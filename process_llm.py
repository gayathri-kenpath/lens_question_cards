"""LensV3 question data processing pipeline (LLM version) - script form of process_llm.ipynb.

Fetches questions live from Postgres (see load_postgres.py), batches them by session_id,
asks Gemini per session to translate, filter non-questions, group into topics and reframe,
then writes the result to output/output_llm.json.

Requires GEMINI_API_KEY and the Postgres settings in .env.

Usage:
    python process_llm.py
    python process_llm.py --out output/my_run.json
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from load_postgres import load_questions

load_dotenv()

OUTPUT_DIR   = Path('output')
OUTPUT_FILE  = OUTPUT_DIR / 'output_llm.json'
NUM_TOPICS   = 6          # every session must be split into exactly this many topics
MAX_ATTEMPTS = 3          # retries if Gemini doesn't return exactly NUM_TOPICS valid topics
MODEL        = 'gemini-3.8-flash'


# ── Structured output & prompt ─────────────────────────────────────────────
# Gemini refers to questions by ID only and never rewrites the originals.
# Originals are always copied from the database, so they can't be altered by the model.

class QuestionResult(BaseModel):
    id: int = Field(description='The ID of the input entry')
    is_question: bool = Field(description='False for commands/acknowledgements like "Yes", "Approved plan", "Move to the next step"')
    english: str = Field(description='The entry translated to English (unchanged if already English)')
    reframed: str = Field(description='A clear, concise, well-formed question in English. Empty string if is_question is false')
    duplicate_of: int | None = Field(description='ID of an earlier meaningful question that asks the same thing in different words, else null')


class Topic(BaseModel):
    label: str = Field(description='Topic title of 1-2 full words, no abbreviations or acronyms, e.g. "Funding Models" or "Agriculture"')
    question_ids: list[int] = Field(description='IDs of the unique (non-duplicate) questions in this topic')


class SessionResult(BaseModel):
    questions: list[QuestionResult]
    topics: list[Topic] = Field(min_length=NUM_TOPICS, max_length=NUM_TOPICS)


SYSTEM_PROMPT = f"""You analyse the questions a user asked in one session of a research assistant about philanthropy and the social sector.

You receive numbered entries. Some are in languages other than English, and some are not questions at all
(commands or acknowledgements such as "Yes", "Approved plan", "Move to the next step", "Mark this complete").

For each entry:
- Translate it to English (keep it unchanged if it is already English).
- Decide whether it is a meaningful question or request for information (is_question).
- If it is, reframe it as a clear, concise question that keeps the original intent and specifics
  (names, places, numbers). Fix spelling. Turn instructions such as "Compare X with Y" into direct
  questions ("How does X compare with Y?"), not generic wrappers.
- If it asks the same thing as an earlier meaningful question (same intent, even if worded differently
  or with a spelling difference), set duplicate_of to the ID of the earliest such question. Otherwise null.
  Questions that are related but ask for different information are not duplicates.

Then group only the meaningful questions into EXACTLY {NUM_TOPICS} topics by what they are about - never more,
never fewer. Every topic must contain at least one question and the topics must have distinct titles. If the
questions seem to fall into fewer themes, split the broadest themes into narrower sub-themes until there are
{NUM_TOPICS}. Give each topic a short title of 1 to 2 words (e.g. "Funding Models", "Agriculture").
Topic titles must use full words only - no abbreviations, acronyms or short forms (not "GGI Report",
"NGO Models" or "CSR"). If you don't know what an acronym stands for, describe the subject instead
(e.g. "Report Analysis").
Place only unique questions (duplicate_of is null) in topics - every unique meaningful question must
appear in exactly one topic, and duplicates must not appear in any topic. Return an entry in `questions` for every input ID."""


# ── Gemini call & validation ───────────────────────────────────────────────

def call_gemini(client, contents: str) -> SessionResult | None:
    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type='application/json',
            response_schema=SessionResult,
            temperature=0,
        ),
    )
    return response.parsed  # None if the reply is blocked or doesn't match the schema


def validate(texts: list[str], result: SessionResult) -> tuple[list[tuple[str, list[int]]], dict, list[str]]:
    """Return (topics, questions_by_id, problems). The result is usable only if problems is empty."""
    by_id = {q.id: q for q in result.questions if 0 <= q.id < len(texts)}
    kept_ids = {i for i, q in by_id.items() if q.is_question}
    problems = []

    # A duplicate is merged into the earlier question it repeats; only valid back-references count
    unique_ids = {i for i in kept_ids
                  if not (by_id[i].duplicate_of is not None and by_id[i].duplicate_of in kept_ids
                          and by_id[i].duplicate_of < i)}

    missing = sorted(set(range(len(texts))) - by_id.keys())
    if missing:
        problems.append(f'entries {missing} are missing from `questions`')

    assigned, topics = set(), []
    for topic in result.topics:
        ids = [i for i in topic.question_ids if i in unique_ids and i not in assigned]
        assigned.update(ids)
        topics.append((topic.label.strip(), ids))

    if len(topics) != NUM_TOPICS:
        problems.append(f'there are {len(topics)} topics instead of exactly {NUM_TOPICS}')
    empty = [label for label, ids in topics if not ids]
    if empty:
        problems.append(f'topics {empty} contain no valid question IDs')
    short_forms = [label for label, _ in topics if any(w.isupper() and len(w) > 1 for w in label.split())]
    if short_forms:
        problems.append(f'topic titles {short_forms} use abbreviations or acronyms')
    labels = [label.lower() for label, _ in topics]
    if len(set(labels)) != len(labels):
        problems.append('topic titles are not distinct')
    unassigned = sorted(unique_ids - assigned)
    if unassigned:
        problems.append(f'questions {unassigned} are not placed in any topic')
    return topics, by_id, problems


def analyse_session(client, texts: list[str]):
    if len(texts) < NUM_TOPICS:
        raise ValueError(f'Only {len(texts)} entries - cannot form {NUM_TOPICS} non-empty topics')

    entries = '\n'.join(f'[{i}] {t}' for i, t in enumerate(texts))
    contents = f'Session entries:\n\n{entries}'
    problems = ['no valid structured reply']
    for attempt in range(1, MAX_ATTEMPTS + 1):
        result = call_gemini(client, contents)
        if result is not None:
            topics, by_id, problems = validate(texts, result)
            if not problems:
                return topics, by_id
        print(f'  attempt {attempt} rejected: {"; ".join(problems)}')
        contents = (f'Session entries:\n\n{entries}\n\nYour previous answer was invalid because '
                    f'{"; ".join(problems)}. Return a corrected answer with exactly {NUM_TOPICS} non-empty topics.')
    raise RuntimeError(f'No valid {NUM_TOPICS}-topic grouping after {MAX_ATTEMPTS} attempts: {"; ".join(problems)}')


# ── Pipeline ───────────────────────────────────────────────────────────────

def load_sessions() -> dict[str, pd.DataFrame]:
    """Step 1: fetch live from Postgres and batch by session_id."""
    df = load_questions()
    df['event_time'] = pd.to_datetime(df['event_time'])
    df['question_text'] = df['question_text'].astype(str).str.strip()
    df = df[df['question_text'] != '']
    df = df.sort_values(['session_id', 'event_time']).reset_index(drop=True)

    session_groups = {str(sid): group for sid, group in df.groupby('session_id')}
    print(f'Loaded {len(df)} rows across {len(session_groups)} sessions')
    for sid, group in session_groups.items():
        print(f'  Session {sid}: {len(group)} entries')
    return session_groups


def process_sessions(client, session_groups):
    """Steps 2-3: one Gemini call per session."""
    results, failed = {}, {}
    for sid, group in session_groups.items():
        texts = list(group['question_text'])
        print(f'Processing session {sid} ({len(texts)} entries)...')
        try:
            results[sid] = (texts, *analyse_session(client, texts))
        except (ValueError, RuntimeError) as e:
            failed[sid] = str(e)
            print(f'  ✗ skipped: {e}')
    print(f'Done ✓  ({len(results)} sessions processed, {len(failed)} failed)')
    return results, failed


def build_output(results, failed):
    """Step 4: shape results into the output JSON."""
    output = []
    for sid, (texts, topics, by_id) in results.items():
        session_out = {'session_id': sid}
        for n, (label, ids) in enumerate(topics, start=1):
            session_out[f'topic{n}'] = {
                'topic_label': label,
                'set_of_original_questions': [texts[i] for i in ids],
                'set_of_reframed_questions': [by_id[i].reframed or by_id[i].english for i in ids],
            }
        output.append(session_out)

        n_unique = sum(len(ids) for _, ids in topics)
        n_kept = sum(q.is_question for q in by_id.values())
        print(f'Session {sid}: {n_unique} unique questions, {n_kept - n_unique} duplicates merged, '
              f'{len(texts) - n_kept} non-questions filtered out')
        for label, ids in topics:
            print(f'    {label}: {len(ids)}')

    if failed:
        print(f'\n⚠ {len(failed)} session(s) left out of the output:')
        for sid, reason in failed.items():
            print(f'    {sid}: {reason}')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=OUTPUT_FILE, help=f'Output JSON path (default: {OUTPUT_FILE})')
    args = parser.parse_args()

    client = genai.Client()  # reads GEMINI_API_KEY
    session_groups = load_sessions()
    results, failed = process_sessions(client, session_groups)
    output = build_output(results, failed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'Output saved to: {args.out}')


if __name__ == '__main__':
    main()
