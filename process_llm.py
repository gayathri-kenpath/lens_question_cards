import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from load_postgres import load_questions

load_dotenv()

OUTPUT_DIR          = Path('output')
NUM_TOPICS          = 6     # topics per org 
QUESTIONS_PER_TOPIC = 6     # one question per topic per day -> 6 days
TARGET_QUESTIONS    = NUM_TOPICS * QUESTIONS_PER_TOPIC
POOL_SIZE           = TARGET_QUESTIONS * 2  # pad with earlier weeks 
MAX_ATTEMPTS        = 3     # retries if Gemini's answer breaks the rules
MODEL               = 'gemini-3.8-flash'


################# Pydantic Validation #####################

# strucuture for qustion
class PickedQuestion(BaseModel):
    id: int = Field(description='The ID of the input entry this question comes from')
    question: str = Field(description='The entry reframed as a clear, concise, well-formed question in English')

# question per topic 
class Topic(BaseModel):
    label: str = Field(description='Topic title of 1-2 full words, no abbreviations or acronyms, e.g. "Funding Models" or "Agriculture"')
    questions: list[PickedQuestion] = Field(max_length=QUESTIONS_PER_TOPIC,description=f'Up to {QUESTIONS_PER_TOPIC} questions, in the order they should be asked (day 1 first)')

# weekly topics
class WeekResult(BaseModel):
    topics: list[Topic] = Field(min_length=NUM_TOPICS, max_length=NUM_TOPICS)



#################################### system prompt #######################################################

SYSTEM_PROMPT = f"""You analyse the questions users of one organisation asked a research assistant about philanthropy
and the social sector during one week. Your output is a weekly plan: {NUM_TOPICS} topics with {QUESTIONS_PER_TOPIC} questions each.
Question 1 of every topic is asked on day 1, question 2 on day 2, and so on.

You receive numbered entries, each tagged (this week) or (earlier). Some are in languages other than English, and
some are not questions at all (commands or acknowledgements such as "Yes", "Approved plan", "Move to the next step").

Rules:
- Ignore entries that are not meaningful questions or requests for information.
- Translate to English and reframe each chosen entry as a clear, concise question that keeps the original intent
  and specifics (names, places, numbers). Fix spelling. Turn instructions such as "Compare X with Y" into direct
  questions ("How does X compare with Y?"), not generic wrappers.
- Never pick two entries that ask the same thing (same intent, even if worded differently). Across the whole plan,
  every question must be distinct.
- Prefer (this week) entries. Use (earlier) entries only when there are not enough distinct (this week) questions.
- Group the chosen questions into EXACTLY {NUM_TOPICS} topics by what they are about, with distinct titles, and put
  {QUESTIONS_PER_TOPIC} questions in every topic ({NUM_TOPICS * QUESTIONS_PER_TOPIC} in total). Only if there are not enough distinct meaningful
  questions may a topic have fewer, but every topic must have at least one.
- Within a topic, order the questions so they make a good day-by-day sequence (for example broad to specific).
- Give each topic a short title of 1 to 2 words (e.g. "Funding Models", "Agriculture"). Topic titles must use full
  words only - no abbreviations, acronyms or short forms (not "GGI Report", "NGO Models" or "CSR"). If you don't know
  what an acronym stands for, describe the subject instead (e.g. "Report Analysis")."""


################################# Gemini call & validation #######################################################

def call_gemini(client, contents: str) -> WeekResult | None:
    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type='application/json',
            response_schema=WeekResult,
            temperature=0,
        ),
    )
    return response.parsed 

############################ Reframed question validation #########################################

def validate(pool: list[dict], result: WeekResult) -> tuple[list[tuple[str, list[PickedQuestion]]], list[str], list[str]]:
    problems, shortfalls = [], []
    used_ids, used_texts, topics = set(), set(), []
    for topic in result.topics:
        picked = []
        for q in topic.questions:
            text = q.question.strip()
            if not (0 <= q.id < len(pool)) or q.id in used_ids or not text or text.lower() in used_texts:
                continue
            used_ids.add(q.id)
            used_texts.add(text.lower())
            picked.append(PickedQuestion(id=q.id, question=text))
        topics.append((topic.label.strip(), picked))
    # Checking if there are 6 topics and each topic has 6 questions
    if len(topics) != NUM_TOPICS:
        problems.append(f'expected exactly {NUM_TOPICS} topics, got {len(topics)}')
    
    
    # Checking if topic labels are unique
    labels = [label for label, _ in topics]
    if len(set(labels)) != len(labels):
        problems.append('topic titles are not distinct')

    short = [label for label, picked in topics if picked and len(picked) < QUESTIONS_PER_TOPIC]
    if short and len(pool) >= TARGET_QUESTIONS:
        shortfalls.append(f'topics {short} have fewer than {QUESTIONS_PER_TOPIC} questions')
    return topics, problems, shortfalls

############################## call LLM to reframe questions and validate the questions generated by the LLM ###########################################

def plan_week(client, pool: list[dict]):
    if len(pool) < NUM_TOPICS:
        raise ValueError(f'Only {len(pool)} questions - cannot form {NUM_TOPICS} non-empty topics')

    entries = '\n'.join(f'[{i}] ({"earlier" if e["earlier"] else "this week"}) {e["text"]}' for i, e in enumerate(pool))
    contents = f'Entries:\n\n{entries}' 
    issues = ['no valid structured reply']
    for attempt in range(1, MAX_ATTEMPTS + 1):
        result = call_gemini(client, contents)
        if result is not None:
            topics, problems, shortfalls = validate(pool, result)
            issues = problems + shortfalls
            if not issues or (not problems and attempt == MAX_ATTEMPTS):
                if shortfalls:
                    print(f'  ⚠ accepted with fewer questions: {"; ".join(shortfalls)}')
                return topics
        print(f'  attempt {attempt} rejected: {"; ".join(issues)}')
        contents = (f'Entries:\n\n{entries}\n\nYour previous answer was invalid because {"; ".join(issues)}. '
                    f'Return a corrected answer with exactly {NUM_TOPICS} topics of {QUESTIONS_PER_TOPIC} distinct questions each.')
    raise RuntimeError(f'No valid plan after {MAX_ATTEMPTS} attempts: {"; ".join(issues)}')


# ── Pipeline ───────────────────────────────────────────────────────────────

# monday to friday cron data (or today + 7 days)
def week_bounds(week: str | None) -> tuple[str, datetime, datetime]:
    if week:
        year, num = week.upper().split('-W')
        monday = date.fromisocalendar(int(year), int(num), 1)    
    else:
        today = datetime.now(timezone.utc).date()
        monday = today - timedelta(days=today.weekday() + 7)
    start = datetime.combine(monday, datetime.min.time(), tzinfo=timezone.utc)
    iso_year, iso_week, _ = monday.isocalendar()
    return f'{iso_year}-W{iso_week:02d}', start, start + timedelta(days=7)

# Step 1: per org, that week's questions, padded with the most recent earlier ones up to POOL_SIZE.
def build_pools(start: datetime, end: datetime) -> dict[str, list[dict]]:   
    by_org = {}
    for row in load_questions(end=end):  # sorted by org, then time
        by_org.setdefault(row.org_id, []).append(row)

    pools = {}
    for org, rows in by_org.items():
        this_week = [r for r in rows if r.event_time >= start]
        if not this_week:
            continue
        earlier = [r for r in reversed(rows) if r.event_time < start]  # most recent first

        pool, seen = [], set()
        for r in this_week + earlier:
            if r.event_time < start and len(pool) >= POOL_SIZE:
                break
            text = r.question_text.strip()
            if text.lower() in seen:  # exact repeats add nothing
                continue
            seen.add(text.lower())
            pool.append({'text': text, 'asked_at': r.event_time, 'earlier': r.event_time < start})
        pools[org] = pool

        n_earlier = sum(e['earlier'] for e in pool)
        print(f'  {org}: {len(pool) - n_earlier} entries this week'
              + (f' + {n_earlier} from earlier weeks' if n_earlier else ''))
    print(f'{len(pools)} orgs asked questions this week')
    return pools

# Step 2: one Gemini call per org.
def process_orgs(client, pools):
    results, failed = {}, {}
    for org, pool in pools.items():
        print(f'Processing {org} ({len(pool)} entries)...')
        try:
            results[org] = plan_week(client, pool)
        except (ValueError, RuntimeError) as e:
            failed[org] = str(e)
            print(f'  ✗ skipped: {e}')
    print(f'Done ✓  ({len(results)} orgs processed, {len(failed)} failed)')
    return results, failed

##################################### writing output to json file ########################################

def build_output(week, start, end, pools, results, failed):
    orgs = []
    for org, topics in results.items():
        pool = pools[org]
        orgs.append({
            'org_id': org,
            'topics': [{
                'topic_label': label,
                'questions': [{
                    'day': day,
                    'question': q.question,
                    'original_question': pool[q.id]['text'],
                    'asked_at': pool[q.id]['asked_at'].isoformat(),
                    'from_earlier_week': pool[q.id]['earlier'],
                } for day, q in enumerate(picked, start=1)],
            } for label, picked in topics],
        })
        n = sum(len(picked) for _, picked in topics)
        n_earlier = sum(pool[q.id]['earlier'] for _, picked in topics for q in picked)
        print(f'{org}: {n} questions ({n_earlier} from earlier weeks)')
        for label, picked in topics:
            print(f'    {label}: {len(picked)}')

    if failed:
        print(f'\n⚠ {len(failed)} org(s) left out of the output:')
        for org, reason in failed.items():
            print(f'    {org}: {reason}')
    return {
        'week': week,
        'week_start': start.isoformat(),
        'week_end': end.isoformat(),
        'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'orgs': orgs,
        'failed_orgs': failed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--week', help='ISO week to process, e.g. 2026-W40 (default: the previous calendar week)')
    parser.add_argument('--out', type=Path, help=f'Output JSON path (default: {OUTPUT_DIR}/<week>.json)')
    parser.add_argument('--overwrite', action='store_true', help='Regenerate even if the output file already exists')
    args = parser.parse_args()

    week, start, end = week_bounds(args.week)
    out = args.out or OUTPUT_DIR / f'{week}.json'
    if out.exists() and not args.overwrite:
        print(f'{out} already exists - nothing to do (use --overwrite to regenerate)')
        return

    print(f'Week {week}: {start:%Y-%m-%d} to {end - timedelta(days=1):%Y-%m-%d} (UTC)')
    client = genai.Client()  # reads GEMINI_API_KEY
    pools = build_pools(start, end)
    results, failed = process_orgs(client, pools)
    output = build_output(week, start, end, pools, results, failed)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'Output saved to: {out}')


if __name__ == '__main__':
    main()
