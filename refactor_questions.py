import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from load_questions import SESSIONS_PER_BATCH, load_questions, resolve_tenant_names

load_dotenv()

OUTPUT_DIR          = Path('output')
CURRENT_FILE        = OUTPUT_DIR / 'questions.json'     # per org: plan from its latest 25 sessions
PREVIOUS_FILE       = OUTPUT_DIR / 'previous_qn.json'   # per org: the plan before that
NUM_TOPICS          = 6     # topics per org 
QUESTIONS_PER_TOPIC = 7     # one question per topic per day -> 7 days
TARGET_QUESTIONS    = NUM_TOPICS * QUESTIONS_PER_TOPIC
POOL_SIZE           = TARGET_QUESTIONS * 2  # pad with the 25 sessions before
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

# topics for one org
class WeekResult(BaseModel):
    topics: list[Topic] = Field(min_length=NUM_TOPICS, max_length=NUM_TOPICS)



#################################### system prompt #######################################################

SYSTEM_PROMPT = f"""You analyse the questions users of one organisation asked a research assistant about philanthropy
and the social sector in their recent sessions. Your output is a plan: {NUM_TOPICS} topics with {QUESTIONS_PER_TOPIC} questions each.
Question 1 of every topic is asked on day 1, question 2 on day 2, and so on.

You receive numbered entries, each tagged (recent) or (earlier). Some are in languages other than English, and
some are not questions at all (commands or acknowledgements such as "Yes", "Approved plan", "Move to the next step").

Rules:
- Ignore entries that are not meaningful questions or requests for information.
- Translate to English and reframe each chosen entry as a clear, concise question that keeps the original intent
  and specifics (names, places, numbers). Fix spelling. Turn instructions such as "Compare X with Y" into direct
  questions ("How does X compare with Y?"), not generic wrappers.
- Remove words/phrases like "previous", "above", "the response", "this discussion", "the document mentioned
  earlier", or "based on the earlier analysis". Instead, restate the subject explicitly.
- Never pick two entries that ask the same thing (same intent, even if worded differently). Across the whole plan,
  every question must be distinct.
- Prefer (recent) entries. Use (earlier) entries only when there are not enough distinct (recent) questions.
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
            # No tools are passed, so turn off automatic function calling (also silences the SDK's AFC warning)
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
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
    # Checking if there are NUM_TOPICS topics and each topic has QUESTIONS_PER_TOPIC questions
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

    entries = '\n'.join(f'[{i}] ({"earlier" if e["earlier"] else "recent"}) {e["text"]}' for i, e in enumerate(pool))
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

#################################### Build set of questions to be asked ######################################################
# Per org: "recent" = one batch of 25 sessions (latest by default, 26-50 with skip_sessions=25);
# "earlier" = the 25 sessions before that batch, used only as padding.
def build_pools(tenant_names: list[str] | None = None, skip_sessions: int = 0) -> dict[str, dict]:
    rows = load_questions(tenant_names=tenant_names, skip_sessions=skip_sessions, num_sessions=2 * SESSIONS_PER_BATCH)
    last_recent_rank = skip_sessions + SESSIONS_PER_BATCH

    by_org = {}
    for r in rows:  # sorted by org, then time
        group = 'recent' if r.session_rank <= last_recent_rank else 'earlier'
        by_org.setdefault(r.org_id, {'recent': [], 'earlier': []})[group].append(r)

    pools = {}
    for org, groups in by_org.items():
        if not groups['recent']:
            continue
        pool, seen = [], set()
        for earlier, group in ((False, groups['recent']), (True, list(reversed(groups['earlier'])))):  # earlier: most recent first
            for r in group:
                if earlier and len(pool) >= POOL_SIZE:
                    break
                text = r.question_text.strip()
                if text.lower() in seen:  # exact repeats add nothing
                    continue
                seen.add(text.lower())
                pool.append({'text': text, 'asked_at': r.event_time, 'session_id': r.session_id, 'earlier': earlier})
        recent = groups['recent']
        pools[org] = {
            'entries': pool,
            'session_ids': sorted({r.session_id for r in recent}),  # identifies this batch; changes when new sessions arrive
        }

        n_earlier = sum(e['earlier'] for e in pool)
        print(f'  {org}: {len(pool) - n_earlier} entries from {len(pools[org]["session_ids"])} sessions'
              + (f' + {n_earlier} from the {SESSIONS_PER_BATCH} sessions before' if n_earlier else ''))
    print(f'{len(pools)} orgs have sessions in this batch')
    return pools

#################################### LLM call for refactoring the questions ########################################################
def process_orgs(client, pools):
    results, failed = {}, {}
    for org, pool in pools.items():
        print(f'Processing {org} ({len(pool["entries"])} entries)...')
        try:
            results[org] = plan_week(client, pool['entries'])
        except (ValueError, RuntimeError) as e:
            failed[org] = str(e)
            print(f'  ✗ skipped: {e}')
    print(f'Done ✓  ({len(results)} orgs processed, {len(failed)} failed)')
    return results, failed

##################################### writing output to json file ########################################

def build_output(pools, results, failed):
    orgs = []
    for org, topics in results.items():
        pool = pools[org]
        entries = pool['entries']
        # The three lists in each topic line up by index and are in day order (index 0 = day 1)
        orgs.append({
            'tenant_name': org,  # org_id in the questions table = the tenant name, spelled as in tenants.json
            'session_id': pool['session_ids'],  # sessions the plan was built from; a change means new sessions arrived
            **{f'topic{n}': {
                'topic_label': label,
                'set_of_original_questions': [entries[q.id]['text'] for q in picked],
                'refactored_questions': [q.question for q in picked],
                'session_ids': [entries[q.id]['session_id'] for q in picked],
            } for n, (label, picked) in enumerate(topics, start=1)},
        })
        n = sum(len(picked) for _, picked in topics)
        n_earlier = sum(entries[q.id]['earlier'] for _, picked in topics for q in picked)
        print(f'{org}: {n} questions ({n_earlier} from earlier sessions)')
        for label, picked in topics:
            print(f'    {label}: {len(picked)}')

    if failed:
        print(f'\n⚠ {len(failed)} org(s) left out of the output:')
        for org, reason in failed.items():
            print(f'    {org}: {reason}')
    return orgs


##################################### questions.json (current) / previous_qn.json (previous) ########################################

def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    # Never merge into (and so overwrite) a file this script didn't write, e.g. old notebook output
    if not (isinstance(data, dict) and isinstance(data.get('tenants'), list)
            and all(isinstance(t, dict) and 'tenant_name' in t for t in data['tenants'])):
        raise RuntimeError(f'{path} is not in the expected format ({{"tenants": [{{"tenant_name": ...}}, ...]}}). '
                           f'Move or rename it, then run again.')
    return data

def write_json(path: Path, data: dict):
    """Write via a temp file + rename so a crash never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def tenant_map(data: dict | None) -> dict[str, dict]:
    return {t['tenant_name']: t for t in (data or {}).get('tenants', [])}

def save_orgs(path: Path, new_orgs: list[dict]):
    """Replace the given tenants' plans in `path`, keep every other tenant as it is."""
    tenants = tenant_map(read_json(path))
    tenants.update({t['tenant_name']: t for t in new_orgs})
    write_json(path, {'tenants': sorted(tenants.values(), key=lambda t: t['tenant_name'])})

##################################### Arg parser to call process data and call LLM to refactor data ########################################

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--orgs', nargs='+', help='One or more org IDs to process (default: all tenants from tenants.json)')
    parser.add_argument('--overwrite', action='store_true', help='Regenerate orgs even if they have no new sessions')
    parser.add_argument('--previous', action='store_true',
                        help=f'Plan from sessions {SESSIONS_PER_BATCH + 1}-{2 * SESSIONS_PER_BATCH} instead and save to {PREVIOUS_FILE}')
    args = parser.parse_args()
    if args.orgs:
        args.orgs = resolve_tenant_names(args.orgs)  # "dasra" -> "DASRA"

    skip = SESSIONS_PER_BATCH if args.previous else 0
    path = PREVIOUS_FILE if args.previous else CURRENT_FILE
    print(f'Sessions {skip + 1}-{skip + SESSIONS_PER_BATCH} per org -> {path}')
    pools = build_pools(tenant_names=args.orgs, skip_sessions=skip)

    # Only orgs whose batch of sessions changed since their last plan need Gemini
    existing = tenant_map(read_json(path))
    todo = {org: pool for org, pool in pools.items()
            if args.overwrite or existing.get(org, {}).get('session_id') != pool['session_ids']}
    for org in pools.keys() - todo.keys():
        print(f'  {org}: no new sessions since its last plan - skipped')
    if not todo:
        print('Nothing to do (use --overwrite to regenerate)')
        return

    client = genai.Client()  # reads GEMINI_API_KEY
    results, failed = process_orgs(client, todo)
    new_orgs = build_output(todo, results, failed)

    if not args.previous:
        # An org's old plan becomes its previous plan when the new one comes from different sessions
        replaced = [existing[t['tenant_name']] for t in new_orgs
                    if t['tenant_name'] in existing and existing[t['tenant_name']].get('session_id') != t['session_id']]
        if replaced:
            save_orgs(PREVIOUS_FILE, replaced)
            print(f'Moved old plans of {[t["tenant_name"] for t in replaced]} to {PREVIOUS_FILE}')
    save_orgs(path, new_orgs)
    print(f'Output saved to: {path}')


if __name__ == '__main__':
    main()
