# LensV3 Question Data Pipeline

Builds question plans for LensV3 organizations by collecting user questions from Postgres and using Gemini to organize them into reusable discussion topics.

For each organization, the questions from its 25 most recent sessions are collected, deduplicated, cleaned, and grouped into **6 topics × 7 questions**. Question *N* from each topic is intended to be shown on day *N*, giving users 6 questions per day over 7 days.

The final output is written to `output/questions.json`.

---

# Repository Overview

| File                         | Purpose                                                                      |
| ---------------------------- | ---------------------------------------------------------------------------- |
| `load_questions.py`          | Loads question data from Postgres                                            |
| `refactor_questions.py`      | Runs the Gemini pipeline and generates question plans                        |
| `tenants.json`               | Tenants the pipeline processes (the default and allowed `--orgs`)            |
| `organizations.json`         | Reference only (deployments, roots and their tenants); not read by the code |
| `notebook/process_llm.ipynb` | Legacy notebook version that processes CSV input                             |
| `run_pipeline.sh`            | Runs the pipeline with interval protection                                   |
| `schedule_pipeline.sh`       | Installs/removes the cron job                                                |

---

# Setup

## 1. Create a virtual environment

```bash
python3 -m venv .venv
```

## 2. Install dependencies

```bash
.venv/bin/pip install python-dotenv sqlalchemy "psycopg[binary]" google-genai pydantic
```

## 3. Configure .env

```env
GEMINI_API_KEY=...
DATABASE_HOST=...
SSH_PORT_NO=5432
DATABASE_NAME=...
UNIQUE_NAME_PG_USER=...
UNIQUE_NAME_PG_PASSWD=...
TABLE_NAME=...
# Optional: use a different tenant list (default: tenants.json next to the scripts)
# TENANTS_FILE=/path/to/tenants.json
```

## 4. Choose tenants

`tenants.json` lists the tenants the pipeline works on:

```json
{
  "tenants": ["APURVA", "DASRA", "SELCO", "..."]
}
```

- Running without `--orgs` processes every tenant in this file.
- `--orgs` only accepts tenants from this file. Names are matched case-insensitively (`dasra` -> `DASRA`);
  an unknown name stops the run with an error instead of silently loading nothing.
- To add or remove a tenant, edit this file - no code change needed.

---

# Quick Start

## Load questions

Loads every question from each org's 25 most recent sessions (sessions are ordered by their latest question).
`--previous` loads the 25 sessions before those (sessions 26-50). Without `--orgs`, all tenants in
`tenants.json` are loaded.

```bash
.venv/bin/python load_questions.py
```

```bash
.venv/bin/python load_questions.py --orgs TENANT1 TENANT2 --out input/questions.csv
```

```bash
.venv/bin/python load_questions.py --orgs TENANT_NAME --previous
```

`--min_questions N` keeps only sessions with at least N questions. The filter is applied before the latest 25 are
picked, so you still get up to 25 sessions per org, all meeting the threshold.

```bash
.venv/bin/python load_questions.py --orgs TENANT_NAME --min_questions 5
```

```bash
.venv/bin/python load_questions.py --min_questions 5
```

```bash
.venv/bin/python load_questions.py --orgs TENANT_NAME --min_questions 5 --previous --out input/questions.csv
```

## Generate plans

Only the orgs need to be passed (default: all tenants in `tenants.json`). For each org the plan is built from its latest
25 sessions; sessions 26-50 are used as "earlier" padding when there are too few distinct questions.

Output:
- `output/questions.json` - each org's current plan
- `output/previous_qn.json` - each org's plan before that

Each file holds `{"tenants": [...]}`, with one entry per tenant from `tenants.json`:

```json
{
  "tenant_name": "SELCO",
  "session_id": ["41b0...", "9c2e...", "..."],
  "topic1": {
    "topic_label": "Solar Financing",
    "set_of_original_questions": ["what r loan options for solar", "..."],
    "refactored_questions": ["What loan options exist for solar?", "..."],
    "session_ids": ["41b0...", "..."]
  },
  "topic2": {"...": "..."}
}
```

`session_id` lists the sessions the plan was built from. Inside each topic (`topic1`-`topic6`), the three lists line up
by index and are in day order: item 0 is shown on day 1, item 1 on day 2, and so on. `session_ids[i]` is the session
that `set_of_original_questions[i]` came from.

An org is only regenerated when its latest 25 sessions have changed (a new session arrived); otherwise it
is skipped, so frequent scheduled runs don't call Gemini for nothing. When an org is regenerated, its old
plan moves from `questions.json` to `previous_qn.json`. `--overwrite` regenerates regardless.

`--previous` builds plans from sessions 26-50 (with 51-75 as padding) and saves them straight to
`previous_qn.json`, leaving `questions.json` untouched. Use it to backfill the previous plan.

```bash
.venv/bin/python refactor_questions.py
```

```bash
.venv/bin/python refactor_questions.py --orgs TENANT_NAME
```

```bash
.venv/bin/python refactor_questions.py --orgs TENANT1 TENANT2
```

```bash
.venv/bin/python refactor_questions.py --orgs TENANT_NAME --overwrite
```

```bash
.venv/bin/python refactor_questions.py --orgs TENANT_NAME --previous
```

`--min_questions N` takes each org's latest 25 sessions and plans only from those with at least N questions
(the padding sessions 26-50 are filtered the same way). Changing N changes which sessions a plan is built from,
so the affected orgs are regenerated on the next run.

```bash
.venv/bin/python refactor_questions.py --min_questions 5
```

```bash
.venv/bin/python refactor_questions.py --orgs TENANT_NAME --min_questions 5
```
