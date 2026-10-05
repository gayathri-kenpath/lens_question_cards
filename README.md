# LensV3 question data pipeline

Builds a weekly question plan for each org. Every week, all the questions an org's users
asked during the previous week (across all sessions) go to Gemini, which picks and reframes
them into **6 topics × 6 questions**. Question N of every topic is served on day N, so each
day shows 6 questions, one per topic, for 6 days. The next week, a fresh plan is generated
from that week's questions, and so on.

The result is written to `output/<week>.json`, for example `output/2026-W40.json`.

## How a weekly plan is built

1. **Week window.** By default the run uses the previous calendar week, Monday 00:00 to
   Sunday 23:59 UTC. Run it on Monday 2026-10-05 and you get `2026-W40`, which is
   2026-09-28 to 2026-10-04.
2. **Question pool per org.** All of the org's questions from that week, minus exact repeats.
   If there are fewer than 72 (twice the 36 needed, to allow for duplicates and
   non-questions), the org's most recent questions from earlier weeks are added until the
   pool reaches 72. Orgs that asked nothing that week are skipped.
3. **Gemini, once per org.** It drops non-questions, translates, removes duplicates, and
   reframes. It then picks 36 distinct questions, preferring this week's, and groups them
   into 6 topics of 6 in day order. If an org doesn't have 36 distinct questions even after
   padding, some topics get fewer.
4. **Output.** One JSON file per week, covering all orgs:

```json
{
  "week": "2026-W40",
  "week_start": "2026-09-28T00:00:00+00:00",
  "week_end": "2026-10-05T00:00:00+00:00",
  "orgs": [
    {
      "org_id": "DASRA",
      "topics": [
        {
          "topic_label": "Funding Models",
          "questions": [
            { "day": 1, "question": "...", "original_question": "...",
              "asked_at": "2026-09-29T10:12:00+00:00", "from_earlier_week": false }
          ]
        }
      ]
    }
  ],
  "failed_orgs": {}
}
```

| File                     | What it does                                                                                                            |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| `load_postgres.py`     | Loads `org_id`, `session_id`, `question_text`, `event_time` from Postgres, only for tenants listed in `organizations.json` |
| `process_llm.py`       | Weekly pipeline: load from Postgres → Gemini per org → `output/<week>.json`                                       |
| `process_llm.ipynb`    | Older per-session notebook version that reads a CSV. It doesn't follow the weekly logic                               |
| `run_pipeline.sh`      | Runs the pipeline once, at most every`PIPELINE_INTERVAL_MINUTES`                                                      |
| `schedule_pipeline.sh` | Installs or removes a cron job that calls`run_pipeline.sh`                                                            |
| `organizations.json`   | Tenant list. Only rows whose`org_id` matches a name in a `tenants` list are loaded                                  |

## 1. Setup

You need Python 3 and network access to the Postgres host.

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install python-dotenv sqlalchemy "psycopg[binary]" google-genai pydantic
```

## 2. Configure `.env`

Create a `.env` file in the project root. It is gitignored, so never commit it.

```
GEMINI_API_KEY=...

DATABASE_HOST=...
SSH_PORT_NO=5432              # Postgres port
DATABASE_NAME=...
UNIQUE_NAME_PG_USER=...
UNIQUE_NAME_PG_PASSWD=...
TABLE_NAME=...                # "table" or "schema.table"

# Optional
PG_SSLMODE=prefer             # Postgres SSL mode
ORGANIZATIONS_FILE=...        # path to a different organizations.json
PIPELINE_INTERVAL_MINUTES=30  # how often the scheduled pipeline runs
```

## 3. Choose which tenants to load

`organizations.json` groups tenants by deployment and root:

```json
{
  "lens-v3": [
    { "rootName": "APURVAROOT", "tenants": ["APURVA", "APURVA_COMMUNITY"] }
  ]
}
```

Every name in a `tenants` list, across all deployments, is allowed. Rows whose `org_id` is
not one of these names are skipped. The match is exact and case-sensitive. `rootName` values
are not used. To add or remove a tenant, edit this file.

## 4. Run

**Load only.** This previews the questions without calling Gemini:

```bash
.venv/bin/python load_postgres.py
```

To also save the rows as CSV:

```bash
.venv/bin/python load_postgres.py --out input/questions.csv
```

**Weekly pipeline.** This writes `output/<previous week>.json`. If that file already
exists, the run does nothing:

```bash
.venv/bin/python process_llm.py
```

To build the plan for a specific ISO week:

```bash
.venv/bin/python process_llm.py --week 2026-W40
```

To regenerate a week that already has a file:

```bash
.venv/bin/python process_llm.py --week 2026-W40 --overwrite
```

To write the output somewhere else:

```bash
.venv/bin/python process_llm.py --out output/my_run.json
```

**From Python:**

```python
from datetime import datetime, timezone
from load_postgres import load_questions

rows = load_questions()   # rows straight from Postgres: row.org_id, row.session_id, row.question_text, row.event_time
rows = load_questions(start=datetime(2026, 9, 28, tzinfo=timezone.utc),
                      end=datetime(2026, 10, 5, tzinfo=timezone.utc))   # start <= event_time < end
rows = load_questions(tenant_names=['APURVA', 'TEDX'])                  # or an explicit tenant list
```

## 5. Run on a schedule (macOS / Linux cron)

To install the cron job:

```bash
./schedule_pipeline.sh install
```

To check whether it's installed and see the current interval:

```bash
./schedule_pipeline.sh status
```

To remove it:

```bash
./schedule_pipeline.sh uninstall
```

Cron calls `run_pipeline.sh` every minute, but the script only runs the pipeline once
`PIPELINE_INTERVAL_MINUTES` have passed since the last run. It reads the interval from
`.env` each time, so changes apply without reinstalling. If a run is still going, the next
one is skipped.

Because `process_llm.py` does nothing once the week's file exists, frequent runs are cheap.
Each week's plan is generated on the first run after Monday 00:00 UTC, and later runs that
week exit straight away. If some orgs fail, they are listed under `failed_orgs` and are not
retried automatically. To retry them, run with `--week <week> --overwrite`.

To run it right away, ignoring the interval:

```bash
./run_pipeline.sh --force
```

Logs are written to `logs/pipeline.log`.

## Troubleshooting

- **`Missing Postgres settings in .env`**: a required key in step 2 is empty or missing.
- **`connection ... Operation timed out`**: the database host isn't reachable from your
  network. You may need a VPN or an SSH tunnel, or your IP may need to be allowlisted.
- **0 rows loaded**: check that the `org_id` values in the table match the tenant names in
  `organizations.json` exactly, including case.
- **`ModuleNotFoundError`**: you used the system `python`. Run `.venv/bin/python` instead.
