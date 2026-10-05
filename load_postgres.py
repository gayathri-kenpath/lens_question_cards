
import argparse
import csv
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import URL, column, create_engine, func, select, table, text

load_dotenv()

QUESTION_COLUMNS = ['org_id', 'session_id', 'question_text', 'event_time']
ORGANIZATIONS_FILE = Path(__file__).with_name('organizations.json')

# Load tenant names from organizations.json
def load_tenant_names(path=None):
    """Return every tenant name listed under "tenants" in organizations.json, across all deployments."""
    path = Path(path or os.getenv('ORGANIZATIONS_FILE') or ORGANIZATIONS_FILE)
    with open(path) as f:
        orgs = json.load(f)
    names = sorted({tenant for roots in orgs.values() for root in roots for tenant in root.get('tenants', [])})
    if not names:
        raise RuntimeError(f'No tenant names found in {path}')
    return names

# Connect to Postgres db
def get_engine():
    required = ('DATABASE_HOST', 'DATABASE_NAME', 'UNIQUE_NAME_PG_USER', 'UNIQUE_NAME_PG_PASSWD')
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError(f'Missing Postgres settings in .env: {", ".join(missing)}')

    url = URL.create(
        'postgresql+psycopg',
        host=os.environ['DATABASE_HOST'],
        port=int(os.getenv('SSH_PORT_NO') or 5432),
        database=os.environ['DATABASE_NAME'],
        username=os.environ['UNIQUE_NAME_PG_USER'],
        password=os.environ['UNIQUE_NAME_PG_PASSWD'],
    )
    return create_engine(url, connect_args={'sslmode': os.getenv('PG_SSLMODE', 'prefer')})

# Fetch the columns from the table
def fetch_rows(query, params=None):
    """Run a read-only query and return its rows (access columns as row.name)."""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text('SET TRANSACTION READ ONLY'))
            return conn.execute(query, params or {}).all()
    finally:
        engine.dispose()

# Fetch QUESTION_COLUMNS from TABLE_NAME for the known tenants, sorted by org and time. Start/end (timezone-aware datetimes) optionally limit rows to start <= event_time < end.
def load_questions(table_name=None, tenant_names=None, start=None, end=None):
    table_name = table_name or os.getenv('TABLE_NAME')
    if not table_name:
        raise RuntimeError('Missing TABLE_NAME in .env')

    # Build the query with SQLAlchemy so the table/column names are quoted safely
    schema, _, name = table_name.rpartition('.')
    tenant_names = tenant_names or load_tenant_names()
    source = table(name, *(column(c) for c in QUESTION_COLUMNS), schema=schema or None)
    query = (select(*source.c)
             .where(source.c.session_id.is_not(None))
             .where(func.trim(source.c.question_text) != '')
             .where(source.c.org_id.in_(tenant_names))
             .order_by(source.c.org_id, source.c.event_time))
    if start is not None:
        query = query.where(source.c.event_time >= start)
    if end is not None:
        query = query.where(source.c.event_time < end)
    return fetch_rows(query)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, help='Optional CSV path to save results (e.g. input/questions.csv)')
    args = parser.parse_args()

    rows = load_questions()
    print(f'Loaded {len(rows)} rows across {len({r.session_id for r in rows})} sessions')
    for row in rows[:5]:
        print(f'  {row.event_time}  {row.org_id}  {row.session_id}  {row.question_text[:80]}')

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(QUESTION_COLUMNS)
            writer.writerows(rows)
        print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()
