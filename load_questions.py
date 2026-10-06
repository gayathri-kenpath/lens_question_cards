
import argparse
import csv
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import URL, and_, column, create_engine, func, select, table, text

load_dotenv()

QUESTION_COLUMNS = ['org_id', 'session_id', 'question_text', 'event_time']
SESSIONS_PER_BATCH = 25   # one batch = an org's 25 most recent sessions
TENANTS_FILE = Path(__file__).with_name('tenants.json')

##################################### Load tenant names from tenants.json #####################################
def load_tenant_names(path=None):
    """Return the tenant names listed under "tenants" in tenants.json."""
    path = Path(path or os.getenv('TENANTS_FILE') or TENANTS_FILE)
    with open(path) as f:
        names = sorted(set(json.load(f).get('tenants', [])))
    if not names:
        raise RuntimeError(f'No tenant names found in {path}')
    return names

def resolve_tenant_names(requested):
    """Map user-typed org names to their exact spelling in tenants.json (case-insensitive, so "dasra" -> "DASRA")."""
    names = load_tenant_names()
    by_lower = {}
    for name in names:
        by_lower.setdefault(name.lower(), []).append(name)
    resolved, unknown, ambiguous = [], [], []
    for n in requested:
        if n in names:  # exact spelling always wins (e.g. "SOCIAL_ALPHA" vs "Social_Alpha")
            resolved.append(n)
        elif len(by_lower.get(n.lower(), [])) == 1:
            resolved.append(by_lower[n.lower()][0])
        elif n.lower() in by_lower:
            ambiguous.append(f'{n} -> {by_lower[n.lower()]}')
        else:
            unknown.append(n)
    if unknown:
        raise ValueError(f'Unknown org(s) {unknown} - not listed in tenants.json')
    if ambiguous:
        raise ValueError(f'Ambiguous org(s), type the exact spelling: {ambiguous}')
    return resolved

######################################### Connect to Postgres db ##############################################
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

############################## Fetch the columns from the table ##########################################
def fetch_rows(query, params=None):
    """Run a read-only query and return its rows (access columns as row.name)."""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text('SET TRANSACTION READ ONLY'))
            return conn.execute(query, params or {}).all()
    finally:
        engine.dispose()

################### Fetch questions by session batch ######################
# Sessions are ranked per org by their latest question (rank 1 = most recent session).
# skip_sessions=0, num_sessions=25 -> the latest 25 sessions of every org; skip_sessions=25 -> sessions 26-50.
# Each row also carries session_rank so callers can split the result into batches.
def load_questions(table_name=None, tenant_names=None, skip_sessions=0, num_sessions=SESSIONS_PER_BATCH):
    table_name = table_name or os.getenv('TABLE_NAME')
    if not table_name:
        raise RuntimeError('Missing TABLE_NAME in .env')

    # Build the query with SQLAlchemy so the table/column names are quoted safely
    schema, _, name = table_name.rpartition('.')
    tenant_names = resolve_tenant_names(tenant_names) if tenant_names else load_tenant_names()
    source = table(name, *(column(c) for c in QUESTION_COLUMNS), schema=schema or None)
    usable = and_(source.c.session_id.is_not(None),
                  func.trim(source.c.question_text) != '',
                  source.c.org_id.in_(tenant_names))

    sessions = (select(source.c.org_id, source.c.session_id, func.max(source.c.event_time).label('last_at'))
                .where(usable)
                .group_by(source.c.org_id, source.c.session_id)
                .subquery())
    ranked = select(sessions.c.org_id, sessions.c.session_id,
                    func.row_number().over(partition_by=sessions.c.org_id,
                                           order_by=(sessions.c.last_at.desc(), sessions.c.session_id))
                    .label('session_rank')).subquery()
    query = (select(*source.c, ranked.c.session_rank)
             .join(ranked, and_(source.c.org_id == ranked.c.org_id, source.c.session_id == ranked.c.session_id))
             .where(usable)
             .where(ranked.c.session_rank > skip_sessions)
             .where(ranked.c.session_rank <= skip_sessions + num_sessions)
             .order_by(source.c.org_id, source.c.event_time))
    return fetch_rows(query)

######################## Arg pasrser to load data from db ##################################
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--orgs', nargs='+', help='One or more org IDs to load (default: all tenants from tenants.json)')
    parser.add_argument('--out', type=Path, help='Optional CSV path to save results (e.g. input/questions.csv)')
    parser.add_argument('--previous', action='store_true',
                        help=f'Load the {SESSIONS_PER_BATCH} sessions before the latest {SESSIONS_PER_BATCH} '
                             f'(sessions {SESSIONS_PER_BATCH + 1}-{2 * SESSIONS_PER_BATCH}) instead')
    args = parser.parse_args()

    skip = SESSIONS_PER_BATCH if args.previous else 0
    rows = load_questions(tenant_names=args.orgs, skip_sessions=skip)
    print(f'Sessions {skip + 1}-{skip + SESSIONS_PER_BATCH} per org: loaded {len(rows)} rows')
    by_org = {}
    for r in rows:
        by_org.setdefault(r.org_id, set()).add(r.session_id)
    for org, sessions in by_org.items():
        print(f'  {org}: {sum(r.org_id == org for r in rows)} rows from {len(sessions)} sessions')

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(QUESTION_COLUMNS)
            writer.writerows([getattr(r, c) for c in QUESTION_COLUMNS] for r in rows)
        print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()
