"""Load question data live from Postgres into a DataFrame.

Connection details and the table name are read from .env (never hard-code them here):

    DATABASE_HOST=...
    SSH_PORT_NO=5432          # Postgres port
    DATABASE_NAME=...
    UNIQUE_NAME_PG_USER=...
    UNIQUE_NAME_PG_PASSWD=...
    TABLE_NAME=...            # "table" or "schema.table"

Usage:
    python load_postgres.py                                  # preview the questions table
    python load_postgres.py --out input/questions.csv        # also save it as CSV

From Python:
    from load_postgres import load_questions
    df = load_questions()
"""
import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import URL, column, create_engine, select, table, text

load_dotenv()

QUESTION_COLUMNS = ['event_time', 'session_id', 'question_text']


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


def load_df(query, params=None):
    """Run a read-only query and return the result as a DataFrame."""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text('SET TRANSACTION READ ONLY'))
            return pd.read_sql(query, conn, params=params)
    finally:
        engine.dispose()


def load_questions(table_name=None):
    """Fetch event_time, session_id, question_text from TABLE_NAME, sorted by session and time."""
    table_name = table_name or os.getenv('TABLE_NAME')
    if not table_name:
        raise RuntimeError('Missing TABLE_NAME in .env')

    # Build the query with SQLAlchemy so the table/column names are quoted safely
    schema, _, name = table_name.rpartition('.')
    source = table(name, *(column(c) for c in QUESTION_COLUMNS), schema=schema or None)
    query = (select(*source.c)
             .where(source.c.question_text.is_not(None))
             .order_by(source.c.session_id, source.c.event_time))
    return load_df(query)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, help='Optional CSV path to save results (e.g. input/questions.csv)')
    args = parser.parse_args()

    df = load_questions()
    print(f'Loaded {len(df)} rows across {df["session_id"].nunique()} sessions')
    print(df.head())

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()
