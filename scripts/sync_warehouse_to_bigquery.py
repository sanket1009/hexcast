"""
Copy the local DuckDB warehouse into BigQuery, table by table.

Airflow still writes to DuckDB on the laptop (storage.backend: local). The
deployed API reads BigQuery, so after a weekly run (or a rescore) this script
makes BigQuery an exact copy of the local schema. Each table is dropped and
reloaded, which keeps the copy exact (no leftover rows from deleted weeks) and
lets the table pick up the partitioning below even if an older run created it
without.

    Partitioned by week_start_date (DATE) when the table has it, so a query
        for one week reads one partition instead of 105 weeks.
    Clustered by city_fips and geoid where present, the columns every API
        query filters or joins on.

The dataset is created (in config gcp.location) if it does not exist yet.

Auth: GCP_SERVICE_ACCOUNT_PATH if set, otherwise Application Default
Credentials (`gcloud auth application-default login`), via
src.utils.bigquery.get_client.

    PYTHONPATH=. python scripts/sync_warehouse_to_bigquery.py --dry-run
    PYTHONPATH=. python scripts/sync_warehouse_to_bigquery.py
    PYTHONPATH=. python scripts/sync_warehouse_to_bigquery.py --tables eval_summary briefings
"""

from __future__ import annotations

import argparse
import time
import warnings
from typing import Any

import duckdb
import pandas as pd

from src.utils.config_loader import load_config

# google-cloud-bigquery 3.43 warns on every DataFrame load that a future
# version will need pandas-gbq; loads work without it today
warnings.filterwarnings("ignore", message=".*pandas-gbq.*", category=FutureWarning)

PARTITION = "week_start_date"
CLUSTER_CANDIDATES = ["city_fips", "geoid"]


def local_tables(con: duckdb.DuckDBPyConnection, schema: str) -> dict[str, int]:
    """Table name -> row count, smallest first."""
    names = [
        r[0]
        for r in con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = ? AND table_type = 'BASE TABLE'",
            [schema],
        ).fetchall()
    ]
    counts = {
        n: con.execute(f'SELECT COUNT(*) FROM "{schema}"."{n}"').fetchone()[0]
        for n in names
    }
    return dict(sorted(counts.items(), key=lambda kv: kv[1]))


def layout(df: pd.DataFrame) -> dict[str, Any]:
    """Partition and cluster fields for one table."""
    partition = None
    if PARTITION in df.columns:
        values = df[PARTITION].dropna()
        # Only a real date column can be a time partition
        if len(values) and pd.api.types.is_datetime64_any_dtype(pd.to_datetime(values)):
            partition = PARTITION
    cluster = [c for c in CLUSTER_CANDIDATES if c in df.columns] or None
    return {"partition": partition, "cluster": cluster}


def ensure_dataset(client: Any, dataset_id: str, location: str) -> None:
    """Create the dataset on the first sync; a no-op once it exists."""
    from google.cloud import bigquery

    dataset = bigquery.Dataset(dataset_id)
    dataset.location = location
    client.create_dataset(dataset, exists_ok=True)


def load_table(client: Any, df: pd.DataFrame, table_id: str) -> None:
    from google.cloud import bigquery

    plan = layout(df)
    if plan["partition"]:
        df = df.assign(**{PARTITION: pd.to_datetime(df[PARTITION]).dt.date})
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        time_partitioning=bigquery.TimePartitioning(field=plan["partition"])
        if plan["partition"]
        else None,
        clustering_fields=plan["cluster"],
    )
    client.delete_table(table_id, not_found_ok=True)
    client.load_table_from_dataframe(df, table_id, job_config=job_config).result()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--city", default="los_angeles")
    parser.add_argument("--tables", nargs="*", help="only these tables")
    parser.add_argument(
        "--dry-run", action="store_true", help="list what would be copied"
    )
    args = parser.parse_args()

    config = load_config(args.city)
    schema = config["gcp"]["bigquery_dataset"]
    project = config["secrets"]["GCP_PROJECT_ID"]
    con = duckdb.connect(config["storage"]["local_db_path"], read_only=True)
    tables = local_tables(con, schema)
    if args.tables:
        missing = set(args.tables) - set(tables)
        if missing:
            raise SystemExit(f"Not in the local warehouse: {sorted(missing)}")
        tables = {t: n for t, n in tables.items() if t in args.tables}

    print(f"{len(tables)} tables from {schema} -> {project}.{schema}")
    if args.dry_run:
        for t, n in tables.items():
            df = con.execute(f'SELECT * FROM "{schema}"."{t}" LIMIT 1000').df()
            plan = layout(df)
            print(
                f"  {t:<24} {n:>9,} rows  partition={plan['partition'] or '-'}"
                f"  cluster={','.join(plan['cluster'] or []) or '-'}"
            )
        return 0

    from src.utils.bigquery import get_client

    client = get_client(config)
    ensure_dataset(client, f"{project}.{schema}", config["gcp"].get("location", "US"))
    for t, n in tables.items():
        start = time.monotonic()
        df = con.execute(f'SELECT * FROM "{schema}"."{t}"').df()
        load_table(client, df, f"{project}.{schema}.{t}")
        remote = client.get_table(f"{project}.{schema}.{t}").num_rows
        status = "ok" if remote == n else f"MISMATCH (BigQuery has {remote:,})"
        print(f"  {t:<24} {n:>9,} rows  {time.monotonic() - start:5.1f}s  {status}")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
