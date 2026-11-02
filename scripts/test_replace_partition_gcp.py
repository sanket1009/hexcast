"""
test_replace_partition_gcp.py
=============================
Live BigQuery test for REPLACE_PARTITION. Run it once before the first
production write on GCP.

It writes to a scratch dataset, checks the row counts after each step,
and deletes the dataset at the end. It never touches the chbindex_06037
dataset.

Usage:
    export GCP_PROJECT_ID=your-project-id
    export GCP_SERVICE_ACCOUNT_PATH=/path/to/key.json   # optional; ADC if unset
    python scripts/test_replace_partition_gcp.py
"""

import datetime as dt
import os
import sys

import pandas as pd
from google.cloud import bigquery

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.bigquery import REPLACE_PARTITION, get_client, write_dataframe  # noqa: E402

SCRATCH_DATASET = "chbindex_test_replace"
TABLE = "replace_partition_test"


def make_config() -> dict:
    return {
        "storage": {"backend": "gcp"},
        "gcp": {"bigquery_dataset": SCRATCH_DATASET, "location": "US"},
        "bigquery": {"write_disposition": "WRITE_APPEND", "partition_field": "week_start_date"},
        "secrets": {
            "GCP_PROJECT_ID": os.environ["GCP_PROJECT_ID"],
            "GCP_SERVICE_ACCOUNT_PATH": os.environ.get("GCP_SERVICE_ACCOUNT_PATH", ""),
        },
    }


def rows(week: str, city: str, n: int, tag: str) -> pd.DataFrame:
    return pd.DataFrame({
        "geoid": [f"{tag}{i}" for i in range(n)],
        "week_start_date": dt.date.fromisoformat(week),
        "city_fips": city,
        "score": [0.5] * n,
    })


def main() -> int:
    config = make_config()
    client = get_client(config)
    project = config["secrets"]["GCP_PROJECT_ID"]
    table_id = f"{project}.{SCRATCH_DATASET}.{TABLE}"

    dataset = bigquery.Dataset(f"{project}.{SCRATCH_DATASET}")
    dataset.location = "US"
    client.create_dataset(dataset, exists_ok=True)
    client.delete_table(table_id, not_found_ok=True)

    def count(week: str | None = None) -> int:
        where = f"WHERE week_start_date = DATE '{week}'" if week else ""
        q = f"SELECT COUNT(*) AS n FROM `{table_id}` {where}"
        return int(next(client.query(q).result()).n)

    checks: list[tuple[str, bool]] = []

    def write(df: pd.DataFrame) -> None:
        write_dataframe(df, TABLE, config, write_disposition=REPLACE_PARTITION)

    try:
        write(rows("2026-06-15", "06037", 3, "a"))
        checks.append(("first write creates 3 rows", count() == 3))

        write(rows("2026-06-22", "06037", 2, "b"))
        checks.append(("new week adds 2 (total 5)", count() == 5))

        write(rows("2026-06-15", "06037", 1, "c"))
        checks.append((
            "replace week 06-15 with 1 row; week 06-22 untouched",
            count() == 3 and count("2026-06-15") == 1 and count("2026-06-22") == 2,
        ))

        write(rows("2026-06-22", "06037", 2, "b"))
        write(rows("2026-06-22", "06037", 2, "b"))
        checks.append(("re-run same week twice: still 2 rows", count("2026-06-22") == 2))

        write(rows("2026-06-22", "06059", 4, "x"))
        checks.append(("other city, same week, kept (2 + 4)", count("2026-06-22") == 6))

        before = count()
        bad = rows("2026-06-15", "06037", 2, "z")
        bad["not_a_column"] = 1
        try:
            write(bad)
            checks.append(("bad insert raises", False))
        except Exception:
            checks.append(("bad insert raises and rolls back", count() == before and count("2026-06-15") == 1))
    finally:
        client.delete_dataset(
            f"{project}.{SCRATCH_DATASET}", delete_contents=True, not_found_ok=True
        )

    failed = 0
    for name, ok in checks:
        print(("PASS" if ok else "FAIL"), name)
        failed += 0 if ok else 1
    print(f"{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
