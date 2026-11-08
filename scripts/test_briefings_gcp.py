"""
test_briefings_gcp.py
=====================
Live BigQuery test for the briefings table. Run it once before the first
production briefings write on GCP.

It writes synthetic briefings (no model calls) to a scratch dataset, checks
row counts and column types, and deletes the dataset at the end. It never
touches the chbindex_06037 dataset.

Usage:
    export GCP_PROJECT_ID=your-project-id
    export GCP_SERVICE_ACCOUNT_PATH=/path/to/key.json   # optional; ADC if unset
    python scripts/test_briefings_gcp.py
"""

import os
import sys
from datetime import date

from google.cloud import bigquery

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag.briefing import Briefing  # noqa: E402
from src.rag.briefings_table import BRIEFINGS_TABLE, save_briefings  # noqa: E402
from src.utils.bigquery import get_client  # noqa: E402

SCRATCH_DATASET = "chbindex_test_briefings"
WEEK_1 = date(2026, 6, 22)
WEEK_2 = date(2026, 6, 29)


def make_config() -> dict:
    return {
        "full_fips": "06037",
        "storage": {"backend": "gcp"},
        "gcp": {"bigquery_dataset": SCRATCH_DATASET, "location": "US"},
        "bigquery": {"write_disposition": "WRITE_APPEND", "partition_field": "week_start_date"},
        "rag": {"urgency_levels": ["MONITOR", "ELEVATED", "CRITICAL"]},
        "secrets": {
            "GCP_PROJECT_ID": os.environ["GCP_PROJECT_ID"],
            "GCP_SERVICE_ACCOUNT_PATH": os.environ.get("GCP_SERVICE_ACCOUNT_PATH", ""),
        },
    }


def make_briefing(geoid: str, urgency: str = "ELEVATED") -> Briefing:
    return Briefing(
        geoid=geoid,
        urgency=urgency,
        summary="Synthetic test briefing.",
        key_risks=["Heat: test risk."],
        actions=[{"action": "Test action.", "citations": ["E1"]}],
        data_gaps=[],
        sources=["E1"],
        retrieved_chunk_ids=["test-chunk"],
        model="claude-haiku-5-5",
        generated_at="2026-06-23T14:00:00+00:00",
    )


def main() -> int:
    config = make_config()
    client = get_client(config)
    project = config["secrets"]["GCP_PROJECT_ID"]
    table_id = f"{project}.{SCRATCH_DATASET}.{BRIEFINGS_TABLE}"

    dataset = bigquery.Dataset(f"{project}.{SCRATCH_DATASET}")
    dataset.location = "US"
    client.create_dataset(dataset, exists_ok=True)
    client.delete_table(table_id, not_found_ok=True)

    def count(week: date | None = None) -> int:
        where = f"WHERE week_start_date = DATE '{week.isoformat()}'" if week else ""
        q = f"SELECT COUNT(*) AS n FROM `{table_id}` {where}"
        return int(next(client.query(q).result()).n)

    checks: list[tuple[str, bool]] = []

    try:
        save_briefings([make_briefing("06037000100"), make_briefing("06037000200", "CRITICAL")],
                       WEEK_1, config)
        checks.append(("first write creates 2 rows", count() == 2))

        save_briefings([make_briefing("06037000300")], WEEK_2, config)
        checks.append(("new week adds 1 (total 3)", count() == 3))

        save_briefings([make_briefing("06037000100", "CRITICAL"), make_briefing("06037000400")],
                       WEEK_1, config)
        checks.append((
            "re-run week 1 with 2 tracts: week 1 has 2, week 2 untouched",
            count(WEEK_1) == 2 and count(WEEK_2) == 1 and count() == 3,
        ))

        save_briefings([make_briefing("06037000100", "CRITICAL")], WEEK_1, config)
        save_briefings([make_briefing("06037000100", "CRITICAL")], WEEK_1, config)
        checks.append(("re-run week 1 twice: still 1 row for that week", count(WEEK_1) == 1))

        schema = {f.name: f.field_type for f in client.get_table(table_id).schema}
        checks.append((
            "column types: DATE week, TIMESTAMP generated_at, STRING json",
            schema.get("week_start_date") == "DATE"
            and schema.get("generated_at") == "TIMESTAMP"
            and schema.get("actions_json") == "STRING",
        ))
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
