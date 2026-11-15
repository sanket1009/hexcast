"""
Check that the API gives the same answers from BigQuery as from DuckDB.

Every Service method the endpoints call is run twice, once on the local
warehouse and once on the BigQuery copy (scripts/sync_warehouse_to_bigquery.py),
and the results are compared after rounding floats to 6 significant digits
(then allowing one unit in the 3rd decimal, the API's coarsest rounding after
an average) and sorting lists of records (row order is not part of the contract where the
SQL has no ORDER BY).

A difference means a query that DuckDB and BigQuery read differently (type
coercion, NULL ordering, date handling), not a data problem: the tables are
copies. Run it after the first sync and after any change to service.py.

    PYTHONPATH=. python scripts/check_bigquery_api.py [--week 2026-06-22]

BigQuery cost: about 25 queries, each billed at the 10 MB minimum.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

from src.serving.service import Service
from src.serving.store import BigQueryStore, LocalStore
from src.utils.config_loader import load_config


def normalize(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return float(f"{value:.6g}")
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        items = [normalize(v) for v in value]
        if items and all(isinstance(v, dict) for v in items):
            items.sort(key=lambda v: json.dumps(v, sort_keys=True, default=str))
        return items
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def first_difference(a: Any, b: Any, path: str = "") -> str | None:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}.{k}: only in {'BigQuery' if k in b else 'DuckDB'}"
            if diff := first_difference(a[k], b[k], f"{path}.{k}"):
                return diff
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: {len(a)} items in DuckDB, {len(b)} in BigQuery"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            if diff := first_difference(x, y, f"{path}[{i}]"):
                return diff
        return None
    if isinstance(a, float) and isinstance(b, float):
        # The API rounds (e.g. Kriging MAE to 3 decimals) after an AVG whose
        # float sum order differs between engines; a value on a rounding
        # boundary can land one unit apart in the last reported digit
        return (
            None
            if math.isclose(a, b, rel_tol=1e-3, abs_tol=1e-3)
            else f"{path}: DuckDB {a!r} vs BigQuery {b!r}"
        )
    return None if a == b else f"{path}: DuckDB {a!r} vs BigQuery {b!r}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="los_angeles")
    parser.add_argument("--week", default="latest")
    args = parser.parse_args()

    config = load_config(args.city)
    dataset = config["gcp"]["bigquery_dataset"]
    local = Service(LocalStore(config["storage"]["local_db_path"], dataset), config)
    remote = Service(
        BigQueryStore(config["secrets"]["GCP_PROJECT_ID"], dataset), config
    )

    week = local.resolve_week(args.week)
    tracts = local.tracts(week)
    top = tracts.sort_values("burden_percentile", ascending=False)["geoid"].tolist()
    geoid = top[0]
    checks = {
        "weeks": lambda s: s.weeks(),
        "summary": lambda s: s.summary(week, None),
        "tracts": lambda s: (
            s.tracts(week).drop(columns=["week_start_date"]).to_dict("records")
        ),
        "tract": lambda s: s.tract(geoid, week, 8),
        "watchlist": lambda s: s.watchlist(week, 50, None),
        "equity": lambda s: s.equity(week),
        "validation": lambda s: s.validation(week),
        "evaluation": lambda s: s.evaluation(),
        "trends": lambda s: s.trends(top[:3], None, None).to_dict("records"),
        "hexes": lambda s: s.hexes().to_dict("records"),
        "search": lambda s: s.search("watts", 10),
    }

    print(f"Week {week}, tract {geoid}")
    failed = 0
    for name, call in checks.items():
        try:
            a, b = normalize(call(local)), normalize(call(remote))
        except Exception as exc:  # report every endpoint, not just the first error
            failed += 1
            print(f"  {name:<11} ERROR {type(exc).__name__}: {str(exc)[:200]}")
            continue
        diff = first_difference(a, b)
        failed += diff is not None
        print(f"  {name:<11} {'same' if diff is None else 'DIFFERENT ' + diff[:200]}")
    print("All endpoints agree." if not failed else f"{failed} endpoint(s) differ.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
