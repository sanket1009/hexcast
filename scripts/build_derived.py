"""
Build the dashboard tables for a range of stored weeks.

    tract_chronic     weeks CRITICAL in the last 52 (src/spatial/chronic.py)
    tract_confidence  correlated Kriging error + weight simulation
                      (src/spatial/confidence.py)
    h3_hexes          populated H3 cells (--hexes; not weekly)

The weekly spatial_transform DAG builds the first two for new weeks; this
script fills history. Each week's Kriging error factor is cached and
reused by the 4 windows it belongs to.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/build_derived.py --first 2024-06-24 --last 2026-06-22 --hexes
"""

import argparse
import time

from src.spatial.chronic import compute_chronic
from src.spatial.confidence import compute_confidence
from src.spatial.hex_grid import build_hex_table
from src.utils.bigquery import read_query
from src.utils.config_loader import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--first", required=True)
    parser.add_argument("--last", required=True)
    parser.add_argument("--city", default="los_angeles")
    parser.add_argument("--hexes", action="store_true", help="also rebuild h3_hexes")
    parser.add_argument("--skip-confidence", action="store_true")
    args = parser.parse_args()

    config = load_config(args.city)
    if args.hexes:
        build_hex_table(config)

    weeks = read_query(
        f"""
        SELECT DISTINCT CAST(week_start_date AS VARCHAR) AS week
        FROM {{dataset}}.compound_scores
        WHERE city_fips = '{config["full_fips"]}'
          AND week_start_date BETWEEN '{args.first}' AND '{args.last}'
        ORDER BY week
        """,
        config,
    )["week"].str[:10]
    cache: dict = {}
    start = time.time()
    for i, week in enumerate(weeks, 1):
        compute_chronic(config, week)
        if not args.skip_confidence:
            compute_confidence(config, week, factor_cache=cache)
        print(f"{i}/{len(weeks)} {week} done ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
