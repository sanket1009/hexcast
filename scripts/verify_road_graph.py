"""
verify_road_graph.py
====================
Check the compact road network (src/spatial/road_graph.py) on real data.

First run: builds the compact cache from the 520 MB GraphML. This loads the
full graph once, so run it on the host (not in the Airflow worker).
Second run: recomputes healthcare access for a week from the compact cache,
compares it with the stored healthcare_access rows, and reports peak memory.
Nothing is written to the warehouse.

Usage:
    PYTHONPATH=. python3 scripts/verify_road_graph.py            # week 2026-06-22
    PYTHONPATH=. python3 scripts/verify_road_graph.py 2026-06-22
"""

import resource
import sys
import time
from datetime import datetime

import numpy as np

from src.spatial import healthcare_access as ha
from src.utils.bigquery import read_query
from src.utils.config_loader import load_config

COMPARE = [
    "reachable",
    "snap_distance_m",
    "access_score_er",
    "er_count_30min",
    "scored",
    "nearest_er_minutes",
    "access_percentile",
]


def peak_memory_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e9 if sys.platform == "darwin" else rss / 1e6  # bytes vs KB


def main() -> int:
    week = sys.argv[1] if len(sys.argv) > 1 else "2026-06-22"
    config = load_config("los_angeles")
    graphml = ha._graphml_path(config)
    compact = graphml.with_name(f"{graphml.stem}_compact")

    if not compact.exists():
        print(f"Building {compact} from {graphml.name} (one time, high memory)...")
        start = time.time()
        ha._get_road_network(config)
        print(f"Built in {time.time() - start:.0f} s, peak memory {peak_memory_gb():.1f} GB")
        print("Run this script again to check results from the compact cache.")
        return 0

    start = time.time()
    tracts = ha._load_demand_tracts(config)
    hospitals = ha._load_hospital_data(config)
    new = ha._compute_access(
        config, tracts, hospitals, datetime.fromisoformat(week)
    ).set_index("geoid")
    elapsed = time.time() - start

    old = read_query(
        f"""
        SELECT * FROM {{dataset}}.healthcare_access
        WHERE week_start_date = '{week}' AND city_fips = '{config["full_fips"]}'
        """,
        config,
    ).set_index("geoid")

    print(f"\nWeek {week}: {len(new)} tracts new, {len(old)} stored")
    ok = set(new.index) == set(old.index)
    print(f"  same tracts: {ok}")
    for col in COMPARE:
        a, b = new[col].reindex(old.index), old[col]
        if a.dtype == bool or b.dtype == bool:
            diff = int((a.astype(bool) != b.astype(bool)).sum())
            print(f"  {col:20} rows that differ: {diff}")
            ok &= diff == 0
        else:
            a, b = a.astype(float), b.astype(float)
            both_nan = a.isna() & b.isna()
            gap = (a - b).abs()[~both_nan]
            nan_mismatch = int((a.isna() != b.isna()).sum())
            worst = float(gap.max()) if len(gap) else 0.0
            print(f"  {col:20} max difference: {worst:.6g}   NaN mismatches: {nan_mismatch}")
            ok &= nan_mismatch == 0 and (np.isnan(worst) or worst < 1e-6)

    print(f"\nCompact path: {elapsed:.0f} s, peak memory {peak_memory_gb():.2f} GB")
    print("IDENTICAL to stored results." if ok else "DIFFERENCES found, see above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
