"""
rescore_weeks.py
================
Recompute compound_scores and hotspot_detection for stored weeks, without
re-running Kriging, LST or healthcare access. Use after a change to the
index formula or its config (e.g. pm25_window_weeks). Sends no alerts.

Run inside the Airflow worker, with no DAG writing to the warehouse:

    python scripts/rescore_weeks.py --first 2025-06-23 --last 2026-06-22

Weeks run oldest first; each table replaces its week, so reruns are safe.
"""

import argparse
import sys
from datetime import date, datetime, timedelta

from src.spatial.compound_score import compute_compound_score
from src.spatial.hotspots import detect_hotspots
from src.utils.config_loader import load_config


def mondays(first: date, last: date) -> list[date]:
    start = first - timedelta(days=first.weekday())
    return [start + timedelta(weeks=i) for i in range((last - start).days // 7 + 1)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="los_angeles")
    parser.add_argument("--first", required=True)
    parser.add_argument("--last", required=True)
    args = parser.parse_args()

    config = load_config(args.city)
    print(f"PM2.5 window: {config['pm25_window_weeks']} weeks")
    weeks = mondays(date.fromisoformat(args.first), date.fromisoformat(args.last))
    for i, week in enumerate(weeks, 1):
        ws = datetime.combine(week, datetime.min.time())
        scores = compute_compound_score(config, week_start=ws)
        spots = detect_hotspots(config, week_start=ws)
        n_crit = int((scores["burden_class"] == "CRITICAL").sum())
        n_hot = int(spots["hotspot_class"].str.startswith("HOTSPOT").sum())
        print(f"[{i}/{len(weeks)}] {week}  CRITICAL {n_crit}  hotspot tracts {n_hot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
