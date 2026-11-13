"""
ml_retrain.py
=============
Run the weekly retrain and scoring by hand (the model_retrain DAG does the
same inside Airflow). Free: no API calls.

    PYTHONPATH=. .venv/bin/python scripts/ml_retrain.py --week 2026-06-22
    PYTHONPATH=. .venv/bin/python scripts/ml_retrain.py --week 2026-06-22 --score-only
    PYTHONPATH=. .venv/bin/python scripts/ml_retrain.py --week 2026-06-22 --drift

Do not run while Airflow is writing the warehouse (DuckDB allows one writer).
"""

import argparse
import json
import warnings
from datetime import date

from src.ml.retrain import retrain, score_week
from src.utils.config_loader import load_config

# numpy 2.x + macOS Accelerate: spurious matmul warnings (see scripts/ml_compare.py)
warnings.filterwarnings(
    "ignore", message=".*encountered in matmul", category=RuntimeWarning
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="los_angeles")
    parser.add_argument("--week", required=True, help="Monday, YYYY-MM-DD")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument(
        "--drift", action="store_true", help="also run the drift checks"
    )
    args = parser.parse_args()

    config = load_config(args.city)
    week = date.fromisoformat(args.week)
    if not args.score_only:
        print(json.dumps(retrain(config, as_of=week), indent=2, default=str))
    preds = score_week(config, week)
    print(f"\nTop 10 of {len(preds):,} non-CRITICAL tracts for {week}:")
    print(
        preds.head(10)[
            ["rank", "geoid", "prob_critical_4w", "burden_percentile"]
        ].to_string(index=False)
    )
    if args.drift:
        from src.ml.drift import check_drift

        print("\nDrift:", json.dumps(check_drift(config, week), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
