"""
Publish the accepted evaluation results to the warehouse (table eval_summary).

Reads evals/results/ml/compare_summary.json and the briefing and retrieval
baselines in evals/baselines/, and rebuilds the table from them. Run after
ml_compare.py (which also calls this) or after accepting a new baseline.

    PYTHONPATH=. python scripts/publish_evals.py [--city los_angeles]
"""

import argparse

from src.evals.publish import publish_eval_summary
from src.utils.config_loader import load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="los_angeles")
    args = parser.parse_args()
    df = publish_eval_summary(load_config(args.city))
    for suite, part in df.groupby("suite"):
        print(f"{suite:<10} {len(part):>4} rows  git {part['git_sha'].iloc[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
