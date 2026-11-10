"""
run_evals.py
============
Score one week of stored briefings. Free: no model calls.

    PYTHONPATH=. python3 scripts/run_evals.py --week 2026-06-22 --rank-by burden_percentile

Options:
    --suite            faithfulness | rules | specificity | all (default: all)
    --rank-by          raw_burden (default) or burden_percentile for briefings
                       written before commit 32d3f38 (see src/evals/dataset.py)
    --no-mlflow        skip MLflow logging
    --update-baseline  save this run's gated metrics as the new baseline

Exit code 1 when a gated metric is worse than the baseline.
"""

import argparse
import sys

import yaml

from src.evals.dataset import load_cases
from src.evals.harness import (
    SUITES,
    compare_to_baseline,
    log_to_mlflow,
    run_suites,
    save_results,
    write_baseline,
)


def load_config() -> dict:
    return {
        **yaml.safe_load(open("config/pipeline.yaml")),
        **yaml.safe_load(open("config/cities/los_angeles.yaml")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", required=True)
    parser.add_argument("--suite", choices=[*SUITES, "all"], default="all")
    parser.add_argument("--rank-by", choices=["raw_burden", "burden_percentile"], default="raw_burden")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()

    config = load_config()
    suites = list(SUITES) if args.suite == "all" else [args.suite]

    cases = load_cases(config, args.week, rank_by=args.rank_by)
    run = run_suites(cases, args.week, suites, config, args.rank_by)

    print(f"\nEval results, week {args.week}, {len(cases)} briefings")
    for name, value in run.metrics.items():
        print(f"  {name:50} {value}")

    print(f"\nSaved: {save_results(run)}")
    if not args.no_mlflow:
        run_id = log_to_mlflow(run, config["ml"]["mlflow_tracking_uri"])
        print(f"MLflow run: {run_id} (experiment chbindex-evals)")

    if args.update_baseline:
        print(f"Baseline updated: {write_baseline(run)}")
        return 0

    regressions = compare_to_baseline(run)
    if regressions:
        print("\nREGRESSIONS:")
        for r in regressions:
            print(f"  {r}")
        return 1
    print("\nNo regressions against the baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
