"""
validate_index.py
=================
B1 index validation: weight sensitivity for one week and week-to-week
stability across all stored weeks. Free: no model calls.

    PYTHONPATH=. python3 scripts/validate_index.py --week 2026-06-22

Reads stored compound_scores (stability) and recomputes the week's signals
with the configured PM2.5 window (sensitivity). Writes
evals/results/index_<week>_<time>.json and logs to the MLflow experiment
chbindex-index-validation (skip with --no-mlflow).
"""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

from src.evals.index_validation import temporal_stability, weight_sensitivity
from src.spatial.compound_score import _impute_missing, _load_signals
from src.utils.bigquery import read_query

EXPERIMENT = "chbindex-index-validation"
RESULTS_DIR = Path("evals/results")


def load_config() -> dict:
    return {
        **yaml.safe_load(open("config/pipeline.yaml")),
        **yaml.safe_load(open("config/cities/los_angeles.yaml")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", required=True)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    config = load_config()
    window = int(config.get("pm25_window_weeks", 1))

    # 1. Weight sensitivity for one week
    signals = _load_signals(config, args.week, pm25_window_weeks=window)
    signals, _ = _impute_missing(signals)
    sensitivity = weight_sensitivity(signals, config)

    # 2. Stability across all stored weeks
    scores = read_query(
        f"""
        SELECT week_start_date, geoid, raw_burden, burden_class
        FROM {{dataset}}.compound_scores
        WHERE city_fips = '{config["full_fips"]}'
        """,
        config,
    )
    per_week, stability = temporal_stability(scores)

    small = sensitivity[sensitivity["weight"] != 0]
    metrics = {
        **{f"stability/{k}": v for k, v in stability.items()},
        "sensitivity/min_spearman_25pct": float(small["spearman"].min()),
        "sensitivity/min_jaccard_25pct": float(small["jaccard"].min()),
        "sensitivity/max_left_critical_25pct": int(small["left_critical"].max()),
    }
    manifest = {
        "week": args.week,
        "pm25_window_weeks": window,
        "git_sha": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }

    print(f"\nWeight sensitivity, week {args.week} (PM2.5 window {window} weeks)")
    print(sensitivity.round(3).to_string(index=False))
    print(f"\nStability across {stability['n_weeks']} stored weeks")
    for k, v in stability.items():
        print(f"  {k:24} {round(v, 3)}")
    print("\nLeast stable weeks")
    print(per_week.nsmallest(5, "retained").round(3).to_string(index=False))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"index_{args.week}_{manifest['evaluated_at'][:19].replace(':', '')}"
    path = RESULTS_DIR / f"{stem}.json"
    path.write_text(
        json.dumps(
            {
                "manifest": manifest,
                "metrics": metrics,
                "sensitivity": sensitivity.to_dict(orient="records"),
                "per_week": per_week.astype({"week_start_date": str}).to_dict(orient="records"),
            },
            indent=2,
        )
    )
    print(f"\nSaved: {path}")

    if not args.no_mlflow:
        import mlflow

        mlflow.set_tracking_uri(Path(config["ml"]["mlflow_tracking_uri"]).resolve().as_uri())
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name=f"index {args.week} pm25 {window}wk") as run:
            mlflow.log_params(manifest)
            mlflow.log_metrics(metrics)
            mlflow.log_table(sensitivity, artifact_file="sensitivity.json")
            mlflow.log_table(per_week.astype({"week_start_date": str}), artifact_file="per_week.json")
        print(f"MLflow run: {run.info.run_id} (experiment {EXPERIMENT})")
    return 0


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()


if __name__ == "__main__":
    sys.exit(main())
