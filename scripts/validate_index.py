"""
validate_index.py
=================
B1 index validation: weight sensitivity for one week, week-to-week
stability across all stored weeks, an external benchmark of the
year-average burden against CalEnviroScreen 5.0 and CDC PLACES, Gi*
hotspot stability, and Kriging quality over the year.
Free: no model calls.

    PYTHONPATH=. python3 scripts/validate_index.py --week 2026-06-22

First run (or after an OEHHA update): add --refresh-ces to pull
CalEnviroScreen into the warehouse. --no-benchmark skips step 3.

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

from src.evals.index_validation import (
    external_benchmark,
    hotspot_stability,
    kriging_year,
    temporal_stability,
    weight_sensitivity,
)
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
    parser.add_argument("--refresh-ces", action="store_true")
    parser.add_argument("--no-benchmark", action="store_true")
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

    # 4. Gi* hotspot stability across all stored weeks
    hotspots = read_query(
        f"""
        SELECT h.week_start_date, h.geoid, h.gi_star_zscore, h.hotspot_class,
            s.raw_burden
        FROM {{dataset}}.hotspot_detection h
        JOIN {{dataset}}.compound_scores s
            USING (geoid, week_start_date, city_fips)
        WHERE h.city_fips = '{config["full_fips"]}'
        """,
        config,
    )
    hotspot_weeks, hotspot_summary = hotspot_stability(hotspots)

    # 5. Kriging quality per week (one row per week)
    kriged = read_query(
        f"""
        SELECT week_start_date,
            ANY_VALUE(cv_mae) AS cv_mae,
            ANY_VALUE(cv_r2) AS cv_r2,
            ANY_VALUE(n_monitors) AS n_monitors,
            ANY_VALUE(variogram_nugget) AS variogram_nugget,
            ANY_VALUE(variogram_psill) AS variogram_psill,
            ANY_VALUE(variogram_range_m) AS variogram_range_m
        FROM {{dataset}}.pm25_kriged
        WHERE city_fips = '{config["full_fips"]}'
        GROUP BY week_start_date
        """,
        config,
    )
    kriging_weeks, kriging_summary = kriging_year(kriged)

    # 3. External benchmark of the year average
    benchmark, comparisons = {}, None
    if not args.no_benchmark:
        benchmark, comparisons = run_benchmark(config, args.refresh_ces)

    small = sensitivity[sensitivity["weight"] != 0]
    metrics = {
        **{f"stability/{k}": v for k, v in stability.items()},
        "sensitivity/min_spearman_25pct": float(small["spearman"].min()),
        "sensitivity/min_jaccard_25pct": float(small["jaccard"].min()),
        "sensitivity/max_left_critical_25pct": int(small["left_critical"].max()),
        **{f"benchmark/{k}": v for k, v in benchmark.items()},
        **{f"hotspots/{k}": v for k, v in hotspot_summary.items()},
        **{f"kriging/{k}": v for k, v in kriging_summary.items()},
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
    if comparisons is not None:
        print(f"\nExternal benchmark, year average ({benchmark['n_tracts']} tracts)")
        print(comparisons[["comparison", "spearman"]].round(3).to_string(index=False))
        print(
            f"  top 10% shared with CES: {benchmark['top_decile_overlap']} "
            f"of {benchmark['top_decile_size']} "
            f"({benchmark['top_decile_overlap_no_access']} without access)"
        )

    print(f"\nGi* hotspot stability across {hotspot_summary['n_weeks']} weeks")
    for k, v in hotspot_summary.items():
        print(f"  {k:28} {round(v, 3)}")
    print("\nKriging over the year")
    for k, v in kriging_summary.items():
        print(f"  {k:28} {round(v, 3)}")
    weak = kriging_weeks[kriging_weeks["weak"]]
    cols = ["week_start_date", "cv_mae", "cv_r2", "variogram_nugget", "variogram_psill"]
    print("Weak Kriging weeks")
    print(weak[cols].round(2).to_string(index=False))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"index_{args.week}_{manifest['evaluated_at'][:19].replace(':', '')}"
    path = RESULTS_DIR / f"{stem}.json"
    path.write_text(
        json.dumps(
            {
                "manifest": manifest,
                "metrics": metrics,
                "sensitivity": sensitivity.to_dict(orient="records"),
                "per_week": per_week.astype({"week_start_date": str}).to_dict(
                    orient="records"
                ),
                "benchmark": None
                if comparisons is None
                else comparisons.to_dict(orient="records"),
                "hotspot_weeks": hotspot_weeks.astype({"week_start_date": str}).to_dict(
                    orient="records"
                ),
                "kriging_weeks": kriging_weeks.astype({"week_start_date": str}).to_dict(
                    orient="records"
                ),
            },
            indent=2,
        )
    )
    print(f"\nSaved: {path}")

    if not args.no_mlflow:
        import mlflow

        mlflow.set_tracking_uri(
            Path(config["ml"]["mlflow_tracking_uri"]).resolve().as_uri()
        )
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name=f"index {args.week} pm25 {window}wk") as run:
            mlflow.log_params(manifest)
            mlflow.log_metrics(metrics)
            mlflow.log_table(sensitivity, artifact_file="sensitivity.json")
            mlflow.log_table(
                per_week.astype({"week_start_date": str}), artifact_file="per_week.json"
            )
            if comparisons is not None:
                mlflow.log_table(comparisons, artifact_file="benchmark.json")
            mlflow.log_table(
                hotspot_weeks.astype({"week_start_date": str}),
                artifact_file="hotspot_weeks.json",
            )
            mlflow.log_table(
                kriging_weeks.astype({"week_start_date": str}),
                artifact_file="kriging_weeks.json",
            )
        print(f"MLflow run: {run.info.run_id} (experiment {EXPERIMENT})")
    return 0


def run_benchmark(config: dict, refresh: bool):
    """Year-average burden vs CalEnviroScreen (pulled on refresh) and CDC PLACES."""
    if refresh:
        from src.ingestion.calenviroscreen import pull_calenviroscreen

        pull_calenviroscreen(config)

    fips = config["full_fips"]
    annual = read_query(
        f"""
        WITH s AS (
            SELECT geoid,
                AVG(raw_burden) AS burden,
                AVG(pm25_z) AS pm25_z,
                AVG(lst_z) AS lst_z,
                AVG(access_z) AS access_z
            FROM {{dataset}}.compound_scores
            WHERE city_fips = '{fips}'
            GROUP BY geoid
        ),
        p AS (
            SELECT geoid, AVG(pm25_kriged) AS pm25_ugm3
            FROM {{dataset}}.pm25_kriged
            WHERE city_fips = '{fips}'
            GROUP BY geoid
        )
        SELECT s.*, p.pm25_ugm3 FROM s JOIN p USING (geoid)
        """,
        config,
    )
    ces = read_query(
        f"SELECT * FROM {{dataset}}.calenviroscreen WHERE city_fips = '{fips}'", config
    )
    places = read_query(
        f"SELECT * FROM {{dataset}}.cdc_places_health WHERE city_fips = '{fips}'",
        config,
    )
    return external_benchmark(
        annual,
        ces.drop(columns=["city_fips", "ces_version"]),
        places.drop(columns=["city_fips"]),
    )


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()


if __name__ == "__main__":
    sys.exit(main())
