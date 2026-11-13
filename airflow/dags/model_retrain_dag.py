"""
model_retrain_dag.py
====================
Weekly champion/challenger retrain and scoring of "enters CRITICAL within
4 weeks", one DAG per city.

Started by spatial_transform_<city> after a live week (not by backfills,
which run with {"notify": false}). Trigger by hand with
{"week_start": "YYYY-MM-DD"} to rerun a week.

Tasks:
    retrain     train a candidate on data up to week_start - 4 weeks minus an
                8-week holdout; promote it over the current model only if its
                holdout PR-AUC is at least as good (src/ml/retrain.py)
    score_week  score every non-CRITICAL tract of the week with the current
                model                                      -> ml_predictions

Everything is computed "as of" week_start: a rerun for a past week never sees
later data. Both tasks read the warehouse and score_week writes it, so they use
the 1-slot duckdb_writer pool with the local backend.
"""

from __future__ import annotations

from airflow.sdk import Param, dag, get_current_context, task
from chbindex_common import (
    CITIES,
    DEFAULT_ARGS,
    WEEK_PATTERN,
    project_config,
    warehouse_pool,
    week_start,
)


def build_retrain_dag(city: str):
    @dag(
        dag_id=f"model_retrain_{city}",
        description=f"Retrain, compare and score the enters-CRITICAL model for {city}",
        schedule=None,  # triggered by spatial_transform_<city>
        max_active_runs=1,
        default_args=DEFAULT_ARGS,
        params={
            "week_start": Param(
                type="string",
                pattern=WEEK_PATTERN,
                description="Monday of the week to score, YYYY-MM-DD.",
            ),
        },
        tags=["chbindex", "ml", city],
    )
    def model_retrain():
        @task
        def resolve_week() -> str:
            from datetime import date

            from src.utils.weeks import monday_of

            week = monday_of(
                date.fromisoformat(get_current_context()["params"]["week_start"])
            )
            print(f"Model retrain and scoring for week starting {week}, {city}")
            return week.isoformat()

        @task(**warehouse_pool())
        def retrain(week: str) -> dict:
            from src.ml.retrain import retrain as run_retrain

            return run_retrain(project_config(city), as_of=week_start(week))

        @task(**warehouse_pool())
        def score_week(week: str) -> dict:
            from src.ml.retrain import score_week as run_score

            df = run_score(project_config(city), week_start(week))
            return {
                "tracts": len(df),
                "model_version": str(df["model_version"].iloc[0]),
                "top10": df.head(10)["geoid"].tolist(),
            }

        week = resolve_week()
        retrain(week) >> score_week(week)

    return model_retrain()


for _city in CITIES:
    globals()[f"model_retrain_{_city}"] = build_retrain_dag(_city)
