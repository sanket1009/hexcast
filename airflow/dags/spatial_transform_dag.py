"""
spatial_transform_dag.py
========================
Turns one week of ingested data into tract scores, one DAG per city.

Started by ingest_weekly_<city> with {"week_start": "YYYY-MM-DD"}; it has
no schedule of its own, so it always processes the week ingest just loaded.
Trigger it by hand with the same conf to rerun a week.

Tasks:
    run_kriging        PM2.5 monitors -> tract values           -> pm25_kriged
    process_lst        LST raster -> tract statistics            -> lst_tract_stats
    healthcare_access  E2SFCA on the road network                -> healthcare_access
    compound_score     signed z-score composite, burden classes  -> compound_scores
    hotspots           Getis-Ord Gi* and Moran's I               -> hotspot_detection
    drift_check        input and output drift vs the trailing 8 weeks -> drift_checks
                       (warns, never fails the run; src/ml/drift.py)
    trigger_alert_dispatch  starts alert_dispatch_<city> for the same week,
    trigger_model_retrain   and model_retrain_<city>, unless the run has
                       {"notify": false} (backfills)

The first three are independent, but with the local DuckDB backend they share
the 1-slot duckdb_writer pool and run one at a time (DuckDB allows a single
writer). Every write replaces its week and city, so retries and reruns never
duplicate rows.
"""

from __future__ import annotations

from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sdk import Param, dag, get_current_context, task
from chbindex_common import (
    CITIES,
    DEFAULT_ARGS,
    WEEK_PATTERN,
    project_config,
    warehouse_pool,
    week_start,
)


def build_spatial_dag(city: str):
    @dag(
        dag_id=f"spatial_transform_{city}",
        description=f"Kriging, LST, healthcare access, compound score and hotspots for {city}",
        schedule=None,  # triggered by ingest_weekly_<city>
        max_active_runs=1,
        default_args=DEFAULT_ARGS,
        params={
            "week_start": Param(
                type="string",
                pattern=WEEK_PATTERN,
                description="Monday of the week to process, YYYY-MM-DD.",
            ),
            "notify": Param(
                True,
                type="boolean",
                description="Start alert_dispatch afterwards. Backfills turn this off.",
            ),
        },
        tags=["chbindex", "spatial", city],
    )
    def spatial_transform():
        @task
        def resolve_week() -> str:
            from datetime import date

            from src.utils.weeks import monday_of

            week = monday_of(
                date.fromisoformat(get_current_context()["params"]["week_start"])
            )
            print(f"Spatial transform for week starting {week}, {city}")
            return week.isoformat()

        @task(**warehouse_pool())
        def run_kriging(week: str) -> int:
            from src.spatial.kriging import run_kriging as krige

            return len(krige(project_config(city), week_start=week_start(week)))

        @task(**warehouse_pool())
        def process_lst(week: str) -> int:
            from src.spatial.lst_processing import process_lst_raster

            return len(
                process_lst_raster(project_config(city), week_start=week_start(week))
            )

        @task(**warehouse_pool())
        def healthcare_access(week: str) -> int:
            from src.spatial.healthcare_access import compute_2sfca

            return len(compute_2sfca(project_config(city), week_start=week_start(week)))

        @task(**warehouse_pool())
        def compound_score(week: str) -> dict:
            from src.spatial.compound_score import compute_compound_score

            df = compute_compound_score(
                project_config(city), week_start=week_start(week)
            )
            return df["burden_class"].value_counts().to_dict()

        @task(**warehouse_pool())
        def hotspots(week: str) -> dict:
            from src.spatial.hotspots import detect_hotspots

            df = detect_hotspots(project_config(city), week_start=week_start(week))
            return df["hotspot_class"].value_counts().to_dict()

        @task(**warehouse_pool())
        def drift_check(week: str) -> dict:
            from src.ml.drift import check_drift

            try:
                return check_drift(project_config(city), week_start(week))
            except Exception as exc:  # monitoring must never block scoring
                print(f"Drift check skipped: {type(exc).__name__}: {exc}")
                return {"skipped": str(exc)}

        week = resolve_week()
        inputs = [run_kriging(week), process_lst(week), healthcare_access(week)]
        score = compound_score(week)

        @task.short_circuit
        def notify_requested() -> bool:
            return bool(get_current_context()["params"].get("notify", True))

        start_alerts = TriggerDagRunOperator(
            task_id="trigger_alert_dispatch",
            trigger_dag_id=f"alert_dispatch_{city}",
            conf={
                "week_start": week
            },  # briefings stay off: those runs are started by hand
        )
        start_retrain = TriggerDagRunOperator(
            task_id="trigger_model_retrain",
            trigger_dag_id=f"model_retrain_{city}",
            conf={"week_start": week},
        )
        gate = notify_requested()
        inputs >> score >> [hotspots(week), drift_check(week)] >> gate
        gate >> [start_alerts, start_retrain]

    return spatial_transform()


for _city in CITIES:
    globals()[f"spatial_transform_{_city}"] = build_spatial_dag(_city)
