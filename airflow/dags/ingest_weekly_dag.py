"""
ingest_weekly_dag.py
====================
Weekly ingestion for one city, one DAG per city in CITIES.

Tasks:
    resolve_week        which Monday this run works on (see below)
    pull_pm25           EPA AQS monitor readings for the week -> epa_pm25_raw
    pull_lst            Landsat land surface temperature raster for the week
    trigger_spatial_transform  starts spatial_transform_<city> for the same week
    reference_requested gate: runs the three refresh tasks only when asked
    refresh_tracts      Census TIGER tracts and buffer tracts
    refresh_cdc_places  CDC PLACES health outcomes
    refresh_hospitals   CDPH hospitals
    refresh_reporting_areas  tract -> SPA, Health District, community (after tracts)

Which week:
    EPA AQS data arrives months late, so a scheduled run processes the
    Monday that is airflow.ingest_lag_weeks before its run date. To process
    a specific week (a backfill, or a rerun), trigger with
    {"week_start": "2026-06-22"}. Any date snaps back to its Monday.

Reference data (tracts, reporting areas, PLACES, hospitals) changes yearly or monthly and
is already loaded, so weekly runs skip it. Trigger with
{"refresh_reference": true} to reload it.

Reruns are safe: PM2.5 rows replace their week and city, and an LST
raster that already exists for the week is reused.

Tasks pass only small values (the week, row counts) between each other.
Data moves through the warehouse, never through Airflow.
"""

from __future__ import annotations

from datetime import date

import pendulum

from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sdk import Param, dag, get_current_context, task
from chbindex_common import (
    AIRFLOW_CFG,
    CITIES,
    DEFAULT_ARGS,
    city_schedule,
    project_config,
    week_start,
)


def build_ingest_dag(city: str):
    schedule = city_schedule(city)

    @dag(
        dag_id=f"ingest_weekly_{city}",
        description=f"Weekly PM2.5 and LST ingestion for {city}",
        schedule=schedule["cron"],
        start_date=pendulum.parse(schedule["start_date"], tz="America/Los_Angeles"),
        catchup=schedule["catchup"],
        max_active_runs=1,  # one week at a time: EPA rate limits, Earth Engine quota
        default_args=DEFAULT_ARGS,
        params={
            "week_start": Param(
                None,
                type=["null", "string"],
                description="Week to process, YYYY-MM-DD. Empty = run date minus the ingest lag.",
            ),
            "refresh_reference": Param(
                False,
                type="boolean",
                description="Also reload tracts, CDC PLACES, and hospitals.",
            ),
        },
        tags=["chbindex", "ingest", city],
    )
    def ingest_weekly():
        @task
        def resolve_week() -> str:
            from src.utils.weeks import monday_of, week_to_process

            ctx = get_current_context()
            override = ctx["params"].get("week_start")
            if override:
                week = monday_of(date.fromisoformat(override))
            else:
                # Manual runs in Airflow 3 can have no logical date; fall back to run_after
                run_date = ctx.get("logical_date") or ctx["dag_run"].run_after
                week = week_to_process(run_date, AIRFLOW_CFG["ingest_lag_weeks"])
            print(f"Processing week starting {week} for {city}")
            return week.isoformat()

        @task
        def pull_pm25(week: str) -> int:
            from src.ingestion.epa_aqs import pull_pm25_readings

            df = pull_pm25_readings(project_config(city), week_start=week_start(week))
            if df.empty:
                raise ValueError(
                    f"No PM2.5 readings for week {week}; AQS may not have it yet"
                )
            return len(df)

        @task
        def pull_lst(week: str) -> str:
            from src.ingestion.gee_lst import pull_lst_raster

            return pull_lst_raster(project_config(city), week_start=week_start(week))

        @task.short_circuit
        def reference_requested() -> bool:
            return bool(get_current_context()["params"].get("refresh_reference"))

        @task
        def refresh_tracts() -> dict:
            from src.ingestion.census_tiger import (
                pull_buffer_tracts,
                pull_tract_geometries,
            )

            config = project_config(city)
            return {
                "tracts": len(pull_tract_geometries(config)),
                "buffer_tracts": len(pull_buffer_tracts(config)),
            }

        @task
        def refresh_cdc_places() -> int:
            from src.ingestion.cdc_places import pull_health_outcomes

            return len(pull_health_outcomes(project_config(city)))

        @task
        def refresh_hospitals() -> int:
            from src.ingestion.hospitals import pull_hospital_locations

            return len(pull_hospital_locations(project_config(city)))

        @task
        def refresh_reporting_areas() -> int:
            from src.ingestion.reporting_areas import pull_reporting_areas

            return len(pull_reporting_areas(project_config(city)))

        week = resolve_week()
        start_spatial = TriggerDagRunOperator(
            task_id="trigger_spatial_transform",
            trigger_dag_id=f"spatial_transform_{city}",
            conf={"week_start": week},
        )
        [pull_pm25(week), pull_lst(week)] >> start_spatial
        gate = reference_requested()
        gate >> [refresh_cdc_places(), refresh_hospitals()]
        (
            gate >> refresh_tracts() >> refresh_reporting_areas()
        )  # areas are assigned to the new tracts

    return ingest_weekly()


for _city in CITIES:
    globals()[f"ingest_weekly_{_city}"] = build_ingest_dag(_city)
