"""
backfill_dag.py
===============
Load many past weeks for one city, then score each one. One DAG per city.

Trigger by hand with {"first_week": "2025-06-23", "last_week": "2026-06-15"}.

Tasks:
    resolve_weeks     every Monday from first_week to last_week (dates snap to Monday)
    pull_pm25         ONE request per county per calendar year for the whole
                      range (6 for a year inside one calendar year), split into
                      weeks; the weekly DAG would make one per county per week
    pull_lst          one Landsat composite per week, mapped over the weeks,
                      at most 4 at a time for Earth Engine quota. A failed week
                      is recorded and skipped, not fatal
    spatial_confs     the weeks that have both PM2.5 and LST
    trigger_spatial   one spatial_transform_<city> run per week with
                      {"notify": false}, so history sends no emails. The spatial
                      DAG runs one week at a time, so these queue in order

Reruns are safe: every table replaces its week, and existing LST rasters
are reused.
"""

from __future__ import annotations

from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sdk import Param, dag, get_current_context, task
from chbindex_common import (
    CITIES,
    DEFAULT_ARGS,
    WEEK_PATTERN,
    project_config,
    week_start,
)

MAX_WEEKS = 160  # about three years; a guard against a mistyped range


def build_backfill_dag(city: str):

    @dag(
        dag_id=f"backfill_{city}",
        description=f"Load and score a range of past weeks for {city}",
        schedule=None,
        max_active_runs=1,
        default_args=DEFAULT_ARGS,
        params={
            "first_week": Param(type="string", pattern=WEEK_PATTERN, description="First Monday, YYYY-MM-DD."),
            "last_week": Param(type="string", pattern=WEEK_PATTERN, description="Last Monday, YYYY-MM-DD."),
        },
        tags=["chbindex", "backfill", city],
    )
    def backfill():

        @task
        def resolve_weeks() -> list[str]:
            from datetime import date, timedelta

            from src.utils.weeks import monday_of

            params = get_current_context()["params"]
            first = monday_of(date.fromisoformat(params["first_week"]))
            last = monday_of(date.fromisoformat(params["last_week"]))
            if last < first:
                raise ValueError("last_week is before first_week")
            weeks = []
            while first <= last:
                weeks.append(first.isoformat())
                first += timedelta(weeks=1)
            if len(weeks) > MAX_WEEKS:
                raise ValueError(f"{len(weeks)} weeks requested; the limit is {MAX_WEEKS}")
            print(f"Backfilling {len(weeks)} weeks: {weeks[0]} to {weeks[-1]}")
            return weeks

        @task
        def pull_pm25(weeks: list[str]) -> dict:
            from src.ingestion.epa_aqs import pull_pm25_range

            return pull_pm25_range(project_config(city), week_start(weeks[0]), week_start(weeks[-1]))

        @task(max_active_tis_per_dagrun=4)
        def pull_lst(week: str) -> dict:
            from src.ingestion.gee_lst import pull_lst_raster

            try:
                return {"week": week, "raster": pull_lst_raster(project_config(city), week_start=week_start(week))}
            except Exception as exc:  # cloud cover, export timeout: record and move on
                print(f"LST failed for {week}: {type(exc).__name__}: {exc}")
                return {"week": week, "raster": None}

        @task
        def spatial_confs(pm25: dict, lst: list[dict]) -> list[dict]:
            lst_ok = {r["week"] for r in lst if r["raster"]}
            ready = [w for w in pm25["weeks_written"] if w in lst_ok]
            skipped = sorted(set(pm25["weeks_missing"]) | {r["week"] for r in lst if not r["raster"]})
            print(f"{len(ready)} weeks ready for scoring; skipped (no PM2.5 or no LST): {skipped}")
            return [{"week_start": w, "notify": False} for w in ready]

        weeks = resolve_weeks()
        confs = spatial_confs(pull_pm25(weeks), pull_lst.expand(week=weeks))
        TriggerDagRunOperator.partial(
            task_id="trigger_spatial",
            trigger_dag_id=f"spatial_transform_{city}",
        ).expand(conf=confs)

    return backfill()


for _city in CITIES:
    globals()[f"backfill_{_city}"] = build_backfill_dag(_city)
