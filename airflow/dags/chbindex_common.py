"""
chbindex_common.py
==================
Shared helpers for the CHBIndex DAG files. Listed in .airflowignore, so the
scheduler imports it from the DAG files but never parses it as a DAG file.
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import yaml

CITIES = ["los_angeles"]  # chicago: config exists, no Illinois hospital source yet


def _project_root() -> Path:
    """The folder holding config/pipeline.yaml (the repo locally, /opt/airflow in Docker)."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "config" / "pipeline.yaml").exists():
            return parent
    raise FileNotFoundError("config/pipeline.yaml not found above the DAG folder")


ROOT = _project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # so tasks can import src.*

# Only YAML is read when the scheduler parses DAG files. Full config loading
# (validation, secrets) happens inside each task.
PIPELINE = yaml.safe_load((ROOT / "config" / "pipeline.yaml").read_text())
AIRFLOW_CFG = PIPELINE["airflow"]

DEFAULT_ARGS = {
    "retries": AIRFLOW_CFG["max_retries"],
    "retry_delay": timedelta(minutes=AIRFLOW_CFG["retry_delay_minutes"]),
    "execution_timeout": timedelta(minutes=AIRFLOW_CFG["task_timeout_minutes"]),
}

WEEK_PATTERN = r"^\d{4}-\d{2}-\d{2}$"


def city_schedule(city: str) -> dict:
    city_yaml = yaml.safe_load((ROOT / "config" / "cities" / f"{city}.yaml").read_text())
    return city_yaml["schedule"]


def week_start(week: str) -> datetime:
    """'2026-06-22' -> datetime at midnight, the type the src functions take."""
    return datetime.combine(date.fromisoformat(week), datetime.min.time())


def project_config(city: str) -> dict:
    """
    Full config for a city, loaded from inside the project folder.

    The src modules use paths relative to the project root (data/warehouse,
    data/processed, ...), so a task works from there before loading config.
    """
    os.chdir(ROOT)
    from src.utils.config_loader import load_config

    return load_config(city)
