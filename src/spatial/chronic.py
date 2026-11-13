"""
chronic.py
==========
Chronic burden: how many of the last 52 weeks each tract was CRITICAL.

One CRITICAL week can be a smoke event; 26 of 52 is a place that stays
burdened. The dashboard counts the people living in chronic tracts.

    weeks_critical   CRITICAL weeks in the window ending at this week
    weeks_observed   scored weeks in that window
    is_chronic       weeks_critical >= chronic_min_weeks, only once the
                     window is full (weeks_observed = chronic_window_weeks);
                     NULL before that (2024-06-24 to 2025-06-09 here)

Output:
    tract_chronic, one row per tract per week (REPLACE_PARTITION)

Usage:
    from src.spatial.chronic import compute_chronic
    df = compute_chronic(config, "2026-06-22")
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from src.spatial.compound_score import _window_start
from src.utils.bigquery import REPLACE_PARTITION, read_query, write_dataframe
from src.utils.logger import get_logger

logger = get_logger(__name__)

TABLE = "tract_chronic"


def compute_chronic(
    config: dict[str, Any], week: str, write: bool = True
) -> pd.DataFrame:
    prod = config["product"]
    window, min_weeks = prod["chronic_window_weeks"], prod["chronic_min_weeks"]
    fips = config["full_fips"]
    df = read_query(
        f"""
        SELECT geoid,
               COUNT(*) FILTER (WHERE burden_class = 'CRITICAL') AS weeks_critical,
               COUNT(DISTINCT week_start_date) AS weeks_observed
        FROM {{dataset}}.compound_scores
        WHERE city_fips = '{fips}'
          AND week_start_date BETWEEN '{_window_start(week, window)}' AND '{week}'
        GROUP BY geoid
        """,
        config,
    )
    if df.empty:
        raise ValueError(f"No compound_scores up to {week}")
    out = add_flags(df, window, min_weeks)
    out["week_start_date"] = date.fromisoformat(week)
    out["city_fips"] = fips
    if write:
        write_dataframe(out, TABLE, config, write_disposition=REPLACE_PARTITION)
    n = int(out["is_chronic"].fillna(False).sum())
    logger.info(
        f"Chronic {week}: {n} tracts CRITICAL in >= {min_weeks} of {window} weeks"
    )
    return out


def add_flags(df: pd.DataFrame, window: int, min_weeks: int) -> pd.DataFrame:
    out = df.copy()
    full = out["weeks_observed"] >= window
    out["is_chronic"] = (
        (out["weeks_critical"] >= min_weeks).where(full).astype("boolean")
    )
    return out
