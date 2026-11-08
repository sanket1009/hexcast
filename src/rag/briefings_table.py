"""
briefings_table.py
==================
Turn generated briefings into the `briefings` warehouse table: one row per
tract per week, keyed the same way as compound_scores
(geoid + week_start_date, partitioned by week_start_date and city_fips).

Re-running a week replaces its rows (REPLACE_PARTITION), so a second run
never leaves duplicate briefings behind.

Nested fields (key risks, actions with citations, sources) are stored as JSON
text. That keeps the table flat, so it loads the same way into DuckDB and
BigQuery, and it can be parsed later with json_extract or JSON functions.

Write path:
    build_briefings_frame()  Briefing objects -> validated DataFrame
    write_briefings()        DataFrame -> local DuckDB or BigQuery
    save_briefings()         both steps, for the caller that has a list
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Iterable

import pandas as pd

from src.utils.bigquery import REPLACE_PARTITION, write_dataframe
from src.utils.logger import get_logger

if TYPE_CHECKING:  # type-only import keeps anthropic and chromadb out of this module
    from src.rag.briefing import Briefing

logger = get_logger(__name__)

BRIEFINGS_TABLE = "briefings"

BRIEFING_COLUMNS = [
    "geoid",
    "week_start_date",
    "city_fips",
    "urgency",
    "summary",
    "key_risks_json",
    "actions_json",
    "data_gaps_json",
    "sources_json",
    "retrieved_chunk_ids_json",
    "model",
    "generated_at",
]

# Columns that must never be empty. Lists (like data_gaps) can be empty,
# but their JSON text is always present.
REQUIRED_COLUMNS = [
    "geoid",
    "week_start_date",
    "city_fips",
    "urgency",
    "summary",
    "key_risks_json",
    "actions_json",
    "model",
    "generated_at",
]


def build_briefings_frame(
    briefings: Iterable["Briefing"],
    week_start: date,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Flatten Briefing objects into one row each, then validate the frame."""
    rows = [_briefing_row(b, week_start, config) for b in briefings]
    if not rows:
        raise ValueError(f"No briefings to write for week {week_start}")

    df = pd.DataFrame(rows, columns=BRIEFING_COLUMNS)
    _validate(df, config)
    return df


def write_briefings(df: pd.DataFrame, config: dict[str, Any]) -> None:
    """Write the frame with REPLACE_PARTITION. Local DuckDB or BigQuery, per config."""
    _validate(df, config)
    write_dataframe(df, BRIEFINGS_TABLE, config, write_disposition=REPLACE_PARTITION)


def save_briefings(
    briefings: Iterable["Briefing"],
    week_start: date,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Build and write one week of briefings. Returns the frame that was written."""
    df = build_briefings_frame(briefings, week_start, config)
    write_briefings(df, config)
    logger.info(
        f"Saved {len(df):,} briefings to {BRIEFINGS_TABLE} for week {week_start}"
    )
    return df


# ---------------------------------------------------------------------------
# Private functions
# ---------------------------------------------------------------------------

def _briefing_row(b: "Briefing", week_start: date, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "geoid": b.geoid,
        "week_start_date": week_start,
        "city_fips": config["full_fips"],
        "urgency": b.urgency,
        "summary": b.summary,
        "key_risks_json": json.dumps(b.key_risks),
        "actions_json": json.dumps(b.actions),
        "data_gaps_json": json.dumps(b.data_gaps),
        "sources_json": json.dumps(b.sources),
        "retrieved_chunk_ids_json": json.dumps(b.retrieved_chunk_ids),
        "model": b.model,
        "generated_at": _naive_utc(b.generated_at),
    }


def _naive_utc(stamp: str | datetime) -> datetime:
    """Store timestamps as naive UTC so DuckDB and BigQuery both read them as TIMESTAMP."""
    ts = pd.Timestamp(stamp)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.to_pydatetime()


def _validate(df: pd.DataFrame, config: dict[str, Any]) -> None:
    if df.empty:
        raise ValueError("Briefings frame is empty")

    missing = [c for c in BRIEFING_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Briefings frame is missing columns: {missing}")

    nulls = df[REQUIRED_COLUMNS].isna().sum()
    nulls = nulls[nulls > 0]
    if not nulls.empty:
        raise ValueError(f"Briefings frame has nulls in required columns: {nulls.to_dict()}")

    if (df["summary"].str.strip() == "").any():
        raise ValueError("Briefings frame has empty summaries")

    # One briefing per tract per week, the grain of the table
    if df.duplicated(subset=["geoid", "week_start_date", "city_fips"]).any():
        raise ValueError("Duplicate tract in briefings frame for one week")

    levels = set(config["rag"]["urgency_levels"])
    invalid = set(df["urgency"].unique()) - levels
    if invalid:
        raise ValueError(f"Invalid urgency levels: {invalid}; allowed {sorted(levels)}")

    if df["city_fips"].nunique() != 1:
        raise ValueError("Briefings frame must cover one city only")
