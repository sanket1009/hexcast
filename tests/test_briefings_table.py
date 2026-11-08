"""
Tests for the briefings table: frame shape, validation, and the local DuckDB
write path with REPLACE_PARTITION. No API calls; Briefing objects are built
directly.
"""

import json
from datetime import date

import duckdb
import pytest

from src.rag.briefing import Briefing
from src.rag.briefings_table import (
    BRIEFING_COLUMNS,
    BRIEFINGS_TABLE,
    build_briefings_frame,
    save_briefings,
)

WEEK_1 = date(2026, 6, 22)
WEEK_2 = date(2026, 6, 29)


def _config(tmp_path):
    return {
        "full_fips": "06037",
        "gcp": {"bigquery_dataset": "chbindex_06037"},
        "bigquery": {"write_disposition": "WRITE_APPEND", "partition_field": "week_start_date"},
        "storage": {"backend": "local", "local_db_path": str(tmp_path / "test.duckdb")},
        "rag": {"urgency_levels": ["MONITOR", "ELEVATED", "CRITICAL"]},
    }


def _briefing(geoid, urgency="ELEVATED", summary="Plain summary."):
    return Briefing(
        geoid=geoid,
        urgency=urgency,
        summary=summary,
        key_risks=["Heat: mean LST 48 °C."],
        actions=[{"action": "Open cooling centers.", "citations": ["E1"]}],
        data_gaps=[],
        sources=["E1"],
        retrieved_chunk_ids=["chunk-1", "chunk-2"],
        model="claude-haiku-5-5",
        generated_at="2026-06-23T14:00:00+00:00",
    )


def test_frame_has_one_row_per_tract_with_expected_columns(tmp_path):
    config = _config(tmp_path)
    df = build_briefings_frame([_briefing("06037000100"), _briefing("06037000200")], WEEK_1, config)

    assert list(df.columns) == BRIEFING_COLUMNS
    assert len(df) == 2
    assert set(df["city_fips"]) == {"06037"}
    assert set(df["week_start_date"]) == {WEEK_1}


def test_nested_fields_round_trip_as_json(tmp_path):
    config = _config(tmp_path)
    df = build_briefings_frame([_briefing("06037000100")], WEEK_1, config)
    row = df.iloc[0]

    assert json.loads(row["actions_json"]) == [{"action": "Open cooling centers.", "citations": ["E1"]}]
    assert json.loads(row["key_risks_json"]) == ["Heat: mean LST 48 °C."]
    assert json.loads(row["retrieved_chunk_ids_json"]) == ["chunk-1", "chunk-2"]


def test_generated_at_is_stored_as_naive_utc(tmp_path):
    config = _config(tmp_path)
    b = _briefing("06037000100")
    b = Briefing(**{**b.__dict__, "generated_at": "2026-06-23T09:00:00-05:00"})
    df = build_briefings_frame([b], WEEK_1, config)

    assert df.iloc[0]["generated_at"].hour == 14
    assert df.iloc[0]["generated_at"].tzinfo is None


def test_local_write_creates_table_and_reads_back(tmp_path):
    config = _config(tmp_path)
    save_briefings([_briefing("06037000100"), _briefing("06037000200", "CRITICAL")], WEEK_1, config)

    con = duckdb.connect(config["storage"]["local_db_path"], read_only=True)
    try:
        rows = con.execute(
            f'SELECT geoid, urgency FROM "chbindex_06037"."{BRIEFINGS_TABLE}" ORDER BY geoid'
        ).fetchall()
    finally:
        con.close()

    assert rows == [("06037000100", "ELEVATED"), ("06037000200", "CRITICAL")]


def test_rerun_same_week_replaces_rows_and_keeps_other_weeks(tmp_path):
    config = _config(tmp_path)
    save_briefings([_briefing("06037000100"), _briefing("06037000200")], WEEK_1, config)
    save_briefings([_briefing("06037000300")], WEEK_2, config)

    # Re-run week 1 with a different set of tracts
    save_briefings([_briefing("06037000100", "CRITICAL"), _briefing("06037000400")], WEEK_1, config)

    con = duckdb.connect(config["storage"]["local_db_path"], read_only=True)
    try:
        week1 = con.execute(
            f'SELECT geoid, urgency FROM "chbindex_06037"."{BRIEFINGS_TABLE}" '
            f"WHERE week_start_date = ? ORDER BY geoid",
            [WEEK_1],
        ).fetchall()
        week2 = con.execute(
            f'SELECT geoid FROM "chbindex_06037"."{BRIEFINGS_TABLE}" WHERE week_start_date = ?',
            [WEEK_2],
        ).fetchall()
    finally:
        con.close()

    assert week1 == [("06037000100", "CRITICAL"), ("06037000400", "ELEVATED")]
    assert week2 == [("06037000300",)]


def test_empty_input_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="No briefings"):
        build_briefings_frame([], WEEK_1, _config(tmp_path))


def test_duplicate_tract_in_one_week_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Duplicate tract"):
        build_briefings_frame(
            [_briefing("06037000100"), _briefing("06037000100")], WEEK_1, _config(tmp_path)
        )


def test_urgency_outside_configured_levels_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Invalid urgency"):
        build_briefings_frame([_briefing("06037000100", "HIGH")], WEEK_1, _config(tmp_path))


def test_empty_summary_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="empty summaries"):
        build_briefings_frame([_briefing("06037000100", summary="   ")], WEEK_1, _config(tmp_path))
