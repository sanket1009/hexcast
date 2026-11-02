"""
Tests for the batch runner: sample selection, retry, checkpoints, and the
no-partial-write rule. Uses a fake generator, so no model calls.
"""

import json
from dataclasses import asdict
from datetime import date

import duckdb
import pytest

from src.rag.briefing import Briefing, BriefingError, TractRisk
from src.rag.briefing_runner import (
    ACCESS_LABEL,
    check_briefing,
    commit_week,
    run_week,
    select_sample,
    write_review,
)
from src.rag.briefings_table import BRIEFINGS_TABLE

WEEK = date(2026, 6, 22)


def _risk(geoid, urgency="ELEVATED", access=5.0):
    return TractRisk(
        geoid=geoid,
        risk_rank=1,
        tract_count=100,
        urgency=urgency,
        drivers=["heat"],
        indicators={ACCESS_LABEL: access},
        vulnerable_groups=[],
    )


def _briefing(geoid, urgency="ELEVATED"):
    return Briefing(
        geoid=geoid,
        urgency=urgency,
        summary="Test summary.",
        key_risks=["Heat: test."],
        actions=[{"action": "Test action.", "citations": ["E1"]}],
        data_gaps=[],
        sources=["E1"],
        retrieved_chunk_ids=["chunk-1"],
        model="claude-haiku-5-5",
        generated_at="2026-06-23T14:00:00+00:00",
    )


class FakeGenerator:
    def __init__(self, fail_once=(), always_fail=()):
        self.calls = []
        self.fail_once = set(fail_once)
        self.always_fail = set(always_fail)

    def generate(self, risk):
        self.calls.append(risk.geoid)
        if risk.geoid in self.always_fail:
            raise BriefingError(f"{risk.geoid}: no valid briefing")
        if risk.geoid in self.fail_once:
            self.fail_once.discard(risk.geoid)
            raise BriefingError(f"{risk.geoid}: cut off")
        return _briefing(risk.geoid, risk.urgency)


def _config(tmp_path):
    return {
        "full_fips": "06037",
        "gcp": {"bigquery_dataset": "chbindex_06037"},
        "bigquery": {"write_disposition": "WRITE_APPEND", "partition_field": "week_start_date"},
        "storage": {"backend": "local", "local_db_path": str(tmp_path / "w.duckdb")},
        "rag": {"urgency_levels": ["MONITOR", "ELEVATED", "CRITICAL"]},
    }


def test_sample_has_25_tracts_by_urgency_and_includes_deserts():
    risks = [_risk(f"0603700{i:04d}", "CRITICAL", access=0.0 if i < 2 else 5.0) for i in range(30)]
    risks += [_risk(f"0603800{i:04d}", "ELEVATED") for i in range(30)]

    chosen = select_sample(risks)

    assert sum(r.urgency == "CRITICAL" for r in chosen) == 13
    assert sum(r.urgency == "ELEVATED" for r in chosen) == 12
    geoids = {r.geoid for r in chosen}
    assert {"06037000000", "06037000001"} <= geoids


def test_sample_is_deterministic():
    risks = [_risk(f"0603800{i:04d}", "ELEVATED") for i in range(40)]
    assert [r.geoid for r in select_sample(risks)] == [r.geoid for r in select_sample(risks)]


def test_failed_tract_is_retried_once(tmp_path):
    gen = FakeGenerator(fail_once={"A"})
    briefings, failures = run_week(
        [_risk("A", "CRITICAL"), _risk("B")], gen, tmp_path / "ckpt.jsonl", retries=1
    )

    assert failures == {}
    assert len(briefings) == 2
    assert gen.calls.count("A") == 2


def test_tract_that_keeps_failing_is_recorded_and_run_continues(tmp_path):
    gen = FakeGenerator(always_fail={"A"})
    briefings, failures = run_week(
        [_risk("A", "CRITICAL"), _risk("B")], gen, tmp_path / "ckpt.jsonl", retries=1
    )

    assert set(failures) == {"A"}
    assert [b.geoid for b in briefings] == ["B"]
    assert gen.calls.count("A") == 2


def test_checkpoint_is_written_and_resume_skips_done_tracts(tmp_path):
    ckpt = tmp_path / "ckpt.jsonl"
    run_week([_risk("A", "CRITICAL")], FakeGenerator(), ckpt)

    gen = FakeGenerator()
    briefings, failures = run_week([_risk("A", "CRITICAL"), _risk("B")], gen, ckpt)

    assert gen.calls == ["B"]
    assert failures == {}
    assert {b.geoid for b in briefings} == {"A", "B"}
    assert len(ckpt.read_text().splitlines()) == 2


def test_commit_refuses_partial_week_by_default(tmp_path):
    config = _config(tmp_path)
    save_first = [_briefing("06037000100")]
    commit_week(save_first, {}, WEEK, config)

    with pytest.raises(RuntimeError, match="not written"):
        commit_week([_briefing("06037000200")], {"06037000300": "boom"}, WEEK, config)

    con = duckdb.connect(config["storage"]["local_db_path"], read_only=True)
    try:
        remaining = con.execute(
            f'SELECT geoid FROM "chbindex_06037"."{BRIEFINGS_TABLE}" ORDER BY geoid'
        ).fetchall()
    finally:
        con.close()
    assert remaining == [("06037000100",)]  # existing week left alone


def test_commit_writes_complete_week(tmp_path):
    config = _config(tmp_path)
    commit_week([_briefing("06037000100"), _briefing("06037000200")], {}, WEEK, config)

    con = duckdb.connect(config["storage"]["local_db_path"], read_only=True)
    try:
        n = con.execute(f'SELECT COUNT(*) FROM "chbindex_06037"."{BRIEFINGS_TABLE}"').fetchone()[0]
    finally:
        con.close()
    assert n == 2


def test_check_briefing_flags_urgency_mismatch():
    risk = _risk("A", "CRITICAL")
    good = check_briefing(_briefing("A", "CRITICAL"), risk)
    bad = check_briefing(_briefing("A", "ELEVATED"), risk)

    assert all(good.values())
    assert bad["urgency matches burden class"] is False


def test_write_review_lists_tracts_and_failures(tmp_path):
    path = tmp_path / "review.md"
    risks = [_risk("A", "CRITICAL"), _risk("B")]
    write_review(path, "2026-06-22", risks, [_briefing("A", "CRITICAL")], {"B": "BriefingError: x"})

    text = path.read_text()
    assert "## A: CRITICAL" in text
    assert "all passed" in text
    assert "## Failed tracts" in text and "- B: BriefingError: x" in text
