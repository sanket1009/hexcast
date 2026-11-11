"""
Tests for the eval scorers and the baseline gate. Built from small hand-made
briefings, so no warehouse and no model calls.
"""

import json

import pandas as pd
import pytest

from src.evals import harness
from src.evals.dataset import EvalCase, _row_to_briefing
from src.evals.faithfulness import (
    drivers_accounted,
    drivers_mentioned,
    numbers_match_data,
    urgency_matches,
)
from src.evals.harness import EvalRun, compare_to_baseline, write_baseline
from src.evals.rules import (
    flesch_kincaid_grade,
    no_jargon,
    no_source_named,
    no_system_talk,
)
from src.evals.specificity import specificity_metrics
from src.rag.briefing import Briefing, TractRisk

HEAT = "Land surface temperature, mean (°C)"
BEDS = "Hospital beds per 1,000 residents, 30-minute drive"


def _case(
    summary="This tract ranks 5 of 2498 city tracts. Mean land surface temperature is 55.61 °C.",
    key_risks=("Extreme heat at 55.6 °C",),
    actions=(("Open cooling centers during heat waves.", "doc.pdf, p. 1"),),
    data_gaps=(),
    urgency="CRITICAL",
    drivers=("heat",),
    geoid="06037000100",
    groups=(),
):
    briefing = Briefing(
        geoid=geoid,
        urgency=urgency,
        summary=summary,
        key_risks=list(key_risks),
        actions=[{"action": a, "citations": [c]} for a, c in actions],
        data_gaps=list(data_gaps),
        sources=sorted({c for _, c in actions}),
        retrieved_chunk_ids=["c1"],
        model="claude-haiku-5-5",
        generated_at="2026-11-08T00:00:00+00:00",
    )
    tract = TractRisk(
        geoid=geoid,
        risk_rank=5,
        tract_count=2498,
        urgency=urgency,
        drivers=list(drivers),
        indicators={HEAT: 55.61, BEDS: 0.0},
        vulnerable_groups=list(groups),
    )
    return EvalCase(briefing, tract)


# -- faithfulness ------------------------------------------------------------


def test_numbers_pass_with_rank_rounding_and_label_numbers():
    case = _case(
        key_risks=("Heat at about 56 °C", "0.0 beds per 1,000 within a 30-minute drive")
    )
    assert numbers_match_data(case).value == 1.0


def test_numbers_fail_on_invented_figure():
    score = numbers_match_data(
        _case(key_risks=("Heat sends 40 people to the ER each week",))
    )
    assert score.value == 0.0
    assert "40" in score.reason


def test_pm25_is_not_read_as_a_number():
    assert numbers_match_data(_case(key_risks=("PM2.5 is a concern",))).value == 1.0


def test_urgency_contradiction_is_caught():
    case = _case(summary="This tract is at elevated risk this week.")
    assert urgency_matches(case).value == 0.0


def test_elevated_as_plain_adjective_is_allowed():
    case = _case(
        summary="Residents face elevated risk of heat illness and elevated temperatures."
    )
    assert urgency_matches(case).value == 1.0


def test_drivers_mentioned_is_a_share():
    case = _case(drivers=("heat", "healthcare_access"), key_risks=("Extreme heat",))
    assert drivers_mentioned(case).value == 0.5


def test_driver_without_action_or_gap_is_flagged():
    case = _case(drivers=("heat", "air_quality"))
    assert drivers_accounted(case).value == 0.0
    case = _case(
        drivers=("heat", "air_quality"), data_gaps=("No excerpt covers PM2.5",)
    )
    assert drivers_accounted(case).value == 1.0


# -- rules -------------------------------------------------------------------


def test_source_in_key_risks_fails_but_in_data_gaps_passes():
    assert (
        no_source_named(_case(key_risks=("Exposure to freeway traffic",))).value == 0.0
    )
    assert (
        no_source_named(
            _case(data_gaps=("Excerpts only cover near-roadway settings",))
        ).value
        == 1.0
    )


def test_system_talk_and_jargon():
    assert no_system_talk(_case(summary="Its z-score is high.")).value == 0.0
    assert no_jargon(_case(summary="Kriged PM2.5 is 9.0.")).value == 0.0
    assert no_system_talk(_case()).value == 1.0


def test_reading_grade_orders_simple_below_complex():
    simple = "The tract is hot. Open cool rooms. Check on old people."
    complex_ = (
        "Comprehensive intergovernmental coordination regarding environmental "
        "vulnerability necessitates multidisciplinary implementation strategies."
    )
    assert flesch_kincaid_grade(simple) < 6 < flesch_kincaid_grade(complex_)


# -- specificity -------------------------------------------------------------


def test_identical_source_sets_are_counted():
    cases = [
        _case(geoid="a"),
        _case(geoid="b"),
        _case(
            geoid="c",
            drivers=("air_quality",),
            actions=(
                ("Install high-efficiency air filters in schools.", "other.pdf, p. 2"),
            ),
        ),
    ]
    m = specificity_metrics(cases)
    assert m["distinct_source_pages"] == 2
    assert m["duplicate_source_sets"] == pytest.approx(2 / 3, abs=1e-3)
    assert m["same_driver_similarity"] > m["diff_driver_similarity"]


def test_group_separation_is_positive_when_groups_change_the_text():
    older = "older adults (65 and over)"
    lep = "limited-English households"
    a_older = ("Call older residents living alone during heat waves.", "a.pdf, p. 1")
    a_lep = ("Send heat warnings in Spanish and Chinese.", "b.pdf, p. 2")
    base = ("Open cooling centers during heat waves.", "doc.pdf, p. 1")
    cases = [
        _case(geoid="1", groups=(older,), actions=(base, a_older)),
        _case(geoid="2", groups=(older,), actions=(base, a_older)),
        _case(geoid="3", groups=(lep,), actions=(base, a_lep)),
        _case(geoid="4", groups=(lep,), actions=(base, a_lep)),
    ]
    m = specificity_metrics(cases)
    assert m["group_separation"] > 0.1


# -- dataset and harness -----------------------------------------------------


def test_row_to_briefing_parses_json_columns():
    row = pd.Series(
        {
            "geoid": "g",
            "urgency": "CRITICAL",
            "summary": "s",
            "key_risks_json": json.dumps(["k"]),
            "actions_json": json.dumps([{"action": "a", "citations": ["c"]}]),
            "data_gaps_json": None,
            "sources_json": json.dumps(["c"]),
            "retrieved_chunk_ids_json": json.dumps(["id"]),
            "model": "m",
            "generated_at": "2026-11-08 00:00:00",
        }
    )
    b = _row_to_briefing(row)
    assert b.actions[0]["citations"] == ["c"]
    assert b.data_gaps == []


def test_baseline_gate_flags_only_real_drops(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "BASELINES_DIR", tmp_path)
    run = EvalRun(
        week="2026-06-22",
        manifest={},
        metrics={
            "faithfulness.urgency_matches": 0.99,
            "rules.reading_grade.median": 12.0,
        },
        per_briefing=pd.DataFrame(),
    )
    write_baseline(run)

    run.metrics = {
        "faithfulness.urgency_matches": 0.985,
        "rules.reading_grade.median": 12.3,
    }
    assert compare_to_baseline(run) == []  # within slack

    run.metrics = {
        "faithfulness.urgency_matches": 0.90,
        "rules.reading_grade.median": 13.0,
    }
    regressions = compare_to_baseline(run)
    assert len(regressions) == 2


# -- Phase 5.1 rules --------------------------------------------------------


def test_on_driver_actions_flags_hazards_the_tract_does_not_have():
    from src.evals.rules import on_driver_actions

    off = _case(
        drivers=("healthcare_access",),
        actions=(("Notify patients of heat risk during extreme heat.", "p.pdf, p. 1"),),
    )
    on = _case(
        drivers=("heat",),
        actions=(("Open cooling centers during extreme heat.", "p.pdf, p. 1"),),
    )
    assert (
        on_driver_actions(off).value == 0.0 and "heat" in on_driver_actions(off).reason
    )
    assert on_driver_actions(on).value == 1.0


def test_concrete_actions_counts_vague_openings():
    from src.evals.rules import concrete_actions

    case = _case(
        actions=(
            ("Encourage residents to use filters.", "p.pdf, p. 1"),
            ("Distribute air cleaners through county clinics.", "p.pdf, p. 2"),
        )
    )
    assert concrete_actions(case).value == 0.5


def test_urgency_separation_is_zero_when_urgency_levels_share_text():
    same = "Open cooling centers during heat waves."
    cases = [
        _case(geoid=f"0603700{i}", urgency=u, actions=((same, "p.pdf, p. 1"),))
        for i, u in enumerate(["CRITICAL", "ELEVATED", "CRITICAL", "ELEVATED"])
    ]
    assert specificity_metrics(cases)["urgency_separation"] == 0.0


def test_air_quality_actions_named_by_device_count_as_accounted():
    case = _case(
        drivers=("air_quality",),
        actions=(("Help residents get portable air cleaners.", "p.pdf, p. 1"),),
    )
    assert drivers_accounted(case).value == 1.0
