"""
Tests for shared action plans: one model call per driver combination, plan
reuse from the cache file, and the tract summary template. Fake retriever and
client, so no model calls.
"""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.evals.dataset import EvalCase
from src.evals.faithfulness import numbers_match_data, urgency_matches
from src.evals.rules import no_jargon, no_source_named, no_system_talk
from src.rag.action_plans import (
    ActionPlanDraft,
    PlanBriefingGenerator,
    plan_key,
    tract_summary,
)
from src.rag.briefing import Action, BriefingError, CallCapError, TractRisk
from src.rag.retriever import RetrievedChunk
from src.rag.risk_inputs import INDICATORS

LABEL = {col: label for label, col in INDICATORS.items()}

CONFIG = {
    "rag": {
        "anthropic": {"model": "claude-haiku-5-5", "max_tokens": 3000},
        "urgency_levels": ["MONITOR", "ELEVATED", "CRITICAL"],
        "driver_queries": {
            "heat": "heat for {group}",
            "air_quality": "air for {group}",
            "healthcare_access": "access for {group}",
        },
        "briefing_excerpts_per_driver": 2,
        "default_vulnerable_group": "vulnerable residents",
    }
}


class FakeRetriever:
    def retrieve(self, query, top_k):
        return [
            RetrievedChunk(f"{query}-{i}", f"text {i}", f"{query}.pdf", i, 0.3)
            for i in range(top_k)
        ]


class FakeClient:
    def __init__(self, stop_reason="end_turn"):
        self.calls = 0
        self.stop_reason = stop_reason
        self.messages = self

    def parse(self, **kwargs):
        self.calls += 1
        draft = ActionPlanDraft(
            recommended_actions=[
                Action(action="Open cooling centers.", excerpt_ids=["E1"]),
                Action(action="Uncited idea.", excerpt_ids=["E99"]),
            ],
            data_gaps=[],
        )
        return SimpleNamespace(stop_reason=self.stop_reason, parsed_output=draft)


def _tract(geoid, drivers, urgency="CRITICAL"):
    return TractRisk(
        geoid=geoid,
        risk_rank=5,
        tract_count=2498,
        urgency=urgency,
        drivers=drivers,
        indicators={
            LABEL["pm25_kriged"]: 8.69,
            LABEL["lst_mean_celsius"]: 55.61,
            LABEL["access_score_er"]: 0.0,
            LABEL["nearest_er_minutes"]: 5.9,
            LABEL["asthma_pct"]: 10.0,
        },
    )


def _generator(client, tmp_path=None):
    return PlanBriefingGenerator(
        CONFIG,
        retriever=FakeRetriever(),
        client=client,
        plan_cache_path=tmp_path / "plans.jsonl" if tmp_path else None,
        group_cache_path=tmp_path / "groups.jsonl" if tmp_path else None,
    )


def test_one_model_call_per_driver_combination():
    client = FakeClient()
    gen = _generator(client)
    tracts = [
        _tract("a", ["heat"]),
        _tract("b", ["heat"]),
        _tract("c", ["heat", "air_quality"]),
        _tract("d", ["air_quality", "heat"]),  # same combination, other order
    ]
    briefings = [gen.generate(t) for t in tracts]
    assert client.calls == 2
    assert briefings[0].actions == briefings[1].actions


def test_uncited_actions_are_dropped():
    b = _generator(FakeClient()).generate(_tract("a", ["heat"]))
    assert [a["action"] for a in b.actions] == ["Open cooling centers."]


def test_plans_are_reused_from_the_cache_file(tmp_path):
    first = FakeClient()
    _generator(first, tmp_path).generate(_tract("a", ["heat"]))
    second = FakeClient()
    _generator(second, tmp_path).generate(_tract("b", ["heat"]))
    assert (first.calls, second.calls) == (1, 0)


def test_refusal_raises_briefing_error():
    with pytest.raises(BriefingError):
        _generator(FakeClient(stop_reason="refusal")).generate(_tract("a", ["heat"]))


def test_plan_key_ignores_order():
    assert plan_key(["heat", "air_quality"], ["b", "a"]) == plan_key(
        ["air_quality", "heat"], ["a", "b"]
    )


def test_template_passes_the_evals():
    tract = _tract("06037900102", ["heat", "healthcare_access", "air_quality"])
    summary, key_risks = tract_summary(tract)
    assert "55.61 °C" in key_risks[0]
    assert "nearest emergency room 5.9 minutes" in key_risks[2]

    briefing = _generator(FakeClient()).generate(tract)
    case = EvalCase(briefing, tract)
    for scorer in (
        numbers_match_data,
        urgency_matches,
        no_source_named,
        no_system_talk,
        no_jargon,
    ):
        assert scorer(case).value == 1.0, scorer.__name__


def test_template_skips_missing_indicators():
    tract = TractRisk("x", 1, 10, "ELEVATED", ["healthcare_access"], {})
    summary, key_risks = tract_summary(tract)
    assert summary.startswith("Elevated urgency this week.")
    assert key_risks == []


# ---------------------------------------------------------------------------
# Vulnerable groups (Phase 3): driver plan + reusable group actions
# ---------------------------------------------------------------------------


def _group_tract(geoid, drivers, groups, urgency="CRITICAL"):
    from src.rag.risk_inputs import GROUP_KEYS, group_median_label, group_share_label

    t = _tract(geoid, drivers, urgency)
    indicators = dict(t.indicators)
    for g in groups:
        indicators[group_share_label(GROUP_KEYS[g])] = 31.2
        indicators[group_median_label(GROUP_KEYS[g])] = 14.5
    return replace(t, indicators=indicators, vulnerable_groups=groups)


OLDER = "older adults (65 and over)"
LEP = "limited-English households"
KIDS = "young children (under 5)"


def test_groups_do_not_change_the_driver_plan():
    # Driver plans stay keyed on drivers only, so cached plans are reused
    client = FakeClient()
    gen = _generator(client)
    gen.generate(_tract("a", ["heat"]))
    calls_before = client.calls
    gen.generate(_group_tract("b", ["heat"], [OLDER]))
    assert client.calls == calls_before + 1  # only the new group actions
    assert list(gen.plans) == ["heat|vulnerable residents"]


def test_group_actions_are_written_once_per_hazard_and_group(tmp_path):
    client = FakeClient()
    gen = _generator(client, tmp_path)
    for i in range(3):
        gen.generate(_group_tract(f"t{i}", ["heat", "air_quality"], [OLDER, LEP]))
    # 1 plan + heat|age_65_plus + heat|limited_english; air quality is not
    # the strongest driver, so it gets no group calls
    assert client.calls == 3
    assert sorted(gen.group_actions) == ["heat|age_65_plus", "heat|limited_english"]

    again = FakeClient()
    _generator(again, tmp_path).generate(
        _group_tract("x", ["heat", "air_quality"], [LEP])
    )
    assert again.calls == 0  # plan and group actions come from the cache files


def test_group_actions_are_labelled_and_capped():
    class ManyActions(FakeClient):
        def parse(self, **kwargs):
            self.calls += 1
            draft = ActionPlanDraft(
                recommended_actions=[
                    Action(action=f"Step {i}.", excerpt_ids=["E1"]) for i in range(4)
                ],
                data_gaps=[],
            )
            return SimpleNamespace(stop_reason="end_turn", parsed_output=draft)

    b = _generator(ManyActions()).generate(_group_tract("a", ["heat"], [OLDER]))
    group_actions = [a for a in b.actions if a.get("group") == OLDER]
    assert len(group_actions) == 2  # MAX_GROUP_ACTIONS
    assert all(a["citations"] for a in group_actions)
    assert b.vulnerable_groups == [OLDER]
    assert b.inputs["vulnerable_groups"] == [OLDER]


def test_uncertain_group_is_noted_in_data_gaps():
    b = _generator(FakeClient()).generate(_group_tract("a", ["heat"], [KIDS]))
    assert any("young children" in g for g in b.data_gaps)


def test_default_group_gets_no_group_actions():
    client = FakeClient()
    gen = _generator(client)
    b = gen.generate(_group_tract("a", ["heat"], []))
    assert client.calls == 1
    assert not any("group" in a for a in b.actions)


def test_call_cap_stops_before_the_call():
    client = FakeClient()
    gen = PlanBriefingGenerator(
        CONFIG, retriever=FakeRetriever(), client=client, max_model_calls=1
    )
    with pytest.raises(CallCapError):
        gen.generate(_group_tract("a", ["heat"], [OLDER]))  # plan + 1 group = 2
    assert client.calls == 1


def test_runner_stops_on_the_cap(tmp_path):
    from src.rag.briefing_runner import run_week

    gen = PlanBriefingGenerator(
        CONFIG, retriever=FakeRetriever(), client=FakeClient(), max_model_calls=2
    )
    tracts = [_group_tract("a", ["heat"], [OLDER]), _group_tract("b", ["heat"], [LEP])]
    with pytest.raises(CallCapError):
        run_week(tracts, gen, tmp_path / "ckpt.jsonl")
    # The first briefing was checkpointed before the run stopped
    lines = (tmp_path / "ckpt.jsonl").read_text().splitlines()
    assert [json.loads(line)["geoid"] for line in lines] == ["a"]


def test_summary_names_groups_and_their_shares():
    tract = _group_tract("06037900102", ["heat"], [OLDER, LEP])
    summary, key_risks = tract_summary(tract)
    assert "older adults (65 and over) and limited-English households" in summary
    assert (
        "Older adults (65 and over): 31.2% of residents in the tract "
        "(county median 14.5%)." in key_risks
    )
    assert (
        "Limited-English households: 31.2% of households in the tract "
        "(county median 14.5%)." in key_risks
    )

    briefing = _generator(FakeClient()).generate(tract)
    case = EvalCase(briefing, tract)
    for scorer in (numbers_match_data, urgency_matches, no_system_talk, no_jargon):
        assert scorer(case).value == 1.0, scorer.__name__


def test_group_gaps_are_written_by_code():
    class NoActions(FakeClient):
        def parse(self, **kwargs):
            self.calls += 1
            draft = ActionPlanDraft(
                recommended_actions=[], data_gaps=["Excerpts lack local detail."]
            )
            return SimpleNamespace(stop_reason="end_turn", parsed_output=draft)

    b = _generator(NoActions()).generate(_group_tract("a", ["air_quality"], [LEP]))
    gap = "No guidance in the source documents on air quality for limited-English households."
    assert b.data_gaps.count(gap) == 1
    # The driver plan's gap (same fake text) is kept; the group's is not added
    assert b.data_gaps.count("Excerpts lack local detail.") == 1

    # A group with actions adds no gap line
    b = _generator(FakeClient()).generate(_group_tract("b", ["heat"], [OLDER]))
    assert not any("older adults" in g for g in b.data_gaps)
