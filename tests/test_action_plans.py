"""
Tests for shared action plans: one model call per driver combination, plan
reuse from the cache file, and the tract summary template. Fake retriever and
client, so no model calls.
"""

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
from src.rag.briefing import Action, BriefingError, TractRisk
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
