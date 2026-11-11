"""
Tests for the A6 judge: sampling, the judge call and its cache, the human
sheet, and the agreement statistics. Fake client, so no model calls.
"""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
from sklearn.metrics import cohen_kappa_score

from src.evals import judge as a6
from src.evals.agreement import agreement, read_human_labels, weighted_kappa
from src.evals.dataset import EvalCase
from src.evals.sampling import stratified_sample, stratum
from src.rag.briefing import Briefing, TractRisk
from src.rag.retriever import RetrievedChunk

CONFIG = {
    "evals": {
        "judge": {
            "model": "claude-sonnet-5-5",
            "max_tokens": 4000,
            "rubric": "evals/rubrics/briefing_v1.md",
            "max_calls": 10,
        }
    }
}
DRIVER_SETS = [["heat"], ["air_quality"], ["air_quality", "heat"]]


def _case(i, urgency="CRITICAL", drivers=("heat",), groups=()):
    tract = TractRisk(
        geoid=f"06037{i:06d}",
        risk_rank=i,
        tract_count=2498,
        urgency=urgency,
        drivers=list(drivers),
        indicators={"Land surface temperature, mean (°C)": 50.0},
        vulnerable_groups=list(groups),
    )
    briefing = Briefing(
        geoid=tract.geoid,
        urgency=urgency,
        summary=f"Critical urgency this week. Rank {i}.",
        key_risks=["Heat: 50 °C."],
        actions=[{"action": "Open cooling centers.", "citations": ["plan.pdf, p. 1"]}],
        data_gaps=[],
        sources=["plan.pdf, p. 1"],
        retrieved_chunk_ids=["c1"],
        model="claude-haiku-5-5",
        generated_at="2026-11-11T00:00:00+00:00",
        vulnerable_groups=list(groups),
        inputs=asdict(tract),
    )
    return EvalCase(briefing, tract)


def _cases(n=90):
    out = []
    for i in range(n):
        urgency = "CRITICAL" if i % 3 else "ELEVATED"
        groups = [["older adults"], ["people with disabilities"], []][i % 3]
        out.append(_case(i, urgency, DRIVER_SETS[i % len(DRIVER_SETS)], groups))
    return out


CHUNKS = {
    "c1": RetrievedChunk(
        "c1", "Open cooling centers in heat waves.", "plan.pdf", 1, 0.0
    )
}


def _verdict(score=4, unsafe=False):
    s = a6.CriterionScore(reason="ok", score=score)
    return a6.JudgeVerdict(
        groundedness=s,
        tract_specificity=s,
        actionability=s,
        urgency_tone=s,
        clarity=s,
        unsafe_reason="",
        unsafe_advice=unsafe,
    )


class FakeClient:
    def __init__(self, score=4):
        self.calls = []
        self.score = score
        self.messages = self

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            stop_reason="end_turn",
            parsed_output=_verdict(self.score),
            usage=SimpleNamespace(input_tokens=3000, output_tokens=600),
        )


# -- sampling ------------------------------------------------------------------


def test_sample_is_deterministic_and_covers_every_stratum():
    cases = _cases()
    a = stratified_sample(cases, 20, seed=42)
    b = stratified_sample(list(reversed(cases)), 20, seed=42)
    assert [c.briefing.geoid for c in a] == [c.briefing.geoid for c in b]
    assert len(a) == 20
    assert {stratum(c) for c in a} == {stratum(c) for c in cases}


def test_sample_prefers_different_group_sets_within_a_stratum():
    cases = [_case(i, groups=[f"g{i % 3}"]) for i in range(30)]
    picked = stratified_sample(cases, 3, seed=1)
    assert len({tuple(c.tract.vulnerable_groups) for c in picked}) == 3


def test_too_many_strata_for_the_sample_raises():
    with pytest.raises(ValueError):
        stratified_sample(_cases(), 2, floor=1)


# -- sample file and human sheet -------------------------------------------------


def test_sample_round_trips_and_flags_human_subset(tmp_path):
    items = a6.build_sample(_cases(), CHUNKS, "2026-06-22", n=12, n_human=6)
    assert [i.item_id for i in items][:2] == ["J01", "J02"]
    assert sum(i.human for i in items) == 6
    path = a6.write_sample(items, tmp_path / "s.jsonl")
    loaded = a6.load_sample(path)
    assert [a6.render_item(i) for i in loaded] == [a6.render_item(i) for i in items]
    assert loaded[0].case().tract.geoid == items[0].briefing.geoid


def test_render_shows_excerpts_citations_and_group_labels():
    item = a6.build_sample(_cases(3), CHUNKS, "w", n=3, n_human=3)[0]
    item.briefing.actions.append(
        {
            "action": "Call older adults.",
            "citations": ["plan.pdf, p. 1"],
            "group": "older adults",
        }
    )
    text = a6.render_item(item)
    assert '<excerpt source="plan.pdf, p. 1">' in text
    assert "[for older adults] Call older adults." in text
    assert "Urgency level: " in text


def test_human_sheet_never_overwrites_labels(tmp_path):
    items = a6.build_sample(_cases(), CHUNKS, "w", n=12, n_human=4)
    csv_path, md_path = tmp_path / "labels.csv", tmp_path / "review.md"
    a6.write_human_sheet(items, "RUBRIC", csv_path, md_path)
    assert len(csv_path.read_text().splitlines()) == 5  # header + 4
    csv_path.write_text("filled")
    a6.write_human_sheet(items, "RUBRIC", csv_path, md_path)
    assert csv_path.read_text() == "filled"
    assert md_path.read_text().endswith("RUBRIC")


# -- judge ----------------------------------------------------------------------


def test_judge_caches_by_input(tmp_path):
    items = a6.build_sample(_cases(), CHUNKS, "w", n=6, n_human=3)
    client = FakeClient()
    judge = a6.Judge(CONFIG, client=client, cache_path=tmp_path / "cache.jsonl")
    first = judge.judge(items[0])
    assert first["verdict"]["groundedness"]["score"] == 4
    assert "temperature" not in client.calls[0]
    assert judge.input_tokens == 3000

    again = a6.Judge(CONFIG, client=client, cache_path=tmp_path / "cache.jsonl")
    assert again.judge(items[0]) == first and len(client.calls) == 1
    assert again.cache_hits == 1

    items[0].briefing.actions[0]["action"] = "Something else."
    again.judge(items[0])  # changed briefing: judged again
    assert len(client.calls) == 2


def test_out_of_range_score_is_an_error():
    item = a6.build_sample(_cases(), CHUNKS, "w", n=6, n_human=3)[0]
    with pytest.raises(a6.JudgeError):
        a6.Judge(CONFIG, client=FakeClient(score=7), cache_path=None).judge(item)


def test_call_cap():
    items = a6.build_sample(_cases(), CHUNKS, "w", n=6, n_human=3)
    config = {"evals": {"judge": CONFIG["evals"]["judge"] | {"max_calls": 1}}}
    judge = a6.Judge(config, client=FakeClient(), cache_path=None)
    judge.judge(items[0])
    with pytest.raises(a6.JudgeError):
        judge.judge(items[1])


def test_metrics_and_strata():
    items = a6.build_sample(_cases(), CHUNKS, "w", n=6, n_human=3)
    judge = a6.Judge(CONFIG, client=FakeClient(), cache_path=None)
    rows = [a6.scores_row(i, judge.judge(i)) for i in items]
    m = a6.judge_metrics(rows)
    assert m["judge.clarity.mean"] == 4 and m["judge.clarity.share_4_plus"] == 1
    assert m["judge.unsafe_rate"] == 0 and m["n_judged"] == 6
    assert sum(v["n"] for v in a6.by_stratum(rows).values()) == 6


# -- agreement ------------------------------------------------------------------


def test_weighted_kappa_matches_sklearn():
    a = [1, 2, 3, 4, 5, 4, 3, 2, 5, 4, 4, 3]
    b = [1, 3, 3, 4, 4, 5, 3, 2, 5, 3, 4, 2]
    expected = cohen_kappa_score(a, b, weights="quadratic", labels=[1, 2, 3, 4, 5])
    assert weighted_kappa(a, b) == pytest.approx(expected)
    assert weighted_kappa(a, a) == pytest.approx(1.0)


def test_kappa_is_undefined_without_variance():
    assert weighted_kappa([4, 4, 4], [4, 4, 4]) != weighted_kappa([4, 4, 4], [4, 4, 4])


def test_agreement_report(tmp_path):
    csv_path = tmp_path / "labels.csv"
    header = "item_id," + ",".join(a6.CRITERIA) + ",unsafe_advice,notes\n"
    csv_path.write_text(
        header
        + "J01,5,4,3,2,1,no,\n"
        + "J02,4,4,4,4,4,yes,\n"
        + "J03,,,,,,,\n"  # unfilled: skipped
    )
    human = read_human_labels(csv_path)
    assert set(human) == {"J01", "J02"}
    judged = {
        "J01": dict(zip(a6.CRITERIA, [5, 4, 3, 2, 1], strict=True))
        | {"unsafe_advice": False},
        "J02": dict(zip(a6.CRITERIA, [4, 4, 4, 4, 5], strict=True))
        | {"unsafe_advice": False},
    }
    result = agreement(human, judged, min_kappa=0.61)
    assert result["n"] == 2
    assert result["criteria"]["groundedness"]["kappa"] == 1.0
    assert result["criteria"]["tract_specificity"]["kappa"] is None  # both all 4s
    assert result["criteria"]["tract_specificity"]["trusted"]  # exact agreement 1.0
    assert result["criteria"]["clarity"]["mean_diff"] == 0.5
    assert result["unsafe_advice"] == {"agreement": 0.5, "human_yes": 1, "judge_yes": 0}


def test_labels_out_of_range_raise(tmp_path):
    csv_path = tmp_path / "labels.csv"
    header = "item_id," + ",".join(a6.CRITERIA) + ",unsafe_advice,notes\n"
    csv_path.write_text(header + "J01,6,4,3,2,1,no,\n")
    with pytest.raises(ValueError):
        read_human_labels(csv_path)
