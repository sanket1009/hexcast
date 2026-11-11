"""Tests for the A9 CI gate: frozen A1 rankings round-trip and regressions are caught."""

import json

import pytest

from src.evals import gate
from src.evals import retrieval as a1


def _gold(tmp_path):
    path = tmp_path / "gold.json"
    path.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "id": "q1",
                        "driver": "heat",
                        "group": "older adults",
                        "query": "cool",
                        "relevant": [{"source": "a.pdf", "page": 1}],
                    },
                    {
                        "id": "neg",
                        "driver": "heat",
                        "group": "x",
                        "query": "water rates",
                        "relevant": [],
                    },
                ]
            }
        )
    )
    return path


def test_frozen_rankings_reproduce_the_metrics(tmp_path):
    gold_path = _gold(tmp_path)
    rankings = {
        "q1": [(("b.pdf", 2), 0.30), (("a.pdf", 1), 0.35)],
        "neg": [(("c.pdf", 3), 0.6)],
    }
    frozen = gate.save_rankings(tmp_path / "r.json", str(gold_path), 0.48, rankings, {})
    metrics = gate.retrieval_metrics(frozen)
    expected, _ = a1.evaluate(a1.load_gold(gold_path), rankings, 0.48)
    assert metrics == expected
    assert metrics["mrr"] == 0.5 and metrics["negative_rejection"] == 1.0


def test_changed_gold_set_is_refused(tmp_path):
    gold_path = _gold(tmp_path)
    frozen = gate.save_rankings(
        tmp_path / "r.json", str(gold_path), 0.48, {"q1": [], "neg": []}, {}
    )
    gold_path.write_text(gold_path.read_text() + " ")
    with pytest.raises(ValueError):
        gate.retrieval_metrics(frozen)


def test_retrieval_regressions(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "BASELINES_DIR", tmp_path)
    assert gate.retrieval_regressions({"mrr": 0.7}) == [
        f"missing baseline {tmp_path / 'ci_retrieval.json'}"
    ]
    gate.write_retrieval_baseline({"mrr": 0.7, "empty_rate": 0.0}, {})
    assert gate.retrieval_regressions({"mrr": 0.698, "empty_rate": 0.0}) == []
    found = gate.retrieval_regressions({"mrr": 0.6, "empty_rate": 0.1})
    assert len(found) == 2
