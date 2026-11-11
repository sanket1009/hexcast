"""
Tests for the A1 retrieval metrics: page collapsing, recall/MRR/nDCG on
hand-built rankings, negatives, and the cutoff sweep. No Chroma, no model.
"""

import json

import pytest

from src.evals import retrieval as a1

REL = frozenset({("a.pdf", 1), ("a.pdf", 2)})


def _q(qid, relevant=REL, driver="heat"):
    return a1.GoldQuery(qid, driver, "", f"query {qid}", frozenset(relevant))


def test_chunks_collapse_to_pages_with_best_distance():
    chunks = [("a.pdf", 1, 0.40), ("a.pdf", 1, 0.20), ("b.pdf", 3, 0.30)]
    assert a1.unique_pages(chunks) == [(("a.pdf", 1), 0.20), (("b.pdf", 3), 0.30)]


def test_pages_keep_ranking_order_not_distance_order():
    # A reranker put the farther chunk first; scoring must keep that order
    chunks = [("b.pdf", 3, 0.40), ("a.pdf", 1, 0.20), ("b.pdf", 3, 0.35)]
    assert a1.unique_pages(chunks) == [(("b.pdf", 3), 0.35), (("a.pdf", 1), 0.20)]


def test_recall_mrr_ndcg_on_a_known_ranking():
    pages = [("x.pdf", 9), ("a.pdf", 2), ("a.pdf", 1)]
    assert a1.recall_at_k(pages, REL, 1) == 0.0
    assert a1.recall_at_k(pages, REL, 3) == 1.0
    assert a1.reciprocal_rank(pages, REL) == 0.5
    # hits at ranks 2 and 3 vs ideal ranks 1 and 2
    ideal = 1 + 1 / 1.584962500721156
    got = 1 / 1.584962500721156 + 1 / 2
    assert a1.ndcg_at_k(pages, REL, 10) == pytest.approx(got / ideal)


def test_perfect_ranking_scores_one():
    pages = [("a.pdf", 1), ("a.pdf", 2)]
    assert a1.ndcg_at_k(pages, REL, 10) == pytest.approx(1.0)
    assert a1.reciprocal_rank(pages, REL) == 1.0


def test_cutoff_drops_far_pages_and_can_empty_a_query():
    ranked = [(("a.pdf", 1), 0.45), (("a.pdf", 2), 0.55)]
    row = a1.score_query(_q("p"), ranked, cutoff=0.5)
    assert row["recall@3"] == 0.5 and not row["empty"]
    row = a1.score_query(_q("p"), ranked, cutoff=0.4)
    assert row["recall@3"] == 0.0 and row["empty"]
    assert row["best_relevant_distance"] == 0.45  # reported even when cut off


def test_negative_is_rejected_only_when_nothing_is_within_cutoff():
    neg = _q("n", relevant=())
    ranked = [(("z.pdf", 4), 0.48)]
    assert a1.score_query(neg, ranked, cutoff=0.45)["rejected"]
    assert not a1.score_query(neg, ranked, cutoff=0.5)["rejected"]


def test_sweep_trades_recall_for_rejection():
    gold = [_q("p"), _q("n", relevant=())]
    rankings = {
        "p": [(("a.pdf", 1), 0.40), (("a.pdf", 2), 0.60)],
        "n": [(("z.pdf", 1), 0.55)],
    }
    low, high = a1.sweep(gold, rankings, [0.45, 0.65])
    assert low["recall@3"] == 0.5 and low["negative_rejection"] == 1.0
    assert high["recall@3"] == 1.0 and high["negative_rejection"] == 0.0


def test_retrieve_all_calls_search_once_per_query():
    calls = []

    def search(query):
        calls.append(query)
        return [("a.pdf", 1, 0.3)]

    gold = [_q("p1"), _q("p2")]
    rankings = a1.retrieve_all(gold, search)
    assert calls == ["query p1", "query p2"]
    assert rankings["p1"] == [(("a.pdf", 1), 0.3)]


def test_by_driver_splits_positives():
    rows = [
        {"driver": "heat", "negative": False, "recall@3": 1.0, "empty": False},
        {"driver": "heat", "negative": False, "recall@3": 0.0, "empty": True},
        {"driver": "air_quality", "negative": False, "recall@3": 0.5, "empty": False},
        {"driver": "", "negative": True},
    ]
    out = a1.by_driver(rows)
    assert out["heat"] == {"recall@3": 0.5, "empty_rate": 0.5, "n": 2}
    assert set(out) == {"air_quality", "heat"}


def test_gold_set_file_is_well_formed():
    data = json.loads(open("evals/datasets/retrieval_gold_v1.json").read())
    gold = a1.load_gold("evals/datasets/retrieval_gold_v1.json")
    assert len(gold) == len(data["queries"]) == len({q.id for q in gold})
    assert sum(q.is_negative for q in gold) >= 5
    assert all(q.query for q in gold)
    for q in gold:
        for source, page in q.relevant:
            assert source.endswith(".pdf") and page >= 1


def test_paired_comparison_counts_wins_and_brackets_the_mean():
    a = [
        {"id": f"q{i}", "negative": False, "recall@3": 0.0, "rr": 0.5, "ndcg@10": 0.5}
        for i in range(10)
    ]
    b = [{**r, "recall@3": 1.0 if i < 6 else 0.0} for i, r in enumerate(a)]
    out = a1.paired_comparison(a, b, n_boot=2000)
    r3 = out["recall@3"]
    assert (r3["wins"], r3["losses"], r3["ties"]) == (6, 0, 4)
    assert r3["ci_low"] <= r3["mean_diff"] == 0.6 <= r3["ci_high"]
    assert out["rr"]["mean_diff"] == 0.0 and out["rr"]["ties"] == 10


def test_negative_diagnostics_use_the_minimum_distance():
    # reranked order: farther page first
    ranked = [(("z.pdf", 1), 0.58), (("y.pdf", 2), 0.51)]
    row = a1.score_query(_q("n", relevant=()), ranked, cutoff=0.48)
    assert row["closest"] == 0.51 and row["rejected"]
