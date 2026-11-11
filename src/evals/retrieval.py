"""
retrieval.py
============
A1: retrieval quality on a hand-labeled gold set. No API cost.

Gold set (evals/datasets/retrieval_gold_v1.json): queries with the policy
pages a person judged relevant. Negatives have no relevant pages: the
corpus cannot answer them, so the right result is nothing within the cutoff.

Scoring is per page, not per chunk. Chunk IDs change when chunk size or
overlap changes; pages do not, so the labels survive re-chunking. The
retriever's ranked chunks are collapsed to unique pages (best distance
per page), which is also what a briefing sees: pick_excerpts keeps one
chunk per page.

Each query is retrieved once with the cutoff off. Every cutoff is then
applied to the stored distances, so the sweep scores all cutoffs on the
same rankings and costs one round of queries.

Metrics (positives only unless stated):
    recall@k             share of a query's relevant pages in its top k pages
                         (k=3: what a briefing gets; k=10: the wider pool)
    mrr                  1 / rank of the first relevant page, 0 if none
    ndcg@10              recall that rewards relevant pages near the top
    empty_rate           positives that get nothing within the cutoff
    negative_rejection   negatives that get nothing within the cutoff
"""

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

Page = tuple[str, int]  # (source file, page number)
# One ranked result: page and its best (lowest) cosine distance
Ranked = list[tuple[Page, float]]

K_BRIEFING = 3
K_POOL = 10


@dataclass(frozen=True)
class GoldQuery:
    id: str
    driver: str
    group: str
    query: str
    relevant: frozenset[Page]

    @property
    def is_negative(self) -> bool:
        return not self.relevant


def load_gold(path: str | Path) -> list[GoldQuery]:
    data = json.loads(Path(path).read_text())
    return [
        GoldQuery(
            id=q["id"],
            driver=q["driver"],
            group=q["group"],
            query=q["query"],
            relevant=frozenset((r["source"], r["page"]) for r in q["relevant"]),
        )
        for q in data["queries"]
    ]


def unique_pages(chunks: list[tuple[str, int, float]]) -> Ranked:
    """Collapse ranked chunks (source, page, distance) to pages.

    Pages keep the order of their first chunk (the ranking: embedding order,
    or reranker order when one is used) and the best distance of any of
    their chunks (what the cutoff tests).
    """
    best: dict[Page, float] = {}
    for source, page, distance in chunks:
        key = (source, page)
        if key not in best or distance < best[key]:
            best[key] = distance
    return list(best.items())


def within(ranked: Ranked, cutoff: float) -> list[Page]:
    return [page for page, distance in ranked if distance <= cutoff]


def recall_at_k(pages: list[Page], relevant: frozenset[Page], k: int) -> float:
    return len(set(pages[:k]) & relevant) / len(relevant)


def reciprocal_rank(pages: list[Page], relevant: frozenset[Page]) -> float:
    for rank, page in enumerate(pages, start=1):
        if page in relevant:
            return 1 / rank
    return 0.0


def ndcg_at_k(pages: list[Page], relevant: frozenset[Page], k: int) -> float:
    """Binary nDCG: a relevant page at rank r adds 1/log2(r+1)."""
    dcg = sum(
        1 / math.log2(rank + 1)
        for rank, page in enumerate(pages[:k], start=1)
        if page in relevant
    )
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def score_query(q: GoldQuery, ranked: Ranked, cutoff: float) -> dict[str, Any]:
    pages = within(ranked, cutoff)
    row: dict[str, Any] = {
        "id": q.id,
        "driver": q.driver,
        "negative": q.is_negative,
        "n_within": len(pages),
        "top3": "; ".join(f"{s} p.{p}" for s, p in pages[:K_BRIEFING]),
    }
    if q.is_negative:
        row["rejected"] = not pages
        row["closest"] = round(ranked[0][1], 3) if ranked else None
        return row
    first_hit = next((round(d, 3) for page, d in ranked if page in q.relevant), None)
    row |= {
        f"recall@{K_BRIEFING}": recall_at_k(pages, q.relevant, K_BRIEFING),
        f"recall@{K_POOL}": recall_at_k(pages, q.relevant, K_POOL),
        "rr": reciprocal_rank(pages, q.relevant),
        f"ndcg@{K_POOL}": ndcg_at_k(pages, q.relevant, K_POOL),
        "empty": not pages,
        "best_relevant_distance": first_hit,  # cutoff-free: how close the answer is
    }
    return row


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    pos = [r for r in rows if not r["negative"]]
    neg = [r for r in rows if r["negative"]]
    metrics = {
        f"recall@{K_BRIEFING}": fmean(r[f"recall@{K_BRIEFING}"] for r in pos),
        f"recall@{K_POOL}": fmean(r[f"recall@{K_POOL}"] for r in pos),
        "mrr": fmean(r["rr"] for r in pos),
        f"ndcg@{K_POOL}": fmean(r[f"ndcg@{K_POOL}"] for r in pos),
        "empty_rate": fmean(r["empty"] for r in pos),
    }
    if neg:
        metrics["negative_rejection"] = fmean(r["rejected"] for r in neg)
    return {k: round(v, 4) for k, v in metrics.items()}


def retrieve_all(
    gold: list[GoldQuery], search: Callable[[str], list[tuple[str, int, float]]]
) -> dict[str, Ranked]:
    """Run every query once, cutoff off. search(query) -> ranked chunks."""
    return {q.id: unique_pages(search(q.query)) for q in gold}


def evaluate(
    gold: list[GoldQuery], rankings: dict[str, Ranked], cutoff: float
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows = [score_query(q, rankings[q.id], cutoff) for q in gold]
    return aggregate(rows), rows


def sweep(
    gold: list[GoldQuery], rankings: dict[str, Ranked], cutoffs: list[float]
) -> list[dict[str, float]]:
    """Aggregate metrics at each cutoff, on the same rankings."""
    return [{"cutoff": c, **evaluate(gold, rankings, c)[0]} for c in cutoffs]


def by_driver(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """recall@3 and empty rate per driver (positives only)."""
    out: dict[str, dict[str, float]] = {}
    for driver in sorted({r["driver"] for r in rows if not r["negative"]}):
        part = [r for r in rows if r["driver"] == driver and not r["negative"]]
        out[driver] = {
            f"recall@{K_BRIEFING}": round(
                fmean(r[f"recall@{K_BRIEFING}"] for r in part), 4
            ),
            "empty_rate": round(fmean(r["empty"] for r in part), 4),
            "n": len(part),
        }
    return out
