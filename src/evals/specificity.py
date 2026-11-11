"""
specificity.py
==============
A5: is each briefing about its own tract, or could it be pasted into any other?

Unlike faithfulness and rules, most of these are measured across the whole
week of briefings, not one briefing at a time, so they return one number each
instead of a Score per briefing.

Week-level metrics (specificity_metrics):
    distinct_source_pages     how many different policy pages are cited at all
    top5_source_share         share of all citations that go to the 5 most cited pages
    duplicate_source_sets     share of briefings whose cited pages equal another briefing's
    same_driver_similarity    mean text similarity of briefings with the same drivers
    diff_driver_similarity    mean text similarity of briefings with different drivers
    driver_separation         same minus different; near 0 means the text barely
                              changes when the drivers change
    group_separation          among briefings with the same drivers: similarity
                              with the same vulnerable groups minus with
                              different groups. Near 0 means the group actions
                              barely change the text (Phase 3 made same-driver
                              briefings differ by group, which lowered
                              driver_separation; this measures that directly)

    urgency_separation    among briefings with the same drivers: similarity
                          with the same urgency minus with the other urgency.
                          Phase 5.1 writes plans per urgency level; before
                          that CRITICAL and ELEVATED shared one plan (near 0)

Per-briefing scorers (SCORERS): length and structure.
"""

import itertools
import random
from collections import Counter
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.evals.base import Score, action_text
from src.evals.dataset import EvalCase

# Pairs sampled per group for the similarity means. 626 briefings give ~196k
# pairs; a fixed-seed sample of 5,000 is plenty and fast.
MAX_PAIRS = 5_000
SEED = 42


def specificity_metrics(cases: list[EvalCase]) -> dict[str, Any]:
    citations = [
        cite for c in cases for a in c.briefing.actions for cite in a["citations"]
    ]
    counts = Counter(citations)
    source_sets = Counter(frozenset(c.briefing.sources) for c in cases)

    same, diff = _driver_similarity(cases)
    same_groups, diff_groups = _group_similarity(cases)
    same_urg, diff_urg = _urgency_similarity(cases)
    return {
        "distinct_source_pages": len(counts),
        "top5_source_share": round(
            sum(n for _, n in counts.most_common(5)) / len(citations), 3
        )
        if citations
        else 0.0,
        "duplicate_source_sets": round(
            sum(n for n in source_sets.values() if n > 1) / len(cases), 3
        ),
        "same_driver_similarity": round(same, 3),
        "diff_driver_similarity": round(diff, 3),
        "driver_separation": round(same - diff, 3),
        "group_separation": round(same_groups - diff_groups, 3),
        "urgency_separation": round(same_urg - diff_urg, 3),
        "top5_source_pages": [page for page, _ in counts.most_common(5)],
    }


def action_count(case: EvalCase) -> Score:
    n = len(case.briefing.actions)
    return Score(float(n), "no actions" if n == 0 else "")


def summary_words(case: EvalCase) -> Score:
    return Score(float(len(case.briefing.summary.split())), "")


SCORERS = {
    "action_count": action_count,
    "summary_words": summary_words,
}


# ---------------------------------------------------------------------------
# Private functions
# ---------------------------------------------------------------------------


def _driver_similarity(cases: list[EvalCase]) -> tuple[float, float]:
    """Mean TF-IDF cosine similarity of action text: same-driver vs different-driver pairs."""
    texts = [action_text(c.briefing) for c in cases]
    vectors = TfidfVectorizer(stop_words="english").fit_transform(texts)
    drivers = [frozenset(c.tract.drivers) for c in cases]

    pairs = list(itertools.combinations(range(len(cases)), 2))
    same_pairs = [p for p in pairs if drivers[p[0]] == drivers[p[1]]]
    diff_pairs = [p for p in pairs if drivers[p[0]] != drivers[p[1]]]
    return _mean_similarity(vectors, same_pairs), _mean_similarity(vectors, diff_pairs)


def _group_similarity(cases: list[EvalCase]) -> tuple[float, float]:
    """Same-driver pairs only: same vulnerable groups vs different groups."""
    texts = [action_text(c.briefing) for c in cases]
    vectors = TfidfVectorizer(stop_words="english").fit_transform(texts)
    drivers = [frozenset(c.tract.drivers) for c in cases]
    groups = [frozenset(c.tract.vulnerable_groups) for c in cases]

    pairs = [
        p
        for p in itertools.combinations(range(len(cases)), 2)
        if drivers[p[0]] == drivers[p[1]]
    ]
    same = [p for p in pairs if groups[p[0]] == groups[p[1]]]
    diff = [p for p in pairs if groups[p[0]] != groups[p[1]]]
    return _mean_similarity(vectors, same), _mean_similarity(vectors, diff)


def _urgency_similarity(cases: list[EvalCase]) -> tuple[float, float]:
    """Same-driver pairs only: same urgency vs different urgency."""
    texts = [action_text(c.briefing) for c in cases]
    vectors = TfidfVectorizer(stop_words="english").fit_transform(texts)
    drivers = [frozenset(c.tract.drivers) for c in cases]
    urgency = [c.tract.urgency for c in cases]

    pairs = [
        p
        for p in itertools.combinations(range(len(cases)), 2)
        if drivers[p[0]] == drivers[p[1]]
    ]
    same = [p for p in pairs if urgency[p[0]] == urgency[p[1]]]
    diff = [p for p in pairs if urgency[p[0]] != urgency[p[1]]]
    return _mean_similarity(vectors, same), _mean_similarity(vectors, diff)


def _mean_similarity(vectors, pairs: list[tuple[int, int]]) -> float:
    if not pairs:
        return float("nan")
    if len(pairs) > MAX_PAIRS:
        pairs = random.Random(SEED).sample(pairs, MAX_PAIRS)
    left = vectors[[i for i, _ in pairs]]
    right = vectors[[j for _, j in pairs]]
    # Row-wise cosine: TF-IDF rows are L2-normalised, so the dot product is the cosine
    sims = np.asarray(left.multiply(right).sum(axis=1)).ravel()
    return float(sims.mean())


__all__ = ["specificity_metrics", "SCORERS"]
