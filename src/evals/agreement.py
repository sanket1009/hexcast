"""
agreement.py
============
A6 judge validation: how well do the judge's scores match the user's own?

Weighted Cohen's kappa (quadratic weights), per criterion:
    Plain agreement rewards luck: if both raters give almost every briefing a
    4, they agree often even if the judge is not reading carefully. Kappa
    subtracts the agreement expected by chance from each rater's own score
    distribution. Quadratic weights count a 4-vs-5 disagreement as small and
    a 1-vs-5 as large, which fits an ordered 1-5 scale.

    kappa = 1 - sum(w * observed) / sum(w * expected), w_ij = (i - j)^2 / 16
    1 = perfect, 0 = chance level, below 0 = worse than chance.
    Rule of thumb (Landis & Koch): 0.61-0.80 substantial, 0.81+ almost perfect.

Kappa is undefined when both raters use a single score for every item (no
variance to explain); those criteria fall back to exact agreement and are
reported as such. With 20 items kappa is noisy, so exact and within-1
agreement and the mean difference (judge minus human) are reported next to it.
"""

import csv
from pathlib import Path
from statistics import fmean
from typing import Any

from src.evals.judge import CRITERIA

SCALE = range(1, 6)


def weighted_kappa(a: list[int], b: list[int]) -> float:
    """Quadratic weighted Cohen's kappa on a 1-5 scale. NaN when undefined."""
    if len(a) != len(b) or not a:
        raise ValueError("Need two equal, non-empty lists of scores")
    k = len(SCALE)
    n = len(a)
    observed = [[0.0] * k for _ in SCALE]
    for x, y in zip(a, b, strict=True):
        observed[x - 1][y - 1] += 1 / n
    hist_a = [sum(row) for row in observed]
    hist_b = [sum(observed[i][j] for i in range(k)) for j in range(k)]

    def w(i: int, j: int) -> float:
        return (i - j) ** 2 / (k - 1) ** 2

    num = sum(w(i, j) * observed[i][j] for i in range(k) for j in range(k))
    den = sum(w(i, j) * hist_a[i] * hist_b[j] for i in range(k) for j in range(k))
    return float("nan") if den == 0 else 1 - num / den


def read_human_labels(path: Path) -> dict[str, dict[str, Any]]:
    """Filled rows of the label CSV, keyed by item_id. Rows with blanks are skipped."""
    labels = {}
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            values = [row.get(c, "").strip() for c in CRITERIA]
            unsafe = row.get("unsafe_advice", "").strip().lower()
            if not all(values) or unsafe not in ("yes", "no"):
                continue
            scores = {c: int(v) for c, v in zip(CRITERIA, values, strict=True)}
            bad = [c for c, s in scores.items() if s not in SCALE]
            if bad:
                raise ValueError(f"{row['item_id']}: scores out of 1-5 for {bad}")
            labels[row["item_id"]] = {**scores, "unsafe_advice": unsafe == "yes"}
    return labels


def agreement(
    human: dict[str, dict[str, Any]],
    judge: dict[str, dict[str, Any]],
    min_kappa: float,
) -> dict[str, Any]:
    """Per-criterion agreement on the items both rated, and a trust verdict."""
    ids = sorted(human.keys() & judge.keys())
    if not ids:
        raise ValueError("No items rated by both the user and the judge")
    out: dict[str, Any] = {"n": len(ids), "criteria": {}}
    for c in CRITERIA:
        h = [human[i][c] for i in ids]
        j = [judge[i][c] for i in ids]
        kappa = weighted_kappa(h, j)
        exact = fmean(x == y for x, y in zip(h, j, strict=True))
        if kappa == kappa:  # not NaN
            trusted = kappa >= min_kappa
            basis = "kappa"
        else:
            trusted = exact >= 0.9
            basis = "exact agreement (no score variance, kappa undefined)"
        out["criteria"][c] = {
            "kappa": None if kappa != kappa else round(kappa, 3),
            "exact": round(exact, 3),
            "within_1": round(
                fmean(abs(x - y) <= 1 for x, y in zip(h, j, strict=True)), 3
            ),
            "mean_diff": round(fmean(j) - fmean(h), 3),  # judge minus human
            "trusted": trusted,
            "basis": basis,
        }
    h_unsafe = [human[i]["unsafe_advice"] for i in ids]
    j_unsafe = [judge[i]["unsafe_advice"] for i in ids]
    out["unsafe_advice"] = {
        "agreement": round(
            fmean(x == y for x, y in zip(h_unsafe, j_unsafe, strict=True)), 3
        ),
        "human_yes": sum(h_unsafe),
        "judge_yes": sum(j_unsafe),
    }
    out["trusted_criteria"] = [c for c in CRITERIA if out["criteria"][c]["trusted"]]
    return out
