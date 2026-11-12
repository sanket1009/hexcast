"""
compare_stats.py
================
Is one scorer really better than another on the out-of-fold rows?

block_bootstrap_diff
    Resample spatial blocks (H3 res-6 cells) with replacement, keeping every
    tract-week of a block together, and recompute the PR-AUC difference. The
    2.5-97.5 percentiles of the differences give a 95% CI. Neighboring tracts
    and repeated weeks of one tract are not independent; resampling blocks
    keeps that dependence inside each draw. Week-to-week correlation across
    blocks (shared weather) is not removed, so the CI is still somewhat narrow.

delong_paired
    DeLong's test for two correlated ROC-AUCs (fast version, Sun & Xu 2014).
    It treats rows as independent, so its p-value is optimistic here; it is
    reported next to the bootstrap CI, not instead of it.
"""

import numpy as np
from scipy.stats import norm, rankdata


def _weighted_ap_presorted(
    y_sorted: np.ndarray, w_sorted: np.ndarray, group_end: np.ndarray
) -> float:
    """Step-wise average precision (as sklearn) with row weights.

    Rows are sorted by score descending; group_end marks the last row of each
    run of tied scores, so ties are scored at one threshold like sklearn does.
    """
    tp = np.cumsum(w_sorted * y_sorted)[group_end]
    seen = np.cumsum(w_sorted)[group_end]
    total_tp = tp[-1]
    if total_tp == 0:
        return float("nan")
    precision = np.divide(tp, seen, out=np.zeros_like(tp), where=seen > 0)
    recall_step = np.diff(np.concatenate([[0.0], tp])) / total_tp
    return float(np.sum(recall_step * precision))


def _presort(score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-score, kind="mergesort")
    s = score[order]
    group_end = np.flatnonzero(np.concatenate([s[1:] != s[:-1], [True]]))
    return order, group_end


def block_bootstrap_diff(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    blocks: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """PR-AUC(a) - PR-AUC(b): point estimate, 95% CI, share of draws <= 0."""
    y = np.asarray(y, dtype=float)
    codes, block_idx = np.unique(np.asarray(blocks), return_inverse=True)
    rng = np.random.default_rng(seed)
    sorted_a, sorted_b = _presort(np.asarray(score_a)), _presort(np.asarray(score_b))
    ones = np.ones(len(codes))

    def ap(order_groups, block_w):
        order, groups = order_groups
        return _weighted_ap_presorted(y[order], block_w[block_idx][order], groups)

    point = ap(sorted_a, ones) - ap(sorted_b, ones)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        counts = np.bincount(
            rng.integers(0, len(codes), len(codes)), minlength=len(codes)
        )
        w = counts.astype(float)
        diffs[i] = ap(sorted_a, w) - ap(sorted_b, w)
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return {
        "diff": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "share_le_0": float(np.mean(diffs <= 0)),
        "n_blocks": int(len(codes)),
        "n_boot": int(n_boot),
    }


def delong_paired(
    y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray
) -> dict[str, float]:
    """ROC-AUC(a) - ROC-AUC(b) with DeLong's z and two-sided p-value."""
    y = np.asarray(y).astype(bool)
    m, n = int(y.sum()), int((~y).sum())
    v10, v01, aucs = [], [], []
    for s in (np.asarray(score_a, float), np.asarray(score_b, float)):
        pos, neg = s[y], s[~y]
        tz = rankdata(np.concatenate([pos, neg]))
        tx, ty = rankdata(pos), rankdata(neg)
        aucs.append((tz[:m].sum() - m * (m + 1) / 2) / (m * n))
        v10.append((tz[:m] - tx) / n)
        v01.append(1 - (tz[m:] - ty) / m)
    s10, s01 = np.cov(np.vstack(v10)), np.cov(np.vstack(v01))
    cov = s10 / m + s01 / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = aucs[0] - aucs[1]
    z = diff / np.sqrt(var) if var > 0 else float("nan")
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "diff": float(diff),
        "z": float(z),
        "p_value": float(2 * norm.sf(abs(z))),
    }
