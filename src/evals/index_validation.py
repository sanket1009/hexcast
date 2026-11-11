"""
index_validation.py
===================
B1 checks for the compound burden index. No API cost.

1. Weight sensitivity (one week): rescore with each component's weight
   changed (x0.75, x1.25, and x0 = dropped) and compare to equal weights.
   - spearman: rank correlation of raw burden with the base ranking
   - jaccard: overlap of the CRITICAL sets, |A and B| / |A or B|
   - left_critical: base CRITICAL tracts that are no longer CRITICAL
   Rescoring uses compound_score._build_composite, so the classes match
   production exactly (including its rounded percentile cutoffs).

2. Temporal stability (all stored weeks): how much the CRITICAL list
   changes from one week to the next.
   - retained: share of last week's CRITICAL tracts still CRITICAL
   - spearman_prev: rank correlation of raw burden with last week
   - ever_critical: tracts CRITICAL in at least one week
   - persistent: tracts CRITICAL in at least half of the weeks

A stable index keeps most of its CRITICAL tracts week to week. Programs
cannot plan outreach around a list that changes by half every week.
"""

from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.spatial.compound_score import SIGNED_SIGNALS, _build_composite

CRITICAL = "CRITICAL"


def jaccard(a: set, b: set) -> float:
    """Overlap of two sets: 1.0 identical, 0.0 disjoint."""
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def critical_set(scores: pd.DataFrame) -> set[str]:
    return set(scores.loc[scores["burden_class"] == CRITICAL, "geoid"])


def weight_sensitivity(
    signals: pd.DataFrame,
    config: dict[str, Any],
    factors: tuple[float, ...] = (0.75, 1.25, 0.0),
) -> pd.DataFrame:
    """
    Rescore one week with each component's weight multiplied by each factor.

    signals: output of _load_signals + _impute_missing (no missing values).
    Returns one row per (component, factor).
    """
    base = _build_composite(signals, config)
    base_critical = critical_set(base)

    rows = []
    for col in SIGNED_SIGNALS:
        for factor in factors:
            out = _build_composite(signals, config, weights={col: factor})
            crit = critical_set(out)
            rows.append(
                {
                    "component": col,
                    "weight": factor,
                    "spearman": float(
                        spearmanr(base["raw_burden"], out["raw_burden"])[0]
                    ),
                    "jaccard": jaccard(base_critical, crit),
                    "left_critical": len(base_critical - crit),
                }
            )
    return pd.DataFrame(rows)


def temporal_stability(scores: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Week-to-week stability of stored compound scores.

    scores: columns week_start_date, geoid, raw_burden, burden_class
    (several weeks). Returns (per-week table, summary metrics).
    """
    scores = scores.copy()
    scores["week_start_date"] = pd.to_datetime(scores["week_start_date"])
    weeks = sorted(scores["week_start_date"].unique())
    if len(weeks) < 2:
        raise ValueError("Temporal stability needs at least two weeks of scores")

    crit = {w: critical_set(scores[scores["week_start_date"] == w]) for w in weeks}
    burden = scores.pivot(index="geoid", columns="week_start_date", values="raw_burden")

    rows = []
    for prev, cur in pairwise(weeks):
        both = burden[[prev, cur]].dropna()
        rows.append(
            {
                "week_start_date": pd.Timestamp(cur).date(),
                "retained": len(crit[prev] & crit[cur]) / len(crit[prev])
                if crit[prev]
                else np.nan,
                "spearman_prev": float(spearmanr(both[prev], both[cur])[0]),
                "n_critical": len(crit[cur]),
            }
        )
    per_week = pd.DataFrame(rows)

    weeks_critical = pd.Series(
        [g for w in weeks for g in crit[w]], dtype="object"
    ).value_counts()
    summary = {
        "n_weeks": len(weeks),
        "retained_median": float(per_week["retained"].median()),
        "retained_min": float(per_week["retained"].min()),
        "spearman_prev_median": float(per_week["spearman_prev"].median()),
        "ever_critical": int(len(weeks_critical)),
        "persistent_critical": int((weeks_critical >= len(weeks) / 2).sum()),
    }
    return per_week, summary
