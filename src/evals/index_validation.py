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

3. External benchmark (year average vs CalEnviroScreen 5.0 and CDC PLACES):
   convergent validity. Each of our components is compared with the CES
   measure closest to it, so a low overall agreement can be traced to the
   component that causes it.
   - Spearman of year-average burden with the CES score, its pollution
     and population parts, and CDC PLACES outcomes
   - top_decile_overlap: tracts in the top 10% of both indexes
   - the same two numbers with access removed (PM2.5 + heat only)
   - with a vulnerability component (Phase 3): hazards only vs CES, and
     vulnerability vs CES population characteristics
   - CES asthma and cardiovascular ED visit rates (observed outcomes).
     CDC PLACES asthma is a model-based estimate driven by demographics;
     in LA it tracks the Asian share of a tract at -0.63, so it is kept
     for reference but the ED rates are the outcome check.

4. Hotspot stability (all stored weeks): Gi* hotspots week to week.
   - retained: share of last week's hotspots still hotspots
   - rank_spearman_prev: rank correlation of Gi* z with last week
   - top_overlap: overlap of the top_n tracts by Gi* z with last week
     (a fixed-size set, so it does not depend on the significance cutoff)
   - count_skew_spearman: rank correlation of the weekly hotspot count
     with the skewness of raw burden. A high value means the count moves
     with the shape of the week's distribution, not with where burden is.

5. Kriging over the year: leave-one-out MAE per week and weak-week flags
   (MAE above mae_limit, or nugget >= partial sill: the variogram finds
   no spatial structure beyond noise). range_at_bound marks weeks whose
   fitted range sits at the fit's upper bound (3x the longest lag): the
   variogram is still rising at the longest distance, so the range is
   not estimated.
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


def top_decile(values: pd.Series, geoids: pd.Series) -> set[str]:
    """GEOIDs above the 90th percentile of values."""
    return set(geoids[values.rank(pct=True) > 0.9])


def external_benchmark(
    annual: pd.DataFrame,
    ces: pd.DataFrame,
    places: pd.DataFrame | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Compare year-average burden with CalEnviroScreen and CDC PLACES.

    annual: geoid, burden, pm25_z, lst_z, access_z, optional
        vulnerability_z (year means of the stored signed components) and
        pm25_ugm3 (year mean Kriged PM2.5)
    ces: calenviroscreen table (ces_score, pollution_score,
        pop_char_score, pm25_ces)
    places: optional cdc_places_health table

    Returns (metrics, comparison table with one row per pair compared).
    Tracts without a CES score are left out.
    """
    d = annual.merge(ces, on="geoid").dropna(subset=["ces_score"])
    d["burden_no_access"] = d["pm25_z"] + d["lst_z"]

    pairs = [
        ("burden", "ces_score", "overall vs overall"),
        ("burden", "pollution_score", "overall vs CES pollution burden"),
        ("burden", "pop_char_score", "overall vs CES population characteristics"),
        ("burden_no_access", "ces_score", "PM2.5 + heat only vs overall"),
        ("pm25_ugm3", "pm25_ces", "Kriged PM2.5 vs CES PM2.5"),
        ("pm25_z", "pollution_score", "PM2.5 component vs CES pollution burden"),
        ("lst_z", "ces_score", "heat component vs CES overall"),
        ("access_z", "ces_score", "access component vs CES overall"),
    ]
    has_vulnerability = "vulnerability_z" in d
    if has_vulnerability:
        d["burden_hazards_only"] = d["burden"] - d["vulnerability_z"]
        pairs += [
            (
                "burden_hazards_only",
                "ces_score",
                "hazards only (no vulnerability) vs overall",
            ),
            (
                "vulnerability_z",
                "pop_char_score",
                "vulnerability component vs CES population characteristics",
            ),
            ("vulnerability_z", "ces_score", "vulnerability component vs CES overall"),
        ]
    for col, label in [
        ("asthma_ed_rate", "asthma ED visits"),
        ("cardio_ed_rate", "cardiovascular ED visits"),
    ]:
        if col in d:
            pairs.append(("burden", col, f"overall vs CES {label}"))
            pairs.append(("pm25_z", col, f"PM2.5 component vs CES {label}"))
            pairs.append(("lst_z", col, f"heat component vs CES {label}"))
    if places is not None:
        d = d.merge(places, on="geoid", how="left")
        for col in ["asthma_pct", "heart_disease_pct", "copd_pct", "diabetes_pct"]:
            pairs.append(("burden", col, f"overall vs CDC PLACES {col}"))

    rows = []
    for ours, theirs, label in pairs:
        both = d[[ours, theirs]].dropna()
        rows.append(
            {
                "comparison": label,
                "ours": ours,
                "theirs": theirs,
                "n": len(both),
                "spearman": float(spearmanr(both[ours], both[theirs])[0]),
            }
        )
    table = pd.DataFrame(rows)

    ces_top = top_decile(d["ces_score"], d["geoid"])
    ours_top = top_decile(d["burden"], d["geoid"])
    no_access_top = top_decile(d["burden_no_access"], d["geoid"])
    rho = table.set_index(["ours", "theirs"])["spearman"]
    metrics = {
        "n_tracts": len(d),
        "spearman_ces": float(rho[("burden", "ces_score")]),
        "spearman_ces_no_access": float(rho[("burden_no_access", "ces_score")]),
        "spearman_pm25_ces": float(rho[("pm25_ugm3", "pm25_ces")]),
        "pm25_mae_ugm3": float((d["pm25_ugm3"] - d["pm25_ces"]).abs().mean()),
        "top_decile_size": len(ces_top),
        "top_decile_overlap": len(ces_top & ours_top),
        "top_decile_overlap_no_access": len(ces_top & no_access_top),
        "ces_top_decile_mean_access_z": float(
            d.loc[d["geoid"].isin(ces_top), "access_z"].mean()
        ),
        "our_top_decile_mean_access_z": float(
            d.loc[d["geoid"].isin(ours_top), "access_z"].mean()
        ),
    }
    if has_vulnerability:
        hazards_top = top_decile(d["burden_hazards_only"], d["geoid"])
        metrics["spearman_ces_hazards_only"] = float(
            rho[("burden_hazards_only", "ces_score")]
        )
        metrics["spearman_vulnerability_pop_char"] = float(
            rho[("vulnerability_z", "pop_char_score")]
        )
        metrics["top_decile_overlap_hazards_only"] = len(ces_top & hazards_top)
    return metrics, table


def hotspot_stability(
    hotspots: pd.DataFrame,
    top_n: int = 100,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Week-to-week stability of Gi* hotspots.

    hotspots: columns week_start_date, geoid, gi_star_zscore,
    hotspot_class, raw_burden (several weeks).
    Returns (per-week table, summary metrics).
    """
    h = hotspots.copy()
    h["week_start_date"] = pd.to_datetime(h["week_start_date"])
    weeks = sorted(h["week_start_date"].unique())
    if len(weeks) < 2:
        raise ValueError("Hotspot stability needs at least two weeks")

    is_hot = h["hotspot_class"].str.startswith("HOTSPOT")
    hot = {w: set(h.loc[is_hot & (h["week_start_date"] == w), "geoid"]) for w in weeks}
    z = h.pivot(index="geoid", columns="week_start_date", values="gi_star_zscore")
    top = {w: set(z[w].nlargest(top_n).index) for w in weeks}
    skew = h.groupby("week_start_date")["raw_burden"].skew()

    rows = []
    for prev, cur in pairwise(weeks):
        both = z[[prev, cur]].dropna()
        rows.append(
            {
                "week_start_date": pd.Timestamp(cur).date(),
                "n_hotspots": len(hot[cur]),
                "retained": len(hot[prev] & hot[cur]) / len(hot[prev])
                if hot[prev]
                else np.nan,
                "rank_spearman_prev": float(spearmanr(both[prev], both[cur])[0]),
                "top_overlap": len(top[prev] & top[cur]) / top_n,
                "burden_skew": float(skew[cur]),
            }
        )
    per_week = pd.DataFrame(rows)

    counts = pd.Series([len(hot[w]) for w in weeks])
    weeks_hot = pd.Series(
        [g for w in weeks for g in hot[w]], dtype="object"
    ).value_counts()
    summary = {
        "n_weeks": len(weeks),
        "count_min": int(counts.min()),
        "count_median": float(counts.median()),
        "count_max": int(counts.max()),
        "retained_median": float(per_week["retained"].median()),
        "rank_spearman_prev_median": float(per_week["rank_spearman_prev"].median()),
        "top_overlap_median": float(per_week["top_overlap"].median()),
        "top_overlap_min": float(per_week["top_overlap"].min()),
        "count_skew_spearman": float(spearmanr(counts, [skew[w] for w in weeks])[0]),
        "ever_hotspot": int(len(weeks_hot)),
        "persistent_hotspot": int((weeks_hot >= len(weeks) / 2).sum()),
    }
    return per_week, summary


def kriging_year(
    kriged: pd.DataFrame,
    mae_limit: float = 2.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Kriging quality over all stored weeks.

    kriged: one row per week with week_start_date, cv_mae, cv_r2,
    n_monitors, variogram_nugget, variogram_psill, variogram_range_m.
    Returns (per-week table with flags, summary metrics).
    """
    k = kriged.sort_values("week_start_date").reset_index(drop=True).copy()
    k["no_structure"] = k["variogram_nugget"] >= k["variogram_psill"]
    k["weak"] = (k["cv_mae"] > mae_limit) | k["no_structure"]
    # The upper bound is 3x the longest lag. It shifts a little with the
    # monitor set (226-242 km in LA with 19-22 monitors), while ranges the
    # fit actually estimates stay far below it (at most 126 km), so 90% of
    # the longest fitted range separates the two
    k["range_at_bound"] = k["variogram_range_m"] >= 0.9 * k["variogram_range_m"].max()
    summary = {
        "n_weeks": len(k),
        "mae_median": float(k["cv_mae"].median()),
        "mae_mean": float(k["cv_mae"].mean()),
        "mae_max": float(k["cv_mae"].max()),
        "weak_weeks": int(k["weak"].sum()),
        "no_structure_weeks": int(k["no_structure"].sum()),
        "range_at_bound_weeks": int(k["range_at_bound"].sum()),
    }
    return k, summary
