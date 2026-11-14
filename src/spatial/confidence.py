"""
confidence.py
=============
How sure is each tract's burden class? Two sources of doubt, simulated
together:

1. PM2.5 interpolation error. Kriging gives each tract a variance, but
   tract errors are not independent: every estimate is a weighted average
   of the same ~21 monitors, so neighbors are wrong together and their
   ranks hold far better than their values. Treating errors as
   independent says 98% of tracts could change class (2026-06-22); that
   is wrong. Here the full error covariance between tracts is rebuilt
   from the week's fitted variogram:

       Sigma = C_tt - C_tm L' - L C_mt + L C_mm L'

   C = covariance from the spherical variogram (sill - gamma(h)), L = the
   ordinary Kriging weights (tracts x monitors). With the nugget on the
   diagonal of C_tt, diag(Sigma) equals PyKrige's stored variance (checked
   in tests). The simulation leaves the nugget out of C_tt: the nugget is
   set at the weekly sampling noise of monitor readings, which is noise in
   the readings, not in the air at a tract. Monitors keep it (C_mm).

   The index uses a 4-week PM2.5 mean. One standard-normal draw is shared
   by the 4 weeks' error factors: monitors barely move week to week, so
   the error pattern repeats (the cautious choice; independent weeks would
   give narrower intervals).

2. Index weights. Each component's weight is drawn from [0.9, 1.1]
   (+/-10%), since equal weights are a choice, not a measurement.

Each of n_draws draws adds one PM2.5 error map and one weight vector,
recomputes the index the way compound_score does (re-standardize PM2.5,
weighted sum, percentile rank, class cutoffs) and records each tract's
class.

The badge is about the decision the product acts on, CRITICAL or not
(the CRITICAL list drives briefings and alerts). p_critical = share of
draws in which the tract is CRITICAL; agreement = p_critical for a
CRITICAL tract, 1 - p_critical otherwise:
    HIGH   agreement >= confidence_high (0.9)
    LOW    agreement <  confidence_low  (0.5)
    MEDIUM otherwise
Exact-class retention is kept as a diagnostic only: ELEVATED (50-75) and
HIGH (75-90) are narrow bands, so on 2026-06-22 not one ELEVATED or HIGH
tract kept its class in 90% of draws; a badge on it would measure band
width, not data quality. The percentile interval covers the rest.
PM2.5-only and weights-only retention (16 corners of the weight box) say
which source drives the doubt.

Output:
    tract_confidence, one row per tract per week (REPLACE_PARTITION)
        geoid, p_critical, class_retention, retention_pm25,
        retention_weights, percentile_p05, percentile_p95, pm25_error_sd (ug/m3, noise free,
        window mean), confidence, n_draws, week_start_date, city_fips

Usage:
    from src.spatial.confidence import compute_confidence
    df = compute_confidence(config, "2026-06-22")
"""

from __future__ import annotations

import itertools
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from src.spatial import kriging as kr
from src.spatial.compound_score import _window_start
from src.utils.bigquery import REPLACE_PARTITION, read_query, write_dataframe
from src.utils.logger import get_logger

logger = get_logger(__name__)

TABLE = "tract_confidence"
COMPONENTS = ["pm25_z", "lst_z", "access_z", "vulnerability_z"]
Z_TOLERANCE = 1e-3  # recomputed PM2.5 z must match the stored one


def compute_confidence(
    config: dict[str, Any],
    week: str,
    factor_cache: dict[str, np.ndarray] | None = None,
    write: bool = True,
) -> pd.DataFrame:
    """
    Simulate class retention for every tract in one week.

    factor_cache: week -> error factor (tracts x tracts). Pass the same dict
    across consecutive weeks in a backfill; each week's factor is reused by
    the 4 windows it belongs to.
    """
    prod = config["product"]
    cache = {} if factor_cache is None else factor_cache
    fips = config["full_fips"]

    tracts = kr._load_tract_centroids(config)
    scores = _load_scores(config, week).set_index("geoid").loc[tracts["geoid"]]
    window = int(config.get("pm25_window_weeks", 4))
    kriged = read_query(
        f"""
        SELECT geoid, week_start_date, pm25_kriged, variogram_nugget,
               variogram_psill, variogram_range_m
        FROM {{dataset}}.pm25_kriged
        WHERE city_fips = '{fips}'
          AND week_start_date BETWEEN '{_window_start(week, window)}' AND '{week}'
        """,
        config,
    )
    weeks = sorted(
        pd.to_datetime(kriged["week_start_date"]).dt.strftime("%Y-%m-%d").unique()
    )

    factors = []
    for wk in weeks:
        if wk not in cache:
            params = kriged[kriged["week_start_date"].astype(str).str[:10] == wk].iloc[
                0
            ]
            monitors = kr._load_monitor_readings(config, wk)
            m_utm, t_utm = kr._project_to_utm(monitors, tracts.copy(), config)
            sigma = error_covariance(
                m_utm[["x_utm", "y_utm"]].to_numpy(),
                t_utm[["x_utm", "y_utm"]].to_numpy(),
                params["variogram_nugget"],
                params["variogram_psill"],
                params["variogram_range_m"],
            )
            cache[wk] = covariance_factor(sigma)
        factors.append(cache[wk])
    for old in [k for k in cache if k < weeks[0]]:
        del cache[old]  # outside every later window

    pm = kriged.groupby("geoid")["pm25_kriged"].mean().loc[tracts["geoid"]].to_numpy()
    _check_pm25_matches(pm, scores["pm25_z"].to_numpy(), week)

    rng = np.random.default_rng(prod["confidence_seed"])
    out = simulate_retention(
        pm=pm,
        components=scores[COMPONENTS].to_numpy(),
        base_class=scores["burden_class"].to_numpy(),
        factor=sum(factors) / len(factors),
        cutoffs=config["burden_percentiles"],
        n_draws=prod["confidence_draws"],
        jitter=prod["confidence_weight_jitter"],
        rng=rng,
    )
    out.insert(0, "geoid", tracts["geoid"].to_numpy())
    is_critical = scores["burden_class"].to_numpy() == "CRITICAL"
    agreement = np.where(is_critical, out["p_critical"], 1 - out["p_critical"])
    out["confidence"] = label(pd.Series(agreement), prod)
    out["n_draws"] = prod["confidence_draws"]
    out["week_start_date"] = date.fromisoformat(week)
    out["city_fips"] = fips

    if write:
        write_dataframe(out, TABLE, config, write_disposition=REPLACE_PARTITION)
    counts = out["confidence"].value_counts().to_dict()
    logger.info(f"Confidence {week}: {counts}; {len(weeks)} Kriging weeks in window")
    return out


def error_covariance(
    monitors_xy: np.ndarray,
    tracts_xy: np.ndarray,
    nugget: float,
    psill: float,
    range_m: float,
    include_nugget: bool = False,
) -> np.ndarray:
    """Ordinary Kriging error covariance between all tract points."""

    def cov(h: np.ndarray) -> np.ndarray:
        r = np.minimum(h / range_m, 1.0)
        return psill * (
            1.0 - (1.5 * r - 0.5 * r**3)
        )  # h > 0: the nugget is a jump at 0

    n, t = len(monitors_xy), len(tracts_xy)
    c_mm = cov(cdist(monitors_xy, monitors_xy))
    c_mm[np.diag_indices(n)] = psill + nugget
    c_tm = cov(cdist(tracts_xy, monitors_xy))

    system = np.ones((n + 1, n + 1))
    system[:n, :n] = c_mm
    system[n, n] = 0.0
    rhs = np.vstack([c_tm.T, np.ones(t)])
    weights = np.linalg.solve(system, rhs)[:n].T  # tracts x monitors

    c_tt = cov(cdist(tracts_xy, tracts_xy))
    c_tt[np.diag_indices(t)] = psill + (nugget if include_nugget else 0.0)
    return c_tt - c_tm @ weights.T - weights @ c_tm.T + weights @ c_mm @ weights.T


def covariance_factor(sigma: np.ndarray) -> np.ndarray:
    """
    Symmetric square root F = V sqrt(D) V' (F F' = sigma; tiny negative
    eigenvalues set to 0). Unlike an eigenvector factor, whose column signs
    are arbitrary, it is unique and changes smoothly with sigma, so one
    shared draw gives the same error pattern in every week of the window.
    """
    vals, vecs = np.linalg.eigh((sigma + sigma.T) / 2)
    return (vecs * np.sqrt(np.clip(vals, 0.0, None))) @ vecs.T


def classes(raw: np.ndarray, cutoffs: dict[str, float]) -> np.ndarray:
    """Class codes 0-3 (LOW..CRITICAL), percentile rank per column, as compound_score."""
    pct = pd.DataFrame(raw).rank(pct=True).mul(100).round(1).to_numpy()
    cuts = np.array([cutoffs["elevated"], cutoffs["high_risk"], cutoffs["critical"]])
    return (pct[..., None] >= cuts).sum(-1), pct


CLASS_CODES = {"LOW": 0, "ELEVATED": 1, "HIGH": 2, "CRITICAL": 3}


def simulate_retention(
    pm: np.ndarray,
    components: np.ndarray,
    base_class: np.ndarray,
    factor: np.ndarray,
    cutoffs: dict[str, float],
    n_draws: int,
    jitter: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    components: tracts x 4 signed contributions in COMPONENTS order (PM2.5
    first). base_class: stored class labels.
    """
    base = np.array([CLASS_CODES[c] for c in base_class])[:, None]
    others = components[:, 1:]

    def pm_z(field: np.ndarray) -> np.ndarray:
        return (field - field.mean(0)) / field.std(0)

    sim = pm[:, None] + factor @ rng.standard_normal((len(pm), n_draws))
    w = rng.uniform(1 - jitter, 1 + jitter, size=(n_draws, 4))
    joint_cls, joint_pct = classes(pm_z(sim) * w[:, 0] + others @ w[:, 1:].T, cutoffs)
    pm_cls, _ = classes(pm_z(sim) + others.sum(1, keepdims=True), cutoffs)

    corners = np.array(list(itertools.product([1 - jitter, 1 + jitter], repeat=4)))
    w_cls, _ = classes(components @ corners.T, cutoffs)

    return pd.DataFrame(
        {
            "p_critical": (joint_cls == CLASS_CODES["CRITICAL"]).mean(1).round(3),
            "class_retention": (joint_cls == base).mean(1).round(3),
            "retention_pm25": (pm_cls == base).mean(1).round(3),
            "retention_weights": (w_cls == base).mean(1).round(3),
            "percentile_p05": np.percentile(joint_pct, 5, axis=1).round(1),
            "percentile_p95": np.percentile(joint_pct, 95, axis=1).round(1),
            "pm25_error_sd": np.linalg.norm(factor, axis=1).round(3),
        }
    )


def label(retention: pd.Series, prod: dict[str, Any]) -> np.ndarray:
    return np.select(
        [retention >= prod["confidence_high"], retention < prod["confidence_low"]],
        ["HIGH", "LOW"],
        default="MEDIUM",
    )


def _load_scores(config: dict[str, Any], week: str) -> pd.DataFrame:
    df = read_query(
        f"""
        SELECT geoid, burden_class, {", ".join(COMPONENTS)}
        FROM {{dataset}}.compound_scores
        WHERE city_fips = '{config["full_fips"]}' AND week_start_date = '{week}'
        """,
        config,
    )
    if df.empty:
        raise ValueError(f"No compound_scores for {week}")
    return df


def _check_pm25_matches(pm: np.ndarray, stored_z: np.ndarray, week: str) -> None:
    """The simulation must start from the exact PM2.5 the index used."""
    z = (pm - pm.mean()) / pm.std()
    gap = np.nanmax(np.abs(z - stored_z))
    if not np.isfinite(gap) or gap > Z_TOLERANCE:
        raise ValueError(
            f"{week}: recomputed PM2.5 z differs from compound_scores by {gap:.4f}"
        )
