"""
hotspots.py
===========
Detect statistically significant compound burden hotspot clusters
using Moran's I (global) and Gi* (local) spatial statistics.

Moran's I answers: is compound burden spatially clustered?
Gi* answers: which specific tracts form significant hotspot clusters?

A tract flagged as a hotspot is not just high-scoring individually —
it is part of a geographically concentrated cluster of high-scoring
tracts. This spatial concentration is more alarming than an
isolated high-scoring tract, because it indicates neighborhood-level
compound risk that affects a contiguous population.

Method:
  - Global: Moran's I with 999 permutations (pseudo p-value), queen
    contiguity weights without self-weights
  - Local: Getis-Ord Gi* with queen contiguity weights plus a self-weight
    (each tract counts in its own neighborhood), row-standardized
  - Significance: two-tailed p<0.05 (|z|>1.96) and p<0.01 (|z|>2.576).
    esda's p_norm is one-tailed; until Nov 2026 the classes used it
    directly, so HOTSPOT_95 meant z>1.645. The stored gi_star_pvalue is
    the two-tailed value.
  - Multiple testing: each week runs one test per tract (2,498 in LA).
    The Benjamini-Hochberg (FDR) count is logged for reference; classes
    use the per-tract cutoffs above.

Note on weekly counts: Gi* compares each neighborhood with that week's
city mean, so the number of hotspots depends on the shape of the week's
burden distribution (a long low tail hides high clusters). Over 53 weeks
the hotspot count tracked burden skewness (Spearman 0.86) while Moran's I
stayed between 0.71 and 0.95. Use Gi* z ranks, not counts, to compare
weeks.

Catalina Island (2 tracts) is its own contiguity component: its tracts
are compared only with each other.

Input:
    BigQuery: compound_scores (weekly compound burden scores)
    Local: tract_geometries GeoParquet (for spatial weights)

Output:
    BigQuery table: hotspot_detection (one row per tract per week)
    Columns: geoid, gi_star_zscore, gi_star_pvalue,
             hotspot_class, morans_i, morans_pvalue,
             week_start_date, city_fips

Usage:
    from src.spatial.hotspots import detect_hotspots
    df = detect_hotspots(config)
"""

import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import geopandas as gpd
import libpysal
import numpy as np
import pandas as pd
from esda.getisord import G_Local
from esda.moran import Moran

from src.utils.bigquery import REPLACE_PARTITION, read_query, write_dataframe
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def detect_hotspots(
    config: dict[str, Any],
    week_start: datetime | None = None,
) -> pd.DataFrame:
    """
    Detect compound burden hotspot clusters for the configured city.

    Parameters
    ----------
    config : dict
        Merged pipeline config.
    week_start : datetime, optional
        Week to detect hotspots for. Defaults to most recent Monday.

    Returns
    -------
    pd.DataFrame
        One row per tract with Gi* scores and hotspot classification.

    Examples
    --------
    >>> config = load_config("los_angeles")
    >>> df = detect_hotspots(config)
    >>> df["hotspot_class"].value_counts()
    NOT_SIGNIFICANT    1987
    HOTSPOT_95          234
    HOTSPOT_99           89
    COLDSPOT_95           0
    """
    if week_start is None:
        today = datetime.now()
        week_start = today - timedelta(days=today.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)

    full_fips = config["full_fips"]
    week_str = week_start.strftime("%Y-%m-%d")

    logger.info(f"Detecting hotspots for {config['city_name']} week of {week_str}")

    # Step 1 — Load compound scores from BigQuery
    scores_df = _load_compound_scores(config, week_str)
    logger.info(f"Loaded compound scores for {len(scores_df):,} tracts")

    # Step 2 — Load tract geometries for spatial weights
    tracts_gdf = _load_tract_geometries(config, scores_df)

    # Step 3 — Build queen contiguity spatial weights
    # Same weights matrix as compound_score.py
    # Row-standardized for Moran's I
    W = _build_spatial_weights(tracts_gdf)
    logger.info(f"Spatial weights: mean neighbors={W.mean_neighbors:.1f}")

    # Step 4 — Global Moran's I
    # One number summarizing spatial clustering for the whole city
    morans_i, morans_pvalue = _compute_morans_i(
        scores_df["compound_score"].values,
        W,
        config,
    )
    logger.info(
        f"Global Moran's I = {morans_i:.4f} "
        f"(p={morans_pvalue:.4f}) — "
        f"{'significant' if morans_pvalue < 0.05 else 'not significant'} "
        f"spatial clustering"
    )

    # Step 5 — Local Gi* hotspot detection
    # One z-score per tract indicating local clustering
    gi_df = _compute_gi_star(
        scores_df["compound_score"].values,
        W,
        scores_df["geoid"].values,
        config,
    )
    logger.info(
        f"Gi* complete: "
        f"{(gi_df['hotspot_class'] == 'HOTSPOT_99').sum()} "
        f"tracts in 99% confidence hotspot clusters"
    )

    # Step 6 — Merge Gi* results with compound scores
    result_df = scores_df[["geoid", "compound_score"]].merge(
        gi_df, on="geoid", how="left"
    )

    # Step 7 — Add global Moran's I values to every row
    # Stored per-row for BigQuery queryability
    result_df["morans_i"] = round(morans_i, 6)
    result_df["morans_pvalue"] = round(morans_pvalue, 6)

    # Step 8 — Add metadata
    result_df["week_start_date"] = week_start.date()
    result_df["city_fips"] = full_fips

    # Step 9 — Validate and write
    _validate_output(result_df, config)
    write_dataframe(
        result_df,
        "hotspot_detection",
        config,
        write_disposition=REPLACE_PARTITION,
    )

    hotspot_99 = (result_df["hotspot_class"] == "HOTSPOT_99").sum()
    hotspot_95 = (result_df["hotspot_class"] == "HOTSPOT_95").sum()

    logger.info(
        f"Hotspot detection complete: "
        f"{hotspot_99} tracts in 99% clusters, "
        f"{hotspot_95} tracts in 95% clusters, "
        f"Moran's I={morans_i:.4f}"
    )

    return result_df


# ---------------------------------------------------------------------------
# Private functions
# ---------------------------------------------------------------------------


def _load_compound_scores(
    config: dict[str, Any],
    week_str: str,
) -> pd.DataFrame:
    """Load compound burden scores from BigQuery for the current week."""
    df = read_query(
        f"""
        SELECT geoid, compound_score, burden_class
        FROM {{dataset}}.compound_scores
        WHERE
            week_start_date = '{week_str}'
            AND city_fips = '{config["full_fips"]}'
        ORDER BY geoid
        """,
        config,
    )

    if df.empty:
        raise ValueError(
            f"No compound scores for week {week_str}. Run compound_score.py first."
        )

    return df


def _load_tract_geometries(
    config: dict[str, Any],
    scores_df: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """
    Load tract geometries aligned to scores DataFrame order.

    Critical: spatial weights matrix rows must exactly match
    the order of the scores array passed to Moran's I and Gi*.
    We sort both by geoid to guarantee alignment.
    """
    full_fips = config["full_fips"]
    local_path = (
        Path("data") / "processed" / "tract_geometries" / f"{full_fips}.parquet"
    )

    if local_path.exists():
        gdf = gpd.read_parquet(local_path)
    else:
        from src.utils.gcs import download_file

        gcs_path = f"tract_geometries/{full_fips}.parquet"
        downloaded = download_file(gcs_path, config, local_path=local_path)
        gdf = gpd.read_parquet(downloaded)

    # Align to scores_df order — both sorted by geoid
    gdf = gdf.set_index("geoid").loc[scores_df["geoid"]].reset_index()

    return gdf


def _build_spatial_weights(
    tracts_gdf: gpd.GeoDataFrame,
) -> libpysal.weights.W:
    """
    Build row-standardized queen contiguity spatial weights matrix.

    compound_score no longer uses spatial weights (PCA was removed), so
    this module owns the weights. Row-standardization is required for
    both Moran's I and Gi*.
    """
    W = libpysal.weights.Queen.from_dataframe(
        tracts_gdf,
        use_index=False,  # rows are aligned to the scores array by position
        silence_warnings=True,
    )
    if W.n_components > 1:
        sizes = np.bincount(W.component_labels)
        logger.info(
            f"Contiguity has {W.n_components} components (sizes "
            f"{sorted(sizes.tolist(), reverse=True)}); small ones are islands"
        )
    W.transform = "r"
    return W


def _gi_star_weights(W: libpysal.weights.W) -> libpysal.weights.W:
    """
    Binary contiguity plus a self-weight of 1, row-standardized.

    Gi* includes each tract in its own neighborhood. Putting the self-weight
    in the matrix makes that explicit; otherwise esda guesses it and warns.
    """
    binary = libpysal.weights.W(W.neighbors, id_order=W.id_order, silence_warnings=True)
    with warnings.catch_warnings():
        # fill_diagonal rebuilds W and repeats the island warning logged above
        warnings.filterwarnings(
            "ignore", message="The weights matrix is not fully connected"
        )
        with_self = libpysal.weights.fill_diagonal(binary, 1.0)
    with_self.transform = "r"
    return with_self


def fdr_count(pvalues: np.ndarray, alpha: float) -> int:
    """
    Number of tests significant under Benjamini-Hochberg at level alpha.

    Sort the p-values; the k smallest are significant for the largest k
    with p_(k) <= k / m * alpha.
    """
    p = np.sort(np.asarray(pvalues, dtype=float))
    m = len(p)
    if m == 0:
        return 0
    passed = np.nonzero(p <= np.arange(1, m + 1) / m * alpha)[0]
    return int(passed[-1] + 1) if len(passed) else 0


def _compute_morans_i(
    scores: np.ndarray,
    W: libpysal.weights.W,
    config: dict[str, Any],
) -> tuple[float, float]:
    """
    Compute Global Moran's I with permutation-based significance testing.

    Moran's I formula:
        I = (N / S0) × (Σ_i Σ_j w_ij (x_i - x̄)(x_j - x̄)) / Σ_i (x_i - x̄)²

    Where:
        N = number of tracts
        S0 = sum of all spatial weights
        w_ij = spatial weight between tracts i and j
        x_i, x_j = compound scores
        x̄ = mean compound score

    Interpretation:
        I > 0: positive spatial autocorrelation (clustering)
        I = 0: random spatial pattern
        I < 0: negative autocorrelation (dispersion — rare)

    P-value from permutation testing:
        Randomly shuffle scores across tracts 999 times.
        Count how many permutations produce I >= observed I.
        p-value = count / 999
        This is a pseudo p-value — no distributional assumptions needed.

    Parameters
    ----------
    scores : np.ndarray
        Compound burden scores, aligned to W.
    W : libpysal.weights.W
        Spatial weights matrix.
    config : dict
        Pipeline config with random seed.

    Returns
    -------
    tuple of (morans_i, p_value)
    """
    n_permutations = 999
    random_seed = config["ml"]["random_seed"]

    logger.debug(
        f"Computing Moran's I with {n_permutations} permutations (seed={random_seed})"
    )

    # esda seeds permutations from NumPy's global generator, not a seed argument
    np.random.seed(random_seed)
    moran = Moran(
        scores,
        W,
        permutations=n_permutations,
    )

    return float(moran.I), float(moran.p_sim)


def _compute_gi_star(
    scores: np.ndarray,
    W: libpysal.weights.W,
    geoids: np.ndarray,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Compute Local Getis-Ord Gi* statistic for each tract.

    Gi* formula for tract i:
        Gi* = (Σ_j w_ij × x_j - x̄ × Σ_j w_ij) /
              (s × sqrt((n × Σ_j w²_ij - (Σ_j w_ij)²) / (n-1)))

    Where:
        w_ij = spatial weight between i and j
        x_j = compound score of tract j
        x̄ = global mean compound score
        s = global standard deviation
        n = number of tracts

    The output z-score tells you:
        Gi* > 1.96 (p<0.05): this tract is in a HIGH cluster (hotspot)
        Gi* > 2.58 (p<0.01): this tract is in a HIGH cluster (99% conf)
        Gi* < -1.96: this tract is in a LOW cluster (coldspot)
        |Gi*| < 1.96: not statistically significant

    Note on difference from LISA (Anselin Local Moran's I):
        Gi* identifies clusters of HIGH values (hotspots) and LOW
        values (coldspots) — it does not identify spatial outliers
        (high tract surrounded by low tracts). For public health
        intervention targeting, hotspot clusters are what we care about.

    Parameters
    ----------
    scores : np.ndarray
        Compound burden scores, aligned to W.
    W : libpysal.weights.W
        Spatial weights matrix.
    geoids : np.ndarray
        Tract GEOIDs, aligned to W.
    config : dict
        Pipeline config. Reads hotspot_significance (standard, high).

    Returns
    -------
    pd.DataFrame
        One row per tract with columns:
        geoid, gi_star_zscore, gi_star_pvalue, hotspot_class
    """
    logger.debug(f"Computing Gi* for {len(scores)} tracts")

    # G_Local computes Gi* (star variant: includes self in calculation)
    # Self-weight comes from _gi_star_weights; transform='R' keeps rows summing to 1
    # permutations=0: use analytical p-values (faster than permutation)
    gi = G_Local(
        scores,
        _gi_star_weights(W),
        star=None,  # the self-weight is already in the matrix
        transform="R",
        permutations=0,  # analytical p-values sufficient for Gi*
    )
    # esda's p_norm is one-tailed (normal tail beyond |z|). Hotspots and
    # coldspots are both tested, so the p-value is two-tailed.
    p_two = np.minimum(2 * gi.p_norm, 1.0)

    # Classify hotspot significance level
    # Using analytical p-values from normal distribution
    standard_sig = config["hotspot_significance"]["standard"]
    high_sig = config["hotspot_significance"]["high"]

    hotspot_classes = []
    for z, p in zip(gi.Zs, p_two, strict=True):
        if z > 0 and p < high_sig:
            hotspot_classes.append("HOTSPOT_99")
        elif z > 0 and p < standard_sig:
            hotspot_classes.append("HOTSPOT_95")
        elif z < 0 and p < high_sig:
            hotspot_classes.append("COLDSPOT_99")
        elif z < 0 and p < standard_sig:
            hotspot_classes.append("COLDSPOT_95")
        else:
            hotspot_classes.append("NOT_SIGNIFICANT")

    df = pd.DataFrame(
        {
            "geoid": geoids,
            "gi_star_zscore": gi.Zs.round(4),
            "gi_star_pvalue": p_two.round(6),
            "hotspot_class": hotspot_classes,
        }
    )

    # Log hotspot summary
    for cls in [
        "HOTSPOT_99",
        "HOTSPOT_95",
        "NOT_SIGNIFICANT",
        "COLDSPOT_95",
        "COLDSPOT_99",
    ]:
        count = (df["hotspot_class"] == cls).sum()
        if count > 0:
            logger.info(f"  {cls}: {count:,} tracts")
    logger.info(
        f"  significant after Benjamini-Hochberg at {standard_sig}: "
        f"{fdr_count(p_two, standard_sig):,} of {len(p_two):,} tracts"
    )

    return df


def _validate_output(df: pd.DataFrame, config: dict[str, Any]) -> None:
    """Validate hotspot detection output before BigQuery write."""
    if df.empty:
        raise ValueError("Hotspot output is empty")

    # Check valid hotspot classes
    valid_classes = {
        "HOTSPOT_99",
        "HOTSPOT_95",
        "NOT_SIGNIFICANT",
        "COLDSPOT_95",
        "COLDSPOT_99",
    }
    invalid = set(df["hotspot_class"].unique()) - valid_classes
    if invalid:
        raise ValueError(f"Invalid hotspot classes: {invalid}")

    # Check Moran's I is in valid range [-1, 1]
    morans_i = df["morans_i"].iloc[0]
    if not -1 <= morans_i <= 1:
        raise ValueError(f"Moran's I={morans_i} is outside [-1, 1]")

    # Warn if no significant hotspots found
    hotspot_count = df["hotspot_class"].str.startswith("HOTSPOT").sum()
    if hotspot_count == 0:
        logger.warning(
            "No statistically significant hotspot clusters found. "
            "This may indicate low spatial autocorrelation this week."
        )

    logger.info(
        f"Hotspot validation passed: {len(df):,} tracts, "
        f"Moran's I={morans_i:.4f}, "
        f"{hotspot_count} hotspot tracts"
    )
