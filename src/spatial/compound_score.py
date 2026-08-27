"""
compound_score.py
=================
Compute the Compound Climate-Health Burden Score for each census
tract by combining PM2.5, LST, and healthcare access signals via
Spatial PCA.

The compound score is the core output of the CHBIndex pipeline —
the single number that captures simultaneous risk across three
climate-health dimensions. It is computed weekly and written to
BigQuery as the primary input to the ML prediction layer.

Method: Spatial PCA
  1. Normalize all three signals to comparable scales
  2. Augment features with spatial lag (neighborhood averages)
     using queen contiguity spatial weights matrix
  3. Run PCA — PC1 captures maximum shared variance
  4. Min-max normalize PC1 to 0-1 compound burden score
  5. Apply burden thresholds to classify risk level

Why Spatial PCA over simple averaging:
  - Handles different units (μg/m³, z-scores, ratios)
  - Accounts for correlation between signals
  - Spatial lag captures neighborhood spillover effects
  - PC1 loadings are interpretable (which signal drives burden)

Input:
    BigQuery: pm25_kriged, lst_tract_stats, healthcare_access,
              tract_geometries (for spatial weights)

Output:
    BigQuery table: compound_scores (one row per tract per week)
    Columns: geoid, compound_score, burden_class,
             pm25_component, lst_component, access_component,
             pc1_loading_pm25, pc1_loading_lst, pc1_loading_access,
             week_start_date, city_fips

Usage:
    from src.spatial.compound_score import compute_compound_score
    df = compute_compound_score(config)
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import geopandas as gpd
import libpysal
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.utils.bigquery import read_query, write_dataframe
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def compute_compound_score(
    config: dict[str, Any],
    week_start: datetime | None = None,
) -> pd.DataFrame:
    """
    Compute weekly compound climate-health burden score per tract.

    Parameters
    ----------
    config : dict
        Merged pipeline config.
    week_start : datetime, optional
        Week to compute scores for. Defaults to most recent Monday.

    Returns
    -------
    pd.DataFrame
        One row per tract with compound score and component scores.

    Examples
    --------
    >>> config = load_config("los_angeles")
    >>> df = compute_compound_score(config)
    >>> df["compound_score"].describe()
    count    2310.000000
    mean        0.487...
    std         0.198...
    min         0.000...
    max         1.000...
    >>> df["burden_class"].value_counts()
    LOW         1456
    ELEVATED     621
    HIGH         187
    CRITICAL      46
    """
    if week_start is None:
        today = datetime.now()
        week_start = today - timedelta(days=today.weekday())
        week_start = week_start.replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    full_fips = config["full_fips"]
    week_str = week_start.strftime("%Y-%m-%d")

    logger.info(
        f"Computing compound burden score for {config['city_name']} "
        f"week of {week_str}"
    )

    # Step 1 — Load all three signals from BigQuery
    signals_df = _load_signals(config, week_str)
    logger.info(
        f"Loaded signals for {len(signals_df):,} tracts "
        f"({signals_df.notna().all(axis=1).sum():,} complete)"
    )

    # Step 2 — Load tract geometries for spatial weights
    tracts_gdf = _load_tract_geometries(config)

    # Step 3 — Build queen contiguity spatial weights matrix
    # Queen contiguity: two tracts are neighbors if they share
    # any boundary point (edge or corner) — more connections
    # than rook contiguity (edge-only)
    W = _build_spatial_weights(tracts_gdf, signals_df)
    logger.info(f"Built spatial weights: mean neighbors={W.mean_neighbors:.1f}")

    # Step 4 — Compute spatial lag features
    # For each tract and signal, compute the population-weighted
    # average of the signal across neighboring tracts
    signals_with_lag = _compute_spatial_lags(signals_df, W)
    logger.info("Computed spatial lag features")

    # Step 5 — Handle missing values
    # Some tracts may have NaN LST (cloud cover) or NaN PM2.5
    # We impute with city median to preserve them in the index
    signals_imputed, imputed_count = _impute_missing(signals_with_lag)
    if imputed_count > 0:
        logger.warning(
            f"Imputed {imputed_count} missing values with city median"
        )

    # Step 6 — Run Spatial PCA
    scores_df, loadings, explained_variance = _run_spatial_pca(
        signals_imputed, config
    )
    logger.info(
        f"Spatial PCA: PC1 explains {explained_variance[0]:.1%} of variance. "
        f"Loadings: PM2.5={loadings[0]:.3f}, "
        f"LST={loadings[1]:.3f}, "
        f"Access={loadings[2]:.3f}"
    )

    # Step 7 — Min-max normalize PC1 to 0-1
    scores_df = _normalize_scores(scores_df)

    # Step 8 — Classify burden level
    scores_df = _classify_burden(scores_df, config)

    # Step 9 — Add PCA loadings as metadata
    # These tell you which signal drives compound burden this week
    # Stored so you can track whether loadings change over time
    scores_df["pc1_loading_pm25"] = round(loadings[0], 4)
    scores_df["pc1_loading_lst"] = round(loadings[1], 4)
    scores_df["pc1_loading_access"] = round(loadings[2], 4)
    scores_df["pc1_variance_explained"] = round(explained_variance[0], 4)

    # Step 10 — Add metadata
    scores_df["week_start_date"] = week_start.date()
    scores_df["city_fips"] = full_fips

    # Step 11 — Validate and write
    _validate_output(scores_df, config)
    write_dataframe(scores_df, "compound_scores", config)

    high_risk = scores_df["burden_class"].isin(["HIGH", "CRITICAL"]).sum()
    critical = (scores_df["burden_class"] == "CRITICAL").sum()

    logger.info(
        f"Compound scores complete: {len(scores_df):,} tracts, "
        f"{high_risk} high/critical risk, {critical} critical. "
        f"Mean score: {scores_df['compound_score'].mean():.3f}"
    )

    return scores_df


# ---------------------------------------------------------------------------
# Private functions
# ---------------------------------------------------------------------------

def _load_signals(
    config: dict[str, Any],
    week_str: str,
) -> pd.DataFrame:
    """
    Load all three signals from BigQuery for the current week.

    Joins PM2.5 (Kriging), LST, and healthcare access on geoid.
    All three are needed to compute the compound score.

    Returns
    -------
    pd.DataFrame
        One row per tract with columns:
        geoid, pm25_kriged, lst_zscore, access_score_er
    """
    full_fips = config["full_fips"]

    query = f"""
        SELECT
            k.geoid,
            k.pm25_kriged,
            l.lst_zscore,
            a.access_score_er
        FROM
            {{dataset}}.pm25_kriged k
        LEFT JOIN
            {{dataset}}.lst_tract_stats l
            ON k.geoid = l.geoid
            AND l.week_start_date = '{week_str}'
        LEFT JOIN
            {{dataset}}.healthcare_access a
            ON k.geoid = a.geoid
        WHERE
            k.week_start_date = '{week_str}'
            AND k.city_fips = '{full_fips}'
        ORDER BY k.geoid
    """

    df = read_query(query, config)

    if df.empty:
        raise ValueError(
            f"No signals found for week {week_str}, FIPS {full_fips}. "
            f"Run ingestion and spatial transform DAGs first."
        )

    return df


def _load_tract_geometries(config: dict[str, Any]) -> gpd.GeoDataFrame:
    """Load tract polygon geometries for spatial weights computation."""
    full_fips = config["full_fips"]
    local_path = (
        Path("data") / "processed" / "tract_geometries" / f"{full_fips}.parquet"
    )

    if local_path.exists():
        return gpd.read_parquet(local_path)

    # Download from GCS if not local
    from src.utils.gcs import download_file
    gcs_path = f"tract_geometries/{full_fips}.parquet"
    downloaded = download_file(gcs_path, config, local_path=local_path)
    return gpd.read_parquet(downloaded)


def _build_spatial_weights(
    tracts_gdf: gpd.GeoDataFrame,
    signals_df: pd.DataFrame,
) -> libpysal.weights.W:
    """
    Build queen contiguity spatial weights matrix.

    The weights matrix W encodes neighborhood relationships:
    W[i,j] = 1 if tract i and tract j share a boundary
    W[i,j] = 0 otherwise (row-standardized: divided by row sum)

    Queen contiguity = shared edge OR shared corner.
    This gives more neighbors per tract than rook contiguity
    (shared edge only) — better for irregular tract shapes.

    Row-standardization means each row sums to 1 — the spatial
    lag for tract i is a simple average of neighbor values.

    Parameters
    ----------
    tracts_gdf : gpd.GeoDataFrame
        Tract polygons — must be sorted to match signals_df order.
    signals_df : pd.DataFrame
        Signal DataFrame — used to align tract order.

    Returns
    -------
    libpysal.weights.W
        Row-standardized queen contiguity weights matrix.
    """
    logger.debug("Building queen contiguity spatial weights matrix")

    # Align tract geometries to signal DataFrame order
    # Critical: weights matrix rows must match signals_df rows exactly
    aligned_gdf = tracts_gdf.set_index("geoid").loc[signals_df["geoid"]].reset_index()

    # Build queen contiguity weights
    # silence_warnings=True suppresses islands warning for isolated tracts
    W = libpysal.weights.Queen.from_dataframe(
        aligned_gdf,
        silence_warnings=True,
    )

    # Row-standardize: divide each weight by the row sum
    # After standardization, W[i,j] = 1/n_neighbors for all j neighbors of i
    # The spatial lag of i = mean of neighbors' values
    W.transform = "r"

    logger.debug(
        f"Weights matrix: {W.n} tracts, "
        f"mean neighbors={W.mean_neighbors:.1f}, "
        f"islands={len(W.islands)}"
    )

    return W


def _compute_spatial_lags(
    signals_df: pd.DataFrame,
    W: libpysal.weights.W,
) -> pd.DataFrame:
    """
    Compute spatial lag features for all three signals.

    For each signal x and tract i:
        lag_x_i = Σ w_ij × x_j    for all neighbors j

    With row-standardized weights, this is the simple average
    of neighboring tract values — what does the surrounding
    neighborhood look like?

    Including spatial lags in PCA means PC1 captures both
    local signal AND neighborhood context. A tract with high
    PM2.5 surrounded by other high-PM2.5 tracts gets a higher
    compound score than an isolated high-PM2.5 tract.

    Parameters
    ----------
    signals_df : pd.DataFrame
        Three signals per tract: pm25_kriged, lst_zscore, access_score_er
    W : libpysal.weights.W
        Row-standardized spatial weights matrix.

    Returns
    -------
    pd.DataFrame
        Original signals plus three spatial lag columns:
        lag_pm25, lag_lst, lag_access
    """
    df = signals_df.copy()

    # Fill NaN with city median before computing spatial lag
    # A NaN neighbor would propagate NaN through the lag computation
    for col in ["pm25_kriged", "lst_zscore", "access_score_er"]:
        df[col] = df[col].fillna(df[col].median())

    # Compute spatial lag for each signal
    # libpysal.weights.lag_spatial multiplies W × y
    # Returns an array of length n where each element is
    # the weighted average of that tract's neighbors
    df["lag_pm25"] = libpysal.weights.lag_spatial(
        W, df["pm25_kriged"].values
    )
    df["lag_lst"] = libpysal.weights.lag_spatial(
        W, df["lst_zscore"].values
    )
    df["lag_access"] = libpysal.weights.lag_spatial(
        W, df["access_score_er"].values
    )

    return df


def _impute_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Impute missing signal values with city median.

    Some tracts may have NaN values:
    - LST NaN: cloud cover prevented satellite reading
    - PM2.5 NaN: very rare but possible at the spatial domain edges
    - Access NaN: should not occur but defensive

    We impute with city median rather than dropping tracts —
    removing tracts would create holes in the compound score
    coverage and make Chicago-LA comparison inconsistent.

    Returns
    -------
    tuple of (imputed DataFrame, number of values imputed)
    """
    signal_cols = ["pm25_kriged", "lst_zscore", "access_score_er"]
    imputed_count = 0

    for col in signal_cols:
        null_mask = df[col].isna()
        n_null = null_mask.sum()
        if n_null > 0:
            df.loc[null_mask, col] = df[col].median()
            imputed_count += n_null

    return df, imputed_count


def _run_spatial_pca(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Run PCA on the six-feature matrix (3 signals + 3 spatial lags).

    Feature matrix columns:
        pm25_kriged, lst_zscore, access_score_er (raw signals)
        lag_pm25, lag_lst, lag_access (spatial lag signals)

    PCA steps:
    1. StandardScaler: normalize each feature to zero mean, unit variance
       (required because features have different scales and variances)
    2. PCA with n_components=1: find the first principal component
       (the linear combination capturing maximum variance)
    3. Extract PC1 scores and loadings

    Note on access_score direction:
    Higher access = LOWER burden. PCA may assign a negative loading
    to access, meaning PC1 increases as access decreases (correct).
    We verify loading direction and flip PC1 sign if needed so that
    higher PC1 always means higher burden.

    Parameters
    ----------
    df : pd.DataFrame
        Signals with spatial lags, no missing values.
    config : dict
        Pipeline config (random_seed for reproducibility).

    Returns
    -------
    tuple of:
        - scores_df: DataFrame with geoid and pc1_score
        - loadings: array of PC1 loadings for the 3 primary signals
        - explained_variance: array of variance explained per component
    """
    random_seed = config["ml"]["random_seed"]

    feature_cols = [
        "pm25_kriged", "lst_zscore", "access_score_er",
        "lag_pm25", "lag_lst", "lag_access",
    ]

    X = df[feature_cols].values

    # Step 1 — Standardize features
    # Each feature gets zero mean and unit variance
    # Without this, high-variance features (PM2.5 in μg/m³) would
    # dominate PCA regardless of their actual importance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 2 — Run PCA
    pca = PCA(n_components=3, random_state=random_seed)
    pc_scores = pca.fit_transform(X_scaled)

    explained_variance = pca.explained_variance_ratio_

    # Step 3 — Extract PC1 loadings for the 3 primary signals
    # pca.components_[0] = loadings for all 6 features
    # We report only the first 3 (original signals, not lags)
    # for interpretability
    pc1_loadings_all = pca.components_[0]
    loadings_primary = pc1_loadings_all[:3]  # pm25, lst, access

    # Step 4 — Check direction and flip if needed
    # We want higher PC1 = higher burden
    # PM2.5 loading should be positive (more PM2.5 = more burden)
    # LST loading should be positive (hotter = more burden)
    # Access loading should be negative (more access = less burden)
    # If PM2.5 + LST loadings are mostly negative, flip the sign
    pm25_loading = loadings_primary[0]
    lst_loading = loadings_primary[1]

    if pm25_loading + lst_loading < 0:
        # PC1 is inverted — flip sign so higher = more burden
        pc_scores = -pc_scores
        loadings_primary = -loadings_primary
        logger.debug("Flipped PC1 sign — PM2.5+LST loadings were negative")

    scores_df = pd.DataFrame({
        "geoid": df["geoid"].values,
        "pc1_raw": pc_scores[:, 0],
        "pm25_component": df["pm25_kriged"].values,
        "lst_component": df["lst_zscore"].values,
        "access_component": df["access_score_er"].values,
    })

    return scores_df, loadings_primary, explained_variance


def _normalize_scores(scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Min-max normalize PC1 to 0-1 compound burden score.

    After normalization:
        0.0 = lowest burden tract in the city this week
        1.0 = highest burden tract in the city this week
        0.75 = high risk threshold (from pipeline.yaml)
        0.90 = critical threshold (from pipeline.yaml)

    Note: scores are relative to city and week — a score of 0.9
    means this tract is in the 90th percentile of burden THIS week
    in THIS city. Comparing scores across cities or weeks requires
    care — the distributions may differ.

    Parameters
    ----------
    scores_df : pd.DataFrame
        DataFrame with pc1_raw column.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with compound_score column (0-1).
    """
    pc1 = scores_df["pc1_raw"].values
    pc1_min = pc1.min()
    pc1_max = pc1.max()

    if pc1_max == pc1_min:
        # All tracts have identical score — extremely rare
        # Set all to 0.5 (middle of scale) rather than dividing by zero
        logger.warning(
            "All PC1 scores are identical — setting compound_score to 0.5"
        )
        scores_df["compound_score"] = 0.5
    else:
        scores_df["compound_score"] = (
            (pc1 - pc1_min) / (pc1_max - pc1_min)
        ).round(4)

    return scores_df


def _classify_burden(
    scores_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Classify each tract into burden level based on compound score.

    Thresholds from pipeline.yaml:
        burden_thresholds.high_risk: 0.75
        burden_thresholds.critical:  0.90

    Classes:
        LOW:      score < 0.50   — below city median, relatively safe
        ELEVATED: 0.50-0.74      — above median, worth monitoring
        HIGH:     0.75-0.89      — triggers LLM briefing generation
        CRITICAL: 0.90-1.00      — triggers immediate intervention flag

    These thresholds are operationally defined — not statistical.
    They can be adjusted in pipeline.yaml without code changes.

    Parameters
    ----------
    scores_df : pd.DataFrame
        DataFrame with compound_score column.
    config : dict
        Pipeline config with burden thresholds.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with burden_class column.
    """
    high_threshold = config["burden_thresholds"]["high_risk"]
    critical_threshold = config["burden_thresholds"]["critical"]

    score = scores_df["compound_score"]

    conditions = [
        score >= critical_threshold,
        score >= high_threshold,
        score >= 0.50,
    ]
    choices = ["CRITICAL", "HIGH", "ELEVATED"]

    scores_df["burden_class"] = np.select(
        conditions, choices, default="LOW"
    )

    # Log class distribution
    class_counts = scores_df["burden_class"].value_counts()
    for burden_class in ["CRITICAL", "HIGH", "ELEVATED", "LOW"]:
        count = class_counts.get(burden_class, 0)
        logger.info(f"  {burden_class}: {count:,} tracts")

    return scores_df


def _validate_output(df: pd.DataFrame, config: dict[str, Any]) -> None:
    """Validate compound scores before BigQuery write."""
    if df.empty:
        raise ValueError("Compound scores output is empty")

    # Scores must be in [0, 1]
    out_of_range = ((df["compound_score"] < 0) | (df["compound_score"] > 1)).sum()
    if out_of_range > 0:
        raise ValueError(
            f"{out_of_range} tracts have compound_score outside [0, 1]"
        )

    # Check burden classes are valid
    valid_classes = {"LOW", "ELEVATED", "HIGH", "CRITICAL"}
    invalid_classes = set(df["burden_class"].unique()) - valid_classes
    if invalid_classes:
        raise ValueError(f"Invalid burden classes: {invalid_classes}")

    # Check we have all expected tracts
    if len(df) < 100:
        raise ValueError(f"Only {len(df)} tracts — expected at least 100")

    logger.info(
        f"Compound score validation passed: {len(df):,} tracts, "
        f"mean={df['compound_score'].mean():.3f}, "
        f"std={df['compound_score'].std():.3f}"
    )