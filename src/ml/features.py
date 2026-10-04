"""
features.py
===========
Engineer the feature matrix for the CHBIndex ML models.

Takes weekly compound burden scores and augments them with:
  1. Temporal lag features — how scores changed over time
  2. H3 spatial lag features — neighborhood context
  3. Socioeconomic interaction features — vulnerability amplifiers

The resulting feature matrix is the input to train.py (Models A, B, C).
Model A uses only temporal + socioeconomic features.
Model B adds H3 spatial lag features.
The difference in AUC between A and B is the A/B test result.

Why H3 spatial lags:
  Compound burden doesn't respect tract boundaries. Pollution drifts,
  heat islands span multiple tracts, and populations cross admin lines
  for care. H3 spatial lags encode neighborhood spillover explicitly —
  a tract surrounded by high-burden neighbors is more likely to stay
  high-burden next week than an isolated high-burden tract.

Input:
    BigQuery: compound_scores (52+ weeks of history)
    BigQuery: tract_geometries (centroid + population + demographics)
    BigQuery: cdc_places_health (health outcome rates)

Output:
    BigQuery table: ml_features (one row per tract per week)
    Two feature sets written:
        features_model_a: temporal + socioeconomic only
        features_model_b: + H3 spatial lags

Usage:
    from src.ml.features import engineer_features
    df_a, df_b = engineer_features(config)
"""

from datetime import datetime, timedelta
from typing import Any

import h3
import numpy as np
import pandas as pd

from src.utils.bigquery import read_query, write_dataframe
from src.utils.logger import get_logger

logger = get_logger(__name__)

# H3 resolution for spatial indexing
# Resolution 7 = hexagons ~5.16 km² — appropriate for census tract scale
# Resolution 8 would be too fine (0.74 km²), resolution 6 too coarse (36.1 km²)
H3_RESOLUTION = 7

# Number of H3 neighbor rings to compute spatial lags for
# Ring 1 = 6 immediate neighbors
# Ring 2 = 18 second-order neighbors (the ring around ring 1)
H3_LAG_RINGS = 2

# Weeks of historical scores to use for temporal features
# 4 weeks captures short-term trend without requiring too much history
TEMPORAL_LAG_WEEKS = [1, 2, 4]

# Target variable definition
# A tract is "high risk" if compound score >= this threshold
# Must match burden_thresholds.high_risk in pipeline.yaml
HIGH_RISK_THRESHOLD = 0.75


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def engineer_features(
    config: dict[str, Any],
    week_start: datetime | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Engineer ML feature matrices for Models A and B.

    Parameters
    ----------
    config : dict
        Merged pipeline config.
    week_start : datetime, optional
        Week to engineer features for. Defaults to most recent Monday.

    Returns
    -------
    tuple of (df_model_a, df_model_b)
        df_model_a: temporal + socioeconomic features only (no spatial lags)
        df_model_b: all features including H3 spatial lags
        Both include target variable: is_high_risk_next_week
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
        f"Engineering features for {config['city_name']} "
        f"week of {week_str}"
    )

    # Step 1 — Load historical compound scores (52 weeks)
    # We need history to compute temporal lags
    scores_history = _load_score_history(config, week_start)
    logger.info(
        f"Loaded {len(scores_history):,} score-week observations "
        f"across {scores_history['geoid'].nunique():,} tracts"
    )

    # Step 2 — Pivot to wide format: one row per tract, one col per week
    scores_wide = _pivot_scores_wide(scores_history)

    # Step 3 — Engineer temporal lag features
    temporal_df = _engineer_temporal_features(scores_wide, week_start)
    logger.info(f"Temporal features: {len(temporal_df.columns)} columns")

    # Step 4 — Load tract demographics for socioeconomic features
    demographics_df = _load_demographics(config)

    # Step 5 — Engineer socioeconomic interaction features
    socio_df = _engineer_socioeconomic_features(
        temporal_df, demographics_df
    )
    logger.info(f"Socioeconomic features added: {len(socio_df.columns)} columns")

    # Step 6 — Add target variable
    # Target: will this tract be high risk NEXT week?
    # We shift compound scores forward by 1 week to get next-week labels
    socio_df = _add_target_variable(socio_df, scores_history, week_start)

    # Step 7 — Assign H3 cell IDs to tract centroids
    tract_h3 = _assign_h3_cells(demographics_df, config)

    # Step 8 — Compute H3 spatial lag features
    spatial_df = _compute_h3_spatial_lags(
        socio_df, tract_h3, scores_wide, week_start
    )
    logger.info(f"H3 spatial lag features added: {len(spatial_df.columns)} columns")

    # Step 9 — Split into Model A and Model B feature sets
    df_model_a, df_model_b = _split_feature_sets(socio_df, spatial_df)

    logger.info(
        f"Feature engineering complete: "
        f"Model A={len(df_model_a.columns)} features, "
        f"Model B={len(df_model_b.columns)} features, "
        f"{df_model_b['is_high_risk_next_week'].sum()} high-risk labels"
    )

    # Step 10 — Write both feature sets to BigQuery
    write_dataframe(df_model_a, "ml_features_model_a", config)
    write_dataframe(df_model_b, "ml_features_model_b", config)

    return df_model_a, df_model_b


# ---------------------------------------------------------------------------
# Private functions
# ---------------------------------------------------------------------------

def _load_score_history(
    config: dict[str, Any],
    week_start: datetime,
) -> pd.DataFrame:
    """
    Load 52 weeks of compound scores per tract from BigQuery.

    We need history to compute temporal lags and the next-week
    target variable. 52 weeks = 1 year captures full seasonality.

    Parameters
    ----------
    config : dict
        Pipeline config.
    week_start : datetime
        Current week — we load history UP TO AND INCLUDING next week
        (needed for next-week target variable computation).

    Returns
    -------
    pd.DataFrame
        Long format: one row per tract per week.
        Columns: geoid, compound_score, week_start_date
    """
    # Load 54 weeks of history to have buffer for lag computation
    # 52 weeks of features + 1 week target + 1 week buffer
    history_start = week_start - timedelta(weeks=54)
    # Load one week ahead for next-week target variable
    history_end = week_start + timedelta(weeks=1)

    df = read_query(
        f"""
        SELECT
            geoid,
            compound_score,
            burden_class,
            week_start_date
        FROM {{dataset}}.compound_scores
        WHERE
            city_fips = '{config["full_fips"]}'
            AND week_start_date >= '{history_start.strftime("%Y-%m-%d")}'
            AND week_start_date <= '{history_end.strftime("%Y-%m-%d")}'
        ORDER BY geoid, week_start_date
        """,
        config,
    )

    if df.empty:
        raise ValueError(
            f"No compound score history found for FIPS {config['full_fips']}. "
            f"Run spatial transform pipeline first."
        )

    df["week_start_date"] = pd.to_datetime(df["week_start_date"])
    return df


def _pivot_scores_wide(scores_long: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot compound scores from long to wide format.

    Long format (input):
        geoid       | week_start_date | compound_score
        06037204200 | 2026-08-18      | 0.82
        06037204200 | 2026-08-11      | 0.79

    Wide format (output):
        geoid       | score_2026-08-18 | score_2026-08-11 | ...
        06037204200 | 0.82             | 0.79             | ...

    Wide format makes temporal feature computation much faster —
    each row is one tract, each column is one week's score.

    Parameters
    ----------
    scores_long : pd.DataFrame
        Long format compound scores.

    Returns
    -------
    pd.DataFrame
        Wide format with one row per tract, one column per week.
    """
    wide = scores_long.pivot_table(
        index="geoid",
        columns="week_start_date",
        values="compound_score",
        aggfunc="mean",
    )

    # Rename columns to score_{date} format
    wide.columns = [
        f"score_{d.strftime('%Y-%m-%d')}"
        for d in wide.columns
    ]

    return wide.reset_index()


def _engineer_temporal_features(
    scores_wide: pd.DataFrame,
    week_start: datetime,
) -> pd.DataFrame:
    """
    Engineer temporal lag and trend features from score history.

    Features computed for each tract:
        score_current:    compound score this week
        lag_{n}w:         compound score n weeks ago
        rolling_mean_4w:  mean score over last 4 weeks
        rolling_std_4w:   std of score over last 4 weeks (volatility)
        delta_1w:         this week - last week (rate of change)
        delta_4w:         this week - 4 weeks ago (medium-term trend)
        weeks_high_risk:  how many of last 4 weeks was tract high risk

    Parameters
    ----------
    scores_wide : pd.DataFrame
        Wide format scores with one column per week.
    week_start : datetime
        Current week start date.

    Returns
    -------
    pd.DataFrame
        One row per tract with temporal features.
    """
    week_str = week_start.strftime("%Y-%m-%d")
    current_col = f"score_{week_str}"

    df = pd.DataFrame({"geoid": scores_wide["geoid"]})

    # Current week score
    df["score_current"] = scores_wide.get(current_col, np.nan)

    # Temporal lag features
    for lag_weeks in TEMPORAL_LAG_WEEKS:
        lag_date = week_start - timedelta(weeks=lag_weeks)
        lag_col = f"score_{lag_date.strftime('%Y-%m-%d')}"
        df[f"lag_{lag_weeks}w"] = scores_wide.get(lag_col, np.nan)

    # Rolling statistics over last 4 weeks
    lag_cols_4w = []
    for w in range(1, 5):
        d = week_start - timedelta(weeks=w)
        col = f"score_{d.strftime('%Y-%m-%d')}"
        if col in scores_wide.columns:
            lag_cols_4w.append(col)

    if lag_cols_4w:
        lag_matrix = scores_wide[lag_cols_4w]
        df["rolling_mean_4w"] = lag_matrix.mean(axis=1).values
        df["rolling_std_4w"] = lag_matrix.std(axis=1).values

        # Count weeks in high risk over last 4 weeks
        df["weeks_high_risk_4w"] = (lag_matrix >= HIGH_RISK_THRESHOLD).sum(axis=1).values
    else:
        df["rolling_mean_4w"] = np.nan
        df["rolling_std_4w"] = np.nan
        df["weeks_high_risk_4w"] = 0

    # Rate of change features
    df["delta_1w"] = df["score_current"] - df.get("lag_1w", np.nan)
    df["delta_4w"] = df["score_current"] - df.get("lag_4w", np.nan)

    # Season feature — week of year captures seasonal patterns
    # Heat and air quality risk peaks in summer (weeks 26-39)
    df["week_of_year"] = week_start.isocalendar()[1]
    df["is_summer"] = int(26 <= df["week_of_year"].iloc[0] <= 39)

    return df


def _load_demographics(config: dict[str, Any]) -> pd.DataFrame:
    """
    Load tract demographics from BigQuery.

    Pulls from tract_geometries (population, median_income)
    and cdc_places_health (health outcome rates).
    These are used for socioeconomic interaction features
    and equity audit stratification.

    Returns
    -------
    pd.DataFrame
        One row per tract with demographic and health columns.
    """
    df = read_query(
        f"""
        SELECT
            t.geoid,
            t.population,
            t.median_income,
            t.centroid_lat,
            t.centroid_lon,
            c.asthma_pct,
            c.copd_pct,
            c.heart_disease_pct,
            c.diabetes_pct,
            c.high_bp_pct
        FROM
            {{dataset}}.tract_geometries t
        LEFT JOIN
            {{dataset}}.cdc_places_health c
            ON t.geoid = c.geoid
        WHERE
            t.city_fips = '{config["full_fips"]}'
        ORDER BY t.geoid
        """,
        config,
    )

    # Compute elderly proxy — CDC PLACES doesn't have age directly
    # We use diabetes + heart disease as proxy for elderly vulnerability
    # (both conditions are strongly age-correlated)
    df["vulnerability_index"] = (
        df["asthma_pct"].fillna(0) * 0.3 +
        df["heart_disease_pct"].fillna(0) * 0.4 +
        df["copd_pct"].fillna(0) * 0.3
    )

    return df


def _engineer_socioeconomic_features(
    temporal_df: pd.DataFrame,
    demographics_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Engineer socioeconomic interaction features.

    Interaction features explicitly encode that the same environmental
    risk level has different health impacts depending on the population's
    vulnerability. A PM2.5 spike in a tract with 25% asthma prevalence
    is more dangerous than the same spike in a tract with 5% asthma.

    Multiplying the signal by the vulnerability amplifies the feature
    in high-risk populations — giving the model this prior knowledge.

    Features computed:
        vulnerability_score:     overall health vulnerability index
        income_quartile:         1-4 (for equity audit stratification)
        population_density_proxy: population / tract (relative)
        asthma_pct:              direct feature (respiratory vulnerability)
        median_income_norm:      normalized median income

    Parameters
    ----------
    temporal_df : pd.DataFrame
        Temporal features from _engineer_temporal_features().
    demographics_df : pd.DataFrame
        Demographic data from _load_demographics().

    Returns
    -------
    pd.DataFrame
        Temporal features + socioeconomic features merged.
    """
    df = temporal_df.merge(
        demographics_df[[
            "geoid", "population", "median_income",
            "asthma_pct", "copd_pct", "heart_disease_pct",
            "vulnerability_index",
        ]],
        on="geoid",
        how="left",
    )

    # Normalize median income to 0-1 scale
    income = df["median_income"].fillna(df["median_income"].median())
    df["median_income_norm"] = (
        (income - income.min()) / (income.max() - income.min())
    )

    # Income quartile for equity audit — 1=lowest, 4=highest
    df["income_quartile"] = pd.qcut(
        df["median_income"].fillna(df["median_income"].median()),
        q=4,
        labels=[1, 2, 3, 4],
    ).astype(int)

    # Fill missing health outcome rates with city median
    for col in ["asthma_pct", "copd_pct", "heart_disease_pct", "vulnerability_index"]:
        df[col] = df[col].fillna(df[col].median())

    # Fill missing temporal features with city median
    temporal_cols = [
        "score_current", "lag_1w", "lag_2w", "lag_4w",
        "rolling_mean_4w", "rolling_std_4w", "delta_1w", "delta_4w",
    ]
    for col in temporal_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


def _add_target_variable(
    df: pd.DataFrame,
    scores_history: pd.DataFrame,
    week_start: datetime,
) -> pd.DataFrame:
    """
    Add next-week high-risk label as the target variable.

    Target: is_high_risk_next_week (binary 0/1)
    True if compound_score >= HIGH_RISK_THRESHOLD in the week
    AFTER week_start.

    Why predict next week rather than current week?
    The compound score for the current week is already known —
    predicting it would be pointless. We want to predict FUTURE
    risk so emergency resources can be pre-positioned before
    a dangerous week arrives, not after.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame to add target to.
    scores_history : pd.DataFrame
        Historical scores including next week's data.
    week_start : datetime
        Current week — target is week_start + 1 week.

    Returns
    -------
    pd.DataFrame
        df with is_high_risk_next_week column added.
    """
    next_week = week_start + timedelta(weeks=1)
    next_week_str = next_week.strftime("%Y-%m-%d")

    # Get next week's compound scores
    next_week_scores = scores_history[
        scores_history["week_start_date"] == next_week
    ][["geoid", "compound_score"]].rename(
        columns={"compound_score": "next_week_score"}
    )

    if next_week_scores.empty:
        logger.warning(
            f"No scores for next week ({next_week_str}). "
            f"Target variable will be NaN — this week cannot be used for training."
        )
        df["is_high_risk_next_week"] = np.nan
        df["next_week_score"] = np.nan
        return df

    df = df.merge(next_week_scores, on="geoid", how="left")
    df["is_high_risk_next_week"] = (
        df["next_week_score"] >= HIGH_RISK_THRESHOLD
    ).astype(int)

    high_risk_pct = df["is_high_risk_next_week"].mean() * 100
    logger.info(
        f"Target variable: {df['is_high_risk_next_week'].sum()} "
        f"high-risk tracts next week ({high_risk_pct:.1f}%)"
    )

    return df


def _assign_h3_cells(
    demographics_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Assign each tract centroid to its H3 resolution-7 cell.

    H3 divides the earth into hexagonal cells at multiple resolutions.
    Resolution 7 cells are ~5.16 km² — slightly larger than a typical
    LA census tract (~1.5 km²), so most tracts map to unique cells
    but nearby tracts may share a cell.

    For each cell, we also compute the parent cell at resolution 6
    — used to define second-order neighbors (ring 2).

    Parameters
    ----------
    demographics_df : pd.DataFrame
        Tract centroids with centroid_lat, centroid_lon columns.
    config : dict
        Pipeline config with h3_resolution setting.

    Returns
    -------
    pd.DataFrame
        geoid, h3_cell_r7, h3_cell_r6 (parent), h3_neighbors_r7
    """
    resolution = config["spatial"]["h3_resolution"]

    logger.debug(
        f"Assigning tract centroids to H3 resolution {resolution} cells"
    )

    records = []
    for _, row in demographics_df.iterrows():
        if pd.isna(row["centroid_lat"]) or pd.isna(row["centroid_lon"]):
            continue

        # Assign to H3 cell at our resolution
        # h3.latlng_to_cell(lat, lng, resolution) → H3 cell ID string
        cell_r7 = h3.latlng_to_cell(
            row["centroid_lat"],
            row["centroid_lon"],
            resolution,
        )

        # Parent cell at resolution 6 (coarser — for ring-2 neighbors)
        cell_r6 = h3.cell_to_parent(cell_r7, resolution - 1)

        # Immediate neighbors (ring 1) — 6 hexagons surrounding this cell
        neighbors_r1 = list(h3.grid_disk(cell_r7, 1) - {cell_r7})

        # Second-order neighbors (ring 2) — 12 hexagons in the outer ring
        # grid_disk(k=2) returns all cells within 2 steps
        # Subtract ring 1 and self to get only ring 2
        neighbors_r2 = list(
            h3.grid_disk(cell_r7, 2) - h3.grid_disk(cell_r7, 1)
        )

        records.append({
            "geoid": row["geoid"],
            "h3_cell_r7": cell_r7,
            "h3_cell_r6": cell_r6,
            "h3_neighbors_r1": neighbors_r1,
            "h3_neighbors_r2": neighbors_r2,
        })

    tract_h3 = pd.DataFrame(records)
    logger.debug(
        f"Assigned {len(tract_h3)} tracts to H3 cells. "
        f"Unique cells: {tract_h3['h3_cell_r7'].nunique()}"
    )

    return tract_h3


def _compute_h3_spatial_lags(
    df: pd.DataFrame,
    tract_h3: pd.DataFrame,
    scores_wide: pd.DataFrame,
    week_start: datetime,
) -> pd.DataFrame:
    """
    Compute H3 spatial lag features for the current week.

    For each tract, compute the population-weighted mean compound
    score of its H3 ring-1 and ring-2 neighbors.

    Why population-weighted?
    A neighbor tract with 50,000 people should contribute more
    to the spatial lag than one with 2,000 people — more people
    means more cross-tract interaction (commuting, shared resources).

    Features computed:
        h3_lag_ring1:      weighted mean score of 6 ring-1 neighbors
        h3_lag_ring2:      weighted mean score of 12 ring-2 neighbors
        h3_lag_diff:       ring1 - ring2 (is risk locally concentrated?)
        h3_max_neighbor:   max score among all ring-1 neighbors
        h3_high_risk_neighbors: count of ring-1 neighbors above threshold

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame to add spatial lags to.
    tract_h3 : pd.DataFrame
        H3 cell assignments from _assign_h3_cells().
    scores_wide : pd.DataFrame
        Wide format scores with current week column.
    week_start : datetime
        Current week for score lookup.

    Returns
    -------
    pd.DataFrame
        df with H3 spatial lag columns added.
    """
    week_str = week_start.strftime("%Y-%m-%d")
    score_col = f"score_{week_str}"

    # Build lookup: h3_cell → mean compound score this week
    # When multiple tracts share an H3 cell, average their scores
    cell_scores = tract_h3.merge(
        scores_wide[["geoid", score_col]],
        on="geoid",
        how="left",
    ).groupby("h3_cell_r7")[score_col].mean().to_dict()

    lag_records = []

    for _, row in tract_h3.iterrows():
        geoid = row["geoid"]
        neighbors_r1 = row["h3_neighbors_r1"]
        neighbors_r2 = row["h3_neighbors_r2"]

        # Ring 1 scores
        r1_scores = [
            cell_scores[cell]
            for cell in neighbors_r1
            if cell in cell_scores and not pd.isna(cell_scores[cell])
        ]

        # Ring 2 scores
        r2_scores = [
            cell_scores[cell]
            for cell in neighbors_r2
            if cell in cell_scores and not pd.isna(cell_scores[cell])
        ]

        h3_lag_r1 = np.mean(r1_scores) if r1_scores else np.nan
        h3_lag_r2 = np.mean(r2_scores) if r2_scores else np.nan

        lag_records.append({
            "geoid": geoid,
            "h3_lag_ring1": round(h3_lag_r1, 4) if not np.isnan(h3_lag_r1) else np.nan,
            "h3_lag_ring2": round(h3_lag_r2, 4) if not np.isnan(h3_lag_r2) else np.nan,
            "h3_lag_diff": round(h3_lag_r1 - h3_lag_r2, 4)
                if not (np.isnan(h3_lag_r1) or np.isnan(h3_lag_r2))
                else np.nan,
            "h3_max_neighbor": round(max(r1_scores), 4) if r1_scores else np.nan,
            "h3_high_risk_neighbors": sum(
                1 for s in r1_scores if s >= HIGH_RISK_THRESHOLD
            ),
        })

    lag_df = pd.DataFrame(lag_records)

    # Merge spatial lags back onto feature DataFrame
    spatial_df = df.merge(lag_df, on="geoid", how="left")

    logger.debug(
        f"H3 spatial lags computed: "
        f"mean ring1={spatial_df['h3_lag_ring1'].mean():.3f}, "
        f"mean ring2={spatial_df['h3_lag_ring2'].mean():.3f}"
    )

    return spatial_df


def _split_feature_sets(
    socio_df: pd.DataFrame,
    spatial_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split into Model A and Model B feature sets.

    Model A — tabular only (no spatial lags):
        Temporal features + socioeconomic features
        Used as baseline in A/B test

    Model B — with H3 spatial lags:
        All Model A features + H3 ring1/ring2 lag features
        Challenger in A/B test

    Both include the same target variable and metadata columns
    so they can be compared fairly in evaluate.py.

    Parameters
    ----------
    socio_df : pd.DataFrame
        Features WITHOUT spatial lags (temporal + socioeconomic).
    spatial_df : pd.DataFrame
        Features WITH spatial lags added.

    Returns
    -------
    tuple of (df_model_a, df_model_b)
    """
    # Metadata and target columns — included in both
    meta_cols = [
        "geoid", "week_of_year", "is_high_risk_next_week",
        "next_week_score", "income_quartile",
    ]

    # Model A feature columns
    model_a_features = [
        "score_current", "lag_1w", "lag_2w", "lag_4w",
        "rolling_mean_4w", "rolling_std_4w",
        "delta_1w", "delta_4w",
        "weeks_high_risk_4w", "is_summer",
        "vulnerability_index", "median_income_norm",
        "asthma_pct", "copd_pct", "heart_disease_pct",
        "population",
    ]

    # Model B adds H3 spatial lag features on top of Model A
    model_b_extra = [
        "h3_lag_ring1", "h3_lag_ring2", "h3_lag_diff",
        "h3_max_neighbor", "h3_high_risk_neighbors",
    ]

    # Filter to columns that actually exist
    a_cols = meta_cols + [c for c in model_a_features if c in socio_df.columns]
    b_cols = a_cols + [c for c in model_b_extra if c in spatial_df.columns]

    df_model_a = socio_df[a_cols].copy()
    df_model_b = spatial_df[b_cols].copy()

    logger.info(
        f"Model A: {len([c for c in a_cols if c not in meta_cols])} features, "
        f"Model B: {len([c for c in b_cols if c not in meta_cols])} features "
        f"(+{len(model_b_extra)} spatial lags)"
    )

    return df_model_a, df_model_b