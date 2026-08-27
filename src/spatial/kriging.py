"""
kriging.py
==========
Ordinary Kriging interpolation of PM2.5 from sparse EPA monitor
locations to all census tract centroids in the configured city.

This module solves the fundamental spatial data problem in the
compound burden index: EPA has 11 monitors for 2,300 LA County
census tracts. Most tracts have no nearby monitor. Kriging uses
the spatial correlation structure of the 11 known readings to
mathematically estimate PM2.5 at all 2,300 tract centroids.

Method: Ordinary Kriging (PyKrige)
  - Fits a spherical variogram to empirical semivariances
  - Solves the Kriging system for optimal weights
  - Produces both predictions and variance estimates
  - Validated with leave-one-out cross-validation

Why Ordinary Kriging over IDW or Nearest Neighbor:
  - IDW assumes correlation decays with 1/distance — arbitrary
  - Kriging learns the actual decay from the data (variogram)
  - Kriging is BLUE: Best Linear Unbiased Estimator
  - Kriging provides uncertainty estimates (variance) per prediction

Input:
    BigQuery table: epa_pm25_raw (monitor readings)
    BigQuery table: tract_geometries (tract centroids)

Output:
    BigQuery table: pm25_kriged (one row per tract per week)
    Columns: geoid, pm25_kriged, pm25_kriged_variance,
             week_start_date, city_fips

Usage:
    from src.spatial.kriging import run_kriging
    df = run_kriging(config)
"""

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging

from src.utils.bigquery import read_query, write_dataframe
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def run_kriging(
    config: dict[str, Any],
    week_start: datetime | None = None,
) -> pd.DataFrame:
    """
    Interpolate PM2.5 from EPA monitors to all census tract centroids.

    Reads monitor readings and tract centroids from BigQuery,
    fits an Ordinary Kriging model, predicts PM2.5 at all tract
    centroids, validates with leave-one-out CV, and writes results
    to BigQuery.

    Parameters
    ----------
    config : dict
        Merged pipeline config from config_loader.load_config().
        Uses: spatial.kriging_variogram_model, spatial.kriging_n_jobs,
              full_fips, crs_projected

    week_start : datetime, optional
        Week to run Kriging for. Defaults to most recent Monday.

    Returns
    -------
    pd.DataFrame
        One row per tract with columns:
        geoid, pm25_kriged, pm25_kriged_variance,
        pm25_percentile, week_start_date, city_fips

    Examples
    --------
    >>> config = load_config("los_angeles")
    >>> df = run_kriging(config)
    >>> len(df)
    2310
    >>> df["pm25_kriged"].describe()
    count    2310.000000
    mean       13.24...
    std         3.81...
    min         8.12...
    max        28.94...
    """
    if week_start is None:
        today = datetime.now()
        week_start = today - timedelta(days=today.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)

    full_fips = config["full_fips"]
    week_str = week_start.strftime("%Y-%m-%d")

    logger.info(f"Running PM2.5 Kriging for {config['city_name']} week of {week_str}")

    # Step 1 — Load monitor readings from BigQuery
    monitors_df = _load_monitor_readings(config, week_str)
    logger.info(f"Loaded {len(monitors_df)} monitor readings for week of {week_str}")

    # Step 2 — Load tract centroids from BigQuery
    tracts_df = _load_tract_centroids(config)
    logger.info(f"Loaded {len(tracts_df):,} tract centroids")

    # Step 3 — Project coordinates to UTM for distance calculations
    # Kriging requires distances in meters — not degrees
    # Using degrees produces wrong variograms because
    # 1 degree longitude ≠ 1 degree latitude in distance
    monitors_utm, tracts_utm = _project_to_utm(
        monitors_df, tracts_df, config
    )

    # Step 4 — Fit Ordinary Kriging model and predict
    kriged_df = _fit_and_predict(
        monitors_utm=monitors_utm,
        tracts_utm=tracts_utm,
        config=config,
    )

    # Step 5 — Leave-one-out cross-validation
    # Validates how well Kriging predicts at unmonitored locations
    # This is the R² number you put on your resume
    cv_r2 = _leave_one_out_cv(monitors_utm, config)
    logger.info(f"Leave-one-out CV R²: {cv_r2:.3f}")

    # Step 6 — Join GEOID back to kriged predictions
    # tracts_utm still has geoid column — merge on index
    result_df = pd.DataFrame({
        "geoid": tracts_df["geoid"].values,
        "pm25_kriged": kriged_df["pm25_kriged"].values,
        "pm25_kriged_variance": kriged_df["pm25_kriged_variance"].values,
    })

    # Step 7 — Compute percentile rank within city
    # Used for compound score normalization
    result_df["pm25_percentile"] = (
        result_df["pm25_kriged"]
        .rank(pct=True)
        .mul(100)
        .round(1)
    )

    # Step 8 — Add metadata
    result_df["week_start_date"] = week_start.date()
    result_df["city_fips"] = full_fips
    result_df["cv_r2"] = round(cv_r2, 4)

    # Step 9 — Validate output
    _validate_output(result_df, len(tracts_df))

    # Step 10 — Write to BigQuery
    write_dataframe(result_df, "pm25_kriged", config)

    logger.info(
        f"Kriging complete: {len(result_df):,} tracts, "
        f"PM2.5 range {result_df['pm25_kriged'].min():.1f}–"
        f"{result_df['pm25_kriged'].max():.1f} μg/m³, "
        f"CV R²={cv_r2:.3f}"
    )

    return result_df


# ---------------------------------------------------------------------------
# Private functions
# ---------------------------------------------------------------------------

def _load_monitor_readings(
    config: dict[str, Any],
    week_str: str,
) -> pd.DataFrame:
    """
    Load EPA PM2.5 monitor readings for the current week from BigQuery.

    Returns monitor locations (lat/lon) and their weekly mean PM2.5.
    These are the known values Kriging interpolates from.

    Parameters
    ----------
    config : dict
        Pipeline config.
    week_str : str
        Week start date as 'YYYY-MM-DD' string.

    Returns
    -------
    pd.DataFrame
        Monitor readings with columns:
        monitor_id, latitude, longitude, pm25_mean
    """
    query = """
        SELECT
            monitor_id,
            latitude,
            longitude,
            pm25_mean
        FROM {dataset}.epa_pm25_raw
        WHERE
            week_start_date = '{week}'
            AND city_fips = '{fips}'
            AND pm25_mean IS NOT NULL
            AND latitude IS NOT NULL
            AND longitude IS NOT NULL
        ORDER BY monitor_id
    """.format(
        dataset="{dataset}",  # substituted by read_query
        week=week_str,
        fips=config["full_fips"],
    )

    df = read_query(query, config)

    if len(df) < 3:
        raise ValueError(
            f"Need at least 3 monitors for Kriging, got {len(df)}. "
            f"Check EPA AQS data for week {week_str}."
        )

    return df


def _load_tract_centroids(config: dict[str, Any]) -> pd.DataFrame:
    """
    Load census tract centroids from BigQuery.

    These are the prediction locations — Kriging predicts PM2.5
    at each centroid. Centroids were computed in census_tiger.py
    using UTM projection for accuracy.

    Parameters
    ----------
    config : dict
        Pipeline config.

    Returns
    -------
    pd.DataFrame
        Tract centroids with columns: geoid, centroid_lat, centroid_lon
    """
    query = """
        SELECT
            geoid,
            centroid_lat,
            centroid_lon
        FROM {dataset}.tract_geometries
        WHERE
            city_fips = '{fips}'
            AND centroid_lat IS NOT NULL
            AND centroid_lon IS NOT NULL
        ORDER BY geoid
    """.format(
        dataset="{dataset}",
        fips=config["full_fips"],
    )

    df = read_query(query, config)

    if df.empty:
        raise ValueError(
            f"No tract centroids found for FIPS {config['full_fips']}. "
            f"Run census_tiger ingestion first."
        )

    return df


def _project_to_utm(
    monitors_df: pd.DataFrame,
    tracts_df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Project monitor and tract coordinates from WGS84 to UTM.

    Kriging computes distances between points. In geographic
    coordinates (degrees), distances are distorted — 1 degree
    of longitude is ~95km in LA but ~111km near the equator,
    and these units are incompatible with each other.

    UTM (Universal Transverse Mercator) projects to a flat plane
    with meters as units — consistent distances in all directions.
    LA uses UTM Zone 11N (EPSG:26911).
    Chicago uses UTM Zone 16N (EPSG:26916).

    The correct UTM zone comes from config["crs_projected"] —
    set per city in los_angeles.yaml / chicago.yaml.

    Parameters
    ----------
    monitors_df : pd.DataFrame
        Monitor readings with latitude/longitude in WGS84.
    tracts_df : pd.DataFrame
        Tract centroids with centroid_lat/centroid_lon in WGS84.
    config : dict
        Pipeline config containing crs_projected.

    Returns
    -------
    tuple of (monitors_utm, tracts_utm)
        Both DataFrames with x_utm, y_utm columns added (meters).
    """
    import pyproj

    crs_projected = config["crs_projected"]

    # Create transformer: WGS84 → UTM
    # always_xy=True ensures consistent x=longitude, y=latitude order
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326",
        crs_projected,
        always_xy=True,
    )

    logger.debug(f"Projecting coordinates to {crs_projected}")

    # Project monitor coordinates
    monitors_utm = monitors_df.copy()
    monitors_utm["x_utm"], monitors_utm["y_utm"] = transformer.transform(
        monitors_df["longitude"].values,
        monitors_df["latitude"].values,
    )

    # Project tract centroid coordinates
    tracts_utm = tracts_df.copy()
    tracts_utm["x_utm"], tracts_utm["y_utm"] = transformer.transform(
        tracts_df["centroid_lon"].values,
        tracts_df["centroid_lat"].values,
    )

    logger.debug(
        f"Monitor UTM range: x={monitors_utm['x_utm'].min():.0f}–"
        f"{monitors_utm['x_utm'].max():.0f}m, "
        f"y={monitors_utm['y_utm'].min():.0f}–"
        f"{monitors_utm['y_utm'].max():.0f}m"
    )

    return monitors_utm, tracts_utm


def _fit_and_predict(
    monitors_utm: pd.DataFrame,
    tracts_utm: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Fit Ordinary Kriging model and predict PM2.5 at tract centroids.

    PyKrige's OrdinaryKriging:
    1. Computes empirical variogram from monitor pairs
    2. Fits the specified variogram model (spherical by default)
    3. Solves the Kriging system for optimal weights
    4. Predicts at each tract centroid with uncertainty estimate

    The spherical variogram model:
        γ(h) = nugget + (sill-nugget) × [1.5(h/range) - 0.5(h/range)³]
                for h ≤ range
        γ(h) = sill    for h > range

    Where h is distance, γ is semivariance.

    Parameters
    ----------
    monitors_utm : pd.DataFrame
        Monitor readings with x_utm, y_utm, pm25_mean columns.
    tracts_utm : pd.DataFrame
        Tract centroids with x_utm, y_utm columns.
    config : dict
        Pipeline config with Kriging settings.

    Returns
    -------
    pd.DataFrame
        Kriging predictions with columns:
        pm25_kriged, pm25_kriged_variance
    """
    variogram_model = config["spatial"]["kriging_variogram_model"]
    n_jobs = config["spatial"]["kriging_n_jobs"]

    logger.info(
        f"Fitting Ordinary Kriging with variogram_model='{variogram_model}' "
        f"on {len(monitors_utm)} monitors"
    )

    # Initialize OrdinaryKriging
    # x = easting (UTM x), y = northing (UTM y), z = PM2.5 value
    # variogram_model: how spatial correlation decays with distance
    # enable_statistics=True: compute prediction variance
    # verbose=False: suppress PyKrige's internal logging
    ok = OrdinaryKriging(
        x=monitors_utm["x_utm"].values,
        y=monitors_utm["y_utm"].values,
        z=monitors_utm["pm25_mean"].values,
        variogram_model=variogram_model,
        enable_statistics=True,
        verbose=False,
        enable_plotting=False,
    )

    # Log fitted variogram parameters
    # These tell you about the spatial structure of PM2.5 in this city
    # Range tells you how far pollution influence extends
    logger.info(
        f"Fitted variogram parameters: "
        f"nugget={ok.variogram_model_parameters[0]:.4f}, "
        f"sill={ok.variogram_model_parameters[1]:.4f}, "
        f"range={ok.variogram_model_parameters[2]:.0f}m"
    )

    # Predict at all tract centroids
    # Returns: predictions array, variance array
    # backend='loop' is more memory efficient than 'vectorized'
    # for large prediction grids (2,300 tracts)
    logger.info(f"Predicting PM2.5 at {len(tracts_utm):,} tract centroids...")

    pm25_pred, pm25_var = ok.execute(
        "points",
        tracts_utm["x_utm"].values,
        tracts_utm["y_utm"].values,
        backend="loop",
        n_closest_points=n_jobs if n_jobs > 0 else None,
    )

    # PyKrige returns masked arrays — convert to regular numpy arrays
    # Masked values (outside variogram range) get filled with NaN
    pm25_pred = np.array(pm25_pred)
    pm25_var = np.array(pm25_var)

    # Clip negative predictions to 0 — PM2.5 cannot be negative
    # Small negative values can occur near the edge of the prediction domain
    pm25_pred = np.clip(pm25_pred, 0, None)
    pm25_var = np.clip(pm25_var, 0, None)

    logger.info(
        f"Kriging predictions complete: "
        f"mean={pm25_pred.mean():.2f} μg/m³, "
        f"std={pm25_pred.std():.2f} μg/m³, "
        f"mean variance={pm25_var.mean():.4f}"
    )

    return pd.DataFrame({
        "pm25_kriged": pm25_pred.round(3),
        "pm25_kriged_variance": pm25_var.round(6),
    })


def _leave_one_out_cv(
    monitors_utm: pd.DataFrame,
    config: dict[str, Any],
) -> float:
    """
    Validate Kriging accuracy with leave-one-out cross-validation.

    For each monitor in turn:
      1. Remove it from the training set
      2. Fit Kriging on the remaining N-1 monitors
      3. Predict PM2.5 at the held-out monitor's location
      4. Compare prediction to actual reading

    This simulates how well Kriging predicts at unmonitored locations
    — which is exactly what we're doing for the 2,300 tract centroids.

    The R² score is your resume number:
      R²=0.78 means Kriging explains 78% of PM2.5 variance
      across unmonitored locations using spatial correlation alone.

    Parameters
    ----------
    monitors_utm : pd.DataFrame
        All monitor readings with UTM coordinates.
    config : dict
        Pipeline config with Kriging settings.

    Returns
    -------
    float
        R² score of leave-one-out cross-validation (0.0 to 1.0).
        Higher is better. Above 0.6 is acceptable for 11 monitors.
    """
    variogram_model = config["spatial"]["kriging_variogram_model"]
    n_monitors = len(monitors_utm)

    if n_monitors < 4:
        logger.warning(
            f"Only {n_monitors} monitors — LOO-CV needs at least 4. "
            f"Skipping CV."
        )
        return float("nan")

    logger.info(f"Running leave-one-out CV on {n_monitors} monitors")

    actuals = []
    predictions = []

    for i in range(n_monitors):
        # Create train set: all monitors except monitor i
        train = monitors_utm.drop(index=monitors_utm.index[i])

        # Held-out monitor: monitor i
        held_out = monitors_utm.iloc[i]

        try:
            # Fit Kriging on N-1 monitors
            ok_cv = OrdinaryKriging(
                x=train["x_utm"].values,
                y=train["y_utm"].values,
                z=train["pm25_mean"].values,
                variogram_model=variogram_model,
                verbose=False,
                enable_plotting=False,
            )

            # Predict at held-out monitor location
            pred, _ = ok_cv.execute(
                "points",
                np.array([held_out["x_utm"]]),
                np.array([held_out["y_utm"]]),
            )

            actuals.append(held_out["pm25_mean"])
            predictions.append(float(np.array(pred)[0]))

        except Exception as e:
            logger.warning(f"LOO-CV failed for monitor {i}: {e}")
            continue

    if len(actuals) < 3:
        logger.warning("Too few successful CV folds — returning NaN")
        return float("nan")

    # Compute R² = 1 - SS_residual / SS_total
    actuals = np.array(actuals)
    predictions = np.array(predictions)

    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - actuals.mean()) ** 2)

    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    logger.info(
        f"LOO-CV complete: R²={r2:.3f}, "
        f"MAE={np.mean(np.abs(actuals - predictions)):.3f} μg/m³"
    )

    return float(r2)


def _validate_output(df: pd.DataFrame, expected_tract_count: int) -> None:
    """
    Validate Kriging output before writing to BigQuery.

    Parameters
    ----------
    df : pd.DataFrame
        Kriging output DataFrame.
    expected_tract_count : int
        Expected number of rows (one per tract).

    Raises
    ------
    ValueError
        If validation fails.
    """
    # Check row count matches tract count
    if len(df) != expected_tract_count:
        raise ValueError(
            f"Kriging output has {len(df)} rows, "
            f"expected {expected_tract_count} (one per tract)"
        )

    # Check no negative PM2.5 predictions
    negative = (df["pm25_kriged"] < 0).sum()
    if negative > 0:
        raise ValueError(
            f"{negative} tracts have negative PM2.5 predictions. "
            f"Kriging clip to 0 should have prevented this."
        )

    # Check for extreme values — above 500 μg/m³ is instrument error
    extreme = (df["pm25_kriged"] > 500).sum()
    if extreme > 0:
        raise ValueError(
            f"{extreme} tracts have PM2.5 > 500 μg/m³. "
            f"Check monitor readings for outliers."
        )

    # Check no null predictions
    null_preds = df["pm25_kriged"].isna().sum()
    if null_preds > 0:
        raise ValueError(
            f"{null_preds} tracts have null PM2.5 predictions."
        )

    # Warn if CV R² is low — predictions may be unreliable
    if "cv_r2" in df.columns:
        r2 = df["cv_r2"].iloc[0]
        if r2 < 0.5:
            logger.warning(
                f"LOO-CV R²={r2:.3f} is below 0.5. "
                f"Kriging predictions may be unreliable this week. "
                f"Check for monitor outages or unusual pollution events."
            )

    logger.info(
        f"Kriging validation passed: {len(df):,} tracts, "
        f"PM2.5 range {df['pm25_kriged'].min():.1f}–"
        f"{df['pm25_kriged'].max():.1f} μg/m³"
    )