"""
lst_processing.py
=================
Extract Land Surface Temperature (LST) statistics per census tract
from Landsat 9 GeoTIFF rasters stored in GCS.

Converts raster pixel values to tract-level summary statistics
using zonal statistics — computing mean, max, and std of all
30m Landsat pixels that fall within each tract polygon.

Also computes:
  - LST z-score: how hot relative to city-wide distribution
  - Year-over-year anomaly: hotter or cooler than same week last year

Input:
    GCS: rasters/lst/{fips}_{week_start}.tif (from gee_lst.py)
    BigQuery: tract_geometries (tract polygons)

Output:
    BigQuery table: lst_tract_stats (one row per tract per week)
    Columns: geoid, lst_mean_celsius, lst_max_celsius, lst_std,
             lst_zscore, lst_yoy_anomaly, valid_pixel_count,
             week_start_date, city_fips

Usage:
    from src.spatial.lst_processing import process_lst_raster
    df = process_lst_raster(config)
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterstats

from src.utils.bigquery import read_query, write_dataframe
from src.utils.gcs import download_file
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Minimum valid pixels per tract
# Tracts with fewer valid pixels get NaN — too cloudy to compute
MIN_PIXEL_COUNT = 10


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def process_lst_raster(
    config: dict[str, Any],
    week_start: datetime | None = None,
) -> pd.DataFrame:
    """
    Compute per-tract LST statistics from Landsat GeoTIFF.

    Parameters
    ----------
    config : dict
        Merged pipeline config.
    week_start : datetime, optional
        Week to process. Defaults to most recent Monday.

    Returns
    -------
    pd.DataFrame
        One row per tract with LST statistics and z-scores.
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
        f"Processing LST raster for {config['city_name']} "
        f"week of {week_str}"
    )

    # Step 1 — Download GeoTIFF from GCS to local disk
    gcs_path = f"rasters/lst/{full_fips}_{week_str}.tif"
    local_path = download_file(gcs_path, config)
    logger.info(f"Downloaded LST raster: {local_path}")

    # Step 2 — Load tract geometries from BigQuery
    tracts_gdf = _load_tract_geometries(config)
    logger.info(f"Loaded {len(tracts_gdf):,} tract polygons")

    # Step 3 — Compute zonal statistics
    # For each tract polygon, summarize all pixels inside it
    zonal_df = _compute_zonal_stats(local_path, tracts_gdf, config)
    logger.info(f"Computed zonal statistics for {len(zonal_df):,} tracts")

    # Step 4 — Compute z-scores across city
    # Normalizes LST to standard deviations from city mean
    # Makes LST comparable to PM2.5 and access scores
    zonal_df = _compute_zscores(zonal_df)

    # Step 5 — Compute year-over-year anomaly
    # How much hotter/cooler than same week last year?
    zonal_df = _compute_yoy_anomaly(zonal_df, config, week_start)

    # Step 6 — Add metadata
    zonal_df["week_start_date"] = week_start.date()
    zonal_df["city_fips"] = full_fips

    # Step 7 — Validate and write to BigQuery
    _validate_output(zonal_df)
    write_dataframe(zonal_df, "lst_tract_stats", config)

    # Step 8 — Clean up local raster file
    # GeoTIFFs are large — don't keep them on disk after processing
    local_path.unlink(missing_ok=True)
    logger.info(f"Cleaned up local raster: {local_path}")

    logger.info(
        f"LST processing complete: {len(zonal_df):,} tracts, "
        f"city mean LST: {zonal_df['lst_mean_celsius'].mean():.1f}°C, "
        f"hottest tract: {zonal_df['lst_mean_celsius'].max():.1f}°C"
    )

    return zonal_df


# ---------------------------------------------------------------------------
# Private functions
# ---------------------------------------------------------------------------

def _load_tract_geometries(config: dict[str, Any]) -> gpd.GeoDataFrame:
    """
    Load tract polygon geometries from GCS GeoParquet.

    We need actual polygon geometries (not just centroids) for
    zonal statistics — rasterstats clips raster pixels to each
    polygon boundary to find which pixels fall inside.

    Returns
    -------
    gpd.GeoDataFrame
        Tract polygons in EPSG:4326 with geoid column.
    """
    from pathlib import Path
    full_fips = config["full_fips"]

    # Try local GeoParquet first — faster than re-downloading
    local_path = (
        Path("data") / "processed" / "tract_geometries" / f"{full_fips}.parquet"
    )

    if local_path.exists():
        logger.debug(f"Loading tract geometries from local: {local_path}")
        gdf = gpd.read_parquet(local_path)
    else:
        # Download from GCS
        logger.debug("Downloading tract geometries from GCS")
        from src.utils.gcs import download_file
        gcs_path = f"tract_geometries/{full_fips}.parquet"
        downloaded = download_file(gcs_path, config, local_path=local_path)
        gdf = gpd.read_parquet(downloaded)

    # Ensure correct CRS
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    cols = ["geoid", "aland", "geometry"] if "aland" in gdf.columns else ["geoid", "geometry"]
    return gdf[cols]


def _compute_zonal_stats(
    raster_path: Path,
    tracts_gdf: gpd.GeoDataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Compute zonal statistics for each tract polygon from LST raster.

    rasterstats.zonal_stats() iterates through each polygon,
    finds all raster pixels whose centers fall inside it,
    and computes summary statistics on those pixel values.

    For a typical LA County tract (~1.5 km²), this means
    averaging roughly 1,500 Landsat pixels (each 30x30m = 900m²).

    Parameters
    ----------
    raster_path : Path
        Local path to the Landsat LST GeoTIFF.
    tracts_gdf : gpd.GeoDataFrame
        Tract polygons in EPSG:4326.
    config : dict
        Pipeline config with zonal stats settings.

    Returns
    -------
    pd.DataFrame
        One row per tract with lst_mean_celsius, lst_max_celsius,
        lst_std, valid_pixel_count columns.
    """
    stats_to_compute = config["spatial"]["lst_zonal_stats"]
    min_pixel_count = config["spatial"]["lst_min_pixel_count"]

    logger.info(
        f"Computing zonal statistics ({stats_to_compute}) "
        f"for {len(tracts_gdf):,} tracts from {raster_path}"
    )

    # Verify raster CRS matches tract CRS
    # If they don't match, zonal stats will be misaligned
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs.to_epsg()
        logger.debug(
            f"Raster CRS: EPSG:{raster_crs}, "
            f"Tract CRS: EPSG:{tracts_gdf.crs.to_epsg()}"
        )
        if raster_crs != 4326:
            raise ValueError(
                f"Raster CRS is EPSG:{raster_crs}, expected EPSG:4326. "
                f"Check GEE export CRS setting."
            )

    # Compute zonal statistics
    # all_touched=False: only pixels whose CENTER falls inside polygon
    # all_touched=True would include edge pixels — less accurate
    # nodata: pixel value representing missing data (from QA masking)
    stats = rasterstats.zonal_stats(
        tracts_gdf,
        str(raster_path),
        stats=stats_to_compute,
        all_touched=False,
        nodata=float("nan"),
        add_stats={"count_valid": _count_valid_pixels},
    )

    # Convert list of dicts to DataFrame
    stats_df = pd.DataFrame(stats)

    # Rename columns to our standard names
    rename_map = {
        "mean": "lst_mean_celsius",
        "max": "lst_max_celsius",
        "std": "lst_std",
        "count_valid": "valid_pixel_count",
    }
    stats_df = stats_df.rename(columns=rename_map)

    # Add geoid from tracts
    stats_df["geoid"] = tracts_gdf["geoid"].values

    # Mark tracts with too few valid pixels as NaN
    # These tracts had heavy cloud cover — no reliable LST reading
    low_pixel_mask = stats_df["valid_pixel_count"] < min_pixel_count
    low_pixel_count = low_pixel_mask.sum()

    if low_pixel_count > 0:
        logger.warning(
            f"{low_pixel_count} tracts have fewer than {min_pixel_count} "
            f"valid pixels — setting LST to NaN (no clear land pixels: "
            f"cloud, water, or outside the raster bbox)"
        )
        for col in ["lst_mean_celsius", "lst_max_celsius", "lst_std"]:
            if col in stats_df.columns:
                stats_df.loc[low_pixel_mask, col] = np.nan

    # Water-only tracts (Census land area 0) have no land surface to
    # measure. The satellite water mask leaves a few shoreline pixels in
    # them (~100 of 200k at 19 °C in the June 22, 2026 run), which would
    # otherwise skew the city z-score. NaN them on geography, not pixels.
    if "aland" in tracts_gdf.columns:
        water_only = (tracts_gdf["aland"].fillna(0).values == 0)
        if water_only.any():
            logger.info(
                f"{water_only.sum()} water-only tracts (land area 0) set to NaN: "
                + ", ".join(stats_df.loc[water_only, "geoid"])
            )
            for col in ["lst_mean_celsius", "lst_max_celsius", "lst_std"]:
                if col in stats_df.columns:
                    stats_df.loc[water_only, col] = np.nan

    return stats_df


def _count_valid_pixels(masked_array) -> int:
    """
    Custom rasterstats statistic: count non-masked pixels.

    Used as add_stats parameter to count how many valid
    (non-cloudy) pixels exist per tract.

    Parameters
    ----------
    masked_array : numpy.ma.MaskedArray
        Pixel values for this tract (masked where NoData).

    Returns
    -------
    int
        Number of valid (unmasked) pixels.
    """
    return int((~masked_array.mask).sum())


def _compute_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute city-wide z-score of mean LST per tract.

    Z-score = (tract_lst - city_mean_lst) / city_std_lst

    This normalizes LST to a dimensionless scale where:
      0 = city average temperature
     +1 = one standard deviation above average (hotter)
     -1 = one standard deviation below average (cooler)
     +2.5 = extremely hot tract (urban heat island)

    Z-scores make LST directly comparable to PM2.5 percentile
    and healthcare access score in the compound index.

    Only computed for tracts with valid LST readings.
    Tracts with NaN LST get NaN z-score.

    Parameters
    ----------
    df : pd.DataFrame
        Tract LST statistics from _compute_zonal_stats().

    Returns
    -------
    pd.DataFrame
        Same DataFrame with lst_zscore column added.
    """
    valid_mask = df["lst_mean_celsius"].notna()
    valid_lst = df.loc[valid_mask, "lst_mean_celsius"]

    city_mean = valid_lst.mean()
    city_std = valid_lst.std()

    logger.info(
        f"City LST statistics: mean={city_mean:.1f}°C, "
        f"std={city_std:.1f}°C, "
        f"range={valid_lst.min():.1f}–{valid_lst.max():.1f}°C"
    )

    # Compute z-score only for valid tracts
    df["lst_zscore"] = np.nan
    df.loc[valid_mask, "lst_zscore"] = (
        (valid_lst - city_mean) / city_std
    ).round(4)

    # Also compute percentile rank
    df["lst_percentile"] = (
        df["lst_mean_celsius"]
        .rank(pct=True)
        .mul(100)
        .round(1)
    )

    return df


def _compute_yoy_anomaly(
    df: pd.DataFrame,
    config: dict[str, Any],
    week_start: datetime,
) -> pd.DataFrame:
    """
    Compute year-over-year LST anomaly per tract.

    Anomaly = this_week_lst - same_week_last_year_lst

    Positive anomaly: hotter than same week last year.
    Negative anomaly: cooler than same week last year.

    This captures unusual heat events — a tract that's always
    hot is less alarming than one that's suddenly much hotter
    than historical baseline. The anomaly captures the "surprise"
    in the heat signal.

    If last year's data doesn't exist in BigQuery yet
    (first run, or data gap), returns NaN for all tracts.

    Parameters
    ----------
    df : pd.DataFrame
        Current week LST statistics.
    config : dict
        Pipeline config.
    week_start : datetime
        Current week start date.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with lst_yoy_anomaly column added.
    """
    # Same week last year
    last_year_week = week_start - timedelta(weeks=52)
    last_year_str = last_year_week.strftime("%Y-%m-%d")

    logger.debug(
        f"Looking up LST for same week last year: {last_year_str}"
    )

    try:
        # Query last year's LST for same city and week
        last_year_df = read_query(
            f"""
            SELECT geoid, lst_mean_celsius as lst_last_year
            FROM {{dataset}}.lst_tract_stats
            WHERE
                week_start_date = '{last_year_str}'
                AND city_fips = '{config["full_fips"]}'
            """,
            config,
        )

        if last_year_df.empty:
            logger.warning(
                f"No LST data for same week last year ({last_year_str}). "
                f"Setting YoY anomaly to NaN."
            )
            df["lst_yoy_anomaly"] = np.nan
            return df

        # Merge current and last year on geoid
        df = df.merge(last_year_df, on="geoid", how="left")

        # Compute anomaly
        df["lst_yoy_anomaly"] = (
            df["lst_mean_celsius"] - df["lst_last_year"]
        ).round(2)

        # Drop the intermediate column
        df = df.drop(columns=["lst_last_year"])

        anomaly_mean = df["lst_yoy_anomaly"].mean()
        logger.info(
            f"YoY LST anomaly: mean={anomaly_mean:+.1f}°C "
            f"({'warmer' if anomaly_mean > 0 else 'cooler'} than last year)"
        )

    except Exception as e:
        # First pipeline run — no historical data exists yet
        logger.warning(f"Could not compute YoY anomaly: {e}. Setting to NaN.")
        df["lst_yoy_anomaly"] = np.nan

    return df


def _validate_output(df: pd.DataFrame) -> None:
    """Validate LST output before BigQuery write."""
    if df.empty:
        raise ValueError("LST output DataFrame is empty")

    # Check required columns exist
    required = ["geoid", "lst_mean_celsius", "lst_zscore", "week_start_date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Check LST values in physically reasonable range
    # Earth's surface LST ranges from ~-60°C to ~70°C
    valid = df["lst_mean_celsius"].dropna()
    if len(valid) > 0:
        if valid.min() < -60 or valid.max() > 70:
            raise ValueError(
                f"LST values out of physical range: "
                f"{valid.min():.1f}–{valid.max():.1f}°C"
            )

    # Check NaN rate — more than 30% NaN suggests cloud cover problem
    nan_rate = df["lst_mean_celsius"].isna().mean()
    if nan_rate > 0.30:
        logger.warning(
            f"{nan_rate:.1%} of tracts have NaN LST — "
            f"heavy cloud cover this week. "
            f"Consider using 30-day composite fallback."
        )

    logger.info(
        f"LST validation passed: {len(df):,} tracts, "
        f"{df['lst_mean_celsius'].notna().sum():,} with valid readings, "
        f"mean z-score={df['lst_zscore'].mean():.3f}"
    )