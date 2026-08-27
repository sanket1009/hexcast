"""
hifld_hospitals.py
==================
Pull hospital and emergency department locations from the
Homeland Infrastructure Foundation-Level Data (HIFLD) dataset.

Hospital locations are the supply side of the 2SFCA healthcare
access model in src/spatial/healthcare_access.py.
Without hospital locations, travel time cannot be computed.

Data source:
    HIFLD Open Data: https://hifld-geoplatform.opendata.arcgis.com
    Dataset: Hospitals (licensed acute care facilities with EDs)
    Updated: quarterly by DHS
    License: Open Government — public domain

Output:
    GeoParquet: data/processed/hifld_hospitals/{fips}.parquet
    BigQuery table: hifld_hospitals (one row per hospital)

Usage:
    from src.ingestion.hifld_hospitals import pull_hospital_locations
    gdf = pull_hospital_locations(config)
"""

from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

from src.utils.bigquery import write_dataframe
from src.utils.gcs import upload_file
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# HIFLD hospitals dataset — ArcGIS REST API endpoint
# Returns hospital point locations with attributes
# Filter: STATUS=OPEN (exclude closed/proposed facilities)
# Filter: TYPE contains "GENERAL ACUTE CARE" (licensed EDs only)
HIFLD_URL = (
    "https://services1.arcgis.com/Hp6G80Pky0om7QvQ/arcgis/rest/services"
    "/Hospitals_1/FeatureServer/0/query"
)

# Fields to retrieve from HIFLD
# OBJECTID: unique identifier
# NAME: hospital name
# ADDRESS, CITY, STATE, ZIP: location
# LATITUDE, LONGITUDE: coordinates
# TYPE: hospital type (General Acute Care, Critical Access, etc.)
# STATUS: Open/Closed/Proposed
# TRAUMA: trauma center level (I, II, III, IV, Not Available)
# BEDS: licensed bed count
HIFLD_FIELDS = (
    "OBJECTID,NAME,ADDRESS,CITY,STATE,ZIP,"
    "LATITUDE,LONGITUDE,TYPE,STATUS,TRAUMA,BEDS"
)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def pull_hospital_locations(config: dict[str, Any]) -> gpd.GeoDataFrame:
    """
    Pull hospital and ER locations for the configured city.

    Fetches from HIFLD, filters to open acute care facilities
    within the city bounding box, validates coordinates, and
    saves as GeoParquet locally and in GCS.

    Parameters
    ----------
    config : dict
        Merged pipeline config. Uses: bbox, state_name,
        healthcare.er_count, full_fips

    Returns
    -------
    gpd.GeoDataFrame
        One row per hospital with columns:
        hospital_id, name, address, city, state, zip,
        latitude, longitude, type, trauma_level, beds,
        geometry, city_fips

    Examples
    --------
    >>> config = load_config("los_angeles")
    >>> gdf = pull_hospital_locations(config)
    >>> len(gdf)
    76
    """
    city = config["city_name"]
    full_fips = config["full_fips"]
    bbox = config["bbox"]

    logger.info(f"Pulling hospital locations for {city}")

    # Step 1 — Fetch from HIFLD API
    raw_df = _fetch_hifld_data(bbox, config)
    logger.info(f"Retrieved {len(raw_df)} hospitals from HIFLD")

    # Step 2 — Convert to GeoDataFrame
    gdf = _build_geodataframe(raw_df, full_fips)

    # Step 3 — Validate
    _validate_output(gdf, config)

    # Step 4 — Save locally as GeoParquet
    local_path = _save_local(gdf, full_fips)

    # Step 5 — Upload to GCS
    gcs_path = f"hospitals/{full_fips}.parquet"
    upload_file(local_path, config, gcs_path=gcs_path)

    # Step 6 — Write to BigQuery (tabular only)
    df_tabular = pd.DataFrame(gdf.drop(columns=["geometry"]))
    write_dataframe(
        df_tabular, "hifld_hospitals", config,
        write_disposition="WRITE_TRUNCATE"
    )

    logger.info(
        f"Hospital ingestion complete: {len(gdf)} facilities, "
        f"{(gdf['trauma_level'] != 'Not Available').sum()} trauma centers"
    )
    return gdf


# ---------------------------------------------------------------------------
# Private functions
# ---------------------------------------------------------------------------

def _fetch_hifld_data(
    bbox: dict,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Fetch hospital records from HIFLD ArcGIS REST API.

    Uses a geometry filter (bounding box) to get hospitals
    within the city's geographic extent. Also filters to
    open, acute care facilities only.

    Parameters
    ----------
    bbox : dict
        City bounding box with west/south/east/north keys.
    config : dict
        Pipeline config.

    Returns
    -------
    pd.DataFrame
        Raw hospital records from HIFLD.
    """
    # ArcGIS REST API geometry filter
    # Format: west,south,east,north
    geometry_filter = (
        f"{bbox['west']},{bbox['south']},"
        f"{bbox['east']},{bbox['north']}"
    )

    params = {
        # Filter: open hospitals only + acute care (have EDs)
        "where": "STATUS='OPEN' AND TYPE='GENERAL ACUTE CARE'",
        "geometry": geometry_filter,
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",           # input CRS: WGS84
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": HIFLD_FIELDS,
        "f": "json",              # return JSON format
        "resultRecordCount": 1000,  # max per request — enough for any county
    }

    logger.debug(f"HIFLD API request: bbox={geometry_filter}")

    response = _request_with_retry(HIFLD_URL, params)
    data = response.json()

    # HIFLD returns features in ArcGIS JSON format
    # Each feature has "attributes" dict with field values
    features = data.get("features", [])
    if not features:
        logger.warning("HIFLD returned no features for bounding box")
        return pd.DataFrame()

    # Extract attributes from each feature
    records = [f["attributes"] for f in features]
    df = pd.DataFrame(records)

    # Standardize column names to lowercase
    df.columns = [col.lower() for col in df.columns]

    # Rename to our standard names
    df = df.rename(columns={
        "objectid": "hospital_id",
        "trauma": "trauma_level",
    })

    return df


def _build_geodataframe(
    df: pd.DataFrame,
    full_fips: str,
) -> gpd.GeoDataFrame:
    """
    Convert hospital DataFrame to GeoDataFrame with Point geometries.

    Parameters
    ----------
    df : pd.DataFrame
        Raw hospital records from _fetch_hifld_data().
    full_fips : str
        County FIPS for city identifier column.

    Returns
    -------
    gpd.GeoDataFrame
        Hospital locations as spatial point features.
    """
    # Cast coordinate columns to float
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # Drop rows with missing coordinates
    # Cannot be used in 2SFCA without a location
    missing = df[df["latitude"].isna() | df["longitude"].isna()]
    if len(missing) > 0:
        logger.warning(f"Dropping {len(missing)} hospitals with missing coordinates")
        df = df.dropna(subset=["latitude", "longitude"])

    # Create Point geometry from lat/lon
    # Point(longitude, latitude) — note: x=lon, y=lat in GIS convention
    geometry = [
        Point(row["longitude"], row["latitude"])
        for _, row in df.iterrows()
    ]

    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Add city identifier
    gdf["city_fips"] = full_fips

    # Cast beds to integer — HIFLD sometimes returns as float
    if "beds" in gdf.columns:
        gdf["beds"] = pd.to_numeric(gdf["beds"], errors="coerce").fillna(0).astype(int)

    return gdf


def _validate_output(gdf: gpd.GeoDataFrame, config: dict[str, Any]) -> None:
    """Validate hospital GeoDataFrame before saving."""
    expected = config["healthcare"]["er_count"]
    actual = len(gdf)

    # Allow ±15 variance — HIFLD updates quarterly and
    # some facilities change status between updates
    if abs(actual - expected) > 15:
        logger.warning(
            f"Expected ~{expected} hospitals, got {actual}. "
            f"HIFLD may have updated since config was written."
        )

    # Check all geometries are valid Points
    null_geoms = gdf["geometry"].isna().sum()
    if null_geoms > 0:
        raise ValueError(f"{null_geoms} hospitals have null geometry")

    logger.info(
        f"Validation passed: {actual} hospitals, "
        f"{(gdf['trauma_level'] != 'Not Available').sum()} trauma centers"
    )


def _save_local(gdf: gpd.GeoDataFrame, full_fips: str) -> Path:
    """Save hospital GeoDataFrame as GeoParquet locally."""
    output_dir = Path("data") / "processed" / "hifld_hospitals"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{full_fips}.parquet"
    gdf.to_parquet(output_path, index=False)
    logger.info(f"Saved hospital locations: {output_path}")
    return output_path


def _request_with_retry(
    url: str,
    params: dict,
    max_retries: int = 3,
    delay_seconds: float = 2.0,
) -> requests.Response:
    """HTTP GET with retry logic — same pattern as other ingestion modules."""
    import time
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as e:
            if e.response.status_code < 500:
                raise
            logger.warning(f"HIFLD server error attempt {attempt}/{max_retries}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"HIFLD network error attempt {attempt}/{max_retries}: {e}")
        if attempt < max_retries:
            time.sleep(delay_seconds)
    raise requests.exceptions.RequestException(
        f"All {max_retries} HIFLD attempts failed"
    )