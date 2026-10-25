"""
hospitals.py
============
Pull hospital locations, bed counts, and emergency department status
for the configured city.

These are the "supply" side of the healthcare access score in
src/spatial/healthcare_access.py (Enhanced 2SFCA): each hospital's
beds are shared among the population within its travel-time catchment.

Why not HIFLD:
    The original source, HIFLD Open (DHS), was shut down in 2025. Its
    hospital layer is now behind HIFLD Secure (login + approved use case)
    or in frozen archive copies that are no longer updated.

Sources (chosen per city in config: healthcare.hospital_source):
    "cdph_ca" — California Department of Public Health, Licensing and
                Certification: Licensed and Certified Healthcare Facility
                Listing (updated monthly, data.ca.gov). Facility locations
                give coordinates and licensed beds; the companion Services
                file gives the licensed emergency service level.

    Other states need their own source (e.g. Illinois IDPH for Chicago).
    Unknown sources stop with a clear error instead of loading wrong data.

Output:
    - GeoParquet: data/processed/hospitals/{fips}.parquet
    - Object store: hospitals/{fips}.parquet
    - Table: hospitals (one row per licensed general acute care hospital)
      hospital_id, name, address, city, zip, latitude, longitude, beds,
      ed_level, has_ed, trauma_level, data_date, source, city_fips

Usage:
    from src.ingestion.hospitals import pull_hospital_locations
    gdf = pull_hospital_locations(config)
"""

import io
from pathlib import Path
from typing import Any, Callable

import geopandas as gpd
import pandas as pd
import requests

from src.utils.bigquery import write_dataframe
from src.utils.gcs import upload_file
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants: California (CDPH)
# ---------------------------------------------------------------------------

# CKAN resource IDs on data.ca.gov. These stay fixed across monthly
# updates; the download URL behind each one changes, so we look it up.
CKAN_RESOURCE_API = "https://data.ca.gov/api/3/action/resource_show"
CDPH_LOCATIONS_RESOURCE = "3d2503d7-56ad-4f38-8435-3d86d27b7407"
CDPH_SERVICES_RESOURCE = "259d1c0e-a07f-4c60-af7d-fe89909e4da8"

# The download host rejects Python's default user agent (HTTP 403)
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (CHBIndex research pipeline)"}

GENERAL_ACUTE_CARE = "GENERAL ACUTE CARE HOSPITAL"

# California licenses emergency departments at three levels
ED_SERVICE_LEVELS = {
    "COMPREHENSIVE EMERGENCY MEDICAL SERVICES": "comprehensive",
    "BASIC EMERGENCY MEDICAL": "basic",
    "STANDBY EMERGENCY MEDICAL SERVICES": "standby",
}
ED_LEVEL_RANK = {"comprehensive": 0, "basic": 1, "standby": 2}


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def pull_hospital_locations(config: dict[str, Any]) -> gpd.GeoDataFrame:
    """
    Pull licensed general acute care hospitals for the configured city.

    Returns
    -------
    gpd.GeoDataFrame
        One row per hospital, EPSG:4326 points. See module docstring
        for columns.

    Examples
    --------
    >>> gdf = pull_hospital_locations(load_config("los_angeles"))
    >>> len(gdf), int(gdf["has_ed"].sum())
    (88, 70)
    """
    city = config["city_name"]
    full_fips = config["full_fips"]
    source = config["healthcare"].get("hospital_source")

    fetch = HOSPITAL_SOURCES.get(source)
    if fetch is None:
        raise NotImplementedError(
            f"No hospital source implemented for '{source}' ({city}). "
            f"Available: {sorted(HOSPITAL_SOURCES)}. Set "
            f"healthcare.hospital_source in the city config."
        )

    logger.info(f"Pulling hospital locations for {city} from source '{source}'")

    df = fetch(config)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    gdf["city_fips"] = full_fips

    gdf = _drop_outside_bbox(gdf, config)
    _validate_output(gdf, config)

    local_path = _save_local(gdf, full_fips)
    upload_file(local_path, config, gcs_path=f"hospitals/{full_fips}.parquet")

    write_dataframe(
        pd.DataFrame(gdf.drop(columns=["geometry"])),
        "hospitals",
        config,
        write_disposition="WRITE_TRUNCATE",  # full monthly snapshot
    )

    logger.info(
        f"Hospital ingestion complete: {len(gdf)} hospitals, "
        f"{int(gdf['has_ed'].sum())} with an ED, {int(gdf['beds'].sum()):,} beds"
    )
    return gdf


# ---------------------------------------------------------------------------
# Source: California CDPH
# ---------------------------------------------------------------------------

def _fetch_cdph_ca(config: dict[str, Any]) -> pd.DataFrame:
    """Licensed general acute care hospitals in the county, with ED level."""
    full_fips = config["full_fips"]

    facilities = _download_ckan_csv(CDPH_LOCATIONS_RESOURCE)
    services = _download_ckan_csv(CDPH_SERVICES_RESOURCE)
    logger.info(f"Downloaded {len(facilities):,} CDPH facilities, {len(services):,} service rows")

    df = facilities[
        (facilities["FIPS_COUNTY_CODE"] == full_fips)
        & (facilities["FAC_FDR"].str.upper() == GENERAL_ACUTE_CARE)
    ].copy()
    n_all = len(df)

    df["beds"] = pd.to_numeric(df["CAPACITY"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")

    # Keep one row per licensed hospital:
    # - beds > 0 drops certification-only duplicates (e.g. a second
    #   "Cedar Sinai" record with 0 beds next to the real 919-bed one)
    #   and consolidated campuses whose beds sit on the parent license
    # - active license and coordinates are required for the access model
    df = df[
        (df["beds"] > 0)
        & df["latitude"].notna()
        & df["longitude"].notna()
        & (df["LICENSE_STATUS_DESCRIPTION"].str.upper() == "ACTIVE")
    ]
    logger.info(f"Kept {len(df)} of {n_all} general acute care rows (beds > 0, active, located)")

    # Emergency department level from the Services file (on-site only)
    ed = services[
        services["SERVICE_TYPE_NAME"].isin(ED_SERVICE_LEVELS)
        & (services["IS_OFFSITE_SERVICE"].fillna("N").str.upper() != "Y")
    ].copy()
    ed["ed_level"] = ed["SERVICE_TYPE_NAME"].map(ED_SERVICE_LEVELS)
    ed = (
        ed.sort_values("ed_level", key=lambda s: s.map(ED_LEVEL_RANK))
        .drop_duplicates("FACID")[["FACID", "ed_level"]]
    )
    df = df.merge(ed, on="FACID", how="left")
    df["has_ed"] = df["ed_level"].notna()

    return pd.DataFrame({
        "hospital_id": df["FACID"],
        "name": df["FACNAME"].str.title(),
        "address": df["ADDRESS"],
        "city": df["CITY"].str.title(),
        "zip": df["ZIP"],
        "latitude": df["latitude"],
        "longitude": df["longitude"],
        "beds": df["beds"].astype(int),
        "ed_level": df["ed_level"],
        "has_ed": df["has_ed"],
        "trauma_level": df["TRAUMA_CTR"].fillna("None"),
        "data_date": df["DATA_DATE"],
        "source": "cdph_ca",
    }).sort_values("hospital_id").reset_index(drop=True)


def _download_ckan_csv(resource_id: str) -> pd.DataFrame:
    """Resolve a data.ca.gov resource to its current URL and read the CSV."""
    meta = requests.get(
        CKAN_RESOURCE_API, params={"id": resource_id}, headers=HTTP_HEADERS, timeout=60
    )
    meta.raise_for_status()
    url = meta.json()["result"]["url"]

    resp = requests.get(url, headers=HTTP_HEADERS, timeout=180)
    resp.raise_for_status()
    return pd.read_csv(io.BytesIO(resp.content), dtype=str, encoding="latin-1")


# Registry of implemented sources, keyed by healthcare.hospital_source
HOSPITAL_SOURCES: dict[str, Callable[[dict[str, Any]], pd.DataFrame]] = {
    "cdph_ca": _fetch_cdph_ca,
}


# ---------------------------------------------------------------------------
# Validation and output
# ---------------------------------------------------------------------------

def _validate_output(gdf: gpd.GeoDataFrame, config: dict[str, Any]) -> None:
    """Sanity-check hospitals before saving."""
    if gdf.empty:
        raise ValueError("No hospitals found for this city")

    if gdf["hospital_id"].duplicated().any():
        raise ValueError("Duplicate hospital_id values")

    expected = config["healthcare"]["er_count"]
    actual_ed = int(gdf["has_ed"].sum())
    if abs(actual_ed - expected) > 10:
        logger.warning(f"Expected ~{expected} hospitals with an ED, found {actual_ed}")

    logger.info(
        f"Validation passed: {len(gdf)} hospitals, {actual_ed} with an ED, "
        f"{int((gdf['trauma_level'] != 'None').sum())} trauma centers"
    )


def _drop_outside_bbox(gdf: gpd.GeoDataFrame, config: dict[str, Any]) -> gpd.GeoDataFrame:
    """
    Drop hospitals outside the city bounding box.

    The access model downloads the road network for the bbox only. A
    hospital outside it would snap to the nearest road inside, creating
    access that doesn't exist. In LA this is Catalina Island Medical
    Center (offshore, in the county but outside the mainland bbox).
    """
    bbox = config["bbox"]
    inside = (
        gdf["longitude"].between(bbox["west"], bbox["east"])
        & gdf["latitude"].between(bbox["south"], bbox["north"])
    )
    if (~inside).any():
        logger.warning(
            f"Dropping {int((~inside).sum())} hospital(s) outside the city bbox "
            f"(unreachable by the road network): {gdf.loc[~inside, 'name'].tolist()}"
        )
    return gdf[inside].reset_index(drop=True)


def _save_local(gdf: gpd.GeoDataFrame, full_fips: str) -> Path:
    """Save hospitals as GeoParquet."""
    output_dir = Path("data") / "processed" / "hospitals"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{full_fips}.parquet"
    gdf.to_parquet(output_path, index=False)
    logger.info(f"Saved hospital locations: {output_path}")
    return output_path
