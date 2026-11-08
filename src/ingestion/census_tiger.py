"""
census_tiger.py
===============
Pull census tract geometries for a configured city from the
US Census Bureau TIGER API.

Census tract polygons are the spatial foundation of the entire
CHBIndex pipeline. Every other data source joins to these geometries
via the GEOID (11-digit tract identifier).

Output:
    - GeoParquet file saved locally: data/processed/tract_geometries/{fips}.parquet
    - Same file uploaded to GCS: tract_geometries/{fips}.parquet
    - BigQuery table: tract_geometries (one row per tract)

Usage:
    from src.ingestion.census_tiger import pull_tract_geometries
    gdf = pull_tract_geometries(config)
"""

import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import shape

from src.utils.bigquery import write_dataframe
from src.utils.gcs import upload_file
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Census Bureau TIGER GeoJSON API endpoint
# Returns tract geometries as GeoJSON for any state/county combination
# Documentation: https://tigerweb.geo.census.gov/arcgis/rest/services
TIGER_API_BASE = (
    "https://tigerweb.geo.census.gov/arcgis/rest/services"
    "/TIGERweb/tigerWMS_Census2020/MapServer/6/query"  # layer 6 = Census Tracts
)

# Census ACS API endpoint for demographic data
# We pull population per tract to weight spatial lag features.
# 2024 5-year is the latest release (2020-2024) and uses the same
# 2020 tract boundaries as TIGER, so GEOIDs join one-to-one.
# Requires a free API key: https://api.census.gov/data/key_signup.html
ACS_API_BASE = "https://api.census.gov/data/2024/acs/acs5"


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def pull_tract_geometries(config: dict[str, Any]) -> gpd.GeoDataFrame:
    """
    Pull census tract geometries for the configured city.

    Fetches tract boundary polygons from the Census TIGER API,
    enriches them with ACS demographic data, saves as GeoParquet
    locally and in GCS, and writes a summary table to BigQuery.

    Parameters
    ----------
    config : dict
        Merged pipeline config from config_loader.load_config().
        Uses: full_fips, state_fips, county_fips, crs, bbox

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with one row per census tract containing:
        - geoid: 11-digit tract identifier (primary key)
        - geometry: tract polygon in EPSG:4326
        - name: tract name
        - aland: land area in square meters
        - population: total population (from ACS 2024 5-year)
        - median_income: median household income
        - tract_area_km2: computed land area in km²

    Examples
    --------
    >>> from src.utils.config_loader import load_config
    >>> config = load_config("los_angeles")
    >>> gdf = pull_tract_geometries(config)
    >>> len(gdf)
    2498
    >>> gdf.columns.tolist()
    ['geoid', 'name', 'aland', 'population', 'median_income',
     'tract_area_km2', 'geometry']
    """
    city = config["city_name"]
    state_fips = config["state_fips"]
    county_fips = config["county_fips"]
    full_fips = config["full_fips"]

    logger.info(f"Pulling census tract geometries for {city} (FIPS: {full_fips})")

    # Step 1 — Pull tract geometries from TIGER API
    gdf = _fetch_tiger_geometries(state_fips, county_fips)
    logger.info(f"Retrieved {len(gdf):,} census tracts from TIGER API")

    # Step 2 — Pull demographic data from ACS API
    demographics = _fetch_acs_demographics(
        state_fips, county_fips, config["secrets"].get("CENSUS_API_KEY", "")
    )
    logger.info(f"Retrieved demographics for {len(demographics):,} tracts from ACS API")

    # Step 3 — Join demographics to geometries
    gdf = gdf.merge(demographics, on="geoid", how="left")

    # Step 4 — Compute derived fields
    gdf = _compute_derived_fields(gdf, config)

    # Step 5 — Validate output
    _validate_output(gdf, full_fips)

    # Step 6 — Save locally as GeoParquet
    local_path = _save_local(gdf, full_fips)

    # Step 7 — Upload to GCS
    gcs_path = f"tract_geometries/{full_fips}.parquet"
    upload_file(local_path, config, gcs_path=gcs_path)

    # Step 8 — Write summary to BigQuery (without geometry — BQ handles separately)
    _write_to_bigquery(gdf, config)

    logger.info(
        f"Census tract ingestion complete: {len(gdf):,} tracts, "
        f"total population {gdf['population'].sum():,.0f}"
    )

    return gdf


def pull_buffer_tracts(config: dict[str, Any]) -> pd.DataFrame:
    """
    Pull population and centroids for neighbor-county tracts near the city.

    These are demand only for the healthcare access model: a hospital just
    outside the county also serves the people on its side of the line, so
    leaving them out would make that hospital look emptier than it is.
    They are not scored and stay out of tract_geometries, so every other
    module still sees city tracts only.

    Keeps tracts in config neighbor_counties whose centroid falls inside
    the bbox grown by healthcare.buffer_km.

    Output table: buffer_tracts
        geoid, county_fips, population, centroid_lat, centroid_lon, city_fips
    """
    from src.utils.config_loader import get_buffered_bbox

    neighbors = config.get("neighbor_counties", [])
    buffer_km = config["healthcare"].get("buffer_km", 0.0)
    if not neighbors or buffer_km <= 0:
        logger.info("No neighbor counties or buffer configured; skipping buffer tracts")
        return pd.DataFrame()

    west, south, east, north = get_buffered_bbox(config, buffer_km)
    api_key = config["secrets"].get("CENSUS_API_KEY", "")
    parts = []

    for fips in neighbors:
        state_fips, county_fips = fips[:2], fips[2:]
        gdf = _fetch_tiger_geometries(state_fips, county_fips)
        demo = _fetch_acs_demographics(state_fips, county_fips, api_key)
        gdf = gdf.merge(demo[["geoid", "population"]], on="geoid", how="left")

        cent = gdf.to_crs(config["crs_projected"]).geometry.centroid.to_crs("EPSG:4326")
        gdf["centroid_lon"], gdf["centroid_lat"] = cent.x, cent.y

        inside = gdf["centroid_lon"].between(west, east) & gdf["centroid_lat"].between(
            south, north
        )
        kept = gdf[inside]
        logger.info(
            f"Buffer tracts {fips}: kept {len(kept)} of {len(gdf)}, "
            f"population {kept['population'].sum():,.0f}"
        )
        parts.append(
            pd.DataFrame(
                {
                    "geoid": kept["geoid"],
                    "county_fips": fips,
                    "population": kept["population"],
                    "centroid_lat": kept["centroid_lat"],
                    "centroid_lon": kept["centroid_lon"],
                }
            )
        )

    df = pd.concat(parts, ignore_index=True)
    df["city_fips"] = config["full_fips"]

    write_dataframe(df, "buffer_tracts", config, write_disposition="WRITE_TRUNCATE")
    logger.info(
        f"Buffer tracts complete: {len(df):,} tracts, "
        f"population {df['population'].sum():,.0f} within {buffer_km} km"
    )
    return df


# ---------------------------------------------------------------------------
# Private functions
# ---------------------------------------------------------------------------


def _fetch_tiger_geometries(
    state_fips: str,
    county_fips: str,
) -> gpd.GeoDataFrame:
    """
    Fetch tract boundary polygons from the Census TIGER API.

    The TIGER API returns GeoJSON with tract geometries.
    We convert to a GeoDataFrame and standardize column names.

    Parameters
    ----------
    state_fips : str
        2-digit state FIPS code. Example: "06" for California.
    county_fips : str
        3-digit county FIPS code. Example: "037" for LA County.

    Returns
    -------
    gpd.GeoDataFrame
        Tract geometries in EPSG:4326 with columns:
        geoid, name, aland, geometry
    """
    logger.debug(
        f"Fetching TIGER geometries for state {state_fips} county {county_fips}"
    )

    # TIGER API query parameters
    # WHERE clause filters to the specific county using FIPS codes
    # outFields=* returns all available attribute fields
    # f=geojson returns GeoJSON format (easier to parse than the default)
    params = {
        # Field names on this layer are STATE/COUNTY (not STATEFP/COUNTYFP)
        "where": f"STATE='{state_fips}' AND COUNTY='{county_fips}'",
        "outFields": "GEOID,NAME,AREALAND",
        "f": "geojson",
        "returnGeometry": "true",
        "outSR": "4326",  # return coordinates in WGS84 (EPSG:4326)
        "resultRecordCount": 5000,  # max records per request — enough for any county
    }

    logger.debug(f"TIGER API params: {params}")

    # Make the API request with retry logic
    response = _request_with_retry(TIGER_API_BASE, params)
    geojson = response.json()

    # Parse GeoJSON features into a GeoDataFrame
    features = geojson.get("features", [])
    if not features:
        raise ValueError(
            f"TIGER API returned no features for state {state_fips} "
            f"county {county_fips}. Check FIPS codes."
        )

    # Convert GeoJSON features to GeoDataFrame
    geometries = [shape(f["geometry"]) for f in features]
    properties = [f["properties"] for f in features]

    gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs="EPSG:4326")

    # Standardize column names to lowercase
    gdf.columns = [col.lower() for col in gdf.columns]

    # Rename columns to our standard names
    gdf = gdf.rename(
        columns={
            "geoid": "geoid",
            "name": "name",
            "arealand": "aland",  # land area in square meters
        }
    )

    # Keep only the columns we need
    gdf = gdf[["geoid", "name", "aland", "geometry"]]

    logger.debug(f"TIGER API returned {len(gdf)} features")

    return gdf


def _fetch_acs_demographics(
    state_fips: str,
    county_fips: str,
    api_key: str,
) -> pd.DataFrame:
    """
    Fetch demographic data from the Census ACS 5-year API.

    Variables pulled:
        B01003_001E — total population
        B19013_001E — median household income

    These are used as:
        - population: for population-weighted spatial lag features
        - median_income: for equity audit stratification

    Parameters
    ----------
    state_fips : str
        2-digit state FIPS code.
    county_fips : str
        3-digit county FIPS code.
    api_key : str
        Census Data API key (CENSUS_API_KEY in .env).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: geoid, population, median_income
    """
    if not api_key:
        raise ValueError(
            "CENSUS_API_KEY is not set. Get a free key at "
            "https://api.census.gov/data/key_signup.html and add it to .env"
        )
    logger.debug("Fetching ACS 5-year demographic data")

    params = {
        "get": "B01003_001E,B19013_001E,NAME",
        "for": "tract:*",
        "in": f"state:{state_fips} county:{county_fips}",
        "key": api_key,
    }

    response = _request_with_retry(ACS_API_BASE, params)

    # A bad or missing key redirects to an HTML page with status 200,
    # which would otherwise fail later with a confusing JSON error
    if "missing_key" in response.url or "invalid_key" in response.url:
        raise ValueError(
            "Census API rejected the key. Check CENSUS_API_KEY in .env "
            "(new keys must be activated from the confirmation email)."
        )
    data = response.json()

    # ACS API returns a list where first row is column headers
    # Subsequent rows are data
    headers = data[0]
    rows = data[1:]

    df = pd.DataFrame(rows, columns=headers)

    # Construct 11-digit GEOID from state + county + tract components
    # This matches the GEOID format from TIGER API
    df["geoid"] = df["state"] + df["county"] + df["tract"]

    # Rename and cast columns
    df = df.rename(
        columns={
            "B01003_001E": "population",
            "B19013_001E": "median_income",
        }
    )

    # Cast to numeric — ACS API returns strings
    # Negative values (-666666666) indicate missing data — replace with NaN
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df["median_income"] = pd.to_numeric(df["median_income"], errors="coerce")
    df["median_income"] = df["median_income"].where(df["median_income"] > 0)

    # Keep only needed columns
    df = df[["geoid", "population", "median_income"]]

    logger.debug(f"ACS API returned data for {len(df)} tracts")

    return df


def _compute_derived_fields(
    gdf: gpd.GeoDataFrame,
    config: dict[str, Any],
) -> gpd.GeoDataFrame:
    """
    Compute derived geographic fields from raw tract data.

    Adds:
        tract_area_km2: land area in square kilometers
        centroid_lat: tract centroid latitude (for Kriging)
        centroid_lon: tract centroid longitude (for Kriging)
        city_fips: the county FIPS code (for BigQuery partitioning)
        state_fips: the state FIPS code

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Raw tract GeoDataFrame from TIGER API.
    config : dict
        Merged pipeline config.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with additional computed columns.
    """
    logger.debug("Computing derived geographic fields")

    # Convert land area from square meters to square kilometers
    # aland is in m² — divide by 1,000,000 to get km²
    gdf["tract_area_km2"] = gdf["aland"] / 1_000_000

    # Compute tract centroids for point-based operations
    # Kriging predicts PM2.5 at centroid locations
    # 2SFCA uses centroids as origin points for travel time calculation
    # We reproject to the UTM CRS for accurate centroid calculation
    # then extract coordinates in WGS84
    crs_projected = config["crs_projected"]

    gdf_projected = gdf.to_crs(crs_projected)
    centroids_projected = gdf_projected.geometry.centroid
    centroids_wgs84 = centroids_projected.to_crs("EPSG:4326")

    gdf["centroid_lon"] = centroids_wgs84.x
    gdf["centroid_lat"] = centroids_wgs84.y

    # Add city identifiers for BigQuery filtering
    gdf["city_fips"] = config["full_fips"]
    gdf["state_fips"] = config["state_fips"]

    logger.debug(
        f"Computed centroids for {len(gdf)} tracts. "
        f"Lat range: {gdf['centroid_lat'].min():.3f} to {gdf['centroid_lat'].max():.3f}"
    )

    return gdf


def _validate_output(
    gdf: gpd.GeoDataFrame,
    full_fips: str,
) -> None:
    """
    Validate the output GeoDataFrame before saving.

    Checks that the data looks geographically correct and
    that required columns are present and non-null.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Tract GeoDataFrame to validate.
    full_fips : str
        Expected county FIPS for error messages.

    Raises
    ------
    ValueError
        If any validation check fails.
    """
    logger.debug("Validating tract geometry output")

    # Check minimum tract count
    # LA County has ~2,310 tracts, Cook County has ~1,319
    # Any county should have at least 100 tracts
    if len(gdf) < 100:
        raise ValueError(
            f"Only {len(gdf)} tracts returned for FIPS {full_fips}. "
            f"Expected at least 100. Check FIPS codes."
        )

    # Check required columns exist
    required_cols = ["geoid", "geometry", "centroid_lat", "centroid_lon"]
    missing = [col for col in required_cols if col not in gdf.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Check no null geometries
    null_geoms = gdf["geometry"].isna().sum()
    if null_geoms > 0:
        raise ValueError(f"{null_geoms} tracts have null geometries")

    # Check CRS is correct
    if gdf.crs.to_epsg() != 4326:
        raise ValueError(f"GeoDataFrame CRS is {gdf.crs} — expected EPSG:4326")

    # Check GEOIDs are 11 digits
    invalid_geoids = gdf[gdf["geoid"].str.len() != 11]
    if len(invalid_geoids) > 0:
        raise ValueError(
            f"{len(invalid_geoids)} tracts have invalid GEOIDs (expected 11 digits)"
        )

    logger.info(
        f"Validation passed: {len(gdf):,} tracts, "
        f"all geometries valid, all GEOIDs correctly formatted"
    )


def _save_local(
    gdf: gpd.GeoDataFrame,
    full_fips: str,
) -> Path:
    """
    Save the GeoDataFrame as a GeoParquet file locally.

    GeoParquet is the cloud-native format for vector spatial data.
    Much faster to read than Shapefile and preserves CRS information.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Tract GeoDataFrame to save.
    full_fips : str
        County FIPS used in the filename.

    Returns
    -------
    Path
        Local path where the file was saved.
    """
    output_dir = Path("data") / "processed" / "tract_geometries"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{full_fips}.parquet"

    logger.info(f"Saving tract geometries to {output_path}")
    gdf.to_parquet(output_path, index=False)
    logger.info(f"Saved {output_path.stat().st_size / 1024:.1f} KB")

    return output_path


def _write_to_bigquery(
    gdf: gpd.GeoDataFrame,
    config: dict[str, Any],
) -> None:
    """
    Write tract summary data to BigQuery.

    We write the tabular columns (not geometry) to BigQuery
    for use in SQL joins with other pipeline outputs.
    Geometry is stored in GCS as GeoParquet.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Full tract GeoDataFrame.
    config : dict
        Merged pipeline config.
    """
    # Select only tabular columns — BigQuery doesn't store geometry natively
    # in the same way as PostGIS. We keep geometry in GeoParquet on GCS.
    tabular_cols = [
        "geoid",
        "name",
        "aland",
        "population",
        "median_income",
        "tract_area_km2",
        "centroid_lat",
        "centroid_lon",
        "city_fips",
        "state_fips",
    ]

    df_tabular = pd.DataFrame(gdf[tabular_cols])

    write_dataframe(
        df_tabular,
        "tract_geometries",
        config,
        write_disposition="WRITE_TRUNCATE",  # replace — geometries don't change
    )


def _request_with_retry(
    url: str,
    params: dict,
    max_retries: int = 3,
    delay_seconds: float = 2.0,
) -> requests.Response:
    """
    Make an HTTP GET request with retry logic.

    Retries on network errors and 5xx server errors.
    Does not retry on 4xx client errors (bad parameters).

    Parameters
    ----------
    url : str
        API endpoint URL.
    params : dict
        Query parameters.
    max_retries : int
        Maximum number of attempts. Default 3.
    delay_seconds : float
        Seconds to wait between retries. Default 2.0.

    Returns
    -------
    requests.Response
        Successful response object.

    Raises
    ------
    requests.exceptions.RequestException
        If all retry attempts fail.
    """
    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(f"API request attempt {attempt}/{max_retries}: {url}")

            response = requests.get(url, params=params, timeout=30)

            # Raise exception for 4xx and 5xx status codes
            response.raise_for_status()

            return response

        except requests.exceptions.HTTPError as e:
            # 4xx errors — bad request parameters, don't retry
            if e.response.status_code < 500:
                logger.error(f"Client error {e.response.status_code}: {e}")
                raise

            # 5xx errors — server problem, retry
            logger.warning(
                f"Server error {e.response.status_code} on attempt "
                f"{attempt}/{max_retries}. Retrying in {delay_seconds}s..."
            )

        except requests.exceptions.RequestException as e:
            # Network errors — retry
            logger.warning(
                f"Network error on attempt {attempt}/{max_retries}: {e}. "
                f"Retrying in {delay_seconds}s..."
            )

        if attempt < max_retries:
            time.sleep(delay_seconds)

    raise requests.exceptions.RequestException(
        f"All {max_retries} attempts failed for {url}"
    )
