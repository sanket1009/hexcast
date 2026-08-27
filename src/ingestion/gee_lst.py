"""
gee_lst.py
==========
Pull Landsat 9 Land Surface Temperature (LST) rasters from
Google Earth Engine for the configured city and week.

LST is the heat signal in the compound burden index — it captures
urban heat islands at 30m resolution across the entire county,
far more spatially detailed than any weather station network.

The output GeoTIFF is read by src/spatial/lst_processing.py which
computes mean LST per census tract polygon.

Data source:
    Google Earth Engine: LANDSAT/LC09/C02/T1_L2
    Band: ST_B10 (thermal infrared → land surface temperature)
    Resolution: 30 meters per pixel
    Revisit: every 16 days per satellite

Output:
    GeoTIFF saved to GCS: rasters/lst/{fips}_{week_start}.tif
    - Single band: LST in Celsius
    - CRS: EPSG:4326
    - Clipped to city bounding box

Authentication:
    Requires Google Earth Engine account registered at:
    https://signup.earthengine.google.com
    For development: run `earthengine authenticate` in terminal
    For production: use service account credentials in .env

Usage:
    from src.ingestion.gee_lst import pull_lst_raster
    gcs_uri = pull_lst_raster(config)
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from src.utils.gcs import blob_exists, get_gcs_uri
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Landsat 9 Collection 2 Tier 1 Level-2 surface temperature product
# This is the calibrated, atmospherically corrected product
# NOT the raw radiance — Level-2 gives us actual surface temperature
LANDSAT9_COLLECTION = "LANDSAT/LC09/C02/T1_L2"

# Thermal infrared band containing land surface temperature
# ST_B10 = Surface Temperature Band 10
# Raw values are in Kelvin × 0.00341802 + 149.0 (scale factor)
# We convert to Celsius in the GEE processing script
LST_BAND = "ST_B10"

# QA_PIXEL band bit flags for cloud masking
# Bit 3 = cloud shadow
# Bit 4 = cloud
# Pixels where these bits are set get masked out
QA_BAND = "QA_PIXEL"

# Scale factor and offset for converting raw DN to Kelvin
# From Landsat Collection 2 documentation
LST_SCALE = 0.00341802
LST_OFFSET = 149.0
KELVIN_TO_CELSIUS = -273.15


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def pull_lst_raster(
    config: dict[str, Any],
    week_start: datetime | None = None,
) -> str:
    """
    Pull Landsat 9 LST raster for the configured city and week.

    Connects to Google Earth Engine, filters Landsat 9 scenes to the
    city bounding box and week date range, applies cloud masking,
    computes mean LST composite, converts to Celsius, and exports
    to GCS as a GeoTIFF.

    GEE exports are asynchronous — this function submits the export
    task and polls until it completes or times out.

    Parameters
    ----------
    config : dict
        Merged pipeline config from config_loader.load_config().
        Uses: gee.collection, gee.scale, gee.cloud_cover_max,
              gee.lst_band, bbox, crs, gcp.gcs_bucket,
              secrets.GEE_SERVICE_ACCOUNT, secrets.GEE_PRIVATE_KEY_PATH

    week_start : datetime, optional
        Start of the week to pull LST for.
        If None, uses the most recent Monday.

    Returns
    -------
    str
        GCS URI of the exported GeoTIFF.
        Format: gs://chbindex-06037-data/rasters/lst/06037_2026-08-18.tif

    Raises
    ------
    RuntimeError
        If GEE export task fails or times out.
    ImportError
        If earthengine-api is not installed.

    Examples
    --------
    >>> config = load_config("los_angeles")
    >>> gcs_uri = pull_lst_raster(config)
    >>> gcs_uri
    'gs://chbindex-06037-data/rasters/lst/06037_2026-08-18.tif'
    """
    # Import GEE here not at top level
    # GEE authentication happens at import time in some versions
    # Keeping it local prevents auth errors on modules that don't need GEE
    try:
        import ee
    except ImportError:
        raise ImportError(
            "earthengine-api is not installed. "
            "Run: pip install earthengine-api"
        )

    # Determine week
    if week_start is None:
        week_start = _get_most_recent_monday()

    week_end = week_start + timedelta(days=6)
    full_fips = config["full_fips"]

    # GCS destination path
    week_str = week_start.strftime("%Y-%m-%d")
    gcs_filename = f"{full_fips}_{week_str}.tif"
    gcs_path = f"rasters/lst/{gcs_filename}"

    logger.info(
        f"Pulling LST raster for {config['city_name']} "
        f"week of {week_str}"
    )

    # Check if this week's raster already exists in GCS
    # Idempotency: skip GEE export if raster already exists
    # This prevents duplicate exports on pipeline reruns
    if blob_exists(gcs_path, config):
        gcs_uri = get_gcs_uri(gcs_path, config)
        logger.info(f"LST raster already exists in GCS: {gcs_uri} — skipping export")
        return gcs_uri

    # Step 1 — Authenticate to Google Earth Engine
    _authenticate_gee(config)

    # Step 2 — Build the LST image composite
    lst_image = _build_lst_composite(
        config=config,
        start_date=week_start,
        end_date=week_end,
    )

    # Step 3 — Export to GCS
    gcs_uri = _export_to_gcs(
        image=lst_image,
        gcs_path=gcs_path,
        config=config,
        ee=ee,
    )

    logger.info(f"LST raster export complete: {gcs_uri}")

    return gcs_uri


# ---------------------------------------------------------------------------
# Private functions
# ---------------------------------------------------------------------------

def _get_most_recent_monday() -> datetime:
    """Return the most recent Monday at midnight."""
    today = datetime.now()
    days_since_monday = today.weekday()
    monday = today - timedelta(days=days_since_monday)
    return monday.replace(hour=0, minute=0, second=0, microsecond=0)


def _authenticate_gee(config: dict[str, Any]) -> None:
    """
    Authenticate to Google Earth Engine.

    Tries service account authentication first (production).
    Falls back to interactive browser authentication (development).

    Service account auth uses:
        GEE_SERVICE_ACCOUNT: service account email
        GEE_PRIVATE_KEY_PATH: path to JSON key file

    Interactive auth uses credentials cached by:
        earthengine authenticate  (run once in terminal)

    Parameters
    ----------
    config : dict
        Pipeline config containing GEE credentials.

    Raises
    ------
    Exception
        If authentication fails with both methods.
    """
    import ee

    sa_email = config["secrets"].get("GEE_SERVICE_ACCOUNT", "")
    sa_key_path = config["secrets"].get("GEE_PRIVATE_KEY_PATH", "")

    if sa_email and sa_key_path and sa_key_path != "/tmp/fake-gee-key.json":
        # Production: service account authentication
        # Does not require browser interaction — works in Docker/Airflow
        logger.debug(f"Authenticating GEE with service account: {sa_email}")
        try:
            credentials = ee.ServiceAccountCredentials(sa_email, sa_key_path)
            ee.Initialize(credentials)
            logger.debug("GEE service account authentication successful")
            return
        except Exception as e:
            logger.warning(f"GEE service account auth failed: {e}. Trying default auth.")

    # Development: use credentials from `earthengine authenticate`
    # These are cached in ~/.config/earthengine/credentials
    logger.debug("Authenticating GEE with default credentials")
    try:
        ee.Initialize()
        logger.debug("GEE default authentication successful")
    except Exception as e:
        raise RuntimeError(
            f"GEE authentication failed: {e}\n"
            f"Run 'earthengine authenticate' in terminal to set up credentials."
        )


def _build_lst_composite(
    config: dict[str, Any],
    start_date: datetime,
    end_date: datetime,
):
    """
    Build a cloud-masked LST composite image for the city and week.

    Process:
    1. Filter Landsat 9 collection to city bbox and date range
    2. Filter to scenes with acceptable cloud cover
    3. Apply cloud mask using QA_PIXEL band
    4. Convert raw DN values to Celsius
    5. Compute median composite across all valid scenes

    We use median (not mean) for compositing because:
    - Mean is pulled by outliers (one extremely hot pixel distorts the mean)
    - Median is robust to remaining cloud artifacts after masking
    - Median preserves the spatial pattern better for urban heat islands

    Parameters
    ----------
    config : dict
        Pipeline config with GEE and bbox settings.
    start_date : datetime
        Start of date range.
    end_date : datetime
        End of date range.

    Returns
    -------
    ee.Image
        Single-band GEE image: LST in Celsius, clipped to city bbox.
    """
    import ee

    bbox = config["bbox"]
    cloud_max = config["gee"]["cloud_cover_max"]

    # Define city bounding box as GEE geometry
    # GEE expects coordinates as [west, south, east, north]
    city_bounds = ee.Geometry.Rectangle([
        bbox["west"],
        bbox["south"],
        bbox["east"],
        bbox["north"],
    ])

    logger.debug(
        f"Building LST composite: "
        f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} "
        f"cloud_max={cloud_max}%"
    )

    # Filter Landsat 9 collection
    collection = (
        ee.ImageCollection(LANDSAT9_COLLECTION)
        .filterBounds(city_bounds)
        .filterDate(
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )
        # Filter scenes with too much cloud cover
        # CLOUD_COVER is a scene-level metadata property
        # We filter here before pixel-level masking for efficiency
        .filter(ee.Filter.lte("CLOUD_COVER", cloud_max))
    )

    # Count available scenes
    scene_count = collection.size().getInfo()
    logger.info(f"Found {scene_count} Landsat 9 scenes with cloud cover ≤ {cloud_max}%")

    if scene_count == 0:
        # No clear scenes this week — use 30-day lookback
        logger.warning(
            f"No clear Landsat 9 scenes found for week of "
            f"{start_date.strftime('%Y-%m-%d')}. "
            f"Falling back to 30-day composite."
        )
        fallback_start = start_date - timedelta(days=30)
        collection = (
            ee.ImageCollection(LANDSAT9_COLLECTION)
            .filterBounds(city_bounds)
            .filterDate(
                fallback_start.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
            )
            .filter(ee.Filter.lte("CLOUD_COVER", cloud_max))
        )
        scene_count = collection.size().getInfo()
        logger.info(f"30-day fallback: found {scene_count} scenes")

    # Apply cloud mask and convert to Celsius to each scene
    def process_scene(image):
        """Apply cloud mask and convert LST band to Celsius."""
        # Cloud mask using QA_PIXEL band
        # Bit 3 = cloud shadow, Bit 4 = cloud
        # bitwiseAnd extracts specific bits from the QA band
        qa = image.select(QA_BAND)
        cloud_shadow_bit = 1 << 3    # bit 3
        cloud_bit = 1 << 4           # bit 4

        # Create mask: 1 = clear pixel, 0 = cloudy pixel
        cloud_mask = (
            qa.bitwiseAnd(cloud_shadow_bit).eq(0)
            .And(qa.bitwiseAnd(cloud_bit).eq(0))
        )

        # Select thermal band and apply scale + offset
        # Raw DN → Kelvin → Celsius
        lst_kelvin = (
            image.select(LST_BAND)
            .multiply(LST_SCALE)
            .add(LST_OFFSET)
        )
        lst_celsius = lst_kelvin.add(KELVIN_TO_CELSIUS)

        # Apply cloud mask — masked pixels become NoData
        return lst_celsius.updateMask(cloud_mask)

    processed = collection.map(process_scene)

    # Compute median composite
    # Each output pixel = median of all valid (unmasked) pixels
    # across all scenes in the collection
    composite = processed.median().clip(city_bounds)

    logger.debug("LST composite built successfully")

    return composite


def _export_to_gcs(
    image,
    gcs_path: str,
    config: dict[str, Any],
    ee,
) -> str:
    """
    Export a GEE image to GCS as a GeoTIFF and wait for completion.

    GEE exports are asynchronous — you submit a task and it runs
    on Google's servers. We poll the task status every N seconds
    until it completes or times out.

    Parameters
    ----------
    image : ee.Image
        The processed LST image to export.
    gcs_path : str
        Destination path inside the GCS bucket.
    config : dict
        Pipeline config with GCS bucket and export settings.
    ee : module
        The earthengine-api module (passed to avoid reimporting).

    Returns
    -------
    str
        GCS URI of the exported file.

    Raises
    ------
    RuntimeError
        If the export task fails or times out.
    """
    bucket = config["gcp"]["gcs_bucket"]
    scale = config["gee"]["scale"]
    poll_interval = config["rate_limits"]["gee_poll_interval_seconds"]
    timeout_minutes = config["rate_limits"]["gee_export_timeout_minutes"]

    # GEE export task configuration
    # fileNamePrefix is the path inside the bucket (without .tif extension)
    task = ee.batch.Export.image.toCloudStorage(
        image=image,
        description=f"CHBIndex LST {gcs_path}",
        bucket=bucket,
        fileNamePrefix=gcs_path.replace(".tif", ""),
        scale=scale,           # 30m per pixel — native Landsat resolution
        crs="EPSG:4326",       # WGS84 — matches our pipeline CRS
        fileFormat="GeoTIFF",
        maxPixels=1e9,         # allow large exports — LA County is big
    )

    task.start()
    logger.info(
        f"GEE export task submitted: {gcs_path} "
        f"(polling every {poll_interval}s, timeout {timeout_minutes}min)"
    )

    # Poll for completion
    max_polls = int((timeout_minutes * 60) / poll_interval)
    for poll_num in range(max_polls):
        time.sleep(poll_interval)

        status = task.status()
        state = status["state"]

        logger.debug(f"GEE task state: {state} (poll {poll_num + 1}/{max_polls})")

        if state == "COMPLETED":
            gcs_uri = get_gcs_uri(gcs_path, config)
            logger.info(f"GEE export completed: {gcs_uri}")
            return gcs_uri

        elif state == "FAILED":
            error_msg = status.get("error_message", "Unknown error")
            raise RuntimeError(
                f"GEE export task failed: {error_msg}\n"
                f"Task: {gcs_path}"
            )

        elif state in ("CANCELLED", "CANCEL_REQUESTED"):
            raise RuntimeError(
                f"GEE export task was cancelled: {gcs_path}"
            )

        # States: READY, RUNNING — still in progress, keep polling
        if poll_num % 5 == 0 and poll_num > 0:
            logger.info(
                f"GEE export still running... "
                f"({poll_num * poll_interval / 60:.1f} min elapsed)"
            )

    raise RuntimeError(
        f"GEE export timed out after {timeout_minutes} minutes: {gcs_path}"
    )