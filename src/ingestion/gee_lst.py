"""
gee_lst.py
==========
Pull Landsat 8 + 9 Land Surface Temperature (LST) rasters from
Google Earth Engine for the configured city and week.

LST is the heat signal in the compound burden index — it captures
urban heat islands at 30m resolution across the entire county,
far more spatially detailed than any weather station network.

The output GeoTIFF is read by src/spatial/lst_processing.py which
computes mean LST per census tract polygon.

Data source:
    Google Earth Engine: LANDSAT/LC08/C02/T1_L2 + LANDSAT/LC09/C02/T1_L2
    Band: ST_B10 (thermal infrared → land surface temperature)
    Resolution: 30 meters per pixel
    Revisit: 16 days per satellite, 8 days combined

Compositing window:
    A single week rarely has a clear pass over the whole county (LA
    spans two satellite paths, and coastal fog is common in early
    summer). Each week's raster is therefore a median of all clear
    pixels in the trailing gee.composite_window_days (default 32)
    ending on the last day of the week.

Output:
    GeoTIFF at rasters/lst/{fips}_{week_start}.tif in the object store
    (GCS bucket, or the local folder in storage.backend: local)
    - Single band: LST in Celsius, float32, NaN = no clear pixel
    - CRS: EPSG:4326
    - Clipped to city bounding box

Delivery:
    gcp   -> Earth Engine batch export straight to the GCS bucket
    local -> the area is split into tiles, each downloaded with
             getDownloadURL (which caps request size), then mosaicked

Authentication:
    Requires Google Earth Engine account registered at:
    https://signup.earthengine.google.com
    For development: run `earthengine authenticate` in terminal
    For production: use service account credentials in .env
    Both need GEE_PROJECT_ID in .env: the Cloud project registered for
    Earth Engine (current earthengine-api requires a project).

Usage:
    from src.ingestion.gee_lst import pull_lst_raster
    gcs_uri = pull_lst_raster(config)
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from src.utils.gcs import blob_exists, get_gcs_uri, upload_file
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Thermal infrared band containing land surface temperature
# ST_B10 = Surface Temperature Band 10
# Raw values are in Kelvin × 0.00341802 + 149.0 (scale factor)
# We convert to Celsius in the GEE processing script
LST_BAND = "ST_B10"

# QA_PIXEL band bit flags for cloud masking (Landsat Collection 2)
# Bit 0 = fill, Bit 1 = dilated cloud, Bit 3 = cloud, Bit 4 = cloud shadow
# Pixels with any of these bits set get masked out
QA_BAND = "QA_PIXEL"
QA_MASK_BITS = (1 << 0) | (1 << 1) | (1 << 3) | (1 << 4)

# Placeholder for masked pixels during download; converted to NaN locally
DOWNLOAD_NODATA = -9999.0

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

    # Step 3 — Deliver the raster to the object store
    if config.get("storage", {}).get("backend", "gcp") == "local":
        gcs_uri = _download_tiled(image=lst_image, gcs_path=gcs_path, config=config, ee=ee)
    else:
        gcs_uri = _export_to_gcs(image=lst_image, gcs_path=gcs_path, config=config, ee=ee)

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
    project = config["secrets"].get("GEE_PROJECT_ID", "")
    if not project:
        raise RuntimeError(
            "GEE_PROJECT_ID is not set in .env. Use the ID of the Cloud project "
            "registered for Earth Engine (console.cloud.google.com, project picker)."
        )

    # Only try the service account when its key file actually exists
    # (the .env template ships a placeholder path)
    if sa_email and sa_key_path and Path(sa_key_path).expanduser().is_file():
        # Production: service account authentication
        # Does not require browser interaction — works in Docker/Airflow
        logger.debug(f"Authenticating GEE with service account: {sa_email}")
        try:
            credentials = ee.ServiceAccountCredentials(sa_email, sa_key_path)
            ee.Initialize(credentials, project=project)
            logger.debug("GEE service account authentication successful")
            return
        except Exception as e:
            logger.warning(f"GEE service account auth failed: {e}. Trying default auth.")

    # Development: use credentials from `earthengine authenticate`
    # These are cached in ~/.config/earthengine/credentials
    logger.debug("Authenticating GEE with default credentials")
    try:
        ee.Initialize(project=project)
        logger.debug(f"GEE default authentication successful (project {project})")
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
    gee_cfg = config["gee"]
    cloud_max = gee_cfg["cloud_cover_max"]
    window_days = gee_cfg["composite_window_days"]

    city_bounds = ee.Geometry.Rectangle([
        bbox["west"], bbox["south"], bbox["east"], bbox["north"],
    ])

    # Trailing window ending on the last day of the week.
    # filterDate's end is exclusive, so stop at the day after week_end.
    window_end = end_date + timedelta(days=1)
    window_start = window_end - timedelta(days=window_days)

    collection = None
    for coll_id in gee_cfg["collections"]:
        c = (
            ee.ImageCollection(coll_id)
            .filterBounds(city_bounds)
            .filterDate(window_start.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d"))
            .filter(ee.Filter.lte("CLOUD_COVER", cloud_max))
        )
        logger.info(f"{coll_id}: {c.size().getInfo()} scenes with cloud cover <= {cloud_max}%")
        collection = c if collection is None else collection.merge(c)

    scene_count = collection.size().getInfo()
    logger.info(
        f"Compositing {scene_count} scenes from "
        f"{window_start.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} "
        f"({window_days}-day window)"
    )
    if scene_count == 0:
        raise RuntimeError(
            f"No Landsat scenes under {cloud_max}% cloud in the {window_days}-day "
            f"window ending {end_date.strftime('%Y-%m-%d')}. Widen "
            f"gee.composite_window_days or raise gee.cloud_cover_max."
        )

    def process_scene(image):
        """Mask fill/cloud/shadow pixels and convert ST_B10 to Celsius."""
        qa = image.select(QA_BAND)
        clear = qa.bitwiseAnd(QA_MASK_BITS).eq(0)
        lst_celsius = (
            image.select(LST_BAND)
            .multiply(LST_SCALE)
            .add(LST_OFFSET)
            .add(KELVIN_TO_CELSIUS)
            .rename("lst_celsius")
        )
        return lst_celsius.updateMask(clear)

    composite = collection.map(process_scene).median().clip(city_bounds)

    # Share of the bbox with at least one clear pixel (sampled at 300 m)
    stats = composite.mask().reduceRegion(
        reducer=ee.Reducer.mean(), geometry=city_bounds, scale=300, maxPixels=1e9
    ).getInfo()
    coverage = float(stats.get("lst_celsius") or 0.0)
    logger.info(f"Clear-pixel coverage of the bbox: {coverage:.1%}")
    if coverage < gee_cfg["min_coverage"]:
        logger.warning(
            f"Only {coverage:.1%} of the area has a clear pixel "
            f"(threshold {gee_cfg['min_coverage']:.0%}). Tracts in gaps will get NaN LST."
        )

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


def _download_tiled(image, gcs_path: str, config: dict[str, Any], ee) -> str:
    """
    Download the composite to the local object store in tiles.

    getDownloadURL caps each request (about 32 MB), and the full LA
    bbox at 30 m is about 4,800 x 4,200 float pixels (~80 MB). We split
    the bbox into an n x n grid, download each tile, and mosaic them
    with rasterio. Masked pixels arrive as DOWNLOAD_NODATA and are
    written as NaN, which lst_processing.py treats as missing.
    """
    import io
    import numpy as np
    import rasterio
    import requests
    from rasterio.merge import merge

    bbox = config["bbox"]
    n = config["gee"]["download_tiles"]
    scale = config["gee"]["scale"]

    image = image.unmask(DOWNLOAD_NODATA).toFloat()
    xs = np.linspace(bbox["west"], bbox["east"], n + 1)
    ys = np.linspace(bbox["south"], bbox["north"], n + 1)

    datasets = []
    try:
        for i in range(n):
            for j in range(n):
                region = ee.Geometry.Rectangle([xs[i], ys[j], xs[i + 1], ys[j + 1]])
                content = _get_tile_with_retry(image, region, scale, ee, requests)
                mem = rasterio.MemoryFile(content)
                datasets.append((mem, mem.open()))
                logger.info(f"Downloaded LST tile {i * n + j + 1}/{n * n} ({len(content) / 1e6:.1f} MB)")

        mosaic, transform = merge([ds for _, ds in datasets], nodata=DOWNLOAD_NODATA)
        profile = datasets[0][1].profile
    finally:
        for mem, ds in datasets:
            ds.close()
            mem.close()

    band = mosaic[0].astype("float32")
    band[band == DOWNLOAD_NODATA] = np.nan

    local_path = Path("data") / "raw" / "lst_rasters" / Path(gcs_path).name
    local_path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(
        driver="GTiff", height=band.shape[0], width=band.shape[1], count=1,
        dtype="float32", nodata=float("nan"), transform=transform, compress="deflate",
    )
    with rasterio.open(local_path, "w", **profile) as dst:
        dst.write(band, 1)

    valid = np.isfinite(band)
    logger.info(
        f"Mosaic {band.shape[1]} x {band.shape[0]} px, {valid.mean():.1%} valid, "
        f"LST {np.nanmin(band):.1f} to {np.nanmax(band):.1f} C (median {np.nanmedian(band):.1f})"
    )
    return upload_file(local_path, config, gcs_path=gcs_path)


def _get_tile_with_retry(image, region, scale: int, ee, requests, max_attempts: int = 5) -> bytes:
    """
    Download one tile, retrying on Earth Engine's transient errors.

    Earth Engine computes each tile on demand and returns 429 (too many
    requests) or 5xx (overloaded) under load. Those are temporary, so we
    wait and retry with exponential backoff: 5 s, 10 s, 20 s, 40 s.
    Other errors (bad region, auth) fail immediately.
    """
    retryable = {429, 500, 502, 503, 504}
    for attempt in range(1, max_attempts + 1):
        # A fresh URL each attempt: the server-side computation is re-queued
        url = image.getDownloadURL({
            "region": region,
            "scale": scale,
            "crs": "EPSG:4326",
            "format": "GEO_TIFF",
        })
        try:
            resp = requests.get(url, timeout=300)
            if resp.status_code not in retryable:
                resp.raise_for_status()
                return resp.content
            reason = f"HTTP {resp.status_code}"
        except requests.exceptions.ConnectionError as e:
            reason = f"connection error: {e}"

        if attempt == max_attempts:
            raise RuntimeError(f"LST tile download failed after {max_attempts} attempts ({reason})")
        wait = 5 * 2 ** (attempt - 1)
        logger.warning(f"Tile download {reason}; retry {attempt}/{max_attempts - 1} in {wait}s")
        time.sleep(wait)
