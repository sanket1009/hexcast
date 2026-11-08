"""
gcs.py
======
Shared Google Cloud Storage utilities for the CHBIndex pipeline.

Handles uploading and downloading files between local disk and GCS buckets.
Used primarily for:
  - Landsat LST GeoTIFF rasters exported from Google Earth Engine
  - GeoParquet files containing census tract geometries
  - MLflow model artifacts
  - Policy PDFs for the RAG corpus

Usage:
    from src.utils.gcs import upload_file, download_file, list_blobs

    # Upload a local GeoTIFF to GCS
    upload_file("data/raw/lst_rasters/la_lst_2026-08-18.tif", config)

    # Download a GeoTIFF from GCS to local disk
    local_path = download_file("rasters/lst/la_lst_2026-08-18.tif", config)

    # List all LST rasters in the bucket
    blobs = list_blobs(config, prefix="rasters/lst/")

Storage backends:
    config["storage"]["backend"] chooses where files live.
      "gcp"   -> the city's GCS bucket (production)
      "local" -> a folder on disk: {local_object_dir}/{bucket_name}/
    Paths inside the bucket are identical in both modes, so pipeline
    modules call these functions the same way either way.
"""

import shutil
from pathlib import Path
from typing import Any

from google.cloud import storage
from google.oauth2 import service_account

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Local backend (folder on disk)
# ---------------------------------------------------------------------------


def _is_local(config: dict[str, Any]) -> bool:
    """True when config selects the local folder backend instead of GCS."""
    return config.get("storage", {}).get("backend", "gcp") == "local"


def _local_path(gcs_path: str, config: dict[str, Any]) -> Path:
    """Map a bucket path to its file in the local object store."""
    root = Path(config["storage"]["local_object_dir"]) / config["gcp"]["gcs_bucket"]
    return root / gcs_path


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------


def get_client(config: dict[str, Any]) -> storage.Client:
    """
    Create and return an authenticated GCS client.

    Uses the same service account as BigQuery — one key file
    authenticates all GCP services.

    Parameters
    ----------
    config : dict
        Merged pipeline config from config_loader.load_config().

    Returns
    -------
    storage.Client
        Authenticated GCS client.

    Examples
    --------
    >>> client = get_client(config)
    >>> client.project
    'your-gcp-project-id'
    """
    project_id = config["secrets"]["GCP_PROJECT_ID"]
    sa_path = config["secrets"]["GCP_SERVICE_ACCOUNT_PATH"]

    logger.debug(f"Creating GCS client for project {project_id}")

    if sa_path and sa_path != "/tmp/fake-key.json":
        credentials = service_account.Credentials.from_service_account_file(
            sa_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        client = storage.Client(
            project=project_id,
            credentials=credentials,
        )
    else:
        # Fall back to Application Default Credentials
        # Works automatically on GCP infrastructure
        client = storage.Client(project=project_id)

    logger.debug("GCS client created successfully")
    return client


def get_bucket(config: dict[str, Any]) -> storage.Bucket:
    """
    Return the GCS bucket for this city's data.

    Bucket name comes from config["gcp"]["gcs_bucket"].
    For LA: "chbindex-06037-data"
    For Chicago: "chbindex-17031-data"

    Parameters
    ----------
    config : dict
        Merged pipeline config.

    Returns
    -------
    storage.Bucket
        The city's GCS bucket object.
    """
    client = get_client(config)
    bucket_name = config["gcp"]["gcs_bucket"]
    logger.debug(f"Accessing GCS bucket: {bucket_name}")
    return client.bucket(bucket_name)


# ---------------------------------------------------------------------------
# Upload operations
# ---------------------------------------------------------------------------


def upload_file(
    local_path: str | Path,
    config: dict[str, Any],
    gcs_path: str | None = None,
) -> str:
    """
    Upload a local file to GCS.

    Parameters
    ----------
    local_path : str or Path
        Path to the local file to upload.
        Example: "data/raw/lst_rasters/la_lst_2026-08-18.tif"
    config : dict
        Merged pipeline config.
    gcs_path : str, optional
        Destination path inside the GCS bucket.
        If None, uses the same relative path as local_path.
        Example: "rasters/lst/la_lst_2026-08-18.tif"

    Returns
    -------
    str
        Full GCS URI of the uploaded file.
        Format: gs://bucket-name/path/to/file.tif

    Raises
    ------
    FileNotFoundError
        If local_path does not exist.

    Examples
    --------
    >>> uri = upload_file("data/raw/lst_rasters/la_lst_2026-08-18.tif", config)
    >>> uri
    'gs://chbindex-06037-data/data/raw/lst_rasters/la_lst_2026-08-18.tif'
    """
    local_path = Path(local_path)

    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")

    # Use local path as GCS path if not specified
    if gcs_path is None:
        gcs_path = str(local_path)

    if _is_local(config):
        dest = _local_path(gcs_path, config)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)
        logger.info(f"Stored {local_path} → local object store {dest}")
        return str(dest)

    bucket = get_bucket(config)
    blob = bucket.blob(gcs_path)

    logger.info(f"Uploading {local_path} → gs://{bucket.name}/{gcs_path}")

    blob.upload_from_filename(str(local_path))

    gcs_uri = f"gs://{bucket.name}/{gcs_path}"
    logger.info(f"Upload complete: {gcs_uri}")

    return gcs_uri


def upload_bytes(
    data: bytes,
    gcs_path: str,
    config: dict[str, Any],
    content_type: str = "application/octet-stream",
) -> str:
    """
    Upload raw bytes to GCS without creating a local file first.

    Useful for small objects generated in memory — like JSON metadata
    or small GeoJSON files — where writing to disk first is wasteful.

    Parameters
    ----------
    data : bytes
        Raw bytes to upload.
    gcs_path : str
        Destination path inside the GCS bucket.
    config : dict
        Merged pipeline config.
    content_type : str
        MIME type of the content.
        "application/geo+json" for GeoJSON
        "image/tiff" for GeoTIFF
        "application/octet-stream" for binary (default)

    Returns
    -------
    str
        Full GCS URI of the uploaded object.
    """
    if _is_local(config):
        dest = _local_path(gcs_path, config)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        logger.info(f"Stored {len(data):,} bytes → local object store {dest}")
        return str(dest)

    bucket = get_bucket(config)
    blob = bucket.blob(gcs_path)

    logger.info(f"Uploading {len(data):,} bytes → gs://{bucket.name}/{gcs_path}")

    blob.upload_from_string(data, content_type=content_type)

    gcs_uri = f"gs://{bucket.name}/{gcs_path}"
    logger.info(f"Upload complete: {gcs_uri}")

    return gcs_uri


# ---------------------------------------------------------------------------
# Download operations
# ---------------------------------------------------------------------------


def download_file(
    gcs_path: str,
    config: dict[str, Any],
    local_path: str | Path | None = None,
) -> Path:
    """
    Download a file from GCS to local disk.

    Parameters
    ----------
    gcs_path : str
        Path to the file inside the GCS bucket.
        Example: "rasters/lst/la_lst_2026-08-18.tif"
    config : dict
        Merged pipeline config.
    local_path : str or Path, optional
        Where to save the downloaded file locally.
        If None, saves to data/raw/ with the same filename as gcs_path.

    Returns
    -------
    Path
        Local path where the file was saved.

    Raises
    ------
    google.cloud.exceptions.NotFound
        If the file does not exist in GCS.

    Examples
    --------
    >>> local_path = download_file("rasters/lst/la_lst_2026-08-18.tif", config)
    >>> local_path
    PosixPath('data/raw/lst_rasters/la_lst_2026-08-18.tif')
    """
    # Determine local save path
    if local_path is None:
        # Save to data/raw/ with the filename from gcs_path
        filename = Path(gcs_path).name
        local_path = Path("data") / "raw" / filename
    else:
        local_path = Path(local_path)

    # Create parent directory if it doesn't exist
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if _is_local(config):
        src = _local_path(gcs_path, config)
        if not src.exists():
            raise FileNotFoundError(f"Not in local object store: {src}")
        if src.resolve() != local_path.resolve():
            shutil.copy2(src, local_path)
        logger.info(f"Fetched {src} → {local_path}")
        return local_path

    bucket = get_bucket(config)
    blob = bucket.blob(gcs_path)

    logger.info(f"Downloading gs://{bucket.name}/{gcs_path} → {local_path}")

    blob.download_to_filename(str(local_path))

    logger.info(
        f"Download complete: {local_path} ({local_path.stat().st_size:,} bytes)"
    )

    return local_path


def download_bytes(
    gcs_path: str,
    config: dict[str, Any],
) -> bytes:
    """
    Download a GCS file as raw bytes without saving to disk.

    Useful for small files processed entirely in memory.

    Parameters
    ----------
    gcs_path : str
        Path to the file inside the GCS bucket.
    config : dict
        Merged pipeline config.

    Returns
    -------
    bytes
        Raw file contents as bytes.
    """
    if _is_local(config):
        return _local_path(gcs_path, config).read_bytes()

    bucket = get_bucket(config)
    blob = bucket.blob(gcs_path)

    logger.info(f"Downloading gs://{bucket.name}/{gcs_path} into memory")

    data = blob.download_as_bytes()

    logger.info(f"Downloaded {len(data):,} bytes")

    return data


# ---------------------------------------------------------------------------
# List and check operations
# ---------------------------------------------------------------------------


def list_blobs(
    config: dict[str, Any],
    prefix: str | None = None,
) -> list[str]:
    """
    List all files in the GCS bucket, optionally filtered by prefix.

    Parameters
    ----------
    config : dict
        Merged pipeline config.
    prefix : str, optional
        Filter to files whose paths start with this prefix.
        Example: "rasters/lst/" returns only LST GeoTIFF files.
        If None, lists all files in the bucket.

    Returns
    -------
    list of str
        GCS paths of all matching files (without bucket name).

    Examples
    --------
    >>> # List all LST rasters
    >>> blobs = list_blobs(config, prefix="rasters/lst/")
    >>> blobs
    ['rasters/lst/la_lst_2026-08-11.tif', 'rasters/lst/la_lst_2026-08-18.tif']
    """
    if _is_local(config):
        root = _local_path("", config)
        if not root.exists():
            return []
        paths = sorted(
            f.relative_to(root).as_posix() for f in root.rglob("*") if f.is_file()
        )
        return [p for p in paths if p.startswith(prefix or "")]

    bucket = get_bucket(config)

    logger.debug(f"Listing blobs in gs://{bucket.name}/{prefix or ''}")

    blobs = [blob.name for blob in bucket.list_blobs(prefix=prefix)]

    logger.debug(f"Found {len(blobs)} files")

    return blobs


def blob_exists(
    gcs_path: str,
    config: dict[str, Any],
) -> bool:
    """
    Check whether a file exists in GCS.

    Used before downloading to avoid errors on missing files,
    and to implement idempotent pipeline steps — skip the GEE
    export if this week's LST raster already exists in GCS.

    Parameters
    ----------
    gcs_path : str
        Path to check inside the GCS bucket.
    config : dict
        Merged pipeline config.

    Returns
    -------
    bool
        True if the file exists, False otherwise.

    Examples
    --------
    >>> # Only export from GEE if we don't already have this week's raster
    >>> if not blob_exists("rasters/lst/la_lst_2026-08-18.tif", config):
    ...     export_from_gee(config)
    """
    if _is_local(config):
        return _local_path(gcs_path, config).is_file()

    bucket = get_bucket(config)
    blob = bucket.blob(gcs_path)
    exists = blob.exists()

    logger.debug(f"gs://{bucket.name}/{gcs_path} exists: {exists}")

    return exists


def get_gcs_uri(
    gcs_path: str,
    config: dict[str, Any],
) -> str:
    """
    Build a full GCS URI from a relative path.

    Parameters
    ----------
    gcs_path : str
        Relative path inside the bucket.
    config : dict
        Merged pipeline config.

    Returns
    -------
    str
        Full GCS URI in format: gs://bucket-name/path/to/file

    Examples
    --------
    >>> get_gcs_uri("rasters/lst/la_lst_2026-08-18.tif", config)
    'gs://chbindex-06037-data/rasters/lst/la_lst_2026-08-18.tif'
    """
    if _is_local(config):
        return str(_local_path(gcs_path, config))

    bucket_name = config["gcp"]["gcs_bucket"]
    return f"gs://{bucket_name}/{gcs_path}"
