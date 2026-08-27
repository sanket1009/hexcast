"""
config_loader.py
================
Single entry point for all pipeline configuration.

Every module in src/ imports this module first:

    from src.utils.config_loader import load_config
    config = load_config("los_angeles")

This returns one merged, validated dictionary containing:
  - All city-specific values from config/cities/{city_name}.yaml
  - All global pipeline values from config/pipeline.yaml
  - All secrets from .env (API keys, GCP credentials)

Design principle: config is loaded ONCE at module startup and passed
around as a plain dictionary. No module reads YAML files directly.
No module reads os.environ directly. Everything flows through here.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, field_validator


# ---------------------------------------------------------------------------
# Project root — all relative paths resolve from here
# ---------------------------------------------------------------------------

# Path(__file__) = .../chbindex/src/utils/config_loader.py
# .parent = .../chbindex/src/utils/
# .parent.parent = .../chbindex/src/
# .parent.parent.parent = .../chbindex/   ← project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# Pydantic models for config validation
# ---------------------------------------------------------------------------
# These define the exact shape of the config we expect.
# If a required key is missing or has the wrong type, Pydantic raises a
# clear ValidationError immediately — not a cryptic KeyError later.
# ---------------------------------------------------------------------------

class BboxConfig(BaseModel):
    """Bounding box for the city — must be valid lat/lon ranges."""
    west: float
    south: float
    east: float
    north: float

    @field_validator("west", "east")
    @classmethod
    def valid_longitude(cls, v: float) -> float:
        """Longitude must be between -180 and 180 degrees."""
        if not -180 <= v <= 180:
            raise ValueError(f"Longitude {v} is out of range [-180, 180]")
        return v

    @field_validator("south", "north")
    @classmethod
    def valid_latitude(cls, v: float) -> float:
        """Latitude must be between -90 and 90 degrees."""
        if not -90 <= v <= 90:
            raise ValueError(f"Latitude {v} is out of range [-90, 90]")
        return v


class EpaAqsConfig(BaseModel):
    """EPA AQS API parameters — required for PM2.5 ingestion."""
    state_code: str
    county_code: str
    parameter_code: str
    monitor_count: int


class GeeConfig(BaseModel):
    """Google Earth Engine parameters — required for LST ingestion."""
    collection: str
    scale: int
    cloud_cover_max: int    
    lst_band: str


class OsmnxConfig(BaseModel):
    """OSMnx road network parameters — required for 2SFCA access."""
    network_type: str
    speed_kmh_default: int


class HealthcareConfig(BaseModel):
    """Healthcare facility parameters — required for 2SFCA access."""
    er_catchment_minutes: int
    primary_care_catchment_minutes: int
    er_count: int


class BurdenThresholdsConfig(BaseModel):
    """Compound burden score thresholds for risk classification."""
    high_risk: float
    critical: float

    @field_validator("high_risk", "critical")
    @classmethod
    def valid_threshold(cls, v: float) -> float:
        """Thresholds must be between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError(f"Threshold {v} must be between 0.0 and 1.0")
        return v


class ScheduleConfig(BaseModel):
    """Airflow schedule configuration."""
    cron: str
    start_date: str
    catchup: bool


class GcpConfig(BaseModel):
    """GCP resource configuration."""
    bigquery_dataset: str
    gcs_bucket: str
    gcs_raster_prefix: str
    location: str


class CityConfig(BaseModel):
    """
    Complete city configuration — validates los_angeles.yaml or chicago.yaml.
    All fields are required unless a default is provided.
    """
    # Identity
    city_name: str
    county_name: str
    state_name: str
    state_fips: str
    county_fips: str
    full_fips: str

    # Geography
    bbox: BboxConfig
    crs: str
    crs_projected: str

    # Population
    population: int
    census_tract_count: int

    # Data sources
    epa_aqs: EpaAqsConfig
    gee: GeeConfig
    osmnx: OsmnxConfig
    healthcare: HealthcareConfig

    # Thresholds
    burden_thresholds: BurdenThresholdsConfig
    hotspot_significance: dict[str, float]

    # Airflow
    schedule: ScheduleConfig

    # GCP
    gcp: GcpConfig

    @field_validator("state_fips")
    @classmethod
    def valid_state_fips(cls, v: str) -> str:
        """State FIPS must be exactly 2 digits."""
        if not v.isdigit() or len(v) != 2:
            raise ValueError(f"state_fips '{v}' must be exactly 2 digits")
        return v

    @field_validator("county_fips")
    @classmethod
    def valid_county_fips(cls, v: str) -> str:
        """County FIPS must be exactly 3 digits."""
        if not v.isdigit() or len(v) != 3:
            raise ValueError(f"county_fips '{v}' must be exactly 3 digits")
        return v

    @field_validator("full_fips")
    @classmethod
    def valid_full_fips(cls, v: str) -> str:
        """Full FIPS must be exactly 5 digits."""
        if not v.isdigit() or len(v) != 5:
            raise ValueError(f"full_fips '{v}' must be exactly 5 digits")
        return v


# ---------------------------------------------------------------------------
# Core loader function
# ---------------------------------------------------------------------------

def load_config(city_name: str) -> dict[str, Any]:
    """
    Load, merge, and validate all configuration for a given city.

    Parameters
    ----------
    city_name : str
        City identifier matching a YAML filename in config/cities/.
        Examples: "los_angeles", "chicago"

    Returns
    -------
    dict[str, Any]
        Merged dictionary containing:
        - All city-specific config from config/cities/{city_name}.yaml
        - All global pipeline config from config/pipeline.yaml
        - Secrets from .env injected under config["secrets"]

    Raises
    ------
    FileNotFoundError
        If the city YAML file does not exist.
    ValidationError
        If any required config key is missing or has wrong type.
    ValueError
        If FIPS codes or geographic bounds are invalid.

    Examples
    --------
    >>> config = load_config("los_angeles")
    >>> config["full_fips"]
    '06037'
    >>> config["ml"]["random_seed"]
    42
    >>> config["bbox"]["west"]
    -118.9448
    """
    # Step 1 — Load secrets from .env into os.environ
    # load_dotenv() reads .env and sets each line as an environment variable.
    # override=False means existing environment variables take precedence —
    # important for CI/CD where secrets are injected by the environment.
    env_path = PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path=env_path, override=False)

    # Step 2 — Load city-specific YAML
    city_config_path = PROJECT_ROOT / "config" / "cities" / f"{city_name}.yaml"

    if not city_config_path.exists():
        raise FileNotFoundError(
            f"City config not found: {city_config_path}\n"
            f"Available cities: {_list_available_cities()}"
        )

    with open(city_config_path, "r") as f:
        city_raw = yaml.safe_load(f)

    # Step 3 — Load global pipeline YAML
    pipeline_config_path = PROJECT_ROOT / "config" / "pipeline.yaml"

    if not pipeline_config_path.exists():
        raise FileNotFoundError(
            f"Pipeline config not found: {pipeline_config_path}"
        )

    with open(pipeline_config_path, "r") as f:
        pipeline_raw = yaml.safe_load(f)

    # Step 4 — Validate city config against Pydantic model
    # This raises ValidationError immediately if any required key is
    # missing or has the wrong type — much better than a cryptic
    # KeyError 500 lines into a spatial computation module.
    try:
        validated_city = CityConfig(**city_raw)
    except ValidationError as e:
        raise ValidationError(
            f"City config validation failed for '{city_name}':\n{e}"
        ) from e

    # Step 5 — Merge city config + pipeline config into one dictionary
    # City config takes precedence over pipeline config if keys conflict.
    # In practice there are no overlapping keys — they cover different domains.
    merged = {
        **pipeline_raw,                    # global settings first
        **validated_city.model_dump(),     # city settings override if conflict
        "city_name_slug": city_name,       # normalized city identifier
        "project_root": str(PROJECT_ROOT), # absolute path for file operations
    }

    # Step 6 — Inject secrets from environment variables
    # These come from .env (local dev) or CI/CD environment injection.
    # They are added under a "secrets" key — never mixed with config values
    # so it's always clear which values are secrets vs configuration.
    merged["secrets"] = _load_secrets()

    return merged


def _load_secrets() -> dict[str, str]:
    """
    Load sensitive values from environment variables.

    These are set by .env locally and by CI/CD secrets in production.
    Any missing secret gets an empty string with a warning — not a crash —
    because some modules don't need all secrets and crashing at config
    load time would break modules that don't use the missing secret.

    Returns
    -------
    dict[str, str]
        Dictionary of secret key-value pairs.
    """
    secrets = {}
    required_secrets = {
        "ANTHROPIC_API_KEY": "Required for LLM briefing generation",
        "GCP_PROJECT_ID": "Required for BigQuery and GCS operations",
        "GCP_SERVICE_ACCOUNT_PATH": "Path to GCP service account JSON key",
        "GEE_SERVICE_ACCOUNT": "Google Earth Engine service account email",
        "GEE_PRIVATE_KEY_PATH": "Path to GEE private key JSON file",
    }

    for key, description in required_secrets.items():
        value = os.environ.get(key, "")
        if not value:
            # Warn but don't crash — let each module decide if it needs this
            print(f"WARNING: Secret '{key}' not set. {description}")
        secrets[key] = value

    return secrets


def _list_available_cities() -> list[str]:
    """
    Return list of available city config files.

    Used in error messages when a city name is not found.
    """
    cities_dir = PROJECT_ROOT / "config" / "cities"
    return [f.stem for f in cities_dir.glob("*.yaml")]


# ---------------------------------------------------------------------------
# Convenience accessor functions
# ---------------------------------------------------------------------------
# These are thin wrappers that make common config lookups more readable.
# Instead of config["gcp"]["bigquery_dataset"] everywhere, modules can
# call get_bigquery_dataset(config) which is self-documenting.
# ---------------------------------------------------------------------------

def get_city_fips(config: dict[str, Any]) -> str:
    """Return the full 5-digit county FIPS code."""
    return config["full_fips"]


def get_bbox_tuple(config: dict[str, Any]) -> tuple[float, float, float, float]:
    """
    Return bounding box as (west, south, east, north) tuple.
    This format is expected by OSMnx and most rasterio operations.
    """
    bbox = config["bbox"]
    return (bbox["west"], bbox["south"], bbox["east"], bbox["north"])


def get_bigquery_dataset(config: dict[str, Any]) -> str:
    """Return the BigQuery dataset name for this city."""
    return config["gcp"]["bigquery_dataset"]


def get_gcs_bucket(config: dict[str, Any]) -> str:
    """Return the GCS bucket name for this city."""
    return config["gcp"]["gcs_bucket"]


def get_high_risk_threshold(config: dict[str, Any]) -> float:
    """Return the compound burden score threshold for high-risk classification."""
    return config["burden_thresholds"]["high_risk"]


def get_critical_threshold(config: dict[str, Any]) -> float:
    """Return the compound burden score threshold for critical classification."""
    return config["burden_thresholds"]["critical"]