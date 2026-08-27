"""
epa_aqs.py
==========
Pull daily PM2.5 monitor readings from the EPA Air Quality System API
for the configured city and week.

These readings are the raw input to the Kriging spatial interpolation
step in src/spatial/kriging.py. Without monitor readings, PM2.5 cannot
be estimated at unmonitored census tract locations.

Data source:
    EPA AQS API: https://aqs.epa.gov/data/api
    Parameter: 88101 (PM2.5 FRM/FEM — regulatory gold standard)
    Authentication: email + key (same value — your registered email)

Output:
    BigQuery table: epa_pm25_raw
    Columns: monitor_id, date, pm25_value, latitude, longitude,
             city_fips, week_start_date

Usage:
    from src.ingestion.epa_aqs import pull_pm25_readings
    df = pull_pm25_readings(config)
"""

import time
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests

from src.utils.bigquery import write_dataframe
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# EPA AQS API base URL
# All endpoints follow the pattern: {BASE}/{endpoint}
AQS_BASE = "https://aqs.epa.gov/data/api"

# Endpoint for daily summary data by county
# Returns one row per monitor per day for the specified date range
DAILY_SUMMARY_ENDPOINT = f"{AQS_BASE}/dailyData/byCounty"

# PM2.5 FRM/FEM parameter code
# 88101 = PM2.5 measured by Federal Reference Method
# This is the regulatory standard — the most accurate PM2.5 measurement
# Do not use 88502 (continuous monitors) — less accurate
PM25_PARAMETER_CODE = "88101"

# EPA annual PM2.5 standard in μg/m³
# Readings above this are flagged in the output
EPA_ANNUAL_STANDARD = 12.0

# EPA daily PM2.5 standard in μg/m³
# Readings above this trigger air quality alerts
EPA_DAILY_STANDARD = 35.0


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def pull_pm25_readings(
    config: dict[str, Any],
    week_start: datetime | None = None,
) -> pd.DataFrame:
    """
    Pull weekly PM2.5 monitor readings for the configured city.

    Fetches daily PM2.5 readings for the 7 days ending on the most
    recent Monday, computes a weekly mean per monitor, and writes
    results to BigQuery.

    Parameters
    ----------
    config : dict
        Merged pipeline config from config_loader.load_config().
        Uses: epa_aqs.state_code, epa_aqs.county_code,
              epa_aqs.parameter_code, secrets.EPA_AQS_EMAIL,
              secrets.EPA_AQS_KEY

    week_start : datetime, optional
        Start date of the week to pull data for.
        If None, uses the most recent Monday.
        Used for historical backfill in Airflow catchup runs.

    Returns
    -------
    pd.DataFrame
        One row per monitor with columns:
        monitor_id, monitor_name, latitude, longitude,
        pm25_mean, pm25_max, reading_count,
        above_annual_standard, above_daily_standard,
        city_fips, week_start_date

    Raises
    ------
    ValueError
        If no monitor readings are returned for the week.
    requests.exceptions.RequestException
        If API calls fail after all retry attempts.

    Examples
    --------
    >>> from src.utils.config_loader import load_config
    >>> config = load_config("los_angeles")
    >>> df = pull_pm25_readings(config)
    >>> len(df)
    11
    >>> df["pm25_mean"].describe()
    count    11.000000
    mean     12.847...
    ...
    """
    city = config["city_name"]
    state_code = config["epa_aqs"]["state_code"]
    county_code = config["epa_aqs"]["county_code"]
    full_fips = config["full_fips"]

    # Determine the week to pull
    # Default: the 7 days of the most recent complete week
    if week_start is None:
        week_start = _get_most_recent_monday()

    week_end = week_start + timedelta(days=6)

    logger.info(
        f"Pulling PM2.5 readings for {city} "
        f"({week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')})"
    )

    # Step 1 — Pull daily readings from EPA AQS API
    daily_df = _fetch_daily_readings(
        state_code=state_code,
        county_code=county_code,
        start_date=week_start,
        end_date=week_end,
        config=config,
    )

    if daily_df.empty:
        raise ValueError(
            f"No PM2.5 readings returned for {city} "
            f"week of {week_start.strftime('%Y-%m-%d')}. "
            f"Check EPA AQS API credentials and FIPS codes."
        )

    logger.info(
        f"Retrieved {len(daily_df):,} daily readings "
        f"from {daily_df['monitor_id'].nunique()} monitors"
    )

    # Step 2 — Aggregate daily readings to weekly summary per monitor
    weekly_df = _aggregate_to_weekly(daily_df, week_start, full_fips)

    logger.info(
        f"Aggregated to {len(weekly_df)} monitor weekly summaries. "
        f"City mean PM2.5: {weekly_df['pm25_mean'].mean():.2f} μg/m³"
    )

    # Step 3 — Flag readings above EPA standards
    weekly_df = _flag_exceedances(weekly_df)

    # Step 4 — Validate output
    _validate_output(weekly_df, config)

    # Step 5 — Write to BigQuery
    write_dataframe(weekly_df, "epa_pm25_raw", config)

    logger.info(
        f"EPA AQS ingestion complete: {len(weekly_df)} monitors, "
        f"{weekly_df['above_annual_standard'].sum()} above annual standard"
    )

    return weekly_df


# ---------------------------------------------------------------------------
# Private functions
# ---------------------------------------------------------------------------

def _get_most_recent_monday() -> datetime:
    """
    Return the most recent Monday as a datetime object.

    The pipeline runs every Monday — data pulled is for the
    previous 7 days (the week just completed).

    Returns
    -------
    datetime
        Most recent Monday at midnight.

    Examples
    --------
    >>> # If today is Wednesday 2026-08-26:
    >>> _get_most_recent_monday()
    datetime(2026, 8, 24, 0, 0)
    """
    today = datetime.now()
    # weekday() returns 0 for Monday, 1 for Tuesday, etc.
    # Subtracting weekday() days gives us the most recent Monday
    days_since_monday = today.weekday()
    most_recent_monday = today - timedelta(days=days_since_monday)
    return most_recent_monday.replace(hour=0, minute=0, second=0, microsecond=0)


def _fetch_daily_readings(
    state_code: str,
    county_code: str,
    start_date: datetime,
    end_date: datetime,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Fetch daily PM2.5 readings from the EPA AQS API.

    The AQS dailyData/byCounty endpoint returns one row per
    monitor per day for the specified county and date range.

    Parameters
    ----------
    state_code : str
        2-digit state FIPS code.
    county_code : str
        3-digit county FIPS code.
    start_date : datetime
        First day of the date range.
    end_date : datetime
        Last day of the date range.
    config : dict
        Pipeline config containing EPA API credentials.

    Returns
    -------
    pd.DataFrame
        Raw daily readings with columns:
        monitor_id, date, pm25_value, latitude, longitude,
        monitor_name, units
    """
    # AQS API date format: YYYYMMDD
    bdate = start_date.strftime("%Y%m%d")
    edate = end_date.strftime("%Y%m%d")

    # API credentials — email is both username and key
    email = config["secrets"].get("EPA_AQS_EMAIL", "")
    key = config["secrets"].get("EPA_AQS_KEY", "")

    params = {
        "email": email,
        "key": key,
        "param": PM25_PARAMETER_CODE,
        "bdate": bdate,
        "edate": edate,
        "state": state_code,
        "county": county_code,
    }

    logger.debug(
        f"EPA AQS API request: state={state_code} county={county_code} "
        f"bdate={bdate} edate={edate}"
    )

    # Make request with retry logic
    # Delay between requests from pipeline.yaml
    delay = config["rate_limits"]["epa_aqs_request_delay_seconds"]
    response = _request_with_retry(
        DAILY_SUMMARY_ENDPOINT,
        params,
        delay_seconds=delay,
    )

    data = response.json()

    # AQS API wraps results in a "Data" key
    # If "Data" is empty, no readings exist for this period
    readings = data.get("Data", [])

    if not readings:
        logger.warning(
            f"EPA AQS returned empty Data array for "
            f"state={state_code} county={county_code} "
            f"bdate={bdate} edate={edate}"
        )
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(readings)

    logger.debug(f"Raw AQS response: {len(df)} rows, columns: {list(df.columns)}")

    # Standardize column names
    # AQS API returns verbose column names — we rename to our standard
    column_mapping = {
        "Site Num": "site_num",
        "Parameter Code": "parameter_code",
        "Arithmetic Mean": "pm25_value",
        "Date Local": "date",
        "Latitude": "latitude",
        "Longitude": "longitude",
        "Local Site Name": "monitor_name",
        "Units of Measure": "units",
        "State Code": "state_code",
        "County Code": "county_code",
    }

    # Only rename columns that exist in the response
    existing_columns = {
        k: v for k, v in column_mapping.items()
        if k in df.columns
    }
    df = df.rename(columns=existing_columns)

    # Construct monitor_id from state + county + site number
    # Format: SS-CCC-NNNN (e.g. 06-037-1103)
    if "site_num" in df.columns:
        df["monitor_id"] = (
            state_code + "-" +
            county_code + "-" +
            df["site_num"].astype(str).str.zfill(4)
        )

    # Cast numeric columns
    df["pm25_value"] = pd.to_numeric(df["pm25_value"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Filter to only PM2.5 readings
    # Response may include other parameters if the monitor is co-located
    if "parameter_code" in df.columns:
        df = df[df["parameter_code"].astype(str) == PM25_PARAMETER_CODE]

    # Drop rows with missing PM2.5 values
    missing_count = df["pm25_value"].isna().sum()
    if missing_count > 0:
        logger.warning(f"Dropping {missing_count} rows with missing PM2.5 values")
        df = df.dropna(subset=["pm25_value"])

    # Keep only needed columns
    keep_cols = [
        "monitor_id", "date", "pm25_value",
        "latitude", "longitude", "monitor_name",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    return df


def _aggregate_to_weekly(
    daily_df: pd.DataFrame,
    week_start: datetime,
    full_fips: str,
) -> pd.DataFrame:
    """
    Aggregate daily monitor readings to weekly summary per monitor.

    For each monitor, compute:
    - Mean PM2.5 across all valid daily readings in the week
    - Maximum PM2.5 reading in the week
    - Number of valid daily readings (out of 7 possible)

    We use weekly mean rather than daily readings because:
    1. Kriging needs one stable value per monitor per week
    2. Weekly means smooth out day-to-day variation
    3. A single bad air day is less concerning than sustained elevation

    Parameters
    ----------
    daily_df : pd.DataFrame
        Daily readings from _fetch_daily_readings().
    week_start : datetime
        Start of the week being processed.
    full_fips : str
        County FIPS code for BigQuery partitioning.

    Returns
    -------
    pd.DataFrame
        One row per monitor with weekly aggregated values.
    """
    logger.debug(f"Aggregating {len(daily_df)} daily readings to weekly summaries")

    # Group by monitor and aggregate
    weekly = daily_df.groupby("monitor_id").agg(
        pm25_mean=("pm25_value", "mean"),
        pm25_max=("pm25_value", "max"),
        pm25_min=("pm25_value", "min"),
        pm25_std=("pm25_value", "std"),
        reading_count=("pm25_value", "count"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        monitor_name=("monitor_name", "first"),
    ).reset_index()

    # Round PM2.5 values to 2 decimal places
    for col in ["pm25_mean", "pm25_max", "pm25_min", "pm25_std"]:
        weekly[col] = weekly[col].round(2)

    # Add metadata columns
    weekly["city_fips"] = full_fips
    weekly["week_start_date"] = week_start.date()

    return weekly


def _flag_exceedances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag monitors whose weekly mean exceeds EPA standards.

    These flags are used in the Kriging step to weight monitors
    more heavily in spatial interpolation — monitors reporting
    elevated PM2.5 are more likely to be in pollution hotspots.

    Parameters
    ----------
    df : pd.DataFrame
        Weekly summary DataFrame from _aggregate_to_weekly().

    Returns
    -------
    pd.DataFrame
        Same DataFrame with two additional boolean columns:
        above_annual_standard, above_daily_standard
    """
    # Annual standard: weekly mean above 12 μg/m³
    df["above_annual_standard"] = df["pm25_mean"] > EPA_ANNUAL_STANDARD

    # Daily standard: any single day above 35 μg/m³
    df["above_daily_standard"] = df["pm25_max"] > EPA_DAILY_STANDARD

    exceedance_count = df["above_annual_standard"].sum()
    if exceedance_count > 0:
        logger.warning(
            f"{exceedance_count} monitors exceed EPA annual standard "
            f"({EPA_ANNUAL_STANDARD} μg/m³) this week"
        )

    return df


def _validate_output(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    """
    Validate the weekly PM2.5 summary before writing to BigQuery.

    Parameters
    ----------
    df : pd.DataFrame
        Weekly summary DataFrame.
    config : dict
        Pipeline config — used for expected monitor count.

    Raises
    ------
    ValueError
        If validation fails.
    """
    expected_monitors = config["epa_aqs"]["monitor_count"]

    # Check monitor count is close to expected
    # Allow ±3 because monitors can go offline temporarily
    actual_monitors = len(df)
    if abs(actual_monitors - expected_monitors) > 3:
        logger.warning(
            f"Expected ~{expected_monitors} monitors, "
            f"got {actual_monitors}. Some monitors may be offline."
        )

    # Check no negative PM2.5 values
    negative_count = (df["pm25_mean"] < 0).sum()
    if negative_count > 0:
        raise ValueError(
            f"{negative_count} monitors have negative PM2.5 mean — "
            f"data quality issue in EPA AQS response"
        )

    # Check PM2.5 values are in reasonable range
    # Values above 500 μg/m³ are instrument errors
    extreme_count = (df["pm25_mean"] > 500).sum()
    if extreme_count > 0:
        raise ValueError(
            f"{extreme_count} monitors have PM2.5 > 500 μg/m³ — "
            f"likely instrument error, check raw data"
        )

    # Check all monitors have coordinates
    missing_coords = df[
        df["latitude"].isna() | df["longitude"].isna()
    ].shape[0]
    if missing_coords > 0:
        raise ValueError(
            f"{missing_coords} monitors have missing coordinates — "
            f"cannot be used in Kriging interpolation"
        )

    logger.info(
        f"Validation passed: {actual_monitors} monitors, "
        f"PM2.5 range {df['pm25_mean'].min():.2f}–"
        f"{df['pm25_mean'].max():.2f} μg/m³"
    )


def _request_with_retry(
    url: str,
    params: dict,
    max_retries: int = 3,
    delay_seconds: float = 1.5,
) -> requests.Response:
    """
    Make an HTTP GET request with retry logic.

    Retries on network errors and 5xx server errors.
    Waits delay_seconds between attempts.

    Parameters
    ----------
    url : str
        API endpoint URL.
    params : dict
        Query parameters including EPA credentials.
    max_retries : int
        Maximum number of attempts. Default 3.
    delay_seconds : float
        Seconds to wait between retries.

    Returns
    -------
    requests.Response
        Successful response.

    Raises
    ------
    requests.exceptions.RequestException
        If all retry attempts fail.
    """
    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(f"EPA AQS request attempt {attempt}/{max_retries}")

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            # Respect rate limit between successful requests
            if attempt < max_retries:
                time.sleep(delay_seconds)

            return response

        except requests.exceptions.HTTPError as e:
            if e.response.status_code < 500:
                # 4xx — bad request, don't retry
                logger.error(
                    f"EPA AQS client error {e.response.status_code}: {e}"
                )
                raise

            logger.warning(
                f"EPA AQS server error {e.response.status_code} "
                f"on attempt {attempt}/{max_retries}. "
                f"Retrying in {delay_seconds}s..."
            )

        except requests.exceptions.RequestException as e:
            logger.warning(
                f"EPA AQS network error on attempt {attempt}/{max_retries}: {e}. "
                f"Retrying in {delay_seconds}s..."
            )

        if attempt < max_retries:
            time.sleep(delay_seconds)

    raise requests.exceptions.RequestException(
        f"All {max_retries} EPA AQS attempts failed for {url}"
    )