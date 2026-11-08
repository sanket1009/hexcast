"""
epa_aqs.py
==========
Pull daily PM2.5 monitor readings from the EPA Air Quality System API
for the configured city and week.

Monitors from neighboring counties (top-level neighbor_counties) are pulled
too and kept if they sit within epa_aqs.neighbor_buffer_km of the city
bbox. They exist only to give Kriging more monitor pairs and coverage at
the county edge; in_city marks the city's own monitors.

These readings are the raw input to the Kriging spatial interpolation
step in src/spatial/kriging.py. Without monitor readings, PM2.5 cannot
be estimated at unmonitored census tract locations.

Data source:
    EPA AQS API: https://aqs.epa.gov/data/api
    Parameter: 88101 (PM2.5 FRM/FEM — regulatory gold standard)
    Authentication: email + key (same value — your registered email)

Output:
    BigQuery table: epa_pm25_raw
    Columns: monitor_id, county_fips, in_city, latitude, longitude,
             pm25_mean/max/min/std, reading_count, city_fips, week_start_date
    city_fips is the city this run belongs to (all rows); county_fips is
    where the monitor physically sits.

Usage:
    from src.ingestion.epa_aqs import pull_pm25_readings, pull_pm25_range
    df = pull_pm25_readings(config)                         # one week
    summary = pull_pm25_range(config, first_week, last_week)  # backfill, 1 request per county-year
"""

import time
from datetime import datetime, timedelta
from typing import Any

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import box

from src.utils.bigquery import REPLACE_PARTITION, write_dataframe
from src.utils.config_loader import get_bbox_tuple
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
# Lowered from 12.0 to 9.0 in Feb 2024; upheld by the D.C. Circuit in
# June 2026. A weekly mean above it is a screening flag, not a violation
# (compliance is judged on a 3-year average).
EPA_ANNUAL_STANDARD = 9.0

# EPA 24-hour PM2.5 standard in μg/m³ (unchanged in the 2024 revision)
EPA_DAILY_STANDARD = 35.0

# Prefer 24-hour measurements over hourly-derived values when a monitor
# reports both for the same day. Lower number = preferred.
SAMPLE_DURATION_PRIORITY = {"24 HOUR": 0, "24-HR BLK AVG": 1, "1 HOUR": 2}


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
    8
    >>> df["pm25_mean"].describe()
    count    11.000000
    mean     12.847...
    ...
    """
    # Default: the 7 days of the most recent complete week
    if week_start is None:
        week_start = _get_most_recent_monday()
    week_end = week_start + timedelta(days=6)

    logger.info(
        f"Pulling PM2.5 readings for {config['city_name']} "
        f"({week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')})"
    )

    daily_df = _fetch_all_counties(config, week_start, week_end)
    if daily_df.empty:
        raise ValueError(
            f"No PM2.5 readings returned for {config['city_name']} "
            f"week of {week_start.strftime('%Y-%m-%d')}. "
            f"AQS holds quality-assured data that agencies submit with a lag "
            f"of roughly two months, so recent weeks are often empty. "
            f"Pass week_start= for an earlier week."
        )

    weekly_df = _summarize_week(daily_df, week_start, config)

    # Replace this week and city, so retries and reruns never duplicate rows
    write_dataframe(
        weekly_df, "epa_pm25_raw", config, write_disposition=REPLACE_PARTITION
    )

    logger.info(
        f"EPA AQS ingestion complete: {len(weekly_df)} monitors, "
        f"{weekly_df['above_annual_standard'].sum()} above annual standard"
    )

    return weekly_df


def pull_pm25_range(
    config: dict[str, Any],
    first_week: datetime,
    last_week: datetime,
) -> dict[str, Any]:
    """
    Pull PM2.5 for many weeks with one AQS request per county per calendar year.

    A weekly pull costs one request per county per week (52 weeks x 6
    counties = 312). AQS lets one request cover a whole calendar year, so a
    backfill fetches each year once, slices the daily readings into weeks,
    and runs every week through the same summary and checks as
    pull_pm25_readings. All weeks are written in one REPLACE_PARTITION
    write, which replaces each week separately.

    Parameters
    ----------
    first_week, last_week : datetime
        Mondays of the first and last week to load (inclusive).

    Returns
    -------
    dict
        weeks_written: list of 'YYYY-MM-DD' Mondays written
        weeks_missing: Mondays with no usable city readings (skipped, not errors)
        rows: total monitor-week rows written
    """
    if last_week < first_week:
        raise ValueError("last_week is before first_week")
    mondays = []
    week = first_week
    while week <= last_week:
        mondays.append(week)
        week += timedelta(weeks=1)

    span_end = last_week + timedelta(days=6)
    logger.info(
        f"Backfilling PM2.5 for {config['city_name']}: {len(mondays)} weeks, "
        f"{first_week:%Y-%m-%d} to {span_end:%Y-%m-%d}"
    )
    daily_df = _fetch_all_counties(config, first_week, span_end)

    frames, written, missing = [], [], []
    for monday in mondays:
        in_week = (
            daily_df[
                (daily_df["date"] >= monday)
                & (daily_df["date"] < monday + timedelta(days=7))
            ]
            if not daily_df.empty
            else daily_df
        )
        try:
            frames.append(_summarize_week(in_week, monday, config))
            written.append(f"{monday:%Y-%m-%d}")
        except ValueError as err:
            logger.warning(f"Week {monday:%Y-%m-%d} skipped: {err}")
            missing.append(f"{monday:%Y-%m-%d}")

    if frames:
        all_weeks = pd.concat(frames, ignore_index=True)
        write_dataframe(
            all_weeks, "epa_pm25_raw", config, write_disposition=REPLACE_PARTITION
        )
    rows = sum(len(f) for f in frames)
    logger.info(
        f"PM2.5 backfill: {len(written)} weeks written ({rows:,} rows), {len(missing)} missing"
    )
    return {"weeks_written": written, "weeks_missing": missing, "rows": rows}


def _fetch_all_counties(
    config: dict[str, Any],
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """
    Daily readings for the city county and its neighbors over [start, end].

    AQS requires bdate and edate in the same calendar year, so the span is
    split at each New Year: one request per county per calendar year.
    """
    aqs_cfg = config["epa_aqs"]
    full_fips = config["full_fips"]
    counties = [(aqs_cfg["state_code"], aqs_cfg["county_code"])] + [
        (fips[:2], fips[2:]) for fips in config.get("neighbor_counties", [])
    ]

    parts = []
    for state_code, county_code in counties:
        is_city = state_code + county_code == full_fips
        for b, e in _year_ranges(start, end):
            try:
                parts.append(
                    _fetch_daily_readings(
                        state_code=state_code,
                        county_code=county_code,
                        start_date=b,
                        end_date=e,
                        config=config,
                    )
                )
            except (ValueError, requests.exceptions.RequestException) as err:
                # The city county is required; a failed neighbor only
                # thins out Kriging support, so log it and carry on
                if is_city:
                    raise
                logger.warning(
                    f"Skipping neighbor county {state_code}{county_code}: {err}"
                )

    parts = [d for d in parts if not d.empty]
    if not parts:
        return pd.DataFrame()
    daily_df = pd.concat(parts, ignore_index=True)
    logger.info(
        f"Retrieved {len(daily_df):,} daily readings "
        f"from {daily_df['monitor_id'].nunique()} monitors"
    )
    return daily_df


def _year_ranges(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """Split [start, end] into pieces that each stay inside one calendar year."""
    ranges = []
    b = start
    while b <= end:
        e = min(end, datetime(b.year, 12, 31))
        ranges.append((b, e))
        b = e + timedelta(days=1)
    return ranges


def _summarize_week(
    daily_df: pd.DataFrame,
    week_start: datetime,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Daily readings for one week -> one checked row per monitor. Raises ValueError if unusable."""
    if daily_df.empty:
        raise ValueError(f"no readings for week of {week_start:%Y-%m-%d}")

    # Aggregate daily readings to a weekly summary per monitor
    weekly_df = _aggregate_to_weekly(daily_df, week_start, config["full_fips"])

    # Drop neighbor monitors outside the buffer around the city
    weekly_df = _filter_to_buffer(weekly_df, config)

    city_rows = weekly_df[weekly_df["in_city"]]
    logger.info(
        f"Week {week_start:%Y-%m-%d}: {len(weekly_df)} monitor summaries "
        f"({len(city_rows)} in city, {len(weekly_df) - len(city_rows)} neighbor). "
        f"City mean PM2.5: {city_rows['pm25_mean'].mean():.2f} μg/m³"
    )

    # Flag readings above EPA standards, then check before writing
    weekly_df = _flag_exceedances(weekly_df)
    _validate_output(weekly_df, config)
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

    # AQS returns HTTP 200 even on failure; the real status is in Header
    header = (data.get("Header") or [{}])[0]
    status = str(header.get("status", ""))
    if status.lower().startswith("failed"):
        raise ValueError(f"EPA AQS request failed: {header.get('error')}")

    readings = data.get("Data", [])

    if not readings:
        logger.warning(
            f"EPA AQS returned no data ({status}) for "
            f"state={state_code} county={county_code} "
            f"bdate={bdate} edate={edate}"
        )
        return pd.DataFrame()

    df = pd.DataFrame(readings)
    logger.debug(f"Raw AQS response: {len(df)} rows")

    # Keep only PM2.5 FRM/FEM rows
    df = df[df["parameter_code"].astype(str) == PM25_PARAMETER_CODE]

    # The API repeats each measurement once per pollutant standard version
    # (1997, 2006, 2012, 2024; annual and 24-hour), so one reading appears
    # up to 8 times with the same value. Collapse to one row per
    # site + instrument (POC) + day + sample duration.
    df = df.drop_duplicates(
        subset=["site_number", "poc", "date_local", "sample_duration_code"]
    )

    # If an instrument reports both a 24-hour and an hourly-derived value
    # for the same day, keep the 24-hour one
    df = df.assign(
        _prio=df["sample_duration"].map(SAMPLE_DURATION_PRIORITY).fillna(9)
    ).sort_values("_prio")
    df = df.drop_duplicates(subset=["site_number", "poc", "date_local"])

    df = df.rename(
        columns={
            "arithmetic_mean": "pm25_value",
            "date_local": "date",
            "local_site_name": "monitor_name",
        }
    )

    # One monitor = one site. Format: SS-CCC-NNNN (e.g. 06-037-1103)
    df["monitor_id"] = (
        state_code
        + "-"
        + county_code
        + "-"
        + df["site_number"].astype(str).str.zfill(4)
    )
    df["county_fips"] = state_code + county_code

    for col in ["pm25_value", "latitude", "longitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    missing_count = df["pm25_value"].isna().sum()
    if missing_count > 0:
        logger.warning(f"Dropping {missing_count} rows with missing PM2.5 values")
        df = df.dropna(subset=["pm25_value"])

    # Sites with several instruments (POCs): average them into one daily
    # value per site, so each day counts once in the weekly summary
    df = df.groupby(["monitor_id", "date"], as_index=False).agg(
        pm25_value=("pm25_value", "mean"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        monitor_name=("monitor_name", "first"),
        county_fips=("county_fips", "first"),
    )

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
    weekly = (
        daily_df.groupby("monitor_id")
        .agg(
            pm25_mean=("pm25_value", "mean"),
            pm25_max=("pm25_value", "max"),
            pm25_min=("pm25_value", "min"),
            pm25_std=("pm25_value", "std"),
            reading_count=("pm25_value", "count"),
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            monitor_name=("monitor_name", "first"),
            county_fips=("county_fips", "first"),
        )
        .reset_index()
    )

    # Round PM2.5 values to 2 decimal places
    for col in ["pm25_mean", "pm25_max", "pm25_min", "pm25_std"]:
        weekly[col] = weekly[col].round(2)

    # Add metadata columns
    weekly["in_city"] = weekly["county_fips"] == full_fips
    weekly["city_fips"] = full_fips
    weekly["week_start_date"] = week_start.date()

    return weekly


def _filter_to_buffer(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Keep city monitors plus neighbor monitors near the city bbox.

    Distance is measured in the city's projected CRS (meters), from each
    monitor to the bbox rectangle. A monitor inside the bbox is distance 0.
    City monitors are always kept, even outside the bbox (island sites).

    Parameters
    ----------
    df : pd.DataFrame
        Weekly summary with latitude, longitude, in_city.
    config : dict
        Uses bbox, crs, crs_projected, epa_aqs.neighbor_buffer_km.

    Returns
    -------
    pd.DataFrame
        Filtered copy with a distance_to_bbox_km column.
    """
    buffer_km = config["epa_aqs"].get("neighbor_buffer_km", 0.0)

    points = gpd.GeoSeries(
        gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=config["crs"],
    ).to_crs(config["crs_projected"])
    city_box = (
        gpd.GeoSeries([box(*get_bbox_tuple(config))], crs=config["crs"])
        .to_crs(config["crs_projected"])
        .iloc[0]
    )

    df = df.copy()
    df["distance_to_bbox_km"] = (points.distance(city_box).values / 1000).round(1)

    keep = df["in_city"] | (df["distance_to_bbox_km"] <= buffer_km)
    dropped = df[~keep]
    if not dropped.empty:
        logger.info(
            f"Dropped {len(dropped)} neighbor monitors beyond {buffer_km} km: "
            + ", ".join(dropped["monitor_id"])
        )

    return df[keep].reset_index(drop=True)


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

    # Check the city's own monitor count is close to expected
    # Allow ±3 because monitors can go offline temporarily
    actual_monitors = int(df["in_city"].sum())
    if actual_monitors == 0:
        raise ValueError("No monitors inside the city county this week")
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
    missing_coords = df[df["latitude"].isna() | df["longitude"].isna()].shape[0]
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
                # Log without the URL: it contains the email and key
                logger.error(
                    f"EPA AQS client error {e.response.status_code} "
                    f"(check EPA_AQS_EMAIL / EPA_AQS_KEY in .env)"
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
