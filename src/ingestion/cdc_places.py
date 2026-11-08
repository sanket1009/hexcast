"""
cdc_places.py
=============
Pull CDC PLACES health outcome estimates for census tracts
in the configured city.

CDC PLACES provides model-based estimates of chronic disease
prevalence at census tract level — asthma, COPD, heart disease,
diabetes, and 37 other health measures. These are used as:

  1. Vulnerability features in the ML model — tracts with high
     pre-existing condition rates are more vulnerable to compound
     climate-health burden
  2. Equity audit stratification — comparing model performance
     across health outcome groups

Data source:
    CDC PLACES API (Socrata): https://data.cdc.gov/resource/cwsq-ngmh.json
    Vintage: 2025 release (2023 BRFSS data for all measures used here).
    Uses 2020 tract boundaries, so GEOIDs match census_tiger.py. Tracts
    with almost no residents (24 in LA) have no PLACES estimates.
    Geography: census tract level, all 50 states

Output:
    BigQuery table: cdc_places_health (one row per tract)

Usage:
    from src.ingestion.cdc_places import pull_health_outcomes
    df = pull_health_outcomes(config)
"""

import time
from typing import Any

import pandas as pd
import requests

from src.utils.bigquery import write_dataframe
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# CDC PLACES Socrata API endpoint
# Returns model-based health estimates at census tract level
CDC_PLACES_URL = "https://data.cdc.gov/resource/cwsq-ngmh.json"

# Health measures we pull — subset of the 40 available measures
# These are the ones most relevant to climate-health compound burden
# Full list: https://www.cdc.gov/places/measure-definitions/
HEALTH_MEASURES = {
    "CASTHMA": "asthma_pct",  # current asthma prevalence
    "CHD": "heart_disease_pct",  # coronary heart disease
    "COPD": "copd_pct",  # chronic obstructive pulmonary disease
    "DIABETES": "diabetes_pct",  # diagnosed diabetes
    "OBESITY": "obesity_pct",  # obesity
    "BPHIGH": "high_bp_pct",  # high blood pressure
    "HIGHCHOL": "high_chol_pct",  # high cholesterol
    "STROKE": "stroke_pct",  # stroke
}

# Socrata page size — max records per API call
# CDC PLACES has ~85,000 tracts nationally — we filter by state FIPS
# Most states have 1,000-5,000 tracts — one page is enough
PAGE_SIZE = 10000


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def pull_health_outcomes(config: dict[str, Any]) -> pd.DataFrame:
    """
    Pull CDC PLACES health outcome estimates for the configured city.

    Parameters
    ----------
    config : dict
        Merged pipeline config. Uses: state_fips, full_fips,
        secrets.CDC_SOCRATA_APP_TOKEN

    Returns
    -------
    pd.DataFrame
        One row per census tract with health outcome columns.
        Columns: geoid + one column per health measure in HEALTH_MEASURES.

    Examples
    --------
    >>> config = load_config("los_angeles")
    >>> df = pull_health_outcomes(config)
    >>> len(df)
    2474
    >>> df.columns.tolist()
    ['geoid', 'asthma_pct', 'heart_disease_pct', 'copd_pct', ...]
    """
    city = config["city_name"]
    state_fips = config["state_fips"]
    full_fips = config["full_fips"]

    logger.info(
        f"Pulling CDC PLACES health outcomes for {city} (state FIPS: {state_fips})"
    )

    # Step 1 — Fetch raw data from Socrata API
    raw_df = _fetch_places_data(full_fips, config)
    logger.info(f"Retrieved {len(raw_df):,} raw records from CDC PLACES API")

    # Step 2 — Filter to county and pivot to wide format
    df = _process_places_data(raw_df, full_fips)
    logger.info(f"Processed to {len(df):,} tracts with {len(df.columns)} columns")

    # Step 3 — Validate
    _validate_output(df, full_fips)

    # Step 4 — Add city identifier and write to BigQuery
    df["city_fips"] = full_fips
    write_dataframe(df, "cdc_places_health", config, write_disposition="WRITE_TRUNCATE")

    logger.info(f"CDC PLACES ingestion complete: {len(df):,} tracts")
    return df


# ---------------------------------------------------------------------------
# Private functions
# ---------------------------------------------------------------------------


def _fetch_places_data(
    full_fips: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Fetch raw CDC PLACES data from Socrata API.

    Socrata API uses SoQL (Socrata Query Language) for filtering.
    We filter by state FIPS and the specific measures we need.
    The app token increases rate limits from 1000 to 50000 req/hour.

    Parameters
    ----------
    full_fips : str
        5-digit county FIPS — filters to one county's tracts.
    config : dict
        Pipeline config containing Socrata app token.

    Returns
    -------
    pd.DataFrame
        Raw API response — one row per tract per measure.
        CDC PLACES returns data in long format (one measure per row).
    """
    app_token = config["secrets"].get("CDC_SOCRATA_APP_TOKEN", "")
    delay = config["rate_limits"]["cdc_places_request_delay_seconds"]

    headers = {}
    if app_token:
        # App token goes in header — not query parameter
        headers["X-App-Token"] = app_token

    # Build measure filter — only pull the 8 measures we need
    # SoQL IN operator: measure_id IN ('CASTHMA', 'CHD', ...)
    measure_list = "', '".join(HEALTH_MEASURES.keys())
    where_clause = (
        f"stateabbr IS NOT NULL "
        f"AND locationid LIKE '{full_fips}%' "
        f"AND measureid IN ('{measure_list}')"
    )

    params = {
        "$where": where_clause,
        "$select": "locationid,measureid,year,data_value,low_confidence_limit,high_confidence_limit",
        # Without a fixed sort order, Socrata can return overlapping or
        # missing rows across pages
        "$order": "locationid,measureid,year",
        "$limit": PAGE_SIZE,
        "$offset": 0,
    }

    all_records = []
    page = 0

    # Paginate through results
    # Most states fit in one page but we handle multiple pages
    # for large states like California or Texas
    while True:
        params["$offset"] = page * PAGE_SIZE
        logger.debug(
            f"Fetching CDC PLACES page {page + 1} (offset={params['$offset']})"
        )

        response = _request_with_retry(CDC_PLACES_URL, params, headers)
        records = response.json()

        if not records:
            break

        all_records.extend(records)
        logger.debug(f"Page {page + 1}: {len(records)} records")

        if len(records) < PAGE_SIZE:
            # Last page — fewer records than page size means no more data
            break

        page += 1
        time.sleep(delay)

    if not all_records:
        logger.warning(f"No CDC PLACES records returned for county FIPS {full_fips}")
        return pd.DataFrame()

    return pd.DataFrame(all_records)


def _process_places_data(
    raw_df: pd.DataFrame,
    full_fips: str,
) -> pd.DataFrame:
    """
    Process raw CDC PLACES data into wide format per tract.

    CDC PLACES returns long format — one row per tract per measure:
        locationid | measureid | data_value
        06037204200 | CASTHMA   | 12.3
        06037204200 | CHD       | 8.1
        06037204300 | CASTHMA   | 15.2

    We pivot to wide format — one row per tract:
        geoid       | asthma_pct | heart_disease_pct
        06037204200 | 12.3       | 8.1
        06037204300 | 15.2       | ...

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw API response from _fetch_places_data().
    full_fips : str
        5-digit county FIPS — filter to this county's tracts only.

    Returns
    -------
    pd.DataFrame
        Wide format with one row per tract, one column per measure.
    """
    if raw_df.empty:
        return pd.DataFrame()

    # Filter to county — locationid is the 11-digit tract GEOID
    # It starts with the 5-digit county FIPS
    county_df = raw_df[raw_df["locationid"].str.startswith(full_fips)].copy()

    logger.debug(f"Filtered to {len(county_df)} records for county FIPS {full_fips}")

    # Cast data_value to numeric
    county_df["data_value"] = pd.to_numeric(
        county_df["data_value"],
        errors="coerce",
    )

    # A release can mix survey years across measures (the 2025 release has
    # 2022 and 2023 rows). Keep only the latest year per tract and measure,
    # so the pivot below never averages two different years together.
    if "year" in county_df.columns:
        county_df = county_df.sort_values("year").drop_duplicates(
            subset=["locationid", "measureid"], keep="last"
        )

    # Pivot from long to wide format
    # index=locationid (tract GEOID)
    # columns=measureid (health measure code)
    # values=data_value (prevalence percentage)
    wide_df = county_df.pivot_table(
        index="locationid",
        columns="measureid",
        values="data_value",
        aggfunc="mean",  # average if duplicate tract+measure rows exist
    ).reset_index()

    # Rename columns to our standard names
    wide_df = wide_df.rename(columns={"locationid": "geoid"})
    wide_df = wide_df.rename(columns=HEALTH_MEASURES)

    # Keep only columns we expect
    expected_cols = ["geoid"] + list(HEALTH_MEASURES.values())
    existing_cols = [c for c in expected_cols if c in wide_df.columns]
    wide_df = wide_df[existing_cols]

    return wide_df


def _validate_output(df: pd.DataFrame, full_fips: str) -> None:
    """Validate CDC PLACES output before BigQuery write."""
    if df.empty:
        raise ValueError(f"No CDC PLACES data for county FIPS {full_fips}")

    if len(df) < 100:
        raise ValueError(
            f"Only {len(df)} tracts in CDC PLACES data for {full_fips}. "
            f"Expected at least 100."
        )

    # Check GEOID format — must be 11 digits
    invalid = df[df["geoid"].str.len() != 11]
    if len(invalid) > 0:
        raise ValueError(f"{len(invalid)} tracts have invalid GEOIDs")

    logger.info(
        f"Validation passed: {len(df):,} tracts, "
        f"asthma mean: {df.get('asthma_pct', pd.Series()).mean():.1f}%"
    )


def _request_with_retry(
    url: str,
    params: dict,
    headers: dict,
    max_retries: int = 3,
    delay_seconds: float = 1.0,
) -> requests.Response:
    """HTTP GET with retry logic — same pattern as epa_aqs.py."""
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as e:
            if e.response.status_code < 500:
                raise
            logger.warning(f"CDC PLACES server error attempt {attempt}/{max_retries}")
        except requests.exceptions.RequestException as e:
            logger.warning(
                f"CDC PLACES network error attempt {attempt}/{max_retries}: {e}"
            )
        if attempt < max_retries:
            time.sleep(delay_seconds)
    raise requests.exceptions.RequestException(
        f"All {max_retries} CDC PLACES attempts failed"
    )
