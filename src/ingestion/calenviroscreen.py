"""
calenviroscreen.py
==================
Pull CalEnviroScreen 5.0 tract results for the configured county.

CalEnviroScreen (CES) is California's official cumulative-impact index
from OEHHA. It combines pollution burden (PM2.5, ozone, diesel, traffic,
toxic sites, ...) with population characteristics (asthma, poverty,
education, ...). It is used here only as an external benchmark for the
compound index (src/evals/index_validation.py), not as an index input.

CES has no heat or hospital access component, so the two indexes are
expected to agree on PM2.5 and disagree where access drives our score.

Data source:
    OEHHA ArcGIS REST service, CalEnviroScreen 5.0 final results
    (released July 1, 2026). 2020 tract boundaries, so GEOIDs match
    census_tiger.py. California only; other states have no CES.
    Tracts with too few residents have no CES score (31 in LA).

Output:
    Table: calenviroscreen (one row per tract, full reload)

Usage:
    from src.ingestion.calenviroscreen import pull_calenviroscreen
    df = pull_calenviroscreen(config)
"""

import time
from typing import Any

import pandas as pd
import requests

from src.utils.bigquery import write_dataframe
from src.utils.logger import get_logger

logger = get_logger(__name__)

CES_VERSION = "5.0 final (2026-07-01)"
CES_URL = (
    "https://services1.arcgis.com/PCHfdHz4GlDNAhBb/ArcGIS/rest/services/"
    "calenviroscreen50results_F_070126_gdb/FeatureServer/2/query"
)
PAGE_SIZE = 2000  # the service returns at most 2,000 rows per request

# CES field -> our column
FIELDS = {
    "tract": "tract",
    "Population": "population",
    "CIscore": "ces_score",
    "CIscoreP": "ces_percentile",
    "PollutionScore": "pollution_score",
    "Pollution_Pctl": "pollution_percentile",
    "PopCharScore": "pop_char_score",
    "PM2_5": "pm25_ces",
}


def pull_calenviroscreen(config: dict[str, Any]) -> pd.DataFrame:
    """Fetch CES 5.0 for the city's county and replace the calenviroscreen table."""
    if config["state_fips"] != "06":
        raise ValueError("CalEnviroScreen covers California only")

    raw = _fetch(config["full_fips"])
    df = _process(raw, config["full_fips"])
    _validate_output(df)
    write_dataframe(df, "calenviroscreen", config, write_disposition="WRITE_TRUNCATE")
    logger.info(
        f"CalEnviroScreen {CES_VERSION}: {len(df):,} tracts, "
        f"{df['ces_score'].isna().sum()} without a score"
    )
    return df


def _fetch(full_fips: str) -> list[dict]:
    """Page through the ArcGIS query endpoint for one county."""
    # CES stores the tract GEOID as a number (6037101110), so filter by range
    lo = int(full_fips) * 1_000_000
    params = {
        "where": f"tract >= {lo} AND tract < {lo + 1_000_000}",
        "outFields": ",".join(FIELDS),
        "returnGeometry": "false",
        "resultRecordCount": PAGE_SIZE,
        "f": "json",
    }
    rows: list[dict] = []
    while True:
        data = _request_with_retry({**params, "resultOffset": len(rows)})
        if "error" in data:
            raise RuntimeError(f"CalEnviroScreen query failed: {data['error']}")
        rows += [f["attributes"] for f in data["features"]]
        if not data.get("exceededTransferLimit"):
            return rows


def _process(rows: list[dict], full_fips: str) -> pd.DataFrame:
    df = pd.DataFrame(rows).rename(columns=FIELDS)
    # Numeric GEOID loses the leading zero of the state FIPS
    df["geoid"] = df.pop("tract").astype("int64").astype(str).str.zfill(11)
    df["city_fips"] = full_fips
    df["ces_version"] = CES_VERSION
    # CES marks missing values as negative numbers in some releases
    score_cols = [c for c in FIELDS.values() if c not in ("tract", "population")]
    df[score_cols] = df[score_cols].mask(df[score_cols] < 0)
    values = [c for c in FIELDS.values() if c != "tract"]
    return df[["geoid", *values, "city_fips", "ces_version"]]


def _validate_output(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("CalEnviroScreen returned no tracts")
    if df["geoid"].duplicated().any():
        raise ValueError("Duplicate tracts in CalEnviroScreen output")
    if df["ces_score"].notna().mean() < 0.9:
        raise ValueError("More than 10% of tracts have no CalEnviroScreen score")


def _request_with_retry(params: dict, max_retries: int = 3) -> dict:
    """GET with retries on server and network errors; 4xx fails at once."""
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(CES_URL, params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code < 500:
                raise
            logger.warning(
                f"CalEnviroScreen server error attempt {attempt}/{max_retries}"
            )
        except requests.exceptions.RequestException as e:
            logger.warning(
                f"CalEnviroScreen network error attempt {attempt}/{max_retries}: {e}"
            )
        if attempt < max_retries:
            time.sleep(2.0)
    raise requests.exceptions.RequestException(
        f"All {max_retries} CalEnviroScreen attempts failed"
    )
