"""
census_blocks.py
================
Pull Census 2020 block populations and interior points for the city's county.

The dashboard paints H3 hexagons only where people live. Tracts are too
coarse for that: an Antelope Valley tract can cover hundreds of square km
of desert with its residents in one corner. Blocks (91,626 in LA County)
show where people actually are. Each block's interior point falls into
one hexagon; a hexagon with no residents is not drawn.

A block GEOID (15 digits) starts with its tract GEOID (11 digits), so the
same table also tells which tract supplies most of a hexagon's people.

Data source:
    TIGERweb, Census 2020 vintage, layer "Census Blocks". One REST layer
    holds both the 2020 count (POP100) and the interior point (INTPTLAT /
    INTPTLON: a point guaranteed to lie inside the block, unlike a
    centroid). No API key, no geometry download.

Checks:
    - row count and population sum match the server's own totals
    - every populated block's tract is in tract_geometries (2020 tracts,
      the same set used by ACS 2020-2024)

Output:
    census_blocks table, one row per block (WRITE_TRUNCATE; decennial data)
        block_geoid, geoid (tract), population, housing_units, lat, lon,
        land_area_m2, city_fips

Usage:
    PYTHONPATH=. python -m src.ingestion.census_blocks
"""

import json
from typing import Any

import pandas as pd
import requests

from src.utils.bigquery import read_query, write_dataframe
from src.utils.logger import get_logger

logger = get_logger(__name__)

TABLE = "census_blocks"
BLOCKS_URL = (
    "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
    "tigerWMS_Census2020/MapServer/10/query"
)
FIELDS = ["GEOID", "POP100", "HU100", "INTPTLAT", "INTPTLON", "AREALAND"]
PAGE_SIZE = 5000
TIMEOUT = 120
HTTP_HEADERS = {"User-Agent": "CHBIndex research pipeline"}


def pull_census_blocks(config: dict[str, Any]) -> pd.DataFrame:
    """Fetch every block in the county, check totals, replace the table."""
    where = f"STATE='{config['state_fips']}' AND COUNTY='{config['county_fips']}'"
    expected = fetch_totals(where)
    df = parse_features(fetch_all(where))
    df["city_fips"] = config["full_fips"]
    _validate(df, expected, config)
    write_dataframe(df, TABLE, config, write_disposition="WRITE_TRUNCATE")

    populated = int((df["population"] > 0).sum())
    logger.info(
        f"Census blocks: {len(df):,} blocks ({populated:,} populated), "
        f"{int(df['population'].sum()):,} people"
    )
    return df


def fetch_totals(where: str) -> dict[str, int]:
    """Block count and population sum, computed by the server."""
    stats = [
        {
            "statisticType": "count",
            "onStatisticField": "OBJECTID",
            "outStatisticFieldName": "n",
        },
        {
            "statisticType": "sum",
            "onStatisticField": "POP100",
            "outStatisticFieldName": "pop",
        },
    ]
    data = _get({"where": where, "outStatistics": json.dumps(stats), "f": "json"})
    attrs = data["features"][0]["attributes"]
    return {"n": int(attrs["n"]), "pop": int(attrs["pop"])}


def fetch_all(where: str) -> list[dict]:
    """Page through the layer in OBJECTID order (stable paging)."""
    features: list[dict] = []
    offset = 0
    while True:
        data = _get(
            {
                "where": where,
                "outFields": ",".join(FIELDS),
                "returnGeometry": "false",
                "orderByFields": "OBJECTID",
                "resultOffset": offset,
                "resultRecordCount": PAGE_SIZE,
                "f": "json",
            }
        )
        page = data.get("features", [])
        features.extend(page)
        logger.debug(f"Blocks page at {offset}: {len(page)}")
        if not page or not data.get("exceededTransferLimit"):
            return features
        offset += len(page)


def parse_features(features: list[dict]) -> pd.DataFrame:
    """ArcGIS features -> one tidy row per block."""
    raw = pd.DataFrame([f["attributes"] for f in features])
    out = pd.DataFrame(
        {
            "block_geoid": raw["GEOID"].astype(str),
            "population": pd.to_numeric(raw["POP100"]).astype("int64"),
            "housing_units": pd.to_numeric(raw["HU100"]).astype("int64"),
            # interior points come as signed strings, e.g. "+34.4590326"
            "lat": pd.to_numeric(raw["INTPTLAT"]),
            "lon": pd.to_numeric(raw["INTPTLON"]),
            "land_area_m2": pd.to_numeric(raw["AREALAND"]),
        }
    )
    out.insert(1, "geoid", out["block_geoid"].str[:11])
    return out


def _get(params: dict) -> dict:
    for attempt in range(3):
        try:
            r = requests.get(
                BLOCKS_URL, params=params, headers=HTTP_HEADERS, timeout=TIMEOUT
            )
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                raise ValueError(f"TIGERweb error: {data['error']}")
            return data
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            logger.warning(f"TIGERweb request failed (attempt {attempt + 1}), retrying")
    raise RuntimeError("unreachable")


def _validate(
    df: pd.DataFrame, expected: dict[str, int], config: dict[str, Any]
) -> None:
    if df["block_geoid"].duplicated().any():
        raise ValueError("Duplicate block GEOIDs (paging overlap)")
    if len(df) != expected["n"]:
        raise ValueError(f"Got {len(df):,} blocks, server reports {expected['n']:,}")
    if int(df["population"].sum()) != expected["pop"]:
        raise ValueError(
            f"Block population {int(df['population'].sum()):,} != server {expected['pop']:,}"
        )
    tracts = set(
        read_query(
            f"SELECT geoid FROM {{dataset}}.tract_geometries "
            f"WHERE city_fips = '{config['full_fips']}'",
            config,
        )["geoid"]
    )
    orphans = df.loc[
        (df["population"] > 0) & ~df["geoid"].isin(tracts), "geoid"
    ].unique()
    if len(orphans):
        raise ValueError(
            f"Populated blocks in tracts missing from tract_geometries: {orphans[:5]}"
        )


if __name__ == "__main__":
    from src.utils.config_loader import load_config

    pull_census_blocks(load_config("los_angeles"))
