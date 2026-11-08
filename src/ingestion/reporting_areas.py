"""
reporting_areas.py
==================
Assign each census tract to the areas a health department plans and
reports with: for LA County, the Service Planning Area (SPA), the Health
District, and the Countywide Statistical Area (a named city or community).
The weekly report uses these names instead of tract IDs.

Each tract goes to the area it overlaps most, measured in the projected CRS.
A centroid can fall outside an oddly shaped tract or in the wrong area; the
largest overlap cannot. The overlap share is kept, so tracts split between
two areas are visible.

Layers come from config (reporting_areas), one entry per level:
    region:    {url, name_field, id_field}   e.g. SPA
    district:  {url, name_field}             e.g. Health District
    community: {url, name_field}             e.g. CSA label
Each url is an ArcGIS REST layer; features are fetched as GeoJSON.

Output:
    Table: tract_reporting_areas, one row per tract
        geoid, <level>_name, <level>_share (and <level>_id when configured),
        city_fips

Usage:
    from src.ingestion.reporting_areas import pull_reporting_areas
    df = pull_reporting_areas(config)
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import pandas as pd
import requests

from src.utils.bigquery import write_dataframe
from src.utils.gcs import download_file
from src.utils.logger import get_logger

logger = get_logger(__name__)

TABLE = "tract_reporting_areas"
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (CHBIndex research pipeline)"}
PAGE_SIZE = 500
LOW_SHARE = 0.6  # below this, a tract is noticeably split between areas


def pull_reporting_areas(config: dict[str, Any]) -> pd.DataFrame:
    """Build and write the tract -> reporting area table for one city."""
    levels = config.get("reporting_areas") or {}
    if not levels:
        raise ValueError("No reporting_areas configured for this city")

    fips = config["full_fips"]
    crs = config["crs_projected"]
    tracts = gpd.read_parquet(download_file(f"tract_geometries/{fips}.parquet", config))
    tracts = tracts[["geoid", "geometry"]].to_crs(crs)

    out = tracts[["geoid"]].copy()
    for level, layer in levels.items():
        fields = [layer["name_field"]] + (
            [layer["id_field"]] if layer.get("id_field") else []
        )
        areas = fetch_layer(layer["url"], fields).to_crs(crs)
        assigned = assign_largest_overlap(
            tracts, areas, layer["name_field"], layer.get("id_field")
        )
        assigned = assigned.rename(
            columns={
                "area_name": f"{level}_name",
                "area_id": f"{level}_id",
                "share": f"{level}_share",
            }
        )
        out = out.merge(assigned, on="geoid", how="left")
        _log_level(level, out, len(areas))

    out["city_fips"] = fips
    write_dataframe(out, TABLE, config, write_disposition="WRITE_TRUNCATE")
    return out


def fetch_layer(url: str, fields: list[str]) -> gpd.GeoDataFrame:
    """All features of an ArcGIS REST layer, paged, as a GeoDataFrame in EPSG:4326."""
    frames, offset = [], 0
    while True:
        resp = requests.get(
            f"{url}/query",
            params={
                "where": "1=1",
                "outFields": ",".join(fields),
                "returnGeometry": "true",
                "outSR": 4326,
                "f": "geojson",
                "resultOffset": offset,
                "resultRecordCount": PAGE_SIZE,
            },
            headers=HTTP_HEADERS,
            timeout=120,
        )
        resp.raise_for_status()
        page = resp.json()
        if "error" in page:
            raise RuntimeError(f"ArcGIS error from {url}: {page['error']}")
        features = page.get("features", [])
        if features:
            frames.append(gpd.GeoDataFrame.from_features(features, crs="EPSG:4326"))
        if len(features) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    if not frames:
        raise RuntimeError(f"No features returned from {url}")
    gdf = pd.concat(frames, ignore_index=True)
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf["geometry"] = gdf.geometry.make_valid()
    return gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")


def assign_largest_overlap(
    tracts: gpd.GeoDataFrame,
    areas: gpd.GeoDataFrame,
    name_field: str,
    id_field: str | None = None,
) -> pd.DataFrame:
    """
    For each tract, the area covering most of it, and that area's share of
    the part of the tract any area covers.
    Both inputs must share a projected CRS. Tracts touching no area are left out.
    """
    keep = [name_field] + ([id_field] if id_field else [])
    pieces = gpd.overlay(
        tracts[["geoid", "geometry"]], areas[keep + ["geometry"]], how="intersection"
    )
    pieces["piece_area"] = pieces.geometry.area
    pieces = pieces[pieces["piece_area"] > 0]

    # Share of the tract's covered area, not its total area: coastal tracts
    # include ocean that no reporting area covers, and that is not a split.
    covered = pieces.groupby("geoid")["piece_area"].sum()
    best = pieces.sort_values("piece_area", ascending=False).drop_duplicates("geoid")
    result = pd.DataFrame(
        {
            "geoid": best["geoid"].values,
            "area_name": best[name_field].astype(str).values,
            "share": (
                best["piece_area"].values / covered.loc[best["geoid"]].values
            ).round(3),
        }
    )
    if id_field:
        result["area_id"] = best[id_field].astype(str).values
    return result


def _log_level(level: str, out: pd.DataFrame, n_areas: int) -> None:
    share = out[f"{level}_share"]
    missing = int(share.isna().sum())
    split = int((share < LOW_SHARE).sum())
    logger.info(
        f"{level}: {out[f'{level}_name'].nunique()} of {n_areas} areas used; "
        f"{missing} tracts unassigned; {split} tracts with under {LOW_SHARE:.0%} in one area"
    )
