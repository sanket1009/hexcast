"""
tract_outlines.py
=================
Simplified tract boundaries for the dashboard map, stored in the warehouse.

The API serves the map's tract outlines from this table, so the serving
container needs no GeoPandas and reads one place (DuckDB locally, BigQuery
in production) for everything.

Simplification: Douglas-Peucker at 0.0001 degrees (about 10 m), then
coordinates snapped to 1e-5 degrees (about 1 m). Full-precision TIGER
outlines are 2.5 MB of GeoJSON for LA County; this is about 1.3 MB before
gzip. Coarser tolerances shrink it little more (0.0002 gives 1.0 MB) and
blur the small dense tracts that matter most. Each tract is simplified on its
own, so neighbors can differ by a few meters along a shared edge; at map zoom
that is invisible.

Output:
    tract_outlines, one row per tract (WRITE_TRUNCATE; changes with TIGER)
        geoid, geometry_geojson (GeoJSON geometry, EPSG:4326), city_fips

Usage:
    PYTHONPATH=. python -m src.spatial.tract_outlines
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import pandas as pd
import shapely

from src.utils.bigquery import write_dataframe
from src.utils.gcs import download_file
from src.utils.logger import get_logger

logger = get_logger(__name__)

TABLE = "tract_outlines"
SIMPLIFY_DEGREES = 0.0001
GRID_DEGREES = 1e-5


def outlines_from_tracts(tracts: gpd.GeoDataFrame) -> pd.DataFrame:
    tracts = tracts.to_crs(4326)
    simple = tracts.geometry.simplify(SIMPLIFY_DEGREES, preserve_topology=True)
    geoms = shapely.set_precision(simple.values, GRID_DEGREES)
    bad = [g for g, geom in zip(tracts["geoid"], geoms, strict=False) if geom.is_empty]
    if bad:
        raise ValueError(f"Simplification emptied {len(bad)} tracts: {bad[:5]}")
    return pd.DataFrame(
        {
            "geoid": tracts["geoid"].to_numpy(),
            "geometry_geojson": shapely.to_geojson(geoms),
        }
    )


def build_tract_outlines(config: dict[str, Any], write: bool = True) -> pd.DataFrame:
    fips = config["full_fips"]
    tracts = gpd.read_parquet(download_file(f"tract_geometries/{fips}.parquet", config))
    out = outlines_from_tracts(tracts[["geoid", "geometry"]])
    out["city_fips"] = fips
    if write:
        write_dataframe(out, TABLE, config, write_disposition="WRITE_TRUNCATE")
    size_kb = out["geometry_geojson"].str.len().sum() / 1024
    logger.info(f"Tract outlines: {len(out):,} tracts, {size_kb:,.0f} KB of GeoJSON")
    return out


if __name__ == "__main__":
    from src.utils.config_loader import load_config

    build_tract_outlines(load_config("los_angeles"))
