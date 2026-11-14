"""Simplified tract outlines for the map (src/spatial/tract_outlines.py)."""

import json

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

from src.spatial.tract_outlines import outlines_from_tracts


def test_outlines_are_simplified_rounded_geojson():
    # A square with 200 points along one edge: simplification keeps the corners
    edge = [(-118 + i * 0.0001 / 4, 34.0) for i in range(200)]
    ring = edge + [(-117.995, 34.0), (-117.995, 34.005), (-118.0, 34.005)]
    tracts = gpd.GeoDataFrame(
        {"geoid": ["06037000100"]}, geometry=[Polygon(ring)], crs=4326
    ).to_crs(3310)  # projected input is reprojected to lon/lat
    out = outlines_from_tracts(tracts)
    geom = json.loads(out["geometry_geojson"].iloc[0])
    assert geom["type"] == "Polygon"
    coords = np.array(geom["coordinates"][0])
    assert len(coords) < 10
    assert np.allclose(coords, np.round(coords, 5))
    assert coords[:, 0].min() > -118.001 and coords[:, 1].max() < 34.006
