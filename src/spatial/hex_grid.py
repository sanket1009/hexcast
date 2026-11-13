"""
hex_grid.py
===========
H3 hexagons where people live, for the dashboard map.

Each populated Census 2020 block's interior point falls into one H3 cell;
a cell's population is the sum of its blocks. Cells with no residents are
not stored, so mountains, forest and open space stay blank.

Each cell takes the tract that supplies most of its people (block GEOIDs
start with the tract GEOID). Resolution 9 (about 0.1 sq km), not 8: at
res 8, 448 small dense tracts (1.38M people) were never a cell's main
tract and would vanish from the hex map; at res 9, 6 tracts (10,250
people) do. Tract outlines drawn on top still show those 6.

Output:
    h3_hexes, one row per populated cell (WRITE_TRUNCATE; changes with the
    decennial census)
        h3_index, population, geoid (main tract), main_share, n_tracts,
        lat, lon (cell center), city_fips

Usage:
    PYTHONPATH=. python -m src.spatial.hex_grid
"""

from __future__ import annotations

from typing import Any

import h3
import pandas as pd

from src.utils.bigquery import read_query, write_dataframe
from src.utils.logger import get_logger

logger = get_logger(__name__)

TABLE = "h3_hexes"


def build_hex_table(config: dict[str, Any], write: bool = True) -> pd.DataFrame:
    fips = config["full_fips"]
    blocks = read_query(
        f"""
        SELECT geoid, population, lat, lon FROM {{dataset}}.census_blocks
        WHERE city_fips = '{fips}' AND population > 0
        """,
        config,
    )
    if blocks.empty:
        raise ValueError(
            "No populated census_blocks; run src.ingestion.census_blocks first"
        )
    out = hexes_from_blocks(blocks, config["product"]["h3_resolution"])
    out["city_fips"] = fips

    missing = set(blocks["geoid"]) - set(out["geoid"])
    lost = int(blocks.loc[blocks["geoid"].isin(missing), "population"].sum())
    if out["population"].sum() != blocks["population"].sum():
        raise ValueError("Hex population does not add up to block population")
    if write:
        write_dataframe(out, TABLE, config, write_disposition="WRITE_TRUNCATE")
    logger.info(
        f"H3 hexes: {len(out):,} populated cells; {len(missing)} tracts "
        f"({lost:,} people) are no cell's main tract"
    )
    return out


def hexes_from_blocks(blocks: pd.DataFrame, resolution: int) -> pd.DataFrame:
    b = blocks.assign(
        h3_index=[
            h3.latlng_to_cell(a, o, resolution)
            for a, o in zip(blocks["lat"], blocks["lon"], strict=False)
        ]
    )
    by_tract = b.groupby(["h3_index", "geoid"], as_index=False)["population"].sum()
    cell = by_tract.groupby("h3_index").agg(
        population=("population", "sum"), n_tracts=("geoid", "nunique")
    )
    main = (
        by_tract.sort_values(["population", "geoid"], ascending=[False, True])
        .drop_duplicates("h3_index")
        .set_index("h3_index")
    )
    cell["geoid"] = main["geoid"]
    cell["main_share"] = (main["population"] / cell["population"]).round(3)
    centers = [h3.cell_to_latlng(c) for c in cell.index]
    cell["lat"] = [c[0] for c in centers]
    cell["lon"] = [c[1] for c in centers]
    return cell.reset_index()[
        ["h3_index", "population", "geoid", "main_share", "n_tracts", "lat", "lon"]
    ]


if __name__ == "__main__":
    from src.utils.config_loader import load_config

    build_hex_table(load_config("los_angeles"))
