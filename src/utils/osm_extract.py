"""
osm_extract.py
==============
Build an OSMnx drive network from a Geofabrik OpenStreetMap extract,
without the Overpass API.

Why:
    Overpass is a shared, volunteer-run query service sized for small
    areas. A metro area plus buffer splits into many heavy sub-queries,
    and both overpass-api.de and the kumi.systems mirror refused or timed
    out on them (Nov 1-2, 2026). Geofabrik publishes daily regional
    extracts as plain files: one download, no rate limits, reproducible.

How:
    1. Download the regional .osm.pbf once (cached; refreshed on request)
    2. Check the extract's own bounds cover the requested bbox
    3. One pass with pyosmium: keep ways that pass OSMnx's "drive" filter
       and have at least one node inside the bbox; write them, plus the
       nodes they reference, to a small OSM XML file
    4. ox.graph_from_xml builds and simplifies the graph exactly as
       graph_from_bbox would after an Overpass download

The drive filter below reproduces osmnx._overpass._get_network_filter
("drive") and settings.default_access from OSMnx 2.1.1. Overpass "!~"
is an unanchored regex test, so the same unanchored re.search is used.
"""

import re
import time
from pathlib import Path

import networkx as nx
import osmium
import osmnx as ox
import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)

# A way is excluded when the tag's value matches the pattern (unanchored,
# as Overpass "!~"). Same keys and values as OSMnx's "drive" filter.
DRIVE_EXCLUDE = {
    "area": re.compile("yes"),
    "access": re.compile("private"),
    "highway": re.compile(
        "abandoned|bridleway|bus_guideway|construction|corridor|cycleway|"
        "elevator|escalator|footway|no|path|pedestrian|planned|platform|"
        "proposed|raceway|razed|rest_area|service|services|steps|track"
    ),
    "motor_vehicle": re.compile("no"),
    "motorcar": re.compile("no"),
    "service": re.compile(
        "alley|driveway|emergency_access|parking|parking_aisle|private"
    ),
}


def is_drive_way(tags) -> bool:
    """True if a way's tags pass OSMnx's "drive" network filter."""
    if "highway" not in tags:
        return False
    for key, pattern in DRIVE_EXCLUDE.items():
        value = tags.get(key)
        if value is not None and pattern.search(value):
            return False
    return True


def download_extract(url: str, path: Path, refresh: bool = False) -> Path:
    """Download a Geofabrik extract to path, streaming to disk."""
    if path.exists() and not refresh:
        logger.info(
            f"Using cached OSM extract: {path} ({path.stat().st_size / 1e6:,.0f} MB)"
        )
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    logger.info(f"Downloading OSM extract {url}")
    start = time.time()
    with requests.get(
        url, stream=True, timeout=60, headers={"User-Agent": "chbindex/1.0"}
    ) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done, next_log = 0, 0.1
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
                done += len(chunk)
                if total and done / total >= next_log:
                    logger.info(f"  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB")
                    next_log += 0.1
    tmp.replace(path)  # only a complete download gets the real name
    logger.info(
        f"Downloaded {path.stat().st_size / 1e6:,.0f} MB in {time.time() - start:.0f}s"
    )
    return path


def check_coverage(pbf: Path, bbox: tuple[float, float, float, float]) -> None:
    """Raise if the extract's declared bounds do not contain the bbox."""
    west, south, east, north = bbox
    box = osmium.io.Reader(str(pbf), osmium.osm.osm_entity_bits.NOTHING).header().box()
    if not box.valid():
        logger.warning("Extract header has no bounds; skipping coverage check")
        return
    bl, tr = box.bottom_left, box.top_right
    logger.info(
        f"Extract bounds W{bl.lon:.3f} S{bl.lat:.3f} E{tr.lon:.3f} N{tr.lat:.3f}"
    )
    if not (bl.lon <= west and bl.lat <= south and tr.lon >= east and tr.lat >= north):
        raise ValueError(
            f"OSM extract {pbf.name} does not cover the bbox "
            f"W{west:.3f} S{south:.3f} E{east:.3f} N{north:.3f}. "
            f"Use a larger extract (osmnx.geofabrik_url in the city config)."
        )


def filter_drive_ways(
    pbf: Path,
    out_osm: Path,
    bbox: tuple[float, float, float, float],
) -> int:
    """
    Write drive ways touching the bbox, with their nodes, to OSM XML.

    One pass over the extract. Node locations are kept in memory so each
    way's nodes can be tested against the bbox; BackReferenceWriter then
    adds every node the kept ways reference, including those just outside
    the bbox, so edge roads are not cut mid-segment.

    Returns the number of ways written.
    """
    west, south, east, north = bbox
    out_osm.parent.mkdir(parents=True, exist_ok=True)
    if out_osm.exists():
        out_osm.unlink()

    kept = 0
    start = time.time()
    fp = (
        osmium.FileProcessor(str(pbf))
        .with_locations()
        .with_filter(osmium.filter.EntityFilter(osmium.osm.WAY))
        .with_filter(osmium.filter.KeyFilter("highway"))
    )
    with osmium.BackReferenceWriter(
        str(out_osm), ref_src=str(pbf), overwrite=True
    ) as writer:
        for way in fp:
            if not is_drive_way(way.tags):
                continue
            inside = False
            for n in way.nodes:
                loc = n.location
                if (
                    loc.valid()
                    and west <= loc.lon <= east
                    and south <= loc.lat <= north
                ):
                    inside = True
                    break
            if inside:
                writer.add_way(way)
                kept += 1

    logger.info(
        f"Filtered {kept:,} drive ways in the bbox to {out_osm.name} "
        f"({out_osm.stat().st_size / 1e6:,.0f} MB) in {time.time() - start:.0f}s"
    )
    return kept


def graph_from_extract(
    url: str,
    bbox: tuple[float, float, float, float],
    work_dir: Path,
    refresh: bool = False,
) -> nx.MultiDiGraph:
    """
    Drive network for bbox from a Geofabrik extract, as an OSMnx graph.

    retain_all=False keeps the largest weakly connected network, matching
    graph_from_bbox(..., retain_all=False).
    """
    pbf = download_extract(url, work_dir / Path(url).name, refresh=refresh)
    check_coverage(pbf, bbox)

    xml = work_dir / "drive_ways_bbox.osm"
    n = filter_drive_ways(pbf, xml, bbox)
    if n == 0:
        raise ValueError("No drive ways found inside the bbox; check bbox and extract")

    logger.info(
        "Building graph from filtered OSM XML (simplify, keep largest component)"
    )
    G = ox.graph_from_xml(xml, simplify=True, retain_all=False)
    xml.unlink(missing_ok=True)  # intermediate; the graph is cached as GraphML upstream
    return G
