"""
healthcare_access.py
====================
Compute Enhanced 2SFCA (Two-Step Floating Catchment Area) emergency
care accessibility for each census tract in the configured city.

2SFCA measures how accessible emergency care is from each tract,
accounting for:
  - Real road network travel time (not straight-line distance)
  - Hospital capacity (licensed beds as supply)
  - Population demand competing for each hospital
  - Gaussian travel-time decay (closer hospitals weighted more)

Method: Enhanced 2SFCA with Gaussian decay
  Step 1: For each hospital j, supply-to-demand ratio
          R_j = beds_j / sum_i(pop_i * w_ij)   over tracts within catchment
  Step 2: For each tract i, access
          A_i = sum_j(R_j * w_ij)              over hospitals within catchment
  Reported per 1,000 residents.

Edge effect handling:
  Hospitals and tract populations from config neighbor_counties within
  healthcare.buffer_km of the city bbox are included, and the road
  network covers the same buffered area. Border tracts then see the ERs
  across the county line they really use, and those ERs carry the demand
  from their own side. Only city tracts are scored. Residual bias: the
  outermost buffer hospitals miss demand from beyond the buffer, which
  slightly over-credits their supply; Gaussian decay keeps this small.

Travel time:
  Measured tract -> hospital (the patient's trip) by running Dijkstra
  from each hospital on the reversed road graph, so one-way streets and
  freeway ramps are respected. Free-flow speeds from OSM tags; no traffic.

Unscored tracts (NaN access):
  - Unreachable: centroid farther than healthcare.snap_max_m from any
    road in the network (Catalina/San Clemente islands, water-only tracts)
  - No residents: population 0 or missing
  A reachable, populated tract with no ER within the catchment scores 0:
  a true ER desert.

Input:
    Table: tract_geometries (city tracts: centroids, population)
    Table: buffer_tracts (neighbor tracts: centroids, population)
    Table: hospitals (city + buffer hospitals; src/ingestion/hospitals.py)
    OSMnx: drive network for the buffered bbox (cached as GraphML)

Output:
    Table: healthcare_access (one row per city tract per week)
    Columns: geoid, access_score_er, er_count_30min, nearest_er_minutes,
             nearest_er_name, nearest_er_in_city, er_in_catchment,
             reachable, snap_distance_m, scored, access_percentile,
             week_start_date, city_fips

Usage:
    from src.spatial.healthcare_access import compute_2sfca
    df = compute_2sfca(config)
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from src.utils.bigquery import read_query, table_exists, write_dataframe
from src.utils.config_loader import get_buffered_bbox
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Gaussian decay bandwidth in minutes
# At d=bandwidth, weight = exp(-0.5) ≈ 0.61
# At d=2×bandwidth, weight = exp(-2) ≈ 0.14
GAUSSIAN_BANDWIDTH = 15.0

# Access is reported as beds per this many residents
PER_RESIDENTS = 1_000

# Road networks change slowly; reuse the downloaded graph across runs
NETWORK_CACHE_DIR = Path("data") / "processed" / "road_network"

# OSMnx caches each finished Overpass sub-query here, so a retry after a
# server error resumes instead of starting over (kept out of git under data/)
OSM_HTTP_CACHE_DIR = Path("data") / "processed" / "osm_http_cache"


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def compute_2sfca(
    config: dict[str, Any],
    week_start: datetime | None = None,
    refresh_network: bool = False,
) -> pd.DataFrame:
    """
    Compute Enhanced 2SFCA emergency care access for every city tract.

    Parameters
    ----------
    config : dict
        Merged pipeline config.
    week_start : datetime, optional
        Week the output is stamped with. Defaults to most recent Monday.
    refresh_network : bool
        Re-download the road network instead of using the cached GraphML.

    Returns
    -------
    pd.DataFrame
        One row per city tract. See module docstring for columns.
    """
    if week_start is None:
        today = datetime.now()
        week_start = today - timedelta(days=today.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)

    hc = config["healthcare"]
    catchment_minutes = hc["er_catchment_minutes"]
    snap_max_m = hc.get("snap_max_m", 2000.0)

    logger.info(
        f"Computing 2SFCA ER access for {config['city_name']} "
        f"(catchment {catchment_minutes} min, buffer {hc.get('buffer_km', 0)} km)"
    )

    # Step 1 — Demand: city tracts (scored) + buffer tracts (demand only)
    tracts_df = _load_demand_tracts(config)

    # Step 2 — Supply: ED hospitals in the city and buffer
    hospitals_df = _load_hospital_data(config)

    # Step 3 — Road network for the buffered bbox (cached)
    G = _get_road_network(config, refresh=refresh_network)

    # Step 4 — Snap to the network; far-off points are unreachable
    tracts_df["node"], tracts_df["snap_distance_m"] = _snap_to_network(
        G, tracts_df["centroid_lon"].values, tracts_df["centroid_lat"].values
    )
    tracts_df["reachable"] = tracts_df["snap_distance_m"] <= snap_max_m

    hospitals_df["node"], hospitals_df["snap_distance_m"] = _snap_to_network(
        G, hospitals_df["longitude"].values, hospitals_df["latitude"].values
    )
    far = hospitals_df["snap_distance_m"] > snap_max_m
    if far.any():
        logger.warning(
            f"Dropping {int(far.sum())} hospital(s) more than {snap_max_m:.0f} m "
            f"from a road: {hospitals_df.loc[far, 'name'].tolist()}"
        )
        hospitals_df = hospitals_df[~far].reset_index(drop=True)

    city_unreachable = tracts_df[tracts_df["in_city"] & ~tracts_df["reachable"]]
    if len(city_unreachable):
        logger.info(
            f"{len(city_unreachable)} city tracts unreachable by road "
            f"(> {snap_max_m:.0f} m to network): "
            + ", ".join(city_unreachable["geoid"])
        )

    # Step 5 — Travel times tract -> hospital within the catchment
    travel_times = _compute_travel_times(
        G, tracts_df[tracts_df["reachable"]], hospitals_df, catchment_minutes
    )
    del G  # free a few GB before the table work

    # Step 6 — 2SFCA step 1: supply-to-demand ratio per hospital
    hospital_ratios = _compute_hospital_ratios(travel_times, hospitals_df, tracts_df)

    # Step 7 — 2SFCA step 2: access per city tract
    access_df = _compute_tract_access(travel_times, hospital_ratios, tracts_df)

    # Step 8 — Nearest ER (plain-English companion metric)
    access_df = _add_nearest_er_info(access_df, travel_times, hospitals_df)

    scored = access_df["scored"]
    access_df["access_percentile"] = np.nan
    access_df.loc[scored, "access_percentile"] = (
        access_df.loc[scored, "access_score_er"].rank(pct=True).mul(100).round(1)
    )

    access_df["week_start_date"] = week_start.date()
    access_df["city_fips"] = config["full_fips"]

    _validate_output(access_df, catchment_minutes)
    write_dataframe(access_df, "healthcare_access", config)

    return access_df


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _load_demand_tracts(config: dict[str, Any]) -> pd.DataFrame:
    """
    City tracts plus neighbor buffer tracts, with population.

    Missing population counts as 0 demand. Filling it with a median would
    invent residents and inflate the load on nearby hospitals.
    """
    fips = config["full_fips"]
    city = read_query(
        f"""
        SELECT geoid, centroid_lat, centroid_lon, population
        FROM {{dataset}}.tract_geometries
        WHERE city_fips = '{fips}'
            AND centroid_lat IS NOT NULL AND centroid_lon IS NOT NULL
        """,
        config,
    )
    city["in_city"] = True
    parts = [city]

    if table_exists("buffer_tracts", config):
        buf = read_query(
            f"""
            SELECT geoid, centroid_lat, centroid_lon, population
            FROM {{dataset}}.buffer_tracts
            WHERE city_fips = '{fips}'
                AND centroid_lat IS NOT NULL AND centroid_lon IS NOT NULL
            """,
            config,
        )
        buf["in_city"] = False
        parts.append(buf)
    elif config["healthcare"].get("buffer_km", 0) > 0:
        logger.warning(
            "buffer_tracts table missing: buffer hospitals will have no demand "
            "from their own side. Run census_tiger.pull_buffer_tracts first."
        )

    df = pd.concat(parts, ignore_index=True)
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    missing = df["population"].isna() & df["in_city"]
    if missing.any():
        logger.warning(f"{int(missing.sum())} city tracts with missing population: treated as 0")
    df["population"] = df["population"].fillna(0).clip(lower=0)

    logger.info(
        f"Demand: {int(df['in_city'].sum()):,} city tracts "
        f"({df.loc[df['in_city'], 'population'].sum():,.0f} people) + "
        f"{int((~df['in_city']).sum()):,} buffer tracts "
        f"({df.loc[~df['in_city'], 'population'].sum():,.0f} people)"
    )
    return df


def _load_hospital_data(config: dict[str, Any]) -> pd.DataFrame:
    """
    ED hospitals in the city and buffer, with beds.

    With healthcare.ed_only (default true), only hospitals with a licensed
    on-site emergency department count as supply: heat illness and
    smoke-related emergencies are treated through EDs.
    """
    ed_filter = "AND has_ed" if config["healthcare"].get("ed_only", True) else ""
    df = read_query(
        f"""
        SELECT hospital_id, name, latitude, longitude, beds, in_city
        FROM {{dataset}}.hospitals
        WHERE city_fips = '{config["full_fips"]}'
            AND latitude IS NOT NULL AND longitude IS NOT NULL
            {ed_filter}
        """,
        config,
    )
    df["beds"] = pd.to_numeric(df["beds"], errors="coerce")
    bad = ~(df["beds"] > 0)
    if bad.any():
        logger.warning(f"Dropping {int(bad.sum())} hospitals without a positive bed count")
        df = df[~bad]
    df = df.reset_index(drop=True)

    logger.info(
        f"Supply: {int(df['in_city'].sum())} city ED hospitals "
        f"({df.loc[df['in_city'], 'beds'].sum():,.0f} beds) + "
        f"{int((~df['in_city']).sum())} buffer ED hospitals "
        f"({df.loc[~df['in_city'], 'beds'].sum():,.0f} beds)"
    )
    return df


def _get_road_network(config: dict[str, Any], refresh: bool = False) -> nx.MultiDiGraph:
    """
    Drive network for the buffered bbox, with travel_time on every edge.

    Downloaded once from OpenStreetMap and saved as GraphML; later runs
    load the file in seconds. The cache name carries the bbox buffer and
    network type, so changing either downloads a fresh network.
    """
    hc = config["healthcare"]
    buffer_km = hc.get("buffer_km", 0.0)
    network_type = config["osmnx"]["network_type"]
    path = NETWORK_CACHE_DIR / (
        f"{config['full_fips']}_{network_type}_buffer{buffer_km:g}km.graphml"
    )

    if path.exists() and not refresh:
        logger.info(f"Loading cached road network: {path}")
        G = ox.load_graphml(path)
    else:
        west, south, east, north = get_buffered_bbox(config, buffer_km)
        logger.info(
            f"Downloading OSM '{network_type}' network for buffered bbox "
            f"W{west:.3f} S{south:.3f} E{east:.3f} N{north:.3f}. "
            f"First download takes a while (large area)."
        )
        G = _download_with_retry(config, (west, south, east, north), network_type)
        # Speed from OSM maxspeed tags; untagged roads get the mean of their
        # highway type, then the configured fallback
        G = ox.add_edge_speeds(G, fallback=config["osmnx"]["speed_kmh_default"])
        G = ox.add_edge_travel_times(G)

        NETWORK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        ox.save_graphml(G, path)
        logger.info(f"Saved road network to {path}")

    logger.info(f"Road network: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def _download_with_retry(
    config: dict[str, Any],
    bbox: tuple[float, float, float, float],
    network_type: str,
) -> nx.MultiDiGraph:
    """
    Download the drive network from Overpass, retrying on server errors.

    A large area is split into many sub-queries, and the public Overpass
    server refuses work when it is busy (HTTP 502/504, "at capacity").
    Each finished sub-query is cached on disk, so every retry only fetches
    what is still missing. Waits double between attempts, as the server
    asks: 1, 2, 4, 8 minutes.
    """
    osm = config["osmnx"]
    attempts = osm.get("download_retries", 5)

    ox.settings.use_cache = True
    ox.settings.cache_folder = str(OSM_HTTP_CACHE_DIR)
    ox.settings.requests_timeout = 300
    if osm.get("overpass_url"):
        ox.settings.overpass_url = osm["overpass_url"]
    logger.info(f"Overpass server: {ox.settings.overpass_url}")

    for attempt in range(1, attempts + 1):
        try:
            # OSMnx 2.x: bbox=(left, bottom, right, top)
            # retain_all=False keeps the largest connected network and drops
            # fragments no car can reach
            return ox.graph_from_bbox(
                bbox=bbox, network_type=network_type, retain_all=False
            )
        except Exception as err:  # Overpass errors surface as several types
            if attempt == attempts:
                raise
            wait = 60 * 2 ** (attempt - 1)
            cached = len(list(OSM_HTTP_CACHE_DIR.glob("*.json")))
            logger.warning(
                f"Overpass download failed (attempt {attempt}/{attempts}): "
                f"{str(err).splitlines()[0][:160]}. "
                f"{cached} sub-queries cached so far. Retrying in {wait // 60} min..."
            )
            time.sleep(wait)


def _snap_to_network(
    G: nx.MultiDiGraph,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
) -> tuple[list[int], np.ndarray]:
    """
    Nearest network node for each point, and the distance to it in meters.

    The distance is what lets us refuse a snap: an island centroid's
    nearest node is on the mainland coast, kilometers away.
    """
    nodes, dists = ox.distance.nearest_nodes(G, longitudes, latitudes, return_dist=True)
    return list(nodes), np.asarray(dists, dtype=float)


# ---------------------------------------------------------------------------
# Travel times and 2SFCA
# ---------------------------------------------------------------------------

def _compute_travel_times(
    G: nx.MultiDiGraph,
    tracts_df: pd.DataFrame,
    hospitals_df: pd.DataFrame,
    catchment_minutes: int,
) -> pd.DataFrame:
    """
    Tract -> hospital travel times within the catchment.

    One Dijkstra per hospital (far fewer hospitals than tracts), run on
    the REVERSED graph: the distance from hospital H to node N on the
    reversed graph equals the drive from N to H on the real one. That is
    the patient's trip, with one-way streets and ramps respected.

    Returns
    -------
    pd.DataFrame
        geoid, hospital_id, travel_time_minutes, decay_weight
    """
    cutoff = catchment_minutes * 60
    R = G.reverse(copy=False)

    node_to_tracts: dict[int, list[str]] = {}
    for geoid, node in zip(tracts_df["geoid"], tracts_df["node"]):
        node_to_tracts.setdefault(node, []).append(geoid)

    logger.info(f"Dijkstra from {len(hospitals_df)} hospitals (cutoff {catchment_minutes} min)")
    records = []
    for k, (hid, hnode) in enumerate(zip(hospitals_df["hospital_id"], hospitals_df["node"]), 1):
        lengths = nx.single_source_dijkstra_path_length(
            R, source=hnode, cutoff=cutoff, weight="travel_time"
        )
        for node, seconds in lengths.items():
            for geoid in node_to_tracts.get(node, ()):
                records.append((geoid, hid, seconds / 60.0))
        if k % 25 == 0:
            logger.info(f"  {k}/{len(hospitals_df)} hospitals done")

    if not records:
        raise ValueError("No tract-hospital pairs within the catchment. Check network and inputs.")

    df = pd.DataFrame(records, columns=["geoid", "hospital_id", "travel_time_minutes"])
    df["decay_weight"] = np.exp(-(df["travel_time_minutes"] ** 2) / (2 * GAUSSIAN_BANDWIDTH ** 2))
    logger.info(f"{len(df):,} tract-hospital pairs within {catchment_minutes} min")
    return df


def _compute_hospital_ratios(
    travel_times: pd.DataFrame,
    hospitals_df: pd.DataFrame,
    tracts_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    2SFCA step 1: R_j = beds_j / sum_i(pop_i * w_ij).

    Demand comes from city AND buffer tracts, so a hospital near the
    county line carries the load from both sides.
    """
    tt = travel_times.merge(tracts_df[["geoid", "population"]], on="geoid", how="left")
    tt["weighted_demand"] = tt["population"] * tt["decay_weight"]

    ratios = (
        tt.groupby("hospital_id")["weighted_demand"].sum()
        .rename("total_weighted_demand").reset_index()
        .merge(hospitals_df[["hospital_id", "beds", "in_city"]], on="hospital_id", how="left")
    )
    # A hospital with no residents in reach has no ratio (it serves nobody
    # in the model); it is dropped rather than divided by ~0
    ratios = ratios[ratios["total_weighted_demand"] > 0].copy()
    ratios["ratio"] = ratios["beds"] / ratios["total_weighted_demand"]

    logger.info(
        f"Hospital ratios: {len(ratios)} hospitals, median "
        f"{ratios['ratio'].median() * PER_RESIDENTS:.2f} beds per {PER_RESIDENTS:,} weighted residents"
    )
    return ratios


def _compute_tract_access(
    travel_times: pd.DataFrame,
    hospital_ratios: pd.DataFrame,
    tracts_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    2SFCA step 2: A_i = sum_j(R_j * w_ij), for city tracts only.

    scored = reachable and has residents. Scored tracts with no ER in
    reach get 0 (a real desert); unscored tracts get NaN.
    """
    tt = travel_times.merge(hospital_ratios[["hospital_id", "ratio"]], on="hospital_id", how="inner")
    tt["weighted_ratio"] = tt["ratio"] * tt["decay_weight"]

    by_tract = tt.groupby("geoid").agg(
        access_score_er=("weighted_ratio", "sum"),
        er_count_30min=("hospital_id", "nunique"),
    ).reset_index()

    city = tracts_df[tracts_df["in_city"]][["geoid", "population", "reachable", "snap_distance_m"]]
    df = city.merge(by_tract, on="geoid", how="left")

    df["scored"] = df["reachable"] & (df["population"] > 0)
    df["er_count_30min"] = df["er_count_30min"].fillna(0).astype(int)
    df["access_score_er"] = (df["access_score_er"].fillna(0) * PER_RESIDENTS).round(4)
    df.loc[~df["scored"], "access_score_er"] = np.nan
    df["er_in_catchment"] = df["er_count_30min"] > 0
    df["snap_distance_m"] = df["snap_distance_m"].round(0)

    deserts = df["scored"] & ~df["er_in_catchment"]
    logger.info(
        f"Scored {int(df['scored'].sum()):,} of {len(df):,} city tracts; "
        f"{int((~df['reachable']).sum())} unreachable, "
        f"{int((df['reachable'] & (df['population'] <= 0)).sum())} without residents, "
        f"{int(deserts.sum())} with no ER within the catchment"
    )
    return df.drop(columns=["population"])


def _add_nearest_er_info(
    access_df: pd.DataFrame,
    travel_times: pd.DataFrame,
    hospitals_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Nearest ER drive time and name per tract.

    "12 minutes to the nearest ER" is the plain-English companion to the
    2SFCA score. NaN when no ER is within the catchment (not a sentinel
    like 999, which would poison any mean or model that reads it).
    nearest_er_in_city shows how often the closest ER is across the line.
    """
    nearest = (
        travel_times.sort_values("travel_time_minutes")
        .drop_duplicates("geoid")[["geoid", "hospital_id", "travel_time_minutes"]]
        .merge(hospitals_df[["hospital_id", "name", "in_city"]], on="hospital_id", how="left")
        .rename(columns={
            "travel_time_minutes": "nearest_er_minutes",
            "name": "nearest_er_name",
            "in_city": "nearest_er_in_city",
        })
        .drop(columns=["hospital_id"])
    )
    nearest["nearest_er_minutes"] = nearest["nearest_er_minutes"].round(1)
    df = access_df.merge(nearest, on="geoid", how="left")

    # Unscored tracts carry no access information
    df.loc[~df["scored"], ["nearest_er_minutes", "nearest_er_name", "nearest_er_in_city"]] = np.nan

    across = df["scored"] & (df["nearest_er_in_city"] == False)  # noqa: E712
    logger.info(f"{int(across.sum())} scored tracts have their nearest ER in a neighbor county")
    return df


def _validate_output(df: pd.DataFrame, catchment_minutes: int) -> None:
    """Sanity checks before writing."""
    if df.empty or not df["scored"].any():
        raise ValueError("No scored tracts in healthcare access output")
    if df["geoid"].duplicated().any():
        raise ValueError("Duplicate geoids in healthcare access output")
    if (df["access_score_er"] < 0).any():
        raise ValueError("Negative access scores")
    if (df["nearest_er_minutes"] > catchment_minutes + 0.1).any():
        raise ValueError("Nearest ER time beyond the catchment")

    s = df.loc[df["scored"]]
    logger.info(
        f"2SFCA validation passed: {len(s):,} scored tracts, access "
        f"median {s['access_score_er'].median():.2f} beds per {PER_RESIDENTS:,} "
        f"(p5 {s['access_score_er'].quantile(0.05):.2f}, p95 {s['access_score_er'].quantile(0.95):.2f}), "
        f"nearest ER median {s['nearest_er_minutes'].median():.1f} min"
    )
