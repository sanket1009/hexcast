"""
healthcare_access.py
====================
Compute 2SFCA (Two-Step Floating Catchment Area) healthcare
accessibility scores for each census tract in the configured city.

2SFCA measures how accessible emergency medical care is from each
tract, accounting for:
  - Real road network travel time (not straight-line distance)
  - Hospital capacity (bed count as proxy for supply)
  - Population demand on each hospital
  - Gaussian distance decay (closer hospitals weighted more)

Method: Enhanced 2SFCA with Gaussian decay
  Step 1: For each hospital, compute supply-to-demand ratio
          (beds / population of all tracts within catchment)
  Step 2: For each tract, sum supply-to-demand ratios of all
          hospitals within catchment, weighted by travel time

Input:
    BigQuery: tract_geometries (centroids + population)
    Table: hospitals (locations, beds, ED status; src/ingestion/hospitals.py)
    OSMnx: road network for city bounding box

Output:
    BigQuery table: healthcare_access (one row per tract)
    Columns: geoid, access_score_er, access_score_primary,
             nearest_er_minutes, nearest_er_name,
             er_count_30min, access_percentile,
             week_start_date, city_fips

Usage:
    from src.spatial.healthcare_access import compute_2sfca
    df = compute_2sfca(config)
"""

from datetime import datetime, timedelta
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from src.utils.bigquery import read_query, write_dataframe
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Gaussian decay bandwidth in minutes
# Controls how quickly accessibility weight falls off with distance
# At d=bandwidth, weight = exp(-0.5) ≈ 0.61
# At d=2×bandwidth, weight = exp(-2) ≈ 0.14
GAUSSIAN_BANDWIDTH = 15.0

# Default travel speed if OSM has no speed limit tagged
DEFAULT_SPEED_KMH = 40.0


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def compute_2sfca(
    config: dict[str, Any],
    week_start: datetime | None = None,
) -> pd.DataFrame:
    """
    Compute Enhanced 2SFCA healthcare accessibility scores.

    Parameters
    ----------
    config : dict
        Merged pipeline config.
    week_start : datetime, optional
        Week to compute access for. Defaults to most recent Monday.

    Returns
    -------
    pd.DataFrame
        One row per tract with 2SFCA accessibility scores.
    """
    if week_start is None:
        today = datetime.now()
        week_start = today - timedelta(days=today.weekday())
        week_start = week_start.replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    full_fips = config["full_fips"]
    week_str = week_start.strftime("%Y-%m-%d")
    catchment_minutes = config["healthcare"]["er_catchment_minutes"]

    logger.info(
        f"Computing 2SFCA healthcare access for {config['city_name']} "
        f"(catchment: {catchment_minutes} min)"
    )

    # Step 1 — Load tract centroids and population from BigQuery
    tracts_df = _load_tract_data(config)
    logger.info(f"Loaded {len(tracts_df):,} tract centroids")

    # Step 2 — Load hospital locations from BigQuery
    hospitals_df = _load_hospital_data(config)
    logger.info(f"Loaded {len(hospitals_df)} hospitals")

    # Step 3 — Download OSMnx road network for city
    # This is the most time-consuming step — 5-15 minutes for LA County
    G = _download_road_network(config)
    logger.info(
        f"Downloaded road network: "
        f"{G.number_of_nodes():,} nodes, "
        f"{G.number_of_edges():,} edges"
    )

    # Step 4 — Snap points to nearest network nodes
    # OSMnx travel time is computed between network nodes
    # We find the nearest node to each tract centroid and hospital
    tract_nodes = _snap_to_network(
        G,
        tracts_df["centroid_lon"].values,
        tracts_df["centroid_lat"].values,
        label="tract centroids",
    )

    hospital_nodes = _snap_to_network(
        G,
        hospitals_df["longitude"].values,
        hospitals_df["latitude"].values,
        label="hospitals",
    )

    # Step 5 — Compute travel times between all tract-hospital pairs
    # This uses Dijkstra's shortest path algorithm on the road network
    travel_times = _compute_travel_times(
        G=G,
        origin_nodes=tract_nodes,
        destination_nodes=hospital_nodes,
        catchment_minutes=catchment_minutes,
        tracts_df=tracts_df,
        hospitals_df=hospitals_df,
    )
    logger.info(
        f"Computed travel times: "
        f"{len(travel_times):,} tract-hospital pairs within catchment"
    )

    # Step 6 — 2SFCA Step 1: compute supply-to-demand ratio per hospital
    hospital_ratios = _compute_hospital_ratios(
        travel_times=travel_times,
        hospitals_df=hospitals_df,
        tracts_df=tracts_df,
        catchment_minutes=catchment_minutes,
    )
    logger.info(
        f"Hospital supply-to-demand ratios: "
        f"mean={hospital_ratios['ratio'].mean():.6f}"
    )

    # Step 7 — 2SFCA Step 2: sum ratios for each tract
    access_df = _compute_tract_access(
        travel_times=travel_times,
        hospital_ratios=hospital_ratios,
        tracts_df=tracts_df,
        catchment_minutes=catchment_minutes,
    )

    # Step 8 — Add nearest ER info and percentile rank
    access_df = _add_nearest_er_info(
        access_df=access_df,
        travel_times=travel_times,
        hospitals_df=hospitals_df,
    )

    access_df["access_percentile"] = (
        access_df["access_score_er"]
        .rank(pct=True)
        .mul(100)
        .round(1)
    )

    # Step 9 — Add metadata
    access_df["week_start_date"] = week_start.date()
    access_df["city_fips"] = full_fips

    # Step 10 — Validate and write to BigQuery
    _validate_output(access_df, config)
    write_dataframe(access_df, "healthcare_access", config)

    logger.info(
        f"2SFCA complete: {len(access_df):,} tracts, "
        f"mean nearest ER: {access_df['nearest_er_minutes'].mean():.1f} min, "
        f"zero-access tracts: {(access_df['access_score_er'] == 0).sum()}"
    )

    return access_df


# ---------------------------------------------------------------------------
# Private functions
# ---------------------------------------------------------------------------

def _load_tract_data(config: dict[str, Any]) -> pd.DataFrame:
    """Load tract centroids and population from BigQuery."""
    df = read_query(
        f"""
        SELECT geoid, centroid_lat, centroid_lon, population
        FROM {{dataset}}.tract_geometries
        WHERE city_fips = '{config["full_fips"]}'
            AND centroid_lat IS NOT NULL
            AND centroid_lon IS NOT NULL
        ORDER BY geoid
        """,
        config,
    )
    # Fill missing population with city median
    # Some tracts have NaN population from ACS gaps
    median_pop = df["population"].median()
    df["population"] = df["population"].fillna(median_pop)
    return df


def _load_hospital_data(config: dict[str, Any]) -> pd.DataFrame:
    """
    Load hospital locations and bed counts.

    With healthcare.ed_only (default true), only hospitals with a licensed
    on-site emergency department count as supply: heat illness and
    smoke-related emergencies are treated through EDs, so a rehab or
    long-term care hospital's beds shouldn't raise a tract's ER access.
    """
    ed_filter = "AND has_ed" if config["healthcare"].get("ed_only", True) else ""
    df = read_query(
        f"""
        SELECT hospital_id, name, latitude, longitude, beds
        FROM {{dataset}}.hospitals
        WHERE city_fips = '{config["full_fips"]}'
            AND latitude IS NOT NULL
            AND longitude IS NOT NULL
            {ed_filter}
        ORDER BY hospital_id
        """,
        config,
    )
    # Guard against missing or non-positive bed counts: fill with the median
    df["beds"] = pd.to_numeric(df["beds"], errors="coerce")
    df.loc[df["beds"] <= 0, "beds"] = np.nan
    median_beds = df["beds"].median()
    df["beds"] = df["beds"].fillna(median_beds)
    return df


def _download_road_network(config: dict[str, Any]) -> nx.MultiDiGraph:
    """
    Download OSMnx driveable road network for city bounding box.

    OSMnx downloads the road network from OpenStreetMap, adds
    travel speed and time attributes, and returns a NetworkX
    directed multigraph where:
      - Nodes = intersections and dead ends
      - Edges = road segments with travel time in seconds

    We add travel time to edges using speed limits from OSM tags.
    Where speed limits are missing, we use the default from config.

    Parameters
    ----------
    config : dict
        Pipeline config with bbox and network settings.

    Returns
    -------
    nx.MultiDiGraph
        Road network graph with travel_time edge attribute (seconds).
    """
    bbox = config["bbox"]
    network_type = config["osmnx"]["network_type"]
    default_speed = config["osmnx"]["speed_kmh_default"]

    logger.info(
        f"Downloading OSMnx {network_type} network for bbox: "
        f"({bbox['south']:.3f}, {bbox['west']:.3f}) to "
        f"({bbox['north']:.3f}, {bbox['east']:.3f})"
    )

    # Download road network within bounding box
    # network_type='drive' = driveable roads only
    # retain_all=False = only keep largest connected component
    #   (removes isolated road fragments that can't be reached)
    G = ox.graph_from_bbox(
        bbox["north"],
        bbox["south"],
        bbox["east"],
        bbox["west"],
        network_type=network_type,
        retain_all=False,
    )

    # Add speed (km/h) to edges from OSM maxspeed tags
    # ox.add_edge_speeds fills missing speeds with highway-type defaults
    G = ox.add_edge_speeds(G, fallback=default_speed)

    # Add travel time (seconds) to edges
    # travel_time = edge_length (meters) / speed (m/s)
    G = ox.add_edge_travel_times(G)

    logger.info(
        f"Road network ready: "
        f"{G.number_of_nodes():,} nodes, "
        f"{G.number_of_edges():,} edges"
    )

    return G


def _snap_to_network(
    G: nx.MultiDiGraph,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    label: str = "points",
) -> list[int]:
    """
    Find nearest network node for each point.

    OSMnx travel times are computed between network nodes
    (intersections). We snap each tract centroid and hospital
    to its nearest node before computing travel times.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Road network graph.
    longitudes : np.ndarray
        Longitude coordinates of points to snap.
    latitudes : np.ndarray
        Latitude coordinates of points to snap.
    label : str
        Description for logging.

    Returns
    -------
    list of int
        Nearest network node ID for each input point.
    """
    logger.debug(f"Snapping {len(longitudes)} {label} to road network")

    nodes = ox.nearest_nodes(G, longitudes, latitudes)

    logger.debug(f"Snapped {len(nodes)} {label} to network nodes")
    return list(nodes)


def _compute_travel_times(
    G: nx.MultiDiGraph,
    origin_nodes: list[int],
    destination_nodes: list[int],
    catchment_minutes: int,
    tracts_df: pd.DataFrame,
    hospitals_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute travel times from each tract to each hospital within catchment.

    Uses Dijkstra's shortest path algorithm from each hospital node
    to all reachable nodes within the catchment threshold.

    We compute from hospitals outward (not from tracts inward) because:
    - There are far fewer hospitals (76) than tracts (2,300)
    - Each hospital-centered Dijkstra covers all tracts at once
    - This is O(H × Dijkstra) vs O(T × Dijkstra) — much faster

    Parameters
    ----------
    G : nx.MultiDiGraph
        Road network with travel_time edge attributes (seconds).
    origin_nodes : list of int
        Network nodes for tract centroids.
    destination_nodes : list of int
        Network nodes for hospitals.
    catchment_minutes : int
        Maximum travel time threshold in minutes.
    tracts_df : pd.DataFrame
        Tract data (for mapping nodes back to GEOIDs).
    hospitals_df : pd.DataFrame
        Hospital data (for mapping nodes back to hospital IDs).

    Returns
    -------
    pd.DataFrame
        One row per reachable tract-hospital pair with columns:
        geoid, hospital_id, travel_time_minutes, decay_weight
    """
    catchment_seconds = catchment_minutes * 60
    records = []

    # Create lookup: network node → tract geoid
    # Multiple tracts may snap to the same node — we handle this
    node_to_tracts = {}
    for i, node in enumerate(origin_nodes):
        if node not in node_to_tracts:
            node_to_tracts[node] = []
        node_to_tracts[node].append(tracts_df.iloc[i]["geoid"])

    logger.info(
        f"Computing shortest paths from {len(destination_nodes)} hospitals "
        f"(catchment: {catchment_minutes} min)"
    )

    for h_idx, hosp_node in enumerate(destination_nodes):
        hospital_id = hospitals_df.iloc[h_idx]["hospital_id"]

        if h_idx % 10 == 0:
            logger.debug(
                f"Processing hospital {h_idx + 1}/{len(destination_nodes)}: "
                f"{hospitals_df.iloc[h_idx]['name']}"
            )

        try:
            # Dijkstra's shortest path from this hospital to all reachable nodes
            # cutoff=catchment_seconds: stop exploring beyond threshold
            # weight='travel_time': edge attribute to minimize
            lengths = nx.single_source_dijkstra_path_length(
                G,
                source=hosp_node,
                cutoff=catchment_seconds,
                weight="travel_time",
            )

            # For each reachable node within catchment
            for node, travel_time_sec in lengths.items():
                if node not in node_to_tracts:
                    continue

                travel_time_min = travel_time_sec / 60.0

                # Gaussian decay weight
                # Weight = 1.0 at same location, decreasing smoothly
                # to near-zero at catchment boundary
                decay_weight = np.exp(
                    -(travel_time_min ** 2) /
                    (2 * GAUSSIAN_BANDWIDTH ** 2)
                )

                # Add record for each tract at this node
                for geoid in node_to_tracts[node]:
                    records.append({
                        "geoid": geoid,
                        "hospital_id": hospital_id,
                        "travel_time_minutes": round(travel_time_min, 2),
                        "decay_weight": round(decay_weight, 6),
                    })

        except nx.NetworkXError as e:
            logger.warning(
                f"Dijkstra failed for hospital {hospital_id}: {e}. "
                f"Skipping."
            )
            continue

    if not records:
        raise ValueError(
            "No tract-hospital pairs found within catchment. "
            "Check road network coverage and hospital locations."
        )

    df = pd.DataFrame(records)

    # Remove duplicate tract-hospital pairs
    # Can occur when multiple tracts snap to the same network node
    df = df.drop_duplicates(subset=["geoid", "hospital_id"])

    return df


def _compute_hospital_ratios(
    travel_times: pd.DataFrame,
    hospitals_df: pd.DataFrame,
    tracts_df: pd.DataFrame,
    catchment_minutes: int,
) -> pd.DataFrame:
    """
    2SFCA Step 1: compute supply-to-demand ratio per hospital.

    For each hospital j:
        R_j = Beds_j / Σ (Population_i × decay_weight_ij)
              for all tracts i within catchment of j

    The decay weight means nearby tracts contribute more to
    the hospital's demand than far tracts — reflecting that
    most patients come from nearby neighborhoods.

    Parameters
    ----------
    travel_times : pd.DataFrame
        Tract-hospital pairs within catchment.
    hospitals_df : pd.DataFrame
        Hospital data with bed counts.
    tracts_df : pd.DataFrame
        Tract data with population.
    catchment_minutes : int
        Catchment threshold in minutes.

    Returns
    -------
    pd.DataFrame
        Hospital supply-to-demand ratios with columns:
        hospital_id, beds, weighted_demand, ratio
    """
    # Merge population onto travel times
    tt_pop = travel_times.merge(
        tracts_df[["geoid", "population"]],
        on="geoid",
        how="left",
    )

    # Weighted demand = population × decay weight
    tt_pop["weighted_demand"] = tt_pop["population"] * tt_pop["decay_weight"]

    # Sum weighted demand per hospital
    demand_by_hosp = (
        tt_pop.groupby("hospital_id")["weighted_demand"]
        .sum()
        .reset_index()
        .rename(columns={"weighted_demand": "total_weighted_demand"})
    )

    # Merge bed counts
    ratios = demand_by_hosp.merge(
        hospitals_df[["hospital_id", "beds"]],
        on="hospital_id",
        how="left",
    )

    # Compute supply-to-demand ratio
    # Add small epsilon to prevent division by zero
    ratios["ratio"] = ratios["beds"] / (
        ratios["total_weighted_demand"] + 1e-10
    )

    return ratios


def _compute_tract_access(
    travel_times: pd.DataFrame,
    hospital_ratios: pd.DataFrame,
    tracts_df: pd.DataFrame,
    catchment_minutes: int,
) -> pd.DataFrame:
    """
    2SFCA Step 2: sum hospital ratios for each tract.

    For each tract i:
        A_i = Σ (R_j × decay_weight_ij)
              for all hospitals j within catchment of i

    Tracts with many nearby high-ratio hospitals have high
    accessibility scores. Tracts with no hospitals within
    catchment have score = 0 (healthcare desert).

    Parameters
    ----------
    travel_times : pd.DataFrame
        Tract-hospital pairs with decay weights.
    hospital_ratios : pd.DataFrame
        Hospital supply-to-demand ratios.
    tracts_df : pd.DataFrame
        All tracts (to include zero-access tracts in output).
    catchment_minutes : int
        Catchment threshold in minutes.

    Returns
    -------
    pd.DataFrame
        One row per tract with access_score_er and er_count_30min.
    """
    # Merge hospital ratios onto travel times
    tt_ratios = travel_times.merge(
        hospital_ratios[["hospital_id", "ratio"]],
        on="hospital_id",
        how="left",
    )

    # Weighted ratio = ratio × decay weight
    tt_ratios["weighted_ratio"] = tt_ratios["ratio"] * tt_ratios["decay_weight"]

    # Sum weighted ratios per tract — this is the 2SFCA access score
    access_by_tract = (
        tt_ratios.groupby("geoid")
        .agg(
            access_score_er=("weighted_ratio", "sum"),
            er_count_30min=("hospital_id", "nunique"),
        )
        .reset_index()
    )

    # Merge back to all tracts — tracts outside all catchments get score=0
    all_tracts = tracts_df[["geoid"]].copy()
    access_df = all_tracts.merge(
        access_by_tract,
        on="geoid",
        how="left",
    )

    # Fill NaN with 0 — no hospital within catchment = zero access
    access_df["access_score_er"] = access_df["access_score_er"].fillna(0)
    access_df["er_count_30min"] = access_df["er_count_30min"].fillna(0).astype(int)

    # Log healthcare desert count
    desert_count = (access_df["access_score_er"] == 0).sum()
    if desert_count > 0:
        logger.warning(
            f"{desert_count} tracts have zero ER access within "
            f"{catchment_minutes} minutes — healthcare deserts"
        )

    return access_df


def _add_nearest_er_info(
    access_df: pd.DataFrame,
    travel_times: pd.DataFrame,
    hospitals_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add nearest ER travel time and name to each tract.

    The nearest ER travel time is the most interpretable
    accessibility metric for non-technical audiences —
    "38 minutes to the nearest ER" is clearer than a
    2SFCA score of 0.0042.

    Parameters
    ----------
    access_df : pd.DataFrame
        Tract access scores.
    travel_times : pd.DataFrame
        All tract-hospital travel times.
    hospitals_df : pd.DataFrame
        Hospital names.

    Returns
    -------
    pd.DataFrame
        access_df with nearest_er_minutes and nearest_er_name added.
    """
    # Find nearest hospital per tract (minimum travel time)
    nearest = (
        travel_times.sort_values("travel_time_minutes")
        .groupby("geoid")
        .first()
        .reset_index()[["geoid", "hospital_id", "travel_time_minutes"]]
        .rename(columns={"travel_time_minutes": "nearest_er_minutes"})
    )

    # Add hospital name
    nearest = nearest.merge(
        hospitals_df[["hospital_id", "name"]].rename(
            columns={"name": "nearest_er_name"}
        ),
        on="hospital_id",
        how="left",
    )

    # Merge onto access df
    access_df = access_df.merge(
        nearest[["geoid", "nearest_er_minutes", "nearest_er_name"]],
        on="geoid",
        how="left",
    )

    # Tracts with no hospital in catchment get large travel time
    # We use 999 to indicate "no hospital found within catchment"
    access_df["nearest_er_minutes"] = access_df["nearest_er_minutes"].fillna(999)
    access_df["nearest_er_name"] = access_df["nearest_er_name"].fillna("None within catchment")

    return access_df


def _validate_output(df: pd.DataFrame, config: dict[str, Any]) -> None:
    """Validate 2SFCA output before BigQuery write."""
    if df.empty:
        raise ValueError("Healthcare access output is empty")

    # Check all tracts have scores
    if len(df) < 100:
        raise ValueError(
            f"Only {len(df)} tracts in output — expected at least 100"
        )

    # Check no negative scores
    negative = (df["access_score_er"] < 0).sum()
    if negative > 0:
        raise ValueError(f"{negative} tracts have negative access scores")

    # Log summary statistics
    desert_pct = (df["access_score_er"] == 0).mean() * 100
    mean_nearest = df["nearest_er_minutes"].replace(999, np.nan).mean()

    logger.info(
        f"2SFCA validation passed: {len(df):,} tracts, "
        f"{desert_pct:.1f}% healthcare deserts, "
        f"mean nearest ER: {mean_nearest:.1f} min"
    )