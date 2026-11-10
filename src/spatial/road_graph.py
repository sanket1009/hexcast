"""
road_graph.py
=============
A compact road network for the healthcare access calculation.

Why:
    The OSMnx GraphML cache for LA plus a 30 km buffer is 520 MB on disk.
    Loading it with ox.load_graphml builds a NetworkX graph that keeps every
    OSM attribute (names, geometries, osmid lists) as Python objects, several
    GB of memory. The Airflow worker ran out of memory doing that.

    The access calculation only needs three things:
        node coordinates   to snap tracts and hospitals to the network
        edge travel times  for the shortest-path search
        connectivity       to find the strongly connected core

    RoadGraph keeps exactly that, as NumPy arrays and a SciPy sparse matrix:
    about 336k nodes and 855k edges in tens of MB. It is saved next to the
    GraphML as two Parquet files and loads in about a second.

Same results as the NetworkX version:
    - Parallel edges keep the fastest one, as NetworkX's Dijkstra does on a
      MultiDiGraph.
    - Snapping uses the haversine distance with OSMnx's Earth radius, as
      ox.distance.nearest_nodes does for an unprojected graph.
    - Dijkstra includes paths exactly at the cutoff, as NetworkX does.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from sklearn.neighbors import BallTree

EARTH_RADIUS_M = 6_371_009  # osmnx.distance.EARTH_RADIUS_M

# SciPy treats a stored 0 as "no edge". A real 0-second edge becomes this.
ZERO_EDGE_SECONDS = 1e-9


@dataclass
class RoadGraph:
    node_ids: np.ndarray  # OSM node id per index
    lon: np.ndarray
    lat: np.ndarray
    travel_time: csr_matrix  # [from, to] -> seconds, fastest parallel edge

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_edges(self) -> int:
        return self.travel_time.nnz

    # -- build, save, load ---------------------------------------------------

    @classmethod
    def from_networkx(cls, G) -> "RoadGraph":
        """From an OSMnx graph with node x/y and edge travel_time."""
        node_ids = np.fromiter(G.nodes, dtype=np.int64, count=G.number_of_nodes())
        index = {n: i for i, n in enumerate(node_ids)}
        lon = np.array([G.nodes[n]["x"] for n in node_ids], dtype=float)
        lat = np.array([G.nodes[n]["y"] for n in node_ids], dtype=float)

        edges = pd.DataFrame(
            [
                (index[u], index[v], float(d["travel_time"]))
                for u, v, d in G.edges(data=True)
                if u != v  # self-loops never shorten a path
            ],
            columns=["u", "v", "seconds"],
        )
        return cls(node_ids, lon, lat, _to_csr(edges, len(node_ids)))

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"node_id": self.node_ids, "lon": self.lon, "lat": self.lat}
        ).to_parquet(directory / "nodes.parquet", index=False)
        coo = self.travel_time.tocoo()
        pd.DataFrame({"u": coo.row, "v": coo.col, "seconds": coo.data}).to_parquet(
            directory / "edges.parquet", index=False
        )

    @classmethod
    def load(cls, directory: Path) -> "RoadGraph":
        nodes = pd.read_parquet(directory / "nodes.parquet")
        edges = pd.read_parquet(directory / "edges.parquet")
        return cls(
            nodes["node_id"].to_numpy(),
            nodes["lon"].to_numpy(),
            nodes["lat"].to_numpy(),
            _to_csr(edges, len(nodes)),
        )

    # -- what healthcare_access needs ----------------------------------------

    def strong_core(self) -> np.ndarray:
        """Indices of the nodes in the largest strongly connected component."""
        _, labels = connected_components(
            self.travel_time, directed=True, connection="strong"
        )
        largest = np.bincount(labels).argmax()
        return np.flatnonzero(labels == largest)

    def nearest(
        self, lons: np.ndarray, lats: np.ndarray, among: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Nearest node index for each point (optionally only among `among`),
        and the great-circle distance to it in meters.
        """
        candidates = np.arange(self.n_nodes) if among is None else np.asarray(among)
        tree = BallTree(
            np.radians(np.column_stack([self.lat[candidates], self.lon[candidates]])),
            metric="haversine",
        )
        dist, pos = tree.query(
            np.radians(np.column_stack([np.asarray(lats), np.asarray(lons)])), k=1
        )
        return candidates[pos[:, 0]], dist[:, 0] * EARTH_RADIUS_M

    def travel_times_to(self, target: int, cutoff_seconds: float) -> dict[int, float]:
        """
        Drive time in seconds from every node that can reach `target` within
        the cutoff, as {node index: seconds}.

        Runs Dijkstra from the target on the reversed graph (the transpose):
        the distance from H to N there equals the drive from N to H on the
        real network, with one-way streets respected.
        """
        reversed_graph = self._reversed()
        seconds = dijkstra(
            reversed_graph, directed=True, indices=target, limit=cutoff_seconds
        )
        reached = np.flatnonzero(np.isfinite(seconds))
        return dict(zip(reached.tolist(), seconds[reached].tolist(), strict=True))

    def _reversed(self) -> csr_matrix:
        if not hasattr(self, "_reversed_cache"):
            self._reversed_cache = self.travel_time.transpose().tocsr()
        return self._reversed_cache


def _to_csr(edges: pd.DataFrame, n_nodes: int) -> csr_matrix:
    """Edge list -> sparse matrix, keeping the fastest of any parallel edges."""
    fastest = edges.groupby(["u", "v"], as_index=False)["seconds"].min()
    seconds = fastest["seconds"].to_numpy(dtype=float)
    seconds[seconds <= 0] = ZERO_EDGE_SECONDS
    return csr_matrix(
        (seconds, (fastest["u"].to_numpy(), fastest["v"].to_numpy())),
        shape=(n_nodes, n_nodes),
    )
