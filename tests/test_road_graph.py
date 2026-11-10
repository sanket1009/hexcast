"""
RoadGraph must give the same answers as the NetworkX/OSMnx code it replaced:
same strongly connected core, same nearest nodes and distances, same travel
times. Checked on small hand-made and random graphs.
"""

import networkx as nx
import numpy as np
import pytest

from src.spatial.road_graph import EARTH_RADIUS_M, RoadGraph


def _random_osm_like_graph(n=300, extra_edges=900, seed=1) -> nx.MultiDiGraph:
    rng = np.random.default_rng(seed)
    G = nx.MultiDiGraph()
    for i in range(n):
        G.add_node(1000 + i, x=-118.5 + rng.random() * 0.5, y=33.8 + rng.random() * 0.5)
    nodes = list(G.nodes)
    for _ in range(extra_edges):
        u, v = rng.choice(nodes, 2, replace=False)
        G.add_edge(u, v, travel_time=float(rng.integers(5, 120)))
        if rng.random() < 0.1:  # parallel edge, sometimes faster
            G.add_edge(u, v, travel_time=float(rng.integers(5, 120)))
    return G


def test_travel_times_match_networkx_on_reversed_graph():
    G = _random_osm_like_graph()
    graph = RoadGraph.from_networkx(G)
    cutoff = 300.0
    for target_index in (0, 17, 150):
        target = graph.node_ids[target_index]
        expected = nx.single_source_dijkstra_path_length(
            G.reverse(copy=False), source=target, cutoff=cutoff, weight="travel_time"
        )
        got = graph.travel_times_to(target_index, cutoff)
        got_by_id = {int(graph.node_ids[i]): s for i, s in got.items()}
        assert got_by_id.keys() == expected.keys()
        for node, seconds in expected.items():
            assert got_by_id[node] == pytest.approx(seconds)


def test_cutoff_is_inclusive_and_parallel_edges_keep_the_fastest():
    G = nx.MultiDiGraph()
    for n in (1, 2, 3):
        G.add_node(n, x=0.0, y=0.0)
    G.add_edge(1, 2, travel_time=50.0)
    G.add_edge(1, 2, travel_time=30.0)  # faster parallel edge
    G.add_edge(2, 3, travel_time=30.0)
    graph = RoadGraph.from_networkx(G)
    # drive times to node 3: from 2 is 30 s, from 1 is 60 s (via the 30 s edge)
    got = graph.travel_times_to(2, cutoff_seconds=60.0)
    assert {int(graph.node_ids[i]): s for i, s in got.items()} == {
        3: 0.0,
        2: 30.0,
        1: 60.0,
    }


def test_zero_second_edges_are_kept():
    G = nx.MultiDiGraph()
    G.add_node(1, x=0.0, y=0.0)
    G.add_node(2, x=0.0, y=0.0)
    G.add_edge(1, 2, travel_time=0.0)
    graph = RoadGraph.from_networkx(G)
    assert set(graph.travel_times_to(1, 10.0)) == {0, 1}


def test_strong_core_matches_networkx():
    G = _random_osm_like_graph(seed=7)
    graph = RoadGraph.from_networkx(G)
    expected = max(nx.strongly_connected_components(G), key=len)
    assert set(graph.node_ids[graph.strong_core()].tolist()) == expected


def test_nearest_uses_haversine_meters_and_respects_candidates():
    G = nx.MultiDiGraph()
    G.add_node(1, x=-118.0, y=34.0)
    G.add_node(2, x=-118.01, y=34.0)
    graph = RoadGraph.from_networkx(G)

    idx, dist = graph.nearest(np.array([-118.0]), np.array([34.0]))
    assert graph.node_ids[idx[0]] == 1
    assert dist[0] == pytest.approx(0.0, abs=1e-6)

    idx, dist = graph.nearest(np.array([-118.0]), np.array([34.0]), among=np.array([1]))
    assert graph.node_ids[idx[0]] == 2
    expected = EARTH_RADIUS_M * np.radians(0.01) * np.cos(np.radians(34.0))
    assert dist[0] == pytest.approx(expected, rel=1e-4)


def test_save_and_load_round_trip(tmp_path):
    graph = RoadGraph.from_networkx(_random_osm_like_graph())
    graph.save(tmp_path / "compact")
    loaded = RoadGraph.load(tmp_path / "compact")
    assert np.array_equal(loaded.node_ids, graph.node_ids)
    assert (loaded.travel_time != graph.travel_time).nnz == 0
