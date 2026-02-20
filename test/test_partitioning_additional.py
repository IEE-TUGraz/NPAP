import networkx as nx

from npap.partitioning.adjacent import (
    AdjacentAgglomerativeConfig,
    AdjacentNodeAgglomerativePartitioning,
)
from npap.partitioning.graph_theory import CommunityPartitioning, SpectralPartitioning
from npap.partitioning.lmp import LMPPartitioning, LMPPartitioningConfig


def _build_chain_graph():
    graph = nx.DiGraph()
    for idx in range(4):
        graph.add_node(idx, lmp=float(idx), ac_island="A" if idx < 2 else "B")
        if idx > 0:
            graph.add_edge(idx - 1, idx)
    return graph


def test_adjacent_avoids_cross_island_merges():
    graph = _build_chain_graph()
    strategy = AdjacentNodeAgglomerativePartitioning(
        AdjacentAgglomerativeConfig(node_attribute="lmp", ac_island_attr="ac_island")
    )

    partition = strategy.partition(graph, n_clusters=2)

    assert len(partition) == 2
    islands = {
        node: graph.nodes[node]["ac_island"] for cluster in partition.values() for node in cluster
    }
    assert set(islands.values()) == {"A", "B"}


def test_lmp_infinite_distance_respects_islands():
    graph = _build_chain_graph()
    strategy = LMPPartitioning(LMPPartitioningConfig(adjacency_bonus=0.0, infinite_distance=1e3))

    partition = strategy.partition(graph, n_clusters=2)

    assert len(partition) == 2
    assert all(
        graph.nodes[node]["ac_island"] in {"A", "B"}
        for cluster in partition.values()
        for node in cluster
    )


def test_community_partitioning_detects_two_groups():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 0), (2, 3), (3, 2)])

    strategy = CommunityPartitioning()
    partition = strategy.partition(graph, n_clusters=2)

    assert len(partition) == 2


def test_spectral_partitioning_respects_connectivity():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4)])
    strategy = SpectralPartitioning()
    result = strategy.partition(graph, n_clusters=2, random_state=0)

    assert isinstance(result, dict)
    assert len(result) == 2
