"""
Property-based tests for partitioning and visualization helpers.
"""

import networkx as nx
from hypothesis import given, settings
from hypothesis import strategies as st

from npap.interfaces import PartitionResult
from npap.managers import PartitioningManager
from npap.partitioning.adjacent import AdjacentAgglomerativeConfig
from npap.visualization import PlotPreset, plot_network


@st.composite
def connected_graph_with_attributes(draw):
    node_count = draw(st.integers(min_value=3, max_value=7))
    nodes = list(range(node_count))

    # Always include a spanning chain so the graph is connected.
    edges = [(i, i + 1) for i in range(node_count - 1)]
    extra_edges = draw(
        st.lists(
            st.tuples(st.integers(0, node_count - 1), st.integers(0, node_count - 1)),
            min_size=0,
            max_size=node_count,
        )
    )
    edges = list(dict.fromkeys(edges + extra_edges))

    graph = nx.DiGraph()
    island_choices = ["A", "B", None]

    for node in nodes:
        graph.add_node(
            node,
            load=draw(st.floats(min_value=0.0, max_value=100.0)),
            ac_island=draw(st.sampled_from(island_choices)),
            lat=draw(st.floats(min_value=-90.0, max_value=90.0)),
            lon=draw(st.floats(min_value=-180.0, max_value=180.0)),
        )

    for u, v in edges:
        if u != v:
            graph.add_edge(u, v)

    n_clusters = draw(st.integers(min_value=1, max_value=node_count))
    return graph, n_clusters


@given(connected_graph_with_attributes())
@settings(max_examples=20)
def test_adjacent_agglomerative_handles_random_graphs(data):
    """Partitioning should return exactly n_clusters for connected graphs."""
    graph, n_clusters = data

    manager = PartitioningManager()
    partition = manager.partition(
        graph,
        "adjacent_agglomerative",
        n_clusters=n_clusters,
        config=AdjacentAgglomerativeConfig(
            node_attribute="load",
            ac_island_attr=None,
        ),
    )

    assert len(partition) == n_clusters
    assert sum(len(nodes) for nodes in partition.values()) == graph.number_of_nodes()


@given(
    st.lists(st.integers(min_value=0, max_value=3), min_size=2, max_size=6),
    st.sampled_from(list(PlotPreset)),
)
@settings(max_examples=15, deadline=None)
def test_plot_network_handles_random_partitions(cluster_assignments, preset):
    """Plotting should accept arbitrary cluster assignments without raising."""
    graph = nx.DiGraph()
    for node_id, cluster_id in enumerate(cluster_assignments):
        graph.add_node(node_id, lat=0.1 * node_id, lon=0.1 * node_id)
    for node_id in range(len(cluster_assignments) - 1):
        graph.add_edge(node_id, node_id + 1)

    partition_map = {}
    for node_id, cluster_id in enumerate(cluster_assignments):
        partition_map.setdefault(cluster_id, []).append(node_id)

    fig = plot_network(
        graph,
        style="clustered",
        partition_map=partition_map,
        preset=preset,
        show=False,
    )

    assert isinstance(fig, object)
    assert len(fig.data) > 0


def test_plot_network_accepts_partition_result_from_manager(simple_digraph):
    """plot_network should accept PartitionResult objects returned by managers."""
    partition_map = {0: [0, 1], 1: [2, 3]}
    partition_result = PartitionResult(
        mapping=partition_map,
        original_graph_hash="hash",
        strategy_name="hypothesis",
        strategy_metadata={},
        n_clusters=len(partition_map),
    )

    fig = plot_network(
        simple_digraph,
        style="clustered",
        partition_result=partition_result,
        show=False,
    )

    assert isinstance(fig, object)
    assert len(fig.data) > 0
