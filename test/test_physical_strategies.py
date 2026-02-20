import networkx as nx

from npap.aggregation.physical_strategies import KronReductionStrategy, PTDFReductionStrategy


def _build_simple_graph() -> tuple[nx.DiGraph, dict[int, list[int]]]:
    graph = nx.DiGraph()
    graph.add_node("A")
    graph.add_node("B")
    graph.add_edge("A", "B", x=0.5)
    partition_map = {0: ["A"], 1: ["B"]}
    return graph, partition_map


def _build_topology() -> nx.DiGraph:
    topology = nx.DiGraph()
    topology.add_nodes_from([0, 1])
    return topology


def test_ptdf_reduction_stores_matrix():
    original, partition_map = _build_simple_graph()
    topology = _build_topology()
    strategy = PTDFReductionStrategy(reactance_property="x")

    aggregated = strategy.aggregate(
        original,
        partition_map,
        topology,
        ["x"],
        parameters=None,
    )

    assert aggregated.graph["reduced_ptdf"]["matrix"].shape[0] >= 0
    assert aggregated.graph["reduced_ptdf"]["nodes"] == [0, 1]
    assert aggregated.has_edge(0, 1) or aggregated.has_edge(1, 0)


def test_kron_reduction_generates_laplacian():
    original, partition_map = _build_simple_graph()
    topology = _build_topology()
    strategy = KronReductionStrategy(reactance_property="x")

    aggregated = strategy.aggregate(
        original,
        partition_map,
        topology,
        ["x"],
        parameters=None,
    )

    laplacian = aggregated.graph["kron_reduced_laplacian"]
    assert laplacian.shape == (2, 2)
    assert aggregated.has_edge(0, 1) or aggregated.has_edge(1, 0)
