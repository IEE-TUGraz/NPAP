"""
Shared pytest fixtures for NPAP test suite.

Provides small, well-defined test graphs that enable deterministic testing
of partitioning and aggregation strategies.
"""

import networkx as nx
import pytest


# =============================================================================
# BASIC TEST GRAPHS
# =============================================================================

@pytest.fixture
def simple_digraph() -> nx.DiGraph:
    """
    Simple 4-node directed graph for basic topology test.

    Structure:
        0 ──→ 1
        │     │
        ↓     ↓
        2 ──→ 3

    All nodes have lat/lon and basic properties.
    All edges have reactance (x) and length properties.
    """
    G = nx.DiGraph()

    # Nodes with properties
    G.add_node(0, lat=0.0, lon=0.0, demand=10.0, name="node_0")
    G.add_node(1, lat=0.0, lon=1.0, demand=20.0, name="node_1")
    G.add_node(2, lat=1.0, lon=0.0, demand=30.0, name="node_2")
    G.add_node(3, lat=1.0, lon=1.0, demand=40.0, name="node_3")

    # Edges with properties (directed: follows arrows above)
    G.add_edge(0, 1, x=0.1, length=100.0, p_max=500.0)
    G.add_edge(0, 2, x=0.2, length=150.0, p_max=600.0)
    G.add_edge(1, 3, x=0.15, length=120.0, p_max=550.0)
    G.add_edge(2, 3, x=0.25, length=180.0, p_max=450.0)

    return G


@pytest.fixture
def geographical_cluster_graph() -> nx.DiGraph:
    """
    6-node graph with two distinct geographical clusters for testing
    geographical partitioning strategies.

    Cluster A (nodes 0,1,2): Located around (0, 0)
    Cluster B (nodes 3,4,5): Located around (10, 10)

    This should naturally partition into 2 clusters with any
    geographical algorithm when n_clusters=2.
    """
    G = nx.DiGraph()

    # Cluster A: nodes near origin
    G.add_node(0, lat=0.0, lon=0.0, voltage=220.0)
    G.add_node(1, lat=0.1, lon=0.1, voltage=220.0)
    G.add_node(2, lat=0.05, lon=-0.05, voltage=220.0)

    # Cluster B: nodes far from origin
    G.add_node(3, lat=10.0, lon=10.0, voltage=220.0)
    G.add_node(4, lat=10.1, lon=10.1, voltage=220.0)
    G.add_node(5, lat=10.05, lon=9.95, voltage=220.0)

    # Edges within clusters
    G.add_edge(0, 1, x=0.1)
    G.add_edge(1, 2, x=0.1)
    G.add_edge(0, 2, x=0.15)

    G.add_edge(3, 4, x=0.1)
    G.add_edge(4, 5, x=0.1)
    G.add_edge(3, 5, x=0.15)

    # Edge between clusters (long transmission line)
    G.add_edge(2, 3, x=1.0)

    return G


@pytest.fixture
def electrical_graph() -> nx.DiGraph:
    """
    Graph designed for electrical distance partitioning test.

    Structure (star topology with center node 0):
        1 ←─┐
            │
        2 ←─0─→ 3
            │
        4 ←─┘

    Nodes 1,2 have low reactance to 0 (electrically close)
    Nodes 3,4 have high reactance to 0 (electrically far)

    All nodes are in the same DC island (dc_island=0).
    """
    G = nx.DiGraph()

    # Central node
    G.add_node(0, lat=0.0, lon=0.0, dc_island=0)

    # Electrically close nodes (low reactance)
    G.add_node(1, lat=1.0, lon=0.0, dc_island=0)
    G.add_node(2, lat=0.0, lon=1.0, dc_island=0)

    # Electrically far nodes (high reactance)
    G.add_node(3, lat=-1.0, lon=0.0, dc_island=0)
    G.add_node(4, lat=0.0, lon=-1.0, dc_island=0)

    # Low reactance edges (close)
    G.add_edge(0, 1, x=0.01)
    G.add_edge(0, 2, x=0.01)

    # High reactance edges (far)
    G.add_edge(0, 3, x=1.0)
    G.add_edge(0, 4, x=1.0)

    # Cross connections for connectivity
    G.add_edge(1, 2, x=0.02)
    G.add_edge(3, 4, x=0.5)

    return G


@pytest.fixture
def multi_island_electrical_graph() -> nx.DiGraph:
    """
    Graph with two separate DC islands for testing DC island isolation.

    DC Island 0: nodes 0, 1, 2 (connected via low reactance)
    DC Island 1: nodes 3, 4, 5 (connected via low reactance)

    The two islands are connected by a DC link (edge 2->3), making the graph
    connected for PTDF calculation. However, the dc_island attribute ensures
    nodes from different islands get infinite electrical distance and thus
    are never clustered together.

    This fixture tests that nodes in different DC islands are never clustered together.
    """
    G = nx.DiGraph()

    # DC Island 0 - nodes near origin
    G.add_node(0, lat=0.0, lon=0.0, dc_island=0)
    G.add_node(1, lat=0.5, lon=0.5, dc_island=0)
    G.add_node(2, lat=0.3, lon=0.8, dc_island=0)

    # DC Island 1 - nodes far from origin
    G.add_node(3, lat=10.0, lon=10.0, dc_island=1)
    G.add_node(4, lat=10.5, lon=10.5, dc_island=1)
    G.add_node(5, lat=10.3, lon=10.8, dc_island=1)

    # Edges within DC Island 0
    G.add_edge(0, 1, x=0.1)
    G.add_edge(1, 2, x=0.15)
    G.add_edge(0, 2, x=0.12)

    # Edges within DC Island 1
    G.add_edge(3, 4, x=0.08)
    G.add_edge(4, 5, x=0.12)
    G.add_edge(3, 5, x=0.1)

    # DC link connecting the islands
    G.add_edge(2, 3, x=0.5, type='dc_link')

    return G


@pytest.fixture
def electrical_graph_no_dc_island() -> nx.DiGraph:
    """
    Graph without dc_island attribute for testing error handling.

    This fixture tests that appropriate error messages are shown
    when dc_island attribute is missing.
    """
    G = nx.DiGraph()

    # Nodes WITHOUT dc_island attribute
    G.add_node(0, lat=0.0, lon=0.0)
    G.add_node(1, lat=1.0, lon=0.0)
    G.add_node(2, lat=0.0, lon=1.0)

    G.add_edge(0, 1, x=0.1)
    G.add_edge(1, 2, x=0.15)
    G.add_edge(0, 2, x=0.12)

    return G


@pytest.fixture
def voltage_aware_graph() -> nx.DiGraph:
    """
    Graph with DC islands and voltage levels for VA partitioning test.

    DC Island 0: nodes 0,1,2 (voltage 220kV)
    DC Island 1: nodes 3,4,5 (voltage 380kV)

    Nodes within same island have same dc_island and voltage attributes.
    """
    G = nx.DiGraph()

    # DC Island 0 - 220kV network
    G.add_node(0, lat=0.0, lon=0.0, voltage=220.0, dc_island=0)
    G.add_node(1, lat=0.5, lon=0.5, voltage=220.0, dc_island=0)
    G.add_node(2, lat=0.3, lon=0.8, voltage=220.0, dc_island=0)

    # DC Island 1 - 380kV network
    G.add_node(3, lat=5.0, lon=5.0, voltage=380.0, dc_island=1)
    G.add_node(4, lat=5.5, lon=5.5, voltage=380.0, dc_island=1)
    G.add_node(5, lat=5.3, lon=5.8, voltage=380.0, dc_island=1)

    # Edges within DC Island 0
    G.add_edge(0, 1, x=0.1, type='line', primary_voltage=220.0, secondary_voltage=220.0)
    G.add_edge(1, 2, x=0.15, type='line', primary_voltage=220.0, secondary_voltage=220.0)

    # Edges within DC Island 1
    G.add_edge(3, 4, x=0.08, type='line', primary_voltage=380.0, secondary_voltage=380.0)
    G.add_edge(4, 5, x=0.12, type='line', primary_voltage=380.0, secondary_voltage=380.0)

    # DC Link connecting the islands (would be added after island detection in real use)
    G.add_edge(2, 3, x=0.5, type='dc_link', primary_voltage=400.0, secondary_voltage=400.0)

    return G


@pytest.fixture
def mixed_voltage_graph() -> nx.DiGraph:
    """
    Graph with multiple voltage levels within same DC island.

    Used to test voltage-aware partitioning respects voltage boundaries.

    DC Island 0:
        - 220kV: nodes 0, 1
        - 380kV: nodes 2, 3
        - Transformer connecting 220kV to 380kV
    """
    G = nx.DiGraph()

    # 220kV nodes
    G.add_node(0, lat=0.0, lon=0.0, voltage=220.0, dc_island=0)
    G.add_node(1, lat=0.5, lon=0.5, voltage=220.0, dc_island=0)

    # 380kV nodes
    G.add_node(2, lat=1.0, lon=1.0, voltage=380.0, dc_island=0)
    G.add_node(3, lat=1.5, lon=1.5, voltage=380.0, dc_island=0)

    # 220kV lines
    G.add_edge(0, 1, x=0.1, type='line', primary_voltage=220.0, secondary_voltage=220.0)

    # 380kV lines
    G.add_edge(2, 3, x=0.08, type='line', primary_voltage=380.0, secondary_voltage=380.0)

    # Transformer 220kV -> 380kV
    G.add_edge(1, 2, x=0.05, type='trafo', primary_voltage=220.0, secondary_voltage=380.0)

    return G


# =============================================================================
# MULTIGRAPH FIXTURES
# =============================================================================

@pytest.fixture
def parallel_edge_multigraph() -> nx.MultiDiGraph:
    """
    MultiDiGraph with parallel edges for testing parallel edge aggregation.

    Structure:
        0 ════→ 1   (two parallel edges)
        │       │
        ↓       ↓
        2 ──→ 3

    Parallel edges between 0 and 1 have different reactances.
    """
    G = nx.MultiDiGraph()

    G.add_node(0, lat=0.0, lon=0.0, demand=10.0)
    G.add_node(1, lat=0.0, lon=1.0, demand=20.0)
    G.add_node(2, lat=1.0, lon=0.0, demand=30.0)
    G.add_node(3, lat=1.0, lon=1.0, demand=40.0)

    # Parallel edges 0 -> 1 (key 0 and key 1)
    G.add_edge(0, 1, key=0, x=0.2, length=100.0, p_max=500.0)
    G.add_edge(0, 1, key=1, x=0.3, length=100.0, p_max=400.0)

    # Single edges
    G.add_edge(0, 2, x=0.15, length=150.0, p_max=600.0)
    G.add_edge(1, 3, x=0.1, length=120.0, p_max=550.0)
    G.add_edge(2, 3, x=0.25, length=180.0, p_max=450.0)

    return G


# =============================================================================
# PARTITION RESULT FIXTURES
# =============================================================================

@pytest.fixture
def simple_partition_map() -> dict:
    """
    Pre-defined partition for simple_digraph.

    Cluster 0: nodes [0, 2] (left column)
    Cluster 1: nodes [1, 3] (right column)
    """
    return {
        0: [0, 2],
        1: [1, 3]
    }


@pytest.fixture
def single_node_partition_map() -> dict:
    """
    Partition where each node is its own cluster.

    Used for testing edge cases.
    """
    return {
        0: [0],
        1: [1],
        2: [2],
        3: [3]
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def nodes_in_same_cluster(partition_map: dict, node_a, node_b) -> bool:
    """Check if two nodes are in the same cluster."""
    for cluster_nodes in partition_map.values():
        if node_a in cluster_nodes and node_b in cluster_nodes:
            return True
    return False


def nodes_in_different_clusters(partition_map: dict, node_a, node_b) -> bool:
    """Check if two nodes are in different clusters."""
    return not nodes_in_same_cluster(partition_map, node_a, node_b)


def get_node_cluster(partition_map: dict, node) -> int:
    """Get the cluster ID for a given node."""
    for cluster_id, nodes in partition_map.items():
        if node in nodes:
            return cluster_id
    return -1


def all_nodes_assigned(partition_map: dict, expected_nodes: list) -> bool:
    """Check if all expected nodes are assigned to exactly one cluster."""
    assigned = []
    for nodes in partition_map.values():
        assigned.extend(nodes)
    return sorted(assigned) == sorted(expected_nodes)
