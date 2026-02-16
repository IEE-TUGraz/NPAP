import itertools
from collections import defaultdict
from typing import Any

import networkx as nx
import numpy as np

from npap.exceptions import AggregationError
from npap.interfaces import EdgePropertyStrategy, NodePropertyStrategy, TopologyStrategy
from npap.logging import LogCategory, log_debug, log_warning

# ============================================================================
# PRECOMPUTATION UTILITIES - Build mappings for fast lookups
# ============================================================================


def build_node_to_cluster_map(partition_map: dict[int, list[Any]]) -> dict[Any, int]:
    """
    Build a reverse mapping from node ID to cluster ID.

    Args:
        partition_map: Dict mapping cluster_id -> list of node IDs

    Returns
    -------
        Dict mapping node_id -> cluster_id for O(1) lookups
    """
    node_to_cluster = {}
    for cluster_id, nodes in partition_map.items():
        for node in nodes:
            node_to_cluster[node] = cluster_id
    return node_to_cluster


def build_cluster_edge_map(
    graph: nx.DiGraph, node_to_cluster: dict[Any, int]
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    """
    Build a mapping from cluster pairs to their original edge data.

    Single pass over all edges - O(E) complexity.

    Args:
        graph: Original NetworkX graph
        node_to_cluster: Mapping from node_id -> cluster_id

    Returns
    -------
        Dict mapping (source_cluster, target_cluster) -> list of edge attribute dicts
    """
    cluster_edges: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)

    for u, v, data in graph.edges(data=True):
        cluster_u = node_to_cluster.get(u)
        cluster_v = node_to_cluster.get(v)

        # Skip edges where nodes aren't in the partition (shouldn't happen normally)
        if cluster_u is None or cluster_v is None:
            continue

        # Skip self-loops at cluster level (internal edges)
        if cluster_u != cluster_v:
            cluster_edges[(cluster_u, cluster_v)].append(data)

    return dict(cluster_edges)


def build_typed_cluster_edge_map(
    graph: nx.DiGraph,
    node_to_cluster: dict[Any, int],
    type_attribute: str = "type",
) -> dict[str, dict[tuple[int, int], list[dict[str, Any]]]]:
    """
    Build cluster edge maps grouped by edge type.

    Like ``build_cluster_edge_map`` but returns a nested dict keyed first by the
    value of the *type_attribute* on each edge, then by ``(source_cluster,
    target_cluster)`` pair.  Edges without the type attribute are collected under
    the key ``"_untyped"``.

    Single pass over all edges – O(E) complexity.

    Parameters
    ----------
    graph : nx.DiGraph
        Original NetworkX graph.
    node_to_cluster : dict[Any, int]
        Mapping from node_id to cluster_id.
    type_attribute : str
        Edge attribute that stores the type label (default ``"type"``).

    Returns
    -------
    dict[str, dict[tuple[int, int], list[dict[str, Any]]]]
        Nested mapping: edge_type -> (src_cluster, tgt_cluster) -> [edge_data].
    """
    typed_edges: dict[str, dict[tuple[int, int], list[dict[str, Any]]]] = {}

    for u, v, data in graph.edges(data=True):
        cluster_u = node_to_cluster.get(u)
        cluster_v = node_to_cluster.get(v)
        if cluster_u is None or cluster_v is None or cluster_u == cluster_v:
            continue

        edge_type = data.get(type_attribute, "_untyped")
        if edge_type not in typed_edges:
            typed_edges[edge_type] = defaultdict(list)
        typed_edges[edge_type][(cluster_u, cluster_v)].append(data)

    # Convert inner defaultdicts to plain dicts
    return {t: dict(m) for t, m in typed_edges.items()}


def build_cluster_connectivity_set(
    graph: nx.DiGraph, node_to_cluster: dict[Any, int]
) -> set[tuple[int, int]]:
    """
    Build a set of connected cluster pairs from original graph edges.

    Single pass over all edges - O(E) complexity.

    Args:
        graph: Original NetworkX graph
        node_to_cluster: Mapping from node_id -> cluster_id

    Returns
    -------
        Set of (source_cluster, target_cluster) tuples where edges exist
    """
    connected_clusters: set[tuple[int, int]] = set()

    for u, v in graph.edges():
        cluster_u = node_to_cluster.get(u)
        cluster_v = node_to_cluster.get(v)

        if cluster_u is not None and cluster_v is not None and cluster_u != cluster_v:
            connected_clusters.add((cluster_u, cluster_v))

    return connected_clusters


# ============================================================================
# TOPOLOGY STRATEGIES - Define graph structure
# ============================================================================


class SimpleTopologyStrategy(TopologyStrategy):
    """
    Simple topology: one node per cluster, edges only where original connections exist.

    This is the most basic topology strategy:

    - Creates one node per cluster
    - Creates a directed edge between two clusters only if there was at least one
      directed edge between nodes in those clusters in the original graph
    - Preserves edge direction from original graph
    - Does NOT create new edges
    """

    def create_topology(self, graph: nx.DiGraph, partition_map: dict[int, list[Any]]) -> nx.DiGraph:
        """Create aggregated topology with basic node and edge mapping."""
        try:
            aggregated = nx.DiGraph()

            # Step 1: Create nodes (one per cluster)
            aggregated.add_nodes_from(partition_map.keys())

            # Step 2: Build node-to-cluster mapping
            node_to_cluster = build_node_to_cluster_map(partition_map)

            # Step 3: Find connected cluster pairs in single pass
            connected_clusters = build_cluster_connectivity_set(graph, node_to_cluster)

            # Step 4: Add edges for connected clusters
            aggregated.add_edges_from(connected_clusters)

            log_debug(
                f"SimpleTopology: {aggregated.number_of_nodes()} nodes, "
                f"{aggregated.number_of_edges()} edges",
                LogCategory.AGGREGATION,
            )

            return aggregated

        except Exception as e:
            raise AggregationError(
                f"Failed to create simple topology: {e}", strategy="simple_topology"
            ) from e

    @property
    def can_create_new_edges(self) -> bool:
        """Simple topology does not create new edges."""
        return False


class ElectricalTopologyStrategy(TopologyStrategy):
    """
    Electrical topology: may create fully connected or partially connected topology
    for subsequent physical aggregation (e.g., Kron reduction).

    This topology strategy is designed for electrical networks where:
    - Physical coupling may exist even without direct edges
    - Kron reduction or other physical methods will determine final connectivity
    - May start with fully connected graph and let physical strategy prune
    """

    def __init__(self, initial_connectivity: str = "existing"):
        """
        Initialize electrical topology strategy.

        Args:
            initial_connectivity: How to initialize edges
                - "existing": Only edges where original connections exist (like simple)
                - "full": Fully connected (all cluster pairs connected)
        """
        self.initial_connectivity = initial_connectivity

    def create_topology(self, graph: nx.DiGraph, partition_map: dict[int, list[Any]]) -> nx.DiGraph:
        """Create topology suitable for electrical aggregation."""
        try:
            aggregated = nx.DiGraph()

            # Step 1: Create nodes
            aggregated.add_nodes_from(partition_map.keys())

            # Step 2: Create edges based on connectivity mode
            if self.initial_connectivity == "full":
                # Add all permutations as edges (excluding self-loops)
                aggregated.add_edges_from(itertools.permutations(partition_map.keys(), 2))
                log_debug(
                    f"ElectricalTopology (full): {len(partition_map)} nodes, "
                    f"{aggregated.number_of_edges()} edges",
                    LogCategory.AGGREGATION,
                )

            elif self.initial_connectivity == "existing":
                # Build node-to-cluster mapping
                node_to_cluster = build_node_to_cluster_map(partition_map)

                # Find connected cluster pairs in single pass
                connected_clusters = build_cluster_connectivity_set(graph, node_to_cluster)

                # Add edges
                aggregated.add_edges_from(connected_clusters)

                log_debug(
                    f"ElectricalTopology (existing): {len(partition_map)} nodes, "
                    f"{aggregated.number_of_edges()} edges",
                    LogCategory.AGGREGATION,
                )

            else:
                raise AggregationError(
                    f"Unknown connectivity mode: {self.initial_connectivity}",
                    strategy="electrical_topology",
                )

            return aggregated

        except Exception as e:
            if isinstance(e, AggregationError):
                raise
            raise AggregationError(
                f"Failed to create electrical topology: {e}",
                strategy="electrical_topology",
            ) from e

    @property
    def can_create_new_edges(self) -> bool:
        """Electrical topology may create new edges depending on connectivity mode."""
        return self.initial_connectivity == "full"


# ============================================================================
# NODE PROPERTY STRATEGIES - Statistical aggregation for node properties
# ============================================================================


def _extract_numeric_node_values(
    graph: nx.DiGraph, nodes: list[Any], property_name: str
) -> np.ndarray:
    """
    Extract numeric values for a property from a list of nodes.

    Uses list comprehension for speed, then converts to NumPy array.

    Returns
    -------
        NumPy array of numeric values (may be empty)
    """
    values = [
        graph.nodes[node][property_name]
        for node in nodes
        if property_name in graph.nodes[node]
        and isinstance(graph.nodes[node][property_name], (int, float))
    ]
    return np.array(values, dtype=np.float64) if values else np.array([], dtype=np.float64)


class SumNodeStrategy(NodePropertyStrategy):
    """Sum numerical properties across nodes in a cluster."""

    def aggregate_property(self, graph: nx.DiGraph, nodes: list[Any], property_name: str) -> Any:
        """Sum property values across nodes using NumPy."""
        try:
            values = _extract_numeric_node_values(graph, nodes, property_name)
            return float(np.sum(values)) if len(values) > 0 else 0.0
        except Exception as e:
            raise AggregationError(
                f"Failed to sum node property '{property_name}': {e}", strategy="sum"
            ) from e


class AverageNodeStrategy(NodePropertyStrategy):
    """Average numerical properties across nodes in a cluster."""

    def aggregate_property(self, graph: nx.DiGraph, nodes: list[Any], property_name: str) -> Any:
        """Average property values across nodes using NumPy."""
        try:
            values = _extract_numeric_node_values(graph, nodes, property_name)
            if len(values) > 0:
                return float(np.mean(values))

            # No numeric values found — fall back to first available value
            log_warning(
                f"No numeric values found for node property '{property_name}'. "
                f"Falling back to first available value.",
                LogCategory.AGGREGATION,
                warn_user=True,
            )
            FirstNodeStrategy.aggregate_property(graph, nodes, property_name)
        except Exception as e:
            raise AggregationError(
                f"Failed to average node property '{property_name}': {e}",
                strategy="average",
            ) from e


class FirstNodeStrategy(NodePropertyStrategy):
    """Take the first available value for non-numerical properties."""

    def aggregate_property(self, graph: nx.DiGraph, nodes: list[Any], property_name: str) -> Any:
        """Take first available property value."""
        try:
            for node in nodes:
                if property_name in graph.nodes[node]:
                    return graph.nodes[node][property_name]
            return None
        except Exception as e:
            raise AggregationError(
                f"Failed to get first node property '{property_name}': {e}",
                strategy="first",
            ) from e


# ============================================================================
# EDGE PROPERTY STRATEGIES - Statistical aggregation for edge properties
# ============================================================================


def _extract_numeric_edge_values(
    original_edges: list[dict[str, Any]], property_name: str
) -> np.ndarray:
    """
    Extract numeric values for a property from a list of edge data dicts.

    Uses list comprehension for speed, then converts to NumPy array.

    Returns
    -------
        NumPy array of numeric values (may be empty)
    """
    values = [
        edge_data[property_name]
        for edge_data in original_edges
        if property_name in edge_data and isinstance(edge_data[property_name], (int, float))
    ]
    return np.array(values, dtype=np.float64) if values else np.array([], dtype=np.float64)


class SumEdgeStrategy(EdgePropertyStrategy):
    """Sum numerical properties across edges."""

    def aggregate_property(self, original_edges: list[dict[str, Any]], property_name: str) -> Any:
        """Sum property values across edges using NumPy."""
        try:
            values = _extract_numeric_edge_values(original_edges, property_name)
            return float(np.sum(values)) if len(values) > 0 else 0.0
        except Exception as e:
            raise AggregationError(
                f"Failed to sum edge property '{property_name}': {e}", strategy="sum"
            ) from e


class AverageEdgeStrategy(EdgePropertyStrategy):
    """Average numerical properties across edges."""

    def aggregate_property(self, original_edges: list[dict[str, Any]], property_name: str) -> Any:
        """Average property values across edges using NumPy."""
        try:
            values = _extract_numeric_edge_values(original_edges, property_name)
            return float(np.mean(values)) if len(values) > 0 else 0.0
        except Exception as e:
            raise AggregationError(
                f"Failed to average edge property '{property_name}': {e}",
                strategy="average",
            ) from e


class FirstEdgeStrategy(EdgePropertyStrategy):
    """Take the first available value for non-numerical properties."""

    def aggregate_property(self, original_edges: list[dict[str, Any]], property_name: str) -> Any:
        """Take first available property value."""
        try:
            for edge_data in original_edges:
                if property_name in edge_data:
                    return edge_data[property_name]
            return None
        except Exception as e:
            raise AggregationError(
                f"Failed to get first edge property '{property_name}': {e}",
                strategy="first",
            ) from e


class EquivalentReactanceStrategy(EdgePropertyStrategy):
    """
    Aggregates reactance for a set of parallel edges.

    This strategy calculates the equivalent reactance by first converting
    each edge's reactance (x) to susceptance (b = 1/x), summing the
    susceptances, and then converting the total susceptance back
    to an equivalent reactance (x_eq = 1 / b_eq).

    This correctly models the physics of parallel lines as:
    b_eq = b_1 + b_2 + ...
    x_eq = 1 / b_eq
    """

    def aggregate_property(self, original_edges: list[dict[str, Any]], property_name: str) -> Any:
        """
        Calculate the equivalent reactance for the given parallel edges.

        Uses NumPy for vectorized susceptance calculation.

        Returns
        -------
            Equivalent reactance value, or float('inf') if non-numeric values present

        Raises
        ------
        AggregationError
            If no edges have the property at all (entirely missing attribute).
        """
        try:
            values = _extract_numeric_edge_values(original_edges, property_name)

            if len(values) == 0:
                # Distinguish: property exists but non-numeric vs. entirely missing
                has_property = any(property_name in edge for edge in original_edges)
                if has_property:
                    log_warning(
                        f"No numeric values found for edge property '{property_name}'. "
                        f"Returning inf (open circuit equivalent).",
                        LogCategory.AGGREGATION,
                        warn_user=False,
                    )
                    return float("inf")
                raise AggregationError(
                    f"No edges have the '{property_name}' attribute. Cannot calculate "
                    f"equivalent reactance.",
                    strategy="equivalent_reactance",
                )

            epsilon = 1e-10

            # Check for zero reactance (short circuit)
            if np.any(np.abs(values) < epsilon):
                return 0.0

            # Vectorized susceptance calculation: b = 1/x
            susceptances = 1.0 / values
            total_susceptance = np.sum(susceptances)

            if abs(total_susceptance) < epsilon:
                return float("inf")

            return float(1.0 / total_susceptance)

        except Exception as e:
            if isinstance(e, AggregationError):
                raise
            raise AggregationError(
                f"Failed to calculate equivalent reactance for '{property_name}': {e}",
                strategy="equivalent_reactance",
            ) from e
