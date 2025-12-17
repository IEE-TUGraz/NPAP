import itertools
from typing import Dict, List, Any

import networkx as nx

from npap.exceptions import AggregationError
from npap.interfaces import (
    TopologyStrategy, NodePropertyStrategy, EdgePropertyStrategy
)


# ============================================================================
# TOPOLOGY STRATEGIES - Define graph structure
# ============================================================================

class SimpleTopologyStrategy(TopologyStrategy):
    """
    Simple topology: one node per cluster, edges only where original connections exist

    This is the most basic topology strategy:
    - Creates one node per cluster
    - Creates a directed edge between two clusters only if there was at least one
      directed edge between nodes in those clusters in the original graph
    - Preserves edge direction from original graph
    - Does NOT create new edges
    """

    def create_topology(self, graph: nx.DiGraph,
                        partition_map: Dict[int, List[Any]]) -> nx.DiGraph:
        """Create aggregated topology with basic node and edge mapping"""
        try:
            aggregated = nx.DiGraph()

            # Step 1: Create nodes (one per cluster)
            for cluster_id in partition_map.keys():
                aggregated.add_node(cluster_id)

            # Step 2: Create edges only where connections exist
            _create_edges_with_existing_connection(partition_map, graph, aggregated)

            return aggregated

        except Exception as e:
            raise AggregationError(
                f"Failed to create simple topology: {e}",
                strategy="simple_topology"
            ) from e

    @property
    def can_create_new_edges(self) -> bool:
        """Simple topology does not create new edges"""
        return False


class ElectricalTopologyStrategy(TopologyStrategy):
    """
    Electrical topology: may create fully connected or partially connected topology
    for subsequent physical aggregation (e.g., Kron reduction)

    This topology strategy is designed for electrical networks where:
    - Physical coupling may exist even without direct edges
    - Kron reduction or other physical methods will determine final connectivity
    - May start with fully connected graph and let physical strategy prune
    """

    def __init__(self, initial_connectivity: str = "existing"):
        """
        Initialize electrical topology strategy

        Args:
            initial_connectivity: How to initialize edges
                - "existing": Only edges where original connections exist (like simple)
                - "full": Fully connected (all cluster pairs connected)
                - "threshold": Based on electrical distance threshold (TODO)
        """
        self.initial_connectivity = initial_connectivity

    def create_topology(self, graph: nx.DiGraph,
                        partition_map: Dict[int, List[Any]]) -> nx.DiGraph:
        """Create topology suitable for electrical aggregation"""
        try:
            aggregated = nx.DiGraph()

            # Step 1: Create nodes
            for cluster_id in partition_map.keys():
                aggregated.add_node(cluster_id)

            # Step 2: Create edges based on connectivity mode
            if self.initial_connectivity == "full":
                # Create fully connected directed graph (both directions)
                for cluster1, cluster2 in itertools.permutations(partition_map.keys(), 2):
                    aggregated.add_edge(cluster1, cluster2)

            elif self.initial_connectivity == "existing":
                _create_edges_with_existing_connection(partition_map, graph, aggregated)

            else:
                raise ValueError(f"Unknown connectivity mode: {self.initial_connectivity}")

            return aggregated

        except Exception as e:
            raise AggregationError(
                f"Failed to create electrical topology: {e}",
                strategy="electrical_topology"
            ) from e

    @property
    def can_create_new_edges(self) -> bool:
        """Electrical topology may create new edges depending on connectivity mode."""
        return self.initial_connectivity == "full"


def _clusters_connected_directed(graph: nx.DiGraph,
                                 source_nodes: List[Any],
                                 target_nodes: List[Any]) -> bool:
    """Return True if any directed edge exists from source_nodes to target_nodes."""
    for n1 in source_nodes:
        for n2 in target_nodes:
            if graph.has_edge(n1, n2):
                return True
    return False


def _create_edges_with_existing_connection(partition_map: Dict[int, List[Any]],
                                           graph: nx.DiGraph, aggregated: nx.DiGraph) -> None:
    """Create edges in aggregated graph where original connections exist."""
    edge_count = 0
    for cluster1, cluster2 in itertools.permutations(partition_map.keys(), 2):
        nodes1 = partition_map[cluster1]
        nodes2 = partition_map[cluster2]

        if _clusters_connected_directed(graph, nodes1, nodes2):
            aggregated.add_edge(cluster1, cluster2)
            edge_count += 1

    log_debug(
        f"ElectricalTopology (existing): {len(partition_map)} nodes, {edge_count} edges",
        LogCategory.AGGREGATION
    )


# ============================================================================
# NODE PROPERTY STRATEGIES - Statistical aggregation for node properties
# ============================================================================

class SumNodeStrategy(NodePropertyStrategy):
    """Sum numerical properties across nodes in a cluster"""

    def aggregate_property(self, graph: nx.DiGraph, nodes: List[Any], property_name: str) -> Any:
        """Sum property values across nodes"""
        try:
            values = []
            for node in nodes:
                if property_name in graph.nodes[node]:
                    value = graph.nodes[node][property_name]
                    if isinstance(value, (int, float)):
                        values.append(value)

            return sum(values) if values else 0
        except Exception as e:
            raise AggregationError(
                f"Failed to sum node property '{property_name}': {e}",
                strategy="sum"
            ) from e


class AverageNodeStrategy(NodePropertyStrategy):
    """Average numerical properties across nodes in a cluster"""

    def aggregate_property(self, graph: nx.DiGraph, nodes: List[Any], property_name: str) -> Any:
        """Average property values across nodes"""
        try:
            values = []
            for node in nodes:
                if property_name in graph.nodes[node]:
                    value = graph.nodes[node][property_name]
                    if isinstance(value, (int, float)):
                        values.append(value)

            return sum(values) / len(values) if values else 0
        except Exception as e:
            raise AggregationError(
                f"Failed to average node property '{property_name}': {e}",
                strategy="average"
            ) from e


class FirstNodeStrategy(NodePropertyStrategy):
    """Take the first available value for non-numerical properties"""

    def aggregate_property(self, graph: nx.DiGraph, nodes: List[Any], property_name: str) -> Any:
        """Take first available property value"""
        try:
            for node in nodes:
                if property_name in graph.nodes[node]:
                    return graph.nodes[node][property_name]
            return None
        except Exception as e:
            raise AggregationError(
                f"Failed to get first node property '{property_name}': {e}",
                strategy="first"
            ) from e


# ============================================================================
# EDGE PROPERTY STRATEGIES - Statistical aggregation for edge properties
# ============================================================================

class SumEdgeStrategy(EdgePropertyStrategy):
    """Sum numerical properties across edges"""

    def aggregate_property(self, original_edges: List[Dict[str, Any]], property_name: str) -> Any:
        """Sum property values across edges"""
        try:
            values = []
            for edge_data in original_edges:
                if property_name in edge_data:
                    value = edge_data[property_name]
                    if isinstance(value, (int, float)):
                        values.append(value)

            return sum(values) if values else 0
        except Exception as e:
            raise AggregationError(
                f"Failed to sum edge property '{property_name}': {e}",
                strategy="sum"
            ) from e


class AverageEdgeStrategy(EdgePropertyStrategy):
    """Average numerical properties across edges"""

    def aggregate_property(self, original_edges: List[Dict[str, Any]], property_name: str) -> Any:
        """Average property values across edges"""
        try:
            values = []
            for edge_data in original_edges:
                if property_name in edge_data:
                    value = edge_data[property_name]
                    if isinstance(value, (int, float)):
                        values.append(value)

            return sum(values) / len(values) if values else 0
        except Exception as e:
            raise AggregationError(
                f"Failed to average edge property '{property_name}': {e}",
                strategy="average"
            ) from e


class FirstEdgeStrategy(EdgePropertyStrategy):
    """Take the first available value for non-numerical properties"""

    def aggregate_property(self, original_edges: List[Dict[str, Any]], property_name: str) -> Any:
        """Take first available property value"""
        try:
            for edge_data in original_edges:
                if property_name in edge_data:
                    return edge_data[property_name]
            return None
        except Exception as e:
            raise AggregationError(
                f"Failed to get first edge property '{property_name}': {e}",
                strategy="first"
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

    def aggregate_property(self, original_edges: List[Dict[str, Any]], property_name: str) -> Any:
        """
        Calculate the equivalent reactance for the given parallel edges.

        Returns:
            Equivalent reactance value, or float('inf') if no edges exist
        """
        try:
            total_susceptance = 0.0
            has_valid_property = False
            epsilon = 1e-10  # Numerical tolerance for zero comparison

            for edge_data in original_edges:
                if property_name in edge_data:
                    reactance = edge_data[property_name]

                    if isinstance(reactance, (int, float)):
                        has_valid_property = True

                        if abs(reactance) < epsilon:
                            # A 0-reactance line (short circuit)
                            # makes the entire parallel group a short circuit.
                            return 0.0

                        # Add this line's susceptance (handles negative reactance for capacitive elements)
                        total_susceptance += (1.0 / reactance)

            if not has_valid_property:
                # No edges had this property - equivalent to open circuit (no connection)
                return float('inf')

            if abs(total_susceptance) < epsilon:
                # This means all valid edges had infinite reactance (open circuit).
                # The equivalent is also an open circuit.
                return float('inf')

            # Return the equivalent reactance
            return 1.0 / total_susceptance
        except Exception as e:
            raise AggregationError(
                f"Failed to calculate equivalent reactance for '{property_name}': {e}",
                strategy="equivalent_reactance"
            ) from e
