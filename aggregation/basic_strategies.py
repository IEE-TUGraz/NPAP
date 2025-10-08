import itertools
from typing import Dict, List, Any

import networkx as nx

from ..exceptions import AggregationError
from ..interfaces import (
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
    - Creates an edge between two clusters only if there was at least one edge
      between nodes in those clusters in the original graph
    - Does NOT create new edges
    """

    def create_topology(self, graph: nx.Graph,
                        partition_map: Dict[int, List[Any]]) -> nx.Graph:
        """Create aggregated topology with basic node and edge mapping"""
        try:
            aggregated = nx.Graph()

            # Step 1: Create nodes (one per cluster)
            for cluster_id in partition_map.keys():
                aggregated.add_node(cluster_id)

            # Step 2: Create edges only where connections exist
            for cluster1, cluster2 in itertools.combinations(partition_map.keys(), 2):
                nodes1 = partition_map[cluster1]
                nodes2 = partition_map[cluster2]

                # Check if there are any edges between these clusters
                has_connection = False
                for n1 in nodes1:
                    for n2 in nodes2:
                        if graph.has_edge(n1, n2):
                            has_connection = True
                            break
                    if has_connection:
                        break

                # Add edge if connection exists
                if has_connection:
                    aggregated.add_edge(cluster1, cluster2)

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

    def create_topology(self, graph: nx.Graph,
                        partition_map: Dict[int, List[Any]]) -> nx.Graph:
        """Create topology suitable for electrical aggregation"""
        try:
            aggregated = nx.Graph()

            # Step 1: Create nodes
            for cluster_id in partition_map.keys():
                aggregated.add_node(cluster_id)

            # Step 2: Create edges based on connectivity mode
            if self.initial_connectivity == "full":
                # Create fully connected graph
                for cluster1, cluster2 in itertools.combinations(partition_map.keys(), 2):
                    aggregated.add_edge(cluster1, cluster2)

            elif self.initial_connectivity == "existing":
                # Only create edges where connections exist (same as simple)
                for cluster1, cluster2 in itertools.combinations(partition_map.keys(), 2):
                    nodes1 = partition_map[cluster1]
                    nodes2 = partition_map[cluster2]

                    has_connection = False
                    for n1 in nodes1:
                        for n2 in nodes2:
                            if graph.has_edge(n1, n2):
                                has_connection = True
                                break
                        if has_connection:
                            break

                    if has_connection:
                        aggregated.add_edge(cluster1, cluster2)

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
        """Electrical topology may create new edges depending on connectivity mode"""
        return self.initial_connectivity == "full"


# ============================================================================
# NODE PROPERTY STRATEGIES - Statistical aggregation for node properties
# ============================================================================

class SumNodeStrategy(NodePropertyStrategy):
    """Sum numerical properties across nodes in a cluster"""

    def aggregate_property(self, graph: nx.Graph, nodes: List[Any], property_name: str) -> Any:
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

    def aggregate_property(self, graph: nx.Graph, nodes: List[Any], property_name: str) -> Any:
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

    def aggregate_property(self, graph: nx.Graph, nodes: List[Any], property_name: str) -> Any:
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
