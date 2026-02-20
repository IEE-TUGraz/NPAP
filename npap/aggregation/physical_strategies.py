from __future__ import annotations

from typing import Any

import networkx as nx

from npap.aggregation.basic_strategies import (
    EquivalentReactanceStrategy,
    build_cluster_edge_map,
    build_node_to_cluster_map,
)
from npap.exceptions import AggregationError
from npap.interfaces import EdgeType, PhysicalAggregationStrategy
from npap.logging import LogCategory, log_warning


class TransformerConservationStrategy(PhysicalAggregationStrategy):
    """
    Preserve transformer-level impedance when reducing the network.

    This strategy treats all transformers between two clusters as parallel elements and
    stores their equivalent reactance/resistance as well as transformer count on the aggregated
    edge.  It is meant to be used with electrical topology and ensures impedance conservation
    before the statistical step runs.
    """

    def __init__(
        self,
        edge_type_attribute: str = "type",
        transformer_type: str = EdgeType.TRAFO.value,
        reactance_property: str = "x",
        resistance_property: str = "r",
    ):
        self.edge_type_attribute = edge_type_attribute
        self.transformer_type = transformer_type
        self.reactance_property = reactance_property
        self.resistance_property = resistance_property
        self._equivalent_reactance = EquivalentReactanceStrategy()

    @property
    def required_properties(self) -> list[str]:
        return [self.reactance_property, self.resistance_property]

    @property
    def modifies_properties(self) -> list[str]:
        return [self.reactance_property, self.resistance_property]

    @property
    def can_create_edges(self) -> bool:
        return False

    @property
    def required_topology(self) -> str:
        return "electrical"

    def aggregate(
        self,
        original_graph: nx.DiGraph,
        partition_map: dict[int, list[Any]],
        topology_graph: nx.DiGraph,
        properties: list[str],
        parameters: dict[str, Any] | None = None,
    ) -> nx.DiGraph:
        params = parameters or {}
        node_to_cluster = params.get("node_to_cluster") or build_node_to_cluster_map(partition_map)
        cluster_edge_map = params.get("cluster_edge_map") or build_cluster_edge_map(
            original_graph, node_to_cluster
        )

        for u, v in topology_graph.edges():
            original_edges = cluster_edge_map.get((u, v), [])
            transformer_edges = [
                edge
                for edge in original_edges
                if edge.get(self.edge_type_attribute) == self.transformer_type
            ]

            if not transformer_edges:
                continue

            reactance = self._calculate_equivalent(transformer_edges, self.reactance_property)
            resistance = self._calculate_equivalent(transformer_edges, self.resistance_property)

            if reactance is not None:
                topology_graph.edges[u, v][self.reactance_property] = reactance
            if resistance is not None:
                topology_graph.edges[u, v][self.resistance_property] = resistance

            topology_graph.edges[u, v]["transformer_count"] = len(transformer_edges)

        return topology_graph

    def _calculate_equivalent(self, edges: list[dict[str, Any]], prop: str) -> float | None:
        try:
            return self._equivalent_reactance.aggregate_property(edges, prop)
        except AggregationError as exc:
            log_warning(
                f"Failed to aggregate '{prop}' for transformer group: {exc}",
                LogCategory.AGGREGATION,
                warn_user=True,
            )
            return None


class KronReductionStrategy(PhysicalAggregationStrategy):
    """
    Kron reduction for DC power flow networks.

    TODO: Implementation pending. This is a placeholder.
    """

    @property
    def required_properties(self) -> list[str]:
        return ["reactance"]

    @property
    def modifies_properties(self) -> list[str]:
        return ["reactance"]

    @property
    def can_create_edges(self) -> bool:
        return True

    @property
    def required_topology(self) -> str:
        return "electrical"

    def aggregate(
        self,
        original_graph: nx.Graph,
        partition_map: dict[int, list[Any]],
        topology_graph: nx.Graph,
        properties: list[str],
        parameters: dict[str, Any] = None,
    ) -> nx.Graph:
        """Kron reduction - TO BE IMPLEMENTED"""
        raise NotImplementedError(
            "Kron reduction is not yet implemented. "
            "Use AggregationMode.SIMPLE or AggregationMode.GEOGRAPHICAL for now."
        )
