from typing import Dict, List, Any

import networkx as nx

from ..interfaces import PhysicalAggregationStrategy


class KronReductionStrategy(PhysicalAggregationStrategy):
    """
    Kron reduction for DC power flow networks

    TODO: Implementation pending. This is a placeholder.
    """

    @property
    def required_properties(self) -> List[str]:
        return ['reactance']

    @property
    def modifies_properties(self) -> List[str]:
        return ['reactance']

    @property
    def can_create_edges(self) -> bool:
        return True

    @property
    def required_topology(self) -> str:
        return "electrical"

    def aggregate(self, original_graph: nx.Graph,
                  partition_map: Dict[int, List[Any]],
                  topology_graph: nx.Graph,
                  properties: List[str],
                  parameters: Dict[str, Any] = None) -> nx.Graph:
        """Kron reduction - TO BE IMPLEMENTED"""
        raise NotImplementedError(
            "Kron reduction is not yet implemented. "
            "Use AggregationMode.SIMPLE or AggregationMode.GEOGRAPHICAL for now."
        )
