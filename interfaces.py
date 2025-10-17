from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import networkx as nx

from utils import validate_required_attributes


@dataclass
class PartitionResult:
    """Enhanced partition result with metadata"""
    mapping: Dict[int, List[Any]]  # cluster_id -> list of node_ids
    original_graph_hash: str  # validation hash of original graph
    strategy_name: str  # which strategy was used
    strategy_metadata: Dict[str, Any]  # strategy-specific data
    n_clusters: int  # target number of clusters


class AggregationMode(Enum):
    """Pre-defined aggregation modes for common use cases"""
    SIMPLE = "simple"  # Sum/avg everything, simple topology
    GEOGRAPHICAL = "geographical"  # Average coordinates, sum other properties
    DC_KRON = "dc_kron"  # Kron reduction for DC networks
    CUSTOM = "custom"  # User-defined profile


@dataclass
class AggregationProfile:
    """
    Enhanced aggregation profile separating physical from statistical aggregation

    This profile distinguishes between:
    1. Topology: How the graph structure is reduced
    2. Physical: Electrical laws that must be preserved (operates on coupled properties)
    3. Statistical: Simple operations on independent properties
    """
    # Topology: How graph structure is reduced
    topology_strategy: str = "simple"  # "simple", "electrical", etc.

    # Physical aggregation (operates on coupled electrical properties)
    physical_strategy: Optional[str] = None  # "kron_reduction", "equivalent_impedance", "ptdf", etc.
    physical_properties: List[str] = field(default_factory=list)  # ["reactance", "resistance"]
    physical_parameters: Dict[str, Any] = field(default_factory=dict)  # Additional params for physical strategies

    # Statistical aggregation (independent properties)
    node_properties: Dict[str, str] = field(default_factory=dict)  # {demand: "sum", name: "first"}
    edge_properties: Dict[str, str] = field(default_factory=dict)  # {length: "average"}

    # Defaults for unspecified properties
    default_node_strategy: str = "average"
    default_edge_strategy: str = "sum"
    warn_on_defaults: bool = True

    # Mode indicator
    mode: AggregationMode = AggregationMode.CUSTOM


class DataLoadingStrategy(ABC):
    """Interface for data loading strategies"""

    @abstractmethod
    def load(self, **kwargs) -> nx.Graph:
        """Load data and return NetworkX graph"""
        pass

    @abstractmethod
    def validate_inputs(self, **kwargs) -> bool:
        """Validate input parameters before loading"""
        pass


class PartitioningStrategy(ABC):
    """Interface for all partitioning algorithms"""

    @validate_required_attributes
    @abstractmethod
    def partition(self, graph: nx.Graph, **kwargs) -> Dict[int, List[Any]]:
        """Partition nodes into clusters"""
        pass

    @property
    @abstractmethod
    def required_attributes(self) -> Dict[str, List[str]]:
        """Required node/edge attributes: {'nodes': [...], 'edges': [...]}"""
        pass


class TopologyStrategy(ABC):
    """
    Interface for topology strategies - defines how graph structure is reduced

    Topology strategies create the skeleton of the aggregated graph:
    - Which nodes exist in the aggregated graph
    - Which edges connect them
    - NO property aggregation at this stage
    """

    @abstractmethod
    def create_topology(self, graph: nx.Graph,
                        partition_map: Dict[int, List[Any]]) -> nx.Graph:
        """
        Create aggregated graph structure (nodes + edges, no properties yet)

        Args:
            graph: Original graph
            partition_map: Cluster_id -> list of original node ids

        Returns:
            Graph with aggregated topology (nodes and edges, but no attributes)
        """
        pass

    @property
    def can_create_new_edges(self) -> bool:
        """Whether this strategy can create edges that didn't exist in original graph"""
        return False

    @staticmethod
    def _clusters_connected(graph: nx.Graph, nodes1: List[Any], nodes2: List[Any]) -> bool:
        """Return True if any edge exists between nodes in nodes1 and nodes2."""
        for n1 in nodes1:
            for n2 in nodes2:
                if graph.has_edge(n1, n2):
                    return True
        return False


class PhysicalAggregationStrategy(ABC):
    """
    Interface for physics-aware aggregation strategies

    These strategies operate on the entire graph and respect physical laws.
    They work on coupled properties (e.g., reactance and resistance together).
    They may create new edges based on electrical coupling.
    """

    @abstractmethod
    def aggregate(self, original_graph: nx.Graph,
                  partition_map: Dict[int, List[Any]],
                  topology_graph: nx.Graph,
                  properties: List[str],
                  parameters: Dict[str, Any] = None) -> nx.Graph:
        """
        Apply physical aggregation to the topology graph

        Args:
            original_graph: Full resolution graph with all properties
            partition_map: Node-to-cluster mapping
            topology_graph: Graph with aggregated structure but no properties yet
            properties: Physical properties to aggregate (coupled)
            parameters: Additional parameters for the strategy

        Returns:
            Graph with physical properties correctly aggregated
        """
        pass

    @property
    @abstractmethod
    def required_properties(self) -> List[str]:
        """Properties this strategy requires (e.g., ['reactance', 'resistance'])"""
        pass

    @property
    @abstractmethod
    def modifies_properties(self) -> List[str]:
        """
        Properties that are modified by this physical strategy

        These properties should NOT be aggregated statistically afterward,
        as they are already handled by the physical strategy.
        """
        pass

    @property
    def can_create_edges(self) -> bool:
        """Whether this strategy can create new edges"""
        return False

    @property
    def required_topology(self) -> str:
        """Required topology strategy for this physical aggregation"""
        return "simple"


class NodePropertyStrategy(ABC):
    """Interface for node property aggregation strategies"""

    @abstractmethod
    def aggregate_property(self, graph: nx.Graph, nodes: List[Any], property_name: str) -> Any:
        """Aggregate a specific property across nodes"""
        pass


class EdgePropertyStrategy(ABC):
    """Interface for edge property aggregation strategies"""

    @abstractmethod
    def aggregate_property(self, original_edges: List[Dict[str, Any]], property_name: str) -> Any:
        """Aggregate a specific property across edges"""
        pass

    @property
    def required_attributes(self) -> Dict[str, List[str]]:
        """Required attributes for this property strategy"""
        return {'nodes': [], 'edges': []}
