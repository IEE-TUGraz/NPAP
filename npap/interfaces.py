from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import networkx as nx

from npap.utils import validate_required_attributes


class EdgeType(Enum):
    """
    Define edge types for voltage-aware strategies.

    Attributes
    ----------
    LINE : str
        Standard transmission or distribution line.
    TRAFO : str
        Transformer connecting different voltage levels.
    DC_LINK : str
        HVDC link connecting AC islands.
    """

    LINE = "line"
    TRAFO = "trafo"
    DC_LINK = "dc_link"


@dataclass
class PartitionResult:
    """
    Store partition result with metadata for validation and tracking.

    Attributes
    ----------
    mapping : dict[int, list[Any]]
        Dictionary mapping cluster_id to list of node_ids.
    original_graph_hash : str
        Hash of the original graph for compatibility validation.
    strategy_name : str
        Name of the partitioning strategy used.
    strategy_metadata : dict[str, Any]
        Strategy-specific metadata and parameters.
    n_clusters : int
        Number of clusters created.
    """

    mapping: dict[int, list[Any]]
    original_graph_hash: str
    strategy_name: str
    strategy_metadata: dict[str, Any]
    n_clusters: int


class AggregationMode(Enum):
    """
    Define pre-defined aggregation modes for common use cases.

    Attributes
    ----------
    SIMPLE : str
        Sum/average everything with simple topology.
    GEOGRAPHICAL : str
        Average coordinates, sum other properties.
    DC_KRON : str
        Kron reduction for DC networks.
    DC_PTDF : str
        PTDF-driven Kron reduction for DC networks.
    CUSTOM : str
        User-defined aggregation profile.
    CONSERVATION : str
        Preserve transformer impedance via dedicated physical strategy.
    """

    SIMPLE = "simple"
    GEOGRAPHICAL = "geographical"
    DC_KRON = "dc_kron"
    DC_PTDF = "dc_ptdf"
    CUSTOM = "custom"
    CONSERVATION = "transformer_conservation"


@dataclass
class AggregationProfile:
    """
    Configure aggregation separating physical from statistical operations.

    This profile distinguishes between:

    1. **Topology**: How the graph structure is reduced
    2. **Physical**: Electrical laws that must be preserved (coupled properties)
    3. **Statistical**: Simple operations on independent properties

    Attributes
    ----------
    topology_strategy : str
        Strategy for graph structure reduction ("simple", "electrical").
    physical_strategy : str or None
        Physical aggregation strategy ("kron_reduction", "equivalent_impedance").
    physical_properties : list[str]
        Properties handled by physical strategy (e.g., ["reactance", "resistance"]).
    physical_parameters : dict[str, Any]
        Additional parameters for physical strategies.
    node_properties : dict[str, str]
        Mapping of node property names to aggregation strategies.
    edge_properties : dict[str, str]
        Mapping of edge property names to aggregation strategies.
    edge_type_properties : dict[str, dict[str, str]]
        Per-edge-type strategy overrides.  Maps edge type values (from the
        ``edge_type_attribute`` attribute on edges) to per-type property
        strategy dicts.  When populated, edges are aggregated separately
        per type and the result graph is a MultiDiGraph with one edge per
        type per cluster pair.  Falls back to ``edge_properties`` for edges
        whose type is not listed here.
    edge_type_attribute : str
        Name of the edge attribute that stores the type label.  Only used
        when ``edge_type_properties`` is populated.  Defaults to ``"type"``.
    default_node_strategy : str
        Default strategy for unspecified node properties.
    default_edge_strategy : str
        Default strategy for unspecified edge properties.
    warn_on_defaults : bool
        Whether to warn when using default strategies.
    mode : AggregationMode
        Indicator of which pre-defined mode is being used.
    """

    topology_strategy: str = "simple"
    physical_strategy: str | None = None
    physical_properties: list[str] = field(default_factory=list)
    physical_parameters: dict[str, Any] = field(default_factory=dict)
    node_properties: dict[str, str] = field(default_factory=dict)
    edge_properties: dict[str, str] = field(default_factory=dict)
    edge_type_properties: dict[str, dict[str, str]] = field(default_factory=dict)
    edge_type_attribute: str = "type"
    default_node_strategy: str = "average"
    default_edge_strategy: str = "sum"
    warn_on_defaults: bool = True
    mode: AggregationMode = AggregationMode.CUSTOM


class DataLoadingStrategy(ABC):
    """Define interface for data loading strategies."""

    @abstractmethod
    def load(self, **kwargs) -> nx.DiGraph | nx.MultiDiGraph:
        """
        Load data and return a NetworkX directed graph.

        Returns
        -------
        nx.DiGraph or nx.MultiDiGraph
            Loaded network graph.
        """
        pass

    @abstractmethod
    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate input parameters before loading.

        Returns
        -------
        bool
            True if validation passes.

        Raises
        ------
        DataLoadingError
            If validation fails.
        """
        pass


class PartitioningStrategy(ABC):
    """Define interface for all partitioning algorithms."""

    @validate_required_attributes
    @abstractmethod
    def partition(self, graph: nx.DiGraph, **kwargs) -> dict[int, list[Any]]:
        """
        Partition nodes into clusters.

        Parameters
        ----------
        graph : nx.DiGraph
            Input graph to partition.
        **kwargs : dict
            Strategy-specific parameters.

        Returns
        -------
        dict[int, list[Any]]
            Mapping of cluster_id to list of node_ids.
        """
        pass

    @property
    @abstractmethod
    def required_attributes(self) -> dict[str, list[str]]:
        """
        Return required node/edge attributes.

        Returns
        -------
        dict[str, list[str]]
            Dictionary with 'nodes' and 'edges' keys listing required attributes.
        """
        pass


class TopologyStrategy(ABC):
    """
    Define interface for topology strategies.

    Topology strategies create the skeleton of the aggregated graph:

    - Which nodes exist in the aggregated graph
    - Which edges connect them
    - NO property aggregation at this stage
    """

    @abstractmethod
    def create_topology(self, graph: nx.DiGraph, partition_map: dict[int, list[Any]]) -> nx.DiGraph:
        """
        Create aggregated graph structure without properties.

        Parameters
        ----------
        graph : nx.DiGraph
            Original directed graph.
        partition_map : dict[int, list[Any]]
            Mapping of cluster_id to list of original node ids.

        Returns
        -------
        nx.DiGraph
            DiGraph with aggregated topology (nodes and edges, no attributes).
        """
        pass

    @property
    def can_create_new_edges(self) -> bool:
        """
        Check whether this strategy can create edges not in original graph.

        Returns
        -------
        bool
            True if strategy can create new edges, False otherwise.
        """
        return False

    @staticmethod
    def _clusters_connected(graph: nx.DiGraph, nodes1: list[Any], nodes2: list[Any]) -> bool:
        """
        Check if any edge exists between two node sets.

        For directed graphs, this checks edges in both directions.
        Use ``_clusters_connected_directed`` for direction-specific checks.

        Parameters
        ----------
        graph : nx.DiGraph
            The directed graph to check.
        nodes1 : list[Any]
            First set of node identifiers.
        nodes2 : list[Any]
            Second set of node identifiers.

        Returns
        -------
        bool
            True if any edge exists between the node sets.
        """
        set_n2 = set(nodes2)

        for node in nodes1:
            for neighbor in graph.successors(node):
                if neighbor in set_n2:
                    return True
            for neighbor in graph.predecessors(node):
                if neighbor in set_n2:
                    return True

        return False

    @staticmethod
    def _clusters_connected_directed(
        graph: nx.DiGraph, source_nodes: list[Any], target_nodes: list[Any]
    ) -> bool:
        """
        Check if any directed edge exists from source to target nodes.

        Parameters
        ----------
        graph : nx.DiGraph
            The directed graph to check.
        source_nodes : list[Any]
            Source node identifiers.
        target_nodes : list[Any]
            Target node identifiers.

        Returns
        -------
        bool
            True if any directed edge exists from source to target.
        """
        target_set = set(target_nodes)

        for node in source_nodes:
            for neighbor in graph.successors(node):
                if neighbor in target_set:
                    return True

        return False


class PhysicalAggregationStrategy(ABC):
    """
    Define interface for physics-aware aggregation strategies.

    These strategies operate on the entire graph and respect physical laws.
    They work on coupled properties (e.g., reactance and resistance together).
    They may create new edges based on electrical coupling.
    """

    @abstractmethod
    def aggregate(
        self,
        original_graph: nx.DiGraph,
        partition_map: dict[int, list[Any]],
        topology_graph: nx.DiGraph,
        properties: list[str],
        parameters: dict[str, Any] = None,
    ) -> nx.DiGraph:
        """
        Apply physical aggregation to the topology graph.

        Parameters
        ----------
        original_graph : nx.DiGraph
            Full resolution directed graph with all properties.
        partition_map : dict[int, list[Any]]
            Mapping of cluster_id to list of node_ids.
        topology_graph : nx.DiGraph
            DiGraph with aggregated structure but no properties yet.
        properties : list[str]
            Physical properties to aggregate (coupled).
        parameters : dict[str, Any], optional
            Additional parameters for the strategy.

        Returns
        -------
        nx.DiGraph
            DiGraph with physical properties correctly aggregated.
        """
        pass

    @property
    @abstractmethod
    def required_properties(self) -> list[str]:
        """
        Return properties this strategy requires.

        Returns
        -------
        list[str]
            List of required property names (e.g., ['reactance', 'resistance']).
        """
        pass

    @property
    @abstractmethod
    def modifies_properties(self) -> list[str]:
        """
        Return properties modified by this physical strategy.

        These properties should NOT be aggregated statistically afterward,
        as they are already handled by the physical strategy.

        Returns
        -------
        list[str]
            List of property names modified by this strategy.
        """
        pass

    @property
    def can_create_edges(self) -> bool:
        """
        Check whether this strategy can create new edges.

        Returns
        -------
        bool
            True if strategy can create new edges.
        """
        return False

    @property
    def required_topology(self) -> str:
        """
        Return required topology strategy for this physical aggregation.

        Returns
        -------
        str
            Name of the required topology strategy.
        """
        return "simple"


class NodePropertyStrategy(ABC):
    """Define interface for node property aggregation strategies."""

    @abstractmethod
    def aggregate_property(self, graph: nx.DiGraph, nodes: list[Any], property_name: str) -> Any:
        """
        Aggregate a specific property across nodes.

        Parameters
        ----------
        graph : nx.DiGraph
            The graph containing node properties.
        nodes : list[Any]
            List of node identifiers to aggregate.
        property_name : str
            Name of the property to aggregate.

        Returns
        -------
        Any
            Aggregated property value.
        """
        pass


class EdgePropertyStrategy(ABC):
    """Define interface for edge property aggregation strategies."""

    @abstractmethod
    def aggregate_property(self, original_edges: list[dict[str, Any]], property_name: str) -> Any:
        """
        Aggregate a specific property across edges.

        Parameters
        ----------
        original_edges : list[dict[str, Any]]
            List of edge attribute dictionaries.
        property_name : str
            Name of the property to aggregate.

        Returns
        -------
        Any
            Aggregated property value.
        """
        pass

    @property
    def required_attributes(self) -> dict[str, list[str]]:
        """
        Return required attributes for this property strategy.

        Returns
        -------
        dict[str, list[str]]
            Dictionary with 'nodes' and 'edges' keys listing required attributes.
        """
        return {"nodes": [], "edges": []}
