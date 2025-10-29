import warnings
from typing import Any, Dict, List, Optional

import networkx as nx

from interfaces import (
    DataLoadingStrategy, PartitioningStrategy, AggregationProfile, AggregationMode,
    TopologyStrategy, PhysicalAggregationStrategy,
    NodePropertyStrategy, EdgePropertyStrategy, PartitionResult
)


class InputDataManager:
    """Manages data loading from different sources"""

    def __init__(self):
        self._strategies: Dict[str, DataLoadingStrategy] = {}
        self._register_default_strategies()

    def register_strategy(self, name: str, strategy: DataLoadingStrategy):
        """Register a new data loading strategy"""
        self._strategies[name] = strategy

    def load(self, strategy_name: str, **kwargs) -> nx.Graph:
        """Load data using specified strategy"""
        if strategy_name not in self._strategies:
            available = ', '.join(self._strategies.keys())
            raise ValueError(f"Unknown data loading strategy: {strategy_name}. Available: {available}")

        strategy = self._strategies[strategy_name]
        strategy.validate_inputs(**kwargs)
        return strategy.load(**kwargs)

    def _register_default_strategies(self):
        """Register built-in loading strategies"""
        from input.csv_loader import CSVFilesStrategy
        from input.networkx_loader import NetworkXDirectStrategy

        self._strategies['csv_files'] = CSVFilesStrategy()
        self._strategies['networkx_direct'] = NetworkXDirectStrategy()


class PartitioningManager:
    """Manages partitioning strategies"""

    def __init__(self):
        self._strategies: Dict[str, PartitioningStrategy] = {}
        self._register_default_strategies()

    def register_strategy(self, name: str, strategy: PartitioningStrategy):
        """Register a new partitioning strategy"""
        self._strategies[name] = strategy

    def partition(self, graph: nx.Graph, method: str, **kwargs) -> Dict[int, List[Any]]:
        """Execute partitioning using specified strategy"""
        if method not in self._strategies:
            available = ', '.join(self._strategies.keys())
            raise ValueError(f"Unknown partitioning strategy: {method}. Available: {available}")

        return self._strategies[method].partition(graph, **kwargs)

    def _register_default_strategies(self):
        """Register built-in partitioning strategies"""
        from partitioning.geographical import GeographicalPartitioning

        self._strategies['geographical_kmeans'] = GeographicalPartitioning(
            algorithm='kmeans',
            distance_metric='euclidean'
        )
        self._strategies['geographical_kmedoids_euclidean'] = GeographicalPartitioning(
            algorithm='kmedoids',
            distance_metric='euclidean'
        )
        self._strategies['geographical_kmedoids_haversine'] = GeographicalPartitioning(
            algorithm='kmedoids',
            distance_metric='haversine'
        )
        self._strategies['geographical_dbscan_euclidean'] = GeographicalPartitioning(
            algorithm='dbscan',
            distance_metric='euclidean'
        )
        self._strategies['geographical_dbscan_haversine'] = GeographicalPartitioning(
            algorithm='dbscan',
            distance_metric='haversine'
        )
        self._strategies['geographical_hierarchical'] = GeographicalPartitioning(
            algorithm='hierarchical',
            distance_metric='euclidean'
        )
        self._strategies['geographical_hdbscan_euclidean'] = GeographicalPartitioning(
            algorithm='hdbscan',
            distance_metric='euclidean'
        )
        self._strategies['geographical_hdbscan_haversine'] = GeographicalPartitioning(
            algorithm='hdbscan',
            distance_metric='haversine'
        )


class AggregationManager:
    """
    Manages aggregation strategies and orchestrates the aggregation process

    Aggregation is a 3-step process:
    1. Topology creation (graph structure)
    2. Physical aggregation (electrical laws)
    3. Statistical property aggregation (independent properties)
    """

    def __init__(self):
        # Topology strategies (how graph structure is reduced)
        self._topology_strategies: Dict[str, TopologyStrategy] = {}

        # Physical aggregation strategies (electrical laws)
        self._physical_strategies: Dict[str, PhysicalAggregationStrategy] = {}

        # Statistical property aggregation strategies
        self._node_strategies: Dict[str, NodePropertyStrategy] = {}
        self._edge_strategies: Dict[str, EdgePropertyStrategy] = {}

        self._register_default_strategies()

    def register_topology_strategy(self, name: str, strategy: TopologyStrategy):
        """Register a topology strategy"""
        self._topology_strategies[name] = strategy

    def register_physical_strategy(self, name: str, strategy: PhysicalAggregationStrategy):
        """Register a physical aggregation strategy"""
        self._physical_strategies[name] = strategy

    def register_node_strategy(self, name: str, strategy: NodePropertyStrategy):
        """Register a node property aggregation strategy"""
        self._node_strategies[name] = strategy

    def register_edge_strategy(self, name: str, strategy: EdgePropertyStrategy):
        """Register an edge property aggregation strategy"""
        self._edge_strategies[name] = strategy

    @staticmethod
    def get_mode_profile(mode: AggregationMode, **overrides) -> AggregationProfile:
        """
        Get pre-defined aggregation profile for a given mode

        Args:
            mode: Aggregation mode
            **overrides: Override specific profile parameters

        Returns:
            AggregationProfile configured for the mode
        """
        from aggregation.modes import get_mode_profile
        return get_mode_profile(mode, **overrides)

    def aggregate(self, graph: nx.Graph, partition_map: Dict[int, List[Any]],
                  profile: AggregationProfile = None) -> nx.Graph:
        """
        Execute aggregation using the specified profile

        Aggregation is a 3-step process:
        1. Create topology (nodes + edge structure)
        2. Apply physical aggregation (if specified)
        3. Aggregate remaining properties statistically
        """
        if profile is None:
            profile = AggregationProfile()  # Use defaults

        # Validate strategies exist
        self._validate_profile(profile)

        # Step 1: Create topology
        topology_strategy = self._topology_strategies[profile.topology_strategy]
        aggregated = topology_strategy.create_topology(graph, partition_map)

        # Track which properties are handled by physical aggregation
        physical_modified_properties = set()

        # Step 2: Apply physical aggregation (if specified)
        if profile.physical_strategy:
            physical_strategy = self._physical_strategies[profile.physical_strategy]

            # Validate topology compatibility
            if topology_strategy.__class__.__name__ != physical_strategy.required_topology:
                warnings.warn(
                    f"Physical strategy '{profile.physical_strategy}' recommends "
                    f"'{physical_strategy.required_topology}' topology, "
                    f"but '{profile.topology_strategy}' is being used. "
                    f"Results may be incorrect."
                )

            # Apply physical aggregation
            aggregated = physical_strategy.aggregate(
                graph, partition_map, aggregated,
                profile.physical_properties,
                profile.physical_parameters
            )

            # Mark properties as modified by physical strategy
            physical_modified_properties = set(physical_strategy.modifies_properties)

            # Warn user if they tried to override physical properties
            self._check_property_conflicts(profile, physical_modified_properties)

        # Step 3: Aggregate node properties (skip properties aggregated in physical step)
        self._aggregate_node_properties(
            graph, partition_map, aggregated, profile, physical_modified_properties
        )

        # Step 4: Aggregate edge properties (skip properties aggregated in physical step)
        self._aggregate_edge_properties(
            graph, partition_map, aggregated, profile, physical_modified_properties
        )

        return aggregated

    def aggregate_parallel_edges(self, graph: nx.MultiGraph, edge_properties: Dict[str, str] = None,
                                 default_strategy: str = "sum", warn_on_defaults: bool = True) -> nx.Graph:
        """
        Collapse parallel edges in a MultiGraph to produce a simple Graph.

        This method aggregates all parallel edges between the same node pairs
        using the specified edge property strategies, converting a MultiGraph
        to a simple Graph that can be partitioned.

        Args:
            graph: MultiGraph with potential parallel edges
            edge_properties: Dict mapping property names to aggregation strategies
                           e.g., {"reactance": "average", "length": "sum"}
            default_strategy: Strategy to use for properties not specified (default: "sum")
            warn_on_defaults: Whether to warn when using default strategy

        Returns:
            Simple Graph with parallel edges aggregated

        Raises:
            ValueError: If graph is not a MultiGraph
            AggregationError: If aggregation fails
        """
        from exceptions import AggregationError

        # Validate input
        if not isinstance(graph, nx.MultiGraph):
            raise ValueError(
                f"Expected MultiGraph, got {type(graph).__name__}. "
                "This method is only for collapsing parallel edges."
            )

        edge_properties = edge_properties or {}

        # Validate strategies exist
        for prop, strategy in edge_properties.items():
            if strategy not in self._edge_strategies:
                available = ', '.join(self._edge_strategies.keys())
                raise ValueError(
                    f"Unknown edge strategy '{strategy}' for property '{prop}'. "
                    f"Available: {available}"
                )

        if default_strategy not in self._edge_strategies:
            available = ', '.join(self._edge_strategies.keys())
            raise ValueError(
                f"Unknown default edge strategy '{default_strategy}'. "
                f"Available: {available}"
            )

        try:
            # Create simple graph
            simple_graph = nx.Graph()

            # Copy all nodes with their attributes
            simple_graph.add_nodes_from(graph.nodes(data=True))

            # Collect all edge properties
            all_properties = set()
            for u, v, data in graph.edges(data=True):
                all_properties.update(data.keys())

            # Process each unique edge (node pair)
            processed_edges = set()
            for u, v in graph.edges():
                # Skip if already processed (for undirected graphs)
                edge_key = tuple(sorted([u, v]))
                if edge_key in processed_edges:
                    continue
                processed_edges.add(edge_key)

                # Get all parallel edges between u and v
                parallel_edges_data = []
                for key in graph[u][v]:
                    parallel_edges_data.append(graph[u][v][key])

                # Aggregate properties
                aggregated_attrs = {}
                for prop in all_properties:
                    if prop in edge_properties:
                        # User specified strategy
                        strategy_name = edge_properties[prop]
                        strategy = self._edge_strategies[strategy_name]
                        aggregated_attrs[prop] = strategy.aggregate_property(
                            parallel_edges_data, prop
                        )
                    else:
                        # Use default strategy
                        if warn_on_defaults:
                            warnings.warn(
                                f"Edge property '{prop}' not specified. "
                                f"Using default strategy '{default_strategy}'"
                            )
                        strategy = self._edge_strategies[default_strategy]
                        aggregated_attrs[prop] = strategy.aggregate_property(
                            parallel_edges_data, prop
                        )

                # Add aggregated edge to simple graph
                simple_graph.add_edge(u, v, **aggregated_attrs)

            return simple_graph

        except Exception as e:
            raise AggregationError(
                f"Failed to aggregate parallel edges: {e}",
                strategy="parallel_edge_aggregation"
            ) from e

    def _validate_profile(self, profile: AggregationProfile):
        """Validate that all strategies in profile exist"""
        if profile.topology_strategy not in self._topology_strategies:
            available = ', '.join(self._topology_strategies.keys())
            raise ValueError(
                f"Unknown topology strategy: {profile.topology_strategy}. "
                f"Available: {available}"
            )

        if profile.physical_strategy and profile.physical_strategy not in self._physical_strategies:
            available = ', '.join(self._physical_strategies.keys())
            raise ValueError(
                f"Unknown physical strategy: {profile.physical_strategy}. "
                f"Available: {available}"
            )

        # Validate node property strategies
        for prop, strategy in profile.node_properties.items():
            if strategy not in self._node_strategies:
                available = ', '.join(self._node_strategies.keys())
                raise ValueError(
                    f"Unknown node strategy '{strategy}' for property '{prop}'. "
                    f"Available: {available}"
                )

        # Validate edge property strategies
        for prop, strategy in profile.edge_properties.items():
            if strategy not in self._edge_strategies:
                available = ', '.join(self._edge_strategies.keys())
                raise ValueError(
                    f"Unknown edge strategy '{strategy}' for property '{prop}'. "
                    f"Available: {available}"
                )

    @staticmethod
    def _check_property_conflicts(profile: AggregationProfile, physical_properties: set):
        """
        Check if user tried to override properties handled by physical strategy
        Issue warnings for conflicts
        """
        # Check node properties
        for prop in profile.node_properties:
            if prop in physical_properties:
                warnings.warn(
                    f"Node property '{prop}' is modified by physical strategy "
                    f"'{profile.physical_strategy}'. Your statistical aggregation "
                    f"for this property will be IGNORED.",
                    UserWarning
                )

        # Check edge properties
        for prop in profile.edge_properties:
            if prop in physical_properties:
                warnings.warn(
                    f"Edge property '{prop}' is modified by physical strategy "
                    f"'{profile.physical_strategy}'. Your statistical aggregation "
                    f"for this property will be IGNORED.",
                    UserWarning
                )

    def _aggregate_node_properties(self, graph: nx.Graph, partition_map: Dict[int, List[Any]],
                                   aggregated: nx.Graph, profile: AggregationProfile,
                                   skip_properties: set = None):
        """Aggregate node properties statistically, skipping properties aggregated in physical step"""
        skip_properties = skip_properties or set()

        # Collect all possible properties
        all_properties = set()
        for nodes in partition_map.values():
            for node in nodes:
                all_properties.update(graph.nodes[node].keys())

        for cluster_id, nodes in partition_map.items():
            node_attrs = {}

            for prop in all_properties:
                # Skip properties handled by physical aggregation
                if prop in skip_properties:
                    continue

                if prop in profile.node_properties:
                    # User specified strategy
                    strategy_name = profile.node_properties[prop]
                    strategy = self._node_strategies[strategy_name]
                    node_attrs[prop] = strategy.aggregate_property(graph, nodes, prop)
                else:
                    # Use default strategy with optional warning
                    if profile.warn_on_defaults:
                        warnings.warn(
                            f"Node property '{prop}' not specified in profile. "
                            f"Using default aggregation strategy '{profile.default_node_strategy}'"
                        )

                    if profile.default_node_strategy in self._node_strategies:
                        strategy = self._node_strategies[profile.default_node_strategy]
                        node_attrs[prop] = strategy.aggregate_property(graph, nodes, prop)
                    else:
                        # Fallback to first value
                        node_attrs[prop] = graph.nodes[nodes[0]].get(prop)

            # Update node attributes
            aggregated.nodes[cluster_id].update(node_attrs)

    def _aggregate_edge_properties(self, graph: nx.Graph, partition_map: Dict[int, List[Any]],
                                   aggregated: nx.Graph, profile: AggregationProfile,
                                   skip_properties: set = None):
        """Aggregate edge properties statistically, skipping properties aggregated in physical step"""
        skip_properties = skip_properties or set()

        # Collect all possible edge properties
        all_properties = set()
        for edge in graph.edges():
            all_properties.update(graph.edges[edge].keys())

        # For each edge in aggregated graph, find corresponding original edges
        for edge in aggregated.edges():
            cluster1, cluster2 = edge
            nodes1 = partition_map[cluster1]
            nodes2 = partition_map[cluster2]

            # Find all original edges between these clusters
            original_edges = []
            for n1 in nodes1:
                for n2 in nodes2:
                    if graph.has_edge(n1, n2):
                        original_edges.append(graph.edges[n1, n2])

            if not original_edges:
                continue

            # Aggregate properties
            edge_attrs = {}
            for prop in all_properties:
                # Skip properties handled by physical aggregation
                if prop in skip_properties:
                    continue

                if prop in profile.edge_properties:
                    # User specified strategy
                    strategy_name = profile.edge_properties[prop]
                    strategy = self._edge_strategies[strategy_name]
                    edge_attrs[prop] = strategy.aggregate_property(original_edges, prop)
                else:
                    # Use default strategy
                    if profile.warn_on_defaults:
                        warnings.warn(
                            f"Edge property '{prop}' not specified in profile. "
                            f"Using default aggregation strategy '{profile.default_edge_strategy}'"
                        )

                    if profile.default_edge_strategy in self._edge_strategies:
                        strategy = self._edge_strategies[profile.default_edge_strategy]
                        edge_attrs[prop] = strategy.aggregate_property(original_edges, prop)
                    else:
                        # Fallback to first value
                        edge_attrs[prop] = original_edges[0].get(prop) if original_edges else None

            # Update edge attributes
            aggregated.edges[edge].update(edge_attrs)

    def _register_default_strategies(self):
        """Register built-in aggregation strategies"""
        from aggregation.basic_strategies import (
            SimpleTopologyStrategy, ElectricalTopologyStrategy,
            SumNodeStrategy, AverageNodeStrategy, FirstNodeStrategy,
            SumEdgeStrategy, AverageEdgeStrategy, FirstEdgeStrategy
        )
        from aggregation.physical_strategies import (
            KronReductionStrategy
        )

        # Topology strategies
        self._topology_strategies['simple'] = SimpleTopologyStrategy()
        self._topology_strategies['electrical'] = ElectricalTopologyStrategy()

        # Physical strategies
        self._physical_strategies['kron_reduction'] = KronReductionStrategy()

        # Node property strategies
        self._node_strategies['sum'] = SumNodeStrategy()
        self._node_strategies['average'] = AverageNodeStrategy()
        self._node_strategies['first'] = FirstNodeStrategy()

        # Edge property strategies
        self._edge_strategies['sum'] = SumEdgeStrategy()
        self._edge_strategies['average'] = AverageEdgeStrategy()
        self._edge_strategies['first'] = FirstEdgeStrategy()


class PartitionAggregatorManager:
    """Main orchestrator - the primary class users interact with"""

    def __init__(self):
        self._current_graph: Optional[nx.Graph] = None
        self._current_graph_hash: Optional[str] = None
        self._current_partition: Optional[PartitionResult] = None

        # Managers for strategies
        self.input_manager = InputDataManager()
        self.partitioning_manager = PartitioningManager()
        self.aggregation_manager = AggregationManager()

    def load_data(self, strategy: str, **kwargs) -> nx.Graph:
        """Load data using specified strategy"""
        self._current_graph = self.input_manager.load(strategy, **kwargs)
        self._current_graph_hash = self._compute_graph_hash(self._current_graph)
        self._current_partition = None  # Clear any existing partition
        return self._current_graph

    def partition(self, strategy: str, **kwargs) -> PartitionResult:
        """Partition current graph and store result"""
        if not self._current_graph:
            raise ValueError("No graph loaded. Call load_data() first.")

        mapping = self.partitioning_manager.partition(
            self._current_graph, strategy, **kwargs
        )

        self._current_partition = PartitionResult(
            mapping=mapping,
            original_graph_hash=self._current_graph_hash,
            strategy_name=strategy,
            strategy_metadata={},
            n_clusters=len(set(mapping.keys()))
        )
        return self._current_partition

    def aggregate(self, partition_result: PartitionResult = None,
                  profile: AggregationProfile = None,
                  mode: AggregationMode = None, **overrides) -> nx.Graph:
        """
        Aggregate using partition result and profile

        Args:
            partition_result: Partition to use (or use stored partition)
            profile: Aggregation profile (custom configuration)
            mode: Aggregation mode (pre-defined configuration)

        Note: If both profile and mode are provided, profile takes precedence
        """
        if not self._current_graph:
            raise ValueError("No graph loaded.")

        # Use provided partition or stored partition
        partition_to_use = partition_result or self._current_partition
        if not partition_to_use:
            raise ValueError(
                "No partition available. Call partition() first or provide partition_result."
            )

        # Validate partition compatibility
        from utils import validate_graph_compatibility
        validate_graph_compatibility(partition_to_use, self._current_graph_hash)

        # Determine profile to use
        if profile is None and mode is not None:
            profile = self.aggregation_manager.get_mode_profile(mode, **overrides)
        elif profile is None:
            profile = AggregationProfile()  # Default

        # Aggregate and update current graph
        self._current_graph = self.aggregation_manager.aggregate(
            self._current_graph, partition_to_use.mapping, profile
        )

        # Update hash and clear partition since graph has changed
        self._current_graph_hash = self._compute_graph_hash(self._current_graph)
        self._current_partition = None

        return self._current_graph

    def aggregate_parallel_edges(self, edge_properties: Dict[str, str] = None,
                                 default_strategy: str = "sum",
                                 warn_on_defaults: bool = True) -> nx.Graph:
        """
        Convert current MultiGraph to simple Graph by aggregating parallel edges.

        This method aggregates all parallel edges between the same node pairs
        into single edges, using the specified aggregation strategies. The graph
        must be a MultiGraph for this operation to be meaningful.

        Args:
            edge_properties: Dict mapping property names to aggregation strategies
                           e.g., {"reactance": "average", "length": "sum"}
            default_strategy: Strategy to use for properties not specified (default: "sum")
            warn_on_defaults: Whether to warn when using default strategy

        Returns:
            Simple Graph with parallel edges aggregated

        Raises:
            ValueError: If no graph is loaded or graph is not a MultiGraph
        """
        if not self._current_graph:
            raise ValueError("No graph loaded. Call load_data() first.")

        if not isinstance(self._current_graph, nx.MultiGraph):
            raise ValueError(
                f"Current graph is not a MultiGraph (it's {type(self._current_graph).__name__}). "
                "This method is only for aggregating parallel edges in MultiGraphs."
            )

        # Aggregate parallel edges
        self._current_graph = self.aggregation_manager.aggregate_parallel_edges(
            self._current_graph,
            edge_properties=edge_properties,
            default_strategy=default_strategy,
            warn_on_defaults=warn_on_defaults
        )

        # Update hash since graph has changed
        self._current_graph_hash = self._compute_graph_hash(self._current_graph)

        # Clear partition since graph structure changed
        self._current_partition = None

        return self._current_graph

    def full_workflow(self, data_strategy: str, partition_strategy: str,
                      aggregation_profile: AggregationProfile = None,
                      aggregation_mode: AggregationMode = None,
                      **kwargs) -> nx.Graph:
        """Execute complete workflow without storing intermediates"""
        # Extract data loading params vs partition params
        data_params = {k: v for k, v in kwargs.items()
                       if k in ['node_file', 'edge_file', 'graph', 'connection_string', 'table_prefix']}
        partition_params = {k: v for k, v in kwargs.items() if k not in data_params}

        # Execute pipeline
        self.load_data(data_strategy, **data_params)
        partition_result = self.partition(partition_strategy, **partition_params)
        return self.aggregate(partition_result, aggregation_profile, aggregation_mode)

    def get_current_graph(self) -> Optional[nx.Graph]:
        """Get the current graph"""
        return self._current_graph

    def get_current_partition(self) -> Optional[PartitionResult]:
        """Get the current partition result"""
        return self._current_partition

    @staticmethod
    def _compute_graph_hash(graph: nx.Graph) -> str:
        """Compute hash for graph validation"""
        from utils import compute_graph_hash
        return compute_graph_hash(graph)

    def plot_network_on_map(self, graph: nx.Graph = None):
        """Plot current graph on geographical map if coordinates are available"""
        if not self._current_graph and graph is None:
            raise ValueError("No graph loaded.")

        from utils import interactive_plot

        if graph is not None:
            interactive_plot(graph)
        else:
            interactive_plot(self._current_graph)
