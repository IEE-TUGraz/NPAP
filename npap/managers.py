import warnings
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from npap.interfaces import (
    DataLoadingStrategy, PartitioningStrategy, AggregationProfile, AggregationMode,
    TopologyStrategy, PhysicalAggregationStrategy,
    NodePropertyStrategy, EdgePropertyStrategy, PartitionResult
)


class InputDataManager:
    """Manages data loading from different sources."""

    def __init__(self):
        self._strategies: Dict[str, DataLoadingStrategy] = {}
        self._register_default_strategies()

    def register_strategy(self, name: str, strategy: DataLoadingStrategy) -> None:
        """Register a new data loading strategy."""
        self._strategies[name] = strategy

    def load(self, strategy_name: str, **kwargs) -> nx.DiGraph | nx.MultiDiGraph:
        """Load data using specified strategy."""
        if strategy_name not in self._strategies:
            available = ', '.join(self._strategies.keys())
            raise ValueError(f"Unknown data loading strategy: {strategy_name}. Available: {available}")

        strategy = self._strategies[strategy_name]
        strategy.validate_inputs(**kwargs)
        return strategy.load(**kwargs)

    def _register_default_strategies(self) -> None:
        """Register built-in loading strategies."""
        from npap.input.csv_loader import CSVFilesStrategy
        from npap.input.networkx_loader import NetworkXDirectStrategy
        from npap.input.va_loader import VoltageAwareStrategy

        self._strategies['csv_files'] = CSVFilesStrategy()
        self._strategies['networkx_direct'] = NetworkXDirectStrategy()
        self._strategies['va_loader'] = VoltageAwareStrategy()


class PartitioningManager:
    """Manages partitioning strategies."""

    def __init__(self):
        self._strategies: Dict[str, PartitioningStrategy] = {}
        self._register_default_strategies()

    def register_strategy(self, name: str, strategy: PartitioningStrategy) -> None:
        """Register a new partitioning strategy."""
        self._strategies[name] = strategy

    def partition(self, graph: nx.DiGraph, method: str, **kwargs) -> Dict[int, List[Any]]:
        """Execute partitioning using specified strategy."""
        if method not in self._strategies:
            available = ', '.join(self._strategies.keys())
            raise ValueError(f"Unknown partitioning strategy: {method}. Available: {available}")

        return self._strategies[method].partition(graph, **kwargs)

    def _register_default_strategies(self) -> None:
        """Register built-in partitioning strategies."""
        from npap.partitioning.geographical import GeographicalPartitioning
        from npap.partitioning.electrical import ElectricalDistancePartitioning
        from npap.partitioning.va_geographical import VAGeographicalPartitioning
        from npap.partitioning.va_geographical import VAGeographicalConfig

        # Geographical distance partitioning strategies
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

        # Electrical distance partitioning strategies
        self._strategies['electrical_kmeans'] = ElectricalDistancePartitioning(
            algorithm='kmeans'
        )
        self._strategies['electrical_kmedoids'] = ElectricalDistancePartitioning(
            algorithm='kmedoids'
        )

        # Voltage-Aware Geographical partitioning strategies - Standard mode
        self._strategies['va_geographical_kmedoids_euclidean'] = VAGeographicalPartitioning(
            algorithm='kmedoids',
            distance_metric='euclidean'
        )
        self._strategies['va_geographical_kmedoids_haversine'] = VAGeographicalPartitioning(
            algorithm='kmedoids',
            distance_metric='haversine'
        )
        self._strategies['va_geographical_hierarchical'] = VAGeographicalPartitioning(
            algorithm='hierarchical',
            distance_metric='haversine'
        )

        # Voltage-Aware Geographical partitioning strategies - Proportional mode
        self._strategies['va_geographical_proportional_kmedoids_euclidean'] = VAGeographicalPartitioning(
            algorithm='kmedoids',
            distance_metric='euclidean',
            config=VAGeographicalConfig(proportional_clustering=True)
        )
        self._strategies['va_geographical_proportional_kmedoids_haversine'] = VAGeographicalPartitioning(
            algorithm='kmedoids',
            distance_metric='haversine',
            config=VAGeographicalConfig(proportional_clustering=True)
        )
        self._strategies['va_geographical_proportional_hierarchical'] = VAGeographicalPartitioning(
            algorithm='hierarchical',
            distance_metric='haversine',
            config=VAGeographicalConfig(proportional_clustering=True)
        )


class AggregationManager:
    """
    Manages aggregation strategies and orchestrates the aggregation process.

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

    def register_topology_strategy(self, name: str, strategy: TopologyStrategy) -> None:
        """Register a topology strategy."""
        self._topology_strategies[name] = strategy

    def register_physical_strategy(self, name: str, strategy: PhysicalAggregationStrategy) -> None:
        """Register a physical aggregation strategy."""
        self._physical_strategies[name] = strategy

    def register_node_strategy(self, name: str, strategy: NodePropertyStrategy) -> None:
        """Register a node property aggregation strategy."""
        self._node_strategies[name] = strategy

    def register_edge_strategy(self, name: str, strategy: EdgePropertyStrategy) -> None:
        """Register an edge property aggregation strategy"""
        self._edge_strategies[name] = strategy

    @staticmethod
    def get_mode_profile(mode: AggregationMode, **overrides) -> AggregationProfile:
        """
        Get pre-defined aggregation profile for a given mode.

        Args:
            mode: Aggregation mode
            **overrides: Override specific profile parameters

        Returns:
            AggregationProfile configured for the mode
        """
        from npap.aggregation.modes import get_mode_profile
        return get_mode_profile(mode, **overrides)

    def aggregate(self, graph: nx.DiGraph, partition_map: Dict[int, List[Any]],
                  profile: AggregationProfile = None) -> nx.DiGraph:
        """
        Execute aggregation using the specified profile.

        Aggregation is a 3-step process:
        1. Create topology (nodes + edge structure)
        2. Apply physical aggregation (if specified)
        3. Aggregate remaining properties statistically
        """
        if profile is None:
            profile = AggregationProfile()

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

        # Step 3: Aggregate node properties
        self._aggregate_node_properties(
            graph, partition_map, aggregated, profile, physical_modified_properties
        )

        # Step 4: Aggregate edge properties
        self._aggregate_edge_properties(
            graph, partition_map, aggregated, profile, physical_modified_properties
        )

        return aggregated

    def aggregate_parallel_edges(self, graph: nx.MultiDiGraph, edge_properties: Dict[str, str] = None,
                                 default_strategy: str = "sum", warn_on_defaults: bool = True) -> nx.DiGraph:
        """
        Collapse parallel edges in a MultiDiGraph to produce a simple DiGraph.

        This method aggregates all parallel edges between the same directed node pairs
        using the specified edge property strategies, converting a MultiDiGraph
        to a simple DiGraph that can be partitioned.

        For directed graphs, edges (A->B) and (B->A) are treated as separate edges
        and are aggregated independently.

        Args:
            graph: MultiDiGraph with potential parallel edges
            edge_properties: Dict mapping property names to aggregation strategies
                           e.g., {"reactance": "average", "length": "sum"}
            default_strategy: Strategy to use for properties not specified (default: "sum")
            warn_on_defaults: Whether to warn when using default strategy

        Returns:
            Simple DiGraph with parallel edges aggregated

        Raises:
            ValueError: If graph is not a MultiDiGraph
            AggregationError: If aggregation fails
        """
        from npap.exceptions import AggregationError

        # Validate input
        if not isinstance(graph, nx.MultiDiGraph):
            raise ValueError(
                f"Expected MultiDiGraph, got {type(graph).__name__}. "
                "This method is only for collapsing parallel edges in directed multigraphs."
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
            # Create simple directed graph
            simple_graph = nx.DiGraph()

            # Copy all nodes with their attributes
            simple_graph.add_nodes_from(graph.nodes(data=True))

            # Collect all edge properties
            all_properties = set()
            for u, v, data in graph.edges(data=True):
                all_properties.update(data.keys())

            # Process each unique directed edge (u->v pair)
            processed_edges = set()

            for u, v in graph.edges():
                edge_key = (u, v)
                if edge_key in processed_edges:
                    continue
                processed_edges.add(edge_key)

                # Get all parallel edges from u to v (same direction only)
                parallel_edges_data = []
                for key in graph[u][v]:
                    parallel_edges_data.append(graph[u][v][key])

                # Aggregate properties
                aggregated_attrs = {}
                warned_properties = set()

                for prop in all_properties:
                    if prop in edge_properties:
                        strategy_name = edge_properties[prop]
                        strategy = self._edge_strategies[strategy_name]
                        aggregated_attrs[prop] = strategy.aggregate_property(
                            parallel_edges_data, prop
                        )
                    else:
                        if warn_on_defaults and prop not in warned_properties:
                            warnings.warn(
                                f"Edge property '{prop}' not specified. "
                                f"Using default strategy '{default_strategy}'"
                            )
                            warned_properties.add(prop)
                        strategy = self._edge_strategies[default_strategy]
                        aggregated_attrs[prop] = strategy.aggregate_property(
                            parallel_edges_data, prop
                        )

                simple_graph.add_edge(u, v, **aggregated_attrs)

            return simple_graph

        except Exception as e:
            raise AggregationError(
                f"Failed to aggregate parallel edges: {e}",
                strategy="parallel_edge_aggregation"
            ) from e

    def _validate_profile(self, profile: AggregationProfile) -> None:
        """Validate that all strategies in profile exist."""
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
    def _check_property_conflicts(profile: AggregationProfile, physical_properties: set) -> None:
        """Check if user tried to override properties handled by physical strategy."""
        for prop in profile.node_properties:
            if prop in physical_properties:
                warnings.warn(
                    f"Node property '{prop}' is modified by physical strategy "
                    f"'{profile.physical_strategy}'. Your statistical aggregation "
                    f"for this property will be IGNORED.",
                    UserWarning
                )

        for prop in profile.edge_properties:
            if prop in physical_properties:
                warnings.warn(
                    f"Edge property '{prop}' is modified by physical strategy "
                    f"'{profile.physical_strategy}'. Your statistical aggregation "
                    f"for this property will be IGNORED.",
                    UserWarning
                )

    def _aggregate_node_properties(self, graph: nx.DiGraph, partition_map: Dict[int, List[Any]],
                                   aggregated: nx.DiGraph, profile: AggregationProfile,
                                   skip_properties: set = None) -> None:
        """Aggregate node properties statistically."""
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
                        node_attrs[prop] = graph.nodes[nodes[0]].get(prop)  # TODO: handle empty nodes list?

            # Update node attributes
            aggregated.nodes[cluster_id].update(node_attrs)

    def _aggregate_edge_properties(self, graph: nx.DiGraph, partition_map: Dict[int, List[Any]],
                                   aggregated: nx.DiGraph, profile: AggregationProfile,
                                   skip_properties: set = None) -> None:
        """Aggregate edge properties statistically."""
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

    def _register_default_strategies(self) -> None:
        """Register built-in aggregation strategies."""
        from npap.aggregation.basic_strategies import (
            SimpleTopologyStrategy, ElectricalTopologyStrategy,
            SumNodeStrategy, AverageNodeStrategy, FirstNodeStrategy,
            SumEdgeStrategy, AverageEdgeStrategy, FirstEdgeStrategy, EquivalentReactanceStrategy
        )
        from npap.aggregation.physical_strategies import (
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
        self._edge_strategies['equivalent_reactance'] = EquivalentReactanceStrategy()


class PartitionAggregatorManager:
    """Main orchestrator - the primary class users interact with."""

    def __init__(self):
        self._current_graph: Optional[nx.DiGraph] = None
        self._current_graph_hash: Optional[str] = None
        self._current_partition: Optional[PartitionResult] = None

        # Managers for strategies
        self.input_manager = InputDataManager()
        self.partitioning_manager = PartitioningManager()
        self.aggregation_manager = AggregationManager()

    def load_data(self, strategy: str, **kwargs) -> nx.DiGraph | nx.MultiDiGraph:
        """Load data using specified strategy."""
        self._current_graph = self.input_manager.load(strategy, **kwargs)
        self._current_graph_hash = self._compute_graph_hash(self._current_graph)
        self._current_partition = None  # Clear any existing partition
        return self._current_graph

    def partition(self, strategy: str, **kwargs) -> PartitionResult:
        """Partition current graph and store result."""
        if not self._current_graph:
            raise ValueError("No graph loaded. Call load_data() first.")

        # Check if graph is MultiDiGraph
        if isinstance(self._current_graph, nx.MultiDiGraph):
            raise ValueError(
                "Cannot partition MultiDiGraph directly. MultiDiGraphs contain parallel edges "
                "that must be aggregated first. Please call manager.aggregate_parallel_edges() before partitioning."
            )

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
                  mode: AggregationMode = None, **overrides) -> nx.DiGraph:
        """
        Aggregate using partition result and profile.

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
        from npap.utils import validate_graph_compatibility
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
                                 warn_on_defaults: bool = True) -> nx.DiGraph:
        """
        Convert current MultiDiGraph to simple DiGraph by aggregating parallel edges.

        This method aggregates all parallel edges between the same directed node pairs
        into single edges, using the specified aggregation strategies. The graph
        must be a MultiDiGraph for this operation to be meaningful.

        For directed graphs, edges (A->B) and (B->A) are treated as separate edges.

        Args:
            edge_properties: Dict mapping property names to aggregation strategies
                           e.g., {"reactance": "average", "length": "sum"}
            default_strategy: Strategy to use for properties not specified (default: "sum")
            warn_on_defaults: Whether to warn when using default strategy

        Returns:
            Simple DiGraph with parallel edges aggregated

        Raises:
            ValueError: If no graph is loaded or graph is not a MultiDiGraph
        """
        if not self._current_graph:
            raise ValueError("No graph loaded. Call load_data() first.")

        if not isinstance(self._current_graph, nx.MultiDiGraph):
            raise ValueError(
                f"Current graph is not a MultiDiGraph (it's {type(self._current_graph).__name__}). "
                "This method is only for aggregating parallel edges in MultiDiGraphs."
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
                      **kwargs) -> nx.DiGraph:
        """
        Execute complete workflow without storing intermediates.

        If a MultiDiGraph is loaded, parallel edges will be automatically aggregated
        before partitioning. If a voltage-aware partitioning strategy is selected,
        voltage levels will be grouped first in 220kV and 380kV voltage levels.

        Args:
            data_strategy: Data loading strategy name
            partition_strategy: Partitioning strategy name
            aggregation_profile: Aggregation profile (custom)
            aggregation_mode: Aggregation mode (pre-defined)
            **kwargs: Parameters for data loading, parallel edge aggregation, partitioning

        Returns:
            Aggregated graph
        """
        # Parameters for data loading
        data_params = {k: v for k, v in kwargs.items()
                       if k in ['node_file', 'edge_file', 'line_file', 'transformer_file',
                                'graph', 'connection_string', 'table_prefix',
                                'delimiter', 'decimal', 'node_id_col', 'edge_from_col', 'edge_to_col',
                                'bidirectional']}

        # Parameters for parallel edge aggregation (if MultiDiGraph)
        parallel_edge_params = {k: v for k, v in kwargs.items()
                                if k in ['edge_properties', 'parallel_edge_default_strategy',
                                         'parallel_edge_warn_on_defaults']}

        # Parameters for partitioning
        partition_params = {k: v for k, v in kwargs.items()
                            if k not in data_params and k not in parallel_edge_params}

        # Step 1: Load data
        self.load_data(data_strategy, **data_params)

        # Step 2: If MultiDiGraph, aggregate parallel edges first
        if isinstance(self._current_graph, nx.MultiDiGraph):
            print("→ MultiDiGraph detected in workflow. Auto-aggregating parallel edges...")
            self.aggregate_parallel_edges(**parallel_edge_params)

        # For voltage-aware strategies, group voltages first
        if partition_strategy.startswith('va_'):
            self.group_by_voltage_levels([220, 380])

        # Step 3: Partition
        partition_result = self.partition(partition_strategy, **partition_params)

        # Step 4: Aggregate clusters
        return self.aggregate(partition_result, aggregation_profile, aggregation_mode)

    def get_current_graph(self) -> Optional[nx.DiGraph]:
        """Get the current graph."""
        return self._current_graph

    def get_current_partition(self) -> Optional[PartitionResult]:
        """Get the current partition result."""
        return self._current_partition

    @staticmethod
    def _compute_graph_hash(graph: nx.DiGraph) -> str:
        """Compute hash for graph validation."""
        from npap.utils import compute_graph_hash
        return compute_graph_hash(graph)

    def plot_network(self, style: str = 'simple', graph: nx.DiGraph = None,
                     show: bool = True, **kwargs):
        """
        Plot the network on an interactive map.

        Args:
            style: Plot style
                - 'simple': All edges same color (fast, minimal)
                - 'voltage_aware': Edges colored by type (line/trafo/dc_link),
                                   thickness by voltage level
                - 'clustered': Nodes colored by cluster assignment (requires
                               prior partitioning)
            graph: Optional graph to plot (uses current graph if not provided)
            show: Whether to display immediately (default: True)
            **kwargs: Additional configuration options:
                - show_lines: bool = True
                - show_trafos: bool = True
                - show_dc_links: bool = True
                - show_nodes: bool = True
                - line_high_voltage_color: str = "#029E73" (green)
                - line_low_voltage_color: str = "#CA9161" (brown)
                - trafo_color: str = "#ECE133" (yellow)
                - dc_link_color: str = "#CC78BC" (pink)
                - node_color: str = "#0173B2" (blue)
                - edge_width: float = 1.5
                - node_size: int = 5
                - title: str = None
                - map_style: str = "carto-positron"
                - map_center_lat: float = 57.5
                - map_center_lon: float = 14.0
                - map_zoom: float = 3.7
                - cluster_colorscale: str = "Viridis" (for clustered style)

        Returns:
            Plotly Figure object
        """
        from npap.visualization import plot_network

        target_graph = graph if graph is not None else self._current_graph
        if target_graph is None:
            raise ValueError("No graph loaded. Call load_data() first.")

        # For clustered style, pass the partition map
        partition_map = None
        if style == 'clustered':
            if self._current_partition is None:
                raise ValueError(
                    "Cannot create clustered plot without partitioning. "
                    "Call partition() first."
                )
            partition_map = self._current_partition.mapping

        return plot_network(
            graph=target_graph,
            style=style,
            partition_map=partition_map,
            show=show,
            **kwargs
        )

    def group_by_voltage_levels(self, target_levels: List[float],
                                voltage_attr: str = 'voltage',
                                store_original: bool = True,
                                handle_missing: str = 'infer') -> Dict[str, Any]:
        """
        Group buses voltage levels to predefined target values.

        This method reassigns each node's voltage to the nearest target voltage level,
        creating clean voltage "islands" for subsequent voltage-aware partitioning.

        Args:
            target_levels: List of target voltage levels (kV) to harmonize to.
                          Example: [220, 380] for European transmission grid.
            voltage_attr: Node attribute containing voltage level (default: 'voltage').
            store_original: If True, stores original voltage in 'original_{voltage_attr}'.
            handle_missing: How to handle nodes without voltage data:
                           - 'infer': Infer from connected neighbors
                           - 'nearest': Assign to nearest target level
                           - 'error': Raise an error
                           - 'skip': Leave as None

        Returns:
            Summary dict with:
                - 'total_nodes': Total number of nodes processed
                - 'reassignments': Dict mapping original_voltage -> (target_voltage, count)
                - 'missing_handled': Number of nodes with missing voltage
                - 'voltage_distribution': Dict mapping target_voltage -> node_count

        Raises:
            ValueError: If no graph loaded, target_levels empty, or handle_missing='error'
                       with missing voltages.
        """
        if not self._current_graph:
            raise ValueError("No graph loaded. Call load_data() first.")

        if not target_levels:
            raise ValueError("target_levels cannot be empty.")

        target_levels = sorted(target_levels)
        graph = self._current_graph

        # Statistics tracking
        reassignments: Dict[Any, Tuple[float, int]] = {}  # original -> (target, count)
        missing_nodes: List[Any] = []
        voltage_distribution: Dict[float, int] = {level: 0 for level in target_levels}

        # First pass: identify missing voltages and reassign known ones
        for node in graph.nodes():
            node_data = graph.nodes[node]
            original_voltage = node_data.get(voltage_attr)

            # Try fallback attributes
            if original_voltage is None:
                original_voltage = node_data.get('voltage', node_data.get('v_nom'))

            if original_voltage is None:
                missing_nodes.append(node)
                continue

            # Store original if requested
            if store_original:
                node_data[f'original_{voltage_attr}'] = original_voltage

            # Find nearest target level
            if isinstance(original_voltage, (int, float)):
                target = min(target_levels, key=lambda t: abs(t - original_voltage))
            else:
                # Non-numeric voltage - try to parse or skip
                try:
                    numeric_v = float(original_voltage)
                    target = min(target_levels, key=lambda t: abs(t - numeric_v))
                except (ValueError, TypeError):
                    missing_nodes.append(node)
                    continue

            # Reassign
            node_data[voltage_attr] = target
            voltage_distribution[target] += 1

            # Track reassignment
            orig_key = round(original_voltage, 1) if isinstance(original_voltage, float) else original_voltage
            if orig_key not in reassignments:
                reassignments[orig_key] = (target, 0)
            reassignments[orig_key] = (reassignments[orig_key][0], reassignments[orig_key][1] + 1)

        # Handle missing voltages
        if missing_nodes:
            if handle_missing == 'error':
                raise ValueError(
                    f"{len(missing_nodes)} nodes have missing voltage data. "
                    f"First few: {missing_nodes[:5]}"
                )

            elif handle_missing == 'infer':
                # Infer from connected neighbors
                inferred_count = 0
                still_missing = []

                for node in missing_nodes:
                    neighbor_voltages = []
                    for neighbor in graph.neighbors(node):
                        nv = graph.nodes[neighbor].get(voltage_attr)
                        if nv is not None and nv in target_levels:
                            neighbor_voltages.append(nv)

                    # Also check predecessors for directed graphs
                    if hasattr(graph, 'predecessors'):
                        for pred in graph.predecessors(node):
                            pv = graph.nodes[pred].get(voltage_attr)
                            if pv is not None and pv in target_levels:
                                neighbor_voltages.append(pv)

                    if neighbor_voltages:
                        # Use most common neighbor voltage
                        from collections import Counter
                        target = Counter(neighbor_voltages).most_common(1)[0][0]
                        graph.nodes[node][voltage_attr] = target
                        if store_original:
                            graph.nodes[node][f'original_{voltage_attr}'] = None
                        voltage_distribution[target] += 1
                        inferred_count += 1
                    else:
                        still_missing.append(node)

                if still_missing:
                    print(f"Warning: {len(still_missing)} nodes could not be inferred. "
                          f"Assigning to nearest target level.")
                    # Assign remaining to first target level (arbitrary but consistent)
                    default_target = target_levels[0]
                    for node in still_missing:
                        graph.nodes[node][voltage_attr] = default_target
                        if store_original:
                            graph.nodes[node][f'original_{voltage_attr}'] = None
                        voltage_distribution[default_target] += 1

                print(f"Inferred voltage for {inferred_count} nodes from neighbors.")

            elif handle_missing == 'nearest':
                # Assign to first target level (arbitrary)
                default_target = target_levels[0]
                for node in missing_nodes:
                    graph.nodes[node][voltage_attr] = default_target
                    if store_original:
                        graph.nodes[node][f'original_{voltage_attr}'] = None
                    voltage_distribution[default_target] += 1

            elif handle_missing == 'skip':
                # Leave as None - will form separate cluster
                print(f"Warning: {len(missing_nodes)} nodes have no voltage (will cluster separately).")

        # Update graph hash since we modified node attributes
        self._current_graph_hash = self._compute_graph_hash(self._current_graph)

        return {
            'total_nodes': len(list(graph.nodes())),
            'reassignments': reassignments,
            'missing_handled': len(missing_nodes),
            'voltage_distribution': voltage_distribution,
            'target_levels': target_levels
        }
