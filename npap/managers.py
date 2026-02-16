from typing import Any

import networkx as nx

from npap.interfaces import (
    AggregationMode,
    AggregationProfile,
    DataLoadingStrategy,
    EdgePropertyStrategy,
    NodePropertyStrategy,
    PartitioningStrategy,
    PartitionResult,
    PhysicalAggregationStrategy,
    TopologyStrategy,
)
from npap.logging import LogCategory, log_debug, log_info, log_warning
from npap.partitioning import VAElectricalDistancePartitioning


class InputDataManager:
    """Manages data loading from different sources."""

    def __init__(self):
        self._strategies: dict[str, DataLoadingStrategy] = {}
        self._register_default_strategies()

    def register_strategy(self, name: str, strategy: DataLoadingStrategy) -> None:
        """Register a new data loading strategy."""
        self._strategies[name] = strategy
        log_debug(f"Registered input strategy: {name}", LogCategory.MANAGER)

    def load(self, strategy_name: str, **kwargs) -> nx.DiGraph | nx.MultiDiGraph:
        """Load data using specified strategy."""
        if strategy_name not in self._strategies:
            available = ", ".join(self._strategies.keys())
            raise ValueError(
                f"Unknown data loading strategy: {strategy_name}. Available: {available}"
            )

        log_info(f"Loading data with strategy: {strategy_name}", LogCategory.INPUT)
        strategy = self._strategies[strategy_name]
        strategy.validate_inputs(**kwargs)
        return strategy.load(**kwargs)

    def _register_default_strategies(self) -> None:
        """Register built-in loading strategies."""
        from npap.input.csv_loader import CSVFilesStrategy
        from npap.input.networkx_loader import NetworkXDirectStrategy
        from npap.input.va_loader import VoltageAwareStrategy

        self._strategies["csv_files"] = CSVFilesStrategy()
        self._strategies["networkx_direct"] = NetworkXDirectStrategy()
        self._strategies["va_loader"] = VoltageAwareStrategy()
        log_debug("Registered default input strategies", LogCategory.MANAGER)


class PartitioningManager:
    """Manages partitioning strategies."""

    def __init__(self):
        self._strategies: dict[str, PartitioningStrategy] = {}
        self._register_default_strategies()

    def register_strategy(self, name: str, strategy: PartitioningStrategy) -> None:
        """Register a new partitioning strategy."""
        self._strategies[name] = strategy
        log_debug(f"Registered partitioning strategy: {name}", LogCategory.MANAGER)

    def partition(self, graph: nx.DiGraph, method: str, **kwargs) -> dict[int, list[Any]]:
        """Execute partitioning using specified strategy."""
        if method not in self._strategies:
            available = ", ".join(self._strategies.keys())
            raise ValueError(f"Unknown partitioning strategy: {method}. Available: {available}")

        log_info(f"Partitioning graph with strategy: {method}", LogCategory.PARTITIONING)
        return self._strategies[method].partition(graph, **kwargs)

    def _register_default_strategies(self) -> None:
        """Register built-in partitioning strategies."""
        from npap.partitioning.electrical import ElectricalDistancePartitioning
        from npap.partitioning.geographical import GeographicalPartitioning
        from npap.partitioning.va_geographical import (
            VAGeographicalConfig,
            VAGeographicalPartitioning,
        )

        # Geographical distance partitioning strategies
        self._strategies["geographical_kmeans"] = GeographicalPartitioning(
            algorithm="kmeans", distance_metric="euclidean"
        )
        self._strategies["geographical_kmedoids_euclidean"] = GeographicalPartitioning(
            algorithm="kmedoids", distance_metric="euclidean"
        )
        self._strategies["geographical_kmedoids_haversine"] = GeographicalPartitioning(
            algorithm="kmedoids", distance_metric="haversine"
        )
        self._strategies["geographical_dbscan_euclidean"] = GeographicalPartitioning(
            algorithm="dbscan", distance_metric="euclidean"
        )
        self._strategies["geographical_dbscan_haversine"] = GeographicalPartitioning(
            algorithm="dbscan", distance_metric="haversine"
        )
        self._strategies["geographical_hierarchical"] = GeographicalPartitioning(
            algorithm="hierarchical", distance_metric="euclidean"
        )
        self._strategies["geographical_hdbscan_euclidean"] = GeographicalPartitioning(
            algorithm="hdbscan", distance_metric="euclidean"
        )
        self._strategies["geographical_hdbscan_haversine"] = GeographicalPartitioning(
            algorithm="hdbscan", distance_metric="haversine"
        )

        # Electrical distance partitioning strategies
        self._strategies["electrical_kmeans"] = ElectricalDistancePartitioning(algorithm="kmeans")
        self._strategies["electrical_kmedoids"] = ElectricalDistancePartitioning(
            algorithm="kmedoids"
        )

        # Voltage-Aware Geographical partitioning strategies - Standard mode
        self._strategies["va_geographical_kmedoids_euclidean"] = VAGeographicalPartitioning(
            algorithm="kmedoids", distance_metric="euclidean"
        )
        self._strategies["va_geographical_kmedoids_haversine"] = VAGeographicalPartitioning(
            algorithm="kmedoids", distance_metric="haversine"
        )
        self._strategies["va_geographical_hierarchical"] = VAGeographicalPartitioning(
            algorithm="hierarchical", distance_metric="haversine"
        )

        # Voltage-Aware Geographical partitioning strategies - Proportional mode
        self._strategies["va_geographical_proportional_kmedoids_euclidean"] = (
            VAGeographicalPartitioning(
                algorithm="kmedoids",
                distance_metric="euclidean",
                config=VAGeographicalConfig(proportional_clustering=True),
            )
        )
        self._strategies["va_geographical_proportional_kmedoids_haversine"] = (
            VAGeographicalPartitioning(
                algorithm="kmedoids",
                distance_metric="haversine",
                config=VAGeographicalConfig(proportional_clustering=True),
            )
        )
        self._strategies["va_geographical_proportional_hierarchical"] = VAGeographicalPartitioning(
            algorithm="hierarchical",
            distance_metric="haversine",
            config=VAGeographicalConfig(proportional_clustering=True),
        )

        # Voltage-Aware Electrical partitioning strategies
        self._strategies["va_electrical_kmedoids"] = VAElectricalDistancePartitioning(
            algorithm="kmedoids"
        )
        self._strategies["va_electrical_hierarchical"] = VAElectricalDistancePartitioning(
            algorithm="hierarchical"
        )

        log_debug("Registered default partitioning strategies", LogCategory.MANAGER)


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
        self._topology_strategies: dict[str, TopologyStrategy] = {}

        # Physical aggregation strategies (electrical laws)
        self._physical_strategies: dict[str, PhysicalAggregationStrategy] = {}

        # Statistical property aggregation strategies
        self._node_strategies: dict[str, NodePropertyStrategy] = {}
        self._edge_strategies: dict[str, EdgePropertyStrategy] = {}

        self._register_default_strategies()

    def register_topology_strategy(self, name: str, strategy: TopologyStrategy) -> None:
        """Register a topology strategy."""
        self._topology_strategies[name] = strategy
        log_debug(f"Registered topology strategy: {name}", LogCategory.MANAGER)

    def register_physical_strategy(self, name: str, strategy: PhysicalAggregationStrategy) -> None:
        """Register a physical aggregation strategy."""
        self._physical_strategies[name] = strategy
        log_debug(f"Registered physical strategy: {name}", LogCategory.MANAGER)

    def register_node_strategy(self, name: str, strategy: NodePropertyStrategy) -> None:
        """Register a node property aggregation strategy."""
        self._node_strategies[name] = strategy

    def register_edge_strategy(self, name: str, strategy: EdgePropertyStrategy) -> None:
        """Register an edge property aggregation strategy."""
        self._edge_strategies[name] = strategy

    @staticmethod
    def get_mode_profile(mode: AggregationMode, **overrides) -> AggregationProfile:
        """
        Get pre-defined aggregation profile for a given mode.

        Parameters
        ----------
        mode : AggregationMode
            Aggregation mode.
        **overrides : dict
            Override specific profile parameters.

        Returns
        -------
        AggregationProfile
            AggregationProfile configured for the mode.
        """
        from npap.aggregation.modes import get_mode_profile

        return get_mode_profile(mode, **overrides)

    def aggregate(
        self,
        graph: nx.DiGraph,
        partition_map: dict[int, list[Any]],
        profile: AggregationProfile = None,
    ) -> nx.DiGraph | nx.MultiDiGraph:
        """
        Execute aggregation using the specified profile.

        Aggregation is a 3-step process:

        1. Create topology (nodes + edge structure)
        2. Apply physical aggregation (if specified)
        3. Aggregate remaining properties statistically

        Parameters
        ----------
        graph : nx.DiGraph
            Graph to aggregate.
        partition_map : dict[int, list[Any]]
            Mapping of cluster_id to list of node_ids.
        profile : AggregationProfile, optional
            Aggregation profile (uses defaults if not provided).

        Returns
        -------
        nx.DiGraph or nx.MultiDiGraph
            Aggregated graph.  Returns a ``MultiDiGraph`` when
            ``profile.edge_type_properties`` is populated.
        """
        if profile is None:
            profile = AggregationProfile()

        log_info(
            f"Aggregating graph: {len(partition_map)} clusters, "
            f"topology={profile.topology_strategy}",
            LogCategory.AGGREGATION,
        )

        # Validate strategies exist
        self._validate_profile(profile)

        # Pre-compute mappings for efficient aggregation
        from npap.aggregation.basic_strategies import (
            build_cluster_edge_map,
            build_node_to_cluster_map,
        )

        node_to_cluster = build_node_to_cluster_map(partition_map)
        cluster_edge_map = build_cluster_edge_map(graph, node_to_cluster)

        # Step 1: Create topology
        topology_strategy = self._topology_strategies[profile.topology_strategy]
        aggregated = topology_strategy.create_topology(graph, partition_map)
        log_info(
            f"Created topology: {aggregated.number_of_nodes()} nodes, "
            f"{aggregated.number_of_edges()} edges",
            LogCategory.AGGREGATION,
        )

        # Track which properties are handled by physical aggregation
        physical_modified_properties = set()

        # Step 2: Apply physical aggregation (if specified)
        if profile.physical_strategy:
            physical_strategy = self._physical_strategies[profile.physical_strategy]

            # Validate topology compatibility
            if topology_strategy.__class__.__name__ != physical_strategy.required_topology:
                log_warning(
                    f"Physical strategy '{profile.physical_strategy}' recommends '{physical_strategy.required_topology}' topology, "
                    f"but '{profile.topology_strategy}' is being used. Results may be incorrect.",
                    LogCategory.AGGREGATION,
                )

            # Apply physical aggregation
            aggregated = physical_strategy.aggregate(
                graph,
                partition_map,
                aggregated,
                profile.physical_properties,
                profile.physical_parameters,
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
        if profile.edge_type_properties:
            aggregated = self._aggregate_typed_edge_properties(
                graph,
                partition_map,
                aggregated,
                profile,
                physical_modified_properties,
            )
        else:
            self._aggregate_edge_properties(
                graph,
                partition_map,
                aggregated,
                profile,
                physical_modified_properties,
                cluster_edge_map,
            )

        log_info(
            f"Aggregation complete: {aggregated.number_of_nodes()} nodes, "
            f"{aggregated.number_of_edges()} edges",
            LogCategory.AGGREGATION,
        )

        return aggregated

    def aggregate_parallel_edges(
        self,
        graph: nx.MultiDiGraph,
        edge_properties: dict[str, str] = None,
        default_strategy: str = "sum",
        warn_on_defaults: bool = True,
    ) -> nx.DiGraph:
        """
        Collapse parallel edges in a MultiDiGraph to produce a simple DiGraph.

        This method aggregates all parallel edges between the same directed node pairs
        using the specified edge property strategies, converting a MultiDiGraph
        to a simple DiGraph that can be partitioned.

        For directed graphs, edges (A->B) and (B->A) are treated as separate edges
        and are aggregated independently.

        Parameters
        ----------
        graph : nx.MultiDiGraph
            MultiDiGraph with potential parallel edges.
        edge_properties : dict[str, str], optional
            Dict mapping property names to aggregation strategies.
            Example: ``{"reactance": "average", "length": "sum"}``.
        default_strategy : str
            Strategy to use for properties not specified.
        warn_on_defaults : bool
            Whether to warn when using default strategy.

        Returns
        -------
        nx.DiGraph
            Simple DiGraph with parallel edges aggregated.

        Raises
        ------
        ValueError
            If graph is not a MultiDiGraph.
        AggregationError
            If aggregation fails.
        """
        from collections import defaultdict

        from npap.exceptions import AggregationError

        # Validate input
        if not isinstance(graph, nx.MultiDiGraph):
            raise ValueError(
                f"Expected MultiDiGraph, got {type(graph).__name__}. "
                "This method is only for collapsing parallel edges in directed multigraphs."
            )

        log_info(
            f"Aggregating parallel edges in MultiDiGraph with {graph.number_of_edges()} edges",
            LogCategory.AGGREGATION,
        )

        edge_properties = edge_properties or {}

        # Validate strategies exist
        for prop, strategy in edge_properties.items():
            if strategy not in self._edge_strategies:
                available = ", ".join(self._edge_strategies.keys())
                raise ValueError(
                    f"Unknown edge strategy '{strategy}' for property '{prop}'. "
                    f"Available: {available}"
                )

        if default_strategy not in self._edge_strategies:
            available = ", ".join(self._edge_strategies.keys())
            raise ValueError(
                f"Unknown default edge strategy '{default_strategy}'. Available: {available}"
            )

        try:
            # Create simple directed graph with all nodes
            simple_graph = nx.DiGraph()
            simple_graph.add_nodes_from(graph.nodes(data=True))

            # Group edges by (u, v) and collect all properties
            edge_groups: dict[tuple, list[dict]] = defaultdict(list)
            all_properties: set[str] = set()

            for u, v, data in graph.edges(data=True):
                edge_groups[(u, v)].append(data)
                all_properties.update(data.keys())

            # Pre-resolve strategies for each property
            property_strategies = self._resolve_property_strategies(
                properties=all_properties,
                user_specified=edge_properties,
                available_strategies=self._edge_strategies,
                default_strategy=default_strategy,
                warn_on_defaults=warn_on_defaults,
                property_type="Edge",
            )

            # Process each unique edge group
            parallel_count = 0

            for (u, v), parallel_edges_data in edge_groups.items():
                if len(parallel_edges_data) > 1:
                    parallel_count += 1

                # Aggregate all properties using pre-resolved strategies
                aggregated_attrs = {
                    prop: strategy.aggregate_property(parallel_edges_data, prop)
                    for prop, strategy in property_strategies.items()
                }

                simple_graph.add_edge(u, v, **aggregated_attrs)

            log_info(
                f"Parallel edge aggregation complete: {parallel_count} parallel edge groups collapsed",
                LogCategory.AGGREGATION,
            )

            return simple_graph

        except Exception as e:
            raise AggregationError(
                f"Failed to aggregate parallel edges: {e}",
                strategy="parallel_edge_aggregation",
            ) from e

    def _validate_profile(self, profile: AggregationProfile) -> None:
        """Validate that all strategies in profile exist."""
        if profile.topology_strategy not in self._topology_strategies:
            available = ", ".join(self._topology_strategies.keys())
            raise ValueError(
                f"Unknown topology strategy: {profile.topology_strategy}. Available: {available}"
            )

        if profile.physical_strategy and profile.physical_strategy not in self._physical_strategies:
            available = ", ".join(self._physical_strategies.keys())
            raise ValueError(
                f"Unknown physical strategy: {profile.physical_strategy}. Available: {available}"
            )

        # Validate node property strategies
        for prop, strategy in profile.node_properties.items():
            if strategy not in self._node_strategies:
                available = ", ".join(self._node_strategies.keys())
                raise ValueError(
                    f"Unknown node strategy '{strategy}' for property '{prop}'. "
                    f"Available: {available}"
                )

        # Validate edge property strategies
        for prop, strategy in profile.edge_properties.items():
            if strategy not in self._edge_strategies:
                available = ", ".join(self._edge_strategies.keys())
                raise ValueError(
                    f"Unknown edge strategy '{strategy}' for property '{prop}'. "
                    f"Available: {available}"
                )

        # Validate per-type edge property strategies
        for edge_type, type_strategies in profile.edge_type_properties.items():
            for prop, strategy in type_strategies.items():
                if strategy not in self._edge_strategies:
                    available = ", ".join(self._edge_strategies.keys())
                    raise ValueError(
                        f"Unknown edge strategy '{strategy}' for property '{prop}' "
                        f"in edge type '{edge_type}'. Available: {available}"
                    )

    @staticmethod
    def _check_property_conflicts(profile: AggregationProfile, physical_properties: set) -> None:
        """Check if user tried to override properties handled by physical strategy."""
        for prop in profile.node_properties:
            if prop in physical_properties:
                log_warning(
                    f"Node property '{prop}' is modified by physical strategy "
                    f"'{profile.physical_strategy}'. User statistical aggregation for this property will be IGNORED.",
                    LogCategory.AGGREGATION,
                )

        for prop in profile.edge_properties:
            if prop in physical_properties:
                log_warning(
                    f"Edge property '{prop}' is modified by physical strategy "
                    f"'{profile.physical_strategy}'. User statistical aggregation for this property will be IGNORED.",
                    LogCategory.AGGREGATION,
                )

    @staticmethod
    def _resolve_property_strategies(
        properties: set[str],
        user_specified: dict[str, str],
        available_strategies: dict[str, Any],
        default_strategy: str,
        warn_on_defaults: bool,
        property_type: str,
    ) -> dict[str, Any]:
        """
        Resolve strategies for a set of properties.

        Parameters
        ----------
        properties : set[str]
            Set of property names to resolve.
        user_specified : dict[str, str]
            User-specified property to strategy name mapping.
        available_strategies : dict[str, Any]
            Available strategy name to strategy object mapping.
        default_strategy : str
            Default strategy name to use.
        warn_on_defaults : bool
            Whether to warn when using default strategy.
        property_type : str
            "Node" or "Edge" for warning messages.

        Returns
        -------
        dict[str, Any]
            Dict mapping property name to strategy object (or None for fallback).
        """
        property_strategies: dict[str, Any] = {}
        warned_properties: set[str] = set()

        for prop in properties:
            if prop in user_specified:
                strategy_name = user_specified[prop]
                property_strategies[prop] = available_strategies[strategy_name]
            elif default_strategy in available_strategies:
                if warn_on_defaults and prop not in warned_properties:
                    log_warning(
                        f"{property_type} property '{prop}' not specified. "
                        f"Using default strategy '{default_strategy}'",
                        LogCategory.AGGREGATION,
                    )
                    warned_properties.add(prop)
                property_strategies[prop] = available_strategies[default_strategy]
            else:
                property_strategies[prop] = None  # Will use fallback

        return property_strategies

    def _aggregate_node_properties(
        self,
        graph: nx.DiGraph,
        partition_map: dict[int, list[Any]],
        aggregated: nx.DiGraph,
        profile: AggregationProfile,
        skip_properties: set = None,
    ) -> None:
        """Aggregate node properties statistically."""
        skip_properties = skip_properties or set()

        # Collect all possible properties in single pass
        all_properties: set[str] = set()
        for nodes in partition_map.values():
            for node in nodes:
                all_properties.update(graph.nodes[node].keys())

        # Remove properties to skip
        properties_to_aggregate = all_properties - skip_properties

        # Pre-resolve strategies
        property_strategies = self._resolve_property_strategies(
            properties=properties_to_aggregate,
            user_specified=profile.node_properties,
            available_strategies=self._node_strategies,
            default_strategy=profile.default_node_strategy,
            warn_on_defaults=profile.warn_on_defaults,
            property_type="Node",
        )

        # Aggregate properties for each cluster
        for cluster_id, nodes in partition_map.items():
            node_attrs = {}

            for prop, strategy in property_strategies.items():
                if strategy is not None:
                    node_attrs[prop] = strategy.aggregate_property(graph, nodes, prop)
                elif nodes:
                    # Fallback to first value
                    node_attrs[prop] = graph.nodes[nodes[0]].get(prop)

            # Batch update node attributes
            aggregated.nodes[cluster_id].update(node_attrs)

    def _aggregate_edge_properties(
        self,
        graph: nx.DiGraph,
        partition_map: dict[int, list[Any]],
        aggregated: nx.DiGraph,
        profile: AggregationProfile,
        skip_properties: set = None,
        cluster_edge_map: dict[tuple[int, int], list[dict[str, Any]]] = None,
    ) -> None:
        """Aggregate edge properties statistically."""
        skip_properties = skip_properties or set()

        # Build cluster_edge_map if not provided
        if cluster_edge_map is None:
            from npap.aggregation.basic_strategies import (
                build_cluster_edge_map,
                build_node_to_cluster_map,
            )

            node_to_cluster = build_node_to_cluster_map(partition_map)
            cluster_edge_map = build_cluster_edge_map(graph, node_to_cluster)

        # Collect all possible edge properties in single pass
        all_properties: set[str] = set()
        for edge_list in cluster_edge_map.values():
            for edge_data in edge_list:
                all_properties.update(edge_data.keys())

        # Remove properties to skip
        properties_to_aggregate = all_properties - skip_properties

        # Pre-resolve strategies
        property_strategies = self._resolve_property_strategies(
            properties=properties_to_aggregate,
            user_specified=profile.edge_properties,
            available_strategies=self._edge_strategies,
            default_strategy=profile.default_edge_strategy,
            warn_on_defaults=profile.warn_on_defaults,
            property_type="Edge",
        )

        # Aggregate properties for each edge using pre-computed mapping (O(1) lookup)
        for edge in aggregated.edges():
            cluster1, cluster2 = edge

            original_edges = cluster_edge_map.get((cluster1, cluster2), [])

            if not original_edges:
                continue

            # Aggregate properties
            edge_attrs = {}
            for prop, strategy in property_strategies.items():
                if strategy is not None:
                    edge_attrs[prop] = strategy.aggregate_property(original_edges, prop)
                elif original_edges:
                    # Fallback to first value
                    edge_attrs[prop] = original_edges[0].get(prop)

            # Batch update edge attributes
            aggregated.edges[edge].update(edge_attrs)

    def _aggregate_typed_edge_properties(
        self,
        graph: nx.DiGraph,
        partition_map: dict[int, list[Any]],
        aggregated: nx.DiGraph,
        profile: AggregationProfile,
        skip_properties: set = None,
    ) -> nx.MultiDiGraph:
        """Aggregate edge properties per edge type, returning a MultiDiGraph.

        When ``profile.edge_type_properties`` is populated, edges are grouped
        by their ``"type"`` attribute and aggregated independently using
        per-type strategy dicts.  The result is a ``MultiDiGraph`` so that
        multiple edges (one per type) can connect the same cluster pair.

        Parameters
        ----------
        graph : nx.DiGraph
            Original graph.
        partition_map : dict[int, list[Any]]
            Cluster mapping.
        aggregated : nx.DiGraph
            Topology graph with aggregated node properties.
        profile : AggregationProfile
            Aggregation profile with ``edge_type_properties``.
        skip_properties : set, optional
            Properties already handled by physical aggregation.

        Returns
        -------
        nx.MultiDiGraph
            Aggregated graph with typed edges.
        """
        skip_properties = skip_properties or set()

        from npap.aggregation.basic_strategies import (
            build_node_to_cluster_map,
            build_typed_cluster_edge_map,
        )

        type_attr = profile.edge_type_attribute

        node_to_cluster = build_node_to_cluster_map(partition_map)
        typed_map = build_typed_cluster_edge_map(graph, node_to_cluster, type_attribute=type_attr)

        # Convert the DiGraph topology to a MultiDiGraph, preserving nodes
        multi = nx.MultiDiGraph()
        multi.add_nodes_from(aggregated.nodes(data=True))

        for edge_type, cluster_edges in typed_map.items():
            # Determine which strategy dict to use for this type
            user_specified = profile.edge_type_properties.get(edge_type, profile.edge_properties)

            # Collect all properties across edges of this type
            all_properties: set[str] = set()
            for edge_list in cluster_edges.values():
                for edge_data in edge_list:
                    all_properties.update(edge_data.keys())

            # Exclude the type attribute itself — it is set explicitly below
            properties_to_aggregate = all_properties - skip_properties - {type_attr}

            # Resolve strategies
            property_strategies = self._resolve_property_strategies(
                properties=properties_to_aggregate,
                user_specified=user_specified,
                available_strategies=self._edge_strategies,
                default_strategy=profile.default_edge_strategy,
                warn_on_defaults=profile.warn_on_defaults,
                property_type=f"Edge[{edge_type}]",
            )

            for (c1, c2), original_edges in cluster_edges.items():
                edge_attrs: dict[str, Any] = {type_attr: edge_type}
                for prop, strategy in property_strategies.items():
                    if strategy is not None:
                        edge_attrs[prop] = strategy.aggregate_property(original_edges, prop)
                    elif original_edges:
                        edge_attrs[prop] = original_edges[0].get(prop)
                multi.add_edge(c1, c2, **edge_attrs)

        log_info(
            f"Typed edge aggregation: {multi.number_of_edges()} edges "
            f"across {len(typed_map)} types",
            LogCategory.AGGREGATION,
        )

        return multi

    def _register_default_strategies(self) -> None:
        """Register built-in aggregation strategies."""
        from npap.aggregation.basic_strategies import (
            AverageEdgeStrategy,
            AverageNodeStrategy,
            ElectricalTopologyStrategy,
            EquivalentReactanceStrategy,
            FirstEdgeStrategy,
            FirstNodeStrategy,
            SimpleTopologyStrategy,
            SumEdgeStrategy,
            SumNodeStrategy,
        )
        from npap.aggregation.physical_strategies import KronReductionStrategy

        # Topology strategies
        self._topology_strategies["simple"] = SimpleTopologyStrategy()
        self._topology_strategies["electrical"] = ElectricalTopologyStrategy()

        # Physical strategies
        self._physical_strategies["kron_reduction"] = KronReductionStrategy()

        # Node property strategies
        self._node_strategies["sum"] = SumNodeStrategy()
        self._node_strategies["average"] = AverageNodeStrategy()
        self._node_strategies["first"] = FirstNodeStrategy()

        # Edge property strategies
        self._edge_strategies["sum"] = SumEdgeStrategy()
        self._edge_strategies["average"] = AverageEdgeStrategy()
        self._edge_strategies["first"] = FirstEdgeStrategy()
        self._edge_strategies["equivalent_reactance"] = EquivalentReactanceStrategy()

        log_debug("Registered default aggregation strategies", LogCategory.MANAGER)


class PartitionAggregatorManager:
    """Main orchestrator - the primary class users interact with."""

    def __init__(self):
        self._current_graph: nx.DiGraph | None = None
        self._current_graph_hash: str | None = None
        self._current_partition: PartitionResult | None = None

        # Managers for strategies
        self.input_manager = InputDataManager()
        self.partitioning_manager = PartitioningManager()
        self.aggregation_manager = AggregationManager()

        log_debug("PartitionAggregatorManager initialized", LogCategory.MANAGER)

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

        mapping = self.partitioning_manager.partition(self._current_graph, strategy, **kwargs)

        self._current_partition = PartitionResult(
            mapping=mapping,
            original_graph_hash=self._current_graph_hash,
            strategy_name=strategy,
            strategy_metadata={},
            n_clusters=len(set(mapping.keys())),
        )

        log_info(
            f"Partitioned into {self._current_partition.n_clusters} clusters",
            LogCategory.PARTITIONING,
        )

        return self._current_partition

    def aggregate(
        self,
        partition_result: PartitionResult = None,
        profile: AggregationProfile = None,
        mode: AggregationMode = None,
        **overrides,
    ) -> nx.DiGraph | nx.MultiDiGraph:
        """
        Aggregate using partition result and profile.

        Parameters
        ----------
        partition_result : PartitionResult, optional
            Partition to use (or use stored partition).
        profile : AggregationProfile, optional
            Aggregation profile (custom configuration).
        mode : AggregationMode, optional
            Aggregation mode (pre-defined configuration).
        **overrides : dict
            Override specific profile parameters when using mode.

        Returns
        -------
        nx.DiGraph or nx.MultiDiGraph
            Aggregated graph.  Returns a ``MultiDiGraph`` when
            ``profile.edge_type_properties`` is populated.

        Notes
        -----
        If both profile and mode are provided, profile takes precedence.
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

    def aggregate_parallel_edges(
        self,
        edge_properties: dict[str, str] = None,
        default_strategy: str = "sum",
        warn_on_defaults: bool = True,
    ) -> nx.DiGraph:
        """
        Convert current MultiDiGraph to simple DiGraph by aggregating parallel edges.

        This method aggregates all parallel edges between the same directed node pairs
        into single edges, using the specified aggregation strategies. The graph
        must be a MultiDiGraph for this operation to be meaningful.

        For directed graphs, edges (A->B) and (B->A) are treated as separate edges.

        Parameters
        ----------
        edge_properties : dict[str, str], optional
            Dict mapping property names to aggregation strategies.
            Example: ``{"reactance": "average", "length": "sum"}``.
        default_strategy : str
            Strategy to use for properties not specified.
        warn_on_defaults : bool
            Whether to warn when using default strategy.

        Returns
        -------
        nx.DiGraph
            Simple DiGraph with parallel edges aggregated.

        Raises
        ------
        ValueError
            If no graph is loaded or graph is not a MultiDiGraph.
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
            warn_on_defaults=warn_on_defaults,
        )

        # Update hash since graph has changed
        self._current_graph_hash = self._compute_graph_hash(self._current_graph)

        # Clear partition since graph structure changed
        self._current_partition = None

        return self._current_graph

    def full_workflow(
        self,
        data_strategy: str,
        partition_strategy: str,
        aggregation_profile: AggregationProfile = None,
        aggregation_mode: AggregationMode = None,
        **kwargs,
    ) -> nx.DiGraph:
        """
        Execute complete workflow without storing intermediates.

        If a MultiDiGraph is loaded, parallel edges will be automatically aggregated
        before partitioning. If a voltage-aware partitioning strategy is selected,
        voltage levels will be grouped first in 220kV and 380kV voltage levels.

        Parameters
        ----------
        data_strategy : str
            Data loading strategy name.
        partition_strategy : str
            Partitioning strategy name.
        aggregation_profile : AggregationProfile, optional
            Aggregation profile (custom).
        aggregation_mode : AggregationMode, optional
            Aggregation mode (pre-defined).
        **kwargs : dict
            Parameters for data loading, parallel edge aggregation, and partitioning.

        Returns
        -------
        nx.DiGraph
            Aggregated graph.

        Examples
        --------
        >>> manager = PartitionAggregatorManager()
        >>> result = manager.full_workflow(
        ...     data_strategy="csv_files",
        ...     partition_strategy="geographical_kmeans",
        ...     node_file="buses.csv",
        ...     edge_file="lines.csv",
        ...     n_clusters=10
        ... )
        """
        log_info("Starting full workflow", LogCategory.MANAGER)

        # Parameters for data loading
        data_params = {
            k: v
            for k, v in kwargs.items()
            if k
            in [
                "node_file",
                "edge_file",
                "line_file",
                "transformer_file",
                "graph",
                "connection_string",
                "table_prefix",
                "delimiter",
                "decimal",
                "node_id_col",
                "edge_from_col",
                "edge_to_col",
                "bidirectional",
            ]
        }

        # Parameters for parallel edge aggregation (if MultiDiGraph)
        parallel_edge_params = {
            k: v
            for k, v in kwargs.items()
            if k
            in [
                "edge_properties",
                "parallel_edge_default_strategy",
                "parallel_edge_warn_on_defaults",
            ]
        }

        # Parameters for partitioning
        partition_params = {
            k: v
            for k, v in kwargs.items()
            if k not in data_params and k not in parallel_edge_params
        }

        # Step 1: Load data
        self.load_data(data_strategy, **data_params)

        # Step 2: If MultiDiGraph, aggregate parallel edges first
        if isinstance(self._current_graph, nx.MultiDiGraph):
            log_info(
                "MultiDiGraph detected, auto-aggregating parallel edges",
                LogCategory.MANAGER,
            )
            self.aggregate_parallel_edges(**parallel_edge_params)

        # For voltage-aware strategies, group voltages first
        if partition_strategy.startswith("va_"):
            self.group_by_voltage_levels([220, 380])

        # Step 3: Partition
        partition_result = self.partition(partition_strategy, **partition_params)

        # Step 4: Aggregate clusters
        result = self.aggregate(partition_result, aggregation_profile, aggregation_mode)

        log_info("Full workflow complete", LogCategory.MANAGER)
        return result

    def get_current_graph(self) -> nx.DiGraph | None:
        """
        Get the current graph.

        Returns
        -------
        nx.DiGraph or None
            Current graph, or None if no graph is loaded.
        """
        return self._current_graph

    def get_current_partition(self) -> PartitionResult | None:
        """
        Get the current partition result.

        Returns
        -------
        PartitionResult or None
            Current partition result, or None if not partitioned.
        """
        return self._current_partition

    @staticmethod
    def _compute_graph_hash(graph: nx.DiGraph) -> str:
        """
        Compute hash for graph validation.

        Parameters
        ----------
        graph : nx.DiGraph
            Graph to compute hash for.

        Returns
        -------
        str
            Hash string for the graph.
        """
        from npap.utils import compute_graph_hash

        return compute_graph_hash(graph)

    def plot_network(
        self,
        style: str = "simple",
        graph: nx.DiGraph = None,
        show: bool = True,
        config=None,
        **kwargs,
    ):
        """
        Plot the network on an interactive map.

        Parameters
        ----------
        style : str
            Plot style:

            - 'simple': All edges same color (fast, minimal)
            - 'voltage_aware': Edges colored by type (line/trafo/dc_link)
            - 'clustered': Nodes colored by cluster assignment
        graph : nx.DiGraph, optional
            Graph to plot (uses current graph if not provided).
        show : bool
            Whether to display immediately.
        config : PlotConfig, optional
            PlotConfig instance to override defaults. If provided,
            kwargs will further override values from this config.
        **kwargs : dict
            Additional configuration options (see PlotConfig for full list).

        Returns
        -------
        go.Figure
            Plotly Figure object.

        Raises
        ------
        ValueError
            If no graph is loaded or clustered style without partition.

        Examples
        --------
        >>> manager.plot_network(style="voltage_aware", title="My Network")
        >>> manager.plot_network(style="clustered")  # After partitioning
        """
        from npap.visualization import plot_network

        target_graph = graph if graph is not None else self._current_graph
        if target_graph is None:
            raise ValueError("No graph loaded. Call load_data() first.")

        # For clustered style, pass the partition map
        partition_map = None
        if style == "clustered":
            if self._current_partition is None:
                raise ValueError(
                    "Cannot create clustered plot without partitioning. Call partition() first."
                )
            partition_map = self._current_partition.mapping

        log_info(f"Creating network plot with style: {style}", LogCategory.VISUALIZATION)

        return plot_network(
            graph=target_graph,
            style=style,
            partition_map=partition_map,
            show=show,
            config=config,
            **kwargs,
        )

    def group_by_voltage_levels(
        self,
        target_levels: list[float],
        voltage_attr: str = "voltage",
        store_original: bool = True,
        handle_missing: str = "infer",
    ) -> dict[str, Any]:
        """
        Group bus voltage levels to predefined target values.

        This method reassigns each node's voltage to the nearest target voltage level,
        creating clean voltage "islands" for subsequent voltage-aware partitioning.

        Parameters
        ----------
        target_levels : list[float]
            List of target voltage levels (kV) to harmonize to.
            Example: ``[220, 380]`` for European transmission grid.
        voltage_attr : str
            Node attribute containing voltage level.
        store_original : bool
            If True, stores original voltage in 'original_{voltage_attr}'.
        handle_missing : str
            How to handle nodes without voltage data:

            - 'infer': Infer from connected neighbors
            - 'nearest': Assign to nearest target level
            - 'error': Raise an error
            - 'skip': Leave as None

        Returns
        -------
        dict[str, Any]
            Summary dict with:

            - 'total_nodes': Total number of nodes processed
            - 'reassignments': Dict mapping original_voltage to (target_voltage, count)
            - 'missing_handled': Number of nodes with missing voltage
            - 'voltage_distribution': Dict mapping target_voltage to node_count

        Raises
        ------
        ValueError
            If no graph loaded, target_levels empty, or handle_missing='error'
            with missing voltages.

        Examples
        --------
        >>> manager.group_by_voltage_levels([220, 380])
        {'total_nodes': 6000, 'voltage_distribution': {220: 3500, 380: 2500}, ...}
        """
        if not self._current_graph:
            raise ValueError("No graph loaded. Call load_data() first.")

        if not target_levels:
            raise ValueError("target_levels cannot be empty.")

        target_levels = sorted(target_levels)
        graph = self._current_graph

        log_info(
            f"Grouping voltages to {len(target_levels)} target levels: {target_levels}",
            LogCategory.MANAGER,
        )

        # Statistics tracking
        reassignments: dict[Any, tuple[float, int]] = {}  # original -> (target, count)
        missing_nodes: list[Any] = []
        voltage_distribution: dict[float, int] = dict.fromkeys(target_levels, 0)

        # First pass: identify missing voltages and reassign known ones
        for node in graph.nodes():
            node_data = graph.nodes[node]
            original_voltage = node_data.get(voltage_attr)

            # Try fallback attributes
            if original_voltage is None:
                original_voltage = node_data.get("voltage", node_data.get("v_nom"))

            if original_voltage is None:
                missing_nodes.append(node)
                continue

            # Store original if requested
            if store_original:
                node_data[f"original_{voltage_attr}"] = original_voltage

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
            orig_key = (
                round(original_voltage, 1)
                if isinstance(original_voltage, float)
                else original_voltage
            )
            if orig_key not in reassignments:
                reassignments[orig_key] = (target, 0)
            reassignments[orig_key] = (
                reassignments[orig_key][0],
                reassignments[orig_key][1] + 1,
            )

        # Handle missing voltages
        if missing_nodes:
            if handle_missing == "error":
                raise ValueError(
                    f"{len(missing_nodes)} nodes have missing voltage data. "
                    f"First few: {missing_nodes[:5]}"
                )

            elif handle_missing == "infer":
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
                    if hasattr(graph, "predecessors"):
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
                            graph.nodes[node][f"original_{voltage_attr}"] = None
                        voltage_distribution[target] += 1
                        inferred_count += 1
                    else:
                        still_missing.append(node)

                if still_missing:
                    log_warning(
                        f"{len(still_missing)} nodes could not be inferred. "
                        f"Assigning to nearest target level.",
                        LogCategory.MANAGER,
                    )
                    default_target = target_levels[0]
                    for node in still_missing:
                        graph.nodes[node][voltage_attr] = default_target
                        if store_original:
                            graph.nodes[node][f"original_{voltage_attr}"] = None
                        voltage_distribution[default_target] += 1

                log_info(
                    f"Inferred voltage for {inferred_count} nodes from neighbors",
                    LogCategory.MANAGER,
                )

            elif handle_missing == "nearest":
                # Assign to first target level (arbitrary)
                default_target = target_levels[0]
                for node in missing_nodes:
                    graph.nodes[node][voltage_attr] = default_target
                    if store_original:
                        graph.nodes[node][f"original_{voltage_attr}"] = None
                    voltage_distribution[default_target] += 1

            elif handle_missing == "skip":
                log_warning(
                    f"{len(missing_nodes)} nodes have no voltage (will cluster separately)",
                    LogCategory.MANAGER,
                )

        # Update graph hash since we modified node attributes
        self._current_graph_hash = self._compute_graph_hash(self._current_graph)

        return {
            "total_nodes": len(list(graph.nodes())),
            "reassignments": reassignments,
            "missing_handled": len(missing_nodes),
            "voltage_distribution": voltage_distribution,
            "target_levels": target_levels,
        }
