from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
from scipy.linalg import LinAlgError, solve

from npap.exceptions import PartitioningError, ValidationError
from npap.interfaces import EdgeType, PartitioningStrategy
from npap.logging import LogCategory, log_debug, log_info, log_warning
from npap.utils import (
    create_partition_map,
    run_hierarchical,
    run_kmedoids,
    validate_partition,
    with_runtime_config,
)


@dataclass
class VAElectricalConfig:
    """
    Configuration for voltage-aware electrical distance partitioning.

    Attributes
    ----------
    zero_reactance_replacement : float
        Reactance value used when edge reactance is zero. Default is 1e-5.
    regularization_factor : float
        Small value added to B matrix diagonal for numerical stability.
        Set to 0.0 to disable regularization. Default 1e-10 provides
        mild regularization that prevents singular matrix issues.
    infinite_distance : float
        Value used to represent "infinite" distance between nodes in different
        voltage levels, DC islands, or disconnected components. Using a large
        finite value instead of np.inf to avoid numerical issues in clustering.
    voltage_tolerance : float
        Tolerance for voltage comparison (kV). Nodes with voltages within
        this tolerance are considered the same voltage level. Default is 1.0 kV.
    hierarchical_linkage : str
        Linkage criterion for hierarchical clustering. Options: 'complete',
        'average', 'single'. Default is 'complete'.
    """

    zero_reactance_replacement: float = 1e-5
    regularization_factor: float = 1e-10
    infinite_distance: float = 1e4
    voltage_tolerance: float = 1.0
    hierarchical_linkage: str = "complete"


class VAElectricalDistancePartitioning(PartitioningStrategy):
    """
    Voltage-Aware Electrical Distance Partitioning.

    This strategy partitions nodes based on electrical distance (PTDF) computed
    over the complete AC network (lines AND transformers), while enforcing
    voltage level boundaries as hard clustering constraints via post-processing.

    Physical Rationale
    ------------------
    In power systems, PTDF (Power Transfer Distribution Factor) describes how
    power injections affect line flows within an AC network. This strategy:

    1. Computes PTDF for the complete AC network (lines + transformers) per
       DC island, capturing full electrical coupling including through transformers.
    2. Post-processes distances to enforce voltage level separation: nodes at
       different voltage levels receive infinite distance, ensuring they never
       cluster together.
    3. Transformer edges and buses at different voltage levels are set to
       infinite distance, preserving voltage hierarchy in the partitioning.

    This approach is physically meaningful because:
    - The full PTDF captures how power flows through the entire AC network
    - Voltage level boundaries are then enforced as hard clustering constraints
    - Transformers become inter-cluster edges after aggregation

    Constraint Hierarchy
    --------------------
    1. DC Islands: Nodes in different DC islands have infinite distance
       (computed via separate PTDF matrices).
    2. Voltage Levels: Nodes at different voltage levels have infinite distance
       (enforced via post-processing).

    Algorithm
    ---------
    1. Group nodes by DC island
    2. For each DC island:
       a. Extract full AC subgraph (lines + transformers, exclude DC links)
       b. Select ONE slack bus for the entire DC island
       c. Compute PTDF including ALL AC elements
       d. Calculate electrical distances from PTDF columns
    3. Combine into full distance matrix with infinite inter-island distances
    4. Post-process: Set infinite distance for node pairs at different voltage levels
    5. Run clustering algorithm on the distance matrix

    Supported Algorithms
    --------------------
    - 'kmedoids': K-Medoids clustering (works with precomputed distance matrices)
    - 'hierarchical': Agglomerative clustering with precomputed distances

    See Also
    --------
    ElectricalDistancePartitioning : Standard electrical partitioning (DC-island aware)
    VAGeographicalPartitioning : Voltage-aware geographical partitioning
    """

    SUPPORTED_ALGORITHMS = ["kmedoids", "hierarchical"]
    SUPPORTED_LINKAGES = ["complete", "average", "single"]

    # Edge types that participate in AC power flow (have reactance)
    AC_EDGE_TYPES = {EdgeType.LINE.value, EdgeType.TRAFO.value}

    # Config parameter names for runtime override detection
    _CONFIG_PARAMS = {
        "zero_reactance_replacement",
        "regularization_factor",
        "infinite_distance",
        "voltage_tolerance",
        "hierarchical_linkage",
    }

    def __init__(
        self,
        algorithm: str = "kmedoids",
        slack_bus: Any | None = None,
        voltage_attr: str = "voltage",
        dc_island_attr: str = "dc_island",
        config: VAElectricalConfig | None = None,
    ):
        """
        Initialize voltage-aware electrical distance partitioning strategy.

        Parameters
        ----------
        algorithm : str, default='kmedoids'
            Clustering algorithm ('kmedoids', 'hierarchical').
        slack_bus : Any, optional
            Specific node to use as slack bus, or None for auto-selection.
        voltage_attr : str, default='voltage'
            Node attribute name containing voltage level.
        dc_island_attr : str, default='dc_island'
            Node attribute name containing DC island ID.
        config : VAElectricalConfig, optional
            Configuration parameters for distance calculations.

        Raises
        ------
        ValueError
            If unsupported algorithm or linkage is specified.
        """
        self.algorithm = algorithm
        self.slack_bus = slack_bus
        self.voltage_attr = voltage_attr
        self.dc_island_attr = dc_island_attr
        self.config = config or VAElectricalConfig()

        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm: {algorithm}. "
                f"Supported: {', '.join(self.SUPPORTED_ALGORITHMS)}"
            )

        # Validate hierarchical linkage
        if self.config.hierarchical_linkage not in self.SUPPORTED_LINKAGES:
            raise ValueError(
                f"Unsupported hierarchical linkage: {self.config.hierarchical_linkage}. "
                f"Supported: {', '.join(self.SUPPORTED_LINKAGES)}"
            )

        log_debug(
            f"Initialized VAElectricalDistancePartitioning: algorithm={algorithm}, "
            f"voltage_attr={voltage_attr}, dc_island_attr={dc_island_attr}",
            LogCategory.PARTITIONING,
        )

    @property
    def required_attributes(self) -> dict[str, list[str]]:
        """Required attributes for voltage-aware electrical partitioning."""
        return {
            "nodes": [self.voltage_attr, self.dc_island_attr],
            "edges": [],  # Reactance validated separately for AC edges
        }

    def _get_strategy_name(self) -> str:
        """Get descriptive strategy name for error messages."""
        return f"va_electrical_{self.algorithm}"

    @with_runtime_config(VAElectricalConfig, _CONFIG_PARAMS)
    def partition(self, graph: nx.DiGraph, **kwargs) -> dict[int, list[Any]]:
        """
        Partition nodes based on voltage-aware electrical distance using PTDF.

        Computes PTDF for the full AC network (lines + transformers) per DC island,
        then enforces voltage level separation via post-processing.

        Parameters
        ----------
        graph : nx.DiGraph
            NetworkX DiGraph with voltage and dc_island on nodes,
            reactance on AC edges (lines and transformers).
        **kwargs : dict
            Additional parameters:
            - n_clusters : int
                Number of clusters (required).
            - config : VAElectricalConfig, optional
                Override instance config.
            - hierarchical_linkage : str, optional
                Override linkage for hierarchical clustering.
            - Individual config parameters to override.

        Returns
        -------
        dict[int, list[Any]]
            Dictionary mapping cluster_id -> list of node_ids.

        Raises
        ------
        PartitioningError
            If partitioning fails.
        ValidationError
            If required attributes are missing.
        ValueError
            If unsupported hierarchical_linkage is specified.
        """
        try:
            effective_config = kwargs.get("_effective_config", self.config)
            effective_slack = kwargs.get("slack_bus", self.slack_bus)
            n_clusters = kwargs.get("n_clusters", 0)

            # Validate hierarchical_linkage if overridden
            if effective_config.hierarchical_linkage not in self.SUPPORTED_LINKAGES:
                raise ValueError(
                    f"Unsupported hierarchical linkage: {effective_config.hierarchical_linkage}. "
                    f"Supported: {', '.join(self.SUPPORTED_LINKAGES)}"
                )

            if n_clusters is None or n_clusters <= 0:
                raise PartitioningError(
                    "Voltage-aware electrical partitioning requires a positive "
                    "'n_clusters' parameter.",
                    strategy=self._get_strategy_name(),
                )

            nodes = list(graph.nodes())
            n_nodes = len(nodes)

            if n_clusters > n_nodes:
                raise PartitioningError(
                    f"Cannot create {n_clusters} clusters from {n_nodes} nodes.",
                    strategy=self._get_strategy_name(),
                )

            # Validate required attributes
            self._validate_node_attributes(graph, nodes)

            # Group nodes by DC island
            dc_islands = self._group_nodes_by_dc_island(graph, nodes)

            # Build voltage key array for post-processing
            voltage_keys = self._build_voltage_key_array(graph, nodes, effective_config)

            log_info(
                f"Starting VA electrical partitioning (PTDF): {self.algorithm}, "
                f"n_clusters={n_clusters}, dc_islands={len(dc_islands)}, "
                f"voltage_levels={len(np.unique(voltage_keys))}",
                LogCategory.PARTITIONING,
            )

            # Calculate electrical distance matrix (full PTDF per DC island)
            distance_matrix = self._calculate_electrical_distance_matrix(
                graph, nodes, dc_islands, effective_config, effective_slack
            )

            # Post-process: enforce voltage level constraints
            self._apply_voltage_constraints(distance_matrix, voltage_keys, effective_config)

            # Perform clustering
            labels = self._run_clustering(distance_matrix, effective_config, **kwargs)

            # Create and validate partition
            partition_map = create_partition_map(nodes, labels)
            validate_partition(partition_map, n_nodes, self._get_strategy_name())

            # Validate cluster consistency
            self._validate_cluster_consistency(graph, partition_map, effective_config)

            log_info(
                f"VA electrical partitioning complete: {len(partition_map)} clusters",
                LogCategory.PARTITIONING,
            )

            return partition_map

        except Exception as e:
            if isinstance(e, (PartitioningError, ValidationError)):
                raise
            raise PartitioningError(
                f"Voltage-aware electrical partitioning failed: {e}",
                strategy=self._get_strategy_name(),
                graph_info={
                    "nodes": len(list(graph.nodes())),
                    "edges": len(graph.edges()),
                },
            ) from e

    def _run_clustering(
        self,
        distance_matrix: np.ndarray,
        config: VAElectricalConfig,
        **kwargs,
    ) -> np.ndarray:
        """
        Dispatch to appropriate clustering algorithm.

        Parameters
        ----------
        distance_matrix : np.ndarray
            Precomputed distance matrix (n_nodes x n_nodes).
        config : VAElectricalConfig
            Configuration with hierarchical_linkage.
        **kwargs : dict
            Additional parameters including n_clusters.

        Returns
        -------
        np.ndarray
            Cluster labels for each node.
        """
        n_clusters = kwargs.get("n_clusters")

        if self.algorithm == "kmedoids":
            log_debug(
                f"Running K-medoids clustering with {n_clusters} clusters",
                LogCategory.PARTITIONING,
            )
            return run_kmedoids(distance_matrix, n_clusters)
        elif self.algorithm == "hierarchical":
            log_debug(
                f"Running hierarchical clustering with {n_clusters} clusters, "
                f"linkage={config.hierarchical_linkage}",
                LogCategory.PARTITIONING,
            )
            return run_hierarchical(distance_matrix, n_clusters, config.hierarchical_linkage)
        else:
            raise PartitioningError(
                f"Unknown algorithm: {self.algorithm}",
                strategy=self._get_strategy_name(),
            )

    # =========================================================================
    # NODE ATTRIBUTE VALIDATION
    # =========================================================================

    def _validate_node_attributes(
        self,
        graph: nx.DiGraph,
        nodes: list[Any],
    ) -> None:
        """
        Validate that all nodes have required dc_island and voltage attributes.

        Parameters
        ----------
        graph : nx.DiGraph
            NetworkX DiGraph with node attributes.
        nodes : list[Any]
            List of nodes to validate.

        Raises
        ------
        ValidationError
            If any node is missing required attributes.
        """
        missing_dc = []
        missing_voltage = []

        for node in nodes:
            node_data = graph.nodes[node]
            if self.dc_island_attr not in node_data:
                missing_dc.append(node)
            if self.voltage_attr not in node_data:
                missing_voltage.append(node)

        if missing_dc:
            sample = missing_dc[:5]
            raise ValidationError(
                f"VA electrical partitioning requires '{self.dc_island_attr}' attribute "
                f"on all nodes. {len(missing_dc)} node(s) missing (first few: {sample}). "
                f"Use 'va_loader' to automatically detect DC islands.",
                missing_attributes={"nodes": [self.dc_island_attr]},
                strategy=self._get_strategy_name(),
            )

        if missing_voltage:
            sample = missing_voltage[:5]
            raise ValidationError(
                f"VA electrical partitioning requires '{self.voltage_attr}' attribute "
                f"on all nodes. {len(missing_voltage)} node(s) missing (first few: {sample}). "
                f"Use 'va_loader' to load voltage-aware data.",
                missing_attributes={"nodes": [self.voltage_attr]},
                strategy=self._get_strategy_name(),
            )

    def _validate_ac_edge_attributes(self, graph: nx.DiGraph) -> None:
        """
        Validate that AC edges (lines and transformers) have reactance attribute.

        Parameters
        ----------
        graph : nx.DiGraph
            NetworkX DiGraph to validate.

        Raises
        ------
        ValidationError
            If any AC edge is missing the 'x' attribute.
        """
        missing_edges = []
        for u, v, data in graph.edges(data=True):
            edge_type = data.get("type", EdgeType.LINE.value)
            if edge_type in self.AC_EDGE_TYPES and "x" not in data:
                missing_edges.append((u, v))

        if missing_edges:
            sample = missing_edges[:5]
            raise ValidationError(
                f"AC edges (lines/transformers) require 'x' (reactance) attribute. "
                f"{len(missing_edges)} edge(s) missing (first few: {sample}).",
                missing_attributes={"edges": ["x"]},
                strategy=self._get_strategy_name(),
            )

    # =========================================================================
    # DC ISLAND GROUPING
    # =========================================================================

    def _group_nodes_by_dc_island(
        self,
        graph: nx.DiGraph,
        nodes: list[Any],
    ) -> dict[Any, list[Any]]:
        """
        Group nodes by DC island attribute.

        Parameters
        ----------
        graph : nx.DiGraph
            NetworkX DiGraph with dc_island attribute on nodes.
        nodes : list[Any]
            List of nodes to group.

        Returns
        -------
        dict[Any, list[Any]]
            Dictionary mapping dc_island_id -> list of nodes.
        """
        islands: dict[Any, list[Any]] = defaultdict(list)

        for node in nodes:
            island_id = graph.nodes[node].get(self.dc_island_attr)
            islands[island_id].append(node)

        return dict(islands)

    # =========================================================================
    # VOLTAGE KEY COMPUTATION
    # =========================================================================

    def _build_voltage_key_array(
        self,
        graph: nx.DiGraph,
        nodes: list[Any],
        config: VAElectricalConfig,
    ) -> np.ndarray:
        """
        Build array of voltage keys for all nodes.

        Voltage keys are integer indices that group nodes by voltage level
        (within tolerance). This enables efficient vectorized comparison.

        Parameters
        ----------
        graph : nx.DiGraph
            NetworkX DiGraph with voltage attribute on nodes.
        nodes : list[Any]
            Ordered list of nodes.
        config : VAElectricalConfig
            Configuration with voltage_tolerance.

        Returns
        -------
        np.ndarray
            Integer array of shape (n_nodes,) where each element is the
            voltage group index for that node.
        """
        n_nodes = len(nodes)
        voltages = np.empty(n_nodes, dtype=np.float64)

        for i, node in enumerate(nodes):
            voltage = graph.nodes[node].get(self.voltage_attr)
            voltages[i] = float(voltage) if isinstance(voltage, (int, float)) else np.nan

        # Quantize voltages to tolerance-based keys
        tolerance = max(config.voltage_tolerance, 0.1)
        voltage_keys_float = np.round(voltages / tolerance) * tolerance

        # Convert to integer indices for efficient comparison
        unique_keys, voltage_indices = np.unique(voltage_keys_float, return_inverse=True)

        log_debug(
            f"Built voltage key array: {len(unique_keys)} unique voltage levels",
            LogCategory.PARTITIONING,
        )

        return voltage_indices

    @staticmethod
    def _get_voltage_key(voltage: Any, config: VAElectricalConfig) -> Any:
        """Get voltage key for tolerance-based matching (scalar version)."""
        if isinstance(voltage, (int, float)):
            tolerance = max(config.voltage_tolerance, 0.1)
            return round(voltage / tolerance) * tolerance
        return voltage

    # =========================================================================
    # ELECTRICAL DISTANCE MATRIX CALCULATION
    # =========================================================================

    def _calculate_electrical_distance_matrix(
        self,
        graph: nx.DiGraph,
        nodes: list[Any],
        dc_islands: dict[Any, list[Any]],
        config: VAElectricalConfig,
        slack_bus: Any | None,
    ) -> np.ndarray:
        """
        Calculate electrical distance matrix with per-DC-island PTDF.

        Computes PTDF including ALL AC elements (lines + transformers) for each
        DC island, then combines into a full distance matrix with infinite
        inter-island distances.

        Parameters
        ----------
        graph : nx.DiGraph
            NetworkX DiGraph with reactance on AC edges.
        nodes : list[Any]
            Ordered list of all nodes.
        dc_islands : dict[Any, list[Any]]
            Mapping of dc_island_id -> list of nodes.
        config : VAElectricalConfig
            Configuration parameters.
        slack_bus : Any, optional
            User-specified slack bus.

        Returns
        -------
        np.ndarray
            Distance matrix (n_nodes x n_nodes) with infinite distances
            between different DC islands.
        """
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        n_nodes = len(nodes)

        # Initialize with infinite distances
        distance_matrix = np.full((n_nodes, n_nodes), config.infinite_distance)
        np.fill_diagonal(distance_matrix, 0.0)

        # Process each DC island
        for island_id, island_nodes in dc_islands.items():
            log_debug(
                f"Processing DC island {island_id}: {len(island_nodes)} nodes",
                LogCategory.PARTITIONING,
            )

            # Skip single-node islands
            if len(island_nodes) == 1:
                log_debug(
                    f"DC island {island_id} has single node, distance = 0",
                    LogCategory.PARTITIONING,
                )
                continue

            # Extract full AC subgraph (lines + transformers, exclude DC links)
            ac_subgraph = self._extract_ac_subgraph(graph, island_nodes)

            # Validate AC edges have reactance
            self._validate_ac_subgraph_reactance(ac_subgraph, island_id)

            # Validate island connectivity
            self._validate_island_connectivity(ac_subgraph, island_id, island_nodes)

            # Select ONE slack bus for the entire DC island
            island_slack = self._select_slack_bus(ac_subgraph, island_nodes, slack_bus, island_id)

            # Compute PTDF-based distances for this island
            island_distances = self._compute_island_ptdf_distances(
                ac_subgraph, island_nodes, island_slack, config
            )

            # Insert into full matrix using vectorized indexing
            island_indices = np.array([node_to_idx[n] for n in island_nodes])
            distance_matrix[np.ix_(island_indices, island_indices)] = island_distances

        log_info(
            f"Processed {len(dc_islands)} DC island(s) for PTDF computation",
            LogCategory.PARTITIONING,
        )

        return distance_matrix

    def _extract_ac_subgraph(
        self,
        graph: nx.DiGraph,
        island_nodes: list[Any],
    ) -> nx.DiGraph:
        """
        Extract AC subgraph containing lines AND transformers (exclude DC links).

        Parameters
        ----------
        graph : nx.DiGraph
            Full NetworkX DiGraph.
        island_nodes : list[Any]
            Nodes in this DC island.

        Returns
        -------
        nx.DiGraph
            Subgraph with only AC edges (lines + transformers).
        """
        island_node_set = set(island_nodes)
        subgraph = nx.DiGraph()

        # Add nodes with attributes
        for node in island_nodes:
            subgraph.add_node(node, **graph.nodes[node])

        # Add AC edges (lines and transformers, exclude DC links)
        for u, v, data in graph.edges(data=True):
            if u in island_node_set and v in island_node_set:
                edge_type = data.get("type", EdgeType.LINE.value)
                if edge_type in self.AC_EDGE_TYPES:
                    subgraph.add_edge(u, v, **data)

        return subgraph

    def _validate_ac_subgraph_reactance(
        self,
        ac_subgraph: nx.DiGraph,
        island_id: Any,
    ) -> None:
        """
        Validate that all edges in AC subgraph have reactance attribute.

        Parameters
        ----------
        ac_subgraph : nx.DiGraph
            AC-only subgraph.
        island_id : Any
            DC island identifier.

        Raises
        ------
        ValidationError
            If any edge is missing the 'x' attribute.
        """
        missing_edges = []
        for u, v, data in ac_subgraph.edges(data=True):
            if "x" not in data:
                missing_edges.append((u, v))

        if missing_edges:
            sample = missing_edges[:5]
            raise ValidationError(
                f"AC edges in DC island {island_id} require 'x' (reactance) "
                f"attribute. {len(missing_edges)} edge(s) missing (first few: {sample}).",
                missing_attributes={"edges": ["x"]},
                strategy=self._get_strategy_name(),
            )

    def _validate_island_connectivity(
        self,
        ac_subgraph: nx.DiGraph,
        island_id: Any,
        island_nodes: list[Any],
    ) -> None:
        """
        Validate that a DC island's AC subgraph is connected.

        Parameters
        ----------
        ac_subgraph : nx.DiGraph
            AC-only subgraph for the island.
        island_id : Any
            DC island identifier.
        island_nodes : list[Any]
            Nodes in this island.

        Raises
        ------
        PartitioningError
            If AC subgraph is not weakly connected.
        """
        if ac_subgraph.number_of_edges() == 0:
            raise PartitioningError(
                f"DC island {island_id} has no AC edges (lines/transformers). "
                f"Cannot compute electrical distances without AC connectivity.",
                strategy=self._get_strategy_name(),
                graph_info={"island_id": island_id, "n_nodes": len(island_nodes)},
            )

        if not nx.is_weakly_connected(ac_subgraph):
            n_components = nx.number_weakly_connected_components(ac_subgraph)
            raise PartitioningError(
                f"DC island {island_id} is not AC-connected. Found {n_components} "
                f"disconnected AC components within the island. This may indicate "
                f"missing line/transformer data or incorrect DC island assignment.",
                strategy=self._get_strategy_name(),
                graph_info={
                    "island_id": island_id,
                    "n_nodes": len(island_nodes),
                    "n_ac_edges": ac_subgraph.number_of_edges(),
                    "n_components": n_components,
                },
            )

    # =========================================================================
    # VOLTAGE CONSTRAINT POST-PROCESSING
    # =========================================================================

    @staticmethod
    def _apply_voltage_constraints(
        distance_matrix: np.ndarray,
        voltage_keys: np.ndarray,
        config: VAElectricalConfig,
    ) -> None:
        """
        Set infinite distance between nodes at different voltage levels.

        This is a post-processing step that enforces voltage level boundaries
        after PTDF distances have been computed. Uses broadcasting for O(n²)
        memory but O(1) NumPy operations, which is highly efficient for large
        graphs due to SIMD vectorization.

        Parameters
        ----------
        distance_matrix : np.ndarray
            Distance matrix to modify in-place (n_nodes x n_nodes).
        voltage_keys : np.ndarray
            Integer array of voltage group indices for each node.
        config : VAElectricalConfig
            Configuration with infinite_distance value.
        """
        # Create boolean mask where voltage levels differ
        # Broadcasting: voltage_keys[:, None] has shape (n, 1)
        #               voltage_keys[None, :] has shape (1, n)
        #               Result has shape (n, n)
        different_voltage_mask = voltage_keys[:, np.newaxis] != voltage_keys[np.newaxis, :]

        # Apply infinite distance where voltage levels differ
        distance_matrix[different_voltage_mask] = config.infinite_distance

        n_modified = np.sum(different_voltage_mask)
        log_debug(
            f"Applied voltage constraints: {n_modified} pairs set to infinite distance",
            LogCategory.PARTITIONING,
        )

    def _validate_cluster_consistency(
        self,
        graph: nx.DiGraph,
        partition_map: dict[int, list[Any]],
        config: VAElectricalConfig,
    ) -> None:
        """
        Validate that clusters don't mix incompatible DC islands or voltage levels.

        Parameters
        ----------
        graph : nx.DiGraph
            Original NetworkX graph.
        partition_map : dict[int, list[Any]]
            Resulting partition mapping.
        config : VAElectricalConfig
            Configuration with voltage_tolerance.
        """
        for cluster_id, cluster_nodes in partition_map.items():
            dc_islands_in_cluster = set()
            voltages_in_cluster = set()

            for node in cluster_nodes:
                node_data = graph.nodes[node]

                dc_island = node_data.get(self.dc_island_attr)
                if dc_island is not None:
                    dc_islands_in_cluster.add(dc_island)

                voltage = node_data.get(self.voltage_attr)
                if voltage is not None:
                    voltages_in_cluster.add(self._get_voltage_key(voltage, config))

            if len(dc_islands_in_cluster) > 1:
                log_warning(
                    f"Cluster {cluster_id} contains nodes from multiple DC islands: "
                    f"{dc_islands_in_cluster}. This should not happen with infinite distances.",
                    LogCategory.PARTITIONING,
                    warn_user=False,
                )

            if len(voltages_in_cluster) > 1:
                log_warning(
                    f"Cluster {cluster_id} contains multiple voltage levels: "
                    f"{voltages_in_cluster}. This should not happen with infinite distances.",
                    LogCategory.PARTITIONING,
                    warn_user=False,
                )

    # =========================================================================
    # SLACK BUS SELECTION
    # =========================================================================

    @staticmethod
    def _select_slack_bus(
        ac_subgraph: nx.DiGraph,
        island_nodes: list[Any],
        user_slack: Any | None,
        island_id: Any,
    ) -> Any:
        """
        Select slack bus for a DC island.

        Parameters
        ----------
        ac_subgraph : nx.DiGraph
            AC subgraph for this DC island.
        island_nodes : list[Any]
            Nodes in this DC island.
        user_slack : Any, optional
            User-specified slack bus.
        island_id : Any
            DC island identifier for logging.

        Returns
        -------
        Any
            Selected slack bus node.
        """
        island_node_set = set(island_nodes)

        if user_slack is not None and user_slack in island_node_set:
            log_debug(
                f"Using user-specified slack bus {user_slack} for DC island {island_id}",
                LogCategory.PARTITIONING,
            )
            return user_slack

        # Auto-select: highest total degree in AC subgraph
        degrees = {n: ac_subgraph.in_degree(n) + ac_subgraph.out_degree(n) for n in island_nodes}
        selected = max(island_nodes, key=lambda n: degrees[n])

        log_debug(
            f"Auto-selected slack bus {selected} for DC island {island_id} "
            f"(degree={degrees[selected]})",
            LogCategory.PARTITIONING,
        )

        return selected

    # =========================================================================
    # PTDF COMPUTATION
    # =========================================================================

    def _compute_island_ptdf_distances(
        self,
        ac_subgraph: nx.DiGraph,
        island_nodes: list[Any],
        slack_bus: Any,
        config: VAElectricalConfig,
    ) -> np.ndarray:
        """
        Compute PTDF-based electrical distances for a DC island.

        Parameters
        ----------
        ac_subgraph : nx.DiGraph
            AC subgraph (lines + transformers) for this DC island.
        island_nodes : list[Any]
            Ordered list of nodes in this island.
        slack_bus : Any
            Slack bus node for this island.
        config : VAElectricalConfig
            Configuration parameters.

        Returns
        -------
        np.ndarray
            Distance matrix for this island (n x n).
        """
        # Build PTDF matrix
        ptdf_matrix, active_nodes = self._build_ptdf_matrix(
            ac_subgraph, island_nodes, slack_bus, config
        )

        log_debug(
            f"Built PTDF matrix: shape {ptdf_matrix.shape}",
            LogCategory.PARTITIONING,
        )

        # Compute distances from PTDF columns
        distance_matrix_active = self._compute_distances_from_ptdf(ptdf_matrix)

        # Integrate slack bus distances
        return self._integrate_slack_bus(
            distance_matrix_active, ptdf_matrix, island_nodes, slack_bus, active_nodes
        )

    def _build_ptdf_matrix(
        self,
        ac_subgraph: nx.DiGraph,
        nodes: list[Any],
        slack_bus: Any,
        config: VAElectricalConfig,
    ) -> tuple[np.ndarray, list[Any]]:
        """
        Build the Power Transfer Distribution Factor (PTDF) matrix.

        PTDF = diag{b} * K_sba * (K_sba^T * diag{b} * K_sba)^(-1)

        Parameters
        ----------
        ac_subgraph : nx.DiGraph
            AC subgraph (lines + transformers).
        nodes : list[Any]
            Ordered list of nodes in island.
        slack_bus : Any
            Slack bus node.
        config : VAElectricalConfig
            Configuration parameters.

        Returns
        -------
        tuple[np.ndarray, list[Any]]
            PTDF matrix (n_edges x n_active) and list of active nodes.
        """
        # Extract edges and susceptances from AC subgraph
        edges, susceptances = self._extract_susceptances(ac_subgraph, config)

        if len(edges) == 0:
            raise PartitioningError(
                "No valid AC edges found for PTDF matrix construction.",
                strategy=self._get_strategy_name(),
            )

        # Build slack-bus-adjusted incidence matrix
        active_nodes = [n for n in nodes if n != slack_bus]
        n_active = len(active_nodes)
        n_edges = len(edges)

        node_to_idx = {node: idx for idx, node in enumerate(active_nodes)}

        # Pre-compute indices for assignment
        u_indices = np.array([node_to_idx.get(u, -1) for u, v in edges], dtype=np.int32)
        v_indices = np.array([node_to_idx.get(v, -1) for u, v in edges], dtype=np.int32)
        edge_indices = np.arange(n_edges, dtype=np.int32)

        # Build incidence matrix K_sba
        K_sba = np.zeros((n_edges, n_active), dtype=np.float64)

        # Assignment for source nodes (u -> edge leaves u)
        valid_u = u_indices >= 0
        K_sba[edge_indices[valid_u], u_indices[valid_u]] = -1.0

        # Assignment for target nodes (v -> edge enters v)
        valid_v = v_indices >= 0
        K_sba[edge_indices[valid_v], v_indices[valid_v]] = 1.0

        # Build susceptance matrix B = K_sba^T @ diag(b) @ K_sba
        # Efficient: (K.T * b) @ K avoids explicit diagonal matrix
        K_scaled = K_sba.T * susceptances
        B_matrix = K_scaled @ K_sba
        B_matrix = (B_matrix + B_matrix.T) / 2.0  # Ensure symmetry

        # Apply Tikhonov regularization for numerical stability
        if config.regularization_factor > 0:
            B_matrix += config.regularization_factor * np.eye(n_active)

        # Compute PTDF: solve B @ X = A^T where A = diag(b) @ K_sba
        A = susceptances[:, np.newaxis] * K_sba

        try:
            # Using assume_a='sym' enables faster symmetric solver
            X = solve(B_matrix, A.T, assume_a="sym")
            ptdf_matrix = X.T
        except LinAlgError:
            log_warning(
                "B matrix is singular, using least-squares solution.",
                LogCategory.PARTITIONING,
            )
            X, _, _, _ = np.linalg.lstsq(B_matrix, A.T, rcond=None)
            ptdf_matrix = X.T

        return ptdf_matrix, active_nodes

    def _extract_susceptances(
        self,
        ac_subgraph: nx.DiGraph,
        config: VAElectricalConfig,
    ) -> tuple[list[tuple[Any, Any]], np.ndarray]:
        """
        Extract edges and susceptances from AC subgraph.

        Parameters
        ----------
        ac_subgraph : nx.DiGraph
            AC subgraph (lines + transformers).
        config : VAElectricalConfig
            Configuration parameters.

        Returns
        -------
        tuple[list[tuple[Any, Any]], np.ndarray]
            List of edges and array of susceptances (b = 1/x).
        """
        edges = []
        susceptances = []
        zero_count = 0

        for u, v, data in ac_subgraph.edges(data=True):
            reactance = data.get("x")

            if reactance is None:
                raise PartitioningError(
                    f"AC edge ({u}, {v}) missing reactance 'x'",
                    strategy=self._get_strategy_name(),
                )

            if not isinstance(reactance, (int, float)):
                raise PartitioningError(
                    f"Edge ({u}, {v}) reactance must be numeric, got {type(reactance)}",
                    strategy=self._get_strategy_name(),
                )

            if reactance == 0:
                zero_count += 1
                reactance = config.zero_reactance_replacement

            edges.append((u, v))
            susceptances.append(1.0 / reactance)

        if zero_count > 0:
            log_warning(
                f"{zero_count} edge(s) with zero reactance replaced with "
                f"{config.zero_reactance_replacement}",
                LogCategory.PARTITIONING,
            )

        return edges, np.array(susceptances, dtype=np.float64)

    def _compute_distances_from_ptdf(self, ptdf_matrix: np.ndarray) -> np.ndarray:
        """
        Compute electrical distances from PTDF column differences.

        d_ij = ||PTDF[:,i] - PTDF[:,j]||_2

        Uses identity: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
        This enables O(n²) computation via optimized BLAS matrix multiply.

        Parameters
        ----------
        ptdf_matrix : np.ndarray
            PTDF matrix (n_edges x n_active).

        Returns
        -------
        np.ndarray
            Distance matrix (n_active x n_active).
        """
        # Use float32 for faster computation on large matrices
        X = ptdf_matrix.T.astype(np.float32, copy=False)

        # Squared norms: ||x_i||^2 for each row
        norms_sq = np.einsum("ij,ij->i", X, X)

        # Gram matrix: <x_i, x_j> for all pairs (BLAS-optimized)
        gram = X @ X.T

        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
        dist_sq = norms_sq[:, np.newaxis] + norms_sq - 2.0 * gram

        # Handle numerical errors (small negative values)
        np.maximum(dist_sq, 0.0, out=dist_sq)

        # Square root in-place
        distances = np.sqrt(dist_sq, out=dist_sq)
        np.fill_diagonal(distances, 0.0)

        if np.any(np.isnan(distances)):
            raise PartitioningError(
                "NaN values in distance matrix after PTDF calculation.",
                strategy=self._get_strategy_name(),
            )

        return distances.astype(np.float64)

    @staticmethod
    def _integrate_slack_bus(
        dist_active: np.ndarray,
        ptdf_matrix: np.ndarray,
        island_nodes: list[Any],
        slack_bus: Any,
        active_nodes: list[Any],
    ) -> np.ndarray:
        """
        Integrate slack bus into distance matrix.

        Slack bus has implicit PTDF column of zeros (reference bus).
        Distance from slack to node i = ||PTDF[:,i]||_2

        Parameters
        ----------
        dist_active : np.ndarray
            Distance matrix for active nodes (n_active x n_active).
        ptdf_matrix : np.ndarray
            PTDF matrix (n_edges x n_active).
        island_nodes : list[Any]
            All nodes in island.
        slack_bus : Any
            Slack bus node.
        active_nodes : list[Any]
            Active nodes (excluding slack).

        Returns
        -------
        np.ndarray
            Full distance matrix including slack bus (n_island x n_island).
        """
        n_island = len(island_nodes)
        full_dist = np.zeros((n_island, n_island), dtype=np.float64)

        slack_idx = island_nodes.index(slack_bus)
        active_indices = np.array([island_nodes.index(node) for node in active_nodes])

        # Copy active distances using fancy indexing
        full_dist[np.ix_(active_indices, active_indices)] = dist_active

        # Slack bus distances: ||PTDF[:,i]||_2 (L2 norm of each column)
        slack_dists = np.linalg.norm(ptdf_matrix, axis=0)
        full_dist[slack_idx, active_indices] = slack_dists
        full_dist[active_indices, slack_idx] = slack_dists

        return full_dist
