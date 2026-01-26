from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from npap.exceptions import PartitioningError
from npap.interfaces import PartitioningStrategy
from npap.logging import LogCategory, log_debug, log_info, log_warning
from npap.utils import (
    compute_geographical_distances,
    create_partition_map,
    run_hierarchical,
    run_kmedoids,
    validate_partition,
    validate_required_attributes,
    with_runtime_config,
)


@dataclass
class VAGeographicalConfig:
    """
    Configuration for voltage-aware geographical partitioning.

    Attributes
    ----------
    voltage_tolerance : float
        Tolerance for voltage comparison (kV). Nodes with voltages within
        this tolerance are considered the same voltage level. Default is 1.0 kV.
    infinite_distance : float
        Value used to represent "infinite" distance between nodes at different
        voltage levels or DC islands. Using a large finite value instead of
        np.inf to avoid numerical issues in clustering algorithms.
    proportional_clustering : bool
        If False (default), runs clustering on full matrix with infinite
        distances between voltage levels. If True, clusters each voltage
        island independently with proportional cluster distribution.
    hierarchical_linkage : str
        Linkage criterion for hierarchical clustering. Must be 'complete',
        'average', or 'single'. Note: 'ward' is NOT supported with precomputed
        distances. Default is 'complete' which works well with infinite
        distances (prevents cross-voltage merging).
    """

    voltage_tolerance: float = 1.0
    infinite_distance: float = 1e4
    proportional_clustering: bool = False
    hierarchical_linkage: str = "complete"


class VAGeographicalPartitioning(PartitioningStrategy):
    """
    Voltage-Aware Geographical Partitioning Strategy with DC Island Support.

    This strategy partitions nodes based on geographical distance while
    respecting both DC island boundaries and voltage level boundaries.

    Constraint Hierarchy:
    1. DC Islands: Nodes in different DC islands are assigned infinite distance.
       DC islands represent separate AC networks connected only via DC links.
    2. Voltage Levels: Within each DC island, nodes at different voltage levels
       are also assigned infinite distance.

    This ensures clustering only occurs within the same DC island AND
    the same voltage level, which is physically meaningful for power networks.

    Two clustering modes are available:

    Standard mode (default):
        - Builds full NxN distance matrix
        - Sets d(i,j) = inf if dc_island(i) != dc_island(j) OR voltage(i) != voltage(j)
        - Runs single clustering algorithm on entire matrix
        - Algorithm handles infinite distances to respect boundaries

    Proportional mode:
        - Groups nodes by (dc_island, voltage_level) combination
        - Distributes n_clusters proportionally among groups
        - Runs clustering independently on each group
        - Guaranteed balanced distribution per group

    Supported algorithms:
        - 'kmedoids': K-Medoids clustering (works naturally with precomputed distances)
        - 'hierarchical': Agglomerative clustering with precomputed distances
                         (uses 'complete' linkage by default, configurable)

    Configuration can be provided at:
    - Instantiation time (via `config` parameter in __init__)
    - Partition time (via `config` or individual parameters in partition())

    Partition-time parameters override instance defaults for that call only.
    """

    SUPPORTED_ALGORITHMS = ["kmedoids", "hierarchical"]
    SUPPORTED_DISTANCE_METRICS = ["euclidean", "haversine"]
    SUPPORTED_LINKAGES = ["complete", "average", "single"]

    # Config parameter names for runtime override detection
    _CONFIG_PARAMS = {
        "voltage_tolerance",
        "infinite_distance",
        "proportional_clustering",
        "hierarchical_linkage",
    }

    def __init__(
        self,
        algorithm: str = "kmedoids",
        distance_metric: str = "euclidean",
        voltage_attr: str = "voltage",
        dc_island_attr: str = "dc_island",
        config: VAGeographicalConfig | None = None,
    ):
        """
        Initialize voltage-aware geographical partitioning strategy.

        Parameters
        ----------
        algorithm : str, default='kmedoids'
            Clustering algorithm ('kmedoids', 'hierarchical').
        distance_metric : str, default='euclidean'
            Distance metric ('euclidean', 'haversine').
        voltage_attr : str, default='voltage'
            Node attribute name containing voltage level.
        dc_island_attr : str, default='dc_island'
            Node attribute name containing DC island ID.
        config : VAGeographicalConfig, optional
            Configuration parameters for the algorithm.

        Raises
        ------
        ValueError
            If unsupported algorithm, distance metric, or linkage.
        """
        self.algorithm = algorithm
        self.distance_metric = distance_metric
        self.voltage_attr = voltage_attr
        self.dc_island_attr = dc_island_attr
        self.config = config or VAGeographicalConfig()

        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm: {algorithm}. "
                f"Supported: {', '.join(self.SUPPORTED_ALGORITHMS)}"
            )

        if distance_metric not in self.SUPPORTED_DISTANCE_METRICS:
            raise ValueError(
                f"Unsupported distance metric: {distance_metric}. "
                f"Supported: {', '.join(self.SUPPORTED_DISTANCE_METRICS)}"
            )

        if self.config.hierarchical_linkage not in self.SUPPORTED_LINKAGES:
            raise ValueError(
                f"Unsupported hierarchical linkage: {self.config.hierarchical_linkage}. "
                f"Supported: {', '.join(self.SUPPORTED_LINKAGES)}. "
                "Note: 'ward' linkage is not supported with precomputed distance matrices."
            )

        log_debug(
            f"Initialized VAGeographicalPartitioning: algorithm={algorithm}, "
            f"metric={distance_metric}, proportional={self.config.proportional_clustering}",
            LogCategory.PARTITIONING,
        )

    @property
    def required_attributes(self) -> dict[str, list[str]]:
        """Required node attributes for voltage-aware geographical partitioning."""
        return {
            "nodes": ["lat", "lon", self.voltage_attr, self.dc_island_attr],
            "edges": [],
        }

    def _get_strategy_name(self) -> str:
        """Get descriptive strategy name for error messages."""
        mode = "proportional" if self.config.proportional_clustering else "standard"
        return f"va_geographical_{mode}_{self.algorithm}"

    @with_runtime_config(VAGeographicalConfig, _CONFIG_PARAMS)
    @validate_required_attributes
    def partition(self, graph: nx.Graph, **kwargs) -> dict[int, list[Any]]:
        """
        Partition nodes based on DC island and voltage-aware geographical distance.

        Parameters
        ----------
        graph : nx.Graph
            NetworkX graph with lat, lon, voltage, and dc_island attributes.
        **kwargs : dict
            Additional parameters:

            - n_clusters : Number of clusters (required)
            - random_state : Random seed for reproducibility (default: 42)
            - max_iter : Maximum iterations for clustering (default: 300)
            - config : VAGeographicalConfig instance to override instance config
            - voltage_tolerance : Override config parameter
            - infinite_distance : Override config parameter
            - proportional_clustering : Override config parameter
            - hierarchical_linkage : Override config parameter

        Returns
        -------
        dict[int, list[Any]]
            Dictionary mapping cluster_id -> list of node_ids.

        Raises
        ------
        PartitioningError
            If partitioning fails.
        """
        try:
            # Get effective config (injected by decorator)
            effective_config = kwargs.get("_effective_config", self.config)

            # Validate hierarchical_linkage if overridden
            if effective_config.hierarchical_linkage not in self.SUPPORTED_LINKAGES:
                raise PartitioningError(
                    f"Unsupported hierarchical linkage: {effective_config.hierarchical_linkage}. "
                    f"Supported: {', '.join(self.SUPPORTED_LINKAGES)}",
                    strategy=self._get_strategy_name(),
                )

            n_clusters = kwargs.get("n_clusters")
            if n_clusters is None or n_clusters <= 0:
                raise PartitioningError(
                    "Voltage-aware geographical partitioning requires a positive 'n_clusters' parameter.",
                    strategy=self._get_strategy_name(),
                )

            # Extract node data
            nodes = list(graph.nodes())
            n_nodes = len(nodes)

            if n_clusters > n_nodes:
                raise PartitioningError(
                    f"Cannot create {n_clusters} clusters from {n_nodes} nodes.",
                    strategy=self._get_strategy_name(),
                )

            # Extract coordinates, voltages, and DC islands
            coordinates, voltages, dc_islands = self._extract_node_data(graph, nodes)

            # Get unique groups summary for validation
            n_groups = self._count_unique_groups(dc_islands, voltages, effective_config)

            # Log summary
            self._log_group_summary(dc_islands, voltages, n_groups)

            if n_clusters < n_groups:
                log_warning(
                    f"Requested {n_clusters} clusters but found {n_groups} "
                    f"distinct (dc_island, voltage_level) groups. Some groups may share clusters, "
                    f"but infinite distance constraints will be respected.",
                    LogCategory.PARTITIONING,
                    warn_user=False,
                )

            log_info(
                f"Starting VA geographical partitioning: {self.algorithm}, "
                f"n_clusters={n_clusters}, groups={n_groups}",
                LogCategory.PARTITIONING,
            )

            # Choose clustering mode based on configuration
            if effective_config.proportional_clustering:
                partition_map = self._proportional_partition(
                    nodes, coordinates, voltages, dc_islands, effective_config, **kwargs
                )
            else:
                partition_map = self._standard_partition(
                    nodes, coordinates, voltages, dc_islands, effective_config, **kwargs
                )

            # Validate using utility function
            validate_partition(partition_map, n_nodes, self._get_strategy_name())

            # Validate DC island and voltage consistency
            self._validate_cluster_consistency(graph, partition_map, effective_config)

            log_info(
                f"VA geographical partitioning complete: {len(partition_map)} clusters",
                LogCategory.PARTITIONING,
            )

            return partition_map

        except Exception as e:
            if isinstance(e, PartitioningError):
                raise
            raise PartitioningError(
                f"Voltage-aware geographical partitioning failed: {e}",
                strategy=self._get_strategy_name(),
                graph_info={
                    "nodes": len(list(graph.nodes())),
                    "edges": len(graph.edges()),
                },
            ) from e

    def _standard_partition(
        self,
        nodes: list[Any],
        coordinates: np.ndarray,
        voltages: np.ndarray,
        dc_islands: np.ndarray,
        config: VAGeographicalConfig,
        **kwargs,
    ) -> dict[int, list[Any]]:
        """
        Partition using single clustering on full matrix with infinite distances.

        This mode builds a distance matrix where nodes in different DC islands
        OR different voltage levels have infinite distance.

        Parameters
        ----------
        nodes : list[Any]
            List of node IDs.
        coordinates : np.ndarray
            Node coordinates [n x 2].
        voltages : np.ndarray
            Node voltage values.
        dc_islands : np.ndarray
            Node DC island IDs.
        config : VAGeographicalConfig
            Configuration instance.
        **kwargs : dict
            Clustering parameters.

        Returns
        -------
        dict[int, list[Any]]
            Partition mapping.
        """
        log_debug("Using standard partitioning mode", LogCategory.PARTITIONING)

        distance_matrix = self._build_aware_distance_matrix(
            coordinates, voltages, dc_islands, config
        )

        # Run clustering algorithm
        labels = self._run_clustering(distance_matrix, config, **kwargs)

        # Create partition mapping using utility function
        return create_partition_map(nodes, labels)

    def _proportional_partition(
        self,
        nodes: list[Any],
        coordinates: np.ndarray,
        voltages: np.ndarray,
        dc_islands: np.ndarray,
        config: VAGeographicalConfig,
        **kwargs,
    ) -> dict[int, list[Any]]:
        """
        Partition each (dc_island, voltage) group independently with proportional distribution.

        Parameters
        ----------
        nodes : list[Any]
            List of node IDs.
        coordinates : np.ndarray
            Node coordinates [n x 2].
        voltages : np.ndarray
            Node voltage values.
        dc_islands : np.ndarray
            Node DC island IDs.
        config : VAGeographicalConfig
            Configuration instance.
        **kwargs : dict
            Clustering parameters.

        Returns
        -------
        dict[int, list[Any]]
            Partition mapping.
        """
        log_debug("Using proportional partitioning mode", LogCategory.PARTITIONING)

        n_clusters = kwargs.get("n_clusters")

        # Group nodes by (dc_island, voltage)
        groups = self._group_by_island_and_voltage(dc_islands, voltages, config)

        # Allocate clusters proportionally
        allocation = self._allocate_clusters(groups, n_clusters)

        log_debug(f"Cluster allocation: {allocation}", LogCategory.PARTITIONING)

        partition_map: dict[int, list[Any]] = {}
        cluster_offset = 0

        for group_key, node_indices in groups.items():
            n_clust = allocation[group_key]
            group_nodes = [nodes[i] for i in node_indices]
            group_coords = coordinates[node_indices]

            # Handle edge cases: more clusters than nodes
            if len(group_nodes) <= n_clust:
                for node_id in group_nodes:
                    partition_map[cluster_offset] = [node_id]
                    cluster_offset += 1
                continue

            # Cluster this group using geographical distances only
            distances = compute_geographical_distances(group_coords, self.distance_metric)
            labels = self._run_clustering(
                distances,
                config,
                n_clusters=n_clust,
                random_state=kwargs.get("random_state", 42),
                max_iter=kwargs.get("max_iter", 300),
            )

            for local_idx, label in enumerate(labels):
                global_id = cluster_offset + int(label)
                if global_id not in partition_map:
                    partition_map[global_id] = []
                partition_map[global_id].append(group_nodes[local_idx])

            cluster_offset += n_clust

        return partition_map

    def _run_clustering(
        self, distance_matrix: np.ndarray, config: VAGeographicalConfig, **kwargs
    ) -> np.ndarray:
        """
        Dispatch to appropriate clustering algorithm using utility functions.

        Parameters
        ----------
        distance_matrix : np.ndarray
            Precomputed distance matrix (n x n).
        config : VAGeographicalConfig
            Configuration instance.
        **kwargs : dict
            Must include 'n_clusters'.

        Returns
        -------
        np.ndarray
            Array of cluster labels.
        """
        n_clusters = kwargs.get("n_clusters")

        if self.algorithm == "kmedoids":
            return run_kmedoids(distance_matrix, n_clusters)
        elif self.algorithm == "hierarchical":
            return run_hierarchical(distance_matrix, n_clusters, config.hierarchical_linkage)
        else:
            raise PartitioningError(
                f"Unknown algorithm: {self.algorithm}",
                strategy=self._get_strategy_name(),
            )

    # =========================================================================
    # DATA EXTRACTION METHODS
    # =========================================================================

    def _extract_node_data(
        self, graph: nx.Graph, nodes: list[Any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract coordinates, voltage levels, and DC island IDs from nodes.

        Parameters
        ----------
        graph : nx.Graph
            NetworkX graph.
        nodes : list[Any]
            List of node IDs.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Tuple of (coordinates array [n x 2], voltages array [n], dc_islands array [n]).

        Raises
        ------
        PartitioningError
            If required attributes are missing.
        """
        coordinates = []
        voltages = []
        dc_islands = []

        for node in nodes:
            node_data = graph.nodes[node]

            lat = node_data.get("lat")
            lon = node_data.get("lon")

            if lat is None or lon is None:
                raise PartitioningError(
                    f"Node {node} missing latitude or longitude",
                    strategy=self._get_strategy_name(),
                    graph_info={"nodes": len(nodes)},
                )

            coordinates.append([lat, lon])
            voltages.append(node_data.get(self.voltage_attr))

            dc_island = node_data.get(self.dc_island_attr)
            if dc_island is None:
                raise PartitioningError(
                    f"Node {node} missing '{self.dc_island_attr}' attribute. "
                    "Ensure the graph was loaded with VoltageAwareStrategy (va_loader).",
                    strategy=self._get_strategy_name(),
                    graph_info={"nodes": len(nodes)},
                )
            dc_islands.append(dc_island)

        return (
            np.array(coordinates),
            np.array(voltages, dtype=object),
            np.array(dc_islands),
        )

    def _group_by_island_and_voltage(
        self, dc_islands: np.ndarray, voltages: np.ndarray, config: VAGeographicalConfig
    ) -> dict[tuple[Any, Any], list[int]]:
        """
        Group node indices by (dc_island, voltage_level) combination.

        Parameters
        ----------
        dc_islands : np.ndarray
            Array of DC island IDs.
        voltages : np.ndarray
            Array of voltage values.
        config : VAGeographicalConfig
            Configuration instance.

        Returns
        -------
        dict[tuple[Any, Any], list[int]]
            Dict mapping (dc_island, voltage_level) -> list of node indices.
        """
        groups: dict[tuple[Any, Any], list[int]] = {}

        for idx in range(len(dc_islands)):
            dc_island = dc_islands[idx]
            voltage = voltages[idx]

            # Find matching group key
            matched_key = None
            for existing_key in groups.keys():
                existing_island, existing_voltage = existing_key
                if self._islands_compatible(
                    dc_island, existing_island
                ) and self._voltages_compatible(voltage, existing_voltage, config):
                    matched_key = existing_key
                    break

            if matched_key is not None:
                groups[matched_key].append(idx)
            else:
                # Create new group key
                island_key = dc_island if dc_island is not None else -1
                voltage_key = voltage if voltage is not None else "unknown"
                groups[(island_key, voltage_key)] = [idx]

        return groups

    @staticmethod
    def _allocate_clusters(
        groups: dict[tuple[Any, Any], list[int]], n_clusters: int
    ) -> dict[tuple[Any, Any], int]:
        """
        Allocate clusters proportionally to groups.

        Parameters
        ----------
        groups : dict[tuple[Any, Any], list[int]]
            Dict mapping (dc_island, voltage) -> node indices.
        n_clusters : int
            Total clusters to allocate.

        Returns
        -------
        dict[tuple[Any, Any], int]
            Dict mapping (dc_island, voltage) -> number of clusters.
        """
        total_nodes = sum(len(indices) for indices in groups.values())
        allocation: dict[tuple[Any, Any], int] = {}

        # Sort by size (largest first) for stable allocation
        sorted_keys = sorted(groups.keys(), key=lambda k: len(groups[k]), reverse=True)

        remaining = n_clusters
        for i, group_key in enumerate(sorted_keys):
            n_nodes = len(groups[group_key])

            if i == len(sorted_keys) - 1:
                # Last group gets remaining clusters
                allocation[group_key] = max(1, remaining)
            else:
                # Proportional allocation
                proportion = n_nodes / total_nodes
                allocated = max(1, int(round(n_clusters * proportion)))
                allocated = min(allocated, n_nodes, remaining - (len(sorted_keys) - i - 1))
                allocation[group_key] = allocated
                remaining -= allocated

        return allocation

    @staticmethod
    def _count_unique_groups(
        dc_islands: np.ndarray, voltages: np.ndarray, config: VAGeographicalConfig
    ) -> int:
        """
        Count unique (dc_island, voltage_level) combinations.

        Parameters
        ----------
        dc_islands : np.ndarray
            Array of DC island IDs.
        voltages : np.ndarray
            Array of voltage values.
        config : VAGeographicalConfig
            Configuration instance.

        Returns
        -------
        int
            Number of unique groups.
        """
        seen_groups = set()

        for i in range(len(dc_islands)):
            dc_island = dc_islands[i]
            voltage = voltages[i]

            # Create a hashable group key
            # Round voltage for tolerance-based matching
            if voltage is not None and isinstance(voltage, (int, float)):
                voltage_key = round(voltage / max(config.voltage_tolerance, 0.1)) * max(
                    config.voltage_tolerance, 0.1
                )
            else:
                voltage_key = voltage

            group_key = (dc_island, voltage_key)
            seen_groups.add(group_key)

        return len(seen_groups)

    # =========================================================================
    # DISTANCE MATRIX METHODS
    # =========================================================================

    def _build_aware_distance_matrix(
        self,
        coordinates: np.ndarray,
        voltages: np.ndarray,
        dc_islands: np.ndarray,
        config: VAGeographicalConfig,
    ) -> np.ndarray:
        """
        Build distance matrix with DC island and voltage awareness.

        For nodes in the same DC island AND same voltage level (within tolerance),
        uses geographical distance. Otherwise, assigns infinite distance.

        Constraint hierarchy:

        1. Different DC islands -> infinite distance
        2. Same DC island, different voltage -> infinite distance
        3. Same DC island, same voltage -> geographical distance

        Parameters
        ----------
        coordinates : np.ndarray
            Array of [lat, lon] coordinates (n x 2).
        voltages : np.ndarray
            Array of voltage levels (n).
        dc_islands : np.ndarray
            Array of DC island IDs (n).
        config : VAGeographicalConfig
            Configuration instance.

        Returns
        -------
        np.ndarray
            Distance matrix (n x n) where:

            - d[i,j] = geographical_distance if same DC island AND same voltage
            - d[i,j] = infinite_distance otherwise
            - d[i,i] = 0 (diagonal)
        """
        n_nodes = len(coordinates)

        # Calculate geographical distances using utility function
        geo_distances = compute_geographical_distances(coordinates, self.distance_metric)

        # same_dc_island[i,j] = True if dc_islands[i] == dc_islands[j]
        same_dc_island = dc_islands[:, np.newaxis] == dc_islands[np.newaxis, :]

        # Handle None DC islands (incompatible with everything)
        dc_not_none = np.array([island is not None for island in dc_islands])
        dc_both_valid = dc_not_none[:, np.newaxis] & dc_not_none[np.newaxis, :]
        dc_compatible = same_dc_island & dc_both_valid

        # Voltage compatibility mask
        voltage_compatible = self._build_voltage_compatibility_mask(voltages, config)

        # Combine masks
        compatible_mask = dc_compatible & voltage_compatible

        # Build distance matrix using vectorized where
        distance_matrix = np.where(compatible_mask, geo_distances, config.infinite_distance)
        np.fill_diagonal(distance_matrix, 0.0)

        # Log statistics (count compatible pairs, excluding diagonal)
        compatible_pairs = (np.sum(compatible_mask) - n_nodes) // 2

        log_debug(
            f"Built aware distance matrix: {compatible_pairs} compatible pairs",
            LogCategory.PARTITIONING,
        )

        return distance_matrix

    @staticmethod
    def _build_voltage_compatibility_mask(
        voltages: np.ndarray, config: VAGeographicalConfig
    ) -> np.ndarray:
        """
        Build voltage compatibility mask.

        Handles three cases:

        1. Both numeric: compatible if within tolerance
        2. Both non-numeric (not None): compatible if exact match
        3. None values: incompatible with everything

        Parameters
        ----------
        voltages : np.ndarray
            Array of voltage values.
        config : VAGeographicalConfig
            Configuration instance.

        Returns
        -------
        np.ndarray
            Boolean mask (n x n) where True indicates voltage compatibility.
        """
        # Categorize voltages
        is_numeric = np.array([isinstance(v, (int, float)) for v in voltages])
        is_none = np.array([v is None for v in voltages])

        # None values are incompatible with everything
        neither_none = ~is_none[:, np.newaxis] & ~is_none[np.newaxis, :]

        # For numeric values: tolerance-based comparison
        float_voltages = np.array(
            [float(v) if isinstance(v, (int, float)) else 0.0 for v in voltages],
            dtype=np.float64,
        )
        voltage_diff = np.abs(float_voltages[:, np.newaxis] - float_voltages[np.newaxis, :])
        numeric_compatible = voltage_diff <= config.voltage_tolerance
        both_numeric = is_numeric[:, np.newaxis] & is_numeric[np.newaxis, :]

        # Initialize with numeric compatibility for numeric pairs
        voltage_compatible = both_numeric & numeric_compatible

        # Handle non-numeric values (not None): exact equality required
        is_non_numeric = ~is_numeric & ~is_none
        if np.any(is_non_numeric):
            # For non-numeric pairs, check exact equality
            both_non_numeric = is_non_numeric[:, np.newaxis] & is_non_numeric[np.newaxis, :]
            if np.any(both_non_numeric):
                # Build equality mask for non-numeric values only
                non_numeric_indices = np.where(is_non_numeric)[0]
                for i in non_numeric_indices:
                    for j in non_numeric_indices:
                        if voltages[i] == voltages[j]:
                            voltage_compatible[i, j] = True

        # Apply None mask
        voltage_compatible &= neither_none

        return voltage_compatible

    # =========================================================================
    # COMPATIBILITY CHECK METHODS
    # =========================================================================

    @staticmethod
    def _islands_compatible(island1: Any, island2: Any) -> bool:
        """
        Check if two DC island IDs are compatible (same island).

        Parameters
        ----------
        island1 : Any
            First DC island ID.
        island2 : Any
            Second DC island ID.

        Returns
        -------
        bool
            True if islands are the same, False otherwise.
        """
        # Handle missing DC island IDs - isolated nodes
        if island1 is None or island2 is None:
            return False

        # Direct comparison (DC island IDs should be integers from component detection)
        return island1 == island2

    @staticmethod
    def _voltages_compatible(v1: Any, v2: Any, config: VAGeographicalConfig) -> bool:
        """
        Check if two voltage levels are compatible (same voltage island).

        Parameters
        ----------
        v1 : Any
            First voltage value.
        v2 : Any
            Second voltage value.
        config : VAGeographicalConfig
            Configuration instance.

        Returns
        -------
        bool
            True if voltages are compatible (within tolerance), False otherwise.
        """
        # Handle missing voltages - nodes without voltage are isolated
        if v1 is None or v2 is None:
            return False

        # Handle non-numeric voltages (exact match required)
        if not isinstance(v1, (int, float)) or not isinstance(v2, (int, float)):
            return v1 == v2

        return abs(float(v1) - float(v2)) <= config.voltage_tolerance

    # =========================================================================
    # LOGGING AND VALIDATION METHODS
    # =========================================================================

    @staticmethod
    def _log_group_summary(dc_islands: np.ndarray, voltages: np.ndarray, n_groups: int) -> None:
        """Log summary of (dc_island, voltage_level) groups found."""
        n_islands = len(set(dc_islands))
        n_voltages = len(set(voltages))

        log_info(
            f"Voltage-aware partitioning: {n_islands} DC island(s), "
            f"{n_voltages} voltage level(s) -> {n_groups} group(s)",
            LogCategory.PARTITIONING,
        )

    def _validate_cluster_consistency(
        self,
        graph: nx.Graph,
        partition_map: dict[int, list[Any]],
        config: VAGeographicalConfig,
    ) -> None:
        """
        Validate that clusters don't mix incompatible DC islands or voltage levels.

        With infinite distances, clusters should never mix:

        1. Different DC islands
        2. Different voltage levels within the same DC island

        Parameters
        ----------
        graph : nx.Graph
            Original NetworkX graph.
        partition_map : dict[int, list[Any]]
            Resulting partition mapping.
        config : VAGeographicalConfig
            Configuration instance.
        """
        for cluster_id, nodes in partition_map.items():
            dc_islands_in_cluster = set()
            voltages_in_cluster = set()

            for node in nodes:
                node_data = graph.nodes[node]

                # Check DC island
                dc_island = node_data.get(self.dc_island_attr)
                if dc_island is not None:
                    dc_islands_in_cluster.add(dc_island)

                # Check voltage
                v = node_data.get(self.voltage_attr)
                if v is None:
                    v = node_data.get("voltage", node_data.get("v_nom"))

                if v is not None and isinstance(v, (int, float)):
                    v_rounded = round(v / max(config.voltage_tolerance, 0.1)) * max(
                        config.voltage_tolerance, 0.1
                    )
                    voltages_in_cluster.add(v_rounded)
                elif v is not None:
                    voltages_in_cluster.add(v)

            # Check for DC island mixing
            if len(dc_islands_in_cluster) > 1:
                log_warning(
                    f"Cluster {cluster_id} contains nodes from multiple DC islands: "
                    f"{dc_islands_in_cluster}. This should not happen with infinite distances.",
                    LogCategory.PARTITIONING,
                    warn_user=False,
                )

            # Check for voltage mixing
            if len(voltages_in_cluster) > 1:
                log_warning(
                    f"Cluster {cluster_id} contains multiple voltage levels: {voltages_in_cluster}.",
                    LogCategory.PARTITIONING,
                    warn_user=False,
                )
