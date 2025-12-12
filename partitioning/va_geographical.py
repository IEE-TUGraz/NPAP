from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import networkx as nx
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances, haversine_distances
from sklearn_extra.cluster import KMedoids

from exceptions import PartitioningError
from interfaces import PartitioningStrategy


@dataclass
class VAGeographicalConfig:
    """
    Configuration for voltage-aware geographical partitioning.

    Attributes:
        voltage_tolerance: Tolerance for voltage comparison (kV).
                          Nodes with voltages within this tolerance are
                          considered the same voltage level. Default is 1.0 kV.
        infinite_distance: Value used to represent "infinite" distance
                          between nodes at different voltage levels or DC islands.
                          Using a large finite value instead of np.inf
                          to avoid numerical issues in clustering algorithms.
        proportional_clustering: If False (default), runs clustering on full
                              matrix with infinite distances between voltage levels.
                              If True, clusters each voltage island independently
                              with proportional cluster distribution.
        hierarchical_linkage: Linkage criterion for hierarchical clustering.
                             Must be 'complete', 'average', or 'single'.
                             Note: 'ward' is NOT supported with precomputed distances.
                             Default is 'complete' which works well with infinite
                             distances (prevents cross-voltage merging).
    """
    voltage_tolerance: float = 1.0
    infinite_distance: float = 1e4
    proportional_clustering: bool = False
    hierarchical_linkage: str = 'complete'


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
    """

    SUPPORTED_ALGORITHMS = ['kmedoids', 'hierarchical']
    SUPPORTED_DISTANCE_METRICS = ['euclidean', 'haversine']
    SUPPORTED_LINKAGES = ['complete', 'average', 'single']

    def __init__(self, algorithm: str = 'kmedoids',
                 distance_metric: str = 'euclidean',
                 voltage_attr: str = 'voltage',
                 dc_island_attr: str = 'dc_island',
                 config: Optional[VAGeographicalConfig] = None):
        """
        Initialize voltage-aware geographical partitioning strategy.

        Args:
            algorithm: Clustering algorithm. Supported options:
                      - 'kmedoids': K-Medoids (robust, works with precomputed distances)
                      - 'hierarchical': Agglomerative clustering (uses linkage from config)
            distance_metric: Distance metric for geographical distances.
                           'euclidean' for flat/projected coordinates,
                           'haversine' for great-circle distance on Earth.
            voltage_attr: Node attribute name containing voltage level.
            dc_island_attr: Node attribute name containing DC island ID.
            config: Configuration parameters for the algorithm.

        Raises:
            ValueError: If unsupported algorithm, distance metric, or linkage is specified.
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

    @property
    def required_attributes(self) -> Dict[str, List[str]]:
        """Required node attributes for voltage-aware geographical partitioning."""
        return {
            'nodes': ['lat', 'lon', self.voltage_attr, self.dc_island_attr],
            'edges': []
        }

    def partition(self, graph: nx.Graph, **kwargs) -> Dict[int, List[Any]]:
        """
        Partition nodes based on DC island and voltage-aware geographical distance.

        Args:
            graph: NetworkX graph with lat, lon, voltage, and dc_island attributes on nodes
            **kwargs: Additional parameters:
                - n_clusters: Number of clusters (required)
                - random_state: Random seed for reproducibility (default: 42)
                - max_iter: Maximum iterations for clustering (default: 300)

        Returns:
            Dictionary mapping cluster_id -> list of node_ids

        Raises:
            PartitioningError: If partitioning fails due to missing data,
                             invalid parameters, or clustering errors.
        """
        try:
            n_clusters = kwargs.get('n_clusters')
            if n_clusters is None or n_clusters <= 0:
                raise PartitioningError(
                    "Voltage-aware geographical partitioning requires a positive 'n_clusters' parameter.",
                    strategy=self._get_strategy_name()
                )

            # Extract node data
            nodes = list(graph.nodes())
            n_nodes = len(nodes)

            if n_clusters > n_nodes:
                raise PartitioningError(
                    f"Cannot create {n_clusters} clusters from {n_nodes} nodes.",
                    strategy=self._get_strategy_name()
                )

            # Extract coordinates, voltages, and DC islands
            coordinates, voltages, dc_islands = self._extract_node_data(graph, nodes)

            # Get unique groups summary for validation
            n_groups = self._count_unique_groups(dc_islands, voltages)

            # Warn if n_clusters < groups (some groups might be forced together)
            if n_clusters < n_groups:
                print(f"Warning: Requested {n_clusters} clusters but found {n_groups} "
                      f"distinct (dc_island, voltage_level) groups. Some groups may share clusters, "
                      f"but infinite distance constraints will be respected.")

            # Log summary
            self._log_group_summary(dc_islands, voltages)

            # Choose clustering mode based on configuration
            if self.config.proportional_clustering:
                partition_map = self._proportional_partition(
                    nodes, coordinates, voltages, dc_islands, **kwargs
                )
            else:
                partition_map = self._standard_partition(
                    nodes, coordinates, voltages, dc_islands, **kwargs
                )

            # Validate result
            self._validate_partition(partition_map, n_nodes)

            # Validate DC island and voltage consistency in clusters
            self._validate_cluster_consistency(graph, partition_map)

            return partition_map

        except Exception as e:
            if isinstance(e, PartitioningError):
                raise
            raise PartitioningError(
                f"Voltage-aware geographical partitioning failed: {e}",
                strategy=self._get_strategy_name(),
                graph_info={'nodes': len(list(graph.nodes())), 'edges': len(graph.edges())}
            ) from e

    def _get_strategy_name(self) -> str:
        """Get descriptive strategy name for error messages."""
        mode = "proportional" if self.config.proportional_clustering else "standard"
        return f"va_geographical_{mode}_{self.algorithm}"

    def _standard_partition(self, nodes: List[Any], coordinates: np.ndarray,
                            voltages: np.ndarray, dc_islands: np.ndarray,
                            **kwargs) -> Dict[int, List[Any]]:
        """
        Partition using single clustering on full matrix with infinite distances.

        This mode builds a distance matrix where nodes in different DC islands
        OR different voltage levels have infinite distance.

        Args:
            nodes: List of node IDs
            coordinates: Node coordinates [n x 2]
            voltages: Node voltage values
            dc_islands: Node DC island IDs
            **kwargs: Clustering parameters

        Returns:
            Partition mapping
        """
        # Build DC island and voltage-aware distance matrix
        distance_matrix = self._build_aware_distance_matrix(coordinates, voltages, dc_islands)

        # Run clustering algorithm
        labels = self._run_clustering(distance_matrix, **kwargs)

        # Create partition mapping
        return self._create_partition_map(nodes, labels)

    def _proportional_partition(self, nodes: List[Any], coordinates: np.ndarray,
                                voltages: np.ndarray, dc_islands: np.ndarray,
                                **kwargs) -> Dict[int, List[Any]]:
        """
        Partition each (dc_island, voltage) group independently with proportional distribution.

        Args:
            nodes: List of node IDs
            coordinates: Node coordinates [n x 2]
            voltages: Node voltage values
            dc_islands: Node DC island IDs
            **kwargs: Clustering parameters

        Returns:
            Partition mapping
        """
        n_clusters = kwargs.get('n_clusters')

        # Group nodes by (dc_island, voltage)
        groups = self._group_by_island_and_voltage(dc_islands, voltages)

        # Allocate clusters proportionally
        allocation = self._allocate_clusters(groups, n_clusters)

        # Partition each group
        partition_map: Dict[int, List[Any]] = {}
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
            distances = self._calculate_geographical_distances(group_coords)
            labels = self._run_clustering(
                distances,
                n_clusters=n_clust,
                random_state=kwargs.get('random_state', 42),
                max_iter=kwargs.get('max_iter', 300)
            )

            for local_idx, label in enumerate(labels):
                global_id = cluster_offset + int(label)
                if global_id not in partition_map:
                    partition_map[global_id] = []
                partition_map[global_id].append(group_nodes[local_idx])

            cluster_offset += n_clust

        return partition_map

    def _group_by_island_and_voltage(self, dc_islands: np.ndarray,
                                     voltages: np.ndarray) -> Dict[Tuple[Any, Any], List[int]]:
        """
        Group node indices by (dc_island, voltage_level) combination.

        Args:
            dc_islands: Array of DC island IDs
            voltages: Array of voltage values

        Returns:
            Dict mapping (dc_island, voltage_level) -> list of node indices
        """
        groups: Dict[Tuple[Any, Any], List[int]] = {}

        for idx in range(len(dc_islands)):
            dc_island = dc_islands[idx]
            voltage = voltages[idx]

            # Find matching group key
            matched_key = None
            for existing_key in groups.keys():
                existing_island, existing_voltage = existing_key
                if self._islands_compatible(dc_island, existing_island) and \
                        self._voltages_compatible(voltage, existing_voltage):
                    matched_key = existing_key
                    break

            if matched_key is not None:
                groups[matched_key].append(idx)
            else:
                # Create new group key
                island_key = dc_island if dc_island is not None else -1
                voltage_key = voltage if voltage is not None else 'unknown'
                groups[(island_key, voltage_key)] = [idx]

        return groups

    @staticmethod
    def _allocate_clusters(groups: Dict[Tuple[Any, Any], List[int]],
                           n_clusters: int) -> Dict[Tuple[Any, Any], int]:
        """
        Allocate clusters proportionally to groups.

        Args:
            groups: Dict mapping (dc_island, voltage) -> node indices
            n_clusters: Total clusters to allocate

        Returns:
            Dict mapping (dc_island, voltage) -> number of clusters
        """
        total_nodes = sum(len(indices) for indices in groups.values())
        allocation: Dict[Tuple[Any, Any], int] = {}

        # Sort by size (largest first) for stable allocation
        sorted_keys = sorted(groups.keys(),
                             key=lambda k: len(groups[k]),
                             reverse=True)

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

    def _run_clustering(self, distance_matrix: np.ndarray, **kwargs) -> np.ndarray:
        """
        Dispatch to the appropriate clustering algorithm.

        Args:
            distance_matrix: Precomputed distance matrix (n x n)
            **kwargs: Clustering parameters

        Returns:
            Array of cluster labels for each node

        Raises:
            PartitioningError: If clustering fails.
        """
        if self.algorithm == 'kmedoids':
            return self._kmedoids_clustering(distance_matrix, **kwargs)
        elif self.algorithm == 'hierarchical':
            return self._hierarchical_clustering(
                distance_matrix,
                linkage=self.config.hierarchical_linkage,
                **kwargs
            )
        else:
            raise PartitioningError(
                f"Unknown algorithm: {self.algorithm}",
                strategy=self._get_strategy_name()
            )

    @staticmethod
    def _kmedoids_clustering(distance_matrix: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform K-Medoids clustering with precomputed distance matrix.

        K-Medoids is robust to outliers and works naturally with precomputed
        distance matrices, making it ideal for voltage-aware clustering.

        Args:
            distance_matrix: Precomputed distance matrix (n x n)
            **kwargs: Clustering parameters:
                - n_clusters: Number of clusters
                - random_state: Random seed (default: 42)
                - max_iter: Maximum iterations (default: 300)

        Returns:
            Array of cluster labels for each node

        Raises:
            PartitioningError: If clustering fails.
        """
        try:
            n_clusters = kwargs.get('n_clusters')
            random_state = kwargs.get('random_state', 42)
            max_iter = kwargs.get('max_iter', 300)

            kmedoids = KMedoids(
                n_clusters=n_clusters,
                metric='precomputed',
                random_state=random_state,
                max_iter=max_iter
            )

            labels = kmedoids.fit_predict(distance_matrix)
            return labels

        except Exception as e:
            raise PartitioningError(
                f"K-Medoids clustering failed: {e}",
                strategy="va_geographical_kmedoids"
            ) from e

    @staticmethod
    def _hierarchical_clustering(distance_matrix: np.ndarray,
                                 linkage: str = 'complete',
                                 **kwargs) -> np.ndarray:
        """
        Perform hierarchical (agglomerative) clustering with precomputed distance matrix.

        Hierarchical clustering builds a tree of clusters and is deterministic
        (no random seed needed). With 'complete' linkage, it handles infinite
        distances well by preventing merges across voltage levels.

        Args:
            distance_matrix: Precomputed distance matrix (n x n)
            linkage: Linkage criterion ('complete', 'average', 'single').
                    Note: 'ward' is not supported with precomputed distances.
            **kwargs: Clustering parameters:
                - n_clusters: Number of clusters (required)

        Returns:
            Array of cluster labels for each node

        Raises:
            PartitioningError: If clustering fails.
        """
        try:
            n_clusters = kwargs.get('n_clusters')

            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage=linkage
            )

            labels = clustering.fit_predict(distance_matrix)
            return labels

        except Exception as e:
            raise PartitioningError(
                f"Hierarchical clustering failed: {e}",
                strategy="va_geographical_hierarchical"
            ) from e

    # =========================================================================
    # DATA EXTRACTION METHODS
    # =========================================================================

    def _extract_node_data(self, graph: nx.Graph,
                           nodes: List[Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract coordinates, voltage levels, and DC island IDs from nodes.

        Args:
            graph: NetworkX graph
            nodes: List of node IDs

        Returns:
            Tuple of (coordinates array [n x 2], voltages array [n], dc_islands array [n])

        Raises:
            PartitioningError: If required attributes are missing.
        """
        coordinates = []
        voltages = []
        dc_islands = []

        for node in nodes:
            node_data = graph.nodes[node]

            # Extract coordinates
            lat = node_data.get('lat')
            lon = node_data.get('lon')

            if lat is None or lon is None:
                raise PartitioningError(
                    f"Node {node} missing latitude or longitude",
                    strategy=self._get_strategy_name(),
                    graph_info={'nodes': len(nodes)}
                )

            coordinates.append([lat, lon])

            # Extract voltage
            voltage = node_data.get(self.voltage_attr)
            voltages.append(voltage)

            # Extract DC island
            dc_island = node_data.get(self.dc_island_attr)
            if dc_island is None:
                raise PartitioningError(
                    f"Node {node} missing '{self.dc_island_attr}' attribute. "
                    "Ensure the graph was loaded with VoltageAwareStrategy (va_loader).",
                    strategy=self._get_strategy_name(),
                    graph_info={'nodes': len(nodes)}
                )
            dc_islands.append(dc_island)

        return (np.array(coordinates),
                np.array(voltages, dtype=object),
                np.array(dc_islands))

    def _count_unique_groups(self, dc_islands: np.ndarray, voltages: np.ndarray) -> int:
        """
        Count unique (dc_island, voltage_level) combinations.

        Args:
            dc_islands: Array of DC island IDs
            voltages: Array of voltage values

        Returns:
            Number of unique groups
        """
        seen_groups = set()

        for i in range(len(dc_islands)):
            dc_island = dc_islands[i]
            voltage = voltages[i]

            # Create a hashable group key
            # Round voltage for tolerance-based matching
            if voltage is not None and isinstance(voltage, (int, float)):
                voltage_key = round(voltage / max(self.config.voltage_tolerance, 0.1)) * \
                              max(self.config.voltage_tolerance, 0.1)
            else:
                voltage_key = voltage

            group_key = (dc_island, voltage_key)
            seen_groups.add(group_key)

        return len(seen_groups)

    # =========================================================================
    # DISTANCE MATRIX METHODS
    # =========================================================================

    def _build_aware_distance_matrix(self, coordinates: np.ndarray,
                                     voltages: np.ndarray,
                                     dc_islands: np.ndarray) -> np.ndarray:
        """
        Build distance matrix with DC island and voltage awareness.

        For nodes in the same DC island AND same voltage level (within tolerance),
        uses geographical distance. Otherwise, assigns infinite distance.

        Constraint hierarchy:
        1. Different DC islands → infinite distance
        2. Same DC island, different voltage → infinite distance
        3. Same DC island, same voltage → geographical distance

        Args:
            coordinates: Array of [lat, lon] coordinates (n x 2)
            voltages: Array of voltage levels (n)
            dc_islands: Array of DC island IDs (n)

        Returns:
            Distance matrix (n x n) where:
                - d[i,j] = geographical_distance if same DC island AND same voltage
                - d[i,j] = infinite_distance otherwise
                - d[i,i] = 0 (diagonal)
        """
        n_nodes = len(coordinates)

        # Calculate geographical distances
        geo_distances = self._calculate_geographical_distances(coordinates)

        # Initialize distance matrix with infinite distances
        distance_matrix = np.full((n_nodes, n_nodes), self.config.infinite_distance)

        # Fill in distances for compatible pairs (same DC island AND same voltage)
        for i in range(n_nodes):
            distance_matrix[i, i] = 0.0  # Diagonal is always zero

            for j in range(i + 1, n_nodes):
                # Check DC island compatibility first (primary constraint)
                if not self._islands_compatible(dc_islands[i], dc_islands[j]):
                    continue  # Keep infinite distance

                # Check voltage compatibility (secondary constraint within same DC island)
                if not self._voltages_compatible(voltages[i], voltages[j]):
                    continue  # Keep infinite distance

                # Both constraints satisfied - use geographical distance
                distance_matrix[i, j] = geo_distances[i, j]
                distance_matrix[j, i] = geo_distances[i, j]  # Symmetric

        return distance_matrix

    def _calculate_geographical_distances(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Calculate geographical distance matrix.

        Args:
            coordinates: Array of [lat, lon] coordinates

        Returns:
            Distance matrix based on configured metric
        """
        if self.distance_metric == 'euclidean':
            return euclidean_distances(coordinates)

        elif self.distance_metric == 'haversine':
            # Convert to radians for haversine
            coords_rad = np.radians(coordinates)
            earth_radius_km = 6371
            return haversine_distances(coords_rad) * earth_radius_km

        else:
            raise PartitioningError(
                f"Unknown distance metric: {self.distance_metric}",
                strategy=self._get_strategy_name()
            )

    # =========================================================================
    # COMPATIBILITY CHECK METHODS
    # =========================================================================

    @staticmethod
    def _islands_compatible(island1: Any, island2: Any) -> bool:
        """
        Check if two DC island IDs are compatible (same island).

        Args:
            island1: First DC island ID
            island2: Second DC island ID

        Returns:
            True if islands are the same, False otherwise.
        """
        # Handle missing DC island IDs - isolated nodes
        if island1 is None or island2 is None:
            return False

        # Direct comparison (DC island IDs should be integers from component detection)
        return island1 == island2

    def _voltages_compatible(self, v1: Any, v2: Any) -> bool:
        """
        Check if two voltage levels are compatible (same voltage island).

        Args:
            v1: First voltage value
            v2: Second voltage value

        Returns:
            True if voltages are compatible (within tolerance), False otherwise.
        """
        # Handle missing voltages - nodes without voltage are isolated
        if v1 is None or v2 is None:
            return False

        # Handle non-numeric voltages (exact match required)
        if not isinstance(v1, (int, float)) or not isinstance(v2, (int, float)):
            return v1 == v2

        # Numeric comparison with tolerance
        return abs(float(v1) - float(v2)) <= self.config.voltage_tolerance

    # =========================================================================
    # LOGGING AND VALIDATION METHODS
    # =========================================================================

    @staticmethod
    def _log_group_summary(dc_islands: np.ndarray, voltages: np.ndarray) -> None:
        """
        Log summary of (dc_island, voltage_level) groups found in the network.

        Args:
            dc_islands: Array of DC island IDs
            voltages: Array of voltage values
        """
        # Count nodes per group
        group_counts: Dict[Tuple[Any, str], int] = {}

        for i in range(len(dc_islands)):
            dc_island = dc_islands[i]
            v = voltages[i]

            if v is None:
                voltage_str = 'Unknown'
            elif isinstance(v, (int, float)):
                voltage_str = f"{int(round(v))} kV"
            else:
                voltage_str = str(v)

            key = (dc_island, voltage_str)
            group_counts[key] = group_counts.get(key, 0) + 1

        n_groups = len(group_counts)
        n_islands = len(set(dc_islands))
        n_voltages = len(set(voltages))

        print(f"\nVoltage-aware partitioning with DC island support:")
        print(f"  {n_islands} DC island(s), {n_voltages} voltage level(s) → {n_groups} group(s)")

    @staticmethod
    def _create_partition_map(nodes: List[Any],
                              labels: np.ndarray) -> Dict[int, List[Any]]:
        """
        Create partition mapping from cluster labels.

        Args:
            nodes: List of node IDs
            labels: Array of cluster labels

        Returns:
            Dictionary mapping cluster_id -> list of node_ids
        """
        partition_map: Dict[int, List[Any]] = {}

        for i, label in enumerate(labels):
            cluster_id = int(label)
            if cluster_id not in partition_map:
                partition_map[cluster_id] = []
            partition_map[cluster_id].append(nodes[i])

        return partition_map

    def _validate_partition(self, partition_map: Dict[int, List[Any]],
                            n_nodes: int) -> None:
        """
        Validate that all nodes were assigned to clusters.

        Args:
            partition_map: Partition mapping
            n_nodes: Expected total number of nodes

        Raises:
            PartitioningError: If node count doesn't match.
        """
        total_assigned = sum(len(nodes) for nodes in partition_map.values())

        if total_assigned != n_nodes:
            raise PartitioningError(
                f"Partition assignment mismatch: {total_assigned} assigned vs {n_nodes} total nodes",
                strategy=self._get_strategy_name()
            )

    def _validate_cluster_consistency(self, graph: nx.Graph,
                                      partition_map: Dict[int, List[Any]]) -> None:
        """
        Validate that clusters don't mix incompatible DC islands or voltage levels.

        With infinite distances, clusters should never mix:
        1. Different DC islands
        2. Different voltage levels within the same DC island

        Args:
            graph: Original NetworkX graph
            partition_map: Resulting partition mapping
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
                    v = node_data.get('voltage', node_data.get('v_nom'))

                if v is not None and isinstance(v, (int, float)):
                    v_rounded = round(v / max(self.config.voltage_tolerance, 0.1)) * \
                                max(self.config.voltage_tolerance, 0.1)
                    voltages_in_cluster.add(v_rounded)
                elif v is not None:
                    voltages_in_cluster.add(v)

            # Check for DC island mixing
            if len(dc_islands_in_cluster) > 1:
                print(f"WARNING: Cluster {cluster_id} contains nodes from multiple DC islands: "
                      f"{dc_islands_in_cluster}. This should not happen with infinite distances.")

            # Check for voltage mixing
            if len(voltages_in_cluster) > 1:
                print(f"Warning: Cluster {cluster_id} contains multiple voltage levels: "
                      f"{voltages_in_cluster}.")
