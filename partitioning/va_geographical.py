from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import networkx as nx
import numpy as np
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
                          between nodes at different voltage levels.
                          Using a large finite value instead of np.inf
                          to avoid numerical issues in clustering algorithms.
        proportional_clustering: If False (default), runs clustering on full
                              matrix with infinite distances between voltage levels.
                              If True, clusters each voltage island independently
                              with proportional cluster distribution.
    """
    voltage_tolerance: float = 1.0
    infinite_distance: float = 1e4
    proportional_clustering: bool = False


class VAGeographicalPartitioning(PartitioningStrategy):
    """
    Voltage-Aware Geographical Partitioning Strategy.

    This strategy partitions nodes based on geographical distance while
    respecting voltage level boundaries. Nodes at different voltage levels
    are assigned infinite distance, ensuring they won't be clustered together.

    This is particularly useful for power networks where:
    - Buses operate at specific voltage levels (e.g., 110kV, 220kV, 380kV)
    - Aggregation should only occur within the same voltage level
    - Transformers connect different voltage levels but shouldn't cause
      those levels to merge during clustering

    Two clustering modes are available:

    Standard mode (per_island_clustering=False, default):
        - Builds full NxN distance matrix
        - Sets d(i,j) = inf if voltage(i) != voltage(j)
        - Runs single clustering algorithm on entire matrix
        - Algorithm handles infinite distances to respect voltage boundaries

    Proportional mode (per_island_clustering=True):
        - Groups nodes by voltage level
        - Distributes n_clusters proportionally among voltage groups
        - Runs clustering independently on each group
        - Guaranteed balanced distribution per voltage level

    Supported algorithms:
        - 'kmedoids': K-Medoids clustering (works naturally with precomputed distances)
    Example distance matrix for N=4 nodes (2 at 220kV, 2 at 380kV):
        ( 0.0,   X,   inf, inf )
        ( X,   0.0,   inf, inf )
        ( inf, inf, 0.0,   Y   )
        ( inf, inf, Y,   0.0   )

    Where X and Y are actual geographical distances within voltage islands.
    """

    SUPPORTED_ALGORITHMS = ['kmedoids']
    SUPPORTED_DISTANCE_METRICS = ['euclidean', 'haversine']

    def __init__(self, algorithm: str = 'kmedoids',
                 distance_metric: str = 'euclidean',
                 voltage_attr: str = 'voltage',
                 config: Optional[VAGeographicalConfig] = None):
        """
        Initialize voltage-aware geographical partitioning strategy.

        Args:
            algorithm: Clustering algorithm. Supported options:
                      - 'kmedoids': K-Medoids (robust, works with precomputed distances)
                      distance matrices.
            distance_metric: Distance metric for geographical distances.
                           'euclidean' for flat/projected coordinates,
                           'haversine' for great-circle distance on Earth.
            voltage_attr: Node attribute name containing voltage level.
            config: Configuration parameters for the algorithm.

        Raises:
            ValueError: If unsupported algorithm or distance metric is specified.
        """
        self.algorithm = algorithm
        self.distance_metric = distance_metric
        self.voltage_attr = voltage_attr
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

        if distance_metric not in ['euclidean', 'haversine']:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")

    @property
    def required_attributes(self) -> Dict[str, List[str]]:
        """Required node attributes for voltage-aware geographical partitioning."""
        return {
            'nodes': ['lat', 'lon', self.voltage_attr],
            'edges': []
        }

    def partition(self, graph: nx.Graph, **kwargs) -> Dict[int, List[Any]]:
        """
        Partition nodes based on voltage-aware geographical distance.

        Args:
            graph: NetworkX graph with lat, lon, and voltage attributes on nodes
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

            # Extract coordinates and voltages
            coordinates, voltages = self._extract_node_data(graph, nodes)

            # Get voltage level summary for validation
            voltage_levels = self._get_unique_voltage_levels(voltages)
            n_voltage_levels = len(voltage_levels)

            # Warn if n_clusters < voltage levels (some levels might be forced together)
            if n_clusters < n_voltage_levels:
                print(f"Warning: Requested {n_clusters} clusters but found {n_voltage_levels} "
                      f"distinct voltage levels. Some voltage levels may share clusters, "
                      f"but infinite distance constraints will be respected.")

            # Log voltage level summary
            self._log_voltage_summary(voltages)

            # Choose clustering mode based on configuration
            if self.config.proportional_clustering:
                partition_map = self._proportional_partition(
                    nodes, coordinates, voltages, **kwargs
                )
            else:
                partition_map = self._standard_partition(
                    nodes, coordinates, voltages, **kwargs
                )

            # Validate result
            self._validate_partition(partition_map, n_nodes)

            # Validate voltage consistency in clusters
            self._validate_voltage_consistency(graph, partition_map)

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

    # =========================================================================
    # STANDARD MODE: Single clustering with infinite distances
    # =========================================================================

    def _standard_partition(self, nodes: List[Any], coordinates: np.ndarray,
                            voltages: np.ndarray, **kwargs) -> Dict[int, List[Any]]:
        """
        Partition using single clustering on full matrix with infinite distances.

        This mode builds a voltage-aware distance matrix where nodes at different
        voltage levels have infinite distance, then runs the clustering algorithm
        on the entire matrix.

        Args:
            nodes: List of node IDs
            coordinates: Node coordinates [n x 2]
            voltages: Node voltage values
            **kwargs: Clustering parameters

        Returns:
            Partition mapping
        """
        # Build voltage-aware distance matrix
        distance_matrix = self._build_voltage_aware_distance_matrix(coordinates, voltages)

        # Run clustering algorithm
        labels = self._run_clustering(distance_matrix, **kwargs)

        # Create partition mapping
        return self._create_partition_map(nodes, labels)

    def _proportional_partition(self, nodes: List[Any], coordinates: np.ndarray,
                                voltages: np.ndarray, **kwargs) -> Dict[int, List[Any]]:
        """
        Partition each voltage island independently with proportional distribution.

        Args:
            nodes: List of node IDs
            coordinates: Node coordinates [n x 2]
            voltages: Node voltage values
            **kwargs: Clustering parameters

        Returns:
            Partition mapping
        """
        n_clusters = kwargs.get('n_clusters')

        # Group nodes by voltage
        voltage_groups = self._group_by_voltage(voltages)

        # Allocate clusters proportionally
        allocation = self._allocate_clusters(voltage_groups, n_clusters)

        # Partition each group
        partition_map: Dict[int, List[Any]] = {}
        cluster_offset = 0

        for voltage, node_indices in voltage_groups.items():
            n_clust = allocation[voltage]
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

    def _group_by_voltage(self, voltages: np.ndarray) -> Dict[Any, List[int]]:
        """
        Group node indices by voltage level.

        Args:
            voltages: Array of voltage values

        Returns:
            Dict mapping voltage_level -> list of node indices
        """
        groups: Dict[Any, List[int]] = {}

        for idx, voltage in enumerate(voltages):
            matched = None
            for existing in groups.keys():
                if self._voltages_compatible(voltage, existing):
                    matched = existing
                    break

            if matched is not None:
                groups[matched].append(idx)
            else:
                key = voltage if voltage is not None else 'unknown'
                groups[key] = [idx]

        return groups

    @staticmethod
    def _allocate_clusters(voltage_groups: Dict[Any, List[int]],
                           n_clusters: int) -> Dict[Any, int]:
        """
        Allocate clusters proportionally to voltage groups.

        Args:
            voltage_groups: Dict mapping voltage -> node indices
            n_clusters: Total clusters to allocate

        Returns:
            Dict mapping voltage -> number of clusters
        """
        total_nodes = sum(len(indices) for indices in voltage_groups.values())
        allocation: Dict[Any, int] = {}

        # Sort by size (largest first) for stable allocation
        sorted_voltages = sorted(voltage_groups.keys(),
                                 key=lambda v: len(voltage_groups[v]),
                                 reverse=True)

        remaining = n_clusters
        for i, voltage in enumerate(sorted_voltages):
            n_nodes = len(voltage_groups[voltage])

            if i == len(sorted_voltages) - 1:
                # Last group gets remaining clusters
                allocation[voltage] = max(1, remaining)
            else:
                # Proportional allocation
                proportion = n_nodes / total_nodes
                allocated = max(1, int(round(n_clusters * proportion)))
                allocated = min(allocated, n_nodes, remaining - (len(sorted_voltages) - i - 1))
                allocation[voltage] = allocated
                remaining -= allocated

        # Log allocation
        print(f"\nCluster allocation (proportional):")
        for voltage in sorted_voltages:
            n_nodes = len(voltage_groups[voltage])
            n_clust = allocation[voltage]
            ratio = n_nodes / n_clust if n_clust > 0 else 0
            voltage_str = f"{int(voltage)} kV" if isinstance(voltage, (int, float)) else str(voltage)
            print(f"  • {voltage_str}: {n_clust} cluster(s) for {n_nodes} nodes (~{ratio:.1f} nodes/cluster)")

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


    # =========================================================================
    # SHARED UTILITIES
    # =========================================================================

    def _extract_node_data(self, graph: nx.Graph,
                           nodes: List[Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract coordinates and voltage levels from nodes.

        Args:
            graph: NetworkX graph
            nodes: List of node IDs

        Returns:
            Tuple of (coordinates array [n x 2], voltages array [n])

        Raises:
            PartitioningError: If required coordinate attributes are missing.
        """
        coordinates = []
        voltages = []

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

            # Extract voltages
            voltage = node_data.get(self.voltage_attr)

            voltages.append(voltage)

        return np.array(coordinates), np.array(voltages, dtype=object)

    def _get_unique_voltage_levels(self, voltages: np.ndarray) -> List[Any]:
        """
        Get unique voltage levels considering tolerance.

        Args:
            voltages: Array of voltage values

        Returns:
            List of unique voltage levels
        """
        seen_voltages = []

        for v in voltages:
            if v is None:
                if None not in seen_voltages:
                    seen_voltages.append(None)
                continue

            # Check if this voltage is compatible with any seen voltage
            is_new = True
            for seen in seen_voltages:
                if seen is not None and self._voltages_compatible(v, seen):
                    is_new = False
                    break

            if is_new:
                seen_voltages.append(v)

        return seen_voltages

    def _build_voltage_aware_distance_matrix(self, coordinates: np.ndarray,
                                             voltages: np.ndarray) -> np.ndarray:
        """
        Build distance matrix with voltage-awareness.

        For nodes with the same voltage level (within tolerance), uses
        geographical distance. For nodes with different voltage levels,
        assigns infinite distance.

        Args:
            coordinates: Array of [lat, lon] coordinates (n x 2)
            voltages: Array of voltage levels (n)

        Returns:
            Distance matrix (n x n) where:
                - d[i,j] = geographical_distance if voltages compatible
                - d[i,j] = infinite_distance if voltages incompatible
                - d[i,i] = 0 (diagonal)
        """
        n_nodes = len(coordinates)

        # Calculate geographical distances
        geo_distances = self._calculate_geographical_distances(coordinates)

        # Initialize distance matrix with infinite distances
        distance_matrix = np.full((n_nodes, n_nodes), self.config.infinite_distance)

        # Fill in distances for compatible voltage pairs
        for i in range(n_nodes):
            distance_matrix[i, i] = 0.0  # Diagonal is always zero

            for j in range(i + 1, n_nodes):
                if self._voltages_compatible(voltages[i], voltages[j]):
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

    @staticmethod
    def _log_voltage_summary(voltages: np.ndarray) -> None:
        """
        Log summary of voltage levels found in the network.

        Args:
            voltages: Array of voltage values
        """
        voltage_counts: Dict[Any, int] = {}

        for v in voltages:
            if v is None:
                key = 'Unknown'
            elif isinstance(v, (int, float)):
                # Round to nearest integer for display
                key = f"{int(round(v))} kV"
            else:
                key = str(v)

            voltage_counts[key] = voltage_counts.get(key, 0) + 1

        print(f"Voltage-aware partitioning - {len(voltage_counts)} voltage level(s) detected:")
        for voltage, count in sorted(voltage_counts.items(),
                                     key=lambda x: (x[0] == 'Unknown', x[0])):
            print(f"  • {voltage}: {count} node(s)")

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

    def _validate_voltage_consistency(self, graph: nx.Graph,
                                      partition_map: Dict[int, List[Any]]) -> None:
        """
        Validate that clusters don't mix incompatible voltage levels.
        With infinite distances, clusters should never mix voltage levels.

        Args:
            graph: Original NetworkX graph
            partition_map: Resulting partition mapping
        """
        for cluster_id, nodes in partition_map.items():
            voltages_in_cluster = set()

            for node in nodes:
                v = graph.nodes[node].get(self.voltage_attr)
                if v is None:
                    v = graph.nodes[node].get('voltage', graph.nodes[node].get('v_nom'))

                if v is not None and isinstance(v, (int, float)):
                    # Round to tolerance for grouping
                    v_rounded = round(v / max(self.config.voltage_tolerance, 0.1)) * max(
                        self.config.voltage_tolerance, 0.1)
                    voltages_in_cluster.add(v_rounded)
                elif v is not None:
                    voltages_in_cluster.add(v)

            if len(voltages_in_cluster) > 1:
                print(f"Warning: Cluster {cluster_id} contains multiple voltage levels: "
                      f"{voltages_in_cluster}.")
