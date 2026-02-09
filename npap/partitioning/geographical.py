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
    run_dbscan,
    run_hdbscan,
    run_hierarchical,
    run_kmeans,
    run_kmedoids,
    validate_partition,
    validate_required_attributes,
    with_runtime_config,
)


@dataclass
class GeographicalConfig:
    """
    Configuration parameters for geographical partitioning.

    Attributes
    ----------
    random_state : int
        Random seed for reproducibility in stochastic algorithms.
    max_iter : int
        Maximum iterations for iterative algorithms (K-means, K-medoids).
    n_init : int
        Number of initializations for K-means.
    hierarchical_linkage : str
        Linkage criterion for hierarchical clustering
        ('ward' for Euclidean, or 'complete'/'average'/'single').
    infinite_distance : float
        Value used to represent "infinite" distance between nodes in different
        AC islands. Using a large finite value instead of np.inf to avoid
        numerical issues in clustering algorithms.
    """

    random_state: int = 42
    max_iter: int = 300
    n_init: int = 10
    hierarchical_linkage: str = "ward"
    infinite_distance: float = 1e4


class GeographicalPartitioning(PartitioningStrategy):
    """
    Partition nodes based on geographical distance.

    This strategy clusters nodes based on their spatial coordinates using
    various clustering algorithms. It supports both Euclidean distance
    (for projected coordinates) and Haversine distance (for lat/lon coordinates).

    AC-Island Awareness:
        When the graph contains data with DC links (nodes have given
        'ac_island' attribute), the strategy automatically respects AC island
        boundaries by assigning infinite distance between nodes in different
        AC islands.

        Note: K-means and hierarchical with 'ward' linkage cannot support
        AC-island awareness because they use raw coordinates instead of
        precomputed distances. When AC islands are detected with these
        algorithms, a warning is issued recommending alternative algorithms.

    Supported algorithms:
        - 'kmeans': K-Means clustering (Euclidean only, fast, no AC-island support)
        - 'kmedoids': K-Medoids clustering (any metric, robust to outliers, AC-island aware)
        - 'dbscan': DBSCAN density-based clustering (automatic cluster count, AC-island aware)
        - 'hierarchical': Agglomerative hierarchical clustering (AC-island aware except ward)
        - 'hdbscan': HDBSCAN density-based clustering (automatic cluster count, AC-island aware)

    Configuration can be provided at:
    - Instantiation time (via `config` parameter in __init__)
    - Partition time (via `config` or individual parameters in partition())

    Partition-time parameters override instance defaults for that call only.
    """

    SUPPORTED_ALGORITHMS = ["kmeans", "kmedoids", "dbscan", "hierarchical", "hdbscan"]
    SUPPORTED_DISTANCE_METRICS = ["euclidean", "haversine"]

    # Algorithms that support AC-island awareness (use precomputed distances)
    AC_ISLAND_AWARE_ALGORITHMS = ["kmedoids", "dbscan", "hdbscan"]

    # Config parameter names for runtime override detection
    _CONFIG_PARAMS = {
        "random_state",
        "max_iter",
        "n_init",
        "hierarchical_linkage",
        "infinite_distance",
    }

    def __init__(
        self,
        algorithm: str = "kmeans",
        distance_metric: str = "euclidean",
        ac_island_attr: str = "ac_island",
        config: GeographicalConfig | None = None,
    ):
        """
        Initialize geographical partitioning strategy.

        Parameters
        ----------
        algorithm : str, default='kmeans'
            Clustering algorithm ('kmeans', 'kmedoids', 'dbscan',
            'hierarchical', 'hdbscan').
        distance_metric : str, default='euclidean'
            Distance metric ('haversine', 'euclidean').
        ac_island_attr : str, default='ac_island'
            Node attribute name containing AC island ID.
        config : GeographicalConfig, optional
            Configuration parameters for the algorithm.

        Raises
        ------
        ValueError
            If unsupported algorithm or distance metric specified.
        """
        self.algorithm = algorithm
        self.distance_metric = distance_metric
        self.ac_island_attr: str = ac_island_attr
        self.config = config or GeographicalConfig()

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

        log_debug(
            f"Initialized GeographicalPartitioning: algorithm={algorithm}, metric={distance_metric}",
            LogCategory.PARTITIONING,
        )

    @property
    def required_attributes(self) -> dict[str, list[str]]:
        """Required node attributes for geographical partitioning."""
        return {"nodes": ["lat", "lon"], "edges": []}

    def _get_strategy_name(self) -> str:
        """Get descriptive strategy name for error messages."""
        return f"geographical_{self.algorithm}"

    @with_runtime_config(GeographicalConfig, _CONFIG_PARAMS)
    @validate_required_attributes
    def partition(self, graph: nx.Graph, **kwargs) -> dict[int, list[Any]]:
        """
        Partition nodes based on geographical coordinates.

        Automatically detects and respects AC island boundaries when the graph
        contains voltage-aware data (nodes have 'ac_island' attribute). When
        AC islands are detected, infinite distance is assigned between nodes
        in different AC islands.

        Parameters
        ----------
        graph : nx.Graph
            NetworkX graph with lat, lon attributes on nodes.
        **kwargs : dict
            Additional parameters:

            - n_clusters : Number of clusters (required for kmeans, kmedoids, hierarchical)
            - eps : Epsilon (required for dbscan)
            - min_samples : Minimum samples (required for dbscan)
            - min_cluster_size : Minimum cluster size for HDBSCAN (default: 5)
            - config : GeographicalConfig instance to override instance config
            - random_state : Override config parameter
            - max_iter : Override config parameter
            - n_init : Override config parameter
            - hierarchical_linkage : Override config parameter
            - infinite_distance : Override config parameter

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
            n_clusters = kwargs.get("n_clusters")

            # Extract coordinates
            nodes = list(graph.nodes())
            coordinates = self._extract_coordinates(graph, nodes)

            # Auto-detect AC island data
            ac_islands = None
            has_ac_islands = self._has_ac_island_data(graph, nodes)

            if has_ac_islands:
                ac_islands = self._extract_ac_islands(graph, nodes)
                n_ac_islands = len(set(ac_islands))

                # Check if algorithm supports AC-island awareness
                if not self._supports_ac_island_awareness(effective_config):
                    log_warning(
                        f"AC islands detected ({n_ac_islands} islands) but '{self.algorithm}' "
                        f"algorithm does not support AC-island-aware partitioning. "
                        f"Consider using 'kmedoids', 'dbscan', or 'hierarchical' (with non-ward linkage) for AC-island awareness.",
                        LogCategory.PARTITIONING,
                    )
                    ac_islands = None  # Disable AC-island awareness for unsupported algorithms
                else:
                    log_info(
                        f"Starting AC-island-aware geographical partitioning: {self.algorithm}, "
                        f"n_clusters={n_clusters}, metric={self.distance_metric}, "
                        f"ac_islands={n_ac_islands}",
                        LogCategory.PARTITIONING,
                    )
            else:
                log_info(
                    f"Starting geographical partitioning: {self.algorithm}, "
                    f"n_clusters={n_clusters}, metric={self.distance_metric}",
                    LogCategory.PARTITIONING,
                )

            log_debug(
                f"Extracted coordinates for {len(nodes)} nodes",
                LogCategory.PARTITIONING,
            )

            # Perform clustering (pass ac_islands for AC-island-aware clustering)
            labels = self._run_clustering(
                coordinates, effective_config, ac_islands=ac_islands, **kwargs
            )

            # Create and validate partition
            partition_map = create_partition_map(nodes, labels)
            validate_partition(partition_map, len(nodes), self._get_strategy_name())

            # Validate AC island consistency if AC-island awareness was applied
            if ac_islands is not None:
                self._validate_cluster_ac_island_consistency(graph, partition_map)

            log_info(
                f"Geographical partitioning complete: {len(partition_map)} clusters",
                LogCategory.PARTITIONING,
            )

            return partition_map

        except Exception as e:
            if isinstance(e, PartitioningError):
                raise
            raise PartitioningError(
                f"Geographical partitioning failed: {e}",
                strategy=self._get_strategy_name(),
                graph_info={
                    "nodes": len(list(graph.nodes())),
                    "edges": len(graph.edges()),
                },
            ) from e

    def _supports_ac_island_awareness(self, config: GeographicalConfig) -> bool:
        """
        Check if current algorithm configuration supports AC-island awareness.

        AC-island awareness requires precomputed distance matrices. Algorithms
        that work directly on raw coordinates (kmeans, hierarchical with ward)
        cannot support this feature.

        Parameters
        ----------
        config : GeographicalConfig
            Current configuration.

        Returns
        -------
        bool
            True if AC-island awareness is supported, False otherwise.
        """
        if self.algorithm in self.AC_ISLAND_AWARE_ALGORITHMS:
            return True

        # Hierarchical supports AC-island awareness with non-ward linkage
        if self.algorithm == "hierarchical" and config.hierarchical_linkage != "ward":
            return True

        return False

    def _extract_coordinates(self, graph: nx.Graph, nodes: list[Any]) -> np.ndarray:
        """
        Extract coordinates from graph nodes.

        Parameters
        ----------
        graph : nx.Graph
            NetworkX graph with lat, lon attributes on nodes.
        nodes : list[Any]
            List of node IDs.

        Returns
        -------
        np.ndarray
            Array of coordinates (n x 2).

        Raises
        ------
        PartitioningError
            If any node is missing latitude or longitude.
        """
        coordinates = []

        for node in nodes:
            node_data = graph.nodes[node]
            lat = node_data.get("lat")
            lon = node_data.get("lon")

            if lat is None or lon is None:
                raise PartitioningError(
                    f"Node {node} missing latitude or longitude",
                    strategy=self._get_strategy_name(),
                    graph_info={"nodes": len(nodes), "edges": len(graph.edges())},
                )

            coordinates.append([lat, lon])

        return np.array(coordinates)

    def _run_clustering(
        self, coordinates: np.ndarray, config: GeographicalConfig, **kwargs
    ) -> np.ndarray:
        """
        Dispatch to appropriate clustering algorithm.

        Parameters
        ----------
        coordinates : np.ndarray
            Array of coordinates (n x 2).
        config : GeographicalConfig
            Configuration parameters.
        **kwargs : dict
            Additional clustering parameters.

        Returns
        -------
        np.ndarray
            Array of cluster labels.
        """
        if self.algorithm == "kmeans":
            return self._kmeans_clustering(coordinates, config, **kwargs)
        elif self.algorithm == "kmedoids":
            return self._kmedoids_clustering(coordinates, config, **kwargs)
        elif self.algorithm == "dbscan":
            return self._dbscan_clustering(coordinates, config, **kwargs)
        elif self.algorithm == "hierarchical":
            return self._hierarchical_clustering(coordinates, config, **kwargs)
        elif self.algorithm == "hdbscan":
            return self._hdbscan_clustering(coordinates, config, **kwargs)
        else:
            raise PartitioningError(
                f"Unknown algorithm: {self.algorithm}",
                strategy=self._get_strategy_name(),
            )

    def _kmeans_clustering(
        self, coordinates: np.ndarray, config: GeographicalConfig, **kwargs
    ) -> np.ndarray:
        """
        Perform K-means clustering on geographical coordinates.

        Parameters
        ----------
        coordinates : np.ndarray
            Array of coordinates (n x 2).
        config : GeographicalConfig
            Configuration parameters.
        **kwargs : dict
            Must include 'n_clusters'.

        Returns
        -------
        np.ndarray
            Array of cluster labels.

        Raises
        ------
        PartitioningError
            If distance metric is not euclidean or n_clusters is invalid.
        """
        # K-means requires Euclidean distance
        if self.distance_metric != "euclidean":
            raise PartitioningError(
                f"K-means does not support {self.distance_metric} distance. "
                "Use 'kmedoids' algorithm for non-Euclidean metrics.",
                strategy=self._get_strategy_name(),
            )

        n_clusters = kwargs.get("n_clusters")
        if n_clusters is None or n_clusters <= 0:
            raise PartitioningError(
                "K-means requires a positive 'n_clusters' parameter.",
                strategy=self._get_strategy_name(),
            )

        cartesian_coordinates = self._convert_spherical_to_cartesian(coordinates)

        log_debug(f"Running K-means with {n_clusters} clusters", LogCategory.PARTITIONING)
        return run_kmeans(
            cartesian_coordinates, n_clusters, config.random_state, config.max_iter, config.n_init
        )

    def _kmedoids_clustering(
        self, coordinates: np.ndarray, config: GeographicalConfig, **kwargs
    ) -> np.ndarray:
        """
        Perform K-medoids clustering on geographical coordinates.

        Parameters
        ----------
        coordinates : np.ndarray
            Array of coordinates (n x 2).
        config : GeographicalConfig
            Configuration parameters.
        **kwargs : dict
            Must include 'n_clusters'. May include 'ac_islands'.

        Returns
        -------
        np.ndarray
            Array of cluster labels.

        Raises
        ------
        PartitioningError
            If n_clusters is invalid.
        """
        n_clusters = kwargs.get("n_clusters")
        if n_clusters is None or n_clusters <= 0:
            raise PartitioningError(
                "K-medoids requires a positive 'n_clusters' parameter.",
                strategy=self._get_strategy_name(),
            )

        ac_islands = kwargs.get("ac_islands")

        if ac_islands is not None:
            log_debug(
                f"Running AC-island-aware K-medoids with {n_clusters} clusters, "
                f"metric={self.distance_metric}",
                LogCategory.PARTITIONING,
            )
            distance_matrix = self._build_ac_island_aware_distance_matrix(
                coordinates, ac_islands, config
            )
        else:
            log_debug(
                f"Running K-medoids with {n_clusters} clusters, metric={self.distance_metric}",
                LogCategory.PARTITIONING,
            )
            distance_matrix = compute_geographical_distances(coordinates, self.distance_metric)

        return run_kmedoids(distance_matrix, n_clusters)

    def _dbscan_clustering(
        self, coordinates: np.ndarray, config: GeographicalConfig, **kwargs
    ) -> np.ndarray:
        """
        Perform DBSCAN clustering on geographical coordinates.

        Parameters
        ----------
        coordinates : np.ndarray
            Array of coordinates (n x 2).
        config : GeographicalConfig
            Configuration parameters.
        **kwargs : dict
            Must include 'eps' and 'min_samples'. May include 'ac_islands'.

        Returns
        -------
        np.ndarray
            Array of cluster labels.

        Raises
        ------
        PartitioningError
            If eps or min_samples is missing.
        """
        eps = kwargs.get("eps")
        min_samples = kwargs.get("min_samples")

        if eps is None or min_samples is None:
            raise PartitioningError(
                "DBSCAN requires 'eps' and 'min_samples' parameters.",
                strategy=self._get_strategy_name(),
            )

        ac_islands = kwargs.get("ac_islands")

        if ac_islands is not None:
            log_debug(
                f"Running AC-island-aware DBSCAN with eps={eps}, min_samples={min_samples}",
                LogCategory.PARTITIONING,
            )
            distance_matrix = self._build_ac_island_aware_distance_matrix(
                coordinates, ac_islands, config
            )
        else:
            log_debug(
                f"Running DBSCAN with eps={eps}, min_samples={min_samples}",
                LogCategory.PARTITIONING,
            )
            distance_matrix = compute_geographical_distances(coordinates, self.distance_metric)

        return run_dbscan(distance_matrix, eps, min_samples)

    def _hierarchical_clustering(
        self, coordinates: np.ndarray, config: GeographicalConfig, **kwargs
    ) -> np.ndarray:
        """
        Perform Hierarchical Clustering on geographical coordinates.

        Parameters
        ----------
        coordinates : np.ndarray
            Array of coordinates (n x 2).
        config : GeographicalConfig
            Configuration parameters.
        **kwargs : dict
            Must include 'n_clusters'. May include 'ac_islands'.

        Returns
        -------
        np.ndarray
            Array of cluster labels.

        Raises
        ------
        PartitioningError
            If n_clusters is invalid or ward linkage used with non-euclidean.
        """
        n_clusters = kwargs.get("n_clusters")
        if n_clusters is None or n_clusters <= 0:
            raise PartitioningError(
                "Hierarchical clustering requires a positive 'n_clusters' parameter.",
                strategy=self._get_strategy_name(),
            )

        linkage = config.hierarchical_linkage
        ac_islands = kwargs.get("ac_islands")

        # Ward linkage only works with Euclidean distance on raw features
        if linkage == "ward":
            if self.distance_metric != "euclidean":
                raise PartitioningError(
                    "Ward linkage for Hierarchical Clustering requires Euclidean distance.",
                    strategy=self._get_strategy_name(),
                )

            log_debug(
                f"Running hierarchical clustering with {n_clusters} clusters, linkage={linkage}",
                LogCategory.PARTITIONING,
            )

            # Use sklearn's AgglomerativeClustering directly with ward linkage
            from sklearn.cluster import AgglomerativeClustering

            clustering = AgglomerativeClustering(
                n_clusters=n_clusters, metric="euclidean", linkage="ward"
            )
            return clustering.fit_predict(coordinates)
        else:
            # For other linkages, use precomputed distance matrix
            if ac_islands is not None:
                log_debug(
                    f"Running AC-island-aware hierarchical clustering with {n_clusters} clusters, "
                    f"linkage={linkage}",
                    LogCategory.PARTITIONING,
                )
                distance_matrix = self._build_ac_island_aware_distance_matrix(
                    coordinates, ac_islands, config
                )
            else:
                log_debug(
                    f"Running hierarchical clustering with {n_clusters} clusters, linkage={linkage}",
                    LogCategory.PARTITIONING,
                )
                distance_matrix = compute_geographical_distances(coordinates, self.distance_metric)

            return run_hierarchical(distance_matrix, n_clusters, linkage)

    def _hdbscan_clustering(
        self, coordinates: np.ndarray, config: GeographicalConfig, **kwargs
    ) -> np.ndarray:
        """
        Perform HDBSCAN clustering on geographical coordinates.

        Parameters
        ----------
        coordinates : np.ndarray
            Array of coordinates (n x 2).
        config : GeographicalConfig
            Configuration parameters.
        **kwargs : dict
            May include 'min_cluster_size' and 'ac_islands'.

        Returns
        -------
        np.ndarray
            Array of cluster labels.
        """
        min_cluster_size = kwargs.get("min_cluster_size", 5)
        ac_islands = kwargs.get("ac_islands")

        if ac_islands is not None:
            log_debug(
                f"Running AC-island-aware HDBSCAN with min_cluster_size={min_cluster_size}",
                LogCategory.PARTITIONING,
            )
            distance_matrix = self._build_ac_island_aware_distance_matrix(
                coordinates, ac_islands, config
            )
            # Use HDBSCAN with precomputed distances
            import hdbscan

            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="precomputed")
            return clusterer.fit_predict(distance_matrix)
        else:
            log_debug(
                f"Running HDBSCAN with min_cluster_size={min_cluster_size}, "
                f"metric={self.distance_metric}",
                LogCategory.PARTITIONING,
            )
            coords_rad = np.radians(coordinates)
            return run_hdbscan(coords_rad, min_cluster_size, self.distance_metric)

    # =========================================================================
    # AC-ISLAND AWARENESS METHODS
    # =========================================================================

    def _has_ac_island_data(self, graph: nx.Graph, nodes: list[Any]) -> bool:
        """
        Check if the graph has AC island data on nodes.

        AC island data is automatically detected by checking for the 'ac_island'
        attribute on nodes, which is set by the VoltageAwareStrategy (va_loader)
        when loading power system data with DC links.

        Parameters
        ----------
        graph : nx.Graph
            NetworkX graph.
        nodes : list[Any]
            List of node IDs.

        Returns
        -------
        bool
            True if all nodes have the ac_island attribute, False otherwise.
        """
        for node in nodes:
            if self.ac_island_attr not in graph.nodes[node]:
                return False
        return True

    def _extract_ac_islands(self, graph: nx.Graph, nodes: list[Any]) -> np.ndarray:
        """
        Extract AC island IDs from graph nodes.

        Parameters
        ----------
        graph : nx.Graph
            NetworkX graph.
        nodes : list[Any]
            List of node IDs.

        Returns
        -------
        np.ndarray
            Array of AC island IDs for each node.

        Raises
        ------
        PartitioningError
            If any node is missing the ac_island attribute.
        """
        ac_islands = []
        missing_nodes = []

        for node in nodes:
            ac_island = graph.nodes[node].get("ac_island")
            if ac_island is None:
                missing_nodes.append(node)
            ac_islands.append(ac_island)

        if missing_nodes:
            sample = missing_nodes[:5]
            raise PartitioningError(
                f"AC-island-aware partitioning requires 'ac_island' attribute "
                f"on all nodes. {len(missing_nodes)} node(s) are missing this attribute "
                f"(first few: {sample}). Use 'va_loader' to automatically detect AC islands.",
                strategy=self._get_strategy_name(),
            )

        return np.array(ac_islands)

    def _build_ac_island_aware_distance_matrix(
        self,
        coordinates: np.ndarray,
        ac_islands: np.ndarray,
        config: GeographicalConfig,
    ) -> np.ndarray:
        """
        Build distance matrix with AC island awareness.

        Nodes in the same AC island use geographical distance.
        Nodes in different AC islands get infinite distance.

        Parameters
        ----------
        coordinates : np.ndarray
            Array of [lat, lon] coordinates (n x 2).
        ac_islands : np.ndarray
            Array of AC island IDs (n).
        config : GeographicalConfig
            Configuration instance.

        Returns
        -------
        np.ndarray
            Distance matrix (n x n) where:

            - d[i,j] = geographical_distance if same AC island
            - d[i,j] = infinite_distance if different AC islands
            - d[i,i] = 0 (diagonal)
        """
        # Calculate geographical distances
        geo_distances = compute_geographical_distances(coordinates, self.distance_metric)

        # same_island[i,j] = True if ac_islands[i] == ac_islands[j]
        same_island_mask = ac_islands[:, np.newaxis] == ac_islands[np.newaxis, :]

        # Build distance matrix: geo_distance if same island, infinite otherwise
        distance_matrix = np.where(same_island_mask, geo_distances, config.infinite_distance)

        # Ensure diagonal is zero (nodes have zero distance to themselves)
        np.fill_diagonal(distance_matrix, 0.0)

        # Log statistics
        n_ac_islands = len(np.unique(ac_islands))
        # Count compatible pairs (same island, excluding diagonal)
        compatible_pairs = (np.sum(same_island_mask) - len(ac_islands)) // 2

        log_debug(
            f"Built AC-island-aware distance matrix: {n_ac_islands} AC island(s), "
            f"{compatible_pairs} compatible pairs",
            LogCategory.PARTITIONING,
        )

        return distance_matrix

    @staticmethod
    def _validate_cluster_ac_island_consistency(
        graph: nx.Graph, partition_map: dict[int, list[Any]]
    ) -> None:
        """
        Validate that clusters don't mix different AC islands.

        With infinite distances between AC islands, clusters should never
        contain nodes from multiple AC islands.

        Parameters
        ----------
        graph : nx.Graph
            Original NetworkX graph.
        partition_map : dict[int, list[Any]]
            Resulting partition mapping.
        """
        for cluster_id, nodes in partition_map.items():
            ac_islands_in_cluster = set()

            for node in nodes:
                ac_island = graph.nodes[node].get("ac_island")
                if ac_island is not None:
                    ac_islands_in_cluster.add(ac_island)

            if len(ac_islands_in_cluster) > 1:
                log_warning(
                    f"Cluster {cluster_id} contains nodes from multiple AC islands: "
                    f"{ac_islands_in_cluster}. This should not happen with infinite distances.",
                    LogCategory.PARTITIONING,
                    warn_user=False,
                )

    @staticmethod
    def _convert_spherical_to_cartesian(coordinates: np.ndarray) -> np.ndarray:
        """
        Convert spherical coordinates (lat, lon) to Cartesian (x, y, z).

        This is used for K-means clustering which operates in Euclidean space.

        Args:
            coordinates: Array of [lat, lon] in degrees (n x 2)

        Returns
        -------
            Array of [x, y, z] Cartesian coordinates (n x 3)
        """
        coords = np.asarray(coordinates)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise PartitioningError(
                "Coordinates must be a (n, 2) array of [lat, lon] pairs.",
                strategy="geographical_conversion",
            )

        lat_rad = np.radians(coords[:, 0])
        lon_rad = np.radians(coords[:, 1])

        cos_lat = np.cos(lat_rad)
        x = cos_lat * np.cos(lon_rad)
        y = cos_lat * np.sin(lon_rad)
        z = np.sin(lat_rad)

        return np.column_stack((x, y, z))
