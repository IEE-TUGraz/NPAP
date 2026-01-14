from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from npap.exceptions import PartitioningError
from npap.interfaces import PartitioningStrategy
from npap.logging import LogCategory, log_debug, log_info
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
        random_state: Random seed for reproducibility in stochastic algorithms
        max_iter: Maximum iterations for iterative algorithms (K-means, K-medoids)
        n_init: Number of initializations for K-means
        hierarchical_linkage: Linkage criterion for hierarchical clustering
                             ('ward' for Euclidean, or 'complete'/'average'/'single')
    """

    random_state: int = 42
    max_iter: int = 300
    n_init: int = 10
    hierarchical_linkage: str = "ward"


class GeographicalPartitioning(PartitioningStrategy):
    """
    Partition nodes based on geographical distance.

    This strategy clusters nodes based on their spatial coordinates using
    various clustering algorithms. It supports both Euclidean distance
    (for projected coordinates) and Haversine distance (for lat/lon coordinates).

    Supported algorithms:
        - 'kmeans': K-Means clustering (Euclidean only, fast)
        - 'kmedoids': K-Medoids clustering (any metric, robust to outliers)
        - 'dbscan': DBSCAN density-based clustering (automatic cluster count)
        - 'hierarchical': Agglomerative hierarchical clustering
        - 'hdbscan': HDBSCAN density-based clustering (automatic cluster count)

    Configuration can be provided at:
    - Instantiation time (via `config` parameter in __init__)
    - Partition time (via `config` or individual parameters in partition())

    Partition-time parameters override instance defaults for that call only.
    """

    SUPPORTED_ALGORITHMS = ["kmeans", "kmedoids", "dbscan", "hierarchical", "hdbscan"]
    SUPPORTED_DISTANCE_METRICS = ["euclidean", "haversine"]

    # Config parameter names for runtime override detection
    _CONFIG_PARAMS = {"random_state", "max_iter", "n_init", "hierarchical_linkage"}

    def __init__(
        self,
        algorithm: str = "kmeans",
        distance_metric: str = "euclidean",
        config: GeographicalConfig | None = None,
    ):
        """
        Initialize geographical partitioning strategy.

        Args:
            algorithm: Clustering algorithm ('kmeans', 'kmedoids', 'dbscan',
                       'hierarchical', 'hdbscan')
            distance_metric: Distance metric ('haversine', 'euclidean')
            config: Configuration parameters for the algorithm

        Raises
        ------
            ValueError: If unsupported algorithm or distance metric specified
        """
        self.algorithm = algorithm
        self.distance_metric = distance_metric
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

        Args:
            graph: NetworkX graph with lat, lon attributes on nodes
            **kwargs: Additional parameters:
                - n_clusters: Number of clusters (required for kmeans, kmedoids, hierarchical)
                - eps: Epsilon (required for dbscan)
                - min_samples: Minimum samples (required for dbscan)
                - min_cluster_size: Minimum cluster size for HDBSCAN (default: 5)
                - config: GeographicalConfig instance to override instance config
                - random_state: Override config parameter
                - max_iter: Override config parameter
                - n_init: Override config parameter
                - hierarchical_linkage: Override config parameter

        Returns
        -------
            Dictionary mapping cluster_id -> list of node_ids

        Raises
        ------
            PartitioningError: If partitioning fails
        """
        try:
            # Get effective config (injected by decorator)
            effective_config = kwargs.get("_effective_config", self.config)
            n_clusters = kwargs.get("n_clusters")

            log_info(
                f"Starting geographical partitioning: {self.algorithm}, "
                f"n_clusters={n_clusters}, metric={self.distance_metric}",
                LogCategory.PARTITIONING,
            )

            # Extract coordinates
            nodes = list(graph.nodes())
            coordinates = self._extract_coordinates(graph, nodes)

            log_debug(
                f"Extracted coordinates for {len(nodes)} nodes",
                LogCategory.PARTITIONING,
            )

            # Perform clustering
            labels = self._run_clustering(coordinates, effective_config, **kwargs)

            # Create and validate partition
            partition_map = create_partition_map(nodes, labels)
            validate_partition(partition_map, len(nodes), self._get_strategy_name())

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

    def _extract_coordinates(self, graph: nx.Graph, nodes: list[Any]) -> np.ndarray:
        """Extract coordinates from graph nodes."""
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
        """Dispatch to appropriate clustering algorithm."""
        if self.algorithm == "kmeans":
            return self._kmeans_clustering(coordinates, config, **kwargs)
        elif self.algorithm == "kmedoids":
            return self._kmedoids_clustering(coordinates, **kwargs)
        elif self.algorithm == "dbscan":
            return self._dbscan_clustering(coordinates, **kwargs)
        elif self.algorithm == "hierarchical":
            return self._hierarchical_clustering(coordinates, config, **kwargs)
        elif self.algorithm == "hdbscan":
            return self._hdbscan_clustering(coordinates, **kwargs)
        else:
            raise PartitioningError(
                f"Unknown algorithm: {self.algorithm}",
                strategy=self._get_strategy_name(),
            )

    def _kmeans_clustering(
        self, coordinates: np.ndarray, config: GeographicalConfig, **kwargs
    ) -> np.ndarray:
        """Perform K-means clustering on geographical coordinates."""
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

        log_debug(f"Running K-means with {n_clusters} clusters", LogCategory.PARTITIONING)
        return run_kmeans(
            coordinates, n_clusters, config.random_state, config.max_iter, config.n_init
        )

    def _kmedoids_clustering(self, coordinates: np.ndarray, **kwargs) -> np.ndarray:
        """Perform K-medoids clustering on geographical coordinates."""
        n_clusters = kwargs.get("n_clusters")
        if n_clusters is None or n_clusters <= 0:
            raise PartitioningError(
                "K-medoids requires a positive 'n_clusters' parameter.",
                strategy=self._get_strategy_name(),
            )

        log_debug(
            f"Running K-medoids with {n_clusters} clusters, metric={self.distance_metric}",
            LogCategory.PARTITIONING,
        )
        distance_matrix = compute_geographical_distances(coordinates, self.distance_metric)

        return run_kmedoids(distance_matrix, n_clusters)

    def _dbscan_clustering(self, coordinates: np.ndarray, **kwargs) -> np.ndarray:
        """Perform DBSCAN clustering on geographical coordinates."""
        eps = kwargs.get("eps")
        min_samples = kwargs.get("min_samples")

        if eps is None or min_samples is None:
            raise PartitioningError(
                "DBSCAN requires 'eps' and 'min_samples' parameters.",
                strategy=self._get_strategy_name(),
            )

        log_debug(
            f"Running DBSCAN with eps={eps}, min_samples={min_samples}",
            LogCategory.PARTITIONING,
        )
        distance_matrix = compute_geographical_distances(coordinates, self.distance_metric)

        return run_dbscan(distance_matrix, eps, min_samples)

    def _hierarchical_clustering(
        self, coordinates: np.ndarray, config: GeographicalConfig, **kwargs
    ) -> np.ndarray:
        """Perform Hierarchical Clustering on geographical coordinates."""
        n_clusters = kwargs.get("n_clusters")
        if n_clusters is None or n_clusters <= 0:
            raise PartitioningError(
                "Hierarchical clustering requires a positive 'n_clusters' parameter.",
                strategy=self._get_strategy_name(),
            )

        linkage = config.hierarchical_linkage

        log_debug(
            f"Running hierarchical clustering with {n_clusters} clusters, linkage={linkage}",
            LogCategory.PARTITIONING,
        )

        # Ward linkage only works with Euclidean distance on raw features
        if linkage == "ward":
            if self.distance_metric != "euclidean":
                raise PartitioningError(
                    "Ward linkage for Hierarchical Clustering requires Euclidean distance.",
                    strategy=self._get_strategy_name(),
                )
            # Use sklearn's AgglomerativeClustering directly with ward linkage
            from sklearn.cluster import AgglomerativeClustering

            clustering = AgglomerativeClustering(
                n_clusters=n_clusters, metric="euclidean", linkage="ward"
            )
            return clustering.fit_predict(coordinates)
        else:
            # For other linkages, use precomputed distance matrix
            distance_matrix = compute_geographical_distances(coordinates, self.distance_metric)
            return run_hierarchical(distance_matrix, n_clusters, linkage)

    def _hdbscan_clustering(self, coordinates: np.ndarray, **kwargs) -> np.ndarray:
        """Perform HDBSCAN clustering on geographical coordinates."""
        min_cluster_size = kwargs.get("min_cluster_size", 5)

        log_debug(
            f"Running HDBSCAN with min_cluster_size={min_cluster_size}, metric={self.distance_metric}",
            LogCategory.PARTITIONING,
        )
        coords_rad = np.radians(coordinates)

        return run_hdbscan(coords_rad, min_cluster_size, self.distance_metric)
