import functools
import hashlib
import json
from collections import defaultdict
from dataclasses import replace
from typing import Any, TypeVar

import networkx as nx
import numpy as np

# Type variable for config classes
ConfigT = TypeVar("ConfigT")


# =============================================================================
# GRAPH VALIDATION UTILITIES
# =============================================================================


def validate_required_attributes(func):
    """
    Decorator for attribute validation.

    - Early exit on first missing attribute
    - Minimal memory overhead
    - Fast attribute checking
    """

    @functools.wraps(func)
    def wrapper(self, graph: nx.Graph, **kwargs):
        required = self.required_attributes

        # Fast early-exit validation for nodes
        if required.get("nodes"):
            required_node_attrs = set(required["nodes"])
            for node_id in graph.nodes():
                node_attrs = graph.nodes[node_id]
                missing_attrs = required_node_attrs - node_attrs.keys()
                if missing_attrs:
                    from npap.exceptions import ValidationError

                    raise ValidationError(
                        f"Node {node_id} missing required attributes",
                        missing_attributes={"nodes": list(missing_attrs)},
                        strategy=self.__class__.__name__,
                    )

        # Fast early-exit validation for edges
        if required.get("edges"):
            required_edge_attrs = set(required["edges"])
            for edge in graph.edges():
                edge_attrs = graph.edges[edge]
                missing_attrs = required_edge_attrs - edge_attrs.keys()
                if missing_attrs:
                    from npap.exceptions import ValidationError

                    raise ValidationError(
                        f"Edge {edge} missing required attributes",
                        missing_attributes={"edges": list(missing_attrs)},
                        strategy=self.__class__.__name__,
                    )

        return func(self, graph, **kwargs)

    return wrapper


def compute_graph_hash(graph: nx.Graph) -> str:
    """
    Compute a hash of the graph structure for validation.

    Creates a fingerprint based on node count, edge count, and node IDs.
    Used to validate that a partition matches its source graph.

    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph.

    Returns
    -------
    str
        Hash string (16 characters).
    """
    graph_data = {
        "n_nodes": len(list(graph.nodes())),
        "n_edges": len(graph.edges()),
        "nodes": sorted([str(n) for n in graph.nodes()]),
    }
    json_str = json.dumps(graph_data, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def validate_graph_compatibility(partition_result, current_graph_hash: str):
    """
    Validate that a partition result is compatible with the current graph.

    Parameters
    ----------
    partition_result : PartitionResult
        PartitionResult object.
    current_graph_hash : str
        Hash of the current graph.

    Raises
    ------
    GraphCompatibilityError
        If hashes don't match.
    """
    if partition_result.original_graph_hash != current_graph_hash:
        from npap.exceptions import GraphCompatibilityError

        raise GraphCompatibilityError(
            "Partition was created from a different graph. "
            "Graph structure has changed since partition was created.",
            expected_hash=partition_result.original_graph_hash,
            actual_hash=current_graph_hash,
        )


# =============================================================================
# CONFIGURATION RESOLUTION
# =============================================================================


def resolve_runtime_config(
    instance_config: ConfigT,
    config_class: type[ConfigT],
    config_params: set[str],
    strategy_name: str,
    **kwargs,
) -> ConfigT:
    """
    Resolve effective configuration for a partition call.

    This utility implements a priority-based configuration resolution:

    1. Individual config parameters in kwargs (highest priority)
    2. Full config object in kwargs
    3. Instance config (lowest priority)

    Parameters
    ----------
    instance_config : ConfigT
        The strategy's default config instance.
    config_class : type[ConfigT]
        The dataclass type for configuration.
    config_params : set[str]
        Set of parameter names that belong to config.
    strategy_name : str
        Strategy name for error messages.
    **kwargs : dict
        Parameters passed to partition().

    Returns
    -------
    ConfigT
        Resolved configuration instance.

    Raises
    ------
    PartitioningError
        If config type is invalid.

    Examples
    --------
    >>> effective_config = resolve_runtime_config(
    ...     instance_config=self.config,
    ...     config_class=ElectricalDistanceConfig,
    ...     config_params={'zero_reactance_replacement', 'use_sparse'},
    ...     strategy_name=self._get_strategy_name(),
    ...     **kwargs
    ... )
    """
    from npap.exceptions import PartitioningError

    effective_config = instance_config

    # Check for full config override
    if "config" in kwargs:
        config_override = kwargs["config"]
        if not isinstance(config_override, config_class):
            raise PartitioningError(
                f"'config' parameter must be {config_class.__name__}, "
                f"got {type(config_override).__name__}",
                strategy=strategy_name,
            )
        effective_config = config_override

    # Check for individual parameter overrides
    override_params = {k: v for k, v in kwargs.items() if k in config_params}

    # Apply individual overrides using dataclass replace
    if override_params:
        effective_config = replace(effective_config, **override_params)

    return effective_config


def with_runtime_config(config_class: type[ConfigT], config_params: set[str]):
    """
    Decorator factory that adds runtime config resolution to partition methods.

    The decorator resolves configuration from kwargs and injects it as
    '_effective_config' keyword argument before calling the original method.

    Parameters
    ----------
    config_class : type[ConfigT]
        The dataclass type for configuration.
    config_params : set[str]
        Set of parameter names that belong to config.

    Returns
    -------
    Callable
        Decorator function.

    Examples
    --------
    >>> @with_runtime_config(VAGeographicalConfig, {'voltage_tolerance', 'infinite_distance'})
    ... def partition(self, graph, **kwargs):
    ...     config = kwargs.get('_effective_config')
    ...     # ... rest of implementation
    """

    def decorator(partition_method):
        @functools.wraps(partition_method)
        def wrapper(self, graph: nx.Graph, **kwargs):
            # Determine strategy name for error messages
            if hasattr(self, "_get_strategy_name"):
                strategy_name = self._get_strategy_name()
            else:
                strategy_name = self.__class__.__name__

            # Resolve configuration
            effective_config = resolve_runtime_config(
                instance_config=self.config,
                config_class=config_class,
                config_params=config_params,
                strategy_name=strategy_name,
                **kwargs,
            )

            # Inject effective config into kwargs
            kwargs["_effective_config"] = effective_config

            return partition_method(self, graph, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# PARTITION MAP UTILITIES
# =============================================================================


def create_partition_map(nodes: list[Any], labels: np.ndarray) -> dict[int, list[Any]]:
    """
    Create partition mapping from cluster labels.

    Parameters
    ----------
    nodes : list[Any]
        List of node IDs (must match order of labels).
    labels : np.ndarray
        Array of cluster labels from clustering algorithm.

    Returns
    -------
    dict[int, list[Any]]
        Dictionary mapping cluster_id -> list of node_ids.
    """
    partition_map: dict[int, list[Any]] = defaultdict(list)

    for i, label in enumerate(labels):
        partition_map[int(label)].append(nodes[i])

    return dict(partition_map)


def validate_partition(
    partition_map: dict[int, list[Any]], n_nodes: int, strategy_name: str
) -> None:
    """
    Validate that all nodes were assigned to clusters.

    Parameters
    ----------
    partition_map : dict[int, list[Any]]
        Partition mapping to validate.
    n_nodes : int
        Expected total number of nodes.
    strategy_name : str
        Strategy name for error messages.

    Raises
    ------
    PartitioningError
        If node count doesn't match.
    """
    from npap.exceptions import PartitioningError

    total_assigned = sum(len(nodes) for nodes in partition_map.values())

    if total_assigned != n_nodes:
        raise PartitioningError(
            f"Partition assignment mismatch: {total_assigned} assigned vs {n_nodes} total",
            strategy=strategy_name,
        )


# =============================================================================
# CLUSTERING ALGORITHMS
# =============================================================================


def run_kmeans(
    features: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    max_iter: int = 300,
    n_init: int = 10,
) -> np.ndarray:
    """
    Perform K-Means clustering on feature matrix.

    K-Means uses Euclidean distance internally. For distance matrix input,
    rows are treated as feature vectors representing distance profiles.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix (n_samples x n_features) or distance matrix.
    n_clusters : int
        Number of clusters to form.
    random_state : int, default=42
        Random seed for reproducibility.
    max_iter : int, default=300
        Maximum iterations for convergence.
    n_init : int, default=10
        Number of initializations to run.

    Returns
    -------
    np.ndarray
        Array of cluster labels for each sample.

    Raises
    ------
    PartitioningError
        If clustering fails.
    """
    from sklearn.cluster import KMeans

    from npap.exceptions import PartitioningError

    if n_clusters is None or n_clusters <= 0:
        raise PartitioningError("K-Means requires a positive 'n_clusters' parameter.")

    try:
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            max_iter=max_iter,
            n_init=n_init,
        )
        return kmeans.fit_predict(features)

    except Exception as e:
        raise PartitioningError(f"K-Means clustering failed: {e}") from e


def run_kmedoids(distance_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Perform K-Medoids clustering with precomputed distance matrix.

    K-Medoids is robust to outliers and works naturally with any distance
    metric through precomputed distance matrices.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Precomputed distance matrix (n x n), must be symmetric.
    n_clusters : int
        Number of clusters to form.

    Returns
    -------
    np.ndarray
        Array of cluster labels for each sample.

    Raises
    ------
    PartitioningError
        If clustering fails.
    """
    import kmedoids

    from npap.exceptions import PartitioningError

    if n_clusters is None or n_clusters <= 0:
        raise PartitioningError("K-Medoids requires a positive 'n_clusters' parameter.")

    try:
        result = kmedoids.fasterpam(distance_matrix, n_clusters)
        return result.labels

    except Exception as e:
        raise PartitioningError(f"K-Medoids clustering failed: {e}") from e


def run_hierarchical(
    distance_matrix: np.ndarray, n_clusters: int, linkage: str = "complete"
) -> np.ndarray:
    """
    Perform hierarchical (agglomerative) clustering with precomputed distance matrix.

    Hierarchical clustering builds a tree of clusters and is deterministic.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Precomputed distance matrix (n x n).
    n_clusters : int
        Number of clusters to form.
    linkage : str, default='complete'
        Linkage criterion ('complete', 'average', 'single').
        Note: 'ward' is NOT supported with precomputed distances.

    Returns
    -------
    np.ndarray
        Array of cluster labels for each sample.

    Raises
    ------
    PartitioningError
        If clustering fails or invalid linkage specified.
    """
    from sklearn.cluster import AgglomerativeClustering

    from npap.exceptions import PartitioningError

    valid_linkages = {"complete", "average", "single"}
    if linkage not in valid_linkages:
        raise PartitioningError(
            f"Unsupported linkage '{linkage}' for precomputed distances. "
            f"Valid options: {', '.join(valid_linkages)}. "
            "Note: 'ward' requires Euclidean distance on raw features."
        )

    if n_clusters is None or n_clusters <= 0:
        raise PartitioningError(
            "Hierarchical clustering requires a positive 'n_clusters' parameter."
        )

    try:
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters, metric="precomputed", linkage=linkage
        )
        return clustering.fit_predict(distance_matrix)

    except Exception as e:
        raise PartitioningError(f"Hierarchical clustering failed: {e}") from e


def run_dbscan(distance_matrix: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """
    Perform DBSCAN clustering with precomputed distance matrix.

    DBSCAN is a density-based algorithm that can find arbitrarily shaped
    clusters and identify noise points (labeled as -1).

    Parameters
    ----------
    distance_matrix : np.ndarray
        Precomputed distance matrix (n x n).
    eps : float
        Maximum distance between two samples to be considered neighbors.
    min_samples : int
        Minimum samples in a neighborhood for a core point.

    Returns
    -------
    np.ndarray
        Array of cluster labels (-1 indicates noise).

    Raises
    ------
    PartitioningError
        If clustering fails.
    """
    from sklearn.cluster import DBSCAN

    from npap.exceptions import PartitioningError

    if eps is None or min_samples is None:
        raise PartitioningError("DBSCAN requires 'eps' and 'min_samples' parameters.")

    try:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
        return dbscan.fit_predict(distance_matrix)

    except Exception as e:
        raise PartitioningError(f"DBSCAN clustering failed: {e}") from e


def run_hdbscan(
    features: np.ndarray, min_cluster_size: int = 5, metric: str = "euclidean"
) -> np.ndarray:
    """
    Perform HDBSCAN clustering.

    HDBSCAN is a hierarchical density-based algorithm that automatically
    determines the number of clusters.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix (coordinates in radians for haversine).
    min_cluster_size : int, default=5
        Minimum size of clusters.
    metric : str, default='euclidean'
        Distance metric ('euclidean' or 'haversine').

    Returns
    -------
    np.ndarray
        Array of cluster labels (-1 indicates noise).

    Raises
    ------
    PartitioningError
        If clustering fails.
    """
    import hdbscan

    from npap.exceptions import PartitioningError

    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, metric=metric, core_dist_n_jobs=-1
        )
        return clusterer.fit_predict(features)

    except Exception as e:
        raise PartitioningError(f"HDBSCAN clustering failed: {e}") from e


# =============================================================================
# DISTANCE MATRIX UTILITIES
# =============================================================================


def compute_euclidean_distances(coordinates: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distance matrix from coordinates.

    Parameters
    ----------
    coordinates : np.ndarray
        Array of coordinates (n x 2) as [lat, lon] or [x, y].

    Returns
    -------
    np.ndarray
        Distance matrix (n x n).
    """
    from sklearn.metrics.pairwise import euclidean_distances

    return euclidean_distances(coordinates)


def compute_haversine_distances(coordinates: np.ndarray) -> np.ndarray:
    """
    Compute Haversine (great-circle) distance matrix from coordinates.

    Parameters
    ----------
    coordinates : np.ndarray
        Array of coordinates (n x 2) as [lat, lon] in DEGREES.

    Returns
    -------
    np.ndarray
        Distance matrix (n x n) in kilometers.
    """
    from sklearn.metrics.pairwise import haversine_distances

    coords_rad = np.radians(coordinates)
    earth_radius_km = 6371
    return haversine_distances(coords_rad) * earth_radius_km


def compute_geographical_distances(
    coordinates: np.ndarray, metric: str = "euclidean"
) -> np.ndarray:
    """
    Compute geographical distance matrix using specified metric.

    Parameters
    ----------
    coordinates : np.ndarray
        Array of coordinates (n x 2) as [lat, lon].
    metric : str, default='euclidean'
        Distance metric ('euclidean' or 'haversine').

    Returns
    -------
    np.ndarray
        Distance matrix (n x n).

    Raises
    ------
    PartitioningError
        If unsupported metric specified.
    """
    from npap.exceptions import PartitioningError

    if metric == "euclidean":
        return compute_euclidean_distances(coordinates)
    elif metric == "haversine":
        return compute_haversine_distances(coordinates)
    else:
        raise PartitioningError(f"Unsupported distance metric: {metric}")
