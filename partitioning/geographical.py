from typing import Dict, List, Any

import hdbscan
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances, haversine_distances
from sklearn_extra.cluster import KMedoids

from exceptions import PartitioningError
from interfaces import PartitioningStrategy


class GeographicalPartitioning(PartitioningStrategy):
    """Partition nodes based on geographical distance"""

    def __init__(self, algorithm: str = 'kmeans', distance_metric: str = 'euclidean'):
        """
        Initialize geographical partitioning strategy

        Args:
            algorithm: Clustering algorithm ('kmeans', 'kmedoids')
            distance_metric: Distance metric ('haversine', 'euclidean')
        """
        self.algorithm = algorithm
        self.distance_metric = distance_metric

        if algorithm not in ['kmeans', 'kmedoids', 'dbscan', 'hierarchical', 'hdbscan']:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        if distance_metric not in ['haversine', 'euclidean']:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")

    @property
    def required_attributes(self) -> Dict[str, List[str]]:
        """Required node attributes for geographical partitioning"""
        return {
            'nodes': ['latitude', 'longitude'],
            'edges': []
        }

    def partition(self, graph: nx.Graph, n_clusters: int, **kwargs) -> Dict[int, List[Any]]:
        """Partition nodes based on geographical coordinates"""
        try:
            # Extract coordinates
            nodes = list(graph.nodes())
            coordinates = []

            for node in nodes:
                node_data = graph.nodes[node]
                lat = node_data.get('latitude')
                lon = node_data.get('longitude')

                if lat is None or lon is None:
                    raise PartitioningError(
                        f"Node {node} missing latitude or longitude",
                        strategy=f"geographical_{self.algorithm}",
                        graph_info={'nodes': len(list(graph.nodes())), 'edges': len(graph.edges())}
                    )

                coordinates.append([lat, lon])

            coordinates = np.array(coordinates)

            # Perform clustering
            if self.algorithm == 'kmeans':
                labels = self._kmeans_clustering(coordinates, **kwargs)
            elif self.algorithm == 'kmedoids':
                labels = self._kmedoids_clustering(coordinates, **kwargs)
            elif self.algorithm == 'dbscan':
                labels = self._dbscan_clustering(coordinates, **kwargs)
            elif self.algorithm == 'hierarchical':
                labels = self._hierarchical_clustering(coordinates, **kwargs)
            elif self.algorithm == 'hdbscan':
                labels = self._hdbscan_clustering(coordinates, **kwargs)
            else:
                raise PartitioningError(f"Unknown algorithm: {self.algorithm}")

            # Create partition mapping
            partition_map = {}
            for i, label in enumerate(labels):
                if int(label) not in partition_map:
                    partition_map[label] = []
                partition_map[label].append(nodes[i])

            # Validate result
            total_assigned = sum(len(cluster_nodes) for cluster_nodes in partition_map.values())
            if total_assigned != len(nodes):
                raise PartitioningError(
                    f"Partition assignment mismatch: {total_assigned} assigned vs {len(nodes)} total nodes"
                )

            return partition_map

        except Exception as e:
            if isinstance(e, PartitioningError):
                raise PartitioningError(
                    f"Geographical partitioning failed: {e}",
                    strategy=f"geographical_{self.algorithm}",
                    graph_info={'nodes': len(list(graph.nodes())), 'edges': len(graph.edges())}
                ) from e

    def _kmeans_clustering(self, coordinates: np.ndarray, **kwargs) -> np.ndarray:
        """Perform K-means clustering on geographical coordinates"""
        try:
            # K-means requires Euclidean distance
            if self.distance_metric != 'euclidean':
                raise PartitioningError(f"K-means does not support {self.distance_metric} distance.")

            # K-means clustering
            random_state = kwargs.get('random_state', 42)
            max_iter = kwargs.get('max_iter', 300)
            n_clusters = kwargs.get('n_clusters')
            if n_clusters is None or n_clusters <= 0:
                raise PartitioningError("K-means requires a positive 'n_clusters' parameter.")

            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                max_iter=max_iter,
                n_init=10
            )

            labels = kmeans.fit_predict(coordinates)
            return labels

        except Exception as e:
            raise PartitioningError(f"K-means clustering failed: {e}") from e

    def _kmedoids_clustering(self, coordinates: np.ndarray, **kwargs) -> np.ndarray:
        """Perform K-medoids clustering on geographical coordinates"""
        try:
            # Calculate distance matrix based on the specified metric
            if self.distance_metric == 'euclidean':
                distance_matrix = euclidean_distances(coordinates)
            elif self.distance_metric == 'haversine':
                coords_rad = np.radians(coordinates)
                earth_radius_km = 6371
                distance_matrix = haversine_distances(coords_rad) * earth_radius_km
            else:
                raise PartitioningError(f"Unsupported distance metric for K-medoids: {self.distance_metric}")

            # Perform K-medoids clustering using the precomputed distance matrix
            random_state = kwargs.get('random_state', 42)
            max_iter = kwargs.get('max_iter', 300)
            n_clusters = kwargs.get('n_clusters')
            if n_clusters is None or n_clusters <= 0:
                raise PartitioningError("K-medoids requires a positive 'n_clusters' parameter.")

            kmedoids = KMedoids(
                n_clusters=n_clusters,
                metric='precomputed',  # tells the model to use the pre-calculated distance matrix
                random_state=random_state,
                max_iter=max_iter
            )

            labels = kmedoids.fit_predict(distance_matrix)
            return labels

        except Exception as e:
            raise PartitioningError(f"K-medoids clustering failed: {e}") from e

    def _dbscan_clustering(self, coordinates: np.ndarray, **kwargs) -> np.ndarray:
        """Perform DBSCAN clustering on geographical coordinates."""
        try:
            eps = kwargs.get('eps')
            min_samples = kwargs.get('min_samples')

            if eps is None or min_samples is None:
                raise PartitioningError("DBSCAN requires 'eps' and 'min_samples' parameters.")

            # Calculate the distance matrix based on the specified metric
            if self.distance_metric == 'euclidean':
                distance_matrix = euclidean_distances(coordinates)
            elif self.distance_metric == 'haversine':
                coords_rad = np.radians(coordinates)
                earth_radius_km = 6371
                distance_matrix = haversine_distances(coords_rad) * earth_radius_km
            else:
                raise PartitioningError(f"Unsupported distance metric for DBSCAN: {self.distance_metric}")

            # Perform DBSCAN clustering
            dbscan = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric='precomputed'  # tells the model to use the pre-calculated distance matrix
            )

            labels = dbscan.fit_predict(distance_matrix)
            return labels

        except Exception as e:
            raise PartitioningError(f"DBSCAN clustering failed: {e}") from e

    def _hierarchical_clustering(self, coordinates: np.ndarray, **kwargs) -> np.ndarray:
        """Perform Hierarchical Clustering on geographical coordinates."""
        try:
            # Ward linkage only works with Euclidean distance
            if self.distance_metric != 'euclidean':
                raise PartitioningError("Ward linkage for Hierarchical Clustering requires Euclidean distance.")

            n_clusters = kwargs.get('n_clusters')
            if n_clusters is None or n_clusters <= 0:
                raise PartitioningError("Hierarchical clustering requires a positive 'n_clusters' parameter.")

            # Perform Agglomerative Clustering
            agg_cluster = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric=self.distance_metric,
                linkage='ward'
            )

            labels = agg_cluster.fit_predict(coordinates)
            return labels

        except Exception as e:
            raise PartitioningError(f"Hierarchical clustering failed: {e}") from e

    def _hdbscan_clustering(self, coordinates: np.ndarray, **kwargs) -> np.ndarray:
        """Perform HDBSCAN clustering on geographical coordinates."""
        try:
            min_cluster_size = kwargs.get('min_cluster_size', 5)

            # Calculate coordinates in radians if using Haversine distance
            if self.distance_metric == 'euclidean':
                coords_rad = np.radians(coordinates)
            elif self.distance_metric == 'haversine':
                coords_rad = np.radians(coordinates)
            else:
                raise PartitioningError(f"Unsupported distance metric for HDBSCAN: {self.distance_metric}")

            # Perform HDBSCAN clustering
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                metric=self.distance_metric,
                core_dist_n_jobs=-1  # use all available CPU cores
            )

            labels = clusterer.fit_predict(coords_rad)
            return labels

        except Exception as e:
            raise PartitioningError(f"HDBSCAN clustering failed: {e}") from e
