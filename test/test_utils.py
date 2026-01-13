"""
Test suite for utility functions.

Tests cover:
- Graph hash computation and validation
- Partition map utilities
- Distance matrix computations
- Clustering algorithm wrappers
"""

import numpy as np
import pytest

from npap.exceptions import GraphCompatibilityError, PartitioningError
from npap.interfaces import PartitionResult
from npap.utils import (
    compute_graph_hash,
    validate_graph_compatibility,
    create_partition_map,
    validate_partition,
    compute_euclidean_distances,
    compute_haversine_distances,
    compute_geographical_distances,
    run_kmeans,
    run_kmedoids,
    run_hierarchical,
    run_dbscan,
    run_hdbscan,
)


# =============================================================================
# GRAPH HASH TESTS
# =============================================================================


class TestGraphHash:
    """Tests for graph hash computation."""

    def test_hash_consistency(self, simple_digraph):
        """Test that same graph produces same hash."""
        hash1 = compute_graph_hash(simple_digraph)
        hash2 = compute_graph_hash(simple_digraph)

        assert hash1 == hash2

    def test_hash_differs_for_different_graphs(
        self, simple_digraph, geographical_cluster_graph
    ):
        """Test that different graphs produce different hashes."""
        hash1 = compute_graph_hash(simple_digraph)
        hash2 = compute_graph_hash(geographical_cluster_graph)

        assert hash1 != hash2

    def test_hash_changes_with_node_addition(self, simple_digraph):
        """Test that adding a node changes the hash."""
        hash_before = compute_graph_hash(simple_digraph)

        G = simple_digraph.copy()
        G.add_node(100, lat=5.0, lon=5.0)
        hash_after = compute_graph_hash(G)

        assert hash_before != hash_after

    def test_hash_changes_with_edge_addition(self, simple_digraph):
        """Test that adding an edge changes the hash."""
        hash_before = compute_graph_hash(simple_digraph)

        G = simple_digraph.copy()
        G.add_edge(1, 2, x=0.5)  # Add new edge
        hash_after = compute_graph_hash(G)

        assert hash_before != hash_after

    def test_hash_is_string(self, simple_digraph):
        """Test that hash is returned as string."""
        hash_value = compute_graph_hash(simple_digraph)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 16  # SHA256 truncated to 16 chars


class TestGraphCompatibilityValidation:
    """Tests for graph compatibility validation."""

    def test_compatible_partition_passes(self, simple_digraph):
        """Test that compatible partition passes validation."""
        graph_hash = compute_graph_hash(simple_digraph)

        partition_result = PartitionResult(
            mapping={0: [0, 1], 1: [2, 3]},
            original_graph_hash=graph_hash,
            strategy_name="test",
            strategy_metadata={},
            n_clusters=2,
        )

        # Should not raise
        validate_graph_compatibility(partition_result, graph_hash)

    def test_incompatible_partition_raises(
        self, simple_digraph, geographical_cluster_graph
    ):
        """Test that incompatible partition raises GraphCompatibilityError."""
        hash1 = compute_graph_hash(simple_digraph)
        hash2 = compute_graph_hash(geographical_cluster_graph)

        partition_result = PartitionResult(
            mapping={0: [0, 1], 1: [2, 3]},
            original_graph_hash=hash1,  # Created from simple_digraph
            strategy_name="test",
            strategy_metadata={},
            n_clusters=2,
        )

        # Should raise when validated against different graph
        with pytest.raises(GraphCompatibilityError):
            validate_graph_compatibility(partition_result, hash2)


# =============================================================================
# PARTITION MAP UTILITIES TESTS
# =============================================================================


class TestPartitionMapUtilities:
    """Tests for partition map utilities."""

    def test_create_partition_map_basic(self):
        """Test basic partition map creation from labels."""
        nodes = ["A", "B", "C", "D"]
        labels = np.array([0, 0, 1, 1])

        partition_map = create_partition_map(nodes, labels)

        assert partition_map[0] == ["A", "B"]
        assert partition_map[1] == ["C", "D"]

    def test_create_partition_map_preserves_order(self):
        """Test that node order is preserved within clusters."""
        nodes = [1, 2, 3, 4, 5]
        labels = np.array([0, 1, 0, 1, 0])

        partition_map = create_partition_map(nodes, labels)

        # Cluster 0: nodes at indices 0, 2, 4 -> [1, 3, 5]
        assert partition_map[0] == [1, 3, 5]
        # Cluster 1: nodes at indices 1, 3 -> [2, 4]
        assert partition_map[1] == [2, 4]

    def test_create_partition_map_handles_negative_labels(self):
        """Test handling of negative labels (noise in DBSCAN)."""
        nodes = ["A", "B", "C"]
        labels = np.array([0, -1, 0])  # -1 is noise

        partition_map = create_partition_map(nodes, labels)

        assert partition_map[0] == ["A", "C"]
        assert partition_map[-1] == ["B"]

    def test_validate_partition_success(self):
        """Test partition validation passes for valid partition."""
        partition_map = {0: [0, 1], 1: [2, 3]}

        # Should not raise
        validate_partition(partition_map, n_nodes=4, strategy_name="test")

    def test_validate_partition_fails_on_mismatch(self):
        """Test partition validation fails when node count doesn't match."""
        partition_map = {0: [0, 1], 1: [2, 3]}

        with pytest.raises(PartitioningError, match="mismatch"):
            validate_partition(partition_map, n_nodes=5, strategy_name="test")


# =============================================================================
# DISTANCE MATRIX TESTS
# =============================================================================


class TestDistanceMatrixComputation:
    """Tests for distance matrix computations."""

    def test_euclidean_distances_basic(self):
        """Test basic Euclidean distance computation."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        distances = compute_euclidean_distances(coords)

        # Distance from (0,0) to (1,0) = 1.0
        assert distances[0, 1] == pytest.approx(1.0)
        # Distance from (0,0) to (0,1) = 1.0
        assert distances[0, 2] == pytest.approx(1.0)
        # Distance from (1,0) to (0,1) = sqrt(2)
        assert distances[1, 2] == pytest.approx(np.sqrt(2))
        # Diagonal is zero
        assert distances[0, 0] == 0.0

    def test_euclidean_distances_symmetry(self):
        """Test that Euclidean distance matrix is symmetric."""
        coords = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])

        distances = compute_euclidean_distances(coords)

        assert np.allclose(distances, distances.T)

    def test_haversine_distances_basic(self):
        """Test basic Haversine distance computation."""
        # Two points on equator, 1 degree apart
        coords = np.array(
            [
                [0.0, 0.0],  # lat, lon in degrees
                [0.0, 1.0],  # 1 degree east
            ]
        )

        distances = compute_haversine_distances(coords)

        # 1 degree at equator ≈ 111 km
        assert 100 < distances[0, 1] < 120
        assert distances[0, 0] == 0.0

    def test_haversine_distances_symmetry(self):
        """Test that Haversine distance matrix is symmetric."""
        coords = np.array(
            [
                [48.8566, 2.3522],  # Paris
                [51.5074, -0.1278],  # London
                [40.7128, -74.0060],  # New York
            ]
        )

        distances = compute_haversine_distances(coords)

        assert np.allclose(distances, distances.T)

    def test_geographical_distances_euclidean(self):
        """Test geographical distances with Euclidean metric."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0]])

        distances = compute_geographical_distances(coords, metric="euclidean")

        assert distances[0, 1] == pytest.approx(1.0)

    def test_geographical_distances_haversine(self):
        """Test geographical distances with Haversine metric."""
        coords = np.array([[0.0, 0.0], [0.0, 1.0]])

        distances = compute_geographical_distances(coords, metric="haversine")

        # Should return distances in km
        assert distances[0, 1] > 0

    def test_geographical_distances_invalid_metric(self):
        """Test that invalid metric raises error."""
        coords = np.array([[0.0, 0.0], [1.0, 1.0]])

        with pytest.raises(PartitioningError, match="Unsupported distance metric"):
            compute_geographical_distances(coords, metric="manhattan")


# =============================================================================
# CLUSTERING ALGORITHM TESTS
# =============================================================================


class TestClusteringAlgorithms:
    """Tests for clustering algorithm wrappers."""

    @pytest.fixture
    def cluster_features(self):
        """Feature matrix with two clear clusters."""
        return np.array(
            [
                [0.0, 0.0],
                [0.1, 0.1],
                [0.05, -0.05],
                [10.0, 10.0],
                [10.1, 10.1],
                [10.05, 9.95],
            ]
        )

    @pytest.fixture
    def cluster_distance_matrix(self, cluster_features):
        """Precomputed distance matrix for cluster_features."""
        return compute_euclidean_distances(cluster_features)

    # -------------------------------------------------------------------------
    # K-Means Tests
    # -------------------------------------------------------------------------

    def test_kmeans_basic(self, cluster_features):
        """Test K-Means clustering."""
        labels = run_kmeans(cluster_features, n_clusters=2, random_state=42)

        assert len(labels) == len(cluster_features)
        assert len(set(labels)) == 2

        # First 3 points should be in same cluster
        assert labels[0] == labels[1] == labels[2]
        # Last 3 points should be in same cluster
        assert labels[3] == labels[4] == labels[5]

    def test_kmeans_invalid_n_clusters(self, cluster_features):
        """Test K-Means with invalid n_clusters."""
        with pytest.raises(PartitioningError, match="n_clusters"):
            run_kmeans(cluster_features, n_clusters=0)

    # -------------------------------------------------------------------------
    # K-Medoids Tests
    # -------------------------------------------------------------------------

    def test_kmedoids_basic(self, cluster_distance_matrix):
        """Test K-Medoids clustering."""
        labels = run_kmedoids(cluster_distance_matrix, n_clusters=2)

        assert len(labels) == 6
        assert len(set(labels)) == 2

    def test_kmedoids_invalid_n_clusters(self, cluster_distance_matrix):
        """Test K-Medoids with invalid n_clusters."""
        with pytest.raises(PartitioningError, match="n_clusters"):
            run_kmedoids(cluster_distance_matrix, n_clusters=-1)

    # -------------------------------------------------------------------------
    # Hierarchical Tests
    # -------------------------------------------------------------------------

    def test_hierarchical_basic(self, cluster_distance_matrix):
        """Test Hierarchical clustering."""
        labels = run_hierarchical(cluster_distance_matrix, n_clusters=2)

        assert len(labels) == 6
        assert len(set(labels)) == 2

    def test_hierarchical_linkages(self, cluster_distance_matrix):
        """Test Hierarchical clustering with different linkages."""
        for linkage in ["complete", "average", "single"]:
            labels = run_hierarchical(
                cluster_distance_matrix, n_clusters=2, linkage=linkage
            )
            assert len(labels) == 6

    def test_hierarchical_invalid_linkage(self, cluster_distance_matrix):
        """Test Hierarchical with invalid linkage."""
        with pytest.raises(PartitioningError, match="Unsupported linkage"):
            run_hierarchical(cluster_distance_matrix, n_clusters=2, linkage="ward")

    # -------------------------------------------------------------------------
    # DBSCAN Tests
    # -------------------------------------------------------------------------

    def test_dbscan_basic(self, cluster_distance_matrix):
        """Test DBSCAN clustering."""
        labels = run_dbscan(cluster_distance_matrix, eps=1.0, min_samples=2)

        assert len(labels) == 6

    def test_dbscan_missing_params(self, cluster_distance_matrix):
        """Test DBSCAN with missing parameters."""
        with pytest.raises(PartitioningError, match="eps.*min_samples"):
            run_dbscan(cluster_distance_matrix, eps=None, min_samples=None)

    # -------------------------------------------------------------------------
    # HDBSCAN Tests
    # -------------------------------------------------------------------------

    def test_hdbscan_basic(self, cluster_features):
        """Test HDBSCAN clustering."""
        labels = run_hdbscan(cluster_features, min_cluster_size=2)

        assert len(labels) == 6


# =============================================================================
# CONFTEST HELPER FUNCTION TESTS
# =============================================================================


class TestConftestHelpers:
    """Tests for helper functions defined in conftest."""

    def test_nodes_in_same_cluster(self):
        """Test nodes_in_same_cluster helper."""
        from test.conftest import nodes_in_same_cluster

        partition = {0: [1, 2, 3], 1: [4, 5]}

        assert nodes_in_same_cluster(partition, 1, 2) is True
        assert nodes_in_same_cluster(partition, 1, 4) is False

    def test_nodes_in_different_clusters(self):
        """Test nodes_in_different_clusters helper."""
        from test.conftest import nodes_in_different_clusters

        partition = {0: [1, 2], 1: [3, 4]}

        assert nodes_in_different_clusters(partition, 1, 3) is True
        assert nodes_in_different_clusters(partition, 1, 2) is False

    def test_get_node_cluster(self):
        """Test get_node_cluster helper."""
        from test.conftest import get_node_cluster

        partition = {0: ["A", "B"], 1: ["C", "D"]}

        assert get_node_cluster(partition, "A") == 0
        assert get_node_cluster(partition, "C") == 1
        assert get_node_cluster(partition, "Z") == -1  # Not found

    def test_all_nodes_assigned(self):
        """Test all_nodes_assigned helper."""
        from test.conftest import all_nodes_assigned

        partition = {0: [1, 2], 1: [3, 4]}

        assert all_nodes_assigned(partition, [1, 2, 3, 4]) is True
        assert all_nodes_assigned(partition, [1, 2, 3, 4, 5]) is False
        assert all_nodes_assigned(partition, [1, 2, 3]) is False
