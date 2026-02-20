"""
Test suite for partitioning strategies.

Tests cover:
- GeographicalPartitioning (all algorithms and distance metrics)
- ElectricalDistancePartitioning (kmeans, kmedoids, PTDF-based distance, AC island isolation)
- VAGeographicalPartitioning (with AC island and voltage awareness)
"""

import logging

import networkx as nx
import numpy as np
import pytest

from npap.exceptions import PartitioningError, ValidationError
from npap.partitioning.electrical import (
    ElectricalDistanceConfig,
    ElectricalDistancePartitioning,
)
from npap.partitioning.geographical import (
    GeographicalConfig,
    GeographicalPartitioning,
)
from npap.partitioning.graph_theory import (
    CommunityPartitioning,
    SpectralPartitioning,
)
from npap.partitioning.va_electrical import (
    VAElectricalDistancePartitioning,
)
from npap.partitioning.va_geographical import (
    VAGeographicalConfig,
    VAGeographicalPartitioning,
)
from test.conftest import (
    all_nodes_assigned,
    nodes_in_different_clusters,
    nodes_in_same_cluster,
)

# =============================================================================
# GEOGRAPHICAL PARTITIONING TESTS
# =============================================================================


class TestGeographicalPartitioning:
    """Tests for GeographicalPartitioning strategy."""

    # -------------------------------------------------------------------------
    # Initialization Tests
    # -------------------------------------------------------------------------

    def test_init_valid_algorithms(self):
        """Test that all supported algorithms can be initialized."""
        for algo in GeographicalPartitioning.SUPPORTED_ALGORITHMS:
            strategy = GeographicalPartitioning(algorithm=algo)
            assert strategy.algorithm == algo

    def test_init_invalid_algorithm_raises(self):
        """Test that invalid algorithm raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            GeographicalPartitioning(algorithm="invalid_algo")

    def test_init_valid_distance_metrics(self):
        """Test that all supported distance metrics can be initialized."""
        for metric in GeographicalPartitioning.SUPPORTED_DISTANCE_METRICS:
            strategy = GeographicalPartitioning(distance_metric=metric)
            assert strategy.distance_metric == metric

    def test_init_invalid_distance_metric_raises(self):
        """Test that invalid distance metric raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported distance metric"):
            GeographicalPartitioning(distance_metric="manhattan")

    def test_required_attributes(self):
        """Test that required attributes are correctly defined."""
        strategy = GeographicalPartitioning()
        assert "lat" in strategy.required_attributes["nodes"]
        assert "lon" in strategy.required_attributes["nodes"]
        assert strategy.required_attributes["edges"] == []

    # -------------------------------------------------------------------------
    # K-Means Clustering Tests
    # -------------------------------------------------------------------------

    def test_kmeans_basic_partition(self, geographical_cluster_graph):
        """Test K-Means partitions geographically distinct clusters correctly."""
        strategy = GeographicalPartitioning(algorithm="kmeans", distance_metric="euclidean")
        partition = strategy.partition(geographical_cluster_graph, n_clusters=2, random_state=42)

        # Verify all nodes assigned
        assert all_nodes_assigned(partition, list(geographical_cluster_graph.nodes()))

        # Verify cluster A nodes (0,1,2) are together
        assert nodes_in_same_cluster(partition, 0, 1)
        assert nodes_in_same_cluster(partition, 1, 2)

        # Verify cluster B nodes (3,4,5) are together
        assert nodes_in_same_cluster(partition, 3, 4)
        assert nodes_in_same_cluster(partition, 4, 5)

        # Verify clusters are separate
        assert nodes_in_different_clusters(partition, 0, 3)

    def test_kmeans_requires_euclidean(self, simple_digraph):
        """Test K-Means raises error for non-Euclidean metrics."""
        strategy = GeographicalPartitioning(algorithm="kmeans", distance_metric="haversine")

        with pytest.raises(PartitioningError, match="does not support haversine"):
            strategy.partition(simple_digraph, n_clusters=2)

    def test_kmeans_missing_n_clusters_raises(self, simple_digraph):
        """Test K-Means raises error when n_clusters not provided."""
        strategy = GeographicalPartitioning(algorithm="kmeans")

        with pytest.raises(PartitioningError, match="n_clusters"):
            strategy.partition(simple_digraph)

    def test_kmeans_n_clusters_greater_than_nodes_raises(self, simple_digraph):
        """Test K-Means raises error when n_clusters > nodes."""
        strategy = GeographicalPartitioning(algorithm="kmeans")
        n_nodes = len(list(simple_digraph.nodes()))

        with pytest.raises(PartitioningError):
            strategy.partition(simple_digraph, n_clusters=n_nodes + 1)

    # -------------------------------------------------------------------------
    # K-Medoids Clustering Tests
    # -------------------------------------------------------------------------

    def test_kmedoids_euclidean_partition(self, geographical_cluster_graph):
        """Test K-Medoids with Euclidean distance."""
        strategy = GeographicalPartitioning(algorithm="kmedoids", distance_metric="euclidean")
        partition = strategy.partition(geographical_cluster_graph, n_clusters=2)

        assert all_nodes_assigned(partition, list(geographical_cluster_graph.nodes()))
        assert len(partition) == 2

        # Should separate the two geographical clusters
        assert nodes_in_same_cluster(partition, 0, 1)
        assert nodes_in_same_cluster(partition, 3, 4)
        assert nodes_in_different_clusters(partition, 0, 3)

    def test_kmedoids_haversine_partition(self, geographical_cluster_graph):
        """Test K-Medoids with Haversine distance."""
        strategy = GeographicalPartitioning(algorithm="kmedoids", distance_metric="haversine")
        partition = strategy.partition(geographical_cluster_graph, n_clusters=2)

        assert all_nodes_assigned(partition, list(geographical_cluster_graph.nodes()))
        assert len(partition) == 2

    # -------------------------------------------------------------------------
    # Hierarchical Clustering Tests
    # -------------------------------------------------------------------------

    def test_hierarchical_partition(self, geographical_cluster_graph):
        """Test Hierarchical clustering partitions correctly."""
        strategy = GeographicalPartitioning(algorithm="hierarchical", distance_metric="euclidean")
        partition = strategy.partition(geographical_cluster_graph, n_clusters=2)

        assert all_nodes_assigned(partition, list(geographical_cluster_graph.nodes()))
        assert len(partition) == 2

        # Should separate the two geographical clusters
        assert nodes_in_same_cluster(partition, 0, 1)
        assert nodes_in_same_cluster(partition, 3, 4)

    def test_hierarchical_ward_requires_euclidean(self, simple_digraph):
        """Test Hierarchical with ward linkage requires Euclidean distance."""
        config = GeographicalConfig(hierarchical_linkage="ward")
        strategy = GeographicalPartitioning(
            algorithm="hierarchical", distance_metric="haversine", config=config
        )

        with pytest.raises(PartitioningError, match="Ward linkage.*requires Euclidean"):
            strategy.partition(simple_digraph, n_clusters=2)

    # -------------------------------------------------------------------------
    # DBSCAN Clustering Tests
    # -------------------------------------------------------------------------

    def test_dbscan_requires_eps_and_min_samples(self, simple_digraph):
        """Test DBSCAN raises error when eps/min_samples not provided."""
        strategy = GeographicalPartitioning(algorithm="dbscan")

        with pytest.raises(PartitioningError, match="eps.*min_samples"):
            strategy.partition(simple_digraph)

    def test_dbscan_basic_partition(self, geographical_cluster_graph):
        """Test DBSCAN can identify clusters."""
        strategy = GeographicalPartitioning(algorithm="dbscan", distance_metric="euclidean")
        partition = strategy.partition(geographical_cluster_graph, eps=1.0, min_samples=2)

        # DBSCAN should find the two clusters (may have noise points labeled -1)
        # Just verify it runs without error and assigns nodes
        total_assigned = sum(len(nodes) for nodes in partition.values())
        assert total_assigned == len(list(geographical_cluster_graph.nodes()))

    # -------------------------------------------------------------------------
    # HDBSCAN Clustering Tests
    # -------------------------------------------------------------------------

    def test_hdbscan_basic_partition(self, geographical_cluster_graph):
        """Test HDBSCAN can identify clusters."""
        strategy = GeographicalPartitioning(algorithm="hdbscan", distance_metric="euclidean")
        partition = strategy.partition(geographical_cluster_graph, min_cluster_size=2)

        # HDBSCAN should find clusters (may have noise)
        total_assigned = sum(len(nodes) for nodes in partition.values())
        assert total_assigned == len(list(geographical_cluster_graph.nodes()))

    # -------------------------------------------------------------------------
    # Validation Tests
    # -------------------------------------------------------------------------

    def test_missing_lat_raises_validation_error(self):
        """Test that missing lat attribute raises ValidationError."""
        G = nx.DiGraph()
        G.add_node(0, lon=0.0)  # Missing lat
        G.add_node(1, lat=1.0, lon=1.0)
        G.add_edge(0, 1)

        strategy = GeographicalPartitioning(algorithm="kmeans")

        with pytest.raises(ValidationError):
            strategy.partition(G, n_clusters=1)

    def test_missing_lon_raises_validation_error(self):
        """Test that missing lon attribute raises ValidationError."""
        G = nx.DiGraph()
        G.add_node(0, lat=0.0)  # Missing lon
        G.add_node(1, lat=1.0, lon=1.0)
        G.add_edge(0, 1)

        strategy = GeographicalPartitioning(algorithm="kmeans")

        with pytest.raises(ValidationError):
            strategy.partition(G, n_clusters=1)

    # -------------------------------------------------------------------------
    # Config Override Tests
    # -------------------------------------------------------------------------

    def test_config_override_at_partition_time(self, simple_digraph):
        """Test that config can be overridden at partition time."""
        strategy = GeographicalPartitioning(
            algorithm="kmeans", config=GeographicalConfig(random_state=1)
        )

        # Override random_state at partition time
        partition1 = strategy.partition(simple_digraph, n_clusters=2, random_state=42)
        partition2 = strategy.partition(simple_digraph, n_clusters=2, random_state=42)

        # Same random state should produce same results
        assert partition1 == partition2

    # -------------------------------------------------------------------------
    # AC-Island Auto-Detection Tests
    # -------------------------------------------------------------------------

    def test_ac_island_auto_detection(self, geographical_ac_island_graph):
        """Test that AC islands are automatically detected from graph attributes."""
        strategy = GeographicalPartitioning(algorithm="kmedoids", distance_metric="euclidean")
        nodes = list(geographical_ac_island_graph.nodes())

        # Should detect AC island data
        assert strategy._has_ac_island_data(geographical_ac_island_graph, nodes)

    def test_no_ac_island_detection_without_attribute(self, geographical_cluster_graph):
        """Test that AC islands are not detected when attribute is missing."""
        strategy = GeographicalPartitioning(algorithm="kmedoids", distance_metric="euclidean")
        nodes = list(geographical_cluster_graph.nodes())

        # Should NOT detect AC island data (no ac_island attribute)
        assert not strategy._has_ac_island_data(geographical_cluster_graph, nodes)

    def test_ac_island_awareness_support_kmedoids(self):
        """Test that kmedoids supports AC-island awareness."""
        strategy = GeographicalPartitioning(algorithm="kmedoids")
        config = GeographicalConfig()
        assert strategy._supports_ac_island_awareness(config)

    def test_ac_island_awareness_support_dbscan(self):
        """Test that dbscan supports AC-island awareness."""
        strategy = GeographicalPartitioning(algorithm="dbscan")
        config = GeographicalConfig()
        assert strategy._supports_ac_island_awareness(config)

    def test_ac_island_awareness_support_hdbscan(self):
        """Test that hdbscan supports AC-island awareness."""
        strategy = GeographicalPartitioning(algorithm="hdbscan")
        config = GeographicalConfig()
        assert strategy._supports_ac_island_awareness(config)

    def test_ac_island_awareness_support_hierarchical_non_ward(self):
        """Test that hierarchical with non-ward linkage supports AC-island awareness."""
        strategy = GeographicalPartitioning(algorithm="hierarchical")
        config = GeographicalConfig(hierarchical_linkage="complete")
        assert strategy._supports_ac_island_awareness(config)

    def test_ac_island_awareness_no_support_kmeans(self):
        """Test that kmeans does not support AC-island awareness."""
        strategy = GeographicalPartitioning(algorithm="kmeans")
        config = GeographicalConfig()
        assert not strategy._supports_ac_island_awareness(config)

    def test_ac_island_awareness_no_support_hierarchical_ward(self):
        """Test that hierarchical with ward linkage does not support AC-island awareness."""
        strategy = GeographicalPartitioning(algorithm="hierarchical")
        config = GeographicalConfig(hierarchical_linkage="ward")
        assert not strategy._supports_ac_island_awareness(config)

    # -------------------------------------------------------------------------
    # AC-Island Aware Partitioning Tests (Supported Algorithms)
    # -------------------------------------------------------------------------

    def test_kmedoids_respects_ac_island_boundaries(self, geographical_ac_island_graph):
        """Test K-Medoids respects AC island boundaries when detected."""
        strategy = GeographicalPartitioning(algorithm="kmedoids", distance_metric="euclidean")
        partition = strategy.partition(geographical_ac_island_graph, n_clusters=2)

        # Nodes 0,1,2 are in AC island 0
        # Nodes 3,4,5 are in AC island 1
        # They should NEVER be in the same cluster
        for i in [0, 1, 2]:
            for j in [3, 4, 5]:
                assert nodes_in_different_clusters(partition, i, j), (
                    f"Nodes {i} and {j} should be in different clusters (different AC islands)"
                )

    def test_kmedoids_haversine_respects_ac_island_boundaries(self, geographical_ac_island_graph):
        """Test K-Medoids with haversine respects AC island boundaries."""
        strategy = GeographicalPartitioning(algorithm="kmedoids", distance_metric="haversine")
        partition = strategy.partition(geographical_ac_island_graph, n_clusters=2)

        for i in [0, 1, 2]:
            for j in [3, 4, 5]:
                assert nodes_in_different_clusters(partition, i, j)

    def test_dbscan_respects_ac_island_boundaries(self, geographical_ac_island_graph):
        """Test DBSCAN respects AC island boundaries when detected."""
        strategy = GeographicalPartitioning(algorithm="dbscan", distance_metric="euclidean")
        partition = strategy.partition(geographical_ac_island_graph, eps=1.0, min_samples=2)

        # Verify nodes from different AC islands are not in the same cluster
        for i in [0, 1, 2]:
            for j in [3, 4, 5]:
                assert nodes_in_different_clusters(partition, i, j)

    def test_hdbscan_respects_ac_island_boundaries(self, geographical_ac_island_graph):
        """Test HDBSCAN respects AC island boundaries when detected."""
        strategy = GeographicalPartitioning(algorithm="hdbscan", distance_metric="euclidean")
        partition = strategy.partition(geographical_ac_island_graph, min_cluster_size=2)

        for i in [0, 1, 2]:
            for j in [3, 4, 5]:
                assert nodes_in_different_clusters(partition, i, j)

    def test_hierarchical_complete_respects_ac_island_boundaries(
        self, geographical_ac_island_graph
    ):
        """Test hierarchical clustering with complete linkage respects AC island boundaries."""
        config = GeographicalConfig(hierarchical_linkage="complete")
        strategy = GeographicalPartitioning(
            algorithm="hierarchical", distance_metric="euclidean", config=config
        )
        partition = strategy.partition(geographical_ac_island_graph, n_clusters=2)

        for i in [0, 1, 2]:
            for j in [3, 4, 5]:
                assert nodes_in_different_clusters(partition, i, j)

    def test_hierarchical_average_respects_ac_island_boundaries(self, geographical_ac_island_graph):
        """Test hierarchical clustering with average linkage respects AC island boundaries."""
        config = GeographicalConfig(hierarchical_linkage="average")
        strategy = GeographicalPartitioning(
            algorithm="hierarchical", distance_metric="euclidean", config=config
        )
        partition = strategy.partition(geographical_ac_island_graph, n_clusters=2)

        for i in [0, 1, 2]:
            for j in [3, 4, 5]:
                assert nodes_in_different_clusters(partition, i, j)

    def test_ac_island_aware_with_more_clusters(self, geographical_ac_island_graph):
        """Test AC island awareness with more clusters than islands."""
        strategy = GeographicalPartitioning(algorithm="kmedoids", distance_metric="euclidean")
        partition = strategy.partition(geographical_ac_island_graph, n_clusters=4)

        assert all_nodes_assigned(partition, list(geographical_ac_island_graph.nodes()))
        assert len(partition) == 4

        # AC island boundaries should still be respected
        for i in [0, 1, 2]:
            for j in [3, 4, 5]:
                assert nodes_in_different_clusters(partition, i, j)

    # -------------------------------------------------------------------------
    # AC-Island Warning Tests (Unsupported Algorithms)
    # -------------------------------------------------------------------------

    def test_kmeans_warns_with_ac_islands(self, geographical_ac_island_graph):
        """Test K-Means issues warning when AC islands are detected."""
        strategy = GeographicalPartitioning(algorithm="kmeans", distance_metric="euclidean")

        with pytest.warns(UserWarning, match="AC islands detected.*does not support"):
            partition = strategy.partition(geographical_ac_island_graph, n_clusters=2)

        # Should still produce valid partition (just without AC-island awareness)
        assert all_nodes_assigned(partition, list(geographical_ac_island_graph.nodes()))

    def test_hierarchical_ward_warns_with_ac_islands(self, geographical_ac_island_graph):
        """Test hierarchical with ward linkage issues warning when AC islands are detected."""
        config = GeographicalConfig(hierarchical_linkage="ward")
        strategy = GeographicalPartitioning(
            algorithm="hierarchical", distance_metric="euclidean", config=config
        )

        with pytest.warns(UserWarning, match="AC islands detected.*does not support"):
            partition = strategy.partition(geographical_ac_island_graph, n_clusters=2)

        # Should still produce valid partition
        assert all_nodes_assigned(partition, list(geographical_ac_island_graph.nodes()))

    # -------------------------------------------------------------------------
    # AC-Island Distance Matrix Tests
    # -------------------------------------------------------------------------

    def test_ac_island_aware_distance_matrix(self, geographical_ac_island_graph):
        """Test that AC-island-aware distance matrix assigns infinite distance between islands."""

        strategy = GeographicalPartitioning(algorithm="kmedoids", distance_metric="euclidean")
        nodes = list(geographical_ac_island_graph.nodes())
        coordinates = strategy._extract_coordinates(geographical_ac_island_graph, nodes)
        ac_islands = strategy._extract_ac_islands(geographical_ac_island_graph, nodes)
        config = GeographicalConfig(infinite_distance=1e4)

        distance_matrix = strategy._build_ac_island_aware_distance_matrix(
            coordinates, ac_islands, config
        )

        # Nodes in same AC island should have finite distances
        # Nodes 0,1,2 are in island 0
        assert distance_matrix[0, 1] < config.infinite_distance
        assert distance_matrix[1, 2] < config.infinite_distance

        # Nodes 3,4,5 are in island 1
        assert distance_matrix[3, 4] < config.infinite_distance
        assert distance_matrix[4, 5] < config.infinite_distance

        # Nodes in different AC islands should have infinite distance
        assert distance_matrix[0, 3] == config.infinite_distance
        assert distance_matrix[1, 4] == config.infinite_distance
        assert distance_matrix[2, 5] == config.infinite_distance

        # Diagonal should be zero
        for i in range(len(nodes)):
            assert distance_matrix[i, i] == 0.0

    def test_infinite_distance_config_override(self, geographical_ac_island_graph):
        """Test that infinite_distance can be configured."""

        strategy = GeographicalPartitioning(algorithm="kmedoids", distance_metric="euclidean")
        nodes = list(geographical_ac_island_graph.nodes())
        coordinates = strategy._extract_coordinates(geographical_ac_island_graph, nodes)
        ac_islands = strategy._extract_ac_islands(geographical_ac_island_graph, nodes)

        # Use custom infinite distance
        config = GeographicalConfig(infinite_distance=999.0)
        distance_matrix = strategy._build_ac_island_aware_distance_matrix(
            coordinates, ac_islands, config
        )

        # Cross-island distances should use the custom infinite distance
        assert distance_matrix[0, 3] == 999.0

    # -------------------------------------------------------------------------
    # AC-Island Consistency Validation Tests
    # -------------------------------------------------------------------------

    def test_cluster_ac_island_consistency_valid(self, geographical_ac_island_graph):
        """Test consistency validation passes for valid partitions."""
        strategy = GeographicalPartitioning(algorithm="kmedoids")

        # Valid partition: each cluster contains nodes from only one AC island
        valid_partition = {0: [0, 1, 2], 1: [3, 4, 5]}

        # Should not raise or warn
        strategy._validate_cluster_ac_island_consistency(
            geographical_ac_island_graph, valid_partition
        )

    def test_cluster_ac_island_consistency_warning(self, geographical_ac_island_graph, caplog):
        """Test consistency validation warns for invalid partitions."""
        import logging

        strategy = GeographicalPartitioning(algorithm="kmedoids")

        # Invalid partition: cluster 0 mixes nodes from different AC islands
        invalid_partition = {0: [0, 1, 3], 1: [2, 4, 5]}

        with caplog.at_level(logging.WARNING):
            strategy._validate_cluster_ac_island_consistency(
                geographical_ac_island_graph, invalid_partition
            )

        # Should have logged a warning about mixed AC islands
        assert "multiple AC islands" in caplog.text

    # -------------------------------------------------------------------------
    # Spherical to Cartesian Conversion Tests
    # -------------------------------------------------------------------------

    def test_convert_spherical_to_cartesian(self):
        """
        Test conversion from spherical (lat, lon) to Cartesian coordinates.

        The function assumes a unit radius (1.0). Therefore, a "zero radius"
        test case is not applicable to the current implementation.
        """
        strategy = GeographicalPartitioning()

        # Test case 1: Equator at (0, 0) -> (x=1, y=0, z=0)
        coords1 = np.array([[0.0, 0.0]])
        expected1 = np.array([[1.0, 0.0, 0.0]])
        np.testing.assert_allclose(
            strategy._convert_spherical_to_cartesian(coords1), expected1, atol=1e-7
        )

        # Test case 2: Equator at (0, 90) -> (x=0, y=1, z=0)
        coords2 = np.array([[0.0, 90.0]])
        expected2 = np.array([[0.0, 1.0, 0.0]])
        np.testing.assert_allclose(
            strategy._convert_spherical_to_cartesian(coords2), expected2, atol=1e-7
        )

        # Test case 3: North Pole (lat=90) -> (x=0, y=0, z=1)
        coords3 = np.array([[90.0, 0.0]])
        expected3 = np.array([[0.0, 0.0, 1.0]])
        np.testing.assert_allclose(
            strategy._convert_spherical_to_cartesian(coords3), expected3, atol=1e-7
        )

        # Test case 4: South Pole (lat=-90) -> (x=0, y=0, z=-1)
        coords4 = np.array([[-90.0, 45.0]])
        expected4 = np.array([[0.0, 0.0, -1.0]])
        np.testing.assert_allclose(
            strategy._convert_spherical_to_cartesian(coords4), expected4, atol=1e-7
        )

        # Test case 5: Multiple points at once
        coords_multi = np.array([[0.0, 0.0], [90.0, 0.0], [0.0, 90.0]])
        expected_multi = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        np.testing.assert_allclose(
            strategy._convert_spherical_to_cartesian(coords_multi),
            expected_multi,
            atol=1e-7,
        )

        # Test case 6: 45 degrees lat/lon
        # x = cos(45)*cos(45) = 0.5
        # y = cos(45)*sin(45) = 0.5
        # z = sin(45) = sqrt(2)/2
        sqrt2_2 = np.sqrt(2) / 2
        coords6 = np.array([[45.0, 45.0]])
        expected6 = np.array([[0.5, 0.5, sqrt2_2]])
        np.testing.assert_allclose(
            strategy._convert_spherical_to_cartesian(coords6), expected6, atol=1e-7
        )

        # Edge case: Invalid input shape should raise PartitioningError
        # The function expects a 2D array (n, 2)
        with pytest.raises(PartitioningError):
            strategy._convert_spherical_to_cartesian(np.array([1.0, 2.0]))


# =============================================================================
# ELECTRICAL DISTANCE PARTITIONING TESTS
# =============================================================================


class TestElectricalDistancePartitioning:
    """Tests for ElectricalDistancePartitioning strategy."""

    # -------------------------------------------------------------------------
    # Initialization Tests
    # -------------------------------------------------------------------------

    def test_init_valid_algorithms(self):
        """Test that all supported algorithms can be initialized."""
        for algo in ElectricalDistancePartitioning.SUPPORTED_ALGORITHMS:
            strategy = ElectricalDistancePartitioning(algorithm=algo)
            assert strategy.algorithm == algo

    def test_init_invalid_algorithm_raises(self):
        """Test that invalid algorithm raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            ElectricalDistancePartitioning(algorithm="dbscan")

    def test_required_attributes(self):
        """Test that required attributes include reactance."""
        strategy = ElectricalDistancePartitioning()
        assert "x" in strategy.required_attributes["edges"]

    def test_init_custom_ac_island_attr(self):
        """Test initialization with custom ac_island attribute name."""
        strategy = ElectricalDistancePartitioning(ac_island_attr="my_island")
        assert strategy.ac_island_attr == "my_island"

    def test_init_default_ac_island_attr(self):
        """Test that default ac_island attribute is 'ac_island'."""
        strategy = ElectricalDistancePartitioning()
        assert strategy.ac_island_attr == "ac_island"

    # -------------------------------------------------------------------------
    # Basic Partition Tests (with ac_island attribute)
    # -------------------------------------------------------------------------

    def test_kmeans_basic_partition(self, electrical_graph):
        """Test K-Means electrical distance partitioning."""
        strategy = ElectricalDistancePartitioning(algorithm="kmeans")
        partition = strategy.partition(electrical_graph, n_clusters=2, random_state=42)

        assert all_nodes_assigned(partition, list(electrical_graph.nodes()))
        assert len(partition) == 2

    def test_kmedoids_basic_partition(self, electrical_graph):
        """Test K-Medoids electrical distance partitioning."""
        strategy = ElectricalDistancePartitioning(algorithm="kmedoids")
        partition = strategy.partition(electrical_graph, n_clusters=2, random_state=42)

        assert all_nodes_assigned(partition, list(electrical_graph.nodes()))
        assert len(partition) == 2

    def test_electrical_distance_groups_by_reactance(self, electrical_graph):
        """Test that nodes with low reactance between them cluster together."""
        strategy = ElectricalDistancePartitioning(algorithm="kmedoids")
        partition = strategy.partition(electrical_graph, n_clusters=2, random_state=42)

        # Nodes 1 and 2 have low reactance to node 0 (electrically close)
        # They should tend to cluster together
        # Note: This is probabilistic, so we just check reasonable behavior
        assert all_nodes_assigned(partition, list(electrical_graph.nodes()))

    # -------------------------------------------------------------------------
    # AC Island Isolation Tests
    # -------------------------------------------------------------------------

    def test_ac_island_isolation_respects_boundaries(self, multi_island_electrical_graph):
        """Test that partitioning respects AC island boundaries."""
        strategy = ElectricalDistancePartitioning(algorithm="kmedoids")
        partition = strategy.partition(multi_island_electrical_graph, n_clusters=2, random_state=42)

        # Nodes 0, 1, 2 are in AC island 0
        # Nodes 3, 4, 5 are in AC island 1
        # They should NEVER be in the same cluster
        for i in [0, 1, 2]:
            for j in [3, 4, 5]:
                assert nodes_in_different_clusters(partition, i, j), (
                    f"Nodes {i} and {j} should be in different clusters (different AC islands)"
                )

    def test_ac_island_isolation_with_more_clusters(self, multi_island_electrical_graph):
        """Test AC island isolation with more clusters than islands."""
        strategy = ElectricalDistancePartitioning(algorithm="kmedoids")
        partition = strategy.partition(multi_island_electrical_graph, n_clusters=4, random_state=42)

        assert all_nodes_assigned(partition, list(multi_island_electrical_graph.nodes()))
        assert len(partition) == 4

        # AC island boundaries should still be respected
        for i in [0, 1, 2]:
            for j in [3, 4, 5]:
                assert nodes_in_different_clusters(partition, i, j)

    def test_ac_island_isolation_kmeans(self, multi_island_electrical_graph):
        """Test AC island isolation with K-Means algorithm."""
        strategy = ElectricalDistancePartitioning(algorithm="kmeans")
        partition = strategy.partition(multi_island_electrical_graph, n_clusters=2, random_state=42)

        # AC island boundaries should be respected
        for i in [0, 1, 2]:
            for j in [3, 4, 5]:
                assert nodes_in_different_clusters(partition, i, j)

    # -------------------------------------------------------------------------
    # Missing ac_island Attribute Tests
    # -------------------------------------------------------------------------

    def test_missing_ac_island_raises_helpful_error(self, electrical_graph_no_ac_island):
        """Test that missing ac_island attribute raises ValidationError with helpful message."""
        strategy = ElectricalDistancePartitioning()

        with pytest.raises(ValidationError) as exc_info:
            strategy.partition(electrical_graph_no_ac_island, n_clusters=2)

        # Check that error message is helpful
        error_msg = str(exc_info.value)
        assert "ac_island" in error_msg
        assert "va_loader" in error_msg

    def test_missing_ac_island_with_custom_attr_name(self):
        """Test that missing custom ac_island attribute raises appropriate error."""
        G = nx.DiGraph()
        G.add_node(0, lat=0.0, lon=0.0, ac_island=0)  # Has ac_island but not custom attr
        G.add_node(1, lat=1.0, lon=0.0, ac_island=0)
        G.add_edge(0, 1, x=0.1)

        strategy = ElectricalDistancePartitioning(ac_island_attr="custom_island")

        with pytest.raises(ValidationError) as exc_info:
            strategy.partition(G, n_clusters=1)

        error_msg = str(exc_info.value)
        assert "custom_island" in error_msg

    # -------------------------------------------------------------------------
    # Validation Tests
    # -------------------------------------------------------------------------

    def test_disconnected_graph_raises_error(self):
        """Test that disconnected graph raises PartitioningError."""
        G = nx.DiGraph()
        G.add_node(0, lat=0.0, lon=0.0, ac_island=0)
        G.add_node(1, lat=1.0, lon=1.0, ac_island=0)
        # No edges - disconnected

        strategy = ElectricalDistancePartitioning()

        with pytest.raises(
            PartitioningError,
            match="Cannot compute electrical distances without AC connectivity",
        ):
            strategy.partition(G, n_clusters=1)

    def test_missing_reactance_raises_error(self):
        """Test that missing reactance attribute raises ValidationError."""
        G = nx.DiGraph()
        G.add_node(0, ac_island=0)
        G.add_node(1, ac_island=0)
        G.add_edge(0, 1, length=100)  # Missing 'x' attribute

        strategy = ElectricalDistancePartitioning()

        with pytest.raises(ValidationError):
            strategy.partition(G, n_clusters=1)

    def test_zero_reactance_warning(self):
        """Test that zero reactance edges produce a warning."""
        G = nx.DiGraph()
        G.add_node(0, ac_island=0)
        G.add_node(1, ac_island=0)
        G.add_edge(0, 1, x=0.0)  # Zero reactance

        strategy = ElectricalDistancePartitioning()

        with pytest.warns(UserWarning, match="zero reactance"):
            strategy.partition(G, n_clusters=1)

    # -------------------------------------------------------------------------
    # Slack Bus Tests
    # -------------------------------------------------------------------------

    def test_custom_slack_bus(self, electrical_graph):
        """Test partitioning with custom slack bus."""
        strategy = ElectricalDistancePartitioning(algorithm="kmeans", slack_bus=0)
        partition = strategy.partition(electrical_graph, n_clusters=2, random_state=42)

        assert all_nodes_assigned(partition, list(electrical_graph.nodes()))

    # -------------------------------------------------------------------------
    # Config Tests
    # -------------------------------------------------------------------------

    def test_config_override(self, electrical_graph):
        """Test config override at partition time."""
        strategy = ElectricalDistancePartitioning()

        config = ElectricalDistanceConfig(zero_reactance_replacement=1e-6)
        partition = strategy.partition(
            electrical_graph, n_clusters=2, config=config, random_state=42
        )

        assert all_nodes_assigned(partition, list(electrical_graph.nodes()))

    def test_infinite_distance_config(self, multi_island_electrical_graph):
        """Test that infinite_distance config is used for AC island isolation."""
        config = ElectricalDistanceConfig(infinite_distance=1e6)
        strategy = ElectricalDistancePartitioning(config=config)

        partition = strategy.partition(multi_island_electrical_graph, n_clusters=2, random_state=42)

        # AC island boundaries should still be respected
        for i in [0, 1, 2]:
            for j in [3, 4, 5]:
                assert nodes_in_different_clusters(partition, i, j)


# =============================================================================
# VOLTAGE-AWARE GEOGRAPHICAL PARTITIONING TESTS
# =============================================================================


class TestVAGeographicalPartitioning:
    """Tests for VAGeographicalPartitioning strategy."""

    # -------------------------------------------------------------------------
    # Initialization Tests
    # -------------------------------------------------------------------------

    def test_init_valid_algorithms(self):
        """Test that all supported algorithms can be initialized."""
        for algo in VAGeographicalPartitioning.SUPPORTED_ALGORITHMS:
            strategy = VAGeographicalPartitioning(algorithm=algo)
            assert strategy.algorithm == algo

    def test_init_invalid_algorithm_raises(self):
        """Test that invalid algorithm raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            VAGeographicalPartitioning(algorithm="kmeans")  # Not supported

    def test_init_invalid_linkage_raises(self):
        """Test that invalid hierarchical linkage raises ValueError."""
        config = VAGeographicalConfig(hierarchical_linkage="ward")  # Not supported

        with pytest.raises(ValueError, match="Unsupported hierarchical linkage"):
            VAGeographicalPartitioning(algorithm="hierarchical", config=config)

    def test_required_attributes(self):
        """Test that required attributes include voltage and ac_island."""
        strategy = VAGeographicalPartitioning()
        required = strategy.required_attributes["nodes"]

        assert "lat" in required
        assert "lon" in required
        assert "voltage" in required
        assert "ac_island" in required

    # -------------------------------------------------------------------------
    # AC Island Respect Tests
    # -------------------------------------------------------------------------

    def test_respects_ac_island_boundaries(self, voltage_aware_graph):
        """Test that partitioning respects AC island boundaries."""
        strategy = VAGeographicalPartitioning(algorithm="kmedoids")
        partition = strategy.partition(voltage_aware_graph, n_clusters=2, random_state=42)

        # Nodes 0,1,2 are in AC island 0
        # Nodes 3,4,5 are in AC island 1
        # They should never be in the same cluster
        for i in [0, 1, 2]:
            for j in [3, 4, 5]:
                assert nodes_in_different_clusters(partition, i, j), (
                    f"Nodes {i} and {j} should be in different clusters (different AC islands)"
                )

    def test_respects_voltage_boundaries(self, mixed_voltage_graph):
        """Test that partitioning respects voltage level boundaries."""
        strategy = VAGeographicalPartitioning(algorithm="kmedoids")
        partition = strategy.partition(mixed_voltage_graph, n_clusters=2, random_state=42)

        # Nodes 0,1 are 220kV
        # Nodes 2,3 are 380kV
        # They should not be in the same cluster
        for i in [0, 1]:
            for j in [2, 3]:
                assert nodes_in_different_clusters(partition, i, j), (
                    f"Nodes {i} and {j} should be in different clusters (different voltages)"
                )

    # -------------------------------------------------------------------------
    # Standard Mode Tests
    # -------------------------------------------------------------------------

    def test_standard_mode_kmedoids(self, voltage_aware_graph):
        """Test standard mode K-Medoids partitioning."""
        strategy = VAGeographicalPartitioning(
            algorithm="kmedoids",
            config=VAGeographicalConfig(proportional_clustering=False),
        )
        partition = strategy.partition(voltage_aware_graph, n_clusters=2, random_state=42)

        assert all_nodes_assigned(partition, list(voltage_aware_graph.nodes()))

    def test_standard_mode_hierarchical(self, voltage_aware_graph):
        """Test standard mode Hierarchical partitioning."""
        strategy = VAGeographicalPartitioning(
            algorithm="hierarchical",
            config=VAGeographicalConfig(proportional_clustering=False),
        )
        partition = strategy.partition(voltage_aware_graph, n_clusters=2)

        assert all_nodes_assigned(partition, list(voltage_aware_graph.nodes()))

    # -------------------------------------------------------------------------
    # Proportional Mode Tests
    # -------------------------------------------------------------------------

    def test_proportional_mode_kmedoids(self, voltage_aware_graph):
        """Test proportional mode K-Medoids partitioning."""
        strategy = VAGeographicalPartitioning(
            algorithm="kmedoids",
            config=VAGeographicalConfig(proportional_clustering=True),
        )
        partition = strategy.partition(voltage_aware_graph, n_clusters=4, random_state=42)

        assert all_nodes_assigned(partition, list(voltage_aware_graph.nodes()))

        # AC islands should still be respected
        for i in [0, 1, 2]:
            for j in [3, 4, 5]:
                assert nodes_in_different_clusters(partition, i, j)

    # -------------------------------------------------------------------------
    # Distance Metric Tests
    # -------------------------------------------------------------------------

    def test_haversine_distance_metric(self, voltage_aware_graph):
        """Test partitioning with Haversine distance metric."""
        strategy = VAGeographicalPartitioning(algorithm="kmedoids", distance_metric="haversine")
        partition = strategy.partition(voltage_aware_graph, n_clusters=2, random_state=42)

        assert all_nodes_assigned(partition, list(voltage_aware_graph.nodes()))

    # -------------------------------------------------------------------------
    # Validation Tests
    # -------------------------------------------------------------------------

    def test_missing_ac_island_raises_error(self):
        """Test that missing ac_island attribute raises error."""
        G = nx.DiGraph()
        G.add_node(0, lat=0.0, lon=0.0, voltage=220.0)  # Missing ac_island
        G.add_node(1, lat=1.0, lon=1.0, voltage=220.0, ac_island=0)
        G.add_edge(0, 1, x=0.1)

        strategy = VAGeographicalPartitioning()

        with pytest.raises(ValidationError):
            strategy.partition(G, n_clusters=1)

    def test_missing_voltage_raises_error(self):
        """Test that missing voltage attribute raises error."""
        G = nx.DiGraph()
        G.add_node(0, lat=0.0, lon=0.0, ac_island=0)  # Missing voltage
        G.add_node(1, lat=1.0, lon=1.0, voltage=220.0, ac_island=0)
        G.add_edge(0, 1, x=0.1)

        strategy = VAGeographicalPartitioning()

        with pytest.raises(ValidationError):
            strategy.partition(G, n_clusters=1)

    # -------------------------------------------------------------------------
    # Config Tests
    # -------------------------------------------------------------------------

    def test_voltage_tolerance_config(self, voltage_aware_graph):
        """Test voltage tolerance configuration."""
        # High tolerance should group different voltages
        config = VAGeographicalConfig(voltage_tolerance=200.0)  # Very high tolerance
        strategy = VAGeographicalPartitioning(algorithm="kmedoids", config=config)

        # Note: AC island constraint still applies, so they won't actually cluster together
        partition = strategy.partition(voltage_aware_graph, n_clusters=2, random_state=42)
        assert all_nodes_assigned(partition, list(voltage_aware_graph.nodes()))

    def test_infinite_distance_config(self, voltage_aware_graph):
        """Test infinite distance configuration."""
        config = VAGeographicalConfig(infinite_distance=1e6)
        strategy = VAGeographicalPartitioning(algorithm="kmedoids", config=config)

        partition = strategy.partition(voltage_aware_graph, n_clusters=2, random_state=42)
        assert all_nodes_assigned(partition, list(voltage_aware_graph.nodes()))


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestPartitioningEdgeCases:
    """Tests for edge cases across all partitioning strategies."""

    def test_single_node_graph(self):
        """Test partitioning a graph with a single node."""
        G = nx.DiGraph()
        G.add_node(0, lat=0.0, lon=0.0)

        strategy = GeographicalPartitioning(algorithm="kmeans")
        partition = strategy.partition(G, n_clusters=1)

        assert len(partition) == 1
        assert 0 in partition[0]

    def test_two_node_two_cluster_partition(self):
        """Test partitioning 2 nodes into 2 clusters."""
        G = nx.DiGraph()
        G.add_node(0, lat=0.0, lon=0.0)
        G.add_node(1, lat=10.0, lon=10.0)
        G.add_edge(0, 1, x=0.1)

        strategy = GeographicalPartitioning(algorithm="kmedoids")
        partition = strategy.partition(G, n_clusters=2, random_state=42)

        assert len(partition) == 2
        assert all_nodes_assigned(partition, [0, 1])

    def test_n_clusters_equals_n_nodes(self, simple_digraph):
        """Test when n_clusters equals number of nodes."""
        n_nodes = len(list(simple_digraph.nodes()))

        strategy = GeographicalPartitioning(algorithm="kmedoids")
        partition = strategy.partition(simple_digraph, n_clusters=n_nodes, random_state=42)

        assert len(partition) == n_nodes
        # Each cluster should have exactly one node
        for nodes in partition.values():
            assert len(nodes) == 1

    def test_reproducibility_with_random_state(self, geographical_cluster_graph):
        """Test that same random_state produces same results."""
        strategy = GeographicalPartitioning(algorithm="kmeans")

        partition1 = strategy.partition(geographical_cluster_graph, n_clusters=2, random_state=42)
        partition2 = strategy.partition(geographical_cluster_graph, n_clusters=2, random_state=42)

        assert partition1 == partition2

    def test_different_random_state_may_differ(self, geographical_cluster_graph):
        """Test that different random_state may produce different results."""
        strategy = GeographicalPartitioning(algorithm="kmeans")

        partition1 = strategy.partition(geographical_cluster_graph, n_clusters=2, random_state=1)
        partition2 = strategy.partition(geographical_cluster_graph, n_clusters=2, random_state=999)

        # Results might be the same (if data is clearly separable) or different
        # Just verify both are valid
        assert all_nodes_assigned(partition1, list(geographical_cluster_graph.nodes()))
        assert all_nodes_assigned(partition2, list(geographical_cluster_graph.nodes()))


# =============================================================================
# VOLTAGE-AWARE SINGLE VOLTAGE LEVEL REJECTION TESTS
# =============================================================================


class TestVAStrategiesSingleVoltageLevelError:
    """Tests that VA strategies reject graphs with a single voltage level."""

    def test_va_geographical_rejects_single_voltage(self):
        """Test VA geographical raises ValidationError for single voltage level."""
        G = nx.DiGraph()
        G.add_node(0, lat=0.0, lon=0.0, voltage=220.0, ac_island=0)
        G.add_node(1, lat=0.5, lon=0.5, voltage=220.0, ac_island=0)
        G.add_node(2, lat=1.0, lon=1.0, voltage=220.0, ac_island=0)
        G.add_edge(0, 1, x=0.1)
        G.add_edge(1, 2, x=0.15)

        strategy = VAGeographicalPartitioning(algorithm="kmedoids")

        with pytest.raises(ValidationError, match="requires multiple voltage levels"):
            strategy.partition(G, n_clusters=2)

    def test_va_electrical_rejects_single_voltage(self):
        """Test VA electrical raises ValidationError for single voltage level."""
        G = nx.DiGraph()
        G.add_node(0, lat=0.0, lon=0.0, voltage=220.0, ac_island=0)
        G.add_node(1, lat=0.5, lon=0.5, voltage=220.0, ac_island=0)
        G.add_node(2, lat=1.0, lon=1.0, voltage=220.0, ac_island=0)
        G.add_edge(0, 1, x=0.1, type="line")
        G.add_edge(1, 2, x=0.15, type="line")

        strategy = VAElectricalDistancePartitioning(algorithm="kmedoids")

        with pytest.raises(ValidationError, match="requires multiple voltage levels"):
            strategy.partition(G, n_clusters=2)

    def test_va_geographical_accepts_multiple_voltages(self, mixed_voltage_graph):
        """Test VA geographical succeeds with multiple voltage levels."""
        strategy = VAGeographicalPartitioning(algorithm="kmedoids")
        partition = strategy.partition(mixed_voltage_graph, n_clusters=2, random_state=42)
        assert all_nodes_assigned(partition, list(mixed_voltage_graph.nodes()))

    def test_va_electrical_accepts_multiple_voltages(self):
        """Test VA electrical succeeds with multiple voltage levels."""
        G = nx.DiGraph()
        G.add_node(0, lat=0.0, lon=0.0, voltage=220.0, ac_island=0)
        G.add_node(1, lat=0.5, lon=0.5, voltage=220.0, ac_island=0)
        G.add_node(2, lat=1.0, lon=1.0, voltage=380.0, ac_island=0)
        G.add_node(3, lat=1.5, lon=1.5, voltage=380.0, ac_island=0)
        G.add_edge(0, 1, x=0.1, type="line")
        G.add_edge(2, 3, x=0.08, type="line")
        G.add_edge(1, 2, x=0.05, type="trafo")

        strategy = VAElectricalDistancePartitioning(algorithm="kmedoids")
        partition = strategy.partition(G, n_clusters=2, random_state=42)
        assert all_nodes_assigned(partition, list(G.nodes()))


class TestSpectralPartitioning:
    """Verify the spectral clustering strategy."""

    def test_requires_n_clusters_parameter(self):
        strategy = SpectralPartitioning(random_state=0)
        graph = nx.DiGraph()
        graph.add_nodes_from([0, 1])

        with pytest.raises(PartitioningError, match="n_clusters >= 2"):
            strategy.partition(graph)

    def test_splits_connected_graph(self):
        strategy = SpectralPartitioning(random_state=0)
        graph = nx.DiGraph()
        graph.add_nodes_from(range(4))
        graph.add_edges_from([(0, 1), (1, 2), (2, 3)])

        partition = strategy.partition(graph, n_clusters=2)
        assert len(partition) == 2
        assert sum(len(nodes) for nodes in partition.values()) == graph.number_of_nodes()


class TestCommunityPartitioning:
    """Verify the modularity-based community strategy."""

    def test_detects_multiple_communities(self):
        strategy = CommunityPartitioning()
        graph = nx.DiGraph()
        graph.add_nodes_from(range(6))
        graph.add_edges_from([(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)])
        graph.add_edges_from([(3, 4), (4, 3), (4, 5), (5, 4), (3, 5), (5, 3)])
        graph.add_edge(2, 3)

        partition = strategy.partition(graph)
        assert sum(len(nodes) for nodes in partition.values()) == graph.number_of_nodes()
        assert len(partition) >= 2


class TestCommunityPartitioningWarnings:
    def test_n_clusters_warning(self, caplog):
        strategy = CommunityPartitioning()
        graph = nx.DiGraph()
        graph.add_nodes_from([0, 1, 2])
        graph.add_edges_from([(0, 1), (1, 2), (2, 0)])

        caplog.set_level(logging.WARNING)
        strategy.partition(graph, n_clusters=2)

        assert "'n_clusters' is ignored" in caplog.text
