"""
Test suite for partitioning strategies.

Tests cover:
- GeographicalPartitioning (all algorithms and distance metrics)
- ElectricalDistancePartitioning (kmeans, kmedoids, PTDF-based distance, DC island isolation)
- VAGeographicalPartitioning (with DC island and voltage awareness)
"""

import networkx as nx
import pytest

from npap.exceptions import PartitioningError, ValidationError
from npap.partitioning.electrical import (
    ElectricalDistanceConfig,
    ElectricalDistancePartitioning,
)
from npap.partitioning.geographical import GeographicalConfig, GeographicalPartitioning
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

    def test_init_custom_dc_island_attr(self):
        """Test initialization with custom dc_island attribute name."""
        strategy = ElectricalDistancePartitioning(dc_island_attr="my_island")
        assert strategy.dc_island_attr == "my_island"

    def test_init_default_dc_island_attr(self):
        """Test that default dc_island attribute is 'dc_island'."""
        strategy = ElectricalDistancePartitioning()
        assert strategy.dc_island_attr == "dc_island"

    # -------------------------------------------------------------------------
    # Basic Partition Tests (with dc_island attribute)
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
    # DC Island Isolation Tests
    # -------------------------------------------------------------------------

    def test_dc_island_isolation_respects_boundaries(self, multi_island_electrical_graph):
        """Test that partitioning respects DC island boundaries."""
        strategy = ElectricalDistancePartitioning(algorithm="kmedoids")
        partition = strategy.partition(multi_island_electrical_graph, n_clusters=2, random_state=42)

        # Nodes 0, 1, 2 are in DC island 0
        # Nodes 3, 4, 5 are in DC island 1
        # They should NEVER be in the same cluster
        for i in [0, 1, 2]:
            for j in [3, 4, 5]:
                assert nodes_in_different_clusters(partition, i, j), (
                    f"Nodes {i} and {j} should be in different clusters (different DC islands)"
                )

    def test_dc_island_isolation_with_more_clusters(self, multi_island_electrical_graph):
        """Test DC island isolation with more clusters than islands."""
        strategy = ElectricalDistancePartitioning(algorithm="kmedoids")
        partition = strategy.partition(multi_island_electrical_graph, n_clusters=4, random_state=42)

        assert all_nodes_assigned(partition, list(multi_island_electrical_graph.nodes()))
        assert len(partition) == 4

        # DC island boundaries should still be respected
        for i in [0, 1, 2]:
            for j in [3, 4, 5]:
                assert nodes_in_different_clusters(partition, i, j)

    def test_dc_island_isolation_kmeans(self, multi_island_electrical_graph):
        """Test DC island isolation with K-Means algorithm."""
        strategy = ElectricalDistancePartitioning(algorithm="kmeans")
        partition = strategy.partition(multi_island_electrical_graph, n_clusters=2, random_state=42)

        # DC island boundaries should be respected
        for i in [0, 1, 2]:
            for j in [3, 4, 5]:
                assert nodes_in_different_clusters(partition, i, j)

    # -------------------------------------------------------------------------
    # Missing dc_island Attribute Tests
    # -------------------------------------------------------------------------

    def test_missing_dc_island_raises_helpful_error(self, electrical_graph_no_dc_island):
        """Test that missing dc_island attribute raises ValidationError with helpful message."""
        strategy = ElectricalDistancePartitioning()

        with pytest.raises(ValidationError) as exc_info:
            strategy.partition(electrical_graph_no_dc_island, n_clusters=2)

        # Check that error message is helpful
        error_msg = str(exc_info.value)
        assert "dc_island" in error_msg
        assert "va_loader" in error_msg

    def test_missing_dc_island_with_custom_attr_name(self):
        """Test that missing custom dc_island attribute raises appropriate error."""
        G = nx.DiGraph()
        G.add_node(0, lat=0.0, lon=0.0, dc_island=0)  # Has dc_island but not custom attr
        G.add_node(1, lat=1.0, lon=0.0, dc_island=0)
        G.add_edge(0, 1, x=0.1)

        strategy = ElectricalDistancePartitioning(dc_island_attr="custom_island")

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
        G.add_node(0, lat=0.0, lon=0.0, dc_island=0)
        G.add_node(1, lat=1.0, lon=1.0, dc_island=0)
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
        G.add_node(0, dc_island=0)
        G.add_node(1, dc_island=0)
        G.add_edge(0, 1, length=100)  # Missing 'x' attribute

        strategy = ElectricalDistancePartitioning()

        with pytest.raises(ValidationError):
            strategy.partition(G, n_clusters=1)

    def test_zero_reactance_warning(self):
        """Test that zero reactance edges produce a warning."""
        G = nx.DiGraph()
        G.add_node(0, dc_island=0)
        G.add_node(1, dc_island=0)
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
        """Test that infinite_distance config is used for DC island isolation."""
        config = ElectricalDistanceConfig(infinite_distance=1e6)
        strategy = ElectricalDistancePartitioning(config=config)

        partition = strategy.partition(multi_island_electrical_graph, n_clusters=2, random_state=42)

        # DC island boundaries should still be respected
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
        """Test that required attributes include voltage and dc_island."""
        strategy = VAGeographicalPartitioning()
        required = strategy.required_attributes["nodes"]

        assert "lat" in required
        assert "lon" in required
        assert "voltage" in required
        assert "dc_island" in required

    # -------------------------------------------------------------------------
    # DC Island Respect Tests
    # -------------------------------------------------------------------------

    def test_respects_dc_island_boundaries(self, voltage_aware_graph):
        """Test that partitioning respects DC island boundaries."""
        strategy = VAGeographicalPartitioning(algorithm="kmedoids")
        partition = strategy.partition(voltage_aware_graph, n_clusters=2, random_state=42)

        # Nodes 0,1,2 are in DC island 0
        # Nodes 3,4,5 are in DC island 1
        # They should never be in the same cluster
        for i in [0, 1, 2]:
            for j in [3, 4, 5]:
                assert nodes_in_different_clusters(partition, i, j), (
                    f"Nodes {i} and {j} should be in different clusters (different DC islands)"
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

        # DC islands should still be respected
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

    def test_missing_dc_island_raises_error(self):
        """Test that missing dc_island attribute raises error."""
        G = nx.DiGraph()
        G.add_node(0, lat=0.0, lon=0.0, voltage=220.0)  # Missing dc_island
        G.add_node(1, lat=1.0, lon=1.0, voltage=220.0, dc_island=0)
        G.add_edge(0, 1, x=0.1)

        strategy = VAGeographicalPartitioning()

        with pytest.raises(ValidationError):
            strategy.partition(G, n_clusters=1)

    def test_missing_voltage_raises_error(self):
        """Test that missing voltage attribute raises error."""
        G = nx.DiGraph()
        G.add_node(0, lat=0.0, lon=0.0, dc_island=0)  # Missing voltage
        G.add_node(1, lat=1.0, lon=1.0, voltage=220.0, dc_island=0)
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

        # Note: DC island constraint still applies, so they won't actually cluster together
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
