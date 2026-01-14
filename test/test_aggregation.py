"""
Test suite for aggregation strategies.

Tests cover:
- Topology strategies (SimpleTopologyStrategy, ElectricalTopologyStrategy)
- Node property strategies (SumNodeStrategy, AverageNodeStrategy, FirstNodeStrategy)
- Edge property strategies (SumEdgeStrategy, AverageEdgeStrategy, FirstEdgeStrategy, EquivalentReactanceStrategy)
- Aggregation modes (SIMPLE, GEOGRAPHICAL)
- Parallel edge aggregation
"""

import networkx as nx
import pytest

from npap import AggregationError
from npap.aggregation.basic_strategies import (
    AverageEdgeStrategy,
    AverageNodeStrategy,
    ElectricalTopologyStrategy,
    EquivalentReactanceStrategy,
    FirstEdgeStrategy,
    FirstNodeStrategy,
    SimpleTopologyStrategy,
    SumEdgeStrategy,
    SumNodeStrategy,
)
from npap.aggregation.modes import get_mode_profile
from npap.interfaces import AggregationMode, AggregationProfile
from npap.managers import AggregationManager

# =============================================================================
# TOPOLOGY STRATEGY TESTS
# =============================================================================


class TestSimpleTopologyStrategy:
    """Tests for SimpleTopologyStrategy."""

    def test_creates_one_node_per_cluster(self, simple_digraph, simple_partition_map):
        """Test that one node is created per cluster."""
        strategy = SimpleTopologyStrategy()
        result = strategy.create_topology(simple_digraph, simple_partition_map)

        assert len(list(result.nodes())) == len(simple_partition_map)
        assert set(result.nodes()) == set(simple_partition_map.keys())

    def test_creates_edges_where_connections_exist(self, simple_digraph, simple_partition_map):
        """Test that edges are created only where original connections exist."""
        # Partition: {0: [0, 2], 1: [1, 3]}
        # Original edges: 0->1, 0->2 (internal), 1->3 (internal), 2->3
        # Expected aggregated edges: 0->1 (from 0->1, 2->3)

        strategy = SimpleTopologyStrategy()
        result = strategy.create_topology(simple_digraph, simple_partition_map)

        # There should be edges between clusters where connections exist
        assert result.has_edge(0, 1)  # From original 0->1 or 2->3

    def test_preserves_edge_direction(self, simple_digraph):
        """Test that edge direction is preserved from original graph."""
        # Custom partition where we can clearly test direction
        partition = {0: [0], 1: [1], 2: [2], 3: [3]}

        strategy = SimpleTopologyStrategy()
        result = strategy.create_topology(simple_digraph, partition)

        # Original: 0->1, 0->2, 1->3, 2->3
        assert result.has_edge(0, 1)
        assert result.has_edge(0, 2)
        assert result.has_edge(1, 3)
        assert result.has_edge(2, 3)

        # Should NOT have reverse edges
        assert not result.has_edge(1, 0)
        assert not result.has_edge(2, 0)

    def test_no_edge_between_unconnected_clusters(self, simple_digraph):
        """Test that no edge is created between unconnected clusters."""
        # Create a partition where some clusters have no connections
        G = nx.DiGraph()
        G.add_node(0, lat=0.0, lon=0.0)
        G.add_node(1, lat=1.0, lon=1.0)
        G.add_node(2, lat=2.0, lon=2.0)
        G.add_edge(0, 1)  # Only 0->1 exists

        partition = {0: [0], 1: [1], 2: [2]}

        strategy = SimpleTopologyStrategy()
        result = strategy.create_topology(G, partition)

        assert result.has_edge(0, 1)
        assert not result.has_edge(0, 2)
        assert not result.has_edge(1, 2)

    def test_can_create_new_edges_property(self):
        """Test that can_create_new_edges returns False."""
        strategy = SimpleTopologyStrategy()
        assert strategy.can_create_new_edges is False


class TestElectricalTopologyStrategy:
    """Tests for ElectricalTopologyStrategy."""

    def test_existing_connectivity_mode(self, simple_digraph, simple_partition_map):
        """Test 'existing' connectivity mode behaves like simple."""
        strategy = ElectricalTopologyStrategy(initial_connectivity="existing")
        result = strategy.create_topology(simple_digraph, simple_partition_map)

        assert len(list(result.nodes())) == len(simple_partition_map)

    def test_full_connectivity_mode(self, simple_digraph, simple_partition_map):
        """Test 'full' connectivity mode creates all possible edges."""
        strategy = ElectricalTopologyStrategy(initial_connectivity="full")
        result = strategy.create_topology(simple_digraph, simple_partition_map)

        n_clusters = len(simple_partition_map)
        # Full connectivity: n*(n-1) directed edges (excluding self-loops)
        expected_edges = n_clusters * (n_clusters - 1)

        assert len(result.edges()) == expected_edges

    def test_full_mode_creates_bidirectional_edges(self, simple_digraph):
        """Test full mode creates edges in both directions."""
        partition = {0: [0, 1], 1: [2, 3]}

        strategy = ElectricalTopologyStrategy(initial_connectivity="full")
        result = strategy.create_topology(simple_digraph, partition)

        # Both directions should exist
        assert result.has_edge(0, 1)
        assert result.has_edge(1, 0)

    def test_can_create_new_edges_depends_on_mode(self):
        """Test that can_create_new_edges depends on connectivity mode."""
        existing_strategy = ElectricalTopologyStrategy(initial_connectivity="existing")
        full_strategy = ElectricalTopologyStrategy(initial_connectivity="full")

        assert existing_strategy.can_create_new_edges is False
        assert full_strategy.can_create_new_edges is True

    def test_invalid_connectivity_mode_raises(self, simple_digraph, simple_partition_map):
        """Test that invalid connectivity mode raises ValueError."""
        strategy = ElectricalTopologyStrategy(initial_connectivity="invalid")

        with pytest.raises(AggregationError, match="Unknown connectivity mode"):
            strategy.create_topology(simple_digraph, simple_partition_map)


# =============================================================================
# NODE PROPERTY STRATEGY TESTS
# =============================================================================


class TestSumNodeStrategy:
    """Tests for SumNodeStrategy."""

    def test_sums_numerical_values(self, simple_digraph):
        """Test that numerical values are summed correctly."""
        strategy = SumNodeStrategy()
        nodes = [0, 1, 2, 3]

        # Demands: 10, 20, 30, 40 -> Sum: 100
        result = strategy.aggregate_property(simple_digraph, nodes, "demand")
        assert result == 100.0

    def test_sum_subset_of_nodes(self, simple_digraph):
        """Test sum for a subset of nodes."""
        strategy = SumNodeStrategy()
        nodes = [0, 2]  # Demands: 10, 30 -> Sum: 40

        result = strategy.aggregate_property(simple_digraph, nodes, "demand")
        assert result == 40.0

    def test_sum_missing_property_returns_zero(self, simple_digraph):
        """Test that missing property returns 0."""
        strategy = SumNodeStrategy()
        nodes = [0, 1]

        result = strategy.aggregate_property(simple_digraph, nodes, "nonexistent")
        assert result == 0

    def test_sum_ignores_non_numerical(self, simple_digraph):
        """Test that non-numerical values are ignored."""
        strategy = SumNodeStrategy()
        nodes = [0, 1]

        # 'name' is a string property
        result = strategy.aggregate_property(simple_digraph, nodes, "name")
        assert result == 0


class TestAverageNodeStrategy:
    """Tests for AverageNodeStrategy."""

    def test_averages_numerical_values(self, simple_digraph):
        """Test that numerical values are averaged correctly."""
        strategy = AverageNodeStrategy()
        nodes = [0, 1, 2, 3]

        # Demands: 10, 20, 30, 40 -> Average: 25
        result = strategy.aggregate_property(simple_digraph, nodes, "demand")
        assert result == 25.0

    def test_average_subset_of_nodes(self, simple_digraph):
        """Test average for a subset of nodes."""
        strategy = AverageNodeStrategy()
        nodes = [0, 2]  # Demands: 10, 30 -> Average: 20

        result = strategy.aggregate_property(simple_digraph, nodes, "demand")
        assert result == 20.0

    def test_average_coordinates(self, simple_digraph):
        """Test averaging lat/lon coordinates."""
        strategy = AverageNodeStrategy()
        nodes = [0, 1, 2, 3]

        # Lats: 0, 0, 1, 1 -> Average: 0.5
        lat_result = strategy.aggregate_property(simple_digraph, nodes, "lat")
        assert lat_result == 0.5

        # Lons: 0, 1, 0, 1 -> Average: 0.5
        lon_result = strategy.aggregate_property(simple_digraph, nodes, "lon")
        assert lon_result == 0.5


class TestFirstNodeStrategy:
    """Tests for FirstNodeStrategy."""

    def test_returns_first_value(self, simple_digraph):
        """Test that first available value is returned."""
        strategy = FirstNodeStrategy()
        nodes = [0, 1, 2]

        # Names: node_0, node_1, node_2 -> First: node_0
        result = strategy.aggregate_property(simple_digraph, nodes, "name")
        assert result == "node_0"

    def test_returns_first_numerical(self, simple_digraph):
        """Test first value for numerical properties."""
        strategy = FirstNodeStrategy()
        nodes = [1, 2, 3]

        # Demands: 20, 30, 40 -> First: 20
        result = strategy.aggregate_property(simple_digraph, nodes, "demand")
        assert result == 20.0

    def test_returns_none_for_missing(self, simple_digraph):
        """Test that None is returned for missing property."""
        strategy = FirstNodeStrategy()
        nodes = [0, 1]

        result = strategy.aggregate_property(simple_digraph, nodes, "nonexistent")
        assert result is None


# =============================================================================
# EDGE PROPERTY STRATEGY TESTS
# =============================================================================


class TestSumEdgeStrategy:
    """Tests for SumEdgeStrategy."""

    def test_sums_edge_values(self):
        """Test that edge values are summed correctly."""
        strategy = SumEdgeStrategy()
        edges = [{"length": 100.0, "p_max": 500.0}, {"length": 150.0, "p_max": 600.0}]

        length_result = strategy.aggregate_property(edges, "length")
        assert length_result == 250.0

        pmax_result = strategy.aggregate_property(edges, "p_max")
        assert pmax_result == 1100.0

    def test_sum_missing_property_returns_zero(self):
        """Test that missing property returns 0."""
        strategy = SumEdgeStrategy()
        edges = [{"x": 0.1}, {"x": 0.2}]

        result = strategy.aggregate_property(edges, "nonexistent")
        assert result == 0


class TestAverageEdgeStrategy:
    """Tests for AverageEdgeStrategy."""

    def test_averages_edge_values(self):
        """Test that edge values are averaged correctly."""
        strategy = AverageEdgeStrategy()
        edges = [{"x": 0.1}, {"x": 0.2}, {"x": 0.3}]

        result = strategy.aggregate_property(edges, "x")
        assert result == pytest.approx(0.2)


class TestFirstEdgeStrategy:
    """Tests for FirstEdgeStrategy."""

    def test_returns_first_value(self):
        """Test that first edge value is returned."""
        strategy = FirstEdgeStrategy()
        edges = [{"type": "line"}, {"type": "trafo"}]

        result = strategy.aggregate_property(edges, "type")
        assert result == "line"


class TestEquivalentReactanceStrategy:
    """Tests for EquivalentReactanceStrategy (parallel edge aggregation)."""

    def test_parallel_reactance_formula(self):
        """Test equivalent reactance for parallel lines: 1/(1/x1 + 1/x2)."""
        strategy = EquivalentReactanceStrategy()

        # Two parallel lines with x=0.2 and x=0.3
        # Equivalent: 1 / (1/0.2 + 1/0.3) = 1 / (5 + 3.333) = 1/8.333 ≈ 0.12
        edges = [{"x": 0.2}, {"x": 0.3}]

        result = strategy.aggregate_property(edges, "x")
        expected = 1.0 / (1.0 / 0.2 + 1.0 / 0.3)  # = 0.12

        assert result == pytest.approx(expected)

    def test_equal_parallel_reactances(self):
        """Test equivalent reactance for equal parallel lines: x/2."""
        strategy = EquivalentReactanceStrategy()

        # Two parallel lines with x=0.4 each
        # Equivalent: 1 / (1/0.4 + 1/0.4) = 1 / 5 = 0.2
        edges = [{"x": 0.4}, {"x": 0.4}]

        result = strategy.aggregate_property(edges, "x")
        assert result == pytest.approx(0.2)

    def test_single_edge(self):
        """Test single edge returns original reactance."""
        strategy = EquivalentReactanceStrategy()
        edges = [{"x": 0.15}]

        result = strategy.aggregate_property(edges, "x")
        assert result == pytest.approx(0.15)

    def test_zero_reactance_short_circuit(self):
        """Test that zero reactance returns 0 (short circuit)."""
        strategy = EquivalentReactanceStrategy()
        edges = [{"x": 0.0}, {"x": 0.3}]

        result = strategy.aggregate_property(edges, "x")
        assert result == 0.0

    def test_missing_property_returns_inf(self):
        """Test that missing property returns infinity (open circuit)."""
        strategy = EquivalentReactanceStrategy()
        edges = [{"length": 100}]  # No 'x' property

        result = strategy.aggregate_property(edges, "x")
        assert result == float("inf")

    def test_three_parallel_lines(self):
        """Test equivalent reactance for three parallel lines."""
        strategy = EquivalentReactanceStrategy()

        # Three parallel lines
        edges = [{"x": 0.3}, {"x": 0.3}, {"x": 0.3}]

        # Equivalent: 1 / (3 * 1/0.3) = 0.3/3 = 0.1
        result = strategy.aggregate_property(edges, "x")
        assert result == pytest.approx(0.1)


# =============================================================================
# AGGREGATION MANAGER TESTS
# =============================================================================


class TestAggregationManager:
    """Tests for AggregationManager orchestration."""

    def test_aggregate_with_simple_profile(self, simple_digraph, simple_partition_map):
        """Test aggregation with SIMPLE mode profile."""
        manager = AggregationManager()
        profile = get_mode_profile(AggregationMode.SIMPLE)

        result = manager.aggregate(simple_digraph, simple_partition_map, profile)

        # Should have 2 nodes (one per cluster)
        assert len(list(result.nodes())) == 2

        # Should have edges where connections existed
        assert len(result.edges()) > 0

    def test_aggregate_with_geographical_profile(self, simple_digraph, simple_partition_map):
        """Test aggregation with GEOGRAPHICAL mode profile."""
        manager = AggregationManager()
        profile = get_mode_profile(AggregationMode.GEOGRAPHICAL, warn_on_defaults=False)

        result = manager.aggregate(simple_digraph, simple_partition_map, profile)

        # Verify lat/lon are averaged
        # Cluster 0: nodes [0, 2] -> lats [0, 1] -> avg 0.5
        assert result.nodes[0]["lat"] == pytest.approx(0.5)

    def test_aggregate_sums_demand(self, simple_digraph, simple_partition_map):
        """Test that demand property is summed correctly."""
        manager = AggregationManager()
        profile = AggregationProfile(
            topology_strategy="simple",
            node_properties={"demand": "sum"},
            default_node_strategy="sum",
            warn_on_defaults=False,
        )

        result = manager.aggregate(simple_digraph, simple_partition_map, profile)

        # Cluster 0: nodes [0, 2] -> demands [10, 30] -> sum 40
        assert result.nodes[0]["demand"] == 40.0

        # Cluster 1: nodes [1, 3] -> demands [20, 40] -> sum 60
        assert result.nodes[1]["demand"] == 60.0

    def test_aggregate_averages_coordinates(self, simple_digraph, simple_partition_map):
        """Test that coordinates are averaged correctly."""
        manager = AggregationManager()
        profile = AggregationProfile(
            topology_strategy="simple",
            node_properties={"lat": "average", "lon": "average"},
            default_node_strategy="average",
            warn_on_defaults=False,
        )

        result = manager.aggregate(simple_digraph, simple_partition_map, profile)

        # Cluster 0: nodes [0, 2] -> lats [0, 1] -> avg 0.5, lons [0, 0] -> avg 0.0
        assert result.nodes[0]["lat"] == pytest.approx(0.5)
        assert result.nodes[0]["lon"] == pytest.approx(0.0)

        # Cluster 1: nodes [1, 3] -> lats [0, 1] -> avg 0.5, lons [1, 1] -> avg 1.0
        assert result.nodes[1]["lat"] == pytest.approx(0.5)
        assert result.nodes[1]["lon"] == pytest.approx(1.0)

    def test_invalid_topology_strategy_raises(self, simple_digraph, simple_partition_map):
        """Test that invalid topology strategy raises ValueError."""
        manager = AggregationManager()
        profile = AggregationProfile(topology_strategy="nonexistent")

        with pytest.raises(ValueError, match="Unknown topology strategy"):
            manager.aggregate(simple_digraph, simple_partition_map, profile)

    def test_invalid_node_strategy_raises(self, simple_digraph, simple_partition_map):
        """Test that invalid node strategy raises ValueError."""
        manager = AggregationManager()
        profile = AggregationProfile(
            topology_strategy="simple",
            node_properties={"demand": "nonexistent_strategy"},
        )

        with pytest.raises(ValueError, match="Unknown node strategy"):
            manager.aggregate(simple_digraph, simple_partition_map, profile)


# =============================================================================
# PARALLEL EDGE AGGREGATION TESTS
# =============================================================================


class TestParallelEdgeAggregation:
    """Tests for parallel edge aggregation in MultiDiGraphs."""

    def test_aggregate_parallel_edges_basic(self, parallel_edge_multigraph):
        """Test basic parallel edge aggregation."""
        manager = AggregationManager()

        result = manager.aggregate_parallel_edges(
            parallel_edge_multigraph,
            edge_properties={"x": "equivalent_reactance"},
            default_strategy="sum",
            warn_on_defaults=False,
        )

        # Result should be a simple DiGraph
        assert isinstance(result, nx.DiGraph)
        assert not isinstance(result, nx.MultiDiGraph)

        # Should have same nodes
        assert set(result.nodes()) == set(parallel_edge_multigraph.nodes())

    def test_parallel_edges_equivalent_reactance(self, parallel_edge_multigraph):
        """Test equivalent reactance calculation for parallel edges."""
        manager = AggregationManager()

        result = manager.aggregate_parallel_edges(
            parallel_edge_multigraph,
            edge_properties={"x": "equivalent_reactance"},
            default_strategy="sum",
            warn_on_defaults=False,
        )

        # Parallel edges 0->1 have x=0.2 and x=0.3
        # Equivalent: 1 / (1/0.2 + 1/0.3) = 0.12
        expected_x = 1.0 / (1.0 / 0.2 + 1.0 / 0.3)

        assert result[0][1]["x"] == pytest.approx(expected_x)

    def test_parallel_edges_sum_length(self, parallel_edge_multigraph):
        """Test summing length for parallel edges."""
        manager = AggregationManager()

        result = manager.aggregate_parallel_edges(
            parallel_edge_multigraph,
            edge_properties={"length": "sum"},
            default_strategy="sum",
            warn_on_defaults=False,
        )

        # Parallel edges 0->1 have length=100 each -> sum=200
        assert result[0][1]["length"] == 200.0

    def test_non_multigraph_raises_error(self, simple_digraph):
        """Test that non-MultiDiGraph raises ValueError."""
        manager = AggregationManager()

        with pytest.raises(ValueError, match="Expected MultiDiGraph"):
            manager.aggregate_parallel_edges(simple_digraph)

    def test_preserves_node_attributes(self, parallel_edge_multigraph):
        """Test that node attributes are preserved."""
        manager = AggregationManager()

        result = manager.aggregate_parallel_edges(
            parallel_edge_multigraph,
            edge_properties={},
            default_strategy="sum",
            warn_on_defaults=False,
        )

        # Check node attributes preserved
        assert result.nodes[0]["lat"] == 0.0
        assert result.nodes[0]["demand"] == 10.0


# =============================================================================
# AGGREGATION MODES TESTS
# =============================================================================


class TestAggregationModes:
    """Tests for pre-defined aggregation modes."""

    def test_simple_mode_profile(self):
        """Test SIMPLE mode profile configuration."""
        profile = get_mode_profile(AggregationMode.SIMPLE)

        assert profile.mode == AggregationMode.SIMPLE
        assert profile.topology_strategy == "simple"
        assert profile.physical_strategy is None
        assert profile.default_node_strategy == "sum"
        assert profile.default_edge_strategy == "sum"
        assert profile.warn_on_defaults is False

    def test_geographical_mode_profile(self):
        """Test GEOGRAPHICAL mode profile configuration."""
        profile = get_mode_profile(AggregationMode.GEOGRAPHICAL)

        assert profile.mode == AggregationMode.GEOGRAPHICAL
        assert profile.topology_strategy == "simple"
        assert profile.node_properties.get("lat") == "average"
        assert profile.node_properties.get("lon") == "average"
        assert profile.default_node_strategy == "average"

    def test_dc_kron_mode_not_implemented(self):
        """Test DC_KRON mode raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            get_mode_profile(AggregationMode.DC_KRON)

    def test_mode_profile_with_overrides(self):
        """Test mode profile with parameter overrides."""
        profile = get_mode_profile(AggregationMode.SIMPLE, default_node_strategy="average")

        assert profile.default_node_strategy == "average"

    def test_mode_profile_dict_override_merges(self):
        """Test that dict overrides merge rather than replace."""
        profile = get_mode_profile(
            AggregationMode.GEOGRAPHICAL, node_properties={"custom_prop": "sum"}
        )

        # Original properties should still exist
        assert profile.node_properties.get("lat") == "average"
        # New property should be added
        assert profile.node_properties.get("custom_prop") == "sum"


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestAggregationEdgeCases:
    """Tests for edge cases in aggregation."""

    def test_single_cluster_partition(self, simple_digraph):
        """Test aggregation when all nodes are in one cluster."""
        partition = {0: [0, 1, 2, 3]}
        manager = AggregationManager()
        profile = get_mode_profile(AggregationMode.SIMPLE)

        result = manager.aggregate(simple_digraph, partition, profile)

        assert len(list(result.nodes())) == 1
        # Self-loops are not created by default
        assert len(result.edges()) == 0

    def test_each_node_own_cluster(self, simple_digraph, single_node_partition_map):
        """Test aggregation when each node is its own cluster."""
        manager = AggregationManager()
        profile = get_mode_profile(AggregationMode.SIMPLE)

        result = manager.aggregate(simple_digraph, single_node_partition_map, profile)

        # Should have same number of nodes
        assert len(list(result.nodes())) == len(list(simple_digraph.nodes()))
        # Should have same number of edges
        assert len(result.edges()) == len(simple_digraph.edges())

    def test_empty_cluster_handling(self, simple_digraph):
        """Test handling of partition with node subset."""
        # Only include some nodes
        partition = {0: [0, 1], 1: [2, 3]}
        manager = AggregationManager()
        profile = get_mode_profile(AggregationMode.SIMPLE)

        result = manager.aggregate(simple_digraph, partition, profile)

        assert len(list(result.nodes())) == 2
