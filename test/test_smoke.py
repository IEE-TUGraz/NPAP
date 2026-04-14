import networkx as nx

import npap


def test_package_metadata():
    """Verify package metadata is correctly accessible."""
    assert npap.__version__ is not None
    assert npap.__version__ != "unknown"
    assert isinstance(npap.__version__, str)


def test_core_components_import():
    """Verify core managers and interfaces can be imported."""
    # Test Manager import
    from npap.managers import PartitionAggregatorManager

    assert PartitionAggregatorManager is not None

    # Test Interface imports
    from npap.interfaces import AggregationMode, AggregationProfile

    assert AggregationMode is not None
    assert AggregationProfile is not None


def test_manager_instantiation():
    """Verify the manager can be instantiated without errors."""
    from npap.managers import PartitionAggregatorManager

    manager = PartitionAggregatorManager()
    assert manager is not None
    # Verify internal state is initialized
    assert hasattr(manager, "partitioning_manager")
    assert hasattr(manager, "aggregation_manager")


def test_copy_graph_returns_deepcopy():
    """Ensure the manager can duplicate the current graph."""
    from npap.managers import PartitionAggregatorManager

    manager = PartitionAggregatorManager()
    graph = nx.DiGraph()
    graph.add_edge("a", "b", x=0.1)
    manager._current_graph = graph

    graph_copy = manager.copy_graph()

    assert graph_copy is not graph
    assert nx.is_isomorphic(graph_copy, graph)

    graph_copy.remove_edge("a", "b")
    assert graph.has_edge("a", "b")
