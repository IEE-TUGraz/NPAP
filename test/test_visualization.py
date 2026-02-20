"""
Tests for the visualization helpers.
"""

import networkx as nx
import plotly.graph_objects as go

from npap.interfaces import PartitionResult
from npap.visualization import clone_graph, export_figure, plot_network


def test_plot_network_accepts_partition_result(simple_digraph):
    """plot_network should accept PartitionResult objects without manual mapping."""
    mapping = {0: [0, 1], 1: [2, 3]}
    partition = PartitionResult(
        mapping=mapping,
        original_graph_hash="hash",
        strategy_name="test",
        strategy_metadata={},
        n_clusters=2,
    )

    fig = plot_network(
        simple_digraph,
        style="clustered",
        partition_result=partition,
        show=False,
    )

    assert fig is not None
    # Ensure the figure contains clustered nodes (should have more than one trace)
    assert len(fig.data) > 0


def test_export_figure_defaults_html(tmp_path):
    """export_figure should write HTML files by default."""
    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1]))
    target = export_figure(fig, tmp_path / "network.html")

    assert target.exists()
    assert target.suffix == ".html"


def test_clone_graph_returns_deep_copy(simple_digraph):
    """clone_graph should not mutate the original graph when the copy changes."""
    copy_graph = clone_graph(simple_digraph)

    assert isinstance(copy_graph, nx.DiGraph)
    assert id(copy_graph) != id(simple_digraph)

    copy_graph.nodes[0]["label"] = "clone"
    assert "label" not in simple_digraph.nodes[0]
