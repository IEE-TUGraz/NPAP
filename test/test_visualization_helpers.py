import networkx as nx
import numpy as np
import plotly.graph_objects as go
import pytest

from npap.visualization import PlotPreset, plot_network, plot_reduced_matrices


def _build_simple_graph():
    graph = nx.DiGraph()
    graph.add_node("A", lat=0.0, lon=0.0)
    graph.add_node("B", lat=0.1, lon=0.1)
    graph.add_edge("A", "B")
    return graph


def test_plot_network_returns_figure():
    graph = _build_simple_graph()

    fig = plot_network(graph, style="simple", preset=PlotPreset.DENSE, show=False)

    assert isinstance(fig, go.Figure)
    assert fig.data


def test_plot_reduced_matrices_requires_diagnostics():
    graph = nx.DiGraph()

    with pytest.raises(ValueError):
        plot_reduced_matrices(graph, matrices=("ptdf",))

    graph.graph["reduced_ptdf"] = {
        "matrix": np.array([[1.0]]),
        "nodes": ["A"],
        "slack": None,
        "edges": [("A", "A")],
    }
    fig = plot_reduced_matrices(graph, matrices=("ptdf",), show=False)
    assert isinstance(fig, go.Figure)
