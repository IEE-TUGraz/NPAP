# SPDX-FileCopyrightText: Contributors to NPAP
# SPDX-License-Identifier: MIT

"""Tests for the visualization module output behaviour (HTML export / renderer)."""

import networkx as nx
import plotly.graph_objects as go
import plotly.io as pio
import pytest

import npap
from npap.visualization import (
    DEFAULT_HTML_FILENAME,
    NetworkPlotter,
    PlotConfig,
    _slugify,
    deliver_figure,
    plot_network,
    resolve_html_path,
)


@pytest.fixture
def geo_graph():
    """Return a small geo-referenced graph suitable for every plot style."""
    graph = nx.DiGraph()
    graph.add_node("bus_1", lat=47.0, lon=15.0, voltage=380)
    graph.add_node("bus_2", lat=47.1, lon=15.1, voltage=380)
    graph.add_node("bus_3", lat=47.2, lon=15.0, voltage=220)
    graph.add_edge("bus_1", "bus_2", x=0.01, type="line")
    graph.add_edge("bus_2", "bus_3", x=0.02, type="trafo")
    return graph


@pytest.fixture
def figure():
    """Return a trivial Plotly figure."""
    return go.Figure()


class TestPathHelpers:
    """Cover filename derivation for the exported HTML."""

    def test_slugify_normalises_title(self):
        assert _slugify("European HV Grid — Raw") == "european_hv_grid_raw"

    def test_slugify_falls_back_when_empty(self):
        assert _slugify("///") == "npap_network"

    def test_default_filename_without_title(self, tmp_path):
        assert resolve_html_path(tmp_path).name == DEFAULT_HTML_FILENAME

    def test_filename_derived_from_title(self, tmp_path):
        assert resolve_html_path(tmp_path, title="My Network").name == "my_network.html"

    def test_explicit_filename_gets_html_suffix(self, tmp_path):
        assert resolve_html_path(tmp_path, filename="grid").name == "grid.html"

    def test_explicit_filename_suffix_preserved(self, tmp_path):
        assert resolve_html_path(tmp_path, filename="grid.html").name == "grid.html"

    def test_defaults_to_working_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert resolve_html_path().parent == tmp_path.resolve()


class TestDeliverFigure:
    """Cover the export/display decision logic."""

    def test_writes_html_outside_notebook(self, figure, tmp_path):
        written = deliver_figure(figure, output_dir=tmp_path, filename="out")
        assert written is not None
        assert written.exists()
        assert written.read_text(encoding="utf-8").lstrip().startswith("<html")

    def test_no_html_inside_notebook(self, figure, tmp_path, monkeypatch):
        monkeypatch.setattr("npap.visualization._in_notebook", lambda: True)
        assert deliver_figure(figure, output_dir=tmp_path) is None
        assert list(tmp_path.iterdir()) == []

    def test_save_html_false_overrides_environment(self, figure, tmp_path):
        assert deliver_figure(figure, save_html=False, output_dir=tmp_path) is None
        assert list(tmp_path.iterdir()) == []

    def test_save_html_true_inside_notebook(self, figure, tmp_path, monkeypatch):
        monkeypatch.setattr("npap.visualization._in_notebook", lambda: True)
        assert deliver_figure(figure, save_html=True, output_dir=tmp_path) is not None

    def test_creates_missing_output_directory(self, figure, tmp_path):
        target = tmp_path / "nested" / "figures"
        written = deliver_figure(figure, output_dir=target, save_html=True)
        assert written.exists()

    def test_renderer_none_does_not_display(self, figure, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(go.Figure, "show", lambda self, **kw: calls.append(kw))
        deliver_figure(figure, output_dir=tmp_path, save_html=True)
        assert calls == []

    def test_renderer_auto_delegates_to_plotly(self, figure, monkeypatch):
        calls = []
        monkeypatch.setattr(go.Figure, "show", lambda self, **kw: calls.append(kw))
        deliver_figure(figure, renderer="auto", save_html=False)
        assert len(calls) == 1
        assert "renderer" not in calls[0]

    def test_explicit_renderer_forwarded(self, figure, monkeypatch):
        calls = []
        monkeypatch.setattr(go.Figure, "show", lambda self, **kw: calls.append(kw))
        deliver_figure(figure, renderer="browser", save_html=False)
        assert calls[0]["renderer"] == "browser"

    def test_does_not_mutate_global_renderer(self, figure, monkeypatch):
        """Regression: NPAP must never change Plotly's global renderer default."""
        monkeypatch.setattr(go.Figure, "show", lambda self, **kw: None)
        before = pio.renderers.default
        deliver_figure(figure, renderer="browser", save_html=False)
        assert pio.renderers.default == before


class TestPlotNetwork:
    """Cover the public plotting entry points."""

    @pytest.mark.parametrize("style", ["simple", "voltage_aware", "clustered"])
    def test_returns_figure_for_every_style(self, geo_graph, tmp_path, style):
        partition_map = {0: ["bus_1", "bus_2"], 1: ["bus_3"]} if style == "clustered" else None
        fig = plot_network(
            geo_graph,
            style=style,
            partition_map=partition_map,
            output_dir=tmp_path,
            save_html=False,
        )
        assert isinstance(fig, go.Figure)

    def test_exports_html_by_default(self, geo_graph, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        plot_network(geo_graph, title="Grid Overview")
        assert (tmp_path / "grid_overview.html").exists()

    def test_unknown_style_raises(self, geo_graph):
        with pytest.raises(ValueError, match="Unknown plot style"):
            plot_network(geo_graph, style="nope", save_html=False)

    def test_clustered_without_partition_raises(self, geo_graph):
        plotter = NetworkPlotter(geo_graph)
        with pytest.raises(ValueError, match="partition_map"):
            plotter.plot_clustered(save_html=False)

    def test_kwargs_override_config(self, geo_graph, tmp_path):
        fig = plot_network(
            geo_graph,
            config=PlotConfig(title="from config"),
            title="from kwargs",
            output_dir=tmp_path,
            save_html=False,
        )
        assert "from kwargs" in str(fig.layout.title.text)


class TestManagerPlotNetwork:
    """Cover the facade wrapper on PartitionAggregatorManager."""

    def test_returns_figure_without_side_effects(self, geo_graph, tmp_path):
        manager = npap.PartitionAggregatorManager()
        manager.load_data("networkx_direct", graph=geo_graph)
        fig = manager.plot_network(save_html=False)
        assert isinstance(fig, go.Figure)
        assert list(tmp_path.iterdir()) == []

    def test_honours_output_dir(self, geo_graph, tmp_path):
        manager = npap.PartitionAggregatorManager()
        manager.load_data("networkx_direct", graph=geo_graph)
        manager.plot_network(output_dir=tmp_path, filename="grid", save_html=True)
        assert (tmp_path / "grid.html").exists()

    def test_requires_loaded_graph(self):
        manager = npap.PartitionAggregatorManager()
        with pytest.raises(ValueError, match="No graph loaded"):
            manager.plot_network(save_html=False)

    def test_clustered_requires_partition(self, geo_graph):
        manager = npap.PartitionAggregatorManager()
        manager.load_data("networkx_direct", graph=geo_graph)
        with pytest.raises(ValueError, match="without partitioning"):
            manager.plot_network(style="clustered", save_html=False)
