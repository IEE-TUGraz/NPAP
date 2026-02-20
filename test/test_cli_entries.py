import argparse
from types import SimpleNamespace

import networkx as nx


class DummyPartitionManager:
    def __init__(self):
        self.graph = nx.DiGraph()

    def load_data(self, *args, **kwargs):
        self.graph.add_node("A", lat=0.0, lon=0.0)
        self.graph.add_node("B", lat=1.0, lon=1.0)
        self.graph.add_edge("A", "B")
        return self.graph

    def partition(self, graph, strategy, n_clusters):
        from npap.interfaces import PartitionResult

        return PartitionResult(
            mapping={0: ["A"], 1: ["B"]},
            original_graph_hash="dummy",
            strategy_name=strategy,
            strategy_metadata={},
            n_clusters=n_clusters,
        )


class DummyAggregationManager:
    def aggregate(self, graph, partition_map, profile):
        agg = nx.DiGraph()
        agg.add_node("agg")
        return agg


def _monkeypatch_parse_args(monkeypatch, args):
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda self: args)


def test_cluster_entry_uses_partition_manager(monkeypatch):
    args = SimpleNamespace(
        node_file="nodes.csv",
        edge_file="edges.csv",
        graph_file=None,
        graph_format=None,
        no_bidirectional=False,
        delimiter=",",
        decimal=".",
        node_id_col=None,
        edge_from_col=None,
        edge_to_col=None,
        partition_strategy="adjacent_agglomerative",
        n_clusters=2,
        partition_output=None,
    )
    _monkeypatch_parse_args(monkeypatch, args)

    recorded = []

    def fake_dump(partition, output=None):
        recorded.append((partition.copy(), output))

    def fake_partition_manager():
        return DummyPartitionManager()

    monkeypatch.setattr("npap.cli.PartitionAggregatorManager", fake_partition_manager)
    monkeypatch.setattr("npap.cli._dump_partition", fake_dump)
    from npap.cli import cluster_entry

    cluster_entry()

    assert recorded
    mapping, output = recorded[0]
    assert mapping == {0: ["A"], 1: ["B"]}
    assert output is None


def test_aggregate_entry_writes_aggregated_graph(monkeypatch, tmp_path):
    args = SimpleNamespace(
        node_file=None,
        edge_file=None,
        graph_file=None,
        graph_format=None,
        no_bidirectional=False,
        delimiter=",",
        decimal=".",
        node_id_col=None,
        edge_from_col=None,
        edge_to_col=None,
        partition_file=str(tmp_path / "part.json"),
        mode="geographical",
        output=str(tmp_path / "agg.graphml"),
    )
    _monkeypatch_parse_args(monkeypatch, args)

    monkeypatch.setattr("npap.cli._load_graph", lambda args, manager: nx.DiGraph())
    monkeypatch.setattr("npap.cli._read_partition", lambda path: {0: ["A"]})

    def fake_aggregation_manager():
        return DummyAggregationManager()

    monkeypatch.setattr("npap.cli.AggregationManager", fake_aggregation_manager)
    monkeypatch.setattr("npap.cli.get_mode_profile", lambda mode: "profile")

    recorded = []

    def fake_write(graph, output, fmt=None):
        recorded.append((graph, output, fmt))

    monkeypatch.setattr("npap.cli._write_graph", fake_write)
    from npap.cli import aggregate_entry

    aggregate_entry()

    assert recorded
    written_graph, path, _ = recorded[0]
    assert path == args.output
    assert isinstance(written_graph, nx.Graph)


def test_plot_entry_exports_figure(monkeypatch, tmp_path):
    args = SimpleNamespace(
        node_file=None,
        edge_file=None,
        graph_file=None,
        graph_format=None,
        no_bidirectional=False,
        delimiter=",",
        decimal=".",
        node_id_col=None,
        edge_from_col=None,
        edge_to_col=None,
        aggregated_file="agg.graphml",
        aggregated_format="graphml",
        partition_file="part.json",
        style="clustered",
        preset="presentation",
        output=str(tmp_path / "figure.html"),
    )
    _monkeypatch_parse_args(monkeypatch, args)

    monkeypatch.setattr("npap.cli._load_graph", lambda args, manager: nx.DiGraph())
    monkeypatch.setattr("npap.cli._load_graph_from_file", lambda path, fmt: nx.DiGraph())
    monkeypatch.setattr("npap.cli._read_partition", lambda path: {0: ["A"]})
    monkeypatch.setattr("npap.cli.plot_network", lambda *args, **kwargs: object())

    recorded = []

    def fake_export(fig, path, **kwargs):
        recorded.append((fig, path, kwargs))

    monkeypatch.setattr("npap.cli.export_figure", fake_export)
    from npap.cli import plot_entry

    plot_entry()

    assert recorded
    _, path, _ = recorded[0]
    assert path == args.output


def test_diagnose_entry_exports_diagnostics(monkeypatch, tmp_path):
    args = SimpleNamespace(
        aggregated_file="agg.graphml",
        aggregated_format="graphml",
        matrix=["ptdf"],
        output=str(tmp_path / "diag.html"),
    )
    _monkeypatch_parse_args(monkeypatch, args)

    monkeypatch.setattr("npap.cli._load_graph_from_file", lambda path, fmt: nx.DiGraph())
    monkeypatch.setattr("npap.cli.plot_reduced_matrices", lambda graph, **kwargs: object())

    recorded = []

    def fake_export(fig, path, **kwargs):
        recorded.append((fig, path))

    monkeypatch.setattr("npap.cli.export_figure", fake_export)
    from npap.cli import diagnose_entry

    diagnose_entry()

    assert recorded
    _, path = recorded[0]
    assert path == args.output
