import json

import networkx as nx

from npap.cli import (
    _dump_partition,
    _load_graph_from_file,
    _read_partition,
    _write_graph,
)


def test_dump_and_read_partition(tmp_path):
    partition = {0: ["A"], 1: ["B"]}
    output = tmp_path / "partition.json"

    _dump_partition(partition, str(output))
    loaded = _read_partition(str(output))

    assert output.exists()
    assert loaded == partition
    assert json.loads(output.read_text()) == {"0": ["A"], "1": ["B"]}


def test_write_graph_with_supported_formats(tmp_path, monkeypatch):
    graph = nx.DiGraph()
    graph.add_node("bus")
    graph.add_edge("bus", "bus")

    target = tmp_path / "grid.graphml"
    if not hasattr(nx, "write_gpickle"):
        monkeypatch.setattr(nx, "write_gpickle", nx.write_graphml, raising=False)

    _write_graph(graph, str(target))

    assert target.exists()
    assert isinstance(nx.read_graphml(target), nx.Graph)


def test_load_graph_from_file(tmp_path):
    graph = nx.DiGraph()
    graph.add_node("A")
    path = tmp_path / "grid.graphml"
    nx.write_graphml(graph, path)

    loaded = _load_graph_from_file(str(path), None)

    assert isinstance(loaded, nx.Graph)
    assert list(loaded.nodes()) == ["A"]
