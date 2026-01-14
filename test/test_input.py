"""
Test suite for data loading strategies.

Tests cover:
- CSVFilesStrategy (csv_loader.py)
- NetworkXDirectStrategy (networkx_loader.py)
- VoltageAwareStrategy (va_loader.py)
- All exception paths and edge cases
"""

import os
import tempfile
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

from npap.exceptions import DataLoadingError
from npap.input.csv_loader import CSVFilesStrategy
from npap.input.networkx_loader import NetworkXDirectStrategy
from npap.input.va_loader import VoltageAwareStrategy

# =============================================================================
# FIXTURES FOR CSV FILES
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def simple_node_csv(temp_dir):
    """Create a simple nodes CSV file."""
    filepath = os.path.join(temp_dir, "nodes.csv")
    df = pd.DataFrame(
        {
            "node_id": [0, 1, 2, 3],
            "lat": [0.0, 0.0, 1.0, 1.0],
            "lon": [0.0, 1.0, 0.0, 1.0],
            "demand": [10.0, 20.0, 30.0, 40.0],
        }
    )
    df.to_csv(filepath, index=False)
    return filepath


@pytest.fixture
def simple_edge_csv(temp_dir):
    """Create a simple edges CSV file."""
    filepath = os.path.join(temp_dir, "edges.csv")
    df = pd.DataFrame(
        {
            "from": [0, 0, 1, 2],
            "to": [1, 2, 3, 3],
            "x": [0.1, 0.2, 0.15, 0.25],
            "length": [100, 150, 120, 180],
        }
    )
    df.to_csv(filepath, index=False)
    return filepath


@pytest.fixture
def parallel_edge_csv(temp_dir):
    """Create edges CSV with parallel edges."""
    filepath = os.path.join(temp_dir, "parallel_edges.csv")
    df = pd.DataFrame(
        {
            "from": [0, 0, 1, 2],  # Two edges from 0->1
            "to": [1, 1, 2, 3],
            "x": [0.1, 0.2, 0.15, 0.25],
        }
    )
    df.to_csv(filepath, index=False)
    return filepath


@pytest.fixture
def empty_node_csv(temp_dir):
    """Create an empty nodes CSV file."""
    filepath = os.path.join(temp_dir, "empty_nodes.csv")
    df = pd.DataFrame(columns=["node_id", "lat", "lon"])
    df.to_csv(filepath, index=False)
    return filepath


@pytest.fixture
def empty_edge_csv(temp_dir):
    """Create an empty edges CSV file."""
    filepath = os.path.join(temp_dir, "empty_edges.csv")
    df = pd.DataFrame(columns=["from", "to", "x"])
    df.to_csv(filepath, index=False)
    return filepath


@pytest.fixture
def invalid_edge_csv(temp_dir, simple_node_csv):
    """Create edges CSV referencing non-existent nodes."""
    filepath = os.path.join(temp_dir, "invalid_edges.csv")
    df = pd.DataFrame(
        {
            "from": [0, 999],  # 999 doesn't exist
            "to": [1, 1],
            "x": [0.1, 0.2],
        }
    )
    df.to_csv(filepath, index=False)
    return filepath


@pytest.fixture
def bus0_bus1_edge_csv(temp_dir):
    """Create edges CSV with bus0/bus1 columns."""
    filepath = os.path.join(temp_dir, "edges_bus.csv")
    df = pd.DataFrame({"bus0": [0, 1], "bus1": [1, 2], "x": [0.1, 0.2]})
    df.to_csv(filepath, index=False)
    return filepath


@pytest.fixture
def source_target_edge_csv(temp_dir):
    """Create edges CSV with source/target columns."""
    filepath = os.path.join(temp_dir, "edges_source.csv")
    df = pd.DataFrame({"source": [0, 1], "target": [1, 2], "x": [0.1, 0.2]})
    df.to_csv(filepath, index=False)
    return filepath


# =============================================================================
# CSV LOADER TESTS
# =============================================================================


class TestCSVFilesStrategy:
    """Tests for CSVFilesStrategy."""

    # -------------------------------------------------------------------------
    # Validation Tests
    # -------------------------------------------------------------------------

    def test_validate_missing_node_file(self, temp_dir, simple_edge_csv):
        """Test validation fails when node_file is missing."""
        strategy = CSVFilesStrategy()

        with pytest.raises(DataLoadingError, match="Missing required parameters"):
            strategy.validate_inputs(edge_file=simple_edge_csv)

    def test_validate_missing_edge_file(self, temp_dir, simple_node_csv):
        """Test validation fails when edge_file is missing."""
        strategy = CSVFilesStrategy()

        with pytest.raises(DataLoadingError, match="Missing required parameters"):
            strategy.validate_inputs(node_file=simple_node_csv)

    def test_validate_nonexistent_file(self, temp_dir):
        """Test validation fails for non-existent files."""
        strategy = CSVFilesStrategy()

        with pytest.raises(DataLoadingError, match="File not found"):
            strategy.validate_inputs(
                node_file="/nonexistent/path/nodes.csv",
                edge_file="/nonexistent/path/edges.csv",
            )

    def test_validate_success(self, simple_node_csv, simple_edge_csv):
        """Test successful validation."""
        strategy = CSVFilesStrategy()
        result = strategy.validate_inputs(node_file=simple_node_csv, edge_file=simple_edge_csv)
        assert result is True

    # -------------------------------------------------------------------------
    # Load Tests
    # -------------------------------------------------------------------------

    def test_load_basic(self, simple_node_csv, simple_edge_csv):
        """Test basic CSV loading."""
        strategy = CSVFilesStrategy()
        graph = strategy.load(simple_node_csv, simple_edge_csv)

        assert isinstance(graph, nx.DiGraph)
        assert len(list(graph.nodes())) == 4
        assert len(graph.edges()) == 4

    def test_load_with_parallel_edges(self, simple_node_csv, parallel_edge_csv):
        """Test loading CSV with parallel edges creates MultiDiGraph."""
        strategy = CSVFilesStrategy()

        with pytest.warns(UserWarning, match="Parallel edges detected"):
            graph = strategy.load(simple_node_csv, parallel_edge_csv)

        assert isinstance(graph, nx.MultiDiGraph)

    def test_load_empty_node_file(self, empty_node_csv, simple_edge_csv):
        """Test loading empty node file raises error."""
        strategy = CSVFilesStrategy()

        with pytest.raises(DataLoadingError, match="Node file is empty"):
            strategy.load(empty_node_csv, simple_edge_csv)

    def test_load_empty_edge_file(self, simple_node_csv, empty_edge_csv):
        """Test loading empty edge file raises error."""
        strategy = CSVFilesStrategy()

        with pytest.raises(DataLoadingError, match="Edge file is empty"):
            strategy.load(simple_node_csv, empty_edge_csv)

    def test_load_invalid_edge_references(self, simple_node_csv, invalid_edge_csv):
        """Test loading edges referencing non-existent nodes."""
        strategy = CSVFilesStrategy()

        with pytest.raises(DataLoadingError, match="non-existent node"):
            strategy.load(simple_node_csv, invalid_edge_csv)

    def test_load_with_custom_delimiter(self, temp_dir):
        """Test loading CSV with custom delimiter."""
        # Create semicolon-delimited files
        node_file = os.path.join(temp_dir, "nodes_semicolon.csv")
        edge_file = os.path.join(temp_dir, "edges_semicolon.csv")

        pd.DataFrame({"node_id": [0, 1], "lat": [0.0, 1.0], "lon": [0.0, 1.0]}).to_csv(
            node_file, index=False, sep=";"
        )

        pd.DataFrame({"from": [0], "to": [1], "x": [0.1]}).to_csv(edge_file, index=False, sep=";")

        strategy = CSVFilesStrategy()
        graph = strategy.load(node_file, edge_file, delimiter=";")

        assert len(list(graph.nodes())) == 2

    def test_load_with_custom_columns(self, temp_dir):
        """Test loading with custom column names."""
        node_file = os.path.join(temp_dir, "custom_nodes.csv")
        edge_file = os.path.join(temp_dir, "custom_edges.csv")

        pd.DataFrame({"my_id": [0, 1], "lat": [0.0, 1.0], "lon": [0.0, 1.0]}).to_csv(
            node_file, index=False
        )

        pd.DataFrame({"start_node": [0], "end_node": [1], "x": [0.1]}).to_csv(
            edge_file, index=False
        )

        strategy = CSVFilesStrategy()
        graph = strategy.load(
            node_file,
            edge_file,
            node_id_col="my_id",
            edge_from_col="start_node",
            edge_to_col="end_node",
        )

        assert len(list(graph.nodes())) == 2

    def test_load_invalid_node_id_column(self, simple_node_csv, simple_edge_csv):
        """Test error when specified node ID column doesn't exist."""
        strategy = CSVFilesStrategy()

        with pytest.raises(DataLoadingError, match="Node ID column.*not found"):
            strategy.load(simple_node_csv, simple_edge_csv, node_id_col="nonexistent")

    def test_load_invalid_edge_from_column(self, simple_node_csv, simple_edge_csv):
        """Test error when edge 'from' column doesn't exist."""
        strategy = CSVFilesStrategy()

        with pytest.raises(DataLoadingError, match="Edge 'from' column.*not found"):
            strategy.load(simple_node_csv, simple_edge_csv, edge_from_col="nonexistent")

    def test_load_invalid_edge_to_column(self, simple_node_csv, simple_edge_csv):
        """Test error when edge 'to' column doesn't exist."""
        strategy = CSVFilesStrategy()

        with pytest.raises(DataLoadingError, match="Edge 'to' column.*not found"):
            strategy.load(simple_node_csv, simple_edge_csv, edge_to_col="nonexistent")

    # -------------------------------------------------------------------------
    # Column Detection Tests
    # -------------------------------------------------------------------------

    def test_detect_bus0_bus1_columns(self, simple_node_csv, bus0_bus1_edge_csv):
        """Test automatic detection of bus0/bus1 columns."""
        # Extend nodes to include node 2
        node_file = Path(simple_node_csv).parent / "nodes3.csv"
        pd.DataFrame({"node_id": [0, 1, 2], "lat": [0.0, 1.0, 2.0], "lon": [0.0, 1.0, 2.0]}).to_csv(
            node_file, index=False
        )

        strategy = CSVFilesStrategy()
        graph = strategy.load(str(node_file), bus0_bus1_edge_csv)

        assert graph.has_edge(0, 1)
        assert graph.has_edge(1, 2)

    def test_detect_source_target_columns(self, temp_dir, source_target_edge_csv):
        """Test automatic detection of source/target columns."""
        node_file = os.path.join(temp_dir, "nodes3.csv")
        pd.DataFrame({"node_id": [0, 1, 2], "lat": [0.0, 1.0, 2.0], "lon": [0.0, 1.0, 2.0]}).to_csv(
            node_file, index=False
        )

        strategy = CSVFilesStrategy()
        graph = strategy.load(node_file, source_target_edge_csv)

        assert graph.has_edge(0, 1)

    def test_detect_id_column_fallback(self, temp_dir):
        """Test ID column detection falls back to first column."""
        node_file = os.path.join(temp_dir, "nodes_custom.csv")
        edge_file = os.path.join(temp_dir, "edges_custom.csv")

        # First column will be used as ID
        pd.DataFrame({"custom_first_col": [0, 1], "lat": [0.0, 1.0], "lon": [0.0, 1.0]}).to_csv(
            node_file, index=False
        )

        # Edge file with non-standard columns - will use first two columns
        pd.DataFrame({"col_a": [0], "col_b": [1], "x": [0.1]}).to_csv(edge_file, index=False)

        strategy = CSVFilesStrategy()
        graph = strategy.load(node_file, edge_file, edge_from_col="col_a", edge_to_col="col_b")

        assert len(list(graph.nodes())) == 2


# =============================================================================
# NETWORKX LOADER TESTS
# =============================================================================


class TestNetworkXDirectStrategy:
    """Tests for NetworkXDirectStrategy."""

    # -------------------------------------------------------------------------
    # Validation Tests
    # -------------------------------------------------------------------------

    def test_validate_missing_graph(self):
        """Test validation fails when graph is not provided."""
        strategy = NetworkXDirectStrategy()

        with pytest.raises(DataLoadingError, match="Missing required parameter"):
            strategy.validate_inputs()

    def test_validate_invalid_type(self):
        """Test validation fails for non-NetworkX objects."""
        strategy = NetworkXDirectStrategy()

        with pytest.raises(DataLoadingError, match="must be a NetworkX Graph"):
            strategy.validate_inputs(graph="not a graph")

    def test_validate_empty_graph(self):
        """Test validation fails for empty graph."""
        strategy = NetworkXDirectStrategy()

        with pytest.raises(DataLoadingError, match="no nodes"):
            strategy.validate_inputs(graph=nx.Graph())

    def test_validate_success(self, simple_digraph):
        """Test successful validation."""
        strategy = NetworkXDirectStrategy()
        result = strategy.validate_inputs(graph=simple_digraph)
        assert result is True

    # -------------------------------------------------------------------------
    # Load Tests - Different Graph Types
    # -------------------------------------------------------------------------

    def test_load_digraph(self, simple_digraph):
        """Test loading DiGraph returns copy."""
        strategy = NetworkXDirectStrategy()
        result = strategy.load(simple_digraph)

        assert isinstance(result, nx.DiGraph)
        assert result is not simple_digraph  # Should be a copy
        assert set(result.nodes()) == set(simple_digraph.nodes())

    def test_load_graph_bidirectional(self):
        """Test loading undirected Graph creates bidirectional edges."""
        G = nx.Graph()
        G.add_node(0, lat=0.0, lon=0.0)
        G.add_node(1, lat=1.0, lon=1.0)
        G.add_edge(0, 1, x=0.1)

        strategy = NetworkXDirectStrategy()
        result = strategy.load(G, bidirectional=True)

        assert isinstance(result, nx.DiGraph)
        assert result.has_edge(0, 1)
        assert result.has_edge(1, 0)  # Bidirectional

    def test_load_graph_unidirectional(self):
        """Test loading undirected Graph with bidirectional=False."""
        G = nx.Graph()
        G.add_node(0, lat=0.0, lon=0.0)
        G.add_node(1, lat=1.0, lon=1.0)
        G.add_edge(0, 1, x=0.1)

        strategy = NetworkXDirectStrategy()
        result = strategy.load(G, bidirectional=False)

        assert isinstance(result, nx.DiGraph)
        # Only one direction
        assert len(result.edges()) == 1

    def test_load_multigraph(self):
        """Test loading MultiGraph creates MultiDiGraph."""
        G = nx.MultiGraph()
        G.add_node(0, lat=0.0, lon=0.0)
        G.add_node(1, lat=1.0, lon=1.0)
        G.add_edge(0, 1, key=0, x=0.1)
        G.add_edge(0, 1, key=1, x=0.2)  # Parallel edge

        strategy = NetworkXDirectStrategy()

        with pytest.warns(UserWarning, match="Parallel edges detected"):
            result = strategy.load(G)

        assert isinstance(result, nx.MultiDiGraph)

    def test_load_multidigraph(self):
        """Test loading MultiDiGraph returns copy."""
        G = nx.MultiDiGraph()
        G.add_node(0, lat=0.0, lon=0.0)
        G.add_node(1, lat=1.0, lon=1.0)
        G.add_edge(0, 1, key=0, x=0.1)

        strategy = NetworkXDirectStrategy()

        with pytest.warns(UserWarning, match="Parallel edges detected"):
            result = strategy.load(G)

        assert isinstance(result, nx.MultiDiGraph)
        assert result is not G

    def test_load_preserves_attributes(self):
        """Test that node and edge attributes are preserved."""
        G = nx.Graph()
        G.add_node(0, lat=0.0, lon=0.0, custom_attr="value")
        G.add_node(1, lat=1.0, lon=1.0)
        G.add_edge(0, 1, x=0.1, length=100)

        strategy = NetworkXDirectStrategy()
        result = strategy.load(G)

        assert result.nodes[0]["custom_attr"] == "value"
        assert result[0][1]["x"] == 0.1
        assert result[0][1]["length"] == 100


# =============================================================================
# VOLTAGE-AWARE LOADER TESTS
# =============================================================================


@pytest.fixture
def va_test_files(temp_dir):
    """Create all required files for VoltageAwareStrategy testing."""
    files = {"node_file": os.path.join(temp_dir, "buses.csv")}

    # Nodes
    pd.DataFrame(
        {
            "bus_id": ["bus_0", "bus_1", "bus_2", "bus_3"],
            "lat": [0.0, 0.5, 5.0, 5.5],
            "lon": [0.0, 0.5, 5.0, 5.5],
            "voltage": [220.0, 220.0, 380.0, 380.0],
        }
    ).to_csv(files["node_file"], index=False)

    # Lines
    files["line_file"] = os.path.join(temp_dir, "lines.csv")
    pd.DataFrame(
        {
            "bus0": ["bus_0", "bus_2"],
            "bus1": ["bus_1", "bus_3"],
            "x": [0.1, 0.08],
            "voltage": [220.0, 380.0],
        }
    ).to_csv(files["line_file"], index=False)

    # Transformers
    files["transformer_file"] = os.path.join(temp_dir, "transformers.csv")
    pd.DataFrame(
        {
            "bus0": ["bus_1"],
            "bus1": ["bus_2"],
            "x": [0.05],
            "primary_voltage": [220.0],
            "secondary_voltage": [380.0],
        }
    ).to_csv(files["transformer_file"], index=False)

    # Converters (empty for basic test)
    files["converter_file"] = os.path.join(temp_dir, "converters.csv")
    pd.DataFrame(columns=["converter_id", "bus0", "bus1", "voltage"]).to_csv(
        files["converter_file"], index=False
    )

    # Links (empty for basic test)
    files["link_file"] = os.path.join(temp_dir, "links.csv")
    pd.DataFrame(columns=["link_id", "bus0", "bus1", "voltage"]).to_csv(
        files["link_file"], index=False
    )

    return files


@pytest.fixture
def va_test_files_with_dc(temp_dir):
    """Create files including DC links for testing."""
    files = {"node_file": os.path.join(temp_dir, "buses.csv")}

    # Nodes - two separate AC networks
    pd.DataFrame(
        {
            "bus_id": ["bus_0", "bus_1", "bus_2", "bus_3"],
            "lat": [0.0, 0.5, 50.0, 50.5],
            "lon": [0.0, 0.5, 50.0, 50.5],
            "voltage": [220.0, 220.0, 220.0, 220.0],
        }
    ).to_csv(files["node_file"], index=False)

    # Lines - only within islands
    files["line_file"] = os.path.join(temp_dir, "lines.csv")
    pd.DataFrame(
        {
            "bus0": ["bus_0", "bus_2"],
            "bus1": ["bus_1", "bus_3"],
            "x": [0.1, 0.1],
            "voltage": [220.0, 220.0],
        }
    ).to_csv(files["line_file"], index=False)

    # No transformers
    files["transformer_file"] = os.path.join(temp_dir, "transformers.csv")
    pd.DataFrame(columns=["bus0", "bus1", "x", "primary_voltage", "secondary_voltage"]).to_csv(
        files["transformer_file"], index=False
    )

    # Converters connecting buses to DC side
    files["converter_file"] = os.path.join(temp_dir, "converters.csv")
    pd.DataFrame(
        {
            "converter_id": ["conv_0", "conv_1"],
            "bus0": ["dc_0", "dc_1"],  # DC side
            "bus1": ["bus_1", "bus_2"],  # AC side
            "voltage": [400.0, 400.0],
        }
    ).to_csv(files["converter_file"], index=False)

    # DC link connecting the converters
    files["link_file"] = os.path.join(temp_dir, "links.csv")
    pd.DataFrame(
        {"link_id": ["link_0"], "bus0": ["dc_0"], "bus1": ["dc_1"], "voltage": [400.0]}
    ).to_csv(files["link_file"], index=False)

    return files


@pytest.mark.filterwarnings("ignore:Converters file is empty. No DC links will be created.")
@pytest.mark.filterwarnings("ignore:Links file is empty. No DC links will be created.")
class TestVoltageAwareStrategy:
    """Tests for VoltageAwareStrategy."""

    # -------------------------------------------------------------------------
    # Validation Tests
    # -------------------------------------------------------------------------

    def test_validate_missing_files(self, temp_dir):
        """Test validation fails when required files are missing."""
        strategy = VoltageAwareStrategy()

        with pytest.raises(DataLoadingError, match="Missing required parameters"):
            strategy.validate_inputs(node_file=os.path.join(temp_dir, "nodes.csv"))

    def test_validate_nonexistent_file(self, temp_dir):
        """Test validation fails for non-existent files."""
        strategy = VoltageAwareStrategy()

        with pytest.raises(DataLoadingError, match="File not found"):
            strategy.validate_inputs(
                node_file="/nonexistent/nodes.csv",
                line_file="/nonexistent/lines.csv",
                transformer_file="/nonexistent/trafos.csv",
                converter_file="/nonexistent/converters.csv",
                link_file="/nonexistent/links.csv",
            )

    def test_validate_success(self, va_test_files):
        """Test successful validation."""
        strategy = VoltageAwareStrategy()
        result = strategy.validate_inputs(**va_test_files)
        assert result is True

    # -------------------------------------------------------------------------
    # Basic Load Tests
    # -------------------------------------------------------------------------

    def test_load_basic(self, va_test_files):
        """Test basic voltage-aware loading."""
        strategy = VoltageAwareStrategy()
        graph = strategy.load(**va_test_files)

        assert isinstance(graph, (nx.DiGraph, nx.MultiDiGraph))
        assert len(list(graph.nodes())) == 4

        # Check dc_island attribute was added
        for node in graph.nodes():
            assert "dc_island" in graph.nodes[node]

    def test_load_with_dc_links(self, va_test_files_with_dc):
        """Test loading with DC links."""
        strategy = VoltageAwareStrategy()
        with pytest.warns(UserWarning, match="Transformers file is empty"):
            graph = strategy.load(**va_test_files_with_dc)

        # Should have 4 nodes
        assert len(list(graph.nodes())) == 4

        # Check DC link was added
        dc_link_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get("type") == "dc_link"]
        assert len(dc_link_edges) == 1

    def test_load_detects_dc_islands(self, va_test_files_with_dc):
        """Test that DC islands are correctly detected."""
        strategy = VoltageAwareStrategy()
        with pytest.warns(UserWarning, match="Transformers file is empty"):
            graph = strategy.load(**va_test_files_with_dc)

        # Before DC links, bus_0/bus_1 and bus_2/bus_3 are separate islands
        island_0 = graph.nodes["bus_0"].get("dc_island")
        island_1 = graph.nodes["bus_1"].get("dc_island")
        island_2 = graph.nodes["bus_2"].get("dc_island")
        island_3 = graph.nodes["bus_3"].get("dc_island")

        # bus_0 and bus_1 should be in same island
        assert island_0 == island_1
        # bus_2 and bus_3 should be in same island
        assert island_2 == island_3
        # But the two pairs should be in different islands
        assert island_0 != island_2

    # -------------------------------------------------------------------------
    # Empty File Tests
    # -------------------------------------------------------------------------

    def test_load_empty_lines_file(self, temp_dir, va_test_files):
        """Test loading with empty lines file."""
        # Create empty lines file
        empty_lines = os.path.join(temp_dir, "empty_lines.csv")
        pd.DataFrame(columns=["bus0", "bus1", "x"]).to_csv(empty_lines, index=False)

        va_test_files["line_file"] = empty_lines

        strategy = VoltageAwareStrategy()

        # EXPECTED: "Lines file is empty" AND "Found isolated nodes"
        # We nest them to catch both expected warnings
        with pytest.warns(UserWarning, match="Lines file is empty"):
            with pytest.warns(UserWarning, match="isolated node"):
                graph = strategy.load(**va_test_files)

        # Graph should still load with just transformers
        assert isinstance(graph, (nx.DiGraph, nx.MultiDiGraph))

    def test_load_empty_transformers_file(self, temp_dir, va_test_files):
        """Test loading with empty transformers file."""
        empty_trafos = os.path.join(temp_dir, "empty_trafos.csv")
        pd.DataFrame(columns=["bus0", "bus1", "x", "primary_voltage", "secondary_voltage"]).to_csv(
            empty_trafos, index=False
        )

        va_test_files["transformer_file"] = empty_trafos

        strategy = VoltageAwareStrategy()

        with pytest.warns(UserWarning, match="Transformers file is empty"):
            graph = strategy.load(**va_test_files)

        assert isinstance(graph, (nx.DiGraph, nx.MultiDiGraph))

    # -------------------------------------------------------------------------
    # Validation Error Tests
    # -------------------------------------------------------------------------

    def test_load_missing_line_columns(self, temp_dir, va_test_files):
        """Test error when lines file missing required columns."""
        bad_lines = os.path.join(temp_dir, "bad_lines.csv")
        pd.DataFrame(
            {
                "from": [0],  # Wrong column names
                "to": [1],
            }
        ).to_csv(bad_lines, index=False)

        va_test_files["line_file"] = bad_lines

        strategy = VoltageAwareStrategy()

        with pytest.raises(DataLoadingError, match="Lines file missing required columns"):
            strategy.load(**va_test_files)

    def test_load_missing_transformer_columns(self, temp_dir, va_test_files):
        """Test error when transformers file missing required columns."""
        bad_trafos = os.path.join(temp_dir, "bad_trafos.csv")
        pd.DataFrame(
            {
                "bus0": ["bus_0"],
                "bus1": ["bus_1"],
                "x": [0.1],
                # Missing primary_voltage, secondary_voltage
            }
        ).to_csv(bad_trafos, index=False)

        va_test_files["transformer_file"] = bad_trafos

        strategy = VoltageAwareStrategy()

        with pytest.raises(DataLoadingError, match="Transformers file missing required columns"):
            strategy.load(**va_test_files)

    def test_load_invalid_transformer_voltage(self, temp_dir, va_test_files):
        """Test error for transformer with invalid voltage."""
        bad_trafos = os.path.join(temp_dir, "bad_voltage_trafos.csv")
        pd.DataFrame(
            {
                "bus0": ["bus_0"],
                "bus1": ["bus_1"],
                "x": [0.1],
                "primary_voltage": [-220.0],  # Negative!
                "secondary_voltage": [380.0],
            }
        ).to_csv(bad_trafos, index=False)

        va_test_files["transformer_file"] = bad_trafos

        strategy = VoltageAwareStrategy()

        with pytest.raises(DataLoadingError, match="positive primary and secondary voltages"):
            strategy.load(**va_test_files)

    def test_load_edge_references_invalid_node(self, temp_dir, va_test_files):
        """Test error when edge references non-existent node."""
        bad_lines = os.path.join(temp_dir, "bad_refs_lines.csv")
        pd.DataFrame(
            {
                "bus0": ["bus_0", "nonexistent"],
                "bus1": ["bus_1", "bus_2"],
                "x": [0.1, 0.2],
            }
        ).to_csv(bad_lines, index=False)

        va_test_files["line_file"] = bad_lines

        strategy = VoltageAwareStrategy()

        with pytest.raises(DataLoadingError, match="references non-existent node"):
            strategy.load(**va_test_files)

    # -------------------------------------------------------------------------
    # DC Link Edge Cases
    # -------------------------------------------------------------------------

    def test_load_skipped_dc_links(self, temp_dir):
        """Test warning when DC links are skipped due to missing converters."""
        files = {"node_file": os.path.join(temp_dir, "buses.csv")}

        pd.DataFrame(
            {
                "bus_id": ["bus_0", "bus_1"],
                "lat": [0.0, 1.0],
                "lon": [0.0, 1.0],
                "voltage": [220.0, 220.0],
            }
        ).to_csv(files["node_file"], index=False)

        files["line_file"] = os.path.join(temp_dir, "lines.csv")
        pd.DataFrame({"bus0": ["bus_0"], "bus1": ["bus_1"], "x": [0.1]}).to_csv(
            files["line_file"], index=False
        )

        files["transformer_file"] = os.path.join(temp_dir, "trafos.csv")
        pd.DataFrame(columns=["bus0", "bus1", "x", "primary_voltage", "secondary_voltage"]).to_csv(
            files["transformer_file"], index=False
        )

        # Converters that don't match links
        files["converter_file"] = os.path.join(temp_dir, "converters.csv")
        pd.DataFrame(
            {
                "converter_id": ["conv_0"],
                "bus0": ["dc_wrong"],  # Won't match link
                "bus1": ["bus_0"],
                "voltage": [400.0],
            }
        ).to_csv(files["converter_file"], index=False)

        files["link_file"] = os.path.join(temp_dir, "links.csv")
        pd.DataFrame(
            {
                "link_id": ["link_0"],
                "bus0": ["dc_0"],  # No matching converter
                "bus1": ["dc_1"],
                "voltage": [400.0],
            }
        ).to_csv(files["link_file"], index=False)

        strategy = VoltageAwareStrategy()

        # We anticipate "Transformers file is empty" AND "DC link skipped"
        with pytest.warns(UserWarning, match="Transformers file is empty"):
            with pytest.warns(UserWarning, match="DC link.*skipped"):
                graph = strategy.load(**files)

        # Graph should still load, just without DC links
        assert isinstance(graph, nx.DiGraph)


# =============================================================================
# EXCEPTION TESTS
# =============================================================================


class TestDataLoadingExceptions:
    """Tests for DataLoadingError exception."""

    def test_exception_with_strategy(self):
        """Test DataLoadingError stores strategy info."""
        error = DataLoadingError("Test error", strategy="test_strategy")
        assert error.strategy == "test_strategy"
        assert str(error) == "Test error"

    def test_exception_with_details(self):
        """Test DataLoadingError stores details."""
        details = {"key": "value", "count": 42}
        error = DataLoadingError("Test error", details=details)
        assert error.details == details

    def test_exception_default_details(self):
        """Test DataLoadingError default details is empty dict."""
        error = DataLoadingError("Test error")
        assert error.details == {}


class TestStrategyNotFoundError:
    """Tests for StrategyNotFoundError exception."""

    def test_exception_basic(self):
        """Test basic StrategyNotFoundError creation."""
        from npap.exceptions import StrategyNotFoundError

        error = StrategyNotFoundError("unknown", "partitioning")
        assert "unknown" in str(error)
        assert "partitioning" in str(error)

    def test_exception_with_available(self):
        """Test StrategyNotFoundError with available strategies."""
        from npap.exceptions import StrategyNotFoundError

        error = StrategyNotFoundError(
            "unknown", "partitioning", available_strategies=["kmeans", "kmedoids"]
        )
        assert "kmeans" in str(error)
        assert "kmedoids" in str(error)


class TestVisualizationError:
    """Tests for VisualizationError exception."""

    def test_exception_basic(self):
        """Test basic VisualizationError creation."""
        from npap.exceptions import VisualizationError

        error = VisualizationError("Plot failed")
        assert str(error) == "Plot failed"
        assert error.details == {}

    def test_exception_with_details(self):
        """Test VisualizationError with details."""
        from npap.exceptions import VisualizationError

        error = VisualizationError("Plot failed", details={"style": "invalid"})
        assert error.details["style"] == "invalid"


class TestElectricalCalculationError:
    """Tests for ElectricalCalculationError exception."""

    def test_exception_basic(self):
        """Test basic ElectricalCalculationError creation."""
        from npap.exceptions import ElectricalCalculationError

        error = ElectricalCalculationError("Calculation failed", calculation_type="kron")
        assert error.calculation_type == "kron"
