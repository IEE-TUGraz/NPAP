from pathlib import Path
from typing import Dict, List, Tuple, Any

import networkx as nx
import pandas as pd
from networkx import DiGraph, MultiDiGraph

from npap.exceptions import DataLoadingError
from npap.interfaces import DataLoadingStrategy, EdgeType


class VoltageAwareStrategy(DataLoadingStrategy):
    """
    Load graph from separate CSV files for nodes, lines, transformers, and DC links.

    This strategy creates a directed graph (DiGraph or MultiDiGraph) where:
    - Nodes represent electrical buses/substations
    - Edges represent transmission lines, transformers, or DC links
    - Edge direction follows defined bus0 -> bus1 convention

    DC Island Detection:
        After loading lines and transformers (before DC links), the loader detects
        disconnected components which represent separate DC islands. Each bus is
        assigned a 'dc_island' attribute indicating which island it belongs to.
        DC links then connect these islands.

    Edge Schema (unified for all types):
        - bus0, bus1: Source and target node IDs
        - type: Edge type ('line', 'trafo', 'dc_link')
        - primary_voltage: Voltage at bus0 side
        - secondary_voltage: Voltage at bus1 side
        - x: Reactance (where applicable)
        - Other type-specific attributes
    """

    REQUIRED_NODE_COLUMNS = ['bus_id']
    REQUIRED_LINE_COLUMNS = ['bus0', 'bus1', 'x']
    REQUIRED_TRANSFORMER_COLUMNS = ['bus0', 'bus1', 'x', 'primary_voltage', 'secondary_voltage']
    REQUIRED_CONVERTER_COLUMNS = ['converter_id', 'bus0', 'bus1', 'voltage']
    REQUIRED_LINK_COLUMNS = ['link_id', 'bus0', 'bus1', 'voltage']

    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate that all required CSV files are provided and exist.

        Args:
            **kwargs: Must include 'node_file', 'line_file', 'transformer_file',
                      'converter_file', and 'link_file'

        Returns:
            True if validation passes

        Raises:
            DataLoadingError: If validation fails
        """
        required_files = ['node_file', 'line_file', 'transformer_file',
                          'converter_file', 'link_file']
        missing = [f for f in required_files if f not in kwargs or kwargs[f] is None]

        if missing:
            raise DataLoadingError(
                f"Missing required parameters: {missing}. "
                "All files (including converter_file and link_file) are mandatory.",
                strategy="va_loader",
                details={
                    'required_params': required_files,
                    'provided_params': list(kwargs.keys())
                }
            )

        # Check if all files exist
        for param in required_files:
            file_path = Path(kwargs[param])
            if not file_path.exists():
                raise DataLoadingError(
                    f"File not found: {file_path}",
                    strategy="va_loader",
                    details={'missing_file': str(file_path)}
                )

        return True

    def load(self, node_file: str, line_file: str, transformer_file: str,
             converter_file: str, link_file: str,
             **kwargs) -> nx.DiGraph | nx.MultiDiGraph:
        """
        Load graph from CSV files: nodes, lines, transformers, and DC links.

        The loading process:
        1. Load nodes, lines, and transformers
        2. Detect DC islands (connected components before DC links)
        3. Assign 'dc_island' attribute to each bus
        4. Add DC links to connect islands
        5. Remove isolated nodes and warn user
        6. Return fully connected graph

        Args:
            node_file: Path to nodes CSV file
            line_file: Path to lines CSV file
            transformer_file: Path to transformers CSV file
            converter_file: Path to converters CSV file (required for DC links)
            link_file: Path to DC links CSV file (required)
            **kwargs: Additional parameters:
                - delimiter: CSV delimiter (default: ',')
                - decimal: Decimal separator (default: '.')
                - node_id_col: Column name for node IDs (auto-detected if not provided)

        Returns:
            DiGraph or MultiDiGraph with combined edges and dc_island attributes

        Raises:
            DataLoadingError: If loading fails
        """
        try:
            delimiter = kwargs.get('delimiter', ',')
            decimal = kwargs.get('decimal', '.')

            # Load and validate nodes
            nodes_df = self._load_nodes(node_file, delimiter, decimal)

            # Load and validate lines
            lines_df = self._load_lines(line_file, delimiter, decimal)

            # Load and validate transformers
            transformers_df = self._load_transformers(transformer_file, delimiter, decimal)

            # Load DC link data
            converters_df = self._load_converters(converter_file, delimiter, decimal)
            links_df = self._load_links(link_file, delimiter, decimal)

            # Get node ID column
            node_id_col = kwargs.get('node_id_col', self._detect_id_column(nodes_df))
            if node_id_col not in nodes_df.columns:
                raise DataLoadingError(
                    f"Node ID column '{node_id_col}' not found in node file",
                    strategy="va_loader",
                    details={'available_columns': list(nodes_df.columns)}
                )

            # Validate edge references for AC elements
            valid_node_ids = set(nodes_df[node_id_col].values)
            self._validate_edge_references(lines_df, valid_node_ids, "lines")
            self._validate_edge_references(transformers_df, valid_node_ids, "transformers")

            # Prepare node tuples (without dc_island yet)
            node_tuples = self._prepare_node_tuples(nodes_df, node_id_col)

            # Prepare AC edge tuples (lines and transformers)
            line_tuples = self._prepare_line_tuples(lines_df)
            transformer_tuples = self._prepare_transformer_tuples(transformers_df)
            ac_edge_tuples = line_tuples + transformer_tuples

            # Step 1: Detect DC islands before adding DC links
            dc_island_map = self._detect_dc_islands(node_tuples, ac_edge_tuples)

            # Update node tuples with dc_island attribute
            node_tuples = self._add_dc_island_to_nodes(node_tuples, dc_island_map)

            # Log DC island summary
            self._log_dc_island_summary(dc_island_map)

            # Prepare DC link tuples
            dc_link_tuples = self._prepare_dc_link_tuples(
                converters_df, links_df, valid_node_ids
            )

            all_edge_tuples = ac_edge_tuples + dc_link_tuples

            # Check for parallel edges
            has_parallel_edges = self._check_parallel_edges(all_edge_tuples)

            # Create appropriate graph type
            if has_parallel_edges:
                graph = nx.MultiDiGraph()
                print("MULTI-DIGRAPH DETECTED: Parallel edges found in the data.")
                print("Call manager.aggregate_parallel_edges() to collapse parallel edges.")
            else:
                graph = nx.DiGraph()

            graph.add_nodes_from(node_tuples)
            graph.add_edges_from(all_edge_tuples)

            # Step 2: Remove isolated nodes after full graph construction
            graph = self._remove_isolated_nodes(graph)

            # Log final summary
            n_lines = len(line_tuples)
            n_trafos = len(transformer_tuples)
            n_dc_links = len(dc_link_tuples)
            print(f"\nLoaded network: {graph.number_of_nodes()} nodes, "
                  f"{n_lines} lines, {n_trafos} transformers, {n_dc_links} DC links")

            # Verify final graph connectivity
            self._verify_final_connectivity(graph)

            return graph

        except DataLoadingError:
            raise
        except pd.errors.EmptyDataError as e:
            raise DataLoadingError(f"Empty CSV file: {e}", strategy="va_loader") from e
        except pd.errors.ParserError as e:
            raise DataLoadingError(f"CSV parsing error: {e}", strategy="va_loader") from e
        except Exception as e:
            raise DataLoadingError(
                f"Unexpected error loading voltage-aware CSV files: {e}",
                strategy="va_loader"
            ) from e

    # =========================================================================
    # DC Island Detection Methods
    # =========================================================================

    @staticmethod
    def _detect_dc_islands(node_tuples: List[Tuple[Any, Dict]],
                           ac_edge_tuples: List[Tuple[Any, Any, Dict]]) -> Dict[Any, int]:
        """
        Detect DC islands by finding connected components before DC links are added.

        Each connected component of lines and transformers represents a separate
        DC island (AC network that will be connected via DC links).

        Args:
            node_tuples: List of (node_id, attributes) tuples
            ac_edge_tuples: List of (from, to, attributes) tuples for lines and trafos

        Returns:
            Dictionary mapping node_id -> dc_island_id
        """
        # Create temporary undirected graph for component detection
        temp_graph = nx.Graph()

        # Add all nodes
        for node_id, _ in node_tuples:
            temp_graph.add_node(node_id)

        # Add AC edges (lines and transformers only)
        for from_node, to_node, _ in ac_edge_tuples:
            temp_graph.add_edge(from_node, to_node)

        # Find connected components
        components = list(nx.connected_components(temp_graph))

        # Create mapping: node_id -> dc_island_id
        dc_island_map: Dict[Any, int] = {}
        for island_id, component in enumerate(components):
            for node_id in component:
                dc_island_map[node_id] = island_id

        return dc_island_map

    @staticmethod
    def _add_dc_island_to_nodes(node_tuples: List[Tuple[Any, Dict]],
                                dc_island_map: Dict[Any, int]) -> List[Tuple[Any, Dict]]:
        """Add dc_island attribute to node tuples."""
        updated_tuples = []
        for node_id, attrs in node_tuples:
            attrs_copy = attrs.copy()
            attrs_copy['dc_island'] = dc_island_map.get(node_id, -1)
            updated_tuples.append((node_id, attrs_copy))
        return updated_tuples

    @staticmethod
    def _log_dc_island_summary(dc_island_map: Dict[Any, int]) -> None:
        """Log summary of detected DC islands."""
        island_counts: Dict[int, int] = {}
        for island_id in dc_island_map.values():
            island_counts[island_id] = island_counts.get(island_id, 0) + 1

        n_islands = len(island_counts)
        print(f"Found {n_islands} DC island(s)")

    # =========================================================================
    # Isolated Node Removal Methods
    # =========================================================================

    @staticmethod
    def _remove_isolated_nodes(graph: nx.DiGraph | nx.MultiDiGraph) -> DiGraph | MultiDiGraph:
        """Remove isolated nodes (nodes with no connections) from the graph."""
        isolated_nodes = list(nx.isolates(graph))

        if isolated_nodes:
            print(f"\nWARNING: Found {len(isolated_nodes)} isolated node(s) with no connections.")
            print("These nodes will be removed from the graph.")
            graph.remove_nodes_from(isolated_nodes)
        return graph

    @staticmethod
    def _verify_final_connectivity(graph: nx.DiGraph | nx.MultiDiGraph) -> None:
        """Verify the final graph connectivity and report status."""
        undirected = graph.to_undirected()
        n_components = nx.number_connected_components(undirected)

        if n_components == 1:
            print("Final graph is fully connected (single component)")
        else:
            print(f"Final graph has {n_components} disconnected component(s)")
            print("This may indicate missing DC links or data issues.")

            # Show component sizes
            components = sorted(nx.connected_components(undirected),
                                key=len, reverse=True)
            for i, comp in enumerate(components[:5]):
                print(f"  • Component {i}: {len(comp)} node(s)")
            if len(components) > 5:
                print(f"  ... and {len(components) - 5} more component(s)")

    # =========================================================================
    # File Loading Methods
    # =========================================================================

    @staticmethod
    def _load_nodes(file_path: str, delimiter: str, decimal: str) -> pd.DataFrame:
        """Load and validate nodes DataFrame."""
        nodes_df = pd.read_csv(file_path, delimiter=delimiter, decimal=decimal)

        if nodes_df.empty:
            raise DataLoadingError("Node file is empty", strategy="va_loader")

        return nodes_df

    def _load_lines(self, file_path: str, delimiter: str, decimal: str) -> pd.DataFrame:
        """Load and validate lines DataFrame."""
        lines_df = pd.read_csv(file_path, delimiter=delimiter, decimal=decimal, quotechar="'")

        if lines_df.empty:
            print("Warning: Lines file is empty. Proceeding with other edge types.")
            return pd.DataFrame(columns=self.REQUIRED_LINE_COLUMNS)

        missing_cols = [col for col in self.REQUIRED_LINE_COLUMNS if col not in lines_df.columns]
        if missing_cols:
            raise DataLoadingError(
                f"Lines file missing required columns: {missing_cols}",
                strategy="va_loader",
                details={'available_columns': list(lines_df.columns)}
            )

        return lines_df

    def _load_transformers(self, file_path: str, delimiter: str, decimal: str) -> pd.DataFrame:
        """Load and validate transformers DataFrame."""
        transformers_df = pd.read_csv(file_path, delimiter=delimiter, decimal=decimal, quotechar="'")

        if transformers_df.empty:
            print("Warning: Transformers file is empty. Proceeding with other edge types.")
            return pd.DataFrame(columns=self.REQUIRED_TRANSFORMER_COLUMNS)

        missing_cols = [col for col in self.REQUIRED_TRANSFORMER_COLUMNS
                        if col not in transformers_df.columns]
        if missing_cols:
            raise DataLoadingError(
                f"Transformers file missing required columns: {missing_cols}",
                strategy="va_loader",
                details={'available_columns': list(transformers_df.columns)}
            )

        # Validate voltage values
        for idx, row in transformers_df.iterrows():
            primary_v = row.get('primary_voltage', 0)
            secondary_v = row.get('secondary_voltage', 0)

            if pd.isna(primary_v) or pd.isna(secondary_v):
                raise DataLoadingError(
                    f"Transformer at row {idx} has missing voltage values",
                    strategy="va_loader"
                )

            if primary_v <= 0 or secondary_v <= 0:
                raise DataLoadingError(
                    f"Transformer at row {idx} must have positive primary and secondary voltages",
                    strategy="va_loader",
                    details={
                        'row': idx,
                        'primary_voltage': primary_v,
                        'secondary_voltage': secondary_v
                    }
                )

        return transformers_df

    def _load_converters(self, file_path: str, delimiter: str, decimal: str) -> pd.DataFrame:
        """Load and validate converters DataFrame."""
        converters_df = pd.read_csv(file_path, delimiter=delimiter, decimal=decimal, quotechar="'")

        if converters_df.empty:
            print("Warning: Converters file is empty. No DC links will be created.")
            return pd.DataFrame(columns=self.REQUIRED_CONVERTER_COLUMNS)

        missing_cols = [col for col in self.REQUIRED_CONVERTER_COLUMNS
                        if col not in converters_df.columns]
        if missing_cols:
            raise DataLoadingError(
                f"Converters file missing required columns: {missing_cols}",
                strategy="va_loader",
                details={'available_columns': list(converters_df.columns)}
            )

        return converters_df

    def _load_links(self, file_path: str, delimiter: str, decimal: str) -> pd.DataFrame:
        """Load and validate DC links DataFrame."""
        links_df = pd.read_csv(file_path, delimiter=delimiter, decimal=decimal, quotechar="'")

        if links_df.empty:
            print("Warning: Links file is empty. No DC links will be created.")
            return pd.DataFrame(columns=self.REQUIRED_LINK_COLUMNS)

        missing_cols = [col for col in self.REQUIRED_LINK_COLUMNS
                        if col not in links_df.columns]
        if missing_cols:
            raise DataLoadingError(
                f"Links file missing required columns: {missing_cols}",
                strategy="va_loader",
                details={'available_columns': list(links_df.columns)}
            )

        return links_df

    # =========================================================================
    # Validation Methods
    # =========================================================================

    @staticmethod
    def _validate_edge_references(edges_df: pd.DataFrame, valid_node_ids: set,
                                  edge_type: str) -> None:
        """Validate that all edge node references exist in the node set."""
        if edges_df.empty:
            return

        for idx, row in edges_df.iterrows():
            bus0 = row['bus0']
            bus1 = row['bus1']

            if bus0 not in valid_node_ids:
                raise DataLoadingError(
                    f"{edge_type.capitalize()} at row {idx} references non-existent node: {bus0}",
                    strategy="va_loader"
                )
            if bus1 not in valid_node_ids:
                raise DataLoadingError(
                    f"{edge_type.capitalize()} at row {idx} references non-existent node: {bus1}",
                    strategy="va_loader"
                )

    # =========================================================================
    # Edge Tuple Preparation Methods
    # =========================================================================

    @staticmethod
    def _prepare_node_tuples(nodes_df: pd.DataFrame, node_id_col: str) -> List[Tuple[Any, Dict]]:
        """Prepare node tuples for graph creation."""
        return [
            (
                row[node_id_col],
                {col: row[col] for col in nodes_df.columns
                 if col != node_id_col and pd.notna(row[col])}
            )
            for _, row in nodes_df.iterrows()
        ]

    @staticmethod
    def _prepare_line_tuples(lines_df: pd.DataFrame) -> List[Tuple[Any, Any, Dict]]:
        """Prepare line edge tuples with unified schema."""
        edge_tuples = []

        for _, row in lines_df.iterrows():
            voltage = row.get('line_voltage', row.get('voltage', 0))

            attrs = {
                'type': EdgeType.LINE.value,
                'primary_voltage': voltage,
                'secondary_voltage': voltage,  # Same voltage for lines
            }

            # Add all other columns as attributes
            for col in lines_df.columns:
                if col not in ['bus0', 'bus1', 'line_voltage', 'voltage'] and pd.notna(row[col]):
                    if col not in attrs:
                        attrs[col] = row[col]

            edge_tuples.append((row['bus0'], row['bus1'], attrs))

        return edge_tuples

    @staticmethod
    def _prepare_transformer_tuples(transformers_df: pd.DataFrame) -> List[Tuple[Any, Any, Dict]]:
        """Prepare transformer edge tuples with unified schema."""
        edge_tuples = []

        for _, row in transformers_df.iterrows():
            attrs = {
                'type': EdgeType.TRAFO.value,
                'primary_voltage': row['primary_voltage'],
                'secondary_voltage': row['secondary_voltage'],
            }

            # Add all other columns as attributes
            for col in transformers_df.columns:
                if col not in ['bus0', 'bus1', 'primary_voltage', 'secondary_voltage'] \
                        and pd.notna(row[col]):
                    if col not in attrs:
                        attrs[col] = row[col]

            edge_tuples.append((row['bus0'], row['bus1'], attrs))

        return edge_tuples

    @staticmethod
    def _prepare_dc_link_tuples(converters_df: pd.DataFrame, links_df: pd.DataFrame,
                                valid_node_ids: set) -> List[Tuple[Any, Any, Dict]]:
        """Prepare DC link edge tuples by resolving converter connections."""
        if converters_df.empty or links_df.empty:
            return []

        # Build converter lookup
        converter_lookup: Dict[Any, Dict] = {}
        for _, row in converters_df.iterrows():
            converter_bus0 = row['bus0']
            converter_lookup[converter_bus0] = {
                'converter_id': row['converter_id'],
                'ac_bus': row['bus1'],
                'dc_voltage': row['voltage'],
                'p_nom': row.get('p_nom', None)
            }

        edge_tuples = []
        skipped_links = []

        for _, row in links_df.iterrows():
            link_id = row['link_id']
            link_bus0 = row['bus0']
            link_bus1 = row['bus1']
            link_voltage = row['voltage']

            # Resolve AC buses through converters
            conv0 = converter_lookup.get(link_bus0)
            conv1 = converter_lookup.get(link_bus1)

            if conv0 is None:
                skipped_links.append((link_id, f"Converter not found for bus0: {link_bus0}"))
                continue

            if conv1 is None:
                skipped_links.append((link_id, f"Converter not found for bus1: {link_bus1}"))
                continue

            ac_bus0 = conv0['ac_bus']
            ac_bus1 = conv1['ac_bus']

            # Validate AC buses exist in the network
            if ac_bus0 not in valid_node_ids:
                skipped_links.append((link_id, f"AC bus not in network: {ac_bus0}"))
                continue

            if ac_bus1 not in valid_node_ids:
                skipped_links.append((link_id, f"AC bus not in network: {ac_bus1}"))
                continue

            # Build edge attributes
            attrs = {
                'type': EdgeType.DC_LINK.value,
                'primary_voltage': link_voltage,
                'secondary_voltage': link_voltage,  # Same voltage for DC links
                'link_id': link_id,
                'converter_bus0': link_bus0,
                'converter_bus1': link_bus1,
                'converter_id_0': conv0['converter_id'],
                'converter_id_1': conv1['converter_id'],
            }

            # Add other link attributes
            for col in links_df.columns:
                if col not in ['link_id', 'bus0', 'bus1', 'voltage'] and pd.notna(row[col]):
                    if col not in attrs:
                        attrs[col] = row[col]

            edge_tuples.append((ac_bus0, ac_bus1, attrs))

        # Report skipped links
        if skipped_links:
            print(f"Warning: {len(skipped_links)} DC links skipped due to missing references:")
            for link_id, reason in skipped_links[:5]:  # Show first 5
                print(f"  - {link_id}: {reason}")
            if len(skipped_links) > 5:
                print(f"  ... and {len(skipped_links) - 5} more")

        return edge_tuples

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @staticmethod
    def _check_parallel_edges(edge_tuples: List[Tuple]) -> bool:
        """Check if there are parallel edges (same source-target pair)."""
        seen_pairs = set()

        for edge in edge_tuples:
            pair = (edge[0], edge[1])
            if pair in seen_pairs:
                return True
            seen_pairs.add(pair)

        return False

    @staticmethod
    def _detect_id_column(df: pd.DataFrame) -> str:
        """Detect the ID column for nodes."""
        candidates = [
            'node_id', 'nodeId', 'node_ID', 'NodeId',
            'bus_id', 'busId', 'bus_ID',
            'id', 'Id', 'ID', 'index', 'name'
        ]

        for candidate in candidates:
            if candidate in df.columns:
                return candidate

        # Fallback to first column
        return df.columns[0]
