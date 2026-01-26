from pathlib import Path

import networkx as nx
import pandas as pd

from npap.exceptions import DataLoadingError
from npap.interfaces import DataLoadingStrategy
from npap.logging import LogCategory, log_debug, log_info, log_warning


class CSVFilesStrategy(DataLoadingStrategy):
    """
    Load graph from separate CSV files for nodes and edges.

    This strategy reads node and edge data from CSV files and constructs
    a NetworkX directed graph. Supports automatic detection of ID columns
    and handles parallel edges by creating a MultiDiGraph.

    Examples
    --------
    >>> from npap.managers import PartitionAggregatorManager
    >>> manager = PartitionAggregatorManager()
    >>> graph = manager.load_data(
    ...     "csv_files",
    ...     node_file="buses.csv",
    ...     edge_file="lines.csv"
    ... )
    """

    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate that required CSV files are provided.

        Parameters
        ----------
        **kwargs : dict
            Must contain 'node_file' and 'edge_file' keys.

        Returns
        -------
        bool
            True if validation passes.

        Raises
        ------
        DataLoadingError
            If required files are missing or don't exist.
        """
        required_files = ["node_file", "edge_file"]
        missing = [file for file in required_files if file not in kwargs or kwargs[file] is None]
        if missing:
            raise DataLoadingError(
                f"Missing required parameters: {missing}",
                strategy="csv_files",
                details={
                    "required_params": required_files,
                    "provided_params": list(kwargs.keys()),
                },
            )

        # Check if files exist
        for param in required_files:
            file_path = Path(kwargs[param])
            if not file_path.exists():
                raise DataLoadingError(
                    f"File not found: {file_path}",
                    strategy="csv_files",
                    details={"missing_file": str(file_path)},
                )

        log_debug(
            f"Validated input files: {kwargs.get('node_file')}, {kwargs.get('edge_file')}",
            LogCategory.INPUT,
        )
        return True

    def load(self, node_file: str, edge_file: str, **kwargs) -> nx.DiGraph | nx.MultiDiGraph:
        """
        Load graph from CSV files as a directed graph.

        Parameters
        ----------
        node_file : str
            Path to CSV file containing node data.
        edge_file : str
            Path to CSV file containing edge data.
        **kwargs : dict
            Additional parameters:

            - delimiter : str, default ','
                CSV delimiter character.
            - decimal : str, default '.'
                Decimal separator character.
            - node_id_col : str, optional
                Column name for node IDs (auto-detected if not provided).
            - edge_from_col : str, optional
                Column name for edge source nodes (auto-detected if not provided).
            - edge_to_col : str, optional
                Column name for edge target nodes (auto-detected if not provided).

        Returns
        -------
        nx.DiGraph or nx.MultiDiGraph
            Loaded graph. Returns MultiDiGraph if parallel edges are detected.

        Raises
        ------
        DataLoadingError
            If files are empty, malformed, or contain invalid references.
        """
        try:
            delimiter = kwargs.get("delimiter", ",")
            decimal = kwargs.get("decimal", ".")

            log_debug(f"Loading nodes from {node_file}", LogCategory.INPUT)

            # Load nodes
            nodes_df = pd.read_csv(node_file, delimiter=delimiter, decimal=decimal)
            if nodes_df.empty:
                raise DataLoadingError("Node file is empty", strategy="csv_files")

            # Determine node ID column
            node_id_col = kwargs.get("node_id_col", self._detect_id_column(nodes_df, "node"))
            if node_id_col not in nodes_df.columns:
                raise DataLoadingError(
                    f"Node ID column '{node_id_col}' not found in node file",
                    strategy="csv_files",
                    details={"available_columns": list(nodes_df.columns)},
                )

            log_debug(f"Loading edges from {edge_file}", LogCategory.INPUT)

            # Load edges
            edges_df = pd.read_csv(edge_file, delimiter=delimiter, decimal=decimal, quotechar="'")
            if edges_df.empty:
                raise DataLoadingError("Edge file is empty", strategy="csv_files")

            # Determine edge columns
            edge_from_col = kwargs.get("edge_from_col", self._detect_edge_column(edges_df, "from"))
            edge_to_col = kwargs.get("edge_to_col", self._detect_edge_column(edges_df, "to"))

            if edge_from_col not in edges_df.columns:
                raise DataLoadingError(
                    f"Edge 'from' column '{edge_from_col}' not found in edge file",
                    strategy="csv_files",
                    details={"available_columns": list(edges_df.columns)},
                )

            if edge_to_col not in edges_df.columns:
                raise DataLoadingError(
                    f"Edge 'to' column '{edge_to_col}' not found in edge file",
                    strategy="csv_files",
                    details={"available_columns": list(edges_df.columns)},
                )

            # Prepare node tuples
            node_records = nodes_df.to_dict("records")
            node_tuples = [
                (
                    record[node_id_col],
                    {k: v for k, v in record.items() if k != node_id_col and pd.notna(v)},
                )
                for record in node_records
            ]

            # Check for parallel edges
            has_parallel_edges = edges_df.duplicated(
                subset=[edge_from_col, edge_to_col], keep=False
            ).any()

            # Validate edge references using vectorized set operations
            valid_node_ids = set(nodes_df[node_id_col].values)
            invalid_from = ~edges_df[edge_from_col].isin(valid_node_ids)
            invalid_to = ~edges_df[edge_to_col].isin(valid_node_ids)

            if invalid_from.any() or invalid_to.any():
                raise DataLoadingError("Edge references non-existent node", strategy="csv_files")

            # Prepare edge tuples
            edge_records = edges_df.to_dict("records")
            edge_tuples = [
                (
                    record[edge_from_col],
                    record[edge_to_col],
                    {
                        k: v
                        for k, v in record.items()
                        if k not in [edge_from_col, edge_to_col] and pd.notna(v)
                    },
                )
                for record in edge_records
            ]

            # Create appropriate directed graph type based on parallel edges
            if has_parallel_edges:
                graph = nx.MultiDiGraph()
                log_warning(
                    "Parallel edges detected in CSV edge file. A MultiDiGraph will be created. "
                    "Call manager.aggregate_parallel_edges() to collapse parallel edges before partitioning.",
                    LogCategory.INPUT,
                )
            else:
                graph = nx.DiGraph()

            graph.add_nodes_from(node_tuples)
            graph.add_edges_from(edge_tuples)

            log_info(
                f"Loaded graph from CSV: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges",
                LogCategory.INPUT,
            )

            return graph

        except pd.errors.EmptyDataError as e:
            raise DataLoadingError(f"Empty CSV file: {e}", strategy="csv_files") from e
        except pd.errors.ParserError as e:
            raise DataLoadingError(f"CSV parsing error: {e}", strategy="csv_files") from e
        except DataLoadingError:
            raise
        except Exception as e:
            raise DataLoadingError(
                f"Unexpected error loading CSV files: {e}", strategy="csv_files"
            ) from e

    @staticmethod
    def _detect_id_column(df: pd.DataFrame, prefix: str) -> str:
        """
        Detect the ID column for nodes/edges.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to search for ID column.
        prefix : str
            Expected prefix for the ID column (e.g., 'node', 'edge').

        Returns
        -------
        str
            Detected column name, or first column if no match found.
        """
        candidates = [
            f"{prefix}_id",
            f"{prefix}Id",
            f"{prefix}_ID",
            f"{prefix}",
            "id",
            "Id",
            "ID",
            "index",
        ]

        for candidate in candidates:
            if candidate in df.columns:
                return candidate

        # If no standard column found, use first column
        return df.columns[0]

    @staticmethod
    def _detect_edge_column(df: pd.DataFrame, direction: str) -> str:
        """
        Detect from/to columns for edges.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to search for edge columns.
        direction : str
            Either 'from' or 'to' to indicate which column to detect.

        Returns
        -------
        str
            Detected column name, or first/second column if no match found.
        """
        if direction == "from":
            candidates = [
                "from",
                "source",
                "from_node",
                "source_node",
                "node1",
                "start",
                "bus0",
            ]
        else:  # direction == 'to'
            candidates = [
                "to",
                "target",
                "to_node",
                "target_node",
                "node2",
                "end",
                "bus1",
            ]

        for candidate in candidates:
            if candidate in df.columns:
                return candidate

        # Fallback: use first two columns
        return df.columns[0] if direction == "from" else df.columns[1]
