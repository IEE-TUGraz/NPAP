from pathlib import Path

import networkx as nx
import pandas as pd

from exceptions import DataLoadingError
from interfaces import DataLoadingStrategy


class CSVFilesStrategy(DataLoadingStrategy):
    """Load graph from separate CSV files for nodes and edges"""

    def validate_inputs(self, **kwargs) -> bool:
        """Validate that required CSV files are provided"""
        required_files = ['node_file', 'edge_file']
        missing = [file for file in required_files if file not in kwargs or kwargs[file] is None]
        if missing:
            raise DataLoadingError(
                f"Missing required parameters: {missing}",
                strategy="csv_files",
                details={'required_params': required_files, 'provided_params': list(kwargs.keys())}
            )

        # Check if files exist
        for param in required_files:
            file_path = Path(kwargs[param])
            if not file_path.exists():
                raise DataLoadingError(
                    f"File not found: {file_path}",
                    strategy="csv_files",
                    details={'missing_file': str(file_path)}
                )

        return True

    def load(self, node_file: str, edge_file: str, **kwargs) -> nx.Graph | nx.MultiGraph:
        """Load graph from CSV files"""
        try:
            delimiter = kwargs["delimiter"] if 'delimiter' in kwargs else ','
            decimal = kwargs["decimal"] if 'decimal' in kwargs else '.'

            # Load nodes
            nodes_df = pd.read_csv(node_file, delimiter=delimiter, decimal=decimal)
            if nodes_df.empty:
                raise DataLoadingError("Node file is empty", strategy="csv_files")

            # Determine node ID column
            node_id_col = kwargs.get('node_id_col', self._detect_id_column(nodes_df, 'node'))
            if node_id_col not in nodes_df.columns:
                raise DataLoadingError(
                    f"Node ID column '{node_id_col}' not found in node file",
                    strategy="csv_files",
                    details={'available_columns': list(nodes_df.columns)}
                )

            # Load edges
            edges_df = pd.read_csv(edge_file, delimiter=delimiter, decimal=decimal)
            if edges_df.empty:
                raise DataLoadingError("Edge file is empty", strategy="csv_files")

            # Determine edge columns
            edge_from_col = kwargs.get('edge_from_col', self._detect_edge_column(edges_df, 'from'))
            edge_to_col = kwargs.get('edge_to_col', self._detect_edge_column(edges_df, 'to'))

            if edge_from_col not in edges_df.columns:
                raise DataLoadingError(
                    f"Edge 'from' column '{edge_from_col}' not found in edge file",
                    strategy="csv_files",
                    details={'available_columns': list(edges_df.columns)}
                )

            if edge_to_col not in edges_df.columns:
                raise DataLoadingError(
                    f"Edge 'to' column '{edge_to_col}' not found in edge file",
                    strategy="csv_files",
                    details={'available_columns': list(edges_df.columns)}
                )

            # Prepare node tuples: (node_id, attr_dict)
            node_tuples = [
                (row[node_id_col],
                 {col: row[col] for col in nodes_df.columns if col != node_id_col and pd.notna(row[col])})
                for _, row in nodes_df.iterrows()
            ]

            # Check for parallel edges (duplicate node pairs)
            edge_pairs = edges_df[[edge_from_col, edge_to_col]].copy()
            # Sort pairs to consider (A,B) and (B,A) as same pair for undirected graph
            edge_pairs['sorted_pair'] = edge_pairs.apply(
                lambda row: tuple(sorted([row[edge_from_col], row[edge_to_col]])), axis=1
            )
            has_parallel_edges = edge_pairs['sorted_pair'].duplicated().any()

            # Prepare edge tuples: (from_node, to_node, attr_dict)
            edge_tuples = []
            for _, row in edges_df.iterrows():
                from_node = row[edge_from_col]
                to_node = row[edge_to_col]
                if from_node not in nodes_df[node_id_col].values or to_node not in nodes_df[node_id_col].values:
                    raise DataLoadingError("Edge references non-existent node", strategy="csv_files")

                attrs = {col: row[col] for col in edges_df.columns if
                         col not in [edge_from_col, edge_to_col] and pd.notna(row[col])}
                edge_tuples.append((from_node, to_node, attrs))

            # Create appropriate graph type based on parallel edges
            if has_parallel_edges:
                graph = nx.MultiGraph()
                print("MULTIGRAPH DETECTED: Parallel edges found in the data.")
                print("The loaded graph contains multiple edges between the same node pairs. MultiGraphs cannot be "
                      "partitioned directly. Call manager.aggregate_parallel_edges() to collapse parallel edges")
            else:
                graph = nx.Graph()

            graph.add_nodes_from(node_tuples)
            graph.add_edges_from(edge_tuples)

            return graph

        except pd.errors.EmptyDataError as e:
            raise DataLoadingError(f"Empty CSV file: {e}", strategy="csv_files") from e
        except pd.errors.ParserError as e:
            raise DataLoadingError(f"CSV parsing error: {e}", strategy="csv_files") from e
        except Exception as e:
            raise DataLoadingError(f"Unexpected error loading CSV files: {e}", strategy="csv_files") from e

    @staticmethod
    def _detect_id_column(df: pd.DataFrame, prefix: str) -> str:
        """Detect the ID column for nodes/edges"""
        # Common variations for ID columns
        candidates = [
            f'{prefix}_id', f'{prefix}Id', f'{prefix}_ID',
            f'{prefix}', 'id', 'Id', 'ID', 'index'
        ]

        for candidate in candidates:
            if candidate in df.columns:
                return candidate

        # If no standard column found, use first column
        return df.columns[0]

    @staticmethod
    def _detect_edge_column(df: pd.DataFrame, direction: str) -> str:
        """Detect from/to columns for edges"""
        if direction == 'from':
            candidates = ['from', 'source', 'from_node', 'source_node', 'node1', 'start']
        else:  # direction == 'to'
            candidates = ['to', 'target', 'to_node', 'target_node', 'node2', 'end']

        for candidate in candidates:
            if candidate in df.columns:
                return candidate

        # Fallback: use first two columns
        return df.columns[0] if direction == 'from' else df.columns[1]
