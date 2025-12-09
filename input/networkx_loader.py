import networkx as nx

from exceptions import DataLoadingError
from interfaces import DataLoadingStrategy


class NetworkXDirectStrategy(DataLoadingStrategy):
    """Use NetworkX graph directly, converting to directed graph"""

    def validate_inputs(self, **kwargs) -> bool:
        """Validate that a NetworkX graph is provided"""
        if 'graph' not in kwargs:
            raise DataLoadingError(
                "Missing required parameter: graph",
                strategy="networkx_direct",
                details={'required_params': ['graph']}
            )

        graph = kwargs['graph']
        if not isinstance(graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
            raise DataLoadingError(
                f"Parameter 'graph' must be a NetworkX Graph, got {type(graph)}",
                strategy="networkx_direct",
                details={'provided_type': str(type(graph))}
            )

        # Additional validation
        if len(list(graph.nodes())) == 0:
            raise DataLoadingError(
                "Provided graph has no nodes",
                strategy="networkx_direct"
            )

        return True

    def load(self, graph: nx.Graph, **kwargs) -> nx.DiGraph | nx.MultiDiGraph:
        """
        Use graph directly, converting to directed graph if needed.

        Supports all NetworkX graph types:
        - Graph -> DiGraph (creates both directions for each edge)
        - DiGraph -> DiGraph (copy)
        - MultiGraph -> MultiDiGraph (creates both directions for each edge)
        - MultiDiGraph -> MultiDiGraph (copy)

        Args:
            graph: Input NetworkX graph
            **kwargs: Additional parameters
                - bidirectional: If True (default), convert undirected edges to
                                 bidirectional directed edges. If False, only
                                 create edges in the original iteration order.

        Returns:
            DiGraph or MultiDiGraph
        """
        try:
            bidirectional = kwargs.get('bidirectional', True)

            if isinstance(graph, nx.MultiDiGraph):
                # Already a MultiDiGraph - create copy
                print("MULTI-DIGRAPH DETECTED: Input is already a MultiDiGraph.")
                print("MultiDiGraphs cannot be partitioned directly.")
                print("Call manager.aggregate_parallel_edges() to collapse parallel edges.")

                result = nx.MultiDiGraph()
                result.add_nodes_from(graph.nodes(data=True))
                result.add_edges_from(graph.edges(data=True, keys=True))

            elif isinstance(graph, nx.DiGraph):
                # Already directed - create copy
                result = graph.copy()

            elif isinstance(graph, nx.MultiGraph):
                # Convert MultiGraph to MultiDiGraph
                print("MULTI-DIGRAPH DETECTED: Converting MultiGraph to MultiDiGraph.")
                print("MultiDiGraphs cannot be partitioned directly.")
                print("Call manager.aggregate_parallel_edges() to collapse parallel edges.")

                result = nx.MultiDiGraph()
                result.add_nodes_from(graph.nodes(data=True))

                # Add edges - for undirected, optionally create both directions
                for u, v, key, data in graph.edges(data=True, keys=True):
                    result.add_edge(u, v, key=key, **data)
                    if bidirectional:
                        result.add_edge(v, u, key=key, **data)

            else:
                # Convert Graph to DiGraph
                result = nx.DiGraph()
                result.add_nodes_from(graph.nodes(data=True))

                # Add edges - for undirected, optionally create both directions
                for u, v, data in graph.edges(data=True):
                    result.add_edge(u, v, **data)
                    if bidirectional:
                        result.add_edge(v, u, **data)

            # Validate the resulting graph
            if len(list(result.nodes())) == 0:
                raise DataLoadingError(
                    "Graph has no nodes after processing",
                    strategy="networkx_direct"
                )

            return result

        except Exception as e:
            raise DataLoadingError(
                f"Error processing NetworkX graph: {e}",
                strategy="networkx_direct"
            ) from e
