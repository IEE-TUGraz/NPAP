import networkx as nx

from npap.exceptions import DataLoadingError
from npap.interfaces import DataLoadingStrategy
from npap.logging import LogCategory, log_debug, log_info, log_warning


class NetworkXDirectStrategy(DataLoadingStrategy):
    """
    Load graph directly from an existing NetworkX graph object.

    This strategy accepts any NetworkX graph type and converts it to a
    directed graph (DiGraph or MultiDiGraph) suitable for partitioning.

    Examples
    --------
    >>> import networkx as nx
    >>> from npap.managers import PartitionAggregatorManager
    >>> G = nx.karate_club_graph()
    >>> manager = PartitionAggregatorManager()
    >>> graph = manager.load_data("networkx_direct", graph=G)
    """

    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate that a NetworkX graph is provided.

        Parameters
        ----------
        **kwargs : dict
            Must contain 'graph' key with a NetworkX graph object.

        Returns
        -------
        bool
            True if validation passes.

        Raises
        ------
        DataLoadingError
            If graph is missing, not a NetworkX graph, or empty.
        """
        if "graph" not in kwargs:
            raise DataLoadingError(
                "Missing required parameter: graph",
                strategy="networkx_direct",
                details={"required_params": ["graph"]},
            )

        graph = kwargs["graph"]
        if not isinstance(graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
            raise DataLoadingError(
                f"Parameter 'graph' must be a NetworkX Graph, got {type(graph)}",
                strategy="networkx_direct",
                details={"provided_type": str(type(graph))},
            )

        if len(list(graph.nodes())) == 0:
            raise DataLoadingError("Provided graph has no nodes", strategy="networkx_direct")

        log_debug(f"Validated NetworkX graph: {type(graph).__name__}", LogCategory.INPUT)
        return True

    def load(self, graph: nx.Graph, **kwargs) -> nx.DiGraph | nx.MultiDiGraph:
        """
        Convert input graph to directed graph if needed.

        Supports all NetworkX graph types:

        - Graph -> DiGraph (creates both directions for each edge)
        - DiGraph -> DiGraph (copy)
        - MultiGraph -> MultiDiGraph (creates both directions for each edge)
        - MultiDiGraph -> MultiDiGraph (copy)

        Parameters
        ----------
        graph : nx.Graph
            Input NetworkX graph (any type).
        **kwargs : dict
            Additional parameters:

            - bidirectional : bool, default True
                If True, convert undirected edges to bidirectional directed
                edges. If False, only create edges in original iteration order.

        Returns
        -------
        nx.DiGraph or nx.MultiDiGraph
            Directed graph suitable for partitioning.

        Raises
        ------
        DataLoadingError
            If graph processing fails.
        """
        try:
            bidirectional = kwargs.get("bidirectional", True)

            if isinstance(graph, nx.MultiDiGraph):
                log_debug("Input is already a MultiDiGraph, creating copy", LogCategory.INPUT)
                log_warning(
                    "Parallel edges detected in input MultiDiGraph. "
                    "Call manager.aggregate_parallel_edges() to collapse parallel edges before partitioning.",
                    LogCategory.INPUT,
                )

                result = nx.MultiDiGraph()
                result.add_nodes_from(graph.nodes(data=True))
                result.add_edges_from(graph.edges(data=True, keys=True))

            elif isinstance(graph, nx.DiGraph):
                log_debug("Input is DiGraph, creating copy", LogCategory.INPUT)
                result = graph.copy()

            elif isinstance(graph, nx.MultiGraph):
                log_debug("Converting MultiGraph to MultiDiGraph", LogCategory.INPUT)
                log_warning(
                    "Parallel edges detected in input MultiGraph. "
                    "Call manager.aggregate_parallel_edges() to collapse parallel edges before partitioning.",
                    LogCategory.INPUT,
                )

                result = nx.MultiDiGraph()
                result.add_nodes_from(graph.nodes(data=True))

                # Add edges - for undirected, optionally create both directions
                for u, v, key, data in graph.edges(data=True, keys=True):
                    result.add_edge(u, v, key=key, **data)
                    if bidirectional:
                        result.add_edge(v, u, key=key, **data)

            else:
                log_debug("Converting Graph to DiGraph", LogCategory.INPUT)
                result = nx.DiGraph()
                result.add_nodes_from(graph.nodes(data=True))

                # Add edges - for undirected, optionally create both directions
                for u, v, data in graph.edges(data=True):
                    result.add_edge(u, v, **data)
                    if bidirectional:
                        result.add_edge(v, u, **data)

            if len(list(result.nodes())) == 0:
                raise DataLoadingError(
                    "Graph has no nodes after processing", strategy="networkx_direct"
                )

            log_info(
                f"Loaded graph from NetworkX: {result.number_of_nodes()} nodes, {result.number_of_edges()} edges",
                LogCategory.INPUT,
            )

            return result

        except DataLoadingError:
            raise
        except Exception as e:
            raise DataLoadingError(
                f"Error processing NetworkX graph: {e}", strategy="networkx_direct"
            ) from e
