from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from npap.exceptions import PartitioningError
from npap.interfaces import PartitioningStrategy
from npap.logging import LogCategory, log_debug, log_info, log_warning
from npap.utils import (
    create_partition_map,
    run_hierarchical,
    validate_partition,
    validate_required_attributes,
    with_runtime_config,
)


@dataclass
class LMPConfig:
    """
    Configuration parameters for LMP-based partitioning.

    Attributes
    ----------
    hierarchical_linkage : str
        Linkage criterion for hierarchical clustering
        ('complete', 'average', 'single').
    infinite_distance : float
        Value used to represent "infinite" distance between nodes in different
        AC islands.
    use_connectivity : bool
        Whether to use graph connectivity as a constraint for clustering.
        If True, only adjacent nodes in the graph can be merged into a cluster.
    """

    hierarchical_linkage: str = "complete"
    infinite_distance: float = 1e4
    use_connectivity: bool = True


class LMPPartitioning(PartitioningStrategy):
    """
    Partition nodes based on Locational Marginal Prices (LMP).

    This strategy clusters nodes based on their LMP values. LMPs can be single
    values or time-series profiles (vectors). The distance between nodes is
    calculated as the Euclidean distance between their LMP profiles.

    AC-Island Awareness:
        When the graph contains AC island data (nodes have 'ac_island' attribute),
        the strategy respects AC island boundaries by assigning infinite
        distance between nodes in different AC islands.

    Connectivity Constraint:
        When `use_connectivity` is enabled in the config, the clustering
        algorithm is constrained by the graph's topology. Only nodes that are
        directly connected by an edge (or belong to the same connected component)
        can be clustered together. This ensures that resulting clusters are
        spatially contiguous in the network.

    Supported algorithms:
        - 'hierarchical': Agglomerative hierarchical clustering.
    """

    SUPPORTED_ALGORITHMS = ["hierarchical"]

    # Config parameter names for runtime override detection
    _CONFIG_PARAMS = {
        "hierarchical_linkage",
        "infinite_distance",
        "use_connectivity",
    }

    def __init__(
        self,
        algorithm: str = "hierarchical",
        lmp_attr: str = "lmp",
        distance_metric: str = "euclidean",
        ac_island_attr: str = "ac_island",
        config: LMPConfig | None = None,
    ):
        """
        Initialize LMP partitioning strategy.

        Parameters
        ----------
        algorithm : str, default='hierarchical'
            Clustering algorithm. Currently only 'hierarchical' is supported.
        lmp_attr : str, default='lmp'
            Node attribute name containing LMP values.
        ac_island_attr : str, default='ac_island'
            Node attribute name containing AC island ID.
        config : LMPConfig, optional
            Configuration parameters for the algorithm.

        Raises
        ------
        ValueError
            If unsupported algorithm is specified.
        """
        self.algorithm = algorithm
        self.lmp_attr = lmp_attr
        self.distance_metric = distance_metric
        self.ac_island_attr = ac_island_attr
        self.config = config or LMPConfig()

        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm: {algorithm}. "
                f"Supported: {', '.join(self.SUPPORTED_ALGORITHMS)}"
            )

        log_debug(
            f"Initialized LMPPartitioning: algorithm={algorithm}, lmp_attr={lmp_attr}",
            LogCategory.PARTITIONING,
        )

    @property
    def required_attributes(self) -> dict[str, list[str]]:
        """Required node attributes for LMP partitioning."""
        return {"nodes": [self.lmp_attr], "edges": []}

    def _get_strategy_name(self) -> str:
        """Get descriptive strategy name for error messages."""
        return f"lmp_{self.algorithm}"

    @with_runtime_config(LMPConfig, _CONFIG_PARAMS)
    @validate_required_attributes
    def partition(self, graph: nx.Graph, **kwargs) -> dict[int, list[Any]]:
        """
        Partition nodes based on Locational Marginal Prices.

        Automatically detects and respects AC island boundaries when available.

        Parameters
        ----------
        graph : nx.Graph
            NetworkX graph with LMP attributes on nodes.
        **kwargs : dict
            Additional parameters:

            - n_clusters : Number of clusters (required for hierarchical)
            - config : LMPConfig instance to override instance config
            - hierarchical_linkage : Override config parameter
            - infinite_distance : Override config parameter
            - use_connectivity : Override config parameter

        Returns
        -------
        dict[int, list[Any]]
            Dictionary mapping cluster_id -> list of node_ids.

        Raises
        ------
        PartitioningError
            If partitioning fails.
        """
        try:
            effective_config = kwargs.get("_effective_config", self.config)
            n_clusters = kwargs.get("n_clusters")

            # Extract LMP data
            nodes = list(graph.nodes())
            lmps = self._extract_lmps(graph, nodes)

            # Auto-detect AC island data
            ac_islands = None
            has_ac_islands = self._has_ac_island_data(graph, nodes)

            if has_ac_islands:
                ac_islands = self._extract_ac_islands(graph, nodes)
                n_ac_islands = len(set(ac_islands))

                log_info(
                    f"Starting AC-island-aware locational marginal prices partitioning: {self.algorithm}, "
                    f"n_clusters={n_clusters}, metric={self.distance_metric}, "
                    f"ac_islands={n_ac_islands}",
                    LogCategory.PARTITIONING,
                )
            else:
                log_info(
                    f"Starting LMP partitioning: {self.algorithm}, n_clusters={n_clusters}",
                    LogCategory.PARTITIONING,
                )

            # Perform clustering
            labels = self._run_clustering(
                graph, lmps, effective_config, ac_islands=ac_islands, **kwargs
            )

            # Create and validate partition
            partition_map = create_partition_map(nodes, labels)
            validate_partition(partition_map, len(nodes), self._get_strategy_name())

            log_info(
                f"LMP partitioning complete: {len(partition_map)} clusters",
                LogCategory.PARTITIONING,
            )

            return partition_map

        except Exception as e:
            if isinstance(e, PartitioningError):
                raise
            raise PartitioningError(
                f"LMP partitioning failed: {e}",
                strategy=self._get_strategy_name(),
                graph_info={
                    "nodes": len(list(graph.nodes())),
                    "edges": len(graph.edges()),
                },
            ) from e

    def _extract_lmps(self, graph: nx.Graph, nodes: list[Any]) -> np.ndarray:
        """
        Extract LMP values from graph nodes.

        Supports both scalar LMPs and time-series vectors.

        Parameters
        ----------
        graph : nx.Graph
            NetworkX graph.
        nodes : list[Any]
            List of node IDs.

        Returns
        -------
        np.ndarray
            Array of LMP profiles (n_nodes x n_timesteps).
        """
        lmp_profiles = []

        for node in nodes:
            val = graph.nodes[node].get(self.lmp_attr)
            if val is None:
                raise PartitioningError(
                    f"Node {node} missing LMP attribute '{self.lmp_attr}'",
                    strategy=self._get_strategy_name(),
                )

            # Ensure data is numeric and array-like
            if isinstance(val, (int, float)):
                lmp_profiles.append([float(val)])
            else:
                lmp_profiles.append(np.atleast_1d(val))

        return np.array(lmp_profiles)

    def _has_ac_island_data(self, graph: nx.Graph, nodes: list[Any]) -> bool:
        """Check if the graph has AC island data on nodes."""
        for node in nodes:
            if self.ac_island_attr not in graph.nodes[node]:
                return False
        return True

    def _extract_ac_islands(self, graph: nx.Graph, nodes: list[Any]) -> np.ndarray:
        """Extract AC island IDs from graph nodes."""
        return np.array([graph.nodes[node][self.ac_island_attr] for node in nodes])

    def _build_ac_island_aware_distance_matrix(
        self,
        lmps: np.ndarray,
        ac_islands: np.ndarray,
        config: LMPConfig,
    ) -> np.ndarray:
        """
        Build distance matrix with AC island awareness.

        Parameters
        ----------
        lmps : np.ndarray
            Array of LMP profiles (n x m).
        ac_islands : np.ndarray
            Array of AC island IDs (n).
        config : LMPConfig
            Configuration instance.

        Returns
        -------
        np.ndarray
            Distance matrix (n x n).
        """
        # Calculate Euclidean distances between LMP profiles
        price_distances = euclidean_distances(lmps)

        # Apply AC island mask: infinite distance between different islands
        same_island_mask = ac_islands[:, np.newaxis] == ac_islands[np.newaxis, :]
        distance_matrix = np.where(same_island_mask, price_distances, config.infinite_distance)

        # Ensure diagonal is zero
        np.fill_diagonal(distance_matrix, 0.0)

        return distance_matrix

    def _run_clustering(
        self, graph: nx.Graph, lmps: np.ndarray, config: LMPConfig, **kwargs
    ) -> np.ndarray:
        """
        Dispatch to appropriate clustering algorithm.

        Parameters
        ----------
        graph : nx.Graph
            Original network graph.
        lmps : np.ndarray
            Array of LMP profiles.
        config : LMPConfig
            Configuration parameters.
        **kwargs : dict
            Additional clustering parameters.

        Returns
        -------
        np.ndarray
            Array of cluster labels.
        """
        if self.algorithm == "hierarchical":
            return self._hierarchical_clustering(graph, lmps, config, **kwargs)
        else:
            raise PartitioningError(
                f"Unknown algorithm: {self.algorithm}",
                strategy=self._get_strategy_name(),
            )

    def _hierarchical_clustering(
        self, graph: nx.Graph, lmps: np.ndarray, config: LMPConfig, **kwargs
    ) -> np.ndarray:
        """
        Perform Hierarchical Clustering on LMP profiles.

        Parameters
        ----------
        graph : nx.Graph
            Original network graph.
        lmps : np.ndarray
            Array of LMP profiles.
        config : LMPConfig
            Configuration parameters.
        **kwargs : dict
            Must include 'n_clusters'. May include 'ac_islands'.

        Returns
        -------
        np.ndarray
            Array of cluster labels.
        """
        n_clusters = kwargs.get("n_clusters")
        if n_clusters is None or n_clusters <= 0:
            raise PartitioningError(
                "Hierarchical clustering requires a positive 'n_clusters' parameter.",
                strategy=self._get_strategy_name(),
            )

        ac_islands = kwargs.get("ac_islands")

        if ac_islands is not None:
            distance_matrix = self._build_ac_island_aware_distance_matrix(lmps, ac_islands, config)
        else:
            distance_matrix = euclidean_distances(lmps)

        # Handle connectivity constraint
        connectivity = None
        if config.use_connectivity:
            log_debug(
                "Using graph connectivity constraint for clustering", LogCategory.PARTITIONING
            )
            connectivity = nx.adjacency_matrix(graph)

        return run_hierarchical(
            distance_matrix,
            n_clusters,
            config.hierarchical_linkage,
            connectivity=connectivity,
        )
