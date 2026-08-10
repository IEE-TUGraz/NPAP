import heapq
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from npap.exceptions import PartitioningError
from npap.interfaces import PartitioningStrategy
from npap.logging import LogCategory, log_debug, log_info
from npap.utils import (
    create_partition_map,
    validate_partition,
    validate_required_attributes,
    with_runtime_config,
)


@dataclass
class _Cluster:
    """
    Internal cluster representation for demand-weighted agglomerative clustering.

    Attributes
    ----------
    id : int
        Unique cluster identifier.
    W : float
        Total cluster demand weight (sum_i d_i).
    mu : np.ndarray
        Demand-weighted mean LMP vector of the cluster.
    members : list[int]
        List of node indices in the cluster.
    neighbors : set[int]
        Set of active adjacent cluster IDs.
    version : int
        Version stamp for lazy heap invalidation.
    """

    id: int
    W: float
    mu: np.ndarray
    members: list[int]
    neighbors: set[int]
    version: int = 0


@dataclass
class LMPConfig:
    """
    Configuration parameters for LMP-based partitioning.

    Attributes
    ----------
    hierarchical_linkage : str
        Linkage criterion for hierarchical clustering
        ('ward', 'complete', 'average', 'single').
        Note: 'single' and 'complete' linkages ignore node weights by construction.
    infinite_distance : float
        Value used to represent "infinite" distance between nodes in different
        AC islands.
    use_connectivity : bool
        Whether to use graph connectivity as a constraint for clustering.
        If True, only adjacent nodes in the graph can be merged into a cluster.
        It is recommended to set this to True, since otherwise nodes might get clustered, which are at different positions in the network.
    distance_threshold : float, optional
        Distance threshold for hierarchical clustering based on tolerated RMS
        (quadratic mean) price difference per timestep. If specified, 'n_clusters' must be None.
    demand_attr : str
        Node attribute name containing demand values for weighting.
    use_demand_weighting : bool
        Whether to weight nodes by their demand during clustering.
    """

    hierarchical_linkage: str = "ward"
    infinite_distance: float = 1e4
    use_connectivity: bool = True
    distance_threshold: float | None = None
    demand_attr: str = "p_load"
    use_demand_weighting: bool = False


class LMPPartitioning(PartitioningStrategy):
    """
    Partition nodes based on Locational Marginal Prices (LMP).

    This strategy clusters nodes based on their LMP values and nodal demand weights.
    LMPs can be single values or time-series profiles (vectors). The clustering
    greedy minimizes the increase of weighted SSE = sum_i d_i * ||lambda_i - mu_c(i)||^2,
    where mu_c is the demand-weighted mean LMP of the zone. Representing each zone
    by mu_c preserves the total load payment exactly: sum_i d_i * lambda_i == sum_c D_c * mu_c.

    AC-Island Awareness:
        When the graph contains AC island data (nodes have 'ac_island' attribute),
        the strategy respects AC island boundaries by removing inter-island edges
        from the adjacency matrix before clustering.

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
        "distance_threshold",
        "demand_attr",
        "use_demand_weighting",
    }

    def __init__(
        self,
        algorithm: str = "hierarchical",
        lmp_attr: str = "lmp",
        demand_attr: str = "demand",
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
        demand_attr : str, default='demand'
            Node attribute name containing demand values for weighting.
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
        self.demand_attr = demand_attr
        self.distance_metric = distance_metric
        self.ac_island_attr = ac_island_attr
        self.config = config or LMPConfig(demand_attr=demand_attr)

        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm: {algorithm}. "
                f"Supported: {', '.join(self.SUPPORTED_ALGORITHMS)}"
            )

        log_debug(
            f"Initialized LMPPartitioning: algorithm={algorithm}, lmp_attr={lmp_attr}, demand_attr={demand_attr}",
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

            - n_clusters : Number of clusters (required if distance_threshold is None)
            - distance_threshold : Distance threshold per timestep (RMS) for hierarchical clustering
            - config : LMPConfig instance to override instance config
            - hierarchical_linkage : Override config parameter
            - infinite_distance : Override config parameter
            - use_connectivity : Override config parameter
            - demand_attr : Override config parameter
            - use_demand_weighting : Override config parameter

        Returns
        -------
        dict[int, list[Any]]
            Dictionary mapping cluster_id -> list of node_ids.

        Raises
        ------
        ValueError
            If graph has more connected components than requested n_clusters.
        PartitioningError
            If partitioning fails.
        """
        try:
            effective_config = kwargs.get("_effective_config", self.config)
            n_clusters = kwargs.get("n_clusters")
            distance_threshold = kwargs.get(
                "distance_threshold", effective_config.distance_threshold
            )

            # Extract LMP and Demand data
            nodes = list(graph.nodes())
            lmps = self._extract_lmps(graph, nodes)
            demands = self._extract_demands(graph, nodes, effective_config.demand_attr)

            log_param_str = (
                f"n_clusters={n_clusters}"
                if n_clusters is not None
                else f"distance_threshold={distance_threshold}"
            )

            # Auto-detect AC island data
            ac_islands = None
            has_ac_islands = self._has_ac_island_data(graph, nodes)

            if has_ac_islands:
                ac_islands = self._extract_ac_islands(graph, nodes)
                n_ac_islands = len(set(ac_islands))

                log_info(
                    f"Starting AC-island-aware locational marginal prices partitioning: {self.algorithm}, "
                    f"{log_param_str}, metric={self.distance_metric}, "
                    f"ac_islands={n_ac_islands}",
                    LogCategory.PARTITIONING,
                )
            else:
                log_info(
                    f"Starting LMP partitioning: {self.algorithm}, {log_param_str}",
                    LogCategory.PARTITIONING,
                )

            # Perform clustering
            labels = self._run_clustering(
                graph, lmps, demands, effective_config, ac_islands=ac_islands, **kwargs
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
            if isinstance(e, (PartitioningError, ValueError)):
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

        Supports both scalar LMPs and time-series profiles.

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
            if isinstance(val, (int, float, np.number)):
                lmp_profiles.append([float(val)])
            else:
                lmp_profiles.append(np.atleast_1d(val))

        return np.array(lmp_profiles, dtype=np.float64)

    def _extract_demands(
        self, graph: nx.Graph, nodes: list[Any], demand_attr: str
    ) -> np.ndarray:
        """
        Extract demand values from graph nodes for weighting.

        Defaults to 1.0 if demand attribute is missing on a node.

        Parameters
        ----------
        graph : nx.Graph
            NetworkX graph.
        nodes : list[Any]
            List of node IDs.
        demand_attr : str
            Node attribute name for demand.

        Returns
        -------
        np.ndarray
            Array of node demand weights (n_nodes).
        """
        demands = []
        for node in nodes:
            val = graph.nodes[node].get(demand_attr, 1.0)
            if isinstance(val, (int, float, np.number)):
                d = float(val)
            else:
                d = float(np.mean(val))
            demands.append(max(d, 1e-12))
        return np.array(demands, dtype=np.float64)

    def _has_ac_island_data(self, graph: nx.Graph, nodes: list[Any]) -> bool:
        """Check if the graph has AC island data on nodes."""
        for node in nodes:
            if self.ac_island_attr not in graph.nodes[node]:
                return False
        return True

    def _extract_ac_islands(self, graph: nx.Graph, nodes: list[Any]) -> np.ndarray:
        """Extract AC island IDs from graph nodes."""
        return np.array([graph.nodes[node][self.ac_island_attr] for node in nodes])

    def _build_adjacency_graph(
        self,
        graph: nx.Graph,
        nodes: list[Any],
        use_connectivity: bool,
        ac_islands: np.ndarray | None = None,
    ) -> nx.Graph:
        """
        Build adjacency graph for connectivity-constrained clustering with AC-island awareness.

        Parameters
        ----------
        graph : nx.Graph
            Original network graph.
        nodes : list[Any]
            List of node IDs.
        use_connectivity : bool
            Whether to restrict merges to graph neighbors.
        ac_islands : np.ndarray | None
            Array of AC island IDs.

        Returns
        -------
        nx.Graph
            Adjacency graph where node identifiers are 0..n-1 indices.
        """
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        n_nodes = len(nodes)
        adj_graph = nx.Graph()
        adj_graph.add_nodes_from(range(n_nodes))

        if not use_connectivity:
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    if ac_islands is None or ac_islands[i] == ac_islands[j]:
                        adj_graph.add_edge(i, j)
            return adj_graph

        for u, v in graph.edges():
            if u in node_to_idx and v in node_to_idx:
                idx_u = node_to_idx[u]
                idx_v = node_to_idx[v]
                if ac_islands is None or ac_islands[idx_u] == ac_islands[idx_v]:
                    adj_graph.add_edge(idx_u, idx_v)

        return adj_graph

    def _run_clustering(
        self,
        graph: nx.Graph,
        lmps: np.ndarray,
        demands: np.ndarray,
        config: LMPConfig,
        **kwargs,
    ) -> np.ndarray:
        """
        Dispatch to appropriate clustering algorithm.

        Parameters
        ----------
        graph : nx.Graph
            Original network graph.
        lmps : np.ndarray
            Array of LMP profiles.
        demands : np.ndarray
            Array of demand weights.
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
            return self._hierarchical_clustering(graph, lmps, demands, config, **kwargs)
        else:
            raise PartitioningError(
                f"Unknown algorithm: {self.algorithm}",
                strategy=self._get_strategy_name(),
            )

    def _hierarchical_clustering(
        self,
        graph: nx.Graph,
        lmps: np.ndarray,
        demands: np.ndarray,
        config: LMPConfig,
        **kwargs,
    ) -> np.ndarray:
        """
        Perform demand-weighted connected agglomerative clustering on LMP profiles.

        Parameters
        ----------
        graph : nx.Graph
            Original network graph.
        lmps : np.ndarray
            Array of LMP profiles.
        demands : np.ndarray
            Array of demand weights.
        config : LMPConfig
            Configuration parameters.
        **kwargs : dict
            May include 'n_clusters', 'distance_threshold', or 'ac_islands'.

        Returns
        -------
        np.ndarray
            Array of cluster labels.

        Raises
        ------
        ValueError
            If graph has more connected components than requested n_clusters.
        PartitioningError
            If clustering parameters are invalid or clustering fails.
        """
        n_clusters = kwargs.get("n_clusters")
        distance_threshold = kwargs.get("distance_threshold", config.distance_threshold)

        if n_clusters is None and distance_threshold is None:
            raise PartitioningError(
                "Hierarchical clustering requires either 'n_clusters' or 'distance_threshold' parameter.",
                strategy=self._get_strategy_name(),
            )
        if n_clusters is not None and distance_threshold is not None:
            raise PartitioningError(
                "Hierarchical clustering cannot take both 'n_clusters' and 'distance_threshold'.",
                strategy=self._get_strategy_name(),
            )

        n_nodes = len(lmps)
        if n_nodes == 0:
            return np.array([], dtype=int)

        nodes = list(graph.nodes())
        ac_islands = kwargs.get("ac_islands")

        adj_graph = self._build_adjacency_graph(
            graph, nodes, config.use_connectivity, ac_islands
        )

        n_components = nx.number_connected_components(adj_graph)
        if n_clusters is not None and n_components > n_clusters:
            raise ValueError(
                f"Graph has {n_components} connected components, which is greater than requested n_clusters={n_clusters}."
            )

        if n_clusters == n_nodes:
            return np.arange(n_nodes)

        scaled_distance_threshold = None
        if distance_threshold is not None:
            if distance_threshold < 0:
                raise PartitioningError(
                    "'distance_threshold' must be non-negative.",
                    strategy=self._get_strategy_name(),
                )
            n_timesteps = lmps.shape[1] if lmps.ndim > 1 else 1
            scaled_distance_threshold = distance_threshold * np.sqrt(n_timesteps)

        linkage = config.hierarchical_linkage
        valid_linkages = {"ward", "complete", "average", "single"}
        if linkage not in valid_linkages:
            raise PartitioningError(
                f"Unsupported linkage '{linkage}'. Valid options: {', '.join(valid_linkages)}.",
                strategy=self._get_strategy_name(),
            )

        # Initialize cluster objects
        clusters: dict[int, _Cluster] = {}
        for i in range(n_nodes):
            w = demands[i] if config.use_demand_weighting else 1.0
            mu = lmps[i].copy()
            neighbors = set(adj_graph.neighbors(i))
            clusters[i] = _Cluster(
                id=i,
                W=w,
                mu=mu,
                members=[i],
                neighbors=neighbors,
            )

        active_clusters = set(range(n_nodes))
        linkage_dist: dict[tuple[int, int], float] = {}

        def compute_pair_cost(u: int, v: int) -> float:
            c_u = clusters[u]
            c_v = clusters[v]
            if linkage == "ward":
                diff = c_u.mu - c_v.mu
                sq_dist = float(np.sum(diff * diff))
                return (c_u.W * c_v.W / (c_u.W + c_v.W)) * sq_dist
            else:
                pair = (min(u, v), max(u, v))
                if pair in linkage_dist:
                    return linkage_dist[pair]
                lmps_u = lmps[c_u.members]
                lmps_v = lmps[c_v.members]
                dists = euclidean_distances(lmps_u, lmps_v)
                if linkage == "single":
                    cost = float(np.min(dists))
                elif linkage == "complete":
                    cost = float(np.max(dists))
                else:  # average
                    cost = float(np.mean(dists))
                linkage_dist[pair] = cost
                return cost

        # Priority queue over costs of adjacent cluster pairs
        pq: list[tuple[float, int, int, int, int]] = []
        for u, v in adj_graph.edges():
            if u < v:
                cost_uv = compute_pair_cost(u, v)
                heapq.heappush(
                    pq,
                    (cost_uv, u, v, clusters[u].version, clusters[v].version),
                )

        next_cluster_id = n_nodes
        target_clusters = n_clusters if n_clusters is not None else 1

        while len(active_clusters) > target_clusters:
            if not pq:
                break

            cost, u, v, ver_u, ver_v = heapq.heappop(pq)

            # Lazy invalidation check
            if u not in active_clusters or v not in active_clusters:
                continue
            if clusters[u].version != ver_u or clusters[v].version != ver_v:
                continue

            if scaled_distance_threshold is not None and cost > scaled_distance_threshold:
                break

            # Merge u, v into new cluster k
            k = next_cluster_id
            next_cluster_id += 1

            c_u = clusters[u]
            c_v = clusters[v]

            w_k = c_u.W + c_v.W
            mu_k = (c_u.W * c_u.mu + c_v.W * c_v.mu) / w_k
            members_k = c_u.members + c_v.members
            neighbors_k = (c_u.neighbors | c_v.neighbors) - {u, v}

            c_k = _Cluster(
                id=k,
                W=w_k,
                mu=mu_k,
                members=members_k,
                neighbors=neighbors_k,
            )
            clusters[k] = c_k

            active_clusters.remove(u)
            active_clusters.remove(v)
            active_clusters.add(k)

            # Update neighbor relationships and push new costs to heap
            for nbr in neighbors_k:
                c_nbr = clusters[nbr]
                c_nbr.neighbors.discard(u)
                c_nbr.neighbors.discard(v)
                c_nbr.neighbors.add(k)

                p_uk = (min(k, nbr), max(k, nbr))
                if linkage == "ward":
                    cost_kn = compute_pair_cost(k, nbr)
                else:
                    p_un = (min(u, nbr), max(u, nbr))
                    p_vn = (min(v, nbr), max(v, nbr))
                    has_un = p_un in linkage_dist
                    has_vn = p_vn in linkage_dist

                    if linkage == "single":
                        if has_un and has_vn:
                            linkage_dist[p_uk] = min(linkage_dist[p_un], linkage_dist[p_vn])
                        elif has_un:
                            linkage_dist[p_uk] = linkage_dist[p_un]
                        elif has_vn:
                            linkage_dist[p_uk] = linkage_dist[p_vn]
                        else:
                            compute_pair_cost(k, nbr)
                    elif linkage == "complete":
                        if has_un and has_vn:
                            linkage_dist[p_uk] = max(linkage_dist[p_un], linkage_dist[p_vn])
                        else:
                            compute_pair_cost(k, nbr)
                    elif linkage == "average":
                        if has_un and has_vn:
                            linkage_dist[p_uk] = (c_u.W * linkage_dist[p_un] + c_v.W * linkage_dist[p_vn]) / w_k
                        else:
                            compute_pair_cost(k, nbr)
                    cost_kn = linkage_dist[p_uk]

                heapq.heappush(
                    pq,
                    (cost_kn, min(k, nbr), max(k, nbr), c_k.version, c_nbr.version),
                )

            del clusters[u]
            del clusters[v]

        # Construct result array of labels
        labels = np.zeros(n_nodes, dtype=int)
        for cluster_label, cid in enumerate(sorted(active_clusters)):
            for member_idx in clusters[cid].members:
                labels[member_idx] = cluster_label

        return labels

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
        price_distances = euclidean_distances(lmps)
        same_island_mask = ac_islands[:, np.newaxis] == ac_islands[np.newaxis, :]
        distance_matrix = np.where(same_island_mask, price_distances, config.infinite_distance)
        np.fill_diagonal(distance_matrix, 0.0)
        return distance_matrix


