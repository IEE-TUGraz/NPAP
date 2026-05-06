from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
from networkx.linalg.graphmatrix import adjacency_matrix
from scipy.linalg import LinAlgError, solve

from npap.exceptions import PartitioningError, ValidationError
from npap.interfaces import EdgeType, PartitioningStrategy
from npap.logging import LogCategory, log_debug, log_info
from npap.utils import (
    create_partition_map,
    run_kmedoids,
    run_hierarchical,
    validate_partition,
    with_runtime_config,
)

from npap.logging import log_warning


@dataclass
class InputDataCongestionSensitiveConfig:
    """
    Configuration parameters for input data congestion sensitive partitioning.

    Attributes
    ----------
    zero_reactance_replacement : float
        Reactance value used when edge reactance is zero.
    regularization_factor : float
        Small value added to B matrix diagonal for numerical stability.
    infinite_distance : float
        Value used to represent "infinite" distance between AC islands.
    """

    zero_reactance_replacement: float = 1e-5
    regularization_factor: float = 1e-10
    infinite_distance: float = 1e4
    use_connectivity = True


class InputDataCongestionSensitivePartitioning(PartitioningStrategy):
    """
    Partition nodes based on congestion sensitivity using PTDF and network data.

    This strategy extends the electrical distance approach by incorporating
    line capacities and nodal generation/demand data to identify areas
    with similar impact on potential network congestion.

    Distance feature basis:
        F^-1 * PTDF * (SUM(max_res_gen + max_disp_gen) - min(demand))

    Where:
        - F: Line capacity (MW)
        - PTDF: Power Transfer Distribution Factor matrix
        - max_res_gen: Installed renewable capacity
        - max_disp_gen: Dispatchable generation capacity
        - demand: Nodal demand

    Configuration can be provided at:
        - Instantiation time (via `config` parameter in __init__)
        - Partition time (via `config` or individual parameters in partition())
    """

    SUPPORTED_ALGORITHMS = ["kmeans", "kmedoids", "hierarchical"]

    # Edge types that participate in AC power flow
    AC_EDGE_TYPES = {EdgeType.LINE.value, EdgeType.TRAFO.value}

    # Config parameter names for runtime override detection
    _CONFIG_PARAMS = {
        "zero_reactance_replacement",
        "regularization_factor",
        "infinite_distance",
    }

    def __init__(
        self,
        algorithm: str = "kmeans",
        slack_bus: Any | None = None,
        ac_island_attr: str = "ac_island",
        config: InputDataCongestionSensitiveConfig | None = None,
    ):
        """
        Initialize the partitioning strategy.

        Parameters
        ----------
        algorithm : str, default='kmeans'
            Clustering algorithm ('kmeans', 'kmedoids').
        slack_bus : Any, optional
            Specific node to use as slack bus, or None for auto-selection.
        ac_island_attr : str, default='ac_island'
            Node attribute name containing AC island ID.
        config : InputDataCongestionSensitiveConfig, optional
            Configuration parameters.
        """
        self.algorithm = algorithm
        self.slack_bus = slack_bus
        self.ac_island_attr = ac_island_attr
        self.config = config or InputDataCongestionSensitiveConfig()

        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm: {algorithm}. "
                f"Supported: {', '.join(self.SUPPORTED_ALGORITHMS)}"
            )

        log_debug(
            f"Initialized InputDataCongestionSensitivePartitioning: algorithm={algorithm}",
            LogCategory.PARTITIONING,
        )

    @property
    def required_attributes(self) -> dict[str, list[str]]:
        """Required attributes for congestion sensitive partitioning."""
        return {
            "nodes": ["renewable_generators", "dispatchable_generators", "demand_min"],
            "edges": ["x", "capacity"],
        }

    def _get_strategy_name(self) -> str:
        """Get descriptive strategy name for error messages."""
        return f"input_data_congestion_sensitive_{self.algorithm}"

    @with_runtime_config(InputDataCongestionSensitiveConfig, _CONFIG_PARAMS)
    def partition(self, graph: nx.DiGraph, **kwargs) -> dict[int, list[Any]]:
        """
        Partition nodes based on congestion sensitivity.

        Parameters
        ----------
        graph : nx.DiGraph
            NetworkX DiGraph with required attributes.
        **kwargs : dict
            Additional parameters:
            - n_clusters : Number of clusters (required)
            - random_state : Random seed
            - max_iter : Maximum iterations for clustering

        Returns
        -------
        dict[int, list[Any]]
            Dictionary mapping cluster_id -> list of node_ids.
        """
        try:
            # Get effective config
            effective_config = kwargs.get("_effective_config", self.config)

            # Resolve slack bus (kwargs override instance default)
            effective_slack = kwargs.get("slack_bus", self.slack_bus)

            # Validate AC island attributes
            self._validate_ac_island_attributes(graph)

            # Validate AC edges have reactance
            self._validate_ac_edge_attributes(graph)

            n_clusters = kwargs.get("n_clusters")
            use_connectivity = self.config.use_connectivity

            if use_connectivity:
                connectivity = adjacency_matrix(graph).toarray()
                kwargs["connectivity"] = connectivity

            nodes = list(graph.nodes())
            n_nodes = len(nodes)

            if n_clusters > n_nodes:
                raise PartitioningError(
                    f"Cannot create {n_clusters} clusters from {n_nodes}.",
                    strategy=self._get_strategy_name(),
                )

            log_info(
                f"Starting input data congestion sensitive partitioning: {self.algorithm}, "
                f"n_clusters={n_clusters}",
                LogCategory.PARTITIONING,
            )

            if n_clusters is None or n_clusters <= 0:
                raise PartitioningError(
                    "Partitioning requires a positive 'n_clusters' parameter.",
                    strategy=self._get_strategy_name(),
                )

            distance_matrix = self._calculate_input_data_congestion_sensitive_distance_matrix(
                graph, nodes, effective_config, effective_slack
            )
            # 3. Perform clustering
            label = self._run_clustering(distance_matrix, **kwargs)

            partition_map = create_partition_map(nodes, label)
            validate_partition(partition_map, n_nodes, self._get_strategy_name())

            # Validate AC island consistency
            self._validate_cluster_ac_island_consistency(graph, partition_map)

            log_info(
                f"Input data congestion sensitive partitioning complete: {len(partition_map)} clusters",
                LogCategory.PARTITIONING,
            )

        except Exception as e:
            if isinstance(e, (PartitioningError, ValidationError, NotImplementedError)):
                raise
            raise PartitioningError(
                f"Input data congestion sensitive partitioning failed: {e}",
                strategy=self._get_strategy_name(),
            ) from e

        return partition_map

    def _run_clustering(self, distance_matrix: np.ndarray, **kwargs) -> np.ndarray:
        """
        Dispatch to appropriate clustering algorithm.

        Parameters
        ----------
        distance_matrix : np.ndarray
            Precomputed distance matrix (n_nodes x n_nodes).
        **kwargs : dict
            Additional parameters including n_clusters, random_state, max_iter.

        Returns
        -------
        np.ndarray
            Array of cluster labels.
        """
        n_clusters = kwargs.get("n_clusters")
        random_state = kwargs.get("random_state", 42)
        max_iter = kwargs.get("max_iter", 300)

        if self.algorithm == "kmedoids":
            log_debug("Running K-medoids on distance matrix", LogCategory.PARTITIONING)
            return run_kmedoids(distance_matrix, n_clusters)
        if self.algorithm == "hierarchical":
            log_debug(
                "Running hierarchical clustering on distance matrix", LogCategory.PARTITIONING
            )
            # Handle connectivity constraint
            connectivity = None
            if self.config.use_connectivity:
                log_debug(
                    "Using graph connectivity constraint for clustering", LogCategory.PARTITIONING
                )
                connectivity = kwargs.get('connectivity')

            return run_hierarchical(distance_matrix, n_clusters, connectivity=connectivity)
        else:
            raise PartitioningError(
                f"Unknown algorithm: {self.algorithm}",
                strategy=self._get_strategy_name(),
            )

    def _calculate_input_data_congestion_sensitive_distance_matrix(
        self,
        graph: nx.DiGraph,
        nodes: list[Any],
        config: InputDataCongestionSensitiveConfig,
        slack_bus: Any | None,
    ) -> np.ndarray:
        # Group nodes by AC island
        islands = self._group_nodes_by_ac_island(graph, nodes)
        n_islands = len(islands)

        log_info(
            f"Processing {n_islands} AC island(s) for PTDF computation",
            LogCategory.PARTITIONING,
        )

        # Build node index mapping for the full matrix
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        n_nodes = len(nodes)

        # Initialize full distance matrix with infinite distances
        distance_matrix = np.full((n_nodes, n_nodes), config.infinite_distance)
        np.fill_diagonal(distance_matrix, 0.0)

        # Process each island independently
        for island_id, island_nodes in islands.items():
            log_debug(
                f"Processing AC island {island_id}: {len(island_nodes)} nodes",
                LogCategory.PARTITIONING,
            )

            # Handle single-node islands
            if len(island_nodes) == 1:
                log_debug(
                    f"Island {island_id} has single node, distance = 0",
                    LogCategory.PARTITIONING,
                )
                continue

            # Extract AC-only subgraph for this island
            ac_subgraph = self._extract_ac_subgraph(graph, island_nodes)

            # Validate island connectivity
            self._validate_island_connectivity(ac_subgraph, island_id, island_nodes)

            # Select slack bus for this island
            island_slack = self._select_slack_bus_for_island(
                ac_subgraph, island_nodes, slack_bus, island_id
            )

            # Compute PTDF-based distances for this island
            island_distances = self._compute_island_distances(
                ac_subgraph, island_nodes, island_slack, config
            )

            # Insert island distances into full matrix
            self._insert_island_distances(
                distance_matrix, island_distances, island_nodes, node_to_idx
            )

        return distance_matrix

    # =========================================================================
    # VALIDATION METHODS
    # =========================================================================

    def _validate_ac_island_attributes(self, graph: nx.DiGraph) -> None:
        """
        Validate that all nodes have the AC island attribute.

        Provides a helpful error message directing users to use va_loader
        or manually add the ac_island attribute.

        Parameters
        ----------
        graph : nx.DiGraph
            NetworkX DiGraph to validate.

        Raises
        ------
        ValidationError
            If any node is missing the ac_island attribute.
        """
        missing_nodes = []
        for node in graph.nodes():
            if self.ac_island_attr not in graph.nodes[node]:
                missing_nodes.append(node)

        if missing_nodes:
            sample = missing_nodes[:5]
            raise ValidationError(
                f"Electrical distance partitioning requires '{self.ac_island_attr}' attribute "
                f"on all nodes for AC island isolation. "
                f"{len(missing_nodes)} node(s) are missing this attribute (first few: {sample}). "
                f"Please use 'va_loader' data loading strategy to automatically detect AC islands, "
                f"or manually add the '{self.ac_island_attr}' attribute to all nodes.",
                missing_attributes={"nodes": [self.ac_island_attr]},
                strategy=self._get_strategy_name(),
            )

    def _validate_ac_edge_attributes(self, graph: nx.DiGraph) -> None:
        """
        Validate that AC edges (lines and transformers) have reactance attribute.

        DC links are excluded from this validation as they don't participate
        in AC power flow and don't require reactance.

        Parameters
        ----------
        graph : nx.DiGraph
            NetworkX DiGraph to validate.

        Raises
        ------
        ValidationError
            If any AC edge is missing the 'x' attribute.
        """
        missing_edges = []
        for u, v, data in graph.edges(data=True):
            edge_type = data.get("type", EdgeType.LINE.value)

            # Only validate AC edges
            if edge_type in self.AC_EDGE_TYPES:
                if "x" not in data:
                    missing_edges.append((u, v))

        if missing_edges:
            sample = missing_edges[:5]
            raise ValidationError(
                f"AC edges (lines/transformers) require 'x' (reactance) attribute. "
                f"{len(missing_edges)} edge(s) are missing this attribute (first few: {sample}).",
                missing_attributes={"edges": ["x"]},
                strategy=self._get_strategy_name(),
            )

    def _validate_island_connectivity(
        self, ac_subgraph: nx.DiGraph, island_id: Any, island_nodes: list[Any]
    ) -> None:
        """
        Validate that an island's AC subgraph is connected.

        Parameters
        ----------
        ac_subgraph : nx.DiGraph
            AC-only subgraph for the island.
        island_id : Any
            AC island identifier.
        island_nodes : list[Any]
            Nodes in this island.

        Raises
        ------
        PartitioningError
            If AC subgraph is not weakly connected.
        """
        if len(island_nodes) == 1:
            # Single node is trivially connected
            return

        if ac_subgraph.number_of_edges() == 0:
            raise PartitioningError(
                f"AC island {island_id} has no AC edges (lines/transformers). "
                f"Cannot compute electrical distances without AC connectivity.",
                strategy=self._get_strategy_name(),
                graph_info={"island_id": island_id, "n_nodes": len(island_nodes)},
            )

        if not nx.is_weakly_connected(ac_subgraph):
            n_components = nx.number_weakly_connected_components(ac_subgraph)
            raise PartitioningError(
                f"AC island {island_id} is not AC-connected. Found {n_components} "
                f"disconnected AC components within the island. This may indicate "
                f"missing line/transformer data or incorrect AC island assignment.",
                strategy=self._get_strategy_name(),
                graph_info={
                    "island_id": island_id,
                    "n_nodes": len(island_nodes),
                    "n_ac_edges": ac_subgraph.number_of_edges(),
                    "n_components": n_components,
                },
            )

    def _validate_cluster_ac_island_consistency(
        self, graph: nx.DiGraph, partition_map: dict[int, list[Any]]
    ) -> None:
        """
        Validate that clusters don't mix different AC islands.

        With infinite distances between AC islands, clusters should never mix
        nodes from different AC islands.

        Parameters
        ----------
        graph : nx.DiGraph
            Original NetworkX graph.
        partition_map : dict[int, list[Any]]
            Resulting partition mapping.
        """
        for cluster_id, nodes in partition_map.items():
            ac_islands_in_cluster = set()

            for node in nodes:
                ac_island = graph.nodes[node].get(self.ac_island_attr)
                if ac_island is not None:
                    ac_islands_in_cluster.add(ac_island)

            if len(ac_islands_in_cluster) > 1:
                log_warning(
                    f"Cluster {cluster_id} contains nodes from multiple AC islands: "
                    f"{ac_islands_in_cluster}. This should not happen with infinite distances.",
                    LogCategory.PARTITIONING,
                    warn_user=False,
                )

    # =========================================================================
    # MULTI AC-ISLAND ELECTRICAL DISTANCE CALCULATION
    # =========================================================================

    def _group_nodes_by_ac_island(
        self, graph: nx.DiGraph, nodes: list[Any]
    ) -> dict[Any, list[Any]]:
        """
        Group nodes by their AC island attribute.

        Parameters
        ----------
        graph : nx.DiGraph
            NetworkX DiGraph with ac_island attribute on nodes.
        nodes : list[Any]
            List of nodes to group.

        Returns
        -------
        dict[Any, list[Any]]
            Dictionary mapping island_id -> list of nodes in that island.
        """
        islands: dict[Any, list[Any]] = defaultdict(list)

        for node in nodes:
            island_id = graph.nodes[node].get(self.ac_island_attr)
            islands[island_id].append(node)

        return dict(islands)

    def _extract_ac_subgraph(self, graph: nx.DiGraph, island_nodes: list[Any]) -> nx.DiGraph:
        """
        Extract AC-only subgraph for a set of nodes.

        Includes only lines and transformers, excluding DC links.

        Parameters
        ----------
        graph : nx.DiGraph
            Full NetworkX DiGraph.
        island_nodes : list[Any]
            Nodes to include in subgraph.

        Returns
        -------
        nx.DiGraph
            Subgraph containing only AC edges between island nodes.
        """
        island_node_set = set(island_nodes)
        ac_subgraph = nx.DiGraph()

        # Add nodes with attributes
        for node in island_nodes:
            ac_subgraph.add_node(node, **graph.nodes[node])

        # Add only AC edges (lines and transformers)
        for u, v, data in graph.edges(data=True):
            if u in island_node_set and v in island_node_set:
                edge_type = data.get("type", EdgeType.LINE.value)
                if edge_type in self.AC_EDGE_TYPES:
                    ac_subgraph.add_edge(u, v, **data)

        return ac_subgraph

    @staticmethod
    def _select_slack_bus_for_island(
        ac_subgraph: nx.DiGraph,
        island_nodes: list[Any],
        user_slack: Any | None,
        island_id: Any,
    ) -> Any:
        """
        Select slack bus for a specific island.

        If user specified a slack bus that's in this island, use it.
        Otherwise, select the node with highest total degree in the AC subgraph.

        Parameters
        ----------
        ac_subgraph : nx.DiGraph
            AC-only subgraph for this island.
        island_nodes : list[Any]
            Nodes in this island.
        user_slack : Any, optional
            User-specified slack bus (may be None or in different island).
        island_id : Any
            AC island identifier for logging.

        Returns
        -------
        Any
            Selected slack bus node for this island.
        """
        island_node_set = set(island_nodes)

        # Check if user-specified slack is in this island
        if user_slack is not None and user_slack in island_node_set:
            log_debug(
                f"Using user-specified slack bus {user_slack} for island {island_id}",
                LogCategory.PARTITIONING,
            )
            return user_slack

        # Auto-select: use node with highest total degree in AC subgraph
        degrees = {n: ac_subgraph.in_degree(n) + ac_subgraph.out_degree(n) for n in island_nodes}
        selected = max(island_nodes, key=lambda n: degrees[n])

        log_debug(
            f"Auto-selected slack bus {selected} for island {island_id} (degree={degrees[selected]})",
            LogCategory.PARTITIONING,
        )

        return selected

    def _compute_island_distances(
        self,
        ac_subgraph: nx.DiGraph,
        island_nodes: list[Any],
        slack_bus: Any,
        config: InputDataCongestionSensitiveConfig,
    ) -> np.ndarray:
        # Build PTDF matrix for this island
        ptdf_matrix, active_nodes = self._build_ptdf_matrix(
            ac_subgraph, island_nodes, slack_bus, config
        )

        log_debug(
            f"Built island PTDF matrix: shape {ptdf_matrix.shape}",
            LogCategory.PARTITIONING,
        )

        # Weight PTDF matrix with additional features (line capacities, generation/demand data)
        weighted_PTDF = self._compute_weighted_ptdf(ptdf_matrix, active_nodes, ac_subgraph)

        # Add slack bus to ptdf as zero columns (since slack bus has no injections)
        n_edges, n_active = weighted_PTDF.shape
        n_total = n_active + 1  # Add slack bus back in
        weighted_ptdf_full = np.zeros((n_edges, n_total))
        active_idx = 0
        for node in island_nodes:
            if node == slack_bus:
                continue  # Skip slack bus column (remains zero)
            weighted_ptdf_full[:, active_idx] = weighted_PTDF[:, active_idx]
            active_idx += 1

        # Compute the distance matrix on the weighted PTDF
        distance_matrix_island = self._compute_weighted_ptdf_distances(weighted_ptdf_full)

        return distance_matrix_island

    @staticmethod
    def _compute_weighted_ptdf(
        ptdf: np.array, active_nodes: list[Any], ac_subgraph: nx.DiGraph
    ) -> np.array:
        """
        Apply weighting to the PTDF matrix based on line capacities and nodal data.

        This is a placeholder for the actual weighting logic, which would involve
        extracting line capacities and nodal generation/demand data, and applying
        them to the PTDF matrix according to the defined distance feature basis.

        Parameters
        ----------
        ptdf : np.ndarray
            PTDF matrix for the island (n_edges × n_active).
        active_nodes : list[Any]
            List of active nodes corresponding to PTDF columns.
        ac_subgraph: nx.DiGraph
                AC-only subgraph for this island, used to extract line capacities and nodal data.

        Returns
        -------
        np.ndarray
            Weighted PTDF matrix.
        """

        # Define vector with reciprocal line capacities (F^-1)
        capacities = []
        for _, _, data in ac_subgraph.edges(data=True):
            cap = data.get("p_max", 1.0)
            capacities.append(1 / cap if cap > 0 else 1)

        cap_inv = np.array(capacities)

        # Define vector with nodal features (SUM(max_res_gen + max_disp_gen) - min(demand))
        nodal_features = []
        for node in active_nodes:
            data = ac_subgraph.nodes[node]

            # Aggregate capacities from generator lists
            res_gen = data.get("renewable_generators", [])
            p_res = sum(g.get("p_nom", 0) for g in res_gen)

            dis_gen = data.get("dispatchable_generators", [])
            p_dis = sum(g.get("p_nom", 0) for g in dis_gen)

            p_demand = data.get("demand_min", 0)

            nodal_features.append(p_res + p_dis - p_demand)

        s_nodal = np.array(nodal_features)

        # Apply weighting to PTDF: F^-1 * PTDF * diag(nodal_features)
        weighted_ptdf = np.multiply(ptdf, cap_inv[:, np.newaxis]) @ np.diag(s_nodal)

        return weighted_ptdf

    def _compute_weighted_ptdf_distances(self, ptdf_matrix: np.ndarray) -> np.ndarray:
        """
        Compute electrical distances using Euclidean distance between PTDF columns.

        Electrical distance: d_ij = ||PTDF[:,i] - PTDF[:,j]||_2

        Uses the identity ||a-b||² = ||a||² + ||b||² - 2⟨a,b⟩ to compute all
        pairwise distances via a single matrix multiplication (Gram matrix),
        which is highly optimized by BLAS libraries.

        Parameters
        ----------
        ptdf_matrix : np.ndarray
            PTDF matrix (n_edges × n_active).

        Returns
        -------
        np.ndarray
            Distance matrix for active nodes (n_active × n_active).

        Raises
        ------
        PartitioningError
            If distance calculation produces invalid values.
        """
        # Use float32 for faster computation
        X = ptdf_matrix.T.astype(np.float32, copy=False)

        # Compute squared norms for each node's PTDF profile
        norms_sq = np.einsum("ij,ij->i", X, X)

        # Compute Gram matrix (all pairwise dot products) - BLAS optimized
        gram = X @ X.T

        # ||a - b||² = ||a||² + ||b||² - 2⟨a,b⟩
        dist_sq = norms_sq[:, np.newaxis] + norms_sq
        dist_sq -= 2.0 * gram  # In-place to save memory

        # Handle numerical errors (small negative values from floating point)
        np.maximum(dist_sq, 0.0, out=dist_sq)

        # Take square root to get actual distances
        distance_matrix = np.sqrt(dist_sq, out=dist_sq)  # In-place

        # Ensure diagonal is exactly zero
        np.fill_diagonal(distance_matrix, 0.0)

        if np.any(np.isnan(distance_matrix)):
            raise PartitioningError(
                "Distance matrix contains NaN values after PTDF distance calculation.",
                strategy=self._get_strategy_name(),
            )

        return distance_matrix.astype(np.float64)

    @staticmethod
    def _insert_island_distances(
        full_matrix: np.ndarray,
        island_distances: np.ndarray,
        island_nodes: list[Any],
        node_to_idx: dict[Any, int],
    ) -> None:
        """
        Insert island distance matrix into the full distance matrix.

        Parameters
        ----------
        full_matrix : np.ndarray
            Full distance matrix (modified in place).
        island_distances : np.ndarray
            Distance matrix for this island.
        island_nodes : list[Any]
            Ordered list of nodes in this island.
        node_to_idx : dict[Any, int]
            Mapping from node to index in full matrix.
        """
        # Build index array for island nodes in full matrix
        island_indices = np.array([node_to_idx[node] for node in island_nodes])

        # Use np.ix_ for efficient block assignment
        full_matrix[np.ix_(island_indices, island_indices)] = island_distances

    # =========================================================================
    # PTDF MATRIX CONSTRUCTION
    # =========================================================================

    def _build_ptdf_matrix(
        self,
        ac_subgraph: nx.DiGraph,
        island_nodes: list[Any],
        slack_bus: Any,
        config: InputDataCongestionSensitiveConfig,
    ) -> tuple[np.ndarray, list[Any]]:
        """
        Build the Power Transfer Distribution Factor (PTDF) matrix for a dc-island.

        PTDF = diag{b} · K_sba · (K_sba^T · diag{b} · K_sba)^(-1)

        Where:

        - K_sba is the slack-bus-adjusted incidence matrix
        - b is the vector of susceptances (1/reactance)
        - The resulting PTDF has shape (n_edges × n_active_nodes)

        Instead of computing B^(-1) explicitly, we solve the linear system
        directly which is significantly faster for large networks.

        Parameters
        ----------
        ac_subgraph : nx.DiGraph
            AC-only DiGraph for this island.
        island_nodes : list[Any]
            Ordered list of nodes in this island.
        slack_bus : Any
            Slack bus node for this island.
        config : ElectricalDistanceConfig
            Configuration instance.

        Returns
        -------
        tuple[np.ndarray, list[Any]]
            Tuple of (PTDF matrix [n_edges × n_active], list of active nodes).

        Raises
        ------
        PartitioningError
            If matrix construction fails.
        """
        try:
            # Extract edges and susceptances (AC edges only)
            edges, susceptances = self._extract_edge_susceptances(ac_subgraph, config)

            if len(edges) == 0:
                raise PartitioningError(
                    "No valid AC edges found for PTDF matrix construction.",
                    strategy=self._get_strategy_name(),
                )

            # Build slack-bus-adjusted incidence matrix
            K_sba, active_nodes = self._build_incidence_matrix(edges, island_nodes, slack_bus)
            log_debug(
                f"Built incidence matrix K_sba: shape {K_sba.shape}",
                LogCategory.PARTITIONING,
            )

            # Build susceptance matrix B = K_sba^T @ diag(b) @ K_sba
            B_matrix = self._compute_B_matrix(K_sba, susceptances)
            log_debug(f"Built B matrix: shape {B_matrix.shape}", LogCategory.PARTITIONING)

            # Compute PTDF using direct linear solve
            ptdf_matrix = self._compute_ptdf(K_sba, susceptances, B_matrix, config)

            return ptdf_matrix, active_nodes

        except Exception as e:
            if isinstance(e, PartitioningError):
                raise
            raise PartitioningError(
                f"Failed to build PTDF matrix: {e}", strategy=self._get_strategy_name()
            ) from e

    def _extract_edge_susceptances(
        self, ac_subgraph: nx.DiGraph, config: InputDataCongestionSensitiveConfig
    ) -> tuple[list[tuple[Any, Any]], np.ndarray]:
        """
        Extract AC edges and their susceptances from the subgraph.

        Only processes lines and transformers (AC edges).
        Susceptance b = 1/x where x is the reactance.

        Parameters
        ----------
        ac_subgraph : nx.DiGraph
            AC-only subgraph.
        config : ElectricalDistanceConfig
            Configuration instance.

        Returns
        -------
        tuple[list[tuple[Any, Any]], np.ndarray]
            Tuple of (list of (from, to) edges, array of susceptances).

        Raises
        ------
        PartitioningError
            If reactance values are invalid.
        """
        edges = []
        susceptances = []
        zero_reactance_count = 0

        for u, v, data in ac_subgraph.edges(data=True):
            reactance = data.get("x")

            if reactance is None:
                raise PartitioningError(
                    f"AC edge ({u}, {v}) missing reactance attribute 'x'",
                    strategy=self._get_strategy_name(),
                )

            if not isinstance(reactance, (int, float)):
                raise PartitioningError(
                    f"Edge ({u}, {v}) reactance must be numeric, got {type(reactance)}",
                    strategy=self._get_strategy_name(),
                )

            if reactance == 0:
                zero_reactance_count += 1
                reactance = config.zero_reactance_replacement

            edges.append((u, v))
            susceptances.append(1.0 / reactance)

        # Emit single warning if any zero reactance edges found
        if zero_reactance_count > 0:
            log_warning(
                f"{zero_reactance_count} edge(s) have zero reactance. "
                f"Using replacement value: {config.zero_reactance_replacement}",
                LogCategory.PARTITIONING,
            )

        return edges, np.array(susceptances)

    @staticmethod
    def _build_incidence_matrix(
        edges: list[tuple[Any, Any]], nodes: list[Any], slack_bus: Any
    ) -> tuple[np.ndarray, list[Any]]:
        """
        Build slack-bus-adjusted incidence matrix K_sba.

        For each directed edge (u → v):

        - K[edge_idx, u] = -1 (edge leaves u)
        - K[edge_idx, v] = +1 (edge enters v)

        The slack bus column is removed to make B invertible.

        Parameters
        ----------
        edges : list[tuple[Any, Any]]
            List of (from_node, to_node) tuples.
        nodes : list[Any]
            Ordered list of all nodes in island.
        slack_bus : Any
            Node to exclude (slack bus).

        Returns
        -------
        tuple[np.ndarray, list[Any]]
            Tuple of (K_sba matrix [n_edges × n_active], list of active nodes).
        """
        # Active nodes (without slack bus)
        active_nodes = [n for n in nodes if n != slack_bus]
        n_active = len(active_nodes)
        n_edges = len(edges)

        # Create node index mapping for active nodes
        node_to_idx = {node: idx for idx, node in enumerate(active_nodes)}

        # Build incidence matrix using vectorized assignment
        K_sba = np.zeros((n_edges, n_active))

        # Pre-compute indices for all edges (-1 means node not in active set)
        u_indices = np.array([node_to_idx.get(u, -1) for u, v in edges])
        v_indices = np.array([node_to_idx.get(v, -1) for u, v in edges])
        edge_indices = np.arange(n_edges)

        # Vectorized assignment for valid source nodes
        valid_u = u_indices >= 0
        K_sba[edge_indices[valid_u], u_indices[valid_u]] = -1.0

        # Vectorized assignment for valid target nodes
        valid_v = v_indices >= 0
        K_sba[edge_indices[valid_v], v_indices[valid_v]] = 1.0

        return K_sba, active_nodes

    @staticmethod
    def _compute_B_matrix(K_sba: np.ndarray, susceptances: np.ndarray) -> np.ndarray:
        """
        Compute susceptance matrix B = K_sba^T @ diag(b) @ K_sba.

        Uses efficient broadcasting: B = (K^T * b) @ K, avoiding explicit
        diagonal matrix construction.

        Parameters
        ----------
        K_sba : np.ndarray
            Slack-bus-adjusted incidence matrix (n_edges × n_active).
        susceptances : np.ndarray
            Array of susceptance values (n_edges,).

        Returns
        -------
        np.ndarray
            Symmetric susceptance matrix (n_active × n_active).
        """
        # Efficient: (K.T * b) @ K where b is broadcast along columns
        # K.T shape: (n_active, n_edges), susceptances shape: (n_edges,)
        K_scaled = K_sba.T * susceptances
        B_matrix = K_scaled @ K_sba

        # Ensure symmetry (handles floating point errors)
        return (B_matrix + B_matrix.T) / 2.0

    @staticmethod
    def _compute_ptdf(
        K_sba: np.ndarray,
        susceptances: np.ndarray,
        B_matrix: np.ndarray,
        config: InputDataCongestionSensitiveConfig,
    ) -> np.ndarray:
        """
        Compute PTDF matrix by solving linear system directly.

        To avoid computing B^(-1) explicitly, we solve:
        B @ X = A^T  where A = diag(b) @ K_sba
        Then PTDF = X^T

        This is mathematically equivalent but 3-5x faster for large matrices.

        Parameters
        ----------
        K_sba : np.ndarray
            Slack-bus-adjusted incidence matrix (n_edges × n_active).
        susceptances : np.ndarray
            Array of susceptance values (n_edges,).
        B_matrix : np.ndarray
            Susceptance matrix (n_active × n_active).
        config : ElectricalDistanceConfig
            Configuration instance.

        Returns
        -------
        np.ndarray
            PTDF matrix (n_edges × n_active).
        """
        # Apply Tikhonov regularization for numerical stability
        if config.regularization_factor > 0:
            B_matrix = B_matrix + config.regularization_factor * np.eye(B_matrix.shape[0])

        # A = diag(b) @ K_sba, shape (n_edges, n_active)
        A = susceptances[:, np.newaxis] * K_sba

        try:
            # Solve B @ X = A^T for X, then PTDF = X^T
            # Using assume_a='sym' tells scipy B is symmetric -> uses faster algorithm
            X = solve(B_matrix, A.T, assume_a="sym")
            ptdf_matrix = X.T

        except LinAlgError:
            # Fallback to least-squares solution if matrix is singular
            log_warning(
                "B matrix is singular, using least-squares solution.",
                LogCategory.PARTITIONING,
            )
            X, _, _, _ = np.linalg.lstsq(B_matrix, A.T, rcond=None)
            ptdf_matrix = X.T

        return ptdf_matrix
