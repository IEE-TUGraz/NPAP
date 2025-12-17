from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import networkx as nx
import numpy as np
from scipy import sparse

from npap.exceptions import PartitioningError
from npap.interfaces import PartitioningStrategy
from npap.logging import log_debug, log_info, log_warning, LogCategory
from npap.utils import (
    with_runtime_config,
    create_partition_map, validate_partition,
    run_kmeans, run_kmedoids, validate_required_attributes
)


@dataclass
class ElectricalDistanceConfig:
    """
    Configuration parameters for electrical distance calculations.

    Centralizes all magic numbers and tolerances used in the electrical
    distance partitioning algorithm for better maintainability and tuning.

    Attributes:
        zero_reactance_replacement: Reactance value used when edge reactance is zero
        slack_distance_fallback: Default distance value when no valid distances exist
        negative_distance_threshold: Threshold for warning about negative distance squared
        use_sparse: Whether to use sparse matrices for large networks
        sparse_threshold: Number of nodes above which sparse matrices are used
        condition_number_threshold: Condition number threshold for B matrix inversion
    """
    zero_reactance_replacement: float = 1e-5
    slack_distance_fallback: float = 1.0
    negative_distance_threshold: float = -1e-10
    use_sparse: bool = True
    sparse_threshold: int = 500
    condition_number_threshold: float = 1e12


class ElectricalDistancePartitioning(PartitioningStrategy):
    """
    Partition nodes based on electrical distance in power networks.

    This strategy uses the Power Transfer Distribution Factor (PTDF) approach
    to calculate electrical distances between nodes. The electrical distance
    is derived from the network's reactance/susceptance matrix and represents
    the electrical coupling between nodes.

    Mathematical basis:
    - Build incidence matrix K from directed network topology
    - Calculate susceptance matrix B = K^T · diag{b} · K
    - Electrical distance: d_ij = sqrt((B^-1_ii + B^-1_jj - 2*B^-1_ij))

    Configuration can be provided at:
    - Instantiation time (via `config` parameter in __init__)
    - Partition time (via `config` or individual parameters in partition())

    Partition-time parameters override instance defaults for that call only.
    """

    SUPPORTED_ALGORITHMS = ['kmeans', 'kmedoids']

    # Config parameter names for runtime override detection
    _CONFIG_PARAMS = {
        'zero_reactance_replacement',
        'slack_distance_fallback',
        'negative_distance_threshold',
        'use_sparse',
        'sparse_threshold',
        'condition_number_threshold'
    }

    def __init__(self, algorithm: str = 'kmeans', slack_bus: Optional[Any] = None,
                 config: Optional[ElectricalDistanceConfig] = None):
        """
        Initialize electrical distance partitioning strategy.

        Args:
            algorithm: Clustering algorithm ('kmeans', 'kmedoids')
            slack_bus: Specific node to use as slack bus, or None for auto-selection
            config: Configuration parameters for distance calculations

        Raises:
            ValueError: If unsupported algorithm is specified
        """
        self.algorithm = algorithm
        self.slack_bus = slack_bus
        self.config = config or ElectricalDistanceConfig()

        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm: {algorithm}. "
                f"Supported: {', '.join(self.SUPPORTED_ALGORITHMS)}"
            )

        log_debug(
            f"Initialized ElectricalDistancePartitioning: algorithm={algorithm}",
            LogCategory.PARTITIONING
        )

    @property
    def required_attributes(self) -> Dict[str, List[str]]:
        """Required attributes for electrical distance partitioning."""
        return {
            'nodes': [],
            'edges': ['x']  # Reactance attribute required on edges
        }

    def _get_strategy_name(self) -> str:
        """Get descriptive strategy name for error messages."""
        return f"electrical_{self.algorithm}"

    @with_runtime_config(ElectricalDistanceConfig, _CONFIG_PARAMS)
    @validate_required_attributes
    def partition(self, graph: nx.DiGraph, **kwargs) -> Dict[int, List[Any]]:
        """
        Partition nodes based on electrical distance.

        Args:
            graph: NetworkX DiGraph with reactance data on edges
            **kwargs: Additional parameters
                - n_clusters: Number of clusters (required)
                - random_state: Random seed for reproducibility
                - max_iter: Maximum iterations for clustering
                - config: ElectricalDistanceConfig instance to override instance config
                - slack_bus: Override the slack bus for this partition call
                - zero_reactance_replacement: Override config parameter
                - slack_distance_fallback: Override config parameter
                - negative_distance_threshold: Override config parameter
                - use_sparse: Override config parameter
                - sparse_threshold: Override config parameter

        Returns:
            Dictionary mapping cluster_id -> list of node_ids

        Raises:
            PartitioningError: If partitioning fails
        """
        try:
            # Get effective config (injected by decorator)
            effective_config = kwargs.get('_effective_config', self.config)

            # Resolve slack bus (kwargs override instance default)
            effective_slack = kwargs.get('slack_bus', self.slack_bus)

            self._validate_network_connectivity(graph)

            n_clusters = kwargs.get('n_clusters')

            log_info(
                f"Starting electrical distance partitioning: {self.algorithm}, n_clusters={n_clusters}",
                LogCategory.PARTITIONING
            )

            self._validate_network_connectivity(graph)

            if n_clusters is None or n_clusters <= 0:
                raise PartitioningError(
                    "Electrical distance partitioning requires a positive 'n_clusters' parameter.",
                    strategy=self._get_strategy_name()
                )

            nodes = list(graph.nodes())
            n_nodes = len(nodes)

            if n_clusters > n_nodes:
                raise PartitioningError(
                    f"Cannot create {n_clusters} clusters from {n_nodes} nodes.",
                    strategy=self._get_strategy_name()
                )

            # Calculate electrical distance matrix
            log_debug("Computing electrical distance matrix", LogCategory.PARTITIONING)
            distance_matrix = self._calculate_electrical_distance_matrix(
                graph, nodes, effective_config, effective_slack
            )

            # Perform clustering
            labels = self._run_clustering(distance_matrix, **kwargs)

            # Create and validate partition
            partition_map = create_partition_map(nodes, labels)
            validate_partition(partition_map, n_nodes, self._get_strategy_name())

            log_info(
                f"Electrical partitioning complete: {len(partition_map)} clusters",
                LogCategory.PARTITIONING
            )

            return partition_map

        except Exception as e:
            if isinstance(e, PartitioningError):
                raise
            raise PartitioningError(
                f"Electrical distance partitioning failed: {e}",
                strategy=self._get_strategy_name(),
                graph_info={'nodes': len(list(graph.nodes())), 'edges': len(graph.edges())}
            ) from e

    def _run_clustering(self, distance_matrix: np.ndarray, **kwargs) -> np.ndarray:
        """Dispatch to appropriate clustering algorithm."""
        n_clusters = kwargs.get('n_clusters')
        random_state = kwargs.get('random_state', 42)
        max_iter = kwargs.get('max_iter', 300)

        if self.algorithm == 'kmeans':
            log_debug(f"Running K-means on distance matrix", LogCategory.PARTITIONING)
            return run_kmeans(distance_matrix, n_clusters, random_state, max_iter)
        elif self.algorithm == 'kmedoids':
            log_debug(f"Running K-medoids on distance matrix", LogCategory.PARTITIONING)
            return run_kmedoids(distance_matrix, n_clusters)
        else:
            raise PartitioningError(
                f"Unknown algorithm: {self.algorithm}",
                strategy=self._get_strategy_name()
            )

    def _validate_network_connectivity(self, graph: nx.DiGraph) -> None:
        """
        Validate network connectivity using weak connectivity for directed graphs.

        For electrical networks, we check weak connectivity (ignoring edge direction)
        because electrical coupling exists regardless of defined edge direction.

        Args:
            graph: NetworkX DiGraph to validate

        Raises:
            PartitioningError: If graph is not weakly connected
        """
        if not nx.is_weakly_connected(graph):
            n_components = nx.number_weakly_connected_components(graph)
            raise PartitioningError(
                f"Graph must be connected for electrical distance partitioning. "
                f"Found {n_components} disconnected components (islands). ",
                strategy=self._get_strategy_name(),
                graph_info={
                    'nodes': len(list(graph.nodes())),
                    'edges': len(graph.edges()),
                    'n_components': n_components
                }
            )

    def _calculate_electrical_distance_matrix(self, graph: nx.DiGraph, nodes: List[Any],
                                              config: ElectricalDistanceConfig, slack_bus: Optional[Any]) -> np.ndarray:
        """
        Calculate electrical distance matrix between all node pairs.

        Orchestrates the calculation by:
        1. Building the susceptance matrix (B matrix)
        2. Inverting the B matrix to get impedance matrix
        3. Computing distances from the impedance matrix
        4. Integrating slack bus distances

        Args:
            graph: NetworkX DiGraph with reactance on edges
            nodes: Ordered list of nodes
            config: ElectricalDistanceConfig instance
            slack_bus: Optional specified slack bus

        Returns:
            Symmetric distance matrix (n_nodes × n_nodes)

        Raises:
            PartitioningError: If distance matrix calculation fails
        """
        selected_slack = self._select_slack_bus(graph, nodes, slack_bus)
        log_debug(f"Selected slack bus: {selected_slack}", LogCategory.PARTITIONING)

        # Build susceptance matrix
        B_matrix, active_nodes = self._build_susceptance_matrix(
            graph, nodes, selected_slack, config
        )
        log_debug(f"Built B matrix: shape {B_matrix.shape}", LogCategory.PARTITIONING)

        # Invert susceptance matrix
        B_inv = self._invert_susceptance_matrix(B_matrix)

        # Calculate distances
        distance_matrix_active = self._compute_distances_vectorized(B_inv, config)

        # Integrate slack bus
        distance_matrix_full = self._integrate_slack_bus_distances(
            distance_matrix_active, nodes, selected_slack, active_nodes, config
        )

        return distance_matrix_full

    def _select_slack_bus(self, graph: nx.DiGraph, nodes: List[Any],
                          slack_bus: Optional[Any]) -> Any:
        """
        Select slack bus node.

        If slack_bus is specified, use it. Otherwise, select the node
        with the highest total degree (in-degree + out-degree).

        Args:
            graph: NetworkX DiGraph
            nodes: List of nodes
            slack_bus: Optional specified slack bus

        Returns:
            Selected slack bus node

        Raises:
            PartitioningError: If specified slack bus not found
        """
        if slack_bus is not None:
            if slack_bus not in nodes:
                raise PartitioningError(
                    f"Specified slack bus {slack_bus} not found in graph.",
                    strategy=self._get_strategy_name()
                )
            return slack_bus

        # Use total degree (in + out) for directed graphs
        degrees = {n: graph.in_degree(n) + graph.out_degree(n) for n in nodes}
        return max(nodes, key=lambda n: degrees[n])

    def _build_susceptance_matrix(self, graph: nx.DiGraph, nodes: List[Any],
                                  slack_bus: Any, config: ElectricalDistanceConfig) -> Tuple[np.ndarray, List[Any]]:
        """
        Build the susceptance matrix (B matrix) for the network.

        Each directed edge in the graph is treated as a unique electrical
        element. The incidence matrix K encodes edge directions:
        - K[edge, from_node] = -1 (edge leaves node)
        - K[edge, to_node] = +1 (edge enters node)

        The B matrix is calculated as: B = K^T · diag{b} · K
        This naturally produces a symmetric matrix.

        Args:
            graph: NetworkX DiGraph with reactance on edges
            nodes: Ordered list of all nodes
            slack_bus: Node designated as slack bus
            config: ElectricalDistanceConfig instance

        Returns:
            Tuple of (B_matrix, active_nodes)

        Raises:
            PartitioningError: If matrix construction fails
        """
        try:
            # Extract edges and susceptances
            edges, susceptances = self._extract_edge_susceptances(graph, config)

            if len(edges) == 0:
                raise PartitioningError(
                    "No valid edges found for B matrix construction.",
                    strategy=self._get_strategy_name()
                )

            # Build incidence matrix (slack-bus-adjusted)
            K_sba, active_nodes = self._build_incidence_matrix(edges, nodes, slack_bus)

            # Determine whether to use sparse matrices
            n_active = len(active_nodes)
            use_sparse = config.use_sparse and n_active > config.sparse_threshold

            if use_sparse:
                log_debug(f"Using sparse matrix operations (n_active={n_active})", LogCategory.PARTITIONING)
                B_matrix = self._compute_B_matrix_sparse(K_sba, susceptances)
            else:
                B_matrix = self._compute_B_matrix_dense(K_sba, susceptances)

            return B_matrix, active_nodes

        except Exception as e:
            if isinstance(e, PartitioningError):
                raise
            raise PartitioningError(
                f"Failed to build susceptance matrix: {e}",
                strategy=self._get_strategy_name()
            ) from e

    def _extract_edge_susceptances(self, graph: nx.DiGraph,
                                   config: ElectricalDistanceConfig) -> Tuple[List[Tuple[Any, Any]], np.ndarray]:
        """
        Extract directed edges and their susceptances from the graph.

        Each directed edge is treated as a unique electrical element.

        Args:
            graph: NetworkX DiGraph with 'x' (reactance) attribute on edges
            config: ElectricalDistanceConfig instance

        Returns:
            Tuple of (list of (from, to) edges, array of susceptances)

        Raises:
            PartitioningError: If reactance values are invalid
        """
        edges = []
        susceptances = []
        zero_reactance_count = 0

        for u, v, data in graph.edges(data=True):
            reactance = data.get('x')

            if not isinstance(reactance, (int, float)):
                raise PartitioningError(
                    f"Edge ({u}, {v}) reactance must be numeric, got {type(reactance)}",
                    strategy=self._get_strategy_name()
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
                LogCategory.PARTITIONING
            )

        return edges, np.array(susceptances)

    @staticmethod
    def _build_incidence_matrix(edges: List[Tuple[Any, Any]], nodes: List[Any],
                                slack_bus: Any) -> Tuple[np.ndarray, List[Any]]:
        """
        Build slack-bus-adjusted incidence matrix K.

        For each directed edge (u → v):
        - K[edge_idx, u] = -1 (edge leaves u)
        - K[edge_idx, v] = +1 (edge enters v)

        The slack bus column is removed to make B invertible.

        Args:
            edges: List of (from_node, to_node) tuples
            nodes: Ordered list of all nodes
            slack_bus: Node to exclude (slack bus)

        Returns:
            Tuple of (K matrix [n_edges × n_active], list of active nodes)
        """
        # Active nodes (without slack bus)
        active_nodes = [n for n in nodes if n != slack_bus]
        n_active = len(active_nodes)
        n_edges = len(edges)

        # Create node index mapping for active nodes
        node_to_idx = {node: idx for idx, node in enumerate(active_nodes)}

        # Build incidence matrix using sparse construction for efficiency
        row_indices = []
        col_indices = []
        values = []

        for edge_idx, (u, v) in enumerate(edges):
            # Edge leaves u: K[edge, u] = -1
            if u in node_to_idx:
                row_indices.append(edge_idx)
                col_indices.append(node_to_idx[u])
                values.append(-1.0)

            # Edge enters v: K[edge, v] = +1
            if v in node_to_idx:
                row_indices.append(edge_idx)
                col_indices.append(node_to_idx[v])
                values.append(1.0)

        # Create sparse matrix and convert to dense
        K_sparse = sparse.csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(n_edges, n_active)
        )

        return K_sparse.toarray(), active_nodes

    @staticmethod
    def _compute_B_matrix_dense(K: np.ndarray, susceptances: np.ndarray) -> np.ndarray:
        """
        Compute B matrix using dense operations: B = K^T · diag(b) · K

        Args:
            K: Incidence matrix (n_edges × n_active)
            susceptances: Array of susceptance values

        Returns:
            Susceptance matrix (n_active × n_active)
        """
        # Efficient computation: K^T @ diag(b) @ K = (K.T * b) @ K
        K_scaled = K.T * susceptances  # Broadcasting: (n_active, n_edges) * (n_edges,)
        B_matrix = K_scaled @ K

        # Ensure symmetry
        return (B_matrix + B_matrix.T) / 2.0

    @staticmethod
    def _compute_B_matrix_sparse(K: np.ndarray, susceptances: np.ndarray) -> np.ndarray:
        """
        Compute B matrix using sparse intermediate operations.

        Args:
            K: Incidence matrix (n_edges × n_active)
            susceptances: Array of susceptance values

        Returns:
            Susceptance matrix as dense array (for subsequent inversion)
        """
        K_sparse = sparse.csr_matrix(K)
        B_diag = sparse.diags(susceptances)
        B_sparse = K_sparse.T @ B_diag @ K_sparse

        # Convert to dense for inversion (sparse inversion is complex)
        B_matrix = B_sparse.toarray()

        # Ensure symmetry
        return (B_matrix + B_matrix.T) / 2.0

    def _invert_susceptance_matrix(self, B_matrix: np.ndarray) -> np.ndarray:
        """
        Invert the susceptance matrix to obtain the impedance matrix.

        Args:
            B_matrix: Susceptance matrix to invert

        Returns:
            Inverted matrix (impedance matrix)

        Raises:
            PartitioningError: If matrix is singular
        """
        try:
            # Check condition number for numerical stability
            cond = np.linalg.cond(B_matrix)
            if cond > self.config.condition_number_threshold:
                log_warning(
                    f"B matrix has high condition number ({cond:.2e}). "
                    "Results may be numerically unstable.",
                    LogCategory.PARTITIONING
                )

            B_inv = np.linalg.inv(B_matrix)

        except np.linalg.LinAlgError as e:
            raise PartitioningError(
                "B matrix is singular and cannot be inverted. "
                "This may indicate the network has disconnected components or numerical issues.",
                strategy=self._get_strategy_name()
            ) from e

        # Ensure symmetry
        return (B_inv + B_inv.T) / 2.0

    def _compute_distances_vectorized(self, B_inv: np.ndarray, config: ElectricalDistanceConfig) -> np.ndarray:
        """
        Compute electrical distances using vectorized numpy operations.

        Electrical distance: d_ij = sqrt(B^-1_ii + B^-1_jj - 2*B^-1_ij)

        This vectorized implementation is O(n²) in memory but much faster
        than nested loops due to numpy's optimized operations.

        Args:
            B_inv: Impedance matrix (inverted susceptance matrix)
            config: ElectricalDistanceConfig instance

        Returns:
            Distance matrix for active nodes (n_active × n_active)

        Raises:
            PartitioningError: If calculation produces invalid values
        """
        # Extract diagonal
        B_inv_diag = np.diag(B_inv)

        # Vectorized distance calculation using broadcasting:
        # d²_ij = B^-1_ii + B^-1_jj - 2*B^-1_ij
        # Shape: (n,1) + (1,n) - 2*(n,n) = (n,n)
        distance_squared = (B_inv_diag[:, np.newaxis] +
                            B_inv_diag[np.newaxis, :] -
                            2 * B_inv)

        # Handle numerical issues: clamp small negatives to zero
        significant_negatives = distance_squared < config.negative_distance_threshold

        if np.any(significant_negatives):
            n_significant = np.sum(significant_negatives)
            min_val = np.min(distance_squared[significant_negatives])
            log_warning(
                f"{n_significant} distance² values significantly negative "
                f"(min: {min_val:.2e}). Setting to zero.",
                LogCategory.PARTITIONING
            )

        distance_squared = np.maximum(distance_squared, 0.0)

        # Compute distances
        distance_matrix = np.sqrt(distance_squared)

        # Ensure diagonal is exactly zero
        np.fill_diagonal(distance_matrix, 0.0)

        if np.any(np.isnan(distance_matrix)):
            raise PartitioningError(
                "Distance matrix contains NaN values. This indicates numerical "
                "instability in the B matrix inversion.",
                strategy=self._get_strategy_name()
            )

        return distance_matrix

    def _integrate_slack_bus_distances(self, distance_matrix_active: np.ndarray,
                                       nodes: List[Any], slack_bus: Any,
                                       active_nodes: List[Any],
                                       config: ElectricalDistanceConfig) -> np.ndarray:
        """
        Integrate slack bus into the full distance matrix.

        The slack bus is assigned the average distance to all other nodes.

        Args:
            distance_matrix_active: Distance matrix for active nodes
            nodes: Complete list of all nodes
            slack_bus: Slack bus node
            active_nodes: List of active nodes (excluding slack bus)
            config: ElectricalDistanceConfig instance

        Returns:
            Full distance matrix including slack bus (n_nodes × n_nodes)
        """
        n_nodes = len(nodes)
        n_active = len(active_nodes)

        # Create full matrix
        distance_matrix_full = np.zeros((n_nodes, n_nodes))

        # Get slack index in full node list
        slack_idx = nodes.index(slack_bus)

        # Map active indices to full indices
        active_to_full = [nodes.index(n) for n in active_nodes]

        # Copy active distances to full matrix
        for i, full_i in enumerate(active_to_full):
            for j, full_j in enumerate(active_to_full):
                distance_matrix_full[full_i, full_j] = distance_matrix_active[i, j]

        # Calculate average distance for slack bus
        if n_active > 0:
            # Use upper triangle to get unique distances
            upper_tri = np.triu(distance_matrix_active, k=1)
            valid_distances = upper_tri[upper_tri > 0]

            if len(valid_distances) > 0:
                avg_distance = np.mean(valid_distances)
            else:
                avg_distance = config.slack_distance_fallback
                log_warning(
                    f"All electrical distances are zero. "
                    f"Using default distance {avg_distance} for slack bus.",
                    LogCategory.PARTITIONING
                )

            # Set slack bus distances
            for full_i in active_to_full:
                distance_matrix_full[slack_idx, full_i] = avg_distance
                distance_matrix_full[full_i, slack_idx] = avg_distance

        # Validate final matrix
        if np.any(np.isnan(distance_matrix_full)):
            raise PartitioningError(
                "Final distance matrix contains NaN values after slack bus integration.",
                strategy=self._get_strategy_name()
            )

        return distance_matrix_full
