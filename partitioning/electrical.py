from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

from exceptions import PartitioningError
from interfaces import PartitioningStrategy


@dataclass
class ElectricalDistanceConfig:
    """
    Configuration parameters for electrical distance calculations.

    Centralizes all magic numbers and tolerances used in the electrical
    distance partitioning algorithm for better maintainability and tuning.

    Attributes:
        regularization: Small value added to B matrix diagonal for numerical stability
        zero_reactance_replacement: Reactance value used when edge reactance is zero
        slack_distance_fallback: Default distance value when no valid distances exist
        numerical_tolerance: Threshold for considering values as zero
        negative_distance_threshold: Threshold for warning about negative distance squared
    """
    regularization: float = 1e-10
    zero_reactance_replacement: float = 1e-5
    slack_distance_fallback: float = 1.0
    numerical_tolerance: float = 1e-10
    negative_distance_threshold: float = -1e-10


class ElectricalDistancePartitioning(PartitioningStrategy):
    """
    Partition nodes based on electrical distance in power networks.

    This strategy uses the Power Transfer Distribution Factor (PTDF) approach
    to calculate electrical distances between nodes. The electrical distance
    is derived from the network's reactance/susceptance matrix and represents
    the electrical coupling between nodes.

    Mathematical basis:
    - Build incidence matrix K from network topology
    - Calculate susceptance matrix B = (K^sba)^T · diag{b} · K^sba
    - Electrical distance: d_ij = sqrt((B^-1_ii + B^-1_jj - 2*B^-1_ij))
    """

    def __init__(self, algorithm: str = 'kmeans', slack_bus: Optional[Any] = None,
                 config: Optional[ElectricalDistanceConfig] = None):
        """
        Initialize electrical distance partitioning strategy.

        Args:
            algorithm: Clustering algorithm ('kmeans', 'kmedoids')
            slack_bus: Specific node to use as slack bus, or None for auto-selection
            config: Configuration parameters for distance calculations
        """
        self.algorithm = algorithm
        self.slack_bus = slack_bus
        self.config = config or ElectricalDistanceConfig()

        if algorithm not in ['kmeans', 'kmedoids']:
            raise ValueError(
                f"Unsupported algorithm: {algorithm}. "
                "Supported: 'kmeans', 'kmedoids'"
            )

    @property
    def required_attributes(self) -> Dict[str, List[str]]:
        """Required attributes for electrical distance partitioning."""
        return {
            'nodes': [],
            'edges': ['x']  # TODO: detect different reactance names
        }

    def partition(self, graph: nx.Graph, **kwargs) -> Dict[int, List[Any]]:
        """
        Partition nodes based on electrical distance.

        Args:
            graph: NetworkX graph with reactance data on edges
            **kwargs: Additional parameters
                - n_clusters: Number of clusters (required)
                - random_state: Random seed for reproducibility
                - max_iter: Maximum iterations for clustering

        Returns:
            Dictionary mapping cluster_id -> list of node_ids

        Raises:
            PartitioningError: If partitioning fails
        """
        try:
            self._validate_network_connectivity(graph)

            n_clusters = kwargs.get('n_clusters')
            if n_clusters is None or n_clusters <= 0:
                raise PartitioningError(
                    f"Electrical distance partitioning requires a positive 'n_clusters' parameter.",
                    strategy=f"electrical_{self.algorithm}"
                )

            # Get node list (preserves order for matrix operations)
            nodes = list(graph.nodes())
            n_nodes = len(nodes)

            if n_clusters > n_nodes:
                raise PartitioningError(
                    f"Cannot create {n_clusters} clusters from {n_nodes} nodes.",
                    strategy=f"electrical_{self.algorithm}"
                )

            # Calculate electrical distance matrix
            distance_matrix = self._calculate_electrical_distance_matrix(graph, nodes)

            # Perform clustering based on electrical distance
            if self.algorithm == 'kmeans':
                labels = self._kmeans_clustering(distance_matrix, **kwargs)
            elif self.algorithm == 'kmedoids':
                labels = self._kmedoids_clustering(distance_matrix, **kwargs)
            else:
                raise PartitioningError(f"Unknown algorithm: {self.algorithm}")

            # Create partition mapping
            partition_map = {}
            for i, label in enumerate(labels):
                if int(label) not in partition_map:
                    partition_map[int(label)] = []
                partition_map[int(label)].append(nodes[i])

            # Validate result
            total_assigned = sum(len(cluster_nodes) for cluster_nodes in partition_map.values())
            if total_assigned != n_nodes:
                raise PartitioningError(
                    f"Partition assignment mismatch: {total_assigned} assigned vs {n_nodes} total nodes",
                    strategy=f"electrical_{self.algorithm}"
                )

            return partition_map

        except Exception as e:
            if isinstance(e, PartitioningError):
                raise
            raise PartitioningError(
                f"Electrical distance partitioning failed: {e}",
                strategy=f"electrical_{self.algorithm}",
                graph_info={'nodes': len(list(graph.nodes())), 'edges': len(graph.edges())}
            ) from e

    def _validate_network_connectivity(self, graph: nx.Graph) -> None:
        """
        Validate network connectivity.

        Args:
            graph: NetworkX graph to validate

        Raises:
            PartitioningError: If validation fails
        """
        if not nx.is_connected(graph):
            n_components = nx.number_connected_components(graph)
            raise PartitioningError(
                f"Graph must be connected for electrical distance partitioning. "
                f"Found {n_components} disconnected components (islands). "
                f"Consider processing each component separately or connecting the network.",
                strategy=f"electrical_{self.algorithm}",
                graph_info={
                    'nodes': len(list(graph.nodes())),
                    'edges': len(graph.edges()),
                    'n_components': n_components
                }
            )

    def _calculate_electrical_distance_matrix(self, graph: nx.Graph,
                                              nodes: List[Any]) -> np.ndarray:
        """
        Calculate electrical distance matrix between all node pairs.

        This method orchestrates the calculation by:
        1. Building the susceptance matrix (B matrix)
        2. Inverting the B matrix to get impedance matrix
        3. Computing distances from the impedance matrix
        4. Integrating slack bus distances

        Args:
            graph: NetworkX graph with reactance on edges
            nodes: Ordered list of nodes

        Returns:
            Symmetric distance matrix (n_nodes × n_nodes)

        Raises:
            PartitioningError: If distance matrix calculation fails
        """
        # Determine slack bus
        slack_bus = self._select_slack_bus(graph, nodes)

        # Build susceptance matrix
        B_matrix, active_nodes = self._build_susceptance_matrix(graph, nodes, slack_bus)

        # Invert susceptance matrix to get impedance matrix
        B_inv = self._invert_susceptance_matrix(B_matrix)

        # Calculate electrical distances from impedance matrix
        distance_matrix_active = self._compute_distances_from_impedance(B_inv, active_nodes)

        # Integrate slack bus into full distance matrix
        distance_matrix_full = self._integrate_slack_bus_distances(
            distance_matrix_active, nodes, slack_bus, active_nodes
        )

        return distance_matrix_full

    def _build_susceptance_matrix(self, graph: nx.Graph, nodes: List[Any],
                                  slack_bus: Any) -> Tuple[np.ndarray, List[Any]]:
        """
        Build the susceptance matrix (B matrix) for the network.

        The B matrix is calculated as: B = (K^sba)^T · diag{b} · K^sba
        where K^sba is the slack-bus-adjusted incidence matrix and b are susceptances.

        Args:
            graph: NetworkX graph with reactance on edges
            nodes: Ordered list of all nodes
            slack_bus: Node designated as slack bus

        Returns:
            Tuple of (B_matrix, active_nodes) where:
                - B_matrix: Susceptance matrix (n_active × n_active)
                - active_nodes: List of nodes excluding slack bus

        Raises:
            PartitioningError: If matrix construction fails
        """
        try:
            # Build slack-bus-adjusted incidence matrix
            K_sba, active_nodes = self._build_slack_bus_adjusted_incidence_matrix(
                graph, nodes, slack_bus
            )

            # Get susceptances (b = 1/x)
            susceptances = self._get_susceptances(graph)

            # Calculate B matrix: B = (K^sba)^T · diag{b} · K^sba
            B_diag = np.diag(susceptances)
            B_matrix = K_sba.T @ B_diag @ K_sba

            # Ensure B matrix is symmetric (fix numerical errors from floating-point operations)
            B_matrix = (B_matrix + B_matrix.T) / 2.0

            # Add small regularization for numerical stability
            B_matrix += self.config.regularization * np.eye(B_matrix.shape[0])

            return B_matrix, active_nodes

        except Exception as e:
            raise PartitioningError(
                f"Failed to build susceptance matrix: {e}",
                strategy=f"electrical_{self.algorithm}"
            ) from e

    def _invert_susceptance_matrix(self, B_matrix: np.ndarray) -> np.ndarray:
        """
        Invert the susceptance matrix to obtain the impedance matrix.

        The impedance matrix (B^-1) represents the electrical impedance between
        nodes and is used to calculate electrical distances.

        Args:
            B_matrix: Susceptance matrix to invert

        Returns:
            Inverted matrix (impedance matrix)

        Raises:
            PartitioningError: If matrix is singular or inversion fails
        """
        try:
            B_inv = np.linalg.inv(B_matrix)
        except np.linalg.LinAlgError as e:
            raise PartitioningError(
                f"B matrix is singular and cannot be inverted. "
                f"This may indicate an islanded network or numerical issues.",
                strategy=f"electrical_{self.algorithm}"
            ) from e

        # Ensure B_inv is symmetric (fix numerical errors from inversion)
        return (B_inv + B_inv.T) / 2.0

    def _compute_distances_from_impedance(self, B_inv: np.ndarray,
                                          active_nodes: List[Any]) -> np.ndarray:
        """
        Compute electrical distances from the impedance matrix.

        Electrical distance between nodes i and j is calculated as:
        d_ij = sqrt(B^-1_ii + B^-1_jj - 2*B^-1_ij)

        Args:
            B_inv: Impedance matrix (inverted susceptance matrix)
            active_nodes: List of active nodes (excluding slack bus)

        Returns:
            Distance matrix for active nodes (n_active × n_active)

        Raises:
            PartitioningError: If distance calculation fails or produces invalid values
        """
        n_active = len(active_nodes)
        distance_matrix = np.zeros((n_active, n_active))

        for i in range(n_active):
            for j in range(i + 1, n_active):
                # Electrical distance: d_ij = sqrt(B^-1_ii + B^-1_jj - 2*B^-1_ij)
                distance_squared = B_inv[i, i] + B_inv[j, j] - 2 * B_inv[i, j]

                # Handle numerical precision: ensure non-negative before sqrt
                if distance_squared < 0:
                    if distance_squared < self.config.negative_distance_threshold:
                        # Significant negative value - warn user
                        print(f"Warning: Negative distance squared ({distance_squared:.2e}) "
                              f"between nodes {active_nodes[i]} and {active_nodes[j]}. "
                              f"Setting to zero.")
                    distance_squared = 0.0

                d_ij = np.sqrt(distance_squared)
                distance_matrix[i, j] = d_ij
                distance_matrix[j, i] = d_ij  # Symmetric

        # Validate distance matrix for NaN values
        if np.any(np.isnan(distance_matrix)):
            raise PartitioningError(
                "Distance matrix contains NaN values. This indicates numerical "
                "instability in the B matrix inversion or distance calculation.",
                strategy=f"electrical_{self.algorithm}"
            )

        return distance_matrix

    def _integrate_slack_bus_distances(self, distance_matrix_active: np.ndarray,
                                       nodes: List[Any], slack_bus: Any,
                                       active_nodes: List[Any]) -> np.ndarray:
        """
        Integrate slack bus into the full distance matrix.

        Since the slack bus is removed during B matrix calculation, we need to
        assign it distances to other nodes. By default, we use the average of
        all non-zero distances in the network.

        Args:
            distance_matrix_active: Distance matrix for active nodes
            nodes: Complete list of all nodes
            slack_bus: Slack bus node
            active_nodes: List of active nodes (excluding slack bus)

        Returns:
            Full distance matrix including slack bus (n_nodes × n_nodes)
        """
        n_nodes = len(nodes)
        n_active = len(active_nodes)
        distance_matrix_full = np.zeros((n_nodes, n_nodes))

        # Map active node distances to full matrix
        slack_idx = nodes.index(slack_bus)
        active_to_full = [i for i in range(n_nodes) if i != slack_idx]

        for i, full_i in enumerate(active_to_full):
            for j, full_j in enumerate(active_to_full):
                distance_matrix_full[full_i, full_j] = distance_matrix_active[i, j]

        # Set slack bus distances (use average distance to other nodes)
        if n_active > 0:
            valid_distances = distance_matrix_active[distance_matrix_active > 0]
            if len(valid_distances) > 0:
                avg_distance = np.mean(valid_distances)
            else:
                # Fallback: if all distances are zero, use default
                avg_distance = self.config.slack_distance_fallback
                print(f"Warning: All electrical distances are zero. "
                      f"Using default distance {avg_distance} for slack bus.")

            for i in range(n_nodes):
                if i != slack_idx:
                    distance_matrix_full[slack_idx, i] = avg_distance
                    distance_matrix_full[i, slack_idx] = avg_distance

        # Final validation to ensure no NaN values in full matrix
        if np.any(np.isnan(distance_matrix_full)):
            raise PartitioningError(
                "Final distance matrix contains NaN values after slack bus integration.",
                strategy=f"electrical_{self.algorithm}"
            )

        return distance_matrix_full

    def _select_slack_bus(self, graph: nx.Graph, nodes: List[Any]) -> Any:
        """
        Select slack bus node.

        If slack_bus is specified, use it. Otherwise, select the node
        with the highest degree (most connections).

        Args:
            graph: NetworkX graph
            nodes: List of nodes

        Returns:
            Selected slack bus node
        """
        if self.slack_bus is not None:
            if self.slack_bus not in nodes:
                raise PartitioningError(
                    f"Specified slack bus {self.slack_bus} not found in graph.",
                    strategy=f"electrical_{self.algorithm}"
                )
            return self.slack_bus

        # Auto-select: node with the highest degree
        degrees = dict(graph.degree())
        slack = max(nodes, key=lambda n: degrees[n])
        return slack

    @staticmethod
    def _build_slack_bus_adjusted_incidence_matrix(graph: nx.Graph,
                                                   nodes: List[Any],
                                                   slack_bus: Any) -> Tuple[np.ndarray, List[Any]]:
        """
        Build slack-bus-adjusted incidence matrix K^sba.

        The incidence matrix K has:
        - Entry -1 if edge leaves the node
        - Entry +1 if edge enters the node
        - Entry 0 if edge not connected to node

        For undirected graphs, we arbitrarily assign direction.
        The slack bus column is removed to make the matrix invertible.

        Args:
            graph: NetworkX graph
            nodes: Ordered list of all nodes
            slack_bus: Node to remove (slack bus)

        Returns:
            Tuple of (K^sba matrix, list of active nodes without slack)
        """
        edges = list(graph.edges())
        n_edges = len(edges)

        # Active nodes (without slack bus)
        active_nodes = [n for n in nodes if n != slack_bus]
        n_active = len(active_nodes)

        # Build K^sba matrix (edges × active_nodes)
        K_sba = np.zeros((n_edges, n_active))

        for edge_idx, (u, v) in enumerate(edges):
            # Assign arbitrary direction: u -> v
            if u in active_nodes:
                u_idx = active_nodes.index(u)
                K_sba[edge_idx, u_idx] = -1  # Edge leaves u

            if v in active_nodes:
                v_idx = active_nodes.index(v)
                K_sba[edge_idx, v_idx] = 1  # Edge enters v

        return K_sba, active_nodes

    def _get_susceptances(self, graph: nx.Graph) -> np.ndarray:
        """
        Extract susceptances (b = 1/x) from edge reactances.

        Args:
            graph: NetworkX graph with 'x' attribute on edges

        Returns:
            Array of susceptances for all edges
        """
        susceptances = []

        for u, v in graph.edges():
            reactance = graph.edges[u, v].get('x')

            if not isinstance(reactance, (int, float)):
                raise PartitioningError(
                    f"Edge ({u}, {v}) reactance must be numeric, got {type(reactance)}",
                    strategy=f"electrical_{self.algorithm}"
                )

            if reactance == 0:
                print(f"Warning: Edge ({u}, {v}) has zero reactance. "
                      f"Assigning reactance to {self.config.zero_reactance_replacement}.")
                reactance = self.config.zero_reactance_replacement

            susceptances.append(1.0 / reactance)

        return np.array(susceptances)

    @staticmethod
    def _kmeans_clustering(distance_matrix: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform K-means clustering using electrical distances.

        Note: K-means expects feature vectors, not distance matrices.
        We use the distance matrix rows as feature vectors, which effectively
        clusters nodes based on their distance profiles to all other nodes.

        Args:
            distance_matrix: Precomputed electrical distance matrix
            **kwargs: Clustering parameters

        Returns:
            Cluster labels
        """
        try:
            n_clusters = kwargs.get('n_clusters')
            random_state = kwargs.get('random_state', 42)
            max_iter = kwargs.get('max_iter', 300)

            # Use distance matrix rows as features
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                max_iter=max_iter,
                n_init=10
            )

            labels = kmeans.fit_predict(distance_matrix)
            return labels

        except Exception as e:
            raise PartitioningError(
                f"K-means clustering with electrical distance failed: {e}",
                strategy="electrical_kmeans"
            ) from e

    @staticmethod
    def _kmedoids_clustering(distance_matrix: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform K-medoids clustering using precomputed electrical distances.

        K-medoids naturally works with distance matrices and is more robust
        to outliers than K-means.

        Args:
            distance_matrix: Precomputed electrical distance matrix
            **kwargs: Clustering parameters

        Returns:
            Cluster labels
        """
        try:
            n_clusters = kwargs.get('n_clusters')
            random_state = kwargs.get('random_state', 42)
            max_iter = kwargs.get('max_iter', 300)

            kmedoids = KMedoids(
                n_clusters=n_clusters,
                metric='precomputed',
                random_state=random_state,
                max_iter=max_iter
            )

            labels = kmedoids.fit_predict(distance_matrix)
            return labels

        except Exception as e:
            raise PartitioningError(
                f"K-medoids clustering with electrical distance failed: {e}",
                strategy="electrical_kmedoids"
            ) from e
