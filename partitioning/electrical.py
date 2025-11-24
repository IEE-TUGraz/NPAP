from typing import Dict, List, Any, Optional

import networkx as nx
import numpy as np

from exceptions import PartitioningError
from interfaces import PartitioningStrategy


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

    def __init__(self, algorithm: str = 'kmeans', slack_bus: Optional[Any] = None):
        """
        Initialize electrical distance partitioning strategy.

        Args:
            algorithm: Clustering algorithm ('kmeans', 'kmedoids')
            slack_bus: Specific node to use as slack bus, or None for auto-selection
        """
        self.algorithm = algorithm
        self.slack_bus = slack_bus

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
                graph_info={'nodes': len(nodes), 'edges': len(graph.edges())}
            ) from e

    def _calculate_electrical_distance_matrix(self, graph: nx.Graph,
                                              nodes: List[Any]) -> np.ndarray:
        """
        Calculate electrical distance matrix between all node pairs.

        The electrical distance is calculated from the inverse of the
        susceptance matrix (B matrix). The B matrix represents the network's
        electrical properties after removing the slack bus.

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
        regularization = 1e-10
        B_matrix += regularization * np.eye(B_matrix.shape[0])

        # Calculate B inverse (impedance matrix)
        try:
            B_inv = np.linalg.inv(B_matrix)
        except np.linalg.LinAlgError as e:
            raise PartitioningError(
                f"B matrix is singular and cannot be inverted. "
                f"This may indicate an islanded network or numerical issues.",
                strategy=f"electrical_{self.algorithm}"
            ) from e

        # Ensure B_inv is symmetric (fix numerical errors from inversion)
        B_inv = (B_inv + B_inv.T) / 2.0

        # Calculate electrical distance matrix for active nodes
        n_active = len(active_nodes)
        distance_matrix_active = np.zeros((n_active, n_active))

        for i in range(n_active):
            for j in range(i + 1, n_active):
                # Electrical distance: d_ij = sqrt(B^-1_ii + B^-1_jj - 2*B^-1_ij)
                distance_squared = B_inv[i, i] + B_inv[j, j] - 2 * B_inv[i, j]

                # Handle numerical precision: ensure non-negative before sqrt
                if distance_squared < 0:
                    if distance_squared < -1e-10:  # Significant negative value
                        print(f"Warning: Negative distance squared ({distance_squared:.2e}) "
                              f"between nodes {active_nodes[i]} and {active_nodes[j]}. "
                              f"Setting to zero.")
                    distance_squared = 0.0

                d_ij = np.sqrt(distance_squared)
                distance_matrix_active[i, j] = d_ij
                distance_matrix_active[j, i] = d_ij  # Symmetric

        # Validate distance matrix for NaN values
        if np.any(np.isnan(distance_matrix_active)):
            raise PartitioningError(
                "Distance matrix contains NaN values. This indicates numerical "
                "instability in the B matrix inversion or distance calculation.",
                strategy=f"electrical_{self.algorithm}"
            )

        # Create full distance matrix including slack bus
        n_nodes = len(nodes)
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
                # Fallback: if all distances are zero, use a small positive value
                avg_distance = 1.0
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
                                                   slack_bus: Any) -> tuple:
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
                print(f"Warning: Edge ({u}, {v}) has zero reactance. Assigning reactance to 10e-5.")
                reactance = 1e-5  # Small value to avoid division by zero

            susceptances.append(1.0 / reactance)

        return np.array(susceptances)

            ) from e
