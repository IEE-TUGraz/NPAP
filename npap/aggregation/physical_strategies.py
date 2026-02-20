from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import networkx as nx
import numpy as np

from npap.aggregation.basic_strategies import (
    EquivalentReactanceStrategy,
    build_cluster_edge_map,
    build_node_to_cluster_map,
)
from npap.exceptions import AggregationError
from npap.interfaces import EdgeType, PhysicalAggregationStrategy
from npap.logging import LogCategory, log_warning

_MIN_REACTANCE = 1e-6
_MIN_SUSCEPTANCE = 1e-12


def _build_admittance_matrix(
    graph: nx.DiGraph, reactance_property: str
) -> tuple[np.ndarray, dict[Any, int]]:
    """Build Laplacian (admittance) matrix for the given graph."""
    nodes = list(graph.nodes())
    n = len(nodes)

    if n == 0:
        raise AggregationError(
            "Graph must contain nodes for PTDF reduction.", strategy="ptdf_reduction"
        )

    node_to_index: dict[Any, int] = {node: idx for idx, node in enumerate(nodes)}

    laplacian = np.zeros((n, n), dtype=float)
    processed = False

    for u, v, data in graph.edges(data=True):
        x_value = data.get(reactance_property)
        if x_value is None or not isinstance(x_value, (int, float)):
            continue

        reactance = float(x_value)
        if abs(reactance) < _MIN_REACTANCE:
            reactance = _MIN_REACTANCE

        susceptance = 1.0 / reactance
        u_idx = node_to_index[u]
        v_idx = node_to_index[v]

        laplacian[u_idx, u_idx] += susceptance
        laplacian[v_idx, v_idx] += susceptance
        laplacian[u_idx, v_idx] -= susceptance
        laplacian[v_idx, u_idx] -= susceptance

        processed = True

    if not processed:
        raise AggregationError(
            f"No numeric '{reactance_property}' values found for PTDF reduction.",
            strategy="ptdf_reduction",
        )

    return laplacian, node_to_index


def _kron_reduce_laplacian(laplacian: np.ndarray, keep_indices: Sequence[int]) -> np.ndarray:
    """Apply Kron reduction to the given Laplacian matrix."""
    n = laplacian.shape[0]
    all_indices = set(range(n))
    keep_set = set(keep_indices)
    eliminate = sorted(all_indices - keep_set)

    reduced = laplacian[np.ix_(keep_indices, keep_indices)].copy()

    if not eliminate:
        return reduced

    lap_kk = laplacian[np.ix_(keep_indices, keep_indices)]
    lap_ke = laplacian[np.ix_(keep_indices, eliminate)]
    lap_ek = laplacian[np.ix_(eliminate, keep_indices)]
    lap_ee = laplacian[np.ix_(eliminate, eliminate)]

    try:
        inv_lap_ee = np.linalg.inv(lap_ee)
    except np.linalg.LinAlgError:
        inv_lap_ee = np.linalg.pinv(lap_ee)

    return lap_kk - lap_ke @ inv_lap_ee @ lap_ek


def _select_representatives(
    partition_map: dict[int, list[Any]], cluster_order: list[int]
) -> list[Any]:
    representatives: list[Any] = []
    for cluster in cluster_order:
        nodes = partition_map.get(cluster)
        if not nodes:
            raise AggregationError(
                f"Cluster {cluster} has no nodes, cannot compute PTDF reduction.",
                strategy="ptdf_reduction",
            )
        representatives.append(nodes[0])
    return representatives


def _compute_reduced_ptdf(graph: nx.DiGraph, reactance_property: str) -> dict[str, Any]:
    """Build a reduced PTDF matrix for the aggregated (cluster) graph."""
    nodes = list(graph.nodes())
    if len(nodes) <= 1:
        return {"matrix": np.zeros((0, len(nodes))), "nodes": nodes, "slack": None, "edges": []}

    node_to_index = {node: idx for idx, node in enumerate(nodes)}
    unique_edges: dict[tuple[Any, Any], dict[str, Any]] = {}

    for u, v, data in graph.edges(data=True):
        key = tuple(sorted((u, v)))
        if key[0] == key[1]:
            continue

        susceptance = data.get("susceptance")
        if susceptance is None:
            reactance = data.get(reactance_property)
            if reactance is None or not isinstance(reactance, (int, float)):
                continue
            susceptance = 1.0 / max(abs(reactance), _MIN_REACTANCE)

        row = unique_edges.setdefault(key, {"u": key[0], "v": key[1], "susceptance": 0.0})
        row["susceptance"] += susceptance

    edges = list(unique_edges.values())
    if not edges:
        return {"matrix": np.zeros((0, len(nodes))), "nodes": nodes, "slack": None, "edges": []}

    edge_susceptances = np.array([edge["susceptance"] for edge in edges], dtype=float)
    incidence = np.zeros((len(edges), len(nodes)), dtype=float)
    for idx, edge in enumerate(edges):
        u_idx = node_to_index[edge["u"]]
        v_idx = node_to_index[edge["v"]]
        incidence[idx, u_idx] = 1.0
        incidence[idx, v_idx] = -1.0

    slack_node = nodes[0]
    slack_idx = node_to_index[slack_node]
    keep_indices = [idx for idx in range(len(nodes)) if idx != slack_idx]
    if not keep_indices:
        return {
            "matrix": np.zeros((len(edges), len(nodes))),
            "nodes": nodes,
            "slack": slack_node,
            "edges": edges,
        }

    incidence_sba = incidence[:, keep_indices]
    weight_diag = np.diag(edge_susceptances)
    b_matrix = incidence_sba.T @ weight_diag @ incidence_sba

    try:
        inv_b = np.linalg.inv(b_matrix)
    except np.linalg.LinAlgError:
        inv_b = np.linalg.pinv(b_matrix)

    ptdf_core = weight_diag @ incidence_sba @ inv_b
    ptdf_full = np.zeros((len(edges), len(nodes)), dtype=float)

    for idx, node_idx in enumerate(keep_indices):
        ptdf_full[:, node_idx] = ptdf_core[:, idx]

    return {
        "matrix": ptdf_full,
        "nodes": nodes,
        "slack": slack_node,
        "edges": [(edge["u"], edge["v"]) for edge in edges],
    }


class TransformerConservationStrategy(PhysicalAggregationStrategy):
    """
    Preserve transformer-level impedance when reducing the network.

    This strategy treats all transformers between two clusters as parallel elements and
    stores their equivalent reactance/resistance as well as transformer count on the aggregated
    edge.  It is meant to be used with electrical topology and ensures impedance conservation
    before the statistical step runs.
    """

    def __init__(
        self,
        edge_type_attribute: str = "type",
        transformer_type: str = EdgeType.TRAFO.value,
        reactance_property: str = "x",
        resistance_property: str = "r",
    ):
        self.edge_type_attribute = edge_type_attribute
        self.transformer_type = transformer_type
        self.reactance_property = reactance_property
        self.resistance_property = resistance_property
        self._equivalent_reactance = EquivalentReactanceStrategy()

    @property
    def required_properties(self) -> list[str]:
        return [self.reactance_property, self.resistance_property]

    @property
    def modifies_properties(self) -> list[str]:
        return [self.reactance_property, self.resistance_property]

    @property
    def can_create_edges(self) -> bool:
        return False

    @property
    def required_topology(self) -> str:
        return "electrical"

    def aggregate(
        self,
        original_graph: nx.DiGraph,
        partition_map: dict[int, list[Any]],
        topology_graph: nx.DiGraph,
        properties: list[str],
        parameters: dict[str, Any] | None = None,
    ) -> nx.DiGraph:
        params = parameters or {}
        node_to_cluster = params.get("node_to_cluster") or build_node_to_cluster_map(partition_map)
        cluster_edge_map = params.get("cluster_edge_map") or build_cluster_edge_map(
            original_graph, node_to_cluster
        )

        for u, v in topology_graph.edges():
            original_edges = cluster_edge_map.get((u, v), [])
            transformer_edges = [
                edge
                for edge in original_edges
                if edge.get(self.edge_type_attribute) == self.transformer_type
            ]

            if not transformer_edges:
                continue

            reactance = self._calculate_equivalent(transformer_edges, self.reactance_property)
            resistance = self._calculate_equivalent(transformer_edges, self.resistance_property)

            if reactance is not None:
                topology_graph.edges[u, v][self.reactance_property] = reactance
            if resistance is not None:
                topology_graph.edges[u, v][self.resistance_property] = resistance

            topology_graph.edges[u, v]["transformer_count"] = len(transformer_edges)

        return topology_graph

    def _calculate_equivalent(self, edges: list[dict[str, Any]], prop: str) -> float | None:
        try:
            return self._equivalent_reactance.aggregate_property(edges, prop)
        except AggregationError as exc:
            log_warning(
                f"Failed to aggregate '{prop}' for transformer group: {exc}",
                LogCategory.AGGREGATION,
                warn_user=True,
            )
            return None


class PTDFReductionStrategy(PhysicalAggregationStrategy):
    """
    PTDF-based reduction for DC networks.

    This strategy builds a Kron-reduced Laplacian for the representative nodes
    of each cluster, derives the susceptance between aggregated nodes, and
    populates the resulting edges with reactance values that match the reduced PTDF.
    """

    def __init__(self, reactance_property: str = "x"):
        self.reactance_property = reactance_property

    @property
    def required_properties(self) -> list[str]:
        return [self.reactance_property]

    @property
    def modifies_properties(self) -> list[str]:
        return [self.reactance_property]

    @property
    def can_create_edges(self) -> bool:
        return True

    @property
    def required_topology(self) -> str:
        return "electrical"

    def aggregate(
        self,
        original_graph: nx.DiGraph,
        partition_map: dict[int, list[Any]],
        topology_graph: nx.DiGraph,
        properties: list[str],
        parameters: dict[str, Any] | None = None,
    ) -> nx.DiGraph:
        del parameters  # unused
        del properties  # unused
        cluster_order = list(topology_graph.nodes())
        representatives = _select_representatives(partition_map, cluster_order)

        laplacian, node_to_index = _build_admittance_matrix(original_graph, self.reactance_property)
        keep_indices = [node_to_index[node] for node in representatives]
        reduced_laplacian = _kron_reduce_laplacian(laplacian, keep_indices)

        aggregated_graph = nx.DiGraph()
        aggregated_graph.add_nodes_from(
            (node, dict(topology_graph.nodes[node])) for node in topology_graph.nodes()
        )

        for src_idx, src_cluster in enumerate(cluster_order):
            for dst_idx, dst_cluster in enumerate(cluster_order):
                if src_idx == dst_idx:
                    continue

                susceptance = -reduced_laplacian[src_idx, dst_idx]
                if susceptance <= _MIN_SUSCEPTANCE or np.isnan(susceptance):
                    continue

                reactance = 1.0 / susceptance
                aggregated_graph.add_edge(
                    src_cluster,
                    dst_cluster,
                    **{
                        self.reactance_property: reactance,
                        "susceptance": susceptance,
                        "aggregation_source": "ptdf_reduction",
                    },
                )

        aggregated_graph.graph["reduced_ptdf"] = _compute_reduced_ptdf(
            aggregated_graph, self.reactance_property
        )

        return aggregated_graph


class KronReductionStrategy(PhysicalAggregationStrategy):
    """
    Kron reduction for DC power flow networks.

    Eliminates interior nodes that belong to each cluster, keeping only the cluster
    representatives. The resulting Laplacian defines the equivalent susceptance
    between clusters, so the aggregated edges inherit physics-consistent reactances.
    """

    def __init__(self, reactance_property: str = "x"):
        self.reactance_property = reactance_property

    @property
    def required_properties(self) -> list[str]:
        return [self.reactance_property]

    @property
    def modifies_properties(self) -> list[str]:
        return [self.reactance_property]

    @property
    def can_create_edges(self) -> bool:
        return True

    @property
    def required_topology(self) -> str:
        return "electrical"

    def aggregate(
        self,
        original_graph: nx.DiGraph,
        partition_map: dict[int, list[Any]],
        topology_graph: nx.DiGraph,
        properties: list[str],
        parameters: dict[str, Any] | None = None,
    ) -> nx.DiGraph:
        del properties  # handled via physical strategy
        cluster_order = list(topology_graph.nodes())
        representatives = _select_representatives(partition_map, cluster_order)

        laplacian, node_to_index = _build_admittance_matrix(original_graph, self.reactance_property)

        try:
            keep_indices = [node_to_index[node] for node in representatives]
        except KeyError as exc:
            raise AggregationError(
                f"Representative node {exc.args[0]} missing reactance '{self.reactance_property}'.",
                strategy="kron_reduction",
            ) from exc

        reduced_laplacian = _kron_reduce_laplacian(laplacian, keep_indices)

        aggregated_graph = nx.DiGraph()
        aggregated_graph.add_nodes_from(
            (node, dict(topology_graph.nodes[node])) for node in topology_graph.nodes()
        )

        n_clusters = len(cluster_order)
        for src_idx in range(n_clusters):
            for dst_idx in range(n_clusters):
                if src_idx == dst_idx:
                    continue

                susceptance = -reduced_laplacian[src_idx, dst_idx]
                if susceptance <= _MIN_SUSCEPTANCE or np.isnan(susceptance):
                    continue

                reactance = 1.0 / susceptance
                src_cluster = cluster_order[src_idx]
                dst_cluster = cluster_order[dst_idx]
                aggregated_graph.add_edge(
                    src_cluster,
                    dst_cluster,
                    **{
                        self.reactance_property: reactance,
                        "susceptance": susceptance,
                        "aggregation_source": "kron_reduction",
                    },
                )

        aggregated_graph.graph["kron_reduced_laplacian"] = reduced_laplacian
        return aggregated_graph
