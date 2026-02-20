from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from npap.exceptions import PartitioningError, ValidationError
from npap.interfaces import PartitioningStrategy
from npap.logging import LogCategory, log_info
from npap.utils import create_partition_map, validate_partition, with_runtime_config


@dataclass
class LMPPartitioningConfig:
    """
    Configuration for LMP-based partitioning.

    Attributes
    ----------
    price_attribute : str
        Node attribute containing the locational marginal price (LMP).
    ac_island_attr : str
        Node attribute that identifies AC island membership (optional).
    adjacency_bonus : float
        Value subtracted from distances for directly connected nodes (clamped at 0).
    infinite_distance : float
        Distance value assigned to nodes that belong to different AC islands.
    """

    price_attribute: str = "lmp"
    ac_island_attr: str = "ac_island"
    adjacency_bonus: float = 0.0
    infinite_distance: float = 1e4


class LMPPartitioning(PartitioningStrategy):
    """
    Partition nodes by locational marginal prices (LMP).

    The strategy clusters nodes whose LMPs (or other custom ``price_attribute``)
    are similar, optionally favouring directly connected buses through the
    ``adjacency_bonus`` parameter. Nodes in different AC islands are separated via
    a large ``infinite_distance`` penalty to preserve electrical isolation.
    """

    _CONFIG_PARAMS = {"price_attribute", "ac_island_attr", "adjacency_bonus", "infinite_distance"}

    def __init__(self, config: LMPPartitioningConfig | None = None):
        self.config = config or LMPPartitioningConfig()

    @property
    def required_attributes(self) -> dict[str, list[str]]:
        return {"nodes": [self.config.price_attribute], "edges": []}

    def _get_strategy_name(self) -> str:
        return "lmp_similarity"

    @with_runtime_config(LMPPartitioningConfig, _CONFIG_PARAMS)
    def partition(self, graph: nx.DiGraph, **kwargs) -> dict[int, list[Any]]:
        """
        Partition nodes using LMP similarity.

        Parameters
        ----------
        graph : nx.DiGraph
            Directed graph with a numeric LMP attribute on each node.
        **kwargs : dict
            Additional parameters:

            - n_clusters : int (required)
            - config : LMPPartitioningConfig instance (overrides the instance config)
            - adjacency_bonus : float (runtime override)
            - price_attribute : str (runtime override)

        Returns
        -------
        dict[int, list[Any]]
            Mapping from cluster ID to node IDs.

        Raises
        ------
        PartitioningError
            If ``n_clusters`` is missing or invalid, or clustering fails.
        ValidationError
            If any node lacks the required LMP attribute.
        """
        effective_config = kwargs.get("_effective_config", self.config)
        price_attribute = effective_config.price_attribute
        ac_attr = effective_config.ac_island_attr
        n_clusters = kwargs.get("n_clusters")

        if n_clusters is None or n_clusters <= 0:
            raise PartitioningError(
                "LMP partitioning requires a positive 'n_clusters' parameter.",
                strategy=self._get_strategy_name(),
            )

        nodes = list(graph.nodes())
        n_nodes = len(nodes)
        if n_nodes > 0 and n_clusters > n_nodes:
            raise PartitioningError(
                f"Cannot create {n_clusters} clusters from {n_nodes} nodes.",
                strategy=self._get_strategy_name(),
            )

        missing_nodes = [
            node
            for node in nodes
            if not isinstance(graph.nodes[node].get(price_attribute), (int, float))
        ]
        if missing_nodes:
            raise ValidationError(
                f"Nodes {missing_nodes} lack a numeric '{price_attribute}' attribute.",
                missing_attributes={"nodes": missing_nodes},
                strategy=self._get_strategy_name(),
            )

        lmp_values = np.array(
            [float(graph.nodes[node][price_attribute]) for node in nodes], dtype=float
        )
        island_values = [graph.nodes[node].get(ac_attr) for node in nodes]

        log_info(
            f"Starting LMP partitioning (n_clusters={n_clusters}, price_attribute={price_attribute}, "
            f"adjacency_bonus={effective_config.adjacency_bonus})",
            LogCategory.PARTITIONING,
        )

        distance_matrix = np.zeros((n_nodes, n_nodes), dtype=float)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if (
                    ac_attr
                    and island_values[i] is not None
                    and island_values[j] is not None
                    and island_values[i] != island_values[j]
                ):
                    dist = effective_config.infinite_distance
                else:
                    diff = abs(lmp_values[i] - lmp_values[j])
                    if effective_config.adjacency_bonus > 0.0:
                        if graph.has_edge(nodes[i], nodes[j]) or graph.has_edge(nodes[j], nodes[i]):
                            diff = max(diff - effective_config.adjacency_bonus, 0.0)
                    dist = diff
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

        if n_nodes == 0:
            return {}

        try:
            clustering = AgglomerativeClustering(
                n_clusters=min(n_clusters, n_nodes),
                metric="precomputed",
                linkage="average",
            )
            labels = clustering.fit_predict(distance_matrix)
        except Exception as exc:
            raise PartitioningError(
                f"LMP partitioning failed: {exc}", strategy=self._get_strategy_name()
            ) from exc

        partition_map = create_partition_map(nodes, labels)
        validate_partition(partition_map, n_nodes, self._get_strategy_name())

        return partition_map
