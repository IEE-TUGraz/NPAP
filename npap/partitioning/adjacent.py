from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from npap.exceptions import PartitioningError, ValidationError
from npap.interfaces import PartitioningStrategy
from npap.logging import LogCategory, log_info
from npap.utils import create_partition_map, validate_partition, with_runtime_config


@dataclass
class AdjacentAgglomerativeConfig:
    """
    Configuration for adjacent-node agglomerative clustering.

    Attributes
    ----------
    node_attribute : str | None
        Optional node attribute used to score merges. If omitted, all nodes get a
        neutral score so merges are driven purely by adjacency.
    ac_island_attr : str | None
        Node attribute that identifies AC islands. When provided, edges connecting
        different islands are ignored so island membership cannot be merged.
    """

    node_attribute: str | None = None
    ac_island_attr: str | None = "ac_island"


class AdjacentNodeAgglomerativePartitioning(PartitioningStrategy):
    """
    Merge clusters only along existing network edges.

    The strategy aggregates clusters using a greedy agglomerative procedure that
    considers only adjacent nodes (or nodes connected through merged clusters).
    An optional ``node_attribute`` lets you prefer merges between similar buses,
    while ``ac_island_attr`` keeps AC islands separate even if DC links exist.
    """

    _CONFIG_PARAMS = {"node_attribute", "ac_island_attr"}

    def __init__(self, config: AdjacentAgglomerativeConfig | None = None):
        self.config = config or AdjacentAgglomerativeConfig()

    @property
    def required_attributes(self) -> dict[str, list[str]]:
        if self.config.node_attribute:
            return {"nodes": [self.config.node_attribute], "edges": []}
        return {"nodes": [], "edges": []}

    def _strategy_name(self) -> str:
        return "adjacent_agglomerative"

    @with_runtime_config(AdjacentAgglomerativeConfig, _CONFIG_PARAMS)
    def partition(self, graph: nx.DiGraph, **kwargs) -> dict[int, list[Any]]:
        """
        Partition nodes so only adjacent clusters merge.

        Parameters
        ----------
        graph : nx.DiGraph
            Directed graph representing network topology.
        **kwargs : dict
            Additional runtime overrides:

            - n_clusters : int (required)
            - config : AdjacentAgglomerativeConfig

        Returns
        -------
        dict[int, list[Any]]
            Mapping from cluster ID to node IDs.

        Raises
        ------
        PartitioningError
            When ``n_clusters`` is invalid or adjacency cannot produce enough
            merges.
        ValidationError
            When required node attributes are missing or not numeric.
        """
        effective_config = kwargs.get("_effective_config", self.config)
        node_attribute = effective_config.node_attribute
        ac_attr = effective_config.ac_island_attr
        n_clusters = kwargs.get("n_clusters")

        if n_clusters is None or n_clusters <= 0:
            raise PartitioningError(
                "n_clusters must be a positive integer.",
                strategy=self._strategy_name(),
            )

        nodes = list(graph.nodes())
        n_nodes = len(nodes)

        if n_nodes == 0:
            return {}

        if n_clusters > n_nodes:
            raise PartitioningError(
                f"Cannot create {n_clusters} clusters from {n_nodes} nodes.",
                strategy=self._strategy_name(),
            )

        node_values: dict[Any, float] = {}
        if node_attribute:
            missing = []
            for node in nodes:
                value = graph.nodes[node].get(node_attribute)
                if not isinstance(value, (int, float)):
                    missing.append(node)
                else:
                    node_values[node] = float(value)
            if missing:
                raise ValidationError(
                    f"Nodes {missing} lack a numeric '{node_attribute}' attribute.",
                    missing_attributes={"nodes": missing},
                    strategy=self._strategy_name(),
                )
        else:
            node_values = dict.fromkeys(nodes, 0.0)

        node_islands: dict[Any, Any] = {}
        if ac_attr:
            for node in nodes:
                node_islands[node] = graph.nodes[node].get(ac_attr)

        log_info(
            f"Adjacent agglomerative partitioning (n_clusters={n_clusters}, attribute={node_attribute})",
            LogCategory.PARTITIONING,
        )

        edges = []
        for u, v in graph.edges():
            if ac_attr:
                island_u = node_islands.get(u)
                island_v = node_islands.get(v)
                if island_u is not None and island_v is not None and island_u != island_v:
                    continue
            edges.append((u, v))

        if not edges and n_clusters < n_nodes:
            raise PartitioningError(
                "Graph has no adjacency edges to perform agglomerative merges.",
                strategy=self._strategy_name(),
            )

        cluster_map = {node: node for node in nodes}
        cluster_nodes: dict[Any, set[Any]] = {node: {node} for node in nodes}
        cluster_stats: dict[Any, tuple[float, int]] = {
            node: (node_values[node], 1) for node in nodes
        }

        def cluster_mean(cluster_id: Any) -> float:
            total, count = cluster_stats[cluster_id]
            return total / count

        while len(cluster_nodes) > n_clusters:
            best: tuple[float, tuple[Any, Any], Any, Any] | None = None
            for u, v in edges:
                cu = cluster_map[u]
                cv = cluster_map[v]
                if cu == cv or cu not in cluster_nodes or cv not in cluster_nodes:
                    continue

                diff = abs(cluster_mean(cu) - cluster_mean(cv))
                order = tuple(sorted((cu, cv)))
                if best is None or diff < best[0] or (diff == best[0] and order < best[1]):
                    best = (diff, order, cu, cv)

            if best is None:
                raise PartitioningError(
                    "Unable to reach requested cluster count via adjacency merges.",
                    strategy=self._strategy_name(),
                )

            _, _, cu, cv = best
            keep, merge = (cu, cv) if cu < cv else (cv, cu)

            cluster_nodes[keep].update(cluster_nodes[merge])
            for node in cluster_nodes[merge]:
                cluster_map[node] = keep

            sum_keep, count_keep = cluster_stats[keep]
            sum_merge, count_merge = cluster_stats[merge]
            cluster_stats[keep] = (sum_keep + sum_merge, count_keep + count_merge)

            del cluster_nodes[merge]
            del cluster_stats[merge]

        labels = np.array([cluster_map[node] for node in nodes], dtype=int)
        partition_map = create_partition_map(nodes, labels)
        validate_partition(partition_map, n_nodes, self._strategy_name())
        return partition_map
