from __future__ import annotations

from typing import Any

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.cluster import SpectralClustering

from npap.exceptions import PartitioningError
from npap.interfaces import PartitioningStrategy
from npap.logging import LogCategory, log_warning
from npap.utils import create_partition_map, validate_partition


class SpectralPartitioning(PartitioningStrategy):
    """
    Partition using spectral clustering on the graph Laplacian.

    This strategy converts the input graph into a symmetric adjacency matrix and applies
    ``sklearn.cluster.SpectralClustering`` with ``affinity="precomputed"``. Because the
    adjacency matrix mirrors the existing connectivity, disconnected AC islands remain
    separate clusters, so the strategy respects island boundaries automatically. Use this
    when you need community-aware partitions on graphs that are not easily handled by
    geometric distances (e.g., weak geographic structure with strong topological signals).

    Parameters
    ----------
    random_state : int | None
        Seed for reproducible k-means assignment inside spectral clustering.
    """

    def __init__(self, random_state: int | None = None):
        self.random_state = random_state

    @property
    def required_attributes(self) -> dict[str, list[str]]:
        return {"nodes": [], "edges": []}

    def _strategy_name(self) -> str:
        return "spectral_clustering"

    def partition(
        self, graph: nx.DiGraph, /, *, n_clusters: int | None = None, **kwargs
    ) -> dict[int, list[Any]]:
        if not n_clusters or n_clusters < 2:
            raise PartitioningError(
                "SpectralPartitioning requires n_clusters >= 2.",
                strategy=self._strategy_name(),
            )

        nodes = list(graph.nodes())
        if len(nodes) < n_clusters:
            raise PartitioningError(
                "Cannot partition into more clusters than nodes.",
                strategy=self._strategy_name(),
            )

        adjacency = nx.to_numpy_array(graph.to_undirected(), nodelist=nodes)

        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=self.random_state,
        )
        labels = model.fit_predict(adjacency)

        partition_map = create_partition_map(nodes, labels)
        validate_partition(partition_map, len(nodes), self._strategy_name())

        return partition_map


class CommunityPartitioning(PartitioningStrategy):
    """
    Partition using greedy modularity community detection.

    This strategy uses NetworkX's ``greedy_modularity_communities`` routine to detect communities
    and is deterministic for a given graph structure.
    """

    def required_attributes(self) -> dict[str, list[str]]:
        return {"nodes": [], "edges": []}

    def _strategy_name(self) -> str:
        return "community_modularity"

    def partition(self, graph: nx.DiGraph, **kwargs) -> dict[int, list[Any]]:
        communities = list(greedy_modularity_communities(graph.to_undirected()))

        if not communities:
            raise PartitioningError(
                "Community detection returned no communities.",
                strategy=self._strategy_name(),
            )

        if "n_clusters" in kwargs:
            log_warning(
                "community_modularity determines cluster count automatically; 'n_clusters' is ignored.",
                LogCategory.PARTITIONING,
                warn_user=False,
            )

        partition_map = {idx: list(cluster) for idx, cluster in enumerate(communities)}
        validate_partition(partition_map, graph.number_of_nodes(), self._strategy_name())

        return partition_map


__all__ = ["CommunityPartitioning", "SpectralPartitioning"]
