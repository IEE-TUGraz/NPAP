"""
Example workflow: price-aware clustering through LMP partitioning,
conservation aggregation, and preset-driven plotting.

This script shows how to combine the locational marginal price
partitioning strategy with the conservation aggregation mode, then
export the clustered figure using the helper presets introduced earlier.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from npap import AggregationMode, PartitionAggregatorManager
from npap.visualization import PlotPreset


def main(
    node_file: str,
    edge_file: str,
    n_clusters: int = 6,
    partition_output: str | None = None,
    aggregated_output: str | None = None,
    figure_output: str | None = None,
):
    """Execute the LMP → conservation workflow with optional exports."""
    manager = PartitionAggregatorManager()
    manager.load_data("csv_files", node_file=node_file, edge_file=edge_file)

    # Price-aware partitioning based on LMP
    partition = manager.partition(
        "lmp_similarity",
        n_clusters=n_clusters,
        adjacency_bonus=0.3,
        infinite_distance=1e5,
    )

    if partition_output:
        manager.plot_network(
            style="clustered",
            partition_map=partition.mapping,
            preset=PlotPreset.CLUSTER_HIGHLIGHT,
            show=False,
        ).write_html(partition_output)

    # Conservation aggregation
    aggregated = manager.aggregate(mode=AggregationMode.CONSERVATION)

    if aggregated_output:
        aggregated.graph["metadata"] = {"source": "lmp_conservation_workflow"}
        aggregated.graph["created_by"] = "lmp_conservation_workflow"
        aggregated.graph["original_partitions"] = partition.mapping.keys()
        agg_path = Path(aggregated_output)
        agg_path.parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(aggregated, agg_path)

    if figure_output:
        manager.plot_network(
            graph=aggregated,
            style="clustered",
            partition_map=partition.mapping,
            preset=PlotPreset.PRESENTATION,
            show=False,
        ).write_html(figure_output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LMP → conservation workflow")
    parser.add_argument("--node-file", required=True, help="Node CSV file path.")
    parser.add_argument("--edge-file", required=True, help="Edge CSV file path.")
    parser.add_argument("--clusters", type=int, default=6, help="Number of LMP clusters.")
    parser.add_argument("--partition-output", help="Optional HTML export for the partitioned graph.")
    parser.add_argument("--figure-output", help="Optional HTML export after conservation aggregation.")

    args = parser.parse_args()

    main(
        args.node_file,
        args.edge_file,
        n_clusters=args.clusters,
        partition_output=args.partition_output,
        figure_output=args.figure_output,
    )
