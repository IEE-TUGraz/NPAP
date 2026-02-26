"""
Command-line helpers for quick NPAP workflows.

These entry points wrap the core managers so you can script clustering,
aggregation, and visualization without writing Python glue code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx

from npap.aggregation.modes import AggregationMode, get_mode_profile
from npap.logging import LogCategory, log_info
from npap.managers import AggregationManager, PartitionAggregatorManager
from npap.visualization import PlotPreset, export_figure, plot_network, plot_reduced_matrices


def _load_graph_from_file(path: str, fmt: str | None = None) -> nx.Graph:
    """
    Read a NetworkX graph from disk in one of the supported formats.
    """
    path_obj = Path(path)
    fmt = (fmt or path_obj.suffix.lstrip(".")).lower()

    if fmt == "graphml":
        return nx.read_graphml(path)
    if fmt == "gexf":
        return nx.read_gexf(path)
    if fmt in {"gpickle", "pickle"}:
        return nx.read_gpickle(path)
    raise ValueError(f"Unsupported graph format: {fmt}")


def _load_graph(args: argparse.Namespace, manager: PartitionAggregatorManager) -> nx.DiGraph:
    if args.node_file and args.edge_file:
        load_kwargs = {
            "delimiter": args.delimiter,
            "decimal": args.decimal,
        }
        if args.node_id_col:
            load_kwargs["node_id_col"] = args.node_id_col
        if args.edge_from_col:
            load_kwargs["edge_from_col"] = args.edge_from_col
        if args.edge_to_col:
            load_kwargs["edge_to_col"] = args.edge_to_col

        return manager.load_data(
            "csv_files",
            node_file=args.node_file,
            edge_file=args.edge_file,
            **{k: v for k, v in load_kwargs.items() if v is not None},
        )

    if args.graph_file:
        graph = _load_graph_from_file(args.graph_file, args.graph_format)
        bidirectional = not args.no_bidirectional
        return manager.load_data("networkx_direct", graph=graph, bidirectional=bidirectional)

    raise ValueError("Either node/edge files or a graph file must be provided.")


def _dump_partition(partition: dict[int, list[int]], output: str | None = None) -> None:
    payload = {str(cluster_id): nodes for cluster_id, nodes in partition.items()}
    if output:
        Path(output).write_text(json.dumps(payload, indent=2))
        log_info(f"Wrote partition mapping to {output}", LogCategory.MANAGER)
    else:
        print(json.dumps(payload, indent=2))


def _read_partition(path: str) -> dict[int, list[int]]:
    payload = json.loads(Path(path).read_text())
    return {int(k): v for k, v in payload.items()}


def _write_graph(graph: nx.Graph, output: str, fmt: str | None = None) -> None:
    fmt = (fmt or Path(output).suffix.lstrip(".")).lower()
    writer = {
        "graphml": nx.write_graphml,
        "gexf": nx.write_gexf,
        "gpickle": nx.write_gpickle,
        "pickle": nx.write_gpickle,
    }.get(fmt)

    if writer is None:
        raise ValueError(f"Unknown graph output format: {fmt}")

    writer(graph, output)
    log_info(f"Exported graph to {output}", LogCategory.AGGREGATION)


def _common_load_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--node-file", help="CSV file with node attributes.")
    parser.add_argument("--edge-file", help="CSV file with edge list.")
    parser.add_argument("--graph-file", help="Path to a NetworkX graph on disk.")
    parser.add_argument(
        "--graph-format",
        choices=["graphml", "gexf", "gpickle", "pickle"],
        help="Force the format of --graph-file.",
    )
    parser.add_argument(
        "--no-bidirectional",
        action="store_true",
        help="Do not create bidirectional edges when converting undirected graphs.",
    )
    parser.add_argument(
        "--delimiter",
        default=",",
        help="CSV delimiter (only used with --node-file/--edge-file).",
    )
    parser.add_argument(
        "--decimal",
        default=".",
        help="Decimal separator (only used with --node-file/--edge-file).",
    )
    parser.add_argument("--node-id-col", help="Override node ID column name.")
    parser.add_argument("--edge-from-col", help="Override edge source column name.")
    parser.add_argument("--edge-to-col", help="Override edge target column name.")


def cluster_entry() -> None:
    """Partition a loaded graph and emit a JSON mapping."""
    parser = argparse.ArgumentParser(description="Partition a grid with NPAP.")
    _common_load_parser(parser)
    parser.add_argument(
        "--partition-strategy",
        default="geographical_kmeans",
        help="Partitioning strategy name registered in PartitioningManager.",
    )
    parser.add_argument(
        "--n-clusters", "-n", type=int, required=True, help="Number of clusters to create."
    )
    parser.add_argument(
        "--partition-output",
        help="Write partition mapping to this JSON file. Prints to stdout if omitted.",
    )

    args = parser.parse_args()
    manager = PartitionAggregatorManager()
    graph = _load_graph(args, manager)

    partition = manager.partition(graph, args.partition_strategy, n_clusters=args.n_clusters)
    _dump_partition(partition.mapping, args.partition_output)


def aggregate_entry() -> None:
    """Aggregate a previously partitioned graph using a predefined mode."""
    parser = argparse.ArgumentParser(description="Aggregate a partitioned grid.")
    _common_load_parser(parser)
    parser.add_argument(
        "--partition-file",
        required=True,
        help="Path to JSON partition map produced by npap-cluster.",
    )
    parser.add_argument(
        "--mode",
        type=AggregationMode,
        choices=list(AggregationMode),
        default=AggregationMode.GEOGRAPHICAL,
        help="Predefined aggregation mode.",
    )
    parser.add_argument(
        "--output",
        help="Write aggregated graph to this path (GraphML/GEXF/GPickle inferred).",
    )

    args = parser.parse_args()
    manager = PartitionAggregatorManager()
    graph = _load_graph(args, manager)
    partition_map = _read_partition(args.partition_file)

    aggregation_manager = AggregationManager()
    profile = get_mode_profile(args.mode)
    aggregated = aggregation_manager.aggregate(graph, partition_map, profile)

    if args.output:
        _write_graph(aggregated, args.output)


def plot_entry() -> None:
    """Render a graph or aggregated network and save the figure."""
    parser = argparse.ArgumentParser(description="Plot NPAP partitions or graphs.")
    _common_load_parser(parser)
    parser.add_argument("--aggregated-file", help="Path to aggregated NetworkX graph.")
    parser.add_argument(
        "--aggregated-format",
        choices=["graphml", "gexf", "gpickle", "pickle"],
        help="Format of the aggregated graph file.",
    )
    parser.add_argument(
        "--partition-file",
        help="Optional partition JSON to color clusters (overrides partition_map).",
    )
    parser.add_argument(
        "--style",
        default="clustered",
        choices=["simple", "voltage_aware", "clustered"],
        help="Plot style.",
    )
    parser.add_argument(
        "--preset",
        choices=[preset.value for preset in PlotPreset],
        default=PlotPreset.DEFAULT.value,
        help="Plot preset for quick styling changes.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the visualization (HTML/default or PNG when format=png).",
    )

    args = parser.parse_args()
    manager = PartitionAggregatorManager()
    graph = _load_graph(args, manager)

    if args.aggregated_file:
        graph = _load_graph_from_file(args.aggregated_file, args.aggregated_format)

    partition_map = _read_partition(args.partition_file) if args.partition_file else None

    fig = plot_network(
        graph,
        style=args.style,
        partition_map=partition_map,
        preset=args.preset,
        show=False,
    )
    export_figure(fig, args.output)


def diagnose_entry() -> None:
    """Visualize reduced PTDF/laplacian matrices from an aggregated graph."""
    parser = argparse.ArgumentParser(description="Inspect reduced PTDF/laplacian matrices.")
    parser.add_argument(
        "--aggregated-file",
        required=True,
        help="Path to a saved aggregated graph (GraphML/GEXF/GPickle).",
    )
    parser.add_argument(
        "--aggregated-format",
        choices=["graphml", "gexf", "gpickle", "pickle"],
        help="Format of the aggregated graph file.",
    )
    parser.add_argument(
        "--matrix",
        choices=["ptdf", "laplacian"],
        action="append",
        default=["ptdf", "laplacian"],
        help="Matrix to visualize; can be repeated.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path where the diagnostic figure will be saved (HTML/PNG).",
    )

    args = parser.parse_args()
    graph = _load_graph_from_file(args.aggregated_file, args.aggregated_format)

    fig = plot_reduced_matrices(graph, matrices=tuple(args.matrix), show=False)
    export_figure(fig, args.output)
