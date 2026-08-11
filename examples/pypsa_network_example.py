"""
Example: European Network Partitioning and Aggregation using PyPSA Data
=======================================================================

This example demonstrates how to use NPAP to process a realistic, large-scale
power system network. It uses the "Prebuilt Electricity Network for PyPSA-Eur"
dataset from Zenodo (record 18619025).

The script will:
1. Download the latest buses, lines, transformers, and DC links data.
2. Load the network using NPAP's VoltageAwareStrategy.
3. Partition the network using geographical K-Means.
4. Aggregate the network using the Transformer Conservation mode.
5. Save a "clustered" visualization showing the partitions.
"""

import urllib.request
from pathlib import Path

from npap import AggregationMode, PartitionAggregatorManager
from npap.logging import LogCategory, log_info

# Zenodo record details
ZENODO_RECORD = "18619025"
FILES = {
    "node_file": "buses.csv",
    "line_file": "lines.csv",
    "transformer_file": "transformers.csv",
    "converter_file": "converters.csv",
    "link_file": "links.csv",
}
BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files"


def download_data(data_dir: Path):
    """Download required CSV files from Zenodo if they don't exist."""
    data_dir.mkdir(exist_ok=True, parents=True)

    for key, filename in FILES.items():
        target = data_dir / filename
        if not target.exists():
            url = f"{BASE_URL}/{filename}?download=1"
            log_info(f"Downloading {filename} from Zenodo...", LogCategory.INPUT)
            urllib.request.urlretrieve(url, target)
        else:
            log_info(f"Using local copy of {filename}", LogCategory.INPUT)


def run_example():
    """
    Execute the PyPSA network partitioning and aggregation example pipeline.

    Downloads necessary CSVs, loads the network, maps geographical coordinates,
    aggregates parallel edges, performs spatial partitioning, visualizes the
    sub-networks, and executes a full topological/physical aggregation.
    """
    # Set up paths
    example_dir = Path(__file__).parent
    data_dir = example_dir / "data"
    output_dir = example_dir / "output"
    output_dir.mkdir(exist_ok=True)

    # 1. Fetch data
    download_data(data_dir)

    # 2. Initialize Manager and Load Data
    log_info("Initializing NPAP Manager...", LogCategory.MANAGER)
    manager = PartitionAggregatorManager()

    # We use VoltageAwareStrategy to respect the hierarchy of AC/DC elements
    manager.load_data(
        strategy="va_loader",
        node_file=str(data_dir / FILES["node_file"]),
        line_file=str(data_dir / FILES["line_file"]),
        transformer_file=str(data_dir / FILES["transformer_file"]),
        converter_file=str(data_dir / FILES["converter_file"]),
        link_file=str(data_dir / FILES["link_file"]),
        node_id_col="bus_id",  # PyPSA uses bus_id
        quotechar="'",  # Required for WKT geometry in these files
    )

    log_info(
        f"Loaded graph with {manager.get_current_graph().number_of_nodes()} nodes.",
        LogCategory.INPUT,
    )

    # 2.5 Aggregate parallel edges
    log_info("Aggregating parallel edges...", LogCategory.MANAGER)
    manager.aggregate_parallel_edges()

    # 2.6 Map 'x' and 'y' to 'lon' and 'lat' for GeographicalPartitioning
    log_info("Mapping x/y coordinates to lon/lat...", LogCategory.MANAGER)
    for node, data in manager.get_current_graph().nodes(data=True):
        if "x" in data and "y" in data:
            data["lon"] = data["x"]
            data["lat"] = data["y"]

    # 3. Partitioning
    # We create 15 clusters using geographical K-Means
    n_clusters = 15
    log_info(f"Partitioning network into {n_clusters} clusters...", LogCategory.MANAGER)
    manager.partition("geographical_kmeans", n_clusters=n_clusters)

    # 4. Visualization of Partitions
    log_info("Generating partition visualization...", LogCategory.VISUALIZATION)
    fig_path = output_dir / "pypsa_european_partitions.html"

    fig = manager.plot_network(style="clustered", show=False)
    fig.write_html(str(fig_path))
    log_info(f"Partition visualization saved to: {fig_path}", LogCategory.VISUALIZATION)

    # 5. Aggregation
    # use CONSERVATION mode to preserve electrical properties
    log_info("Aggregating network (Conservation mode)...", LogCategory.AGGREGATION)
    aggregated = manager.aggregate(mode=AggregationMode.CONSERVATION)

    log_info(f"Aggregated graph has {aggregated.number_of_nodes()} nodes.", LogCategory.AGGREGATION)
    log_info("Example complete.", LogCategory.MANAGER)


if __name__ == "__main__":
    run_example()
