"""
NPAP Example: European Electricity Network Partitioning and Aggregation.

This example showcases how to use NPAP to:
1. Load a real-world European high-voltage electricity network (based on PyPSA-Eur OSM data).
2. Partition the network using geographical and graph-theory-based strategies.
3. Aggregate the network to a reduced representation.
4. Visualize the results.

Data Source:
Xiong, B., Fioriti, D., Neumann, F., Riepin, I., Brown, T. 
Prebuilt Electricity Network for PyPSA-Eur based on OpenStreetMap Data. 
Zenodo (2025). https://zenodo.org/records/18619025
"""

import os
import requests
import pandas as pd
from pathlib import Path

from npap.managers import PartitionAggregatorManager
from npap.interfaces import AggregationMode
from npap.visualization import NetworkPlotter

# --- 1. CONFIGURATION ---

# Data files from Zenodo
ZENODO_URL = "https://zenodo.org/records/18619025/files/"
FILES = ["buses.csv", "lines.csv", "transformers.csv"]
DATA_DIR = Path("examples/data")

# Partitioning settings
N_CLUSTERS = 50  # Target number of clusters

# --- 2. DATA PREPARATION ---

def download_data():
    """Download required data files from Zenodo if they don't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    for file_name in FILES:
        target_path = DATA_DIR / file_name
        if not target_path.exists():
            print(f"Downloading {file_name} from Zenodo...")
            response = requests.get(f"{ZENODO_URL}{file_name}?download=1")
            response.raise_for_status()
            with open(target_path, "wb") as f:
                f.write(response.content)
            print(f"Saved to {target_path}")
        else:
            print(f"Using local file: {target_path}")

# --- 3. MAIN WORKFLOW ---

def main():
    # Ensure data is available
    download_data()
    
    # Initialize NPAP Manager
    manager = PartitionAggregatorManager()
    
    # 3.1 Load the network
    # We use CSVFilesStrategy as the Zenodo data is in CSV format
    print("\nLoading network from CSV files...")
    graph = manager.load_data(
        "csv_files",
        node_file=str(DATA_DIR / "buses.csv"),
        edge_file=str(DATA_DIR / "lines.csv"),
        node_id_col="Bus",      # column in Zenodo buses.csv
        edge_from_col="bus0",   # source column in lines.csv
        edge_to_col="bus1"      # target column in lines.csv
    )
    
    print(f"Original network: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # 3.2 Partitioning
    # We'll use a geographical approach for this large-scale European grid
    print(f"\nPartitioning network into {N_CLUSTERS} clusters (Geographical)...")
    partition_result = manager.partition_graph(
        "geographical",
        n_clusters=N_CLUSTERS
    )
    
    print(f"Partitioning complete. Resulting in {partition_result.n_clusters} clusters.")
    
    # 3.3 Aggregation
    # Reduce the network according to the partition
    print("\nAggregating network...")
    aggregated_graph = manager.aggregate_graph(
        mode=AggregationMode.GEOGRAPHICAL
    )
    
    print(f"Aggregated network: {aggregated_graph.number_of_nodes()} nodes, {aggregated_graph.number_of_edges()} edges")
    print(f"Compression ratio: {graph.number_of_nodes() / aggregated_graph.number_of_nodes():.2f}x")
    
    # 3.4 Visualization
    # Create an interactive plot of the clustering results
    print("\nGenerating visualization...")
    plotter = NetworkPlotter(graph, partition_map=partition_result.mapping)
    fig = plotter.plot_clustered()
    
    # Show results
    # fig.show()  # Uncomment to open in browser
    
    # Save the plot for reference
    output_plot = "examples/clustered_european_grid.html"
    fig.write_html(output_plot)
    print(f"Visualization saved to {output_plot}")

if __name__ == "__main__":
    main()
