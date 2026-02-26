# NPAP Examples

This directory contains examples demonstrating the use and potential applications of the **Network Partitioning and Aggregation Package (NPAP)**.

## Available Examples

### 1. European Electricity Network Partitioning
- **Script**: `pypsa_network_example.py`
- **Description**: This example process a topologically connected representation of the European high-voltage grid (220 kV to 750 kV) constructed from OpenStreetMap data. 
- **Features Demonstrated**:
    - Data loading from CSV files (compatible with PyPSA-Eur datasets).
    - Geographical partitioning.
    - Geographical aggregation.
    - Interactive visualization of clustered networks.
- **Data Source**: [Zenodo Record 18619025](https://zenodo.org/records/18619025).

## Running the Examples

Ensure you have the required dependencies installed (including `requests` for data downloading):

```bash
pip install requests pandas plotly networkx
```

To run the PyPSA example:

```bash
python examples/pypsa_network_example.py
```

The script will automatically download the necessary CSV files from Zenodo (approx. 36 MB) into the `examples/data/` directory.

## Outputs
Running the example will generate an HTML file (e.g., `examples/clustered_european_grid.html`) which can be opened in any web browser to explore the clustered network.
