# NPAP Examples

This directory contains examples demonstrating how to use NPAP (Network Partitioning & Aggregation Package) for large-scale power system network reduction.

## Examples

### 1. PyPSA European Network Example (`pypsa_network_example.py`)

This example showcases the full NPAP pipeline using a realistic dataset:
- **Data Source**: [Prebuilt Electricity Network for PyPSA-Eur](https://zenodo.org/records/18619025) (Zenodo).
- **Loading**: Uses the `VoltageAwareStrategy` to handle AC lines, transformers, and DC links.
- **Partitioning**: Groups thousands of buses into a user-specified number of clusters based on geographical coordinates.
- **Aggregation**: Reduces the network while conserving physical properties (impedance/reactance) using the `Conservation` mode.
- **Visualization**: Generates an interactive HTML map showing the partitions.

#### Running the example

Ensure you have installed NPAP in editable mode:
```bash
pip install -e "."
```

Run the script:
```bash
python examples/pypsa_network_example.py
```

The script will:
1. Automatically download the required CSV data (~35 MB) from Zenodo if not present in `examples/data/`.
2. Process the network and save an interactive visualization to `examples/output/pypsa_european_partitions.html`.
