<!-- GitHub dark/light mode banner -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/IEE-TUGraz/NPAP/main/docs/assets/NPAP-Banner-light.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/IEE-TUGraz/NPAP/main/docs/assets/NPAP-Banner-dark.svg">
    <img alt="NPAP - Network Partitioning & Aggregation Package" src="https://raw.githubusercontent.com/IEE-TUGraz/NPAP/main/docs/assets/NPAP-Banner-dark.svg" width="800">
  </picture>
</p>

<!-- Badges -->
[![PyPI version](https://img.shields.io/pypi/v/npap.svg?cacheSeconds=0)](https://pypi.org/project/npap/)
[![Python versions](https://img.shields.io/pypi/pyversions/npap.svg?cacheSeconds=0)](https://pypi.org/project/npap/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Documentation](https://readthedocs.org/projects/npap/badge/?version=latest)](https://npap.readthedocs.io)
[![codecov](https://codecov.io/gh/IEE-TUGraz/npap/branch/main/graph/badge.svg)](https://codecov.io/gh/IEE-TUGraz/npap)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/IEE-TUGraz/NPAP/main.svg)](https://results.pre-commit.ci/latest/github/IEE-TUGraz/NPAP/main)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

---

**NPAP** is an open-source Python library for **partitioning and aggregating network graphs**, with a special focus on electrical power systems. Built
on top of [NetworkX](https://networkx.org/), it provides a clean strategy-based architecture that makes it easy to cluster networks and reduce their
complexity while preserving essential properties.

Whether you're working with power grids, transportation networks, or any graph-based spatial data, NPAP helps you simplify complex networks into
manageable pieces.

> [!NOTE]
> Project contributors from [IEE](https://www.tugraz.at/institute/iee/home) (Institute of Electricity Economics and Energy Innovation) at the [Technical University of Graz](https://www.tugraz.at/)
are supported by the [Research Center Energetic](https://www.tugraz.at/forschung/forschung-an-der-tu-graz/research-centers/research-center-for-energy-economics-and-energy-analytics-energetic) and funded by the European Union (ERC,
NetZero-Opt, [101116212](https://cordis.europa.eu/project/id/101116212)).

## Documentation

For comprehensive guides, API reference, and tutorials, visit the official documentation:

**[https://npap.readthedocs.io](https://npap.readthedocs.io)**

## Features

- **Multiple Partitioning Algorithms** - K-means, K-medoids, DBSCAN, HDBSCAN, and hierarchical clustering
- **Distance Metrics** - Euclidean for local coordinates, Haversine for geographic data
- **Electrical Distance** - Partition based on PTDF-derived electrical proximity
- **Voltage-Aware Clustering** - Respects voltage levels and transformer boundaries
- **Flexible Aggregation** - Sum, average, or custom strategies for node/edge properties
- **Extensible Design** - Easy to add your own partitioning or aggregation strategies

## Installation

```bash
pip install npap
```

## Quick Start

```python
from npap import PartitionAggregatorManager, AggregationProfile, AggregationMode

# 1. Initialize the manager
manager = PartitionAggregatorManager()

# 2. Load data (voltage-aware loader for power systems)
manager.load_data(
    strategy='va_loader',
    node_file="buses.csv",
    line_file="lines.csv",
    transformer_file="transformers.csv",
    converter_file="converters.csv",
    link_file="dc_links.csv"
)

# 3. Aggregate parallel edges (optional)
manager.aggregate_parallel_edges(
    edge_properties={"x": "equivalent_reactance", "type": "first"},
    default_strategy="average"
)

# 4. Partition the network
manager.partition(strategy="electrical_kmedoids", n_clusters=50)

# 5. Visualize the partitioned network
manager.plot_network(style='clustered', title='Partitioned Network')

# 6. Aggregate with a custom profile
profile = AggregationProfile(
    mode=AggregationMode.GEOGRAPHICAL,
    topology_strategy="simple",
    node_properties={"lat": "average", "lon": "average", "voltage": "average"},
    edge_properties={"p_max": "sum", "x": "equivalent_reactance"},
    default_node_strategy="average",
    default_edge_strategy="average"
)
aggregated_network = manager.aggregate(profile=profile)

# 7. Visualize the reduced network
manager.plot_network(style='simple', title='Aggregated Network')
```

## Contributing

We warmly welcome contributions from everyone! Whether it's fixing a typo, improving documentation, reporting bugs, or implementing new features —
every contribution matters.

Please read our [Contributing Guide](CONTRIBUTING.md) to get started, or visit the full [Contributing Documentation](https://npap.readthedocs.io/en/latest/contributing.html) for detailed guidelines. Don't hesitate to open an issue if you have questions or ideas!

## License

NPAP is released under the [MIT License](LICENSE).

## Acknowledgements

Funded by the European Union (ERC, NetZero-Opt, 101116212). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them.

<p align="center">
  <img src="https://raw.githubusercontent.com/IEE-TUGraz/NPAP/main/docs/assets/ERC.png" alt="ERC" height="80">
  &nbsp;&nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/IEE-TUGraz/NPAP/main/docs/assets/Netzero-opt.png" alt="NetZero-Opt" height="80">
</p>

---

<p align="center">
  <i>Built with care for the open-source community 🫰</i>
</p>
