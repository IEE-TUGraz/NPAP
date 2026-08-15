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
[![JOSS status](https://joss.theoj.org/papers/a48865bb0af0faffe1c9074db581d072/status.svg)](https://joss.theoj.org/papers/a48865bb0af0faffe1c9074db581d072)

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

This example runs as-is: NPAP downloads and caches the network for you. It
reduces the European high-voltage grid from 6,863 buses to 50, and takes about a
minute.

```python
import npap
from npap import AggregationProfile
from npap.datasets import fetch_pypsa_eur

files = fetch_pypsa_eur()

manager = npap.PartitionAggregatorManager()

manager.load_data(
    strategy="csv_files",
    node_file=str(files["buses.csv"]),
    edge_file=str(files["lines.csv"]),
)

manager.aggregate_parallel_edges(
    edge_properties={"x": "equivalent_reactance", "s_nom": "sum"},
    default_strategy="average",
)

manager.partition(strategy="geographical_kmedoids_haversine", n_clusters=50)

profile = AggregationProfile(
    topology_strategy="simple",
    node_properties={"lat": "average", "lon": "average", "voltage": "first"},
    edge_properties={"x": "equivalent_reactance", "s_nom": "sum"},
    default_node_strategy="average",
    default_edge_strategy="average",
)
reduced = manager.aggregate(profile=profile)

manager.plot_network(graph=reduced, style="simple", title="Reduced network")
```

The result is a 50-bus, 153-line network — 0.7% of the buses you started with.

The [Quick Start guide](https://npap.readthedocs.io/en/latest/user-guide/quick-start.html)
explains each step and how the configuration choices affect the outcome, and the
[example notebooks](https://npap.readthedocs.io/en/latest/user-guide/examples.html)
run the full voltage-aware pipeline, which respects transformers, DC links and
voltage levels.

## Contributing

We warmly welcome contributions from everyone! Whether it's fixing a typo, improving documentation, reporting bugs, or implementing new features —
every contribution matters.

Please read our [Contributing Guide](CONTRIBUTING.md) to get started, or visit the full [Contributing Documentation](https://npap.readthedocs.io/en/latest/contributing.html) for detailed guidelines.

## Reporting Issues and Asking Questions

Found a bug, want to request a feature, or have a question about using NPAP?
**[Open an issue](https://github.com/IEE-TUGraz/NPAP/issues/new/choose)** — the
template picker will guide you to the right form. Please
[search the existing issues](https://github.com/IEE-TUGraz/NPAP/issues) first, in
case it has already been reported.

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
