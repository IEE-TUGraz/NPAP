# Quick Start

## Why reduce a network?

The European high-voltage grid has around 6,800 substations. Feeding all of them
into a capacity-expansion or unit-commitment model with hourly resolution is
rarely tractable: the problem size grows with the number of buses, and most of
that detail does not change the answer to questions asked at continental scale.

The usual response is to cluster the network — group nearby or electrically
similar buses, then collapse each group into one representative bus. Doing that
by hand is tedious and easy to get wrong: reactances do not average, capacities
do not, and merging buses across a DC link or across voltage levels produces a
grid that no longer means anything physically.

NPAP does this reduction for you, and lets you choose how much detail to keep.
Running the same pipeline on the real European grid at two settings gives:

| `n_clusters` | Buses | Edges | Smallest cluster | Largest cluster |
|-------------:|------:|------:|-----------------:|----------------:|
| — (original) | 6,863 | 8,065 | — | — |
| 200 | 200 | 567 | 5 | 190 |
| 50 | 50 | 153 | 34 | 310 |

Fewer clusters means a smaller model but coarser geography, and the spread
between the smallest and largest group widens.

## Installing

```bash
pip install npap
```

Every example in this documentation runs on the same network — the prebuilt
PyPSA-Eur grid derived from OpenStreetMap. NPAP downloads and caches it for you,
so there is nothing to prepare by hand:

```python
from npap.datasets import fetch_pypsa_eur

files = fetch_pypsa_eur()
```

The first call downloads about 20 MB from Zenodo and leaves roughly 2.7 MB in a
local cache; later calls return immediately. See
{doc}`the datasets API <../api/datasets>` for where that cache lives and how to
clear it.

## A complete example

This runs as-is and takes about a minute. It reduces the European grid from
6,863 buses to 50: load the network, collapse the double-circuit lines, cluster
geographically, then aggregate each cluster into a single representative bus.

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
Run as a script, the last line also writes a standalone HTML map next to you;
inside a notebook it renders inline. Pass `renderer="browser"` to open it
directly from an IDE.

## Going further: the example notebooks

The example above is deliberately minimal. **Everything else — choosing between
geographical and electrical distance, harmonising voltage levels, handling AC
islands and DC links, and applying the physically correct aggregation strategy
per edge type — is covered step by step in the two example notebooks.**

You can read them two ways:

- **Here in the documentation.** The pages linked below are the notebooks
  rendered from their stored outputs, interactive maps included. Nothing to
  install, nothing to run.
- **On your own machine.** The notebooks live in the
  [`examples/`](https://github.com/IEE-TUGraz/NPAP/tree/main/examples) directory
  of the repository and are meant to be executed and modified.

```{toctree}
:hidden:
:maxdepth: 1

examples/getting_started
examples/european_network_pypsa
```

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Getting Started
:link: examples/getting_started
:link-type: doc

The whole pipeline in ten minutes, first on a plain graph and then with the
power-systems features enabled, so the difference between the two is immediate.
:::

:::{grid-item-card} European High-Voltage Network
:link: examples/european_network_pypsa
:link-type: doc

A deeper walkthrough of the same grid: voltage-level harmonisation, AC-island
handling, and per-edge-type aggregation with the physically correct strategy
for lines, transformers and DC links.
:::

::::

### Getting the notebook files

The rendered pages above are read-only. To run or modify them, take the `.ipynb`
files from the repository:

| Notebook | Direct link |
|----------|-------------|
| Getting Started | <https://github.com/IEE-TUGraz/NPAP/blob/main/examples/getting_started.ipynb> |
| European High-Voltage Network | <https://github.com/IEE-TUGraz/NPAP/blob/main/examples/european_network_pypsa.ipynb> |

Or clone the repository and launch Jupyter from the `examples/` directory:

```bash
git clone https://github.com/IEE-TUGraz/NPAP.git
cd NPAP
pip install -e ".[dev,test,docs]"
jupyter lab examples/
```

## Where to go next

- {doc}`available-strategies` — every partitioning and aggregation option
- {doc}`data-loading` — CSV column conventions and the voltage-aware formats
- {doc}`partitioning/index` — choosing between geographical and electrical distance
- {doc}`aggregation` — how the three-tier aggregation system works
- {doc}`visualization` — styling the interactive maps
- {doc}`extending` — registering your own loaders, partitioners and strategies
