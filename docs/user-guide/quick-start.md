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

NPAP does this reduction for you, and lets you choose how much detail to keep:

| Clusters | Buses | Edges | Reduction |
|---------:|------:|------:|-----------|
| 6,863 (original) | 6,863 | 8,065 | — |
| 200 | 200 | 567 | to 2.9% of buses |
| 50 | 50 | 153 | to 0.7% of buses |

The rest of this page walks through producing exactly that, on the real network,
in about twenty lines of code.

## Getting the data

Every example in this documentation runs on the same network: the prebuilt
PyPSA-Eur grid derived from OpenStreetMap. NPAP downloads and caches it for you,
so there is nothing to prepare by hand.

```python
from npap.datasets import fetch_pypsa_eur

files = fetch_pypsa_eur()
```

The first call downloads about 20 MB from Zenodo and leaves roughly 2.7 MB in a
local cache; later calls return immediately. See
{doc}`the datasets API <../api/datasets>` for where that cache lives.

## Loading the network

`PartitionAggregatorManager` is the single entry point for the whole workflow.
The `csv_files` strategy needs only a node file and an edge file, and treats
every edge alike — a good starting point when you do not need the full grid
topology.

```python
import npap

manager = npap.PartitionAggregatorManager()

graph = manager.load_data(
    strategy="csv_files",
    node_file=str(files["buses.csv"]),
    edge_file=str(files["lines.csv"]),
)
```

This loads 6,863 buses and 9,162 lines. NPAP warns that it returned a
`MultiDiGraph` rather than a `DiGraph`, because the dataset contains
double-circuit lines: two separate lines between the same pair of buses.

## Collapsing parallel edges

Partitioning strategies operate on simple graphs, so those parallel lines have
to be merged first. This is the first place where the physics matters, and where
NPAP asks you to be explicit: two parallel lines carry the *sum* of their
capacities but the *equivalent* of their reactances, not the average.

```python
graph = manager.aggregate_parallel_edges(
    edge_properties={
        "x": "equivalent_reactance",
        "r": "equivalent_reactance",
        "s_nom": "sum",
        "length": "average",
    },
    default_strategy="average",
)
```

That collapses 1,097 parallel edges, leaving 8,065.

## Partitioning

Partitioning assigns every bus to a cluster without yet modifying the graph.
Here we use geographical k-medoids with Haversine distance, which keeps clusters
geographically compact and measures distance on the sphere rather than on a flat
projection.

```python
partition = manager.partition(
    strategy="geographical_kmedoids_haversine",
    n_clusters=50,
)
```

The result carries the mapping and its metadata, so you can inspect the outcome
before committing to it:

```python
import pandas as pd

sizes = pd.Series({k: len(v) for k, v in partition.mapping.items()})
print(f"{partition.n_clusters} clusters, sizes {sizes.min()}–{sizes.max()}")
```

With 50 clusters the groups range from 34 to 310 buses, averaging 137.

## Aggregating

Aggregation builds the reduced graph. An `AggregationProfile` states what should
happen to every property: coordinates get averaged so the representative bus
sits at the centre of its cluster, capacities are summed, and reactances use the
parallel-impedance formula.

```python
from npap import AggregationProfile

profile = AggregationProfile(
    topology_strategy="simple",
    node_properties={"lat": "average", "lon": "average", "voltage": "first"},
    edge_properties={
        "x": "equivalent_reactance",
        "r": "equivalent_reactance",
        "s_nom": "sum",
        "length": "average",
    },
    default_node_strategy="average",
    default_edge_strategy="average",
)

reduced = manager.aggregate(profile=profile)
```

The network is now 50 buses and 153 lines — 0.7% of the buses and 1.9% of the
edges you started with.

## Visualising

Every plotting call returns a Plotly figure. Run as a script it also writes a
standalone HTML file next to you; inside a notebook it renders inline.

```python
manager.plot_network(style="clustered", title="50 clusters")
manager.plot_network(graph=reduced, style="simple", title="Reduced network")
```

Pass `renderer="browser"` to open the interactive map directly from an IDE. See
{doc}`visualization` for styling options.

## Choosing how much to reduce

`n_clusters` is the main lever, and its effect is easy to quantify. Running the
same pipeline at two settings on this network gives:

| `n_clusters` | Buses | Edges | Smallest cluster | Largest cluster |
|-------------:|------:|------:|-----------------:|----------------:|
| 200 | 200 | 567 | 5 | 190 |
| 50 | 50 | 153 | 34 | 310 |

Fewer clusters means a smaller model but coarser geography, and the spread
between the smallest and largest cluster widens — with 50 clusters one group
holds 310 buses while another holds 34. If that imbalance matters for your
application, the voltage-aware strategies offer a proportional mode that
distributes clusters across voltage levels instead.

The other lever is the strategy itself. Geographical strategies cluster on
distance; electrical ones cluster on impedance, so buses that are far apart but
electrically close end up together. {doc}`available-strategies` lists all of
them.

## Power systems: the voltage-aware workflow

The workflow above treats every edge alike. Real grids have transformers and DC
links, and merging buses across them is physically wrong. The `va_loader`
strategy reads all five PyPSA-Eur files, classifies each edge as a line,
transformer or DC link, and assigns every bus to an AC island — the set of buses
reachable without crossing a DC link.

```python
va_manager = npap.PartitionAggregatorManager()

va_graph = va_manager.load_data(
    strategy="va_loader",
    node_file=str(files["buses.csv"]),
    line_file=str(files["lines.csv"]),
    transformer_file=str(files["transformers.csv"]),
    converter_file=str(files["converters.csv"]),
    link_file=str(files["links.csv"]),
)

va_manager.partition(strategy="va_geographical_kmedoids_haversine", n_clusters=50)
```

Buses in different AC islands or at different voltage levels are then never
merged. {doc}`partitioning/index` covers this in depth, and the
{doc}`example notebooks <examples>` run the full voltage-aware pipeline end to
end with per-edge-type aggregation.

## Using your own network

Nothing above is specific to power systems, or to CSV files. Any NetworkX
directed graph with `lat` and `lon` attributes works:

```python
import networkx as nx

graph = nx.DiGraph()
graph.add_node("a", lat=47.0, lon=15.0)
graph.add_node("b", lat=47.1, lon=15.1)
graph.add_edge("a", "b", x=0.01)

manager = npap.PartitionAggregatorManager()
manager.load_data("networkx_direct", graph=graph)
```

{doc}`data-loading` documents the CSV column conventions and the voltage-aware
file formats, and {doc}`extending` shows how to register a loader for a format
NPAP does not know about.

## Next steps

- {doc}`examples` — the same pipeline as runnable notebooks, with figures
- {doc}`available-strategies` — every partitioning and aggregation option
- {doc}`aggregation` — how the three-tier aggregation system works
- {doc}`partitioning/index` — choosing between geographical and electrical distance
