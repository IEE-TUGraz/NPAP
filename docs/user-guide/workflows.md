# Workflows

This page highlights ready-to-copy workflows that combine the new graph-theory partitioners, transformer-conserving aggregation mode, and preset-driven visualization to solve production-grade problems.

## Voltage-aware case study

Target scenario: a multi-voltage power grid where geographical anchors are weak, but community structure and voltage hierarchies still guide reduction.

```{mermaid}
flowchart LR
    A[Voltage-Aware Loader] --> B[Community Detection]
    B --> C[Spectral / Conservation]
    C --> D[Transformer Constrained Aggregation]
    D --> E[Visualize with Preset]

    style A fill:#0fad6b,stroke:#076b3f,color:#fff
    style B fill:#2993B5,stroke:#1d6f8a,color:#fff
    style C fill:#2993B5,stroke:#1d6f8a,color:#fff
    style D fill:#0fad6b,stroke:#076b3f,color:#fff
    style E fill:#2993B5,stroke:#1d6f8a,color:#fff
```

```python
from npap import AggregationMode, PartitionAggregatorManager

manager = PartitionAggregatorManager()
manager.load_data(
    strategy="va_loader",
    node_file="buses.csv",
    line_file="lines.csv",
    transformer_file="transformers.csv",
    converter_file="converters.csv",
    link_file="dc_links.csv",
)

# Use graph-theory partitioning to respect community structure, then
# aggregate with the transformer conservation mode.
partition = manager.partition("community_modularity")
aggregated = manager.aggregate(mode=AggregationMode.CONSERVATION)

manager.plot_network(
    style="clustered",
    preset="cluster_highlight",
    partition_map=partition.mapping,
    show=False,
)
```

**Why this works**

1. The `community_modularity` strategy finds structural clusters even when lat/lon are noisy.
2. `AggregationMode.CONSERVATION` invokes the transformer conservation physical strategy to keep reactance/resistance faithful.
3. Presets such as `cluster_highlight` (see [Visualization](visualization.md)) simplify presentation-ready exports.

## Custom aggregation profile with transformer conservation

If you need more control than a predefined mode, mix in the new physical strategy manually:

```python
from npap import AggregationProfile

profile = AggregationProfile(
    topology_strategy="electrical",
    physical_strategy="transformer_conservation",
    physical_properties=["x", "r"],
    edge_properties={"p_max": "sum"},
    default_node_strategy="average",
    default_edge_strategy="sum",
)

aggregated = manager.aggregate(profile=profile)
```

This profile lets you combine the transformer-preserving physical strategy with any node/edge aggregation strategy you need (for example, splitting transformers by type in `edge_type_properties`).

## Visualization presets

Use the new `preset` argument in `plot_network` or pass `PlotPreset` directly to unlock consistent layout and styling for different audiences:

| Preset | Description |
|--------|-------------|
| `default` | Balanced settings for data exploration |
| `presentation` | Larger canvas, thicker edges, open-street-map tiles |
| `dense` | Compact view with higher voltage thresholds for busy networks |
| `cluster_highlight` | Turbo colorscale with bold nodes for partition-focused slides |

```python
from npap.visualization import PlotPreset

manager.plot_network(style="voltage_aware", preset=PlotPreset.PRESENTATION)
```

Presets are composable with `PlotConfig`: pass `config=PlotConfig(...)` and overrides via `kwargs` to fine-tune individual charts while keeping a consistent baseline.

## Command-line helpers

Three lightweight CLIs wrap the PartitionAggregatorManager, AggregationManager, and visualization helpers so you can automate common workflows without writing Python:

1. `npap-cluster` – load CSV node/edge files (or a saved NetworkX graph), partition with any registered strategy, and emit a JSON mapping.
2. `npap-aggregate` – read the partition JSON, aggregate via a predefined `AggregationMode`, and export the reduced graph (GraphML/GEXF/GPickle).
3. `npap-plot` – load either the original or aggregated graph, optionally color it with a partition JSON, and save the figure as HTML/PNG/SVG using the Plot presets.

```bash
npap-cluster \
  --node-file buses.csv \
  --edge-file lines.csv \
  --partition-strategy geographical_kmeans \
  --n-clusters 8 \
  --partition-output partitions.json

npap-aggregate \
  --node-file buses.csv \
  --edge-file lines.csv \
  --partition-file partitions.json \
  --mode dc_ptdf \
  --output aggregated.graphml

npap-plot \
  --aggregated-file aggregated.graphml \
  --partition-file partitions.json \
  --style clustered \
  --preset cluster_highlight \
  --output reports/clustered.html
```

The CLI helpers respect the same loaders as the Python API and reuse the preset-driven visualization pipeline introduced earlier. Install them via `pip install -e ".[test]"` (needed for Hypothesis and the CLI entry points) before running the commands.
