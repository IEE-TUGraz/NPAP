# Visualization

NPAP provides interactive network visualization using Plotly, enabling geographic map displays with customizable styling for nodes, edges, and clusters.

## Overview

The visualization system renders networks on interactive maps with support for:

- Geographic coordinates (lat/lon)
- Edge type differentiation (lines, transformers, DC links)
- Voltage level coloring
- Cluster visualization
- Customizable styling

## Basic Usage

### Quick Plot

```python
import npap

manager = npap.PartitionAggregatorManager()
manager.load_data("networkx_direct", graph=G)

# Simple visualization
manager.plot_network()
```

### With Partition Coloring

```python
# Partition first
partition = manager.partition("geographical_kmeans", n_clusters=10)

# Plot with cluster colors
manager.plot_network(style="clustered")
```

## Plot Styles

NPAP provides three built-in plot styles:

### SIMPLE Style

Basic visualization with uniform edge styling:

```python
manager.plot_network(style="simple")
```

- All edges have the same color
- Fastest rendering
- Best for quick inspection

### VOLTAGE_AWARE Style

Edges colored by type and voltage level:

```python
manager.plot_network(style="voltage_aware")
```

- Lines colored by voltage (high/low)
- Transformers highlighted
- DC links distinguished
- Best for power system visualization

### CLUSTERED Style

Nodes colored by cluster assignment:

```python
partition = manager.partition("geographical_kmeans", n_clusters=10)
manager.plot_network(style="clustered")
```

- Each cluster has a distinct color
- Shows partition results
- Best for partition analysis

## Plot Configuration

Fine-tune visualizations with `PlotConfig`:

```python
from npap.visualization import PlotConfig

config = PlotConfig(
    # Display toggles
    show_lines=True,
    show_trafos=True,
    show_dc_links=True,
    show_nodes=True,

    # Voltage threshold (kV)
    line_voltage_threshold=300,

    # Colors (hex)
    line_high_voltage_color="#e74c3c",   # Red for HV
    line_low_voltage_color="#3498db",    # Blue for LV
    trafo_color="#9b59b6",               # Purple for transformers
    dc_link_color="#2ecc71",             # Green for DC links
    node_color="#34495e",                # Dark gray for nodes

    # Geometry
    edge_width=1.5,
    node_size=6,

    # Map settings
    map_center_lat=47.0,
    map_center_lon=15.0,
    map_zoom=5,
    map_style="carto-positron"
)

manager.plot_network(style="voltage_aware", config=config)
```

### Configuration Parameters

#### Display Toggles

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `show_lines` | bool | `True` | Show transmission lines |
| `show_trafos` | bool | `True` | Show transformers |
| `show_dc_links` | bool | `True` | Show DC links |
| `show_nodes` | bool | `True` | Show nodes |

#### Voltage Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `line_voltage_threshold` | float | `300.0` | Threshold (kV) for high/low voltage |

#### Colors

| Parameter | Default | Description |
|-----------|---------|-------------|
| `line_high_voltage_color` | `"#029E73"` | High voltage lines (>threshold) |
| `line_low_voltage_color` | `"#CA9161"` | Low voltage lines (≤threshold) |
| `trafo_color` | `"#ECE133"` | Transformers |
| `dc_link_color` | `"#CC78BC"` | DC links |
| `node_color` | `"#0173B2"` | Nodes (non-clustered) |

#### Geometry

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `edge_width` | float | `1.5` | Line width in pixels |
| `node_size` | int | `5` | Node marker size |

#### Map Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `map_center_lat` | float | `57.5` | Map center latitude |
| `map_center_lon` | float | `14.0` | Map center longitude |
| `map_zoom` | float | `3.7` | Initial zoom level |
| `map_style` | str | `"carto-positron"` | Map tile style |

### Map Styles

Available Plotly map styles:

| Style | Description |
|-------|-------------|
| `"carto-positron"` | Light, clean style (default) |
| `"carto-darkmatter"` | Dark theme |
| `"open-street-map"` | OpenStreetMap tiles |
| `"stamen-terrain"` | Terrain with shading |
| `"stamen-toner"` | High-contrast black/white |

## Customizing Cluster Colors

For clustered plots, customize the color scale:

```python
config = PlotConfig(
    cluster_colorscale="Viridis"  # Plotly colorscale name
)

manager.plot_network(style="clustered", config=config)
```

Available colorscales include:
- `"Viridis"` (default)
- `"Plasma"`
- `"Inferno"`
- `"Magma"`
- `"Cividis"`
- `"Rainbow"`
- `"Jet"`
- `"Turbo"`

## Preset Configurations

Use the `preset` argument to apply curated styling for different audiences without re-writing `PlotConfig`.

| Preset | Description |
|--------|-------------|
| `default` | Balanced defaults for data exploration |
| `presentation` | Wide canvas with thicker lines and `"open-street-map"` tiles |
| `dense` | Compact view, higher voltage threshold, and dark tiles |
| `cluster_highlight` | Turbo colorscale with large nodes and white background |

```python
from npap import PlotPreset

manager.plot_network(style="voltage_aware", preset=PlotPreset.PRESENTATION)
```

Presets layer on top of `config`/`kwargs`; any explicit `PlotConfig` parameter overrides the preset values.

## Working with Figures

### Getting the Figure Object

```python
# Get the Plotly figure without displaying
fig = manager.plot_network(style="voltage_aware", show=False)

# Customize further
fig.update_layout(
    title="My Network",
    title_x=0.5
)

# Save to file
fig.write_html("network.html")
fig.write_image("network.png")

# Display
fig.show()
```

### Adding Custom Traces

```python
import plotly.graph_objects as go

fig = manager.plot_network(style="simple", show=False)

# Add custom markers
fig.add_trace(go.Scattermapbox(
    lat=[47.0, 47.5],
    lon=[15.0, 15.5],
    mode="markers",
    marker=dict(size=15, color="red"),
    name="Critical Nodes"
))

fig.show()
```

## Visualizing Different Graphs

### Original vs Aggregated

```python
# Load and partition
manager.load_data("csv_files", node_file="...", edge_file="...")
partition = manager.partition("geographical_kmeans", n_clusters=20)

# Plot original with clusters
fig_original = manager.plot_network(
    style="clustered",
    title="Original Network (Clustered)",
    show=False
)

# Aggregate
aggregated = manager.aggregate(mode=AggregationMode.GEOGRAPHICAL)

# Plot aggregated
fig_aggregated = manager.plot_network(
    graph=aggregated,
    style="simple",
    title="Aggregated Network",
    show=False
)

# Display side by side (in Jupyter)
from plotly.subplots import make_subplots
# ... or save both
fig_original.write_html("original.html")
fig_aggregated.write_html("aggregated.html")
```

### Plotting External Graphs

Plot any NetworkX graph with geographic coordinates:

```python
import networkx as nx

# Create or load a graph
G = nx.DiGraph()
G.add_node("A", lat=47.0, lon=15.0)
G.add_node("B", lat=47.1, lon=15.1)
G.add_edge("A", "B")

# Plot it
fig = manager.plot_network(graph=G, style="simple")
```

## Performance Optimization

### Large Networks

For networks with >10,000 edges:

```python
config = PlotConfig(
    show_nodes=False,    # Hide nodes for faster rendering
    edge_width=0.5       # Thinner lines
)

manager.plot_network(style="simple", config=config)
```

### Edge Grouping

NPAP automatically groups edges by type for efficient rendering:

- All high-voltage lines → single trace
- All low-voltage lines → single trace
- All transformers → single trace
- All DC links → single trace

This dramatically improves performance compared to rendering each edge separately.

## Interactive Features

Plotly figures are interactive:

- **Zoom**: Scroll wheel or pinch
- **Pan**: Click and drag
- **Hover**: Shows node/edge information
- **Legend**: Click to toggle visibility
- **Save**: Download as PNG from toolbar

### Hover Information

Default hover shows:
- **Nodes**: Node ID, coordinates
- **Edges**: Source, target, type

Customize with Plotly's hovertemplate:

```python
fig = manager.plot_network(style="simple", show=False)

# Update node hover
for trace in fig.data:
    if trace.name == "Nodes":
        trace.hovertemplate = "Node: %{text}<br>Lat: %{lat}<br>Lon: %{lon}"

fig.show()
```

## Examples

### Power System with All Edge Types

```python
# Load voltage-aware data
manager.load_data(
    "va_loader",
    node_file="buses.csv",
    line_file="lines.csv",
    transformer_file="transformers.csv",
    link_file="dc_links.csv"
)

# Configure for power system visualization
config = PlotConfig(
    line_voltage_threshold=300,        # 300 kV threshold
    line_high_voltage_color="#c0392b", # Dark red for EHV
    line_low_voltage_color="#2980b9",  # Blue for HV
    trafo_color="#8e44ad",             # Purple for transformers
    dc_link_color="#27ae60",           # Green for DC
    map_style="carto-darkmatter"       # Dark theme
)

manager.plot_network(style="voltage_aware", config=config)
```

### Partition Analysis

```python
# Compare different partition sizes
for n_clusters in [5, 10, 20, 50]:
    partition = manager.partition("geographical_kmeans", n_clusters=n_clusters)

    fig = manager.plot_network(
        style="clustered",
        title=f"Partitioning with {n_clusters} Clusters",
        show=False
    )

    fig.write_html(f"partition_{n_clusters}.html")
```

### Animation (Advanced)

Create animated visualizations:

```python
import plotly.graph_objects as go

# Create frames for different partition sizes
frames = []
for n_clusters in range(5, 51, 5):
    partition = manager.partition("geographical_kmeans", n_clusters=n_clusters)
    fig = manager.plot_network(style="clustered", show=False)
    frames.append(go.Frame(data=fig.data, name=str(n_clusters)))

# Create animated figure
animated_fig = go.Figure(
    data=frames[0].data,
    frames=frames,
    layout=go.Layout(
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play", method="animate")]
        )]
    )
)

animated_fig.show()
```

## Troubleshooting

### Missing Coordinates

```python
# Check for nodes without coordinates
missing = [n for n, d in graph.nodes(data=True)
           if "lat" not in d or "lon" not in d]

if missing:
    print(f"Nodes without coordinates: {missing[:5]}...")
```

### Empty Plot

If the plot appears empty:

1. Check that nodes have `lat` and `lon` attributes
2. Verify coordinate ranges are reasonable (lat: -90 to 90, lon: -180 to 180)
3. Adjust `map_center_lat`, `map_center_lon`, and `map_zoom`

### Slow Rendering

For slow rendering:

1. Reduce edge width
2. Hide nodes
3. Use `style="simple"` instead of `"voltage_aware"`
4. Aggregate the network before plotting
