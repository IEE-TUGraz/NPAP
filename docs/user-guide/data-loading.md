# Data Loading

NPAP provides three data loading strategies to import network data from various sources. Each strategy is designed for different use cases and data formats.

## Overview

| Strategy | Use Case                           | Output |
|----------|------------------------------------|--------|
| `networkx_direct` | Programmatic or pre-defined graphs | DiGraph or MultiDiGraph |
| `csv_files` | Separate node/edge CSV files       | DiGraph or MultiDiGraph |
| `va_loader` | Voltage-aware power systems        | DiGraph or MultiDiGraph |

All strategies are accessed through the {py:class}`~npap.PartitionAggregatorManager`:

```python
import npap

manager = npap.PartitionAggregatorManager()
graph = manager.load_data("strategy_name", **kwargs)
```

## NetworkX Direct Strategy

The simplest approach for loading graphs created programmatically or from other sources.

### Basic Usage

```python
import networkx as nx
import npap

# Create a graph
G = nx.DiGraph()
G.add_node("bus_1", lat=47.0, lon=15.0, load=100)
G.add_node("bus_2", lat=47.1, lon=15.1, load=150)
G.add_edge("bus_1", "bus_2", x=0.01, p_max=500)

# Load into NPAP
manager = npap.PartitionAggregatorManager()
graph = manager.load_data("networkx_direct", graph=G)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `graph` | NetworkX graph | *required* | Any NetworkX graph type |
| `bidirectional` | bool | `True` | Create both directions for undirected edges |

### Graph Type Handling

The strategy handles different NetworkX graph types:

```python
# Undirected graphs are converted to directed
G_undirected = nx.Graph()
G_undirected.add_edge("A", "B", weight=1.0)

# With bidirectional=True (default), creates edges A→B and B→A
graph = manager.load_data("networkx_direct", graph=G_undirected)

# With bidirectional=False, only creates A→B
graph = manager.load_data("networkx_direct", graph=G_undirected, bidirectional=False)
```

### Supported Input Types

- `nx.Graph` → Converted to `nx.DiGraph`
- `nx.DiGraph` → Used directly
- `nx.MultiGraph` → Converted to `nx.MultiDiGraph`
- `nx.MultiDiGraph` → Used directly

## CSV Files Strategy

Load network data from separate CSV files for nodes and edges.

### Basic Usage

```python
graph = manager.load_data(
    "csv_files",
    node_file="path/to/nodes.csv",
    edge_file="path/to/edges.csv"
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `node_file` | str | *required* | Path to nodes CSV file |
| `edge_file` | str | *required* | Path to edges CSV file |
| `node_id_col` | str | auto-detect | Column name for node IDs |
| `edge_from_col` | str | auto-detect | Column name for edge source |
| `edge_to_col` | str | auto-detect | Column name for edge target |
| `delimiter` | str | `","` | CSV delimiter character |
| `decimal` | str | `"."` | Decimal separator |

### File Format

**nodes.csv**:
```text
id,lat,lon,load,voltage
bus_1,47.0667,15.4333,100.0,380
bus_2,47.1234,15.5678,150.0,380
bus_3,48.2089,16.3726,200.0,220
```

**edges.csv**:
```text
from,to,x,r,p_max,length
bus_1,bus_2,0.0123,0.0045,500,125.5
bus_2,bus_3,0.0234,0.0089,400,98.2
```

### Auto-Detection

The strategy automatically detects:

- **Node ID column**: Looks for columns named `id`, `node_id`, `bus_id`, `name`
- **Edge from column**: Looks for `from`, `source`, `from_bus`, `bus0`
- **Edge to column**: Looks for `to`, `target`, `to_bus`, `bus1`

### Handling Parallel Edges

If the CSV contains multiple edges between the same node pair, a `MultiDiGraph` is returned.

```{warning}
A warning in the console will be issued if parallel edges are detected because a MultiDigraph cannot be partitioned. You will see the message:

"Parallel edges detected in CSV edge file. A MultiDiGraph will be created. Call manager.aggregate_parallel_edges() to collapse parallel edges before partitioning."
```

## Voltage-Aware Strategy

The most comprehensive loader for power system networks with multiple voltage levels, transformers, and DC links.

### Basic Usage

```python
graph = manager.load_data(
    "va_loader",
    node_file="buses.csv",
    line_file="lines.csv",
    transformer_file="transformers.csv",
    converter_file="converters.csv",
    link_file="dc_links.csv"
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `node_file` | str | *required* | Path to buses/substations CSV |
| `line_file` | str | *required* | Path to AC transmission lines CSV |
| `transformer_file` | str | *required* | Path to transformers CSV |
| `converter_file` | str | *required* | Path to AC/DC converters CSV |
| `link_file` | str | *required* | Path to DC links CSV |

### File Formats

**buses.csv** (nodes):
```text
id,lat,lon,voltage,p_load,q_load
bus_001,47.0667,15.4333,380,100.0,30.0
bus_002,47.1234,15.5678,380,150.0,45.0
bus_003,48.2089,16.3726,220,200.0,60.0
```

**lines.csv** (AC transmission lines):
```text
from,to,x,r,b,p_max,length
bus_001,bus_002,0.0123,0.0045,0.001,500,125.5
bus_002,bus_004,0.0234,0.0089,0.002,400,98.2
```

**transformers.csv**:
```text
from,to,x,r,s_nom,tap_ratio
bus_002,bus_003,0.05,0.01,400,1.0
bus_005,bus_006,0.04,0.008,500,1.05
```

**converters.csv** (AC/DC converters):
```text
id,bus,p_max
conv_1,bus_010,1000
conv_2,bus_020,1000
```

**dc_links.csv** (HVDC links):
```text
from,to,p_max,converter_from,converter_to
conv_1,conv_2,800,conv_1,conv_2
```

### AC Island Detection

The voltage-aware loader automatically detects **AC islands** - groups of AC-connected buses separated by DC links:

```{mermaid}
flowchart LR
    subgraph Island1[AC Island 0]
        A[Bus A] --- B[Bus B]
        B --- C[Bus C]
    end

    subgraph Island2[AC Island 1]
        D[Bus D] --- E[Bus E]
        E --- F[Bus F]
    end

    C -.->|DC Link| D

    style A fill:#2993B5,stroke:#1d6f8a,color:#fff
    style B fill:#2993B5,stroke:#1d6f8a,color:#fff
    style C fill:#2993B5,stroke:#1d6f8a,color:#fff
    style D fill:#0fad6b,stroke:#076b3f,color:#fff
    style E fill:#0fad6b,stroke:#076b3f,color:#fff
    style F fill:#0fad6b,stroke:#076b3f,color:#fff
```

Each node receives a `ac_island` attribute:

```python
graph = manager.load_data("va_loader", ...)

# Check AC island assignments
for node, data in graph.nodes(data=True):
    print(f"{node}: AC Island {data['ac_island']}")
```

### Edge Type Classification

All edges are classified by type:

```python
from npap import EdgeType

for u, v, data in graph.edges(data=True):
    edge_type = data["type"]
    if edge_type == EdgeType.LINE:
        print(f"AC Line: {u} -> {v}")
    elif edge_type == EdgeType.TRAFO:
        print(f"Transformer: {u} -> {v}")
    elif edge_type == EdgeType.DC_LINK:
        print(f"DC Link: {u} -> {v}")
```

### Unified Edge Schema

All edges have a unified attribute schema:

| Attribute | Description |
|-----------|-------------|
| `type` | Edge type (EdgeType enum) |
| `primary_voltage` | Voltage at source node |
| `secondary_voltage` | Voltage at target node |
| `x` | Reactance (AC elements only) |
| `r` | Resistance (if available) |
| `p_max` | Maximum power flow |


## Aggregating Parallel Edges

After loading, you may want to aggregate parallel edges (e.g., parallel transmission lines) into single edges:

```python
# Load data
graph = manager.load_data("csv_files", ...)

# Check if parallel edges exist
if isinstance(graph, nx.MultiDiGraph):
    # Aggregate parallel edges
    manager.aggregate_parallel_edges(
        edge_properties={
            "x": "equivalent_reactance",  # Parallel reactance formula
            "r": "equivalent_reactance",  # Same for resistance
            "p_max": "sum",               # Sum capacities
            "length": "average"           # Average length
        },
        default_strategy="average",
        warn_on_defaults=True
    )
```

```{note}
MultiDiGraphs cannot be partitioned directly. First, you will need to collapse parallel edges using the  `aggregate_parallel_edges()` method.
```

### Aggregation Strategies for Parallel Edges

| Strategy | Formula | Use Case |
|----------|---------|----------|
| `sum` | $\sum x_i$ | Capacities, power |
| `average` | $\frac{1}{n}\sum x_i$ | General numeric |
| `first` | $x_1$ | Non-numeric attributes |
| `equivalent_reactance` | $\frac{1}{\sum \frac{1}{x_i}}$ | Parallel impedances |

## Voltage Level Grouping

For voltage-aware networks, you can group voltage levels:

```python
# Load voltage-aware data
graph = manager.load_data("va_loader", ...)

# Group similar voltage levels
manager.group_by_voltage_levels(
    target_levels=[220, 380],      # Target voltage levels in kV
    voltage_attr="voltage",         # Attribute containing voltage
)
```

This maps similar voltages to standard levels:
- 225 kV → 220 kV
- 375 kV → 380 kV
- 400 kV → 380 kV

## Best Practices

### 1. Validate Your Data

Check that required attributes exist after loading:

```python
graph = manager.load_data("csv_files", ...)

# Check for geographic coordinates
missing_coords = [n for n, d in graph.nodes(data=True)
                  if "lat" not in d or "lon" not in d]
if missing_coords:
    print(f"Warning: {len(missing_coords)} nodes missing coordinates")
```

```{tip}
Check as well all required attributes are named correctly in your CSV files before loading.
```

### 2. Handle Missing Values

NPAP strategies may fail if required attributes are missing. Clean your data first:

```python
import pandas as pd

# Load and clean CSV before using NPAP
nodes_df = pd.read_csv("nodes.csv")
nodes_df = nodes_df.dropna(subset=["lat", "lon"])
nodes_df.to_csv("nodes_clean.csv", index=False)

graph = manager.load_data("csv_files",
                          node_file="nodes_clean.csv",
                          edge_file="edges.csv")
```

### 3. Choose the Right Loader

| Scenario | Recommended Strategy |
|----------|---------------------|
| Simple networks, testing | `networkx_direct` |
| Standard CSV exports | `csv_files` |
| Power systems with voltage levels | `va_loader` |
| Networks with DC links | `va_loader` |
