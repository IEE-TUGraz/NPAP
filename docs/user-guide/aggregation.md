# Aggregation

Aggregation reduces a partitioned network by merging nodes within each cluster into single representative nodes. NPAP uses a **three-tier aggregation system** that separates topology creation, physical aggregation, and statistical property aggregation.

## Overview

### The Three-Tier System

```{mermaid}
flowchart LR
    A[Partitioned Graph] --> B[1. Topology Creation]
    B --> C[2. Physical Aggregation]
    C --> D[3. Statistical Aggregation]
    D --> E[Aggregated Graph]

    style A fill:#2993B5,stroke:#1d6f8a,color:#fff
    style B fill:#2993B5,stroke:#1d6f8a,color:#fff
    style C fill:#0fad6b,stroke:#076b3f,stroke-dasharray: 5 5,color:#fff
    style D fill:#2993B5,stroke:#1d6f8a,color:#fff
    style E fill:#2993B5,stroke:#1d6f8a,color:#fff
```

1. **Topology Creation**: Creates the structure of the aggregated graph (nodes and edges)
2. **Physical Aggregation**: Applies electrical laws to preserve physical behavior (optional)
3. **Statistical Aggregation**: Aggregates node and edge properties

### Quick Start

```python
import npap
from npap import AggregationMode

manager = npap.PartitionAggregatorManager()
manager.load_data("networkx_direct", graph=G)
partition = manager.partition("geographical_kmeans", n_clusters=10)

# Aggregate using a predefined mode
aggregated = manager.aggregate(mode=AggregationMode.GEOGRAPHICAL)
```

## Aggregation Modes

NPAP provides predefined aggregation modes for common use cases.

### Available Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `SIMPLE` | Sum all numeric properties | Basic reduction |
| `GEOGRAPHICAL` | Average coordinates, sum loads | Spatial analysis |
| `DC_KRON` | Kron reduction for DC networks | DC network analysis |
| `CUSTOM` | User-defined profile | Advanced use |

### SIMPLE Mode

Sums all numeric properties across clusters:

```python
from npap import AggregationMode

aggregated = manager.aggregate(mode=AggregationMode.SIMPLE)
```

Configuration:
- Topology: `simple`
- Node properties: `sum`
- Edge properties: `sum`

### GEOGRAPHICAL Mode

Designed for networks with geographic coordinates:

```python
aggregated = manager.aggregate(mode=AggregationMode.GEOGRAPHICAL)
```

Configuration:
- Topology: `simple`
- Node `lat`, `lon`: `average` (geographic center)
- Node `base_voltage`: `average`
- Edge `p_max`: `sum`
- Edge `x`: `average`
- Default: `average`

### CUSTOM Mode

For full control, create an {py:class}`~npap.AggregationProfile`:

```python
from npap import AggregationProfile

profile = AggregationProfile(
    topology_strategy="simple",
    physical_strategy=None,
    node_properties={
        "lat": "average",
        "lon": "average",
        "load": "sum",
        "generation": "sum"
    },
    edge_properties={
        "x": "equivalent_reactance",
        "r": "equivalent_reactance",
        "p_max": "sum"
    },
    default_node_strategy="sum",
    default_edge_strategy="average"
)

aggregated = manager.aggregate(profile=profile)
```

## Aggregation Profile

The {py:class}`~npap.AggregationProfile` controls all aspects of aggregation.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `topology_strategy` | str | `"simple"` | How to create graph structure |
| `physical_strategy` | str \| None | `None` | Physical aggregation method |
| `physical_properties` | list[str] | `[]` | Properties for physical aggregation |
| `physical_parameters` | dict | `{}` | Extra parameters for physical strategy |
| `node_properties` | dict[str, str] | `{}` | Property → aggregation method mapping |
| `edge_properties` | dict[str, str] | `{}` | Property → aggregation method mapping |
| `default_node_strategy` | str | `"average"` | Fallback for unmapped node properties |
| `default_edge_strategy` | str | `"sum"` | Fallback for unmapped edge properties |
| `warn_on_defaults` | bool | `True` | Warn when using default strategies |
| `mode` | AggregationMode | `CUSTOM` | Associated mode |

### Example Profiles

**Power Flow Analysis:**
```python
power_flow_profile = AggregationProfile(
    topology_strategy="simple",
    node_properties={
        "p_load": "sum",
        "q_load": "sum",
        "p_gen": "sum",
        "q_gen": "sum",
        "voltage": "average"
    },
    edge_properties={
        "x": "equivalent_reactance",
        "r": "equivalent_reactance",
        "b": "sum",
        "p_max": "sum"
    }
)
```

**Geographic Visualization:**
```python
geo_profile = AggregationProfile(
    topology_strategy="simple",
    node_properties={
        "lat": "average",
        "lon": "average",
        "name": "first"
    },
    edge_properties={
        "length": "sum"
    },
    default_node_strategy="first",
    default_edge_strategy="first"
)
```

## Topology Strategies

Topology strategies determine how the aggregated graph structure is created.

### Simple Topology

Creates edges only where connections existed in the original graph:

```python
profile = AggregationProfile(topology_strategy="simple")
```

- One node per cluster
- Edge between clusters if any original edge connected them
- Preserves network structure

```{mermaid}
flowchart LR
    subgraph Original
        A1[A1] --> B1[B1]
        A2[A2] --> B1
        A1 --> A2
    end

    subgraph Aggregated
        A[Cluster A] --> B[Cluster B]
    end

    Original --> Aggregated

    style A1 fill:#64748b,stroke:#475569,color:#fff
    style A2 fill:#64748b,stroke:#475569,color:#fff
    style B1 fill:#64748b,stroke:#475569,color:#fff
    style A fill:#2993B5,stroke:#1d6f8a,color:#fff
    style B fill:#0fad6b,stroke:#076b3f,color:#fff
```

### Electrical Topology

For use with physical aggregation strategies:

```python
profile = AggregationProfile(
    topology_strategy="electrical",
    connectivity="existing"  # or "full"
)
```

**Connectivity modes:**
- `existing`: Same as simple topology
- `full`: Creates fully connected graph

## Property Aggregation Strategies

### Node Property Strategies

| Strategy | Formula | Use Case |
|----------|---------|----------|
| `sum` | $\sum_{i \in C} x_i$ | Loads, generation, counts |
| `average` | $\frac{1}{\lvert C\rvert}\sum_{i \in C} x_i$ | Coordinates, voltage |
| `first` | $x_1$ | Names, IDs, categories |

**Examples:**
```python
node_properties={
    "load": "sum",        # Total load in cluster
    "lat": "average",     # Geographic center
    "name": "first",      # Keep first name
    "voltage": "average"  # Average voltage level
}
```

### Edge Property Strategies

| Strategy | Formula                                      | Use Case |
|----------|----------------------------------------------|----------|
| `sum` | $\sum_{e \in E} x_e$                         | Capacity, flow limits |
| `average` | $\frac{1}{\lvert E\rvert}\sum_{e \in E} x_e$ | General properties |
| `first` | $x_1$                                        | Type, category |
| `equivalent_reactance` | $\frac{1}{\sum_{e \in E} \frac{1}{x_e}}$     | Parallel impedances |

**Examples:**
```python
edge_properties={
    "p_max": "sum",              # Total capacity
    "x": "equivalent_reactance",  # Parallel reactance
    "length": "average",          # Average length
    "type": "first"              # Keep first type
}
```

### Equivalent Reactance

For parallel transmission lines, reactances combine as:

$$x_{eq} = \frac{1}{\sum_{i=1}^{n} \frac{1}{x_i}}$$

This is the electrical equivalent of parallel resistors/inductors.

```python
edge_properties={
    "x": "equivalent_reactance",
    "r": "equivalent_reactance"
}
```

## Physical Aggregation

Physical aggregation strategies preserve electrical laws during network reduction.

### Kron Reduction (Planned)

```{note}
Kron reduction is planned for a future release.
```

## Handling Defaults

When a property isn't explicitly mapped, NPAP uses the default strategy:

```python
profile = AggregationProfile(
    node_properties={"load": "sum"},   # Only load is mapped
    default_node_strategy="average",   # Everything else uses average
    warn_on_defaults=True              # Warn when using defaults
)
```

### Warnings

With `warn_on_defaults=True`, you'll see warnings like:

```
UserWarning: Using default strategy 'average' for node property 'voltage'
```

This helps identify properties you may want to explicitly configure.

## Aggregating Parallel Edges

Before partitioning, you need to aggregate parallel edges in a `MultiDiGraph`:

```python
# Load data (may return MultiDiGraph)
graph = manager.load_data("csv_files", node_file="...", edge_file="...")

# Aggregate parallel edges
if isinstance(graph, nx.MultiDiGraph):
    manager.aggregate_parallel_edges(
        edge_properties={
            "x": "equivalent_reactance",
            "r": "equivalent_reactance",
            "p_max": "sum",
            "length": "average"
        },
        default_strategy="average",
        warn_on_defaults=False
    )
```

This converts the `MultiDiGraph` to a `DiGraph` with single edges.

## Complete Workflow Example

```python
import npap
from npap import AggregationProfile, AggregationMode
import networkx as nx

# Create manager
manager = npap.PartitionAggregatorManager()

# Load data
graph = manager.load_data("csv_files",
                          node_file="nodes.csv",
                          edge_file="edges.csv")

# Handle parallel edges if present
if isinstance(graph, nx.MultiDiGraph):
    manager.aggregate_parallel_edges(
        edge_properties={"x": "equivalent_reactance", "p_max": "sum"}
    )

# Partition
partition = manager.partition("geographical_kmeans", n_clusters=20)

# Option 1: Use predefined mode
aggregated = manager.aggregate(mode=AggregationMode.GEOGRAPHICAL)

# Option 2: Use custom profile
custom_profile = AggregationProfile(
    topology_strategy="simple",
    node_properties={
        "lat": "average",
        "lon": "average",
        "load": "sum",
        "generation": "sum"
    },
    edge_properties={
        "x": "equivalent_reactance",
        "p_max": "sum"
    },
    default_node_strategy="sum",
    default_edge_strategy="sum"
)
aggregated = manager.aggregate(profile=custom_profile)

# Inspect result
print(f"Original: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
print(f"Aggregated: {aggregated.number_of_nodes()} nodes, {aggregated.number_of_edges()} edges")
```

## Accessing Mode Profiles

You can retrieve the profile for any predefined mode:

```python
from npap import get_mode_profile, AggregationMode

# Get the GEOGRAPHICAL mode profile
geo_profile = get_mode_profile(AggregationMode.GEOGRAPHICAL)

print(geo_profile.topology_strategy)
# 'simple'

print(geo_profile.node_properties)
# {'lat': 'average', 'lon': 'average', 'base_voltage': 'average'}
```

This is useful for:
- Inspecting what a mode does
- Using a mode as a starting point for customization

```python
# Start from GEOGRAPHICAL and customize
profile = get_mode_profile(AggregationMode.GEOGRAPHICAL)
profile.node_properties["custom_attr"] = "sum"
profile.edge_properties["custom_edge"] = "average"

aggregated = manager.aggregate(profile=profile)
```

## Best Practices

### 1. Choose Appropriate Strategies

| Property Type | Recommended Strategy |
|---------------|---------------------|
| Extensive (load, generation) | `sum` |
| Intensive (voltage, temperature) | `average` |
| Impedance (reactance, resistance) | `equivalent_reactance` |
| Categorical (name, type) | `first` |
| Capacity (p_max, rating) | `sum` |
| Length, distance | `sum` or `average` |

### 2. Validate Results

Check that aggregation preserves important quantities:

```python
# Total load should be preserved
original_load = sum(graph.nodes[n].get("load", 0) for n in graph.nodes())
aggregated_load = sum(aggregated.nodes[n].get("load", 0) for n in aggregated.nodes())

assert abs(original_load - aggregated_load) < 1e-6, "Load not conserved!"
```

### 3. Handle Missing Properties

Properties may not exist on all nodes/edges:

```python
# Check for missing properties before aggregation
missing = [n for n in graph.nodes() if "load" not in graph.nodes[n]]
if missing:
    print(f"Warning: {len(missing)} nodes missing 'load' attribute")
```

### 4. Document Your Profile

When using custom profiles, document your choices:

```python
profile = AggregationProfile(
    topology_strategy="simple",
    node_properties={
        "lat": "average",     # Geographic center of cluster
        "lon": "average",     # Geographic center of cluster
        "p_load": "sum",      # Total active load (MW)
        "q_load": "sum",      # Total reactive load (MVAr)
    },
    edge_properties={
        "x": "equivalent_reactance",  # Parallel combination
        "p_max": "sum",               # Total transfer capacity
    }
)
```
