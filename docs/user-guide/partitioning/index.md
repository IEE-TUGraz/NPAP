# Partitioning

Partitioning divides a network into clusters of nodes based on distance metrics. NPAP provides four partitioning strategy families, each suited for different use cases.

## Overview

| Strategy Family | Distance Metric            | Use Case                                  |
|-----------------|----------------------------|-------------------------------------------|
| Geographical | Euclidean / Haversine      | Spatial proximity clustering              |
| Electrical | PTDF-based                 | Electrical behavior clustering            |
| VA Geographical | Geographical + Voltage aware | Multi-voltage-level networks              |
| VA Electrical | PTDF + Voltage aware       | Multi-voltage-level electrical clustering |

## Basic Usage

All partitioning is done through the {py:class}`~npap.PartitionAggregatorManager`:

```python
import npap

manager = npap.PartitionAggregatorManager()
manager.load_data("networkx_direct", graph=G)

# Partition the network
partition_result = manager.partition(
    strategy="geographical_kmeans",
    n_clusters=10
)
```

The result is a {py:class}`~npap.PartitionResult` object:

```python
print(partition_result.mapping)
# {0: ['bus_1', 'bus_2'], 1: ['bus_3', 'bus_4'], ...}

print(partition_result.n_clusters)
# 10

print(partition_result.strategy_name)
# 'geographical_kmeans'
```

```{toctree}
:hidden:
:maxdepth: 2

geographical
electrical
va-geographical
va-electrical
```

## DC-Island Awareness

### What Are DC Islands?

In networks with HVDC links, AC-connected regions form **DC islands**. These islands:

- Are electrically separate in AC sense
- Connected only through DC links
- Have independent power flow characteristics

### How NPAP Handles DC Islands

When nodes have a `dc_island` attribute (automatically set by `va_loader`):

1. **Distance matrices** include infinite distances between islands
2. **Clustering algorithms** using precomputed distances respect these boundaries
3. **Partitions** never span across DC island boundaries

### Algorithm Support

| Algorithm | DC-Island Support |
|-----------|-------------------|
| K-Means | No (works on raw coordinates) |
| K-Medoids | Yes |
| DBSCAN | Yes |
| HDBSCAN | Yes |
| Hierarchical (non-ward) | Yes |
| Hierarchical (ward) | No |

```{note}
K-Means and Ward linkage will issue a warning if DC islands are detected but will proceed without respecting island boundaries.
```

## Choosing a Strategy

### Decision Guide

```{mermaid}
flowchart TD
    A[Start] --> B{Multi-voltage?}
    B -->|Yes| C{Need electrical distance?}
    B -->|No| D{Need electrical distance?}
    C -->|Yes| E[va_electrical_*]
    C -->|No| F[va_geographical_*]
    D -->|Yes| G[electrical_*]
    D -->|No| H{DC islands?}
    H -->|Yes| I[geographical_kmedoids_*]
    H -->|No| K[geographical_kmeans]

    style A fill:#2993B5,stroke:#1d6f8a,color:#fff
    style B fill:#FFBF00,stroke:#cc9900,color:#1e293b
    style C fill:#FFBF00,stroke:#cc9900,color:#1e293b
    style D fill:#FFBF00,stroke:#cc9900,color:#1e293b
    style E fill:#0fad6b,stroke:#076b3f,color:#fff
    style F fill:#0fad6b,stroke:#076b3f,color:#fff
    style G fill:#0fad6b,stroke:#076b3f,color:#fff
    style H fill:#FFBF00,stroke:#cc9900,color:#1e293b
    style I fill:#0fad6b,stroke:#076b3f,color:#fff
    style K fill:#0fad6b,stroke:#076b3f,color:#fff
```

### Recommendations by Use Case

| Use Case                     | Recommended Strategy |
|------------------------------|---------------------|
| Geographical clustering      | `geographical_kmeans` |
| Geographical with DC islands | `geographical_kmedoids_haversine` |
| Electrical behavior grouping | `electrical_kmedoids` |
| Multi-voltage network        | `va_geographical_kmedoids_haversine` |
| Unknown cluster count        | `geographical_dbscan_*` or `geographical_hdbscan_*` |

### Performance Comparison

| Strategy | Speed | Memory | Robustness |
|----------|-------|--------|------------|
| K-Means | Fast | Low | Medium |
| K-Medoids | Medium | High (distance matrix) | High |
| DBSCAN | Medium | Medium | High |
| HDBSCAN | Slow | High | Very High |
| Hierarchical | Medium | High | High |

## Configuration Options

All geographical partitioning strategies share a common configuration:

```python
from npap.partitioning import GeographicalConfig

config = GeographicalConfig(
    random_state=42,              # Reproducibility
    max_iter=300,                 # K-Means iterations
    n_init=10,                    # K-Means initializations
    hierarchical_linkage="ward",  # Hierarchical linkage
    infinite_distance=1e4         # DC island separation
)
```

Electrical strategies have their own configuration:

```python
from npap.partitioning import ElectricalDistanceConfig

config = ElectricalDistanceConfig(
    zero_reactance_replacement=1e-5,
    regularization_factor=1e-10,
    infinite_distance=1e4
)
```

## Working with Partition Results

### Inspecting Results

```python
partition = manager.partition("geographical_kmeans", n_clusters=5)

# Cluster mapping
print(partition.mapping)
# {0: ['bus_1', 'bus_2'], 1: ['bus_3'], ...}

# Number of clusters
print(partition.n_clusters)

# Strategy used
print(partition.strategy_name)

# Graph hash for validation
print(partition.original_graph_hash)
```

### Reusing Partitions

Partition results can be reused for multiple aggregations:

```python
# Partition once
partition = manager.partition("geographical_kmeans", n_clusters=10)

# Aggregate with different profiles
simple_agg = manager.aggregate(partition, mode=AggregationMode.SIMPLE)
geo_agg = manager.aggregate(partition, mode=AggregationMode.GEOGRAPHICAL)
```

### Validation

NPAP validates that partitions match their source graph:

```python
# This will raise GraphCompatibilityError if graph changed
aggregated = manager.aggregate(partition)
```
