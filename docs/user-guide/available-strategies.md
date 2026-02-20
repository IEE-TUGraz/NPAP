# Available Strategies

NPAP uses the [**Strategy Pattern**](https://refactoring.guru/design-patterns/strategy) to provide flexible, interchangeable algorithms for data loading, partitioning, and aggregation. This page provides an overview of all available strategies and how they work together.

## The Workflow

A typical NPAP workflow consists of four main steps:

```{mermaid}
flowchart LR
    A[Load Data] --> B[Partition]
    B --> C[Aggregate]
    C --> D[Visualize]

    style A fill:#2993B5,stroke:#1d6f8a,color:#fff
    style B fill:#2993B5,stroke:#1d6f8a,color:#fff
    style C fill:#2993B5,stroke:#1d6f8a,color:#fff
    style D fill:#2993B5,stroke:#1d6f8a,color:#fff
```

Each step uses a specific strategy that can be swapped based on your requirements:

| Step | Manager | Strategy Type | Purpose |
|------|---------|---------------|---------|
| Load Data | InputDataManager | DataLoadingStrategy | Import network from various sources |
| Partition | PartitioningManager | PartitioningStrategy | Cluster nodes |
| Aggregate | AggregationManager | Multiple strategies | Reduce network by merging clusters |
| Visualize | (built-in) | PlotStyle | Create interactive maps |

## Strategy Pattern Architecture

```{mermaid}
flowchart TB
    subgraph Managers
        PAM[PartitionAggregatorManager]
        IM[InputDataManager]
        PM[PartitioningManager]
        AM[AggregationManager]
    end

    subgraph Strategies
        DL[Data Loading Strategies]
        PS[Partitioning Strategies]
        AS[Aggregation Strategies]
    end

    PAM --> IM
    PAM --> PM
    PAM --> AM
    IM --> DL
    PM --> PS
    AM --> AS

    style PAM fill:#2993B5,stroke:#1d6f8a,color:#fff
    style IM fill:#2993B5,stroke:#1d6f8a,color:#fff
    style PM fill:#2993B5,stroke:#1d6f8a,color:#fff
    style AM fill:#2993B5,stroke:#1d6f8a,color:#fff
    style DL fill:#0fad6b,stroke:#076b3f,color:#fff
    style PS fill:#0fad6b,stroke:#076b3f,color:#fff
    style AS fill:#0fad6b,stroke:#076b3f,color:#fff
```

The {py:class}`~npap.PartitionAggregatorManager` is the main entry point that coordinates three specialized managers. Each manager handles its category of strategies.

### How Strategies Work

Strategies are registered with their managers and invoked by name:

```python
import npap

manager = npap.PartitionAggregatorManager()

# Loading strategy is selected by name
manager.load_data("networkx_direct", graph=G)

# Partitioning strategy is selected by name
partition = manager.partition("geographical_kmeans", n_clusters=10)

# Aggregation uses profiles and modes
aggregated = manager.aggregate(mode=npap.AggregationMode.GEOGRAPHICAL)
```

## Data Loading Strategies

| Strategy | Description                                   | Required Parameters |
|----------|-----------------------------------------------|---------------------|
| `networkx_direct` | Load from existing NetworkX graph             | `graph` |
| `csv_files` | Load from separate CSV files                  | `node_file`, `edge_file` |
| `va_loader` | Voltage-aware power system loader (CSV files) | `node_file`, `line_file`, `transformer_file` |

### When to Use Each Loader

| Scenario | Recommended Strategy |
|----------|---------------------|
| Programmatic graph creation | `networkx_direct` |
| Standard CSV exports | `csv_files` |
| Multi-voltage power systems | `va_loader` |
| Networks with DC links | `va_loader` |

See [Data Loading](data-loading.md) for detailed documentation.

## Partitioning Strategies

NPAP provides four families of partitioning strategies:

### Geographical Partitioning

Clusters nodes based on geographic coordinates.

| Strategy | Algorithm | Metric | AC-Island Aware |
|----------|-----------|--------|-----------------|
| `geographical_kmeans` | K-Means | Euclidean | No |
| `geographical_kmedoids_euclidean` | K-Medoids | Euclidean | Yes |
| `geographical_kmedoids_haversine` | K-Medoids | Haversine | Yes |
| `geographical_dbscan_euclidean` | DBSCAN | Euclidean | Yes |
| `geographical_dbscan_haversine` | DBSCAN | Haversine | Yes |
| `geographical_hierarchical` | Agglomerative | Euclidean | Yes* |
| `geographical_hdbscan_euclidean` | HDBSCAN | Euclidean | Yes |
| `geographical_hdbscan_haversine` | HDBSCAN | Haversine | Yes |

*Ward linkage does not support AC-island awareness

**Required node attributes**: `lat`, `lon`

### Electrical Partitioning

Clusters nodes based on electrical distance (PTDF approach) respecting AC islands.

| Strategy | Algorithm | Description                                           |
|----------|-----------|-------------------------------------------------------|
| `electrical_kmeans` | K-Means | Provides arbitrary centroid node                      |
| `electrical_kmedoids` | K-Medoids | Provides existing centroid node (e.g. Kron-Reduction) |

**Required attributes**: Nodes: `ac_island` | Edges: `x` (reactance)

### Voltage-Aware Geographical Partitioning

Combines geographical distance with voltage level and AC island constraints.

Voltage-aware geographical partitioning provides two modes for defining the number of clusters per voltage level:
1. **Standard**: Only the total number of clusters over all voltage levels is specified by the user. The number of clusters per voltage level is set by the clustering algorithm.
2. **Proportional**: The user specifies the total number of clusters and the number of clusters per voltage level gets defined proportionally to the number of nodes at each voltage level.

| Strategy | Mode | Description                       |
|----------|------|-----------------------------------|
| `va_geographical_kmedoids_euclidean` | Standard | K-Medoids with Euclidean distance |
| `va_geographical_kmedoids_haversine` | Standard | K-Medoids with Haversine distance |
| `va_geographical_hierarchical` | Standard | Agglomerative clustering          |
| `va_geographical_proportional_kmedoids_euclidean` | Proportional | Proportional by number of nodes   |
| `va_geographical_proportional_kmedoids_haversine` | Proportional | Proportional by number of nodes                 |
| `va_geographical_proportional_hierarchical` | Proportional | Proportional by number of nodes             |

**Required node attributes**: `lat`, `lon`, `voltage`, `ac_island`

### Voltage-Aware Electrical Partitioning

Combines electrical distance (PTDF approach) with voltage level and AC island constraints.

| Strategy | Algorithm |
|----------|-----------|
| `va_electrical_kmedoids` | K-Medoids |
| `va_electrical_hierarchical` | Agglomerative |

**Required attributes**: Nodes: `voltage`, `ac_island` | Edges: `x` (reactance)

### Graph-theory Partitioning

Graph-theory strategies rely on matrix-based clustering or community detection rather than physical distances.

| Strategy | Algorithm | Description |
|----------|-----------|-------------|
| `spectral_clustering` | Spectral clustering (precomputed adjacency) | Ideal for networks with loose geographic signals; adapts to the graph Laplacian. |
| `community_modularity` | Greedy modularity communities | Detects naturally modular regions without any tunable `n_clusters`. |

Spectral clustering expects `n_clusters` as an argument and splits the adjacency matrix via Eigen-decomposition, while the community strategy returns the modularity-maximizing partition automatically.

### Adjacent-node Agglomerative Clustering

`adjacent_agglomerative` merges groups only via edges that already exist in the graph. It is useful for topology-aware workflows that should not combine disconnected buses even when their attributes look similar.

**Required node attributes**: depends on `AdjacentAgglomerativeConfig.node_attribute`; any numeric attribute can guide the merging order.

**Configuration**: use {py:class}`~npap.partitioning.adjacent.AdjacentAgglomerativeConfig` to choose the attribute that scores merges and to keep AC islands separate via `ac_island_attr`. When the attribute is omitted, merges fall back to any adjacent pair in deterministic order.

```python
from npap.partitioning import AdjacentAgglomerativeConfig
from npap import PartitionAggregatorManager

config = AdjacentAgglomerativeConfig(node_attribute="load", ac_island_attr="ac_island")
manager = PartitionAggregatorManager()
partition = manager.partition("adjacent_agglomerative", n_clusters=4, config=config)
```

### LMP Partitioning

The `lmp_similarity` strategy groups buses with similar locational marginal prices (LMPs, typically provided by an OPF or market simulation) and optionally favours directly connected nodes through the `adjacency_bonus` configuration parameter. Nodes with different `ac_island` values are always separated by a large `infinite_distance` to keep islands apart.

**Required node attributes**: `lmp` (or a custom attribute specified via `price_attribute`)

**Configuration**: Use {py:class}`~npap.partitioning.lmp.LMPPartitioningConfig` to override the attribute names, adjacency bonus, or infinite-distance penalty.

```python
from npap.partitioning import LMPPartitioningConfig
from npap import PartitionAggregatorManager

config = LMPPartitioningConfig(
    price_attribute="locational_price",
    adjacency_bonus=0.2,
    infinite_distance=1e5,
)

manager = PartitionAggregatorManager()
partition = manager.partition("lmp_similarity", n_clusters=3, config=config)
```

### Choosing a Partitioning Strategy

```{mermaid}
flowchart TD
    A[Start] --> B{Multi-voltage?}
    B -->|Yes| C{Need electrical distance?}
    B -->|No| D{Need electrical distance?}
    C -->|Yes| E[va_electrical_*]
    C -->|No| F[va_geographical_*]
    D -->|Yes| G[electrical_*]
    D -->|No| H{AC islands?}
    H -->|Yes| I[geographical_kmedoids_*]
    H -->|No| L[geographical_*]

    style A fill:#2993B5,stroke:#1d6f8a,color:#fff
    style B fill:#FFBF00,stroke:#cc9900,color:#1e293b
    style C fill:#FFBF00,stroke:#cc9900,color:#1e293b
    style D fill:#FFBF00,stroke:#cc9900,color:#1e293b
    style E fill:#0fad6b,stroke:#076b3f,color:#fff
    style F fill:#0fad6b,stroke:#076b3f,color:#fff
    style G fill:#0fad6b,stroke:#076b3f,color:#fff
    style H fill:#FFBF00,stroke:#cc9900,color:#1e293b
    style I fill:#0fad6b,stroke:#076b3f,color:#fff
    style L fill:#0fad6b,stroke:#076b3f,color:#fff
```

See [Partitioning](partitioning/index.md) for detailed documentation.

## Aggregation Profiles

Aggregation uses {py:class}`~npap.AggregationProfile` that define how the network is reduced.

### Aggregation Modes

Predefined {py:class}`~npap.AggregationMode` for common use cases:

| Mode | Topology | Node Properties | Edge Properties                    |
|------|----------|-----------------|------------------------------------|
| `SIMPLE` | simple | sum all | sum all                            |
| `GEOGRAPHICAL` | simple | avg coords, sum loads | sum capacity, equivalent reactance |
| `DC_PTDF` | electrical | PTDF reduction | PTDF-derived reactance values      |
| `DC_KRON` | electrical | Kron reduction | Kron reduction                    |
| `CONSERVATION` | electrical | Equivalent reactance + transformer preservation | Preserves transformer count & impedance |
| `CUSTOM` | user-defined | user-defined | user-defined                       |

```python
from npap import AggregationMode

# Use a predefined mode
aggregated = manager.aggregate(mode=AggregationMode.GEOGRAPHICAL)
```

### PTDF Reduction Mode

`AggregationMode.DC_PTDF` wires the `"ptdf_reduction"` physical strategy into an
electrical topology, so the reduced graph stores PTDF-consistent reactances for
each aggregated edge and exposes a `reduced_ptdf` matrix in `aggregated.graph`.
This mode keeps `x` coupled to the physical reduction step instead of applying
statistical averaging.

### Kron Reduction Mode

`AggregationMode.DC_KRON` applies the `"kron_reduction"` strategy to an
electrical topology. The strategy eliminates nodes that belong to each cluster
and keeps only the representative nodes, so aggregated reactances follow a
Kron-reduced susceptance matrix. The resulting Laplacian is stored in
`aggregated.graph["kron_reduced_laplacian"]` for debugging or downstream DC
studies.

### Custom Aggregation Profile

For full control over the aggregation step, create an {py:class}`~npap.AggregationProfile`:

```python
from npap import AggregationProfile

profile = AggregationProfile(
    topology_strategy="simple",
    node_properties={
        "lat": "average",
        "lon": "average",
        "load": "sum",
    },
    edge_properties={
        "x": "equivalent_reactance",
        "p_max": "sum",
    },
    default_node_strategy="sum",
    default_edge_strategy="average"
)

aggregated = manager.aggregate(profile=profile)
```

### Property Aggregation Strategies

**Node properties:**

| Strategy | Formula | Use Case |
|----------|---------|----------|
| `sum` | $\sum x_i$ | Loads, generation |
| `average` | $\frac{1}{n}\sum x_i$ | Coordinates, voltage |
| `first` | $x_1$ | Names, categories |

**Edge properties:**

| Strategy | Formula | Use Case          |
|----------|---------|-------------------|
| `sum` | $\sum x_i$ | Capacity    |
| `average` | $\frac{1}{n}\sum x_i$ | Length            |
| `first` | $x_1$ | Type, category    |
| `equivalent_reactance` | $\frac{1}{\sum \frac{1}{x_i}}$ | Parallel impedances |

See [Aggregation](aggregation.md) for detailed documentation.

### Transformer Conservation Mode

`AggregationMode.CONSERVATION` wires the new `"transformer_conservation"` physical strategy into an electrical topology, so transformer reactance (`x`) and resistance (`r`) are calculated using parallel impedance rules before statistical aggregation steps run. This mode is the foundation for the voltage-aware workflows on this page.

## Key Classes

### Main Entry Point

| Class | Purpose |
|-------|---------|
| {py:class}`~npap.PartitionAggregatorManager` | Main orchestrator for all operations |

### Data Structures

| Class | Purpose |
|-------|---------|
| {py:class}`~npap.PartitionResult` | Output of partitioning (cluster assignments) |
| {py:class}`~npap.AggregationProfile` | Configuration for network aggregation |
| {py:class}`~npap.AggregationMode` | Predefined aggregation modes (enum) |

### Managers

| Class | Purpose |
|-------|---------|
| {py:class}`~npap.InputDataManager` | Handles data loading strategies |
| {py:class}`~npap.PartitioningManager` | Handles partitioning strategies |
| {py:class}`~npap.AggregationManager` | Handles aggregation strategies |

### Enums

| Enum | Values |
|------|--------|
| {py:class}`~npap.EdgeType` | `LINE`, `TRAFO`, `DC_LINK` |
| {py:class}`~npap.AggregationMode` | `SIMPLE`, `GEOGRAPHICAL`, `DC_PTDF`, `DC_KRON`, `CUSTOM`, `CONSERVATION` |

### Exceptions

| Exception | When Raised |
|-----------|-------------|
| {py:class}`~npap.NPAPError` | Base exception for all NPAP errors |
| {py:class}`~npap.DataLoadingError` | Problems loading input data |
| {py:class}`~npap.PartitioningError` | Partitioning algorithm failures |
| {py:class}`~npap.AggregationError` | Aggregation failures |
| {py:class}`~npap.ValidationError` | Input validation failures |
| {py:class}`~npap.GraphCompatibilityError` | Partition/graph mismatch |

## Registering Custom Strategies

You can extend NPAP by registering custom strategies:

```python
# Custom data loader
manager.input_manager.register_strategy("my_loader", MyLoaderStrategy())

# Custom partitioning
manager.partitioning_manager.register_strategy("my_partitioner", MyPartitioningStrategy())

# Custom node/edge aggregation
manager.aggregation_manager.register_node_strategy("my_node_agg", MyNodeStrategy())
manager.aggregation_manager.register_edge_strategy("my_edge_agg", MyEdgeStrategy())
```

See [Extending NPAP](extending.md) for detailed instructions on creating custom strategies.
