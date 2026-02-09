# Quick Start

This guide demonstrates a complete NPAP workflow using a voltage-aware (VA) power network. By the end, you'll understand how to load data, partition a network, aggregate it, and visualize the results.

## The NPAP Workflow

```{mermaid}
flowchart LR
    A[Load Data] --> B[Aggregate Parallel Edges]
    B --> C[Group by Voltage]
    C --> D[Partition]
    D --> E[Aggregate]
    E --> F[Visualize]

    style A fill:#2993B5,stroke:#1d6f8a,color:#fff
    style B fill:#0fad6b,stroke:#076b3f,stroke-dasharray: 5 5,color:#fff
    style C fill:#0fad6b,stroke:#076b3f,stroke-dasharray: 5 5,color:#fff
    style D fill:#2993B5,stroke:#1d6f8a,color:#fff
    style E fill:#2993B5,stroke:#1d6f8a,color:#fff
    style F fill:#2993B5,stroke:#1d6f8a,color:#fff
```

*Dashed boxes indicate optional steps*

## Complete Example: Voltage-Aware Network

This example shows the full workflow for a multi-voltage power system with transmission lines, transformers, and DC links.

```python
import npap
from npap import AggregationMode
import networkx as nx

# =============================================================================
# Step 1: Initialize the Manager
# =============================================================================

manager = npap.PartitionAggregatorManager()

# =============================================================================
# Step 2: Load Voltage-Aware Data
# =============================================================================

# The VA loader handles multi-voltage networks with transformers and DC links
graph = manager.load_data(
    strategy="va_loader",
    node_file="buses.csv",           # Bus/substation data
    line_file="lines.csv",           # AC transmission lines
    transformer_file="transformers.csv",  # Transformers between voltage levels
    converter_file="converters.csv", # AC/DC converters
    link_file="dc_links.csv"         # HVDC links
)

print(f"Loaded network: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

# =============================================================================
# Step 3: Aggregate Parallel Edges (if present)
# =============================================================================

# Power networks often have parallel transmission lines
if isinstance(graph, nx.MultiDiGraph):
    manager.aggregate_parallel_edges(
        edge_properties={
            "x": "equivalent_reactance",  # Parallel reactance formula: 1/(1/x1 + 1/x2)
            "r": "equivalent_reactance",  # Same for resistance
            "p_max": "sum",               # Sum the capacities
        },
        default_strategy="average",
        warn_on_defaults=False
    )
    print(f"After parallel edge aggregation: {manager.get_current_graph().number_of_edges()} edges")

# =============================================================================
# Step 4: Partition the Network
# =============================================================================

# VA-geographical partitioning respects voltage levels and DC islands
partition_result = manager.partition(
    strategy="va_geographical_kmedoids_haversine",
    n_clusters=50
)

print(f"Created {partition_result.n_clusters} clusters")
print(f"Strategy used: {partition_result.strategy_name}")

# Inspect cluster assignments
for cluster_id, nodes in list(partition_result.mapping.items())[:3]:
    print(f"  Cluster {cluster_id}: {len(nodes)} nodes")

# =============================================================================
# Step 5: Visualize Partitioned Network
# =============================================================================

# Plot the network colored by cluster assignment
manager.plot_network(
    style="clustered",
    title="Partitioned VA Network"
)

# =============================================================================
# Step 6: Aggregate the Network
# =============================================================================

# Aggregate using the GEOGRAPHICAL mode (averages coordinates, sums loads)
aggregated_graph = manager.aggregate(mode=AggregationMode.GEOGRAPHICAL)

print(f"Aggregated network: {aggregated_graph.number_of_nodes()} nodes, "
      f"{aggregated_graph.number_of_edges()} edges")

# =============================================================================
# Step 7: Visualize Aggregated Network
# =============================================================================

manager.plot_network(
    graph=aggregated_graph,
    style="voltage_aware",
    title="Aggregated Network"
)
```

## Understanding the Data Files

### buses.csv (Nodes)

```text
id,lat,lon,voltage,p_load,q_load
bus_001,47.0667,15.4333,380,100.0,30.0
bus_002,47.1234,15.5678,380,150.0,45.0
bus_003,48.2089,16.3726,220,200.0,60.0
```

### lines.csv (AC Transmission Lines)

```text
from,to,x,r,p_max,length
bus_001,bus_002,0.0123,0.0045,500,125.5
bus_002,bus_004,0.0234,0.0089,400,98.2
```

### transformers.csv

```text
from,to,x,r,s_nom,tap_ratio
bus_002,bus_003,0.05,0.01,400,1.0
```

### converters.csv (Optional)

```text
id,bus,p_max
conv_1,bus_010,1000
conv_2,bus_020,1000
```

### dc_links.csv (Optional)

```text
from,to,p_max,converter_from,converter_to
conv_1,conv_2,800,conv_1,conv_2
```

## Simple Example: NetworkX Graph

For simpler use cases without CSV files:

```python
import npap
import networkx as nx

# Create a graph programmatically
G = nx.DiGraph()

# Add nodes with geographic coordinates
G.add_node("bus_1", lat=47.0, lon=15.0, load=100.0)
G.add_node("bus_2", lat=47.1, lon=15.1, load=150.0)
G.add_node("bus_3", lat=47.2, lon=15.0, load=200.0)
G.add_node("bus_4", lat=47.0, lon=15.2, load=120.0)

# Add edges with electrical properties
G.add_edge("bus_1", "bus_2", x=0.01, p_max=500)
G.add_edge("bus_2", "bus_3", x=0.02, p_max=400)
G.add_edge("bus_3", "bus_4", x=0.015, p_max=450)
G.add_edge("bus_4", "bus_1", x=0.018, p_max=300)

# Initialize manager and load
manager = npap.PartitionAggregatorManager()
manager.load_data("networkx_direct", graph=G)

# Partition
partition = manager.partition("geographical_kmeans", n_clusters=2)

# Aggregate
aggregated = manager.aggregate(mode=npap.AggregationMode.GEOGRAPHICAL)

# Visualize
manager.plot_network(style="voltage_aware")
```

## Custom Aggregation Profile

For more control over how properties are aggregated:

```python
from npap import AggregationProfile

# Define custom aggregation rules
profile = AggregationProfile(
    topology_strategy="simple",
    node_properties={
        "lat": "average",           # Geographic center
        "lon": "average",
        "p_load": "sum",            # Total load
        "q_load": "sum",
        "voltage": "average",       # Average voltage
    },
    edge_properties={
        "x": "equivalent_reactance",  # Parallel impedance formula
        "r": "equivalent_reactance",
        "p_max": "sum",               # Total capacity
        "length": "sum",              # Total length
    },
    default_node_strategy="sum",
    default_edge_strategy="average"
)

aggregated = manager.aggregate(profile=profile)
```

## Key Points

1. **Manager Pattern**: All operations go through {py:class}`~npap.PartitionAggregatorManager`
2. **Strategy Selection**: Choose strategies based on your data and requirements
3. **Voltage Awareness**: Use `va_loader` and `va_geographical_*` or `va_electrical_*` strategies for multi-voltage networks
4. **DC Island Handling**: NPAP automatically detects and respects DC island boundaries (if `dc_island` attribute is present)
5. **Parallel Edges**: Aggregate them before partitioning

## Next Steps

- **[Available Strategies](available-strategies.md)** - Explore all partitioning and aggregation options
- **[Data Loading](data-loading.md)** - Detailed guide on loading different data formats
- **[Partitioning](partitioning/index.md)** - Deep dive into clustering algorithms
- **[Aggregation](aggregation.md)** - Learn about the three-tier aggregation system
