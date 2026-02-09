# Voltage-Aware Geographical Partitioning

Combines geographic distance with voltage level and DC island constraints.

## Required Attributes

- **Nodes**: `lat`, `lon`, `voltage`, `dc_island`

## Available Strategies

| Strategy | Mode | Description |
|----------|------|-------------|
| `va_geographical_kmedoids_euclidean` | Standard | K-Medoids with Euclidean distance |
| `va_geographical_kmedoids_haversine` | Standard | K-Medoids with Haversine distance |
| `va_geographical_hierarchical` | Standard | Agglomerative clustering |
| `va_geographical_proportional_kmedoids_euclidean` | Proportional | Balanced clustering by group |
| `va_geographical_proportional_kmedoids_haversine` | Proportional | Balanced clustering by group |
| `va_geographical_proportional_hierarchical` | Proportional | Balanced hierarchical |

## Standard Mode

Nodes in different voltage levels or DC islands receive infinite distance:

```python
partition = manager.partition(
    "va_geographical_kmedoids_haversine",
    n_clusters=20
)
```

## Proportional Mode

Distributes clusters proportionally among voltage/island groups:

```python
# If network has:
# - 100 nodes at 380 kV
# - 50 nodes at 220 kV
# And n_clusters=15, approximately:
# - 10 clusters for 380 kV nodes
# - 5 clusters for 220 kV nodes

partition = manager.partition(
    "va_geographical_proportional_kmedoids_haversine",
    n_clusters=15
)
```

## Configuration

```python
from npap.partitioning import VAGeographicalConfig

config = VAGeographicalConfig(
    voltage_tolerance=1.0,        # kV tolerance for grouping
    infinite_distance=1e4,
    proportional_clustering=False,
    hierarchical_linkage="complete"
)

partition = manager.partition(
    "va_geographical_kmedoids_haversine",
    n_clusters=20,
    config=config
)
```
