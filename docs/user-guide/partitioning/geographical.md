# Geographical Partitioning

Clusters nodes based on geographic coordinates using various clustering algorithms.

## Required Attributes

- **Nodes**: `lat` (latitude), `lon` (longitude)

## Available Strategies

| Strategy | Algorithm | Metric | DC-Island Aware |
|----------|-----------|--------|-----------------|
| `geographical_kmeans` | K-Means | Euclidean | No |
| `geographical_kmedoids_euclidean` | K-Medoids | Euclidean | Yes |
| `geographical_kmedoids_haversine` | K-Medoids | Haversine | Yes |
| `geographical_dbscan_euclidean` | DBSCAN | Euclidean | Yes |
| `geographical_dbscan_haversine` | DBSCAN | Haversine | Yes |
| `geographical_hierarchical` | Agglomerative | Euclidean | Yes* |
| `geographical_hdbscan_euclidean` | HDBSCAN | Euclidean | Yes |
| `geographical_hdbscan_haversine` | HDBSCAN | Haversine | Yes |

*Ward linkage does not support DC-island awareness

## K-Means Clustering

Fast clustering on raw coordinates. Best for quick partitioning when DC-island boundaries don't matter.

```python
partition = manager.partition(
    "geographical_kmeans",
    n_clusters=10,
    random_state=42,
    max_iter=300,
    n_init=10
)
```

**Parameters:**
- `n_clusters`: Number of clusters (required)
- `random_state`: Random seed for reproducibility
- `max_iter`: Maximum iterations
- `n_init`: Number of initializations

```{warning}
K-Means does **not** support DC-island awareness. If your network has DC links, consider using K-Medoids instead.
```

## K-Medoids Clustering

More robust clustering using precomputed distance matrices. Supports DC-island constraints.

```python
# Euclidean distance (for projected coordinates)
partition = manager.partition(
    "geographical_kmedoids_euclidean",
    n_clusters=10
)

# Haversine distance (for lat/lon on Earth's surface)
partition = manager.partition(
    "geographical_kmedoids_haversine",
    n_clusters=10
)
```
**Parameters:**
- `n_clusters`: Number of clusters (required)

## DBSCAN Clustering

Density-based clustering that automatically determines the number of clusters.

```python
partition = manager.partition(
    "geographical_dbscan_euclidean",
    eps=0.5,          # Maximum distance between points in a cluster
    min_samples=5     # Minimum points to form a cluster
)

print(f"Found {partition.n_clusters} clusters")
```

**Parameters:**
- `eps`: Maximum distance between two samples in the same neighborhood
- `min_samples`: Minimum number of samples in a neighborhood for a point to be a core point

```{note}
DBSCAN may classify some nodes as noise (cluster -1). These are typically isolated nodes.
```

## Hierarchical Clustering

Agglomerative clustering with different linkage methods.

```python
partition = manager.partition(
    "geographical_hierarchical",
    n_clusters=10,
    linkage="complete"  # 'complete', 'average', 'single'
)
```

**Parameters:**
- `n_clusters`: Number of clusters
- `linkage`: Linkage criterion
  - `complete`: Maximum distance between clusters
  - `average`: Average distance between clusters
  - `single`: Minimum distance between clusters

```{warning}
Ward linkage (`ward`) is **not supported** when DC-island awareness is needed, as it requires Euclidean distances and cannot use precomputed distance matrices.
```

## HDBSCAN Clustering

Hierarchical density-based clustering for robust automatic cluster detection.

```python
partition = manager.partition(
    "geographical_hdbscan_haversine",
    min_cluster_size=5
)
```

**Parameters:**
- `min_cluster_size`: Minimum size of clusters
