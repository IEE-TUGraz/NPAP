# Electrical Partitioning

Clusters nodes based on **electrical distance** computed from Power Transfer Distribution Factors (PTDF).

## Required Attributes

- **Nodes**: `ac_island` (AC island identifier)
- **Edges**: `x` (reactance)

## Mathematical Background

Electrical distance is based on how power flows through the network:

1. **Incidence Matrix (K)**: Describes network topology
2. **Susceptance Matrix (B)**: $\mathbf{B} = \mathbf{K}_\mathrm{sba}^\top \cdot \text{diag}(\mathbf{b}) \cdot \mathbf{K}_\mathrm{sba}$
3. **PTDF Matrix**: $\mathbf{PTDF} = \text{diag}(\mathrm{b}) \cdot \mathbf{K}_\mathrm{sba} \cdot \mathbf{B}^{-1}$
4. **Electrical Distance**: $d_{ij} = ||\mathrm{PTDF}_{:,i} - \mathrm{PTDF}_{:,j}||_2$

Nodes with similar PTDF columns have similar impact on power flows of lines.

## Available Strategies

| Strategy | Algorithm | Description                                                                |
|----------|-----------|----------------------------------------------------------------------------|
| `electrical_kmeans` | K-Means | Electrical clustering with arbitrary node as centroid                      |
| `electrical_kmedoids` | K-Medoids | Electrical clustering with existing node as centroid (e.g. Kron-Reduction) |

## Basic Usage

```python
partition = manager.partition(
    "electrical_kmeans",
    n_clusters=10
)
```

## AC-Island Handling

Electrical partitioning is always AC-island aware:

- PTDF is computed independently for each AC island
- Nodes in different islands have infinite distance
- Each island is clustered separately

```{mermaid}
flowchart TB
    subgraph Island0[AC Island 0]
        A[Cluster 0] --- B[Cluster 1]
    end

    subgraph Island1[AC Island 1]
        C[Cluster 2] --- D[Cluster 3]
    end

    Island0 -.->|DC Link| Island1

    style A fill:#2993B5,stroke:#1d6f8a,color:#fff
    style B fill:#2993B5,stroke:#1d6f8a,color:#fff
    style C fill:#0fad6b,stroke:#076b3f,color:#fff
    style D fill:#0fad6b,stroke:#076b3f,color:#fff
```

## Configuration

```python
from npap.partitioning import ElectricalDistanceConfig

# Custom configuration
config = ElectricalDistanceConfig(
    zero_reactance_replacement=1e-5,  # Replace zero reactance
    regularization_factor=1e-10,       # Matrix regularization
    infinite_distance=1e4              # Inter-island distance
)

partition = manager.partition(
    "electrical_kmeans",
    n_clusters=10,
    config=config
)
```
