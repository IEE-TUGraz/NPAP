# Voltage-Aware Electrical Partitioning

Combines electrical distance (PTDF) with voltage level constraints.

## Required Attributes

- **Nodes**: `voltage`, `dc_island`
- **Edges**: `x` (reactance)

## Available Strategies

| Strategy | Algorithm |
|----------|-----------|
| `va_electrical_kmedoids` | K-Medoids |
| `va_electrical_hierarchical` | Agglomerative |

## Basic Usage

```python
partition = manager.partition(
    "va_electrical_kmedoids",
    n_clusters=20
)
```

## How It Works

1. Computes PTDF for the full AC network (lines + transformers)
2. Calculates electrical distances from PTDF
3. Sets infinite distance between nodes at different voltage levels
4. Clusters respecting both electrical behavior and voltage boundaries

```{mermaid}
flowchart TB
    subgraph V380[380 kV Level]
        A[Cluster 0] --- B[Cluster 1]
    end

    subgraph V220[220 kV Level]
        C[Cluster 2] --- D[Cluster 3]
    end

    V380 -->|Transformer| V220

    style A fill:#2993B5,stroke:#1d6f8a,color:#fff
    style B fill:#2993B5,stroke:#1d6f8a,color:#fff
    style C fill:#0fad6b,stroke:#076b3f,color:#fff
    style D fill:#0fad6b,stroke:#076b3f,color:#fff
```
