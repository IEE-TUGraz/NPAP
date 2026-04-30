# Locational Marginal Price Partitioning

Clusters nodes based on Locational Marginal Prices (LMP) or price profiles. The implementation of this strategy is based on the findings in the
paper "Congestion-Sensitive Grid Aggregation for DC Optimal Power Flow" by B. Stöckl et al. ([2025](https://doi.org/10.1109/PowerTech59965.2025.11180585)).

## Required Attributes

- **Nodes**: `lmp` (can be a scalar value or a time-series vector/array)

## Available Strategies

| Strategy                     | Algorithm    | Description                                            | AC-Island Aware |
|------------------------------|--------------|--------------------------------------------------------|-----------------|
| `lmp_hierarchical_connected` | Hierarchical | Agglomerative clustering with connectivity constraints | Yes             |

## Overview

LMP-based partitioning is specifically designed for market-based network analysis. It identifies regions with similar price behaviors, which is often
used to define bidding zones or identify structural congestion in a power system.

The strategy calculates the Euclidean distance between LMP profiles (vectors of prices over time) to determine nodal similarity.

### Connectivity Constraint

By default, the `lmp_hierarchical_connected` strategy uses the graph's topology as a constraint. This ensures that resulting clusters are **spatially
contiguous**—nodes in the same cluster must be physically connected in the network. This prevents "scattered" clusters that would be unrealistic for
geographical price zones.

### Mathematical Background

Locational Marginal Prices (LMPs) are often used in power systems to reflect the cost of supplying the next increment of load at a specific location,
considering generation costs and network constraints.

In an optimal power flow (OPF) with a bus angle formulation, they can be mathematically defined as the dual variables $\lambda$ associated with the
nodal balance constraints.

In an optimal power flow implementation based on PTDF power flows, the LMPs can be expressed as:
$$\mathrm{LMP} = \lambda_{slack} + \mathrm{PTDF}^\top (\bar\varphi - \underline\varphi)$$
where:

- $ \lambda_{slack} $ is the LMP at the slack bus (reference node).
- $ \mathrm{PTDF} $ is the Power Transfer Distribution Factor matrix, which describes how power injections at nodes affect line flows.
- $ \bar\varphi $ and $ \underline\varphi $ are the dual variables associated with the upper and lower line flow constraints, respectively.

### AC-Island Awareness

If the graph contains `ac_island` data (automatically set by `va_loader`), the strategy respects AC island boundaries by assigning infinite distance
between nodes in different islands. This ensures that price zones do not span across asynchronous regions.

## Usage

```python
import npap

manager = npap.PartitionAggregatorManager()
manager.load_data("va_loader", node_file="nodes.csv", line_file="lines.csv")

# Partition based on LMP profiles
partition = manager.partition(
    strategy="lmp_hierarchical_connected",
    n_clusters=15,
    lmp_attr="lmp",  # Node attribute containing prices
    use_connectivity=True  # Ensure contiguous clusters
)
```

## Configuration

The LMP strategy uses {py:class}`~npap.partitioning.locational_marginal_price.LMPConfig`:

```python
from npap.partitioning.locational_marginal_price import LMPConfig

config = LMPConfig(
    hierarchical_linkage="complete",
    infinite_distance=1e4,
    use_connectivity=True
)

partition = manager.partition(
    "lmp_hierarchical_connected",
    n_clusters=15,
    config=config
)
```

| Parameter              | Default      | Description                                                   |
|------------------------|--------------|---------------------------------------------------------------|
| `hierarchical_linkage` | `"complete"` | Linkage criterion for hierarchical clustering.                |
| `infinite_distance`    | `1e4`        | Value used to separate nodes in different AC islands.         |
| `use_connectivity`     | `True`       | Whether to use graph connectivity as a clustering constraint. |
