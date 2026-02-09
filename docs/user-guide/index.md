# User Guide

```{toctree}
:hidden:
:maxdepth: 2
:caption: Getting Started

installation
quick-start
available-strategies
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Architecture

data-loading
partitioning/index
aggregation
visualization
extending
```

Welcome to the NPAP User Guide! 🚀

This guide covers everything you need to know to use NPAP for network partitioning and aggregation.

---

## Getting Started

New to NPAP? Start here to get up and running.

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Installation
:link: installation
:link-type: doc

Install NPAP from PyPI or source, including development dependencies.
:::

:::{grid-item-card} Quick Start
:link: quick-start
:link-type: doc

Complete workflow example using a voltage-aware power network.
:::

:::{grid-item-card} Available Strategies
:link: available-strategies
:link-type: doc

Overview of all strategies, key classes, and the strategy pattern architecture.
:::

::::

---

## Architecture

Deep dive into each component of the NPAP workflow.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Data Loading
:link: data-loading
:link-type: doc

Load networks from CSV files, NetworkX graphs, or voltage-aware power system formats.
:::

:::{grid-item-card} Partitioning
:link: partitioning/index
:link-type: doc

Geographical, electrical, and voltage-aware partitioning strategies.
:::

:::{grid-item-card} Aggregation
:link: aggregation
:link-type: doc

Three-tier aggregation system with topology, physical, and statistical strategies.
:::

:::{grid-item-card} Visualization
:link: visualization
:link-type: doc

Interactive Plotly maps with customizable styling.
:::

::::

---

## Data Flow Overview

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

## Working with Graphs

### Graph Types

NPAP works with [NetworkX](https://networkx.org/documentation/stable/reference/classes/index.html) directed graphs:

- **DiGraph**: Simple directed graph (one edge per node pair)
- **MultiDiGraph**: Directed graph with parallel edges

```python
import networkx as nx

# Simple directed graph
G = nx.DiGraph()

# Multi-edge directed graph (for parallel lines)
G = nx.MultiDiGraph()
```

### Required Attributes

Different strategies require specific node and edge attributes:

| Strategy Type | Node Attributes | Edge Attributes |
|---------------|-----------------|-----------------|
| Geographical | `lat`, `lon` | — |
| Electrical | `ac_island` | `x` (reactance) |
| Voltage-Aware | `lat`, `lon`, `voltage`, `ac_island` | `x`, `type` |

## Error Handling

NPAP provides a comprehensive exception hierarchy:

```python
from npap import (
    NPAPError,              # Base exception
    DataLoadingError,       # Input/loading issues
    PartitioningError,      # Partitioning failures
    AggregationError,       # Aggregation issues
    ValidationError,        # Input validation
    GraphCompatibilityError # Partition/graph mismatch
)

try:
    result = manager.partition("unknown_strategy", n_clusters=5)
except PartitioningError as e:
    print(f"Partitioning failed: {e}")
except NPAPError as e:
    print(f"NPAP error: {e}")
```

## Logging

Configure logging to monitor NPAP operations:

```python
from npap.logging import configure_logging, LogCategory
import logging

# Enable debug logging
configure_logging(level=logging.DEBUG)

# Or disable all logging
from npap.logging import disable_logging
disable_logging()
```

Log categories include: `INPUT`, `PARTITIONING`, `AGGREGATION`, `VISUALIZATION`, `VALIDATION`, `MANAGER`.
