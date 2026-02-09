# Extending NPAP

NPAP's strategy pattern architecture makes it easy to add custom strategies. This guide shows how to contribute new data loaders, partitioning algorithms, and aggregation strategies.

## Architecture Overview

```{mermaid}
flowchart TB
    subgraph Interfaces
        DLS[DataLoadingStrategy]
        PS[PartitioningStrategy]
        TS[TopologyStrategy]
        PAS[PhysicalAggregationStrategy]
        NPS[NodePropertyStrategy]
        EPS[EdgePropertyStrategy]
    end

    subgraph Your Code
        CL[Custom Loader]
        CP[Custom Partitioner]
        CA[Custom Aggregator]
    end

    CL -->|inherits| DLS
    CP -->|inherits| PS
    CA -->|inherits| NPS

    style DLS fill:#2993B5,stroke:#1d6f8a,color:#fff
    style PS fill:#2993B5,stroke:#1d6f8a,color:#fff
    style TS fill:#2993B5,stroke:#1d6f8a,color:#fff
    style PAS fill:#2993B5,stroke:#1d6f8a,color:#fff
    style NPS fill:#2993B5,stroke:#1d6f8a,color:#fff
    style EPS fill:#2993B5,stroke:#1d6f8a,color:#fff
    style CL fill:#0fad6b,stroke:#076b3f,color:#fff
    style CP fill:#0fad6b,stroke:#076b3f,color:#fff
    style CA fill:#0fad6b,stroke:#076b3f,color:#fff
```

All strategies inherit from abstract base classes in `npap.interfaces`.

## Adding a Data Loading Strategy

### Step 1: Create the Strategy Class

```python
# npap/input/my_loader.py
from npap.interfaces import DataLoadingStrategy
import networkx as nx

class MyFormatStrategy(DataLoadingStrategy):
    """Load network from my custom format."""

    def validate_inputs(self, **kwargs) -> bool:
        """Validate input parameters."""
        return "file_path" in kwargs

    def load(self, **kwargs) -> nx.DiGraph:
        """Load and return the graph."""
        file_path = kwargs["file_path"]
        # ... loading logic ...
        return G
```

### Step 2: Register the Strategy

In `npap/managers.py`, add registration in `InputDataManager.__init__`:

```python
self.register_strategy("my_format", MyFormatStrategy())
```

### Step 3: Add Tests

```python
# test/test_input.py
def test_my_format_loader():
    manager = PartitionAggregatorManager()
    graph = manager.load_data("my_format", file_path="test_data.xyz")
    assert graph.number_of_nodes() > 0
```

### Step 4: Update Documentation

Add documentation in `docs/user-guide/data-loading.md`.

## Adding a Partitioning Strategy

### Step 1: Create the Strategy Class

```python
# npap/partitioning/my_partitioning.py
from npap.interfaces import PartitioningStrategy
import networkx as nx

class MyPartitioning(PartitioningStrategy):
    """Partition using my algorithm."""

    @property
    def required_attributes(self) -> dict[str, list[str]]:
        """Required node and edge attributes."""
        return {
            "nodes": ["my_attribute"],
            "edges": []
        }

    def partition(
        self,
        graph: nx.DiGraph,
        n_clusters: int = 10,
        **kwargs
    ) -> dict[int, list]:
        """Partition the graph.

        Parameters
        ----------
        graph : nx.DiGraph
            Network to partition.
        n_clusters : int
            Number of clusters.

        Returns
        -------
        dict[int, list]
            Mapping of cluster_id -> list of node IDs.
        """
        # ... partitioning logic ...
        return clusters
```

### Step 2: Register the Strategy

In `npap/managers.py`, add registration in `PartitioningManager.__init__`:

```python
self.register_strategy("my_algorithm", MyPartitioning())
```

### Step 3: Add Tests

```python
# test/test_partitioning.py
class TestMyPartitioning:
    def test_basic_partition(self):
        G = create_test_graph()
        strategy = MyPartitioning()
        result = strategy.partition(G, n_clusters=3)

        assert len(result) == 3
        all_nodes = set()
        for nodes in result.values():
            all_nodes.update(nodes)
        assert all_nodes == set(G.nodes())

    def test_missing_attributes(self):
        G = nx.DiGraph()
        G.add_node(1)  # Missing required attribute
        strategy = MyPartitioning()

        with pytest.raises(ValidationError):
            strategy.partition(G, n_clusters=1)
```

### Step 4: Update Documentation

Add documentation in `docs/user-guide/partitioning/`.

## Adding an Aggregation Strategy

### Node Property Strategy

```python
# npap/aggregation/my_aggregation.py
from npap.interfaces import NodePropertyStrategy
import networkx as nx

class MedianNodeStrategy(NodePropertyStrategy):
    """Aggregate using median value."""

    def aggregate_property(
        self,
        graph: nx.DiGraph,
        nodes: list,
        property_name: str
    ):
        """Compute median across nodes."""
        import numpy as np

        values = [
            graph.nodes[n][property_name]
            for n in nodes
            if property_name in graph.nodes[n]
        ]
        return float(np.median(values)) if values else 0
```

### Edge Property Strategy

```python
from npap.interfaces import EdgePropertyStrategy
from typing import Any

class MaxEdgeStrategy(EdgePropertyStrategy):
    """Aggregate using maximum value."""

    def aggregate_property(
        self,
        original_edges: list[dict[str, Any]],
        property_name: str
    ):
        """Return maximum value across edges."""
        values = [
            edge[property_name]
            for edge in original_edges
            if property_name in edge
        ]
        return max(values) if values else 0
```

### Registration

```python
# In AggregationManager.__init__
self.register_node_strategy("median", MedianNodeStrategy())
self.register_edge_strategy("max", MaxEdgeStrategy())
```

## Best Practices

### 1. Follow the Interface Contract

Always implement all abstract methods:

```python
class MyStrategy(PartitioningStrategy):
    @property
    def required_attributes(self):
        return {"nodes": [], "edges": []}  # Must implement

    def partition(self, graph, **kwargs):
        pass  # Must implement
```

### 2. Handle Edge Cases

```python
def partition(self, graph, n_clusters=10, **kwargs):
    # Handle empty graph
    if graph.number_of_nodes() == 0:
        return {}

    # Handle n_clusters > nodes
    n_clusters = min(n_clusters, graph.number_of_nodes())

    # Proceed with partitioning
    ...
```

### 3. Use Type Hints

```python
from typing import Any
import networkx as nx

def partition(
    self,
    graph: nx.DiGraph,
    n_clusters: int = 10,
    **kwargs: Any
) -> dict[int, list[Any]]:
    ...
```

### 4. Write Comprehensive Docstrings

```python
class MyPartitioning(PartitioningStrategy):
    """Short description of the strategy.

    This strategy partitions networks based on [algorithm description].

    Parameters
    ----------
    param1 : type
        Description.

    Attributes
    ----------
    required_attributes : dict
        Requires 'lat', 'lon' on nodes.

    Examples
    --------
    >>> strategy = MyPartitioning()
    >>> result = strategy.partition(graph, n_clusters=5)
    """
```

### 5. Raise Appropriate Exceptions

```python
from npap import PartitioningError, ValidationError

class MyPartitioning(PartitioningStrategy):
    def partition(self, graph, n_clusters=10, **kwargs):
        if n_clusters < 1:
            raise ValidationError(
                "n_clusters must be positive",
                strategy="my_partitioning"
            )

        try:
            # Algorithm logic
            ...
        except Exception as e:
            raise PartitioningError(
                f"Partitioning failed: {e}",
                strategy="my_partitioning"
            )
```

## Contribution Checklist

Before submitting your new strategy:

- [ ] Strategy inherits from appropriate base class
- [ ] All abstract methods implemented
- [ ] Type hints added
- [ ] NumPy-style docstrings written
- [ ] Unit tests added and passing
- [ ] Strategy registered in manager
- [ ] Documentation updated
- [ ] Code passes `ruff check .` and `ruff format .`

## Next Steps

- [Pull Requests](pull-requests.md) - How to submit your contribution
- [User Guide: Extending NPAP](../user-guide/extending.md) - End-user documentation for custom strategies
