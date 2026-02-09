# Extending NPAP

NPAP's strategy pattern architecture makes it easy to add custom data loaders, partitioning algorithms, and aggregation strategies. This guide shows how to extend each component.

```{tip}
Want to contribute your custom strategy to NPAP? See the [Contributing to NPAP](../contributing.md) guide for details on forking the repository, development setup, and submitting pull requests.
```

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
    CA -->|inherits| EPS

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

## Custom Data Loading Strategy

### Interface

```python
from abc import ABC, abstractmethod
import networkx as nx

class DataLoadingStrategy(ABC):
    @abstractmethod
    def load(self, **kwargs) -> nx.DiGraph | nx.MultiDiGraph:
        """Load data and return a NetworkX directed graph."""
        pass

    @abstractmethod
    def validate_inputs(self, **kwargs) -> bool:
        """Validate input parameters before loading."""
        pass
```

### Example: JSON Loader

```python
from npap.interfaces import DataLoadingStrategy
import networkx as nx
import json

class JSONFileStrategy(DataLoadingStrategy):
    """Load network from a JSON file."""

    def validate_inputs(self, **kwargs) -> bool:
        """Validate that file path is provided and file exists."""
        if "file_path" not in kwargs:
            return False

        import os
        return os.path.exists(kwargs["file_path"])

    def load(self, **kwargs) -> nx.DiGraph:
        """Load graph from JSON file.

        Parameters
        ----------
        file_path : str
            Path to JSON file with 'nodes' and 'edges' keys.

        Returns
        -------
        nx.DiGraph
            Loaded network graph.
        """
        file_path = kwargs["file_path"]

        with open(file_path, "r") as f:
            data = json.load(f)

        G = nx.DiGraph()

        # Add nodes with attributes
        for node in data["nodes"]:
            node_id = node.pop("id")
            G.add_node(node_id, **node)

        # Add edges with attributes
        for edge in data["edges"]:
            source = edge.pop("from")
            target = edge.pop("to")
            G.add_edge(source, target, **edge)

        return G
```

### Registering the Strategy

```python
import npap

manager = npap.PartitionAggregatorManager()

# Register custom strategy
manager.input_manager.register_strategy("json_file", JSONFileStrategy())

# Use it
graph = manager.load_data("json_file", file_path="network.json")
```

## Custom Partitioning Strategy

### Interface

```python
from abc import ABC, abstractmethod
import networkx as nx

class PartitioningStrategy(ABC):
    @property
    @abstractmethod
    def required_attributes(self) -> dict[str, list[str]]:
        """Required node and edge attributes.

        Returns
        -------
        dict
            {'nodes': ['attr1', 'attr2'], 'edges': ['attr3']}
        """
        pass

    @abstractmethod
    def partition(
        self,
        graph: nx.DiGraph,
        **kwargs
    ) -> dict[int, list]:
        """Partition the graph into clusters.

        Parameters
        ----------
        graph : nx.DiGraph
            The network to partition.
        **kwargs
            Algorithm-specific parameters.

        Returns
        -------
        dict[int, list]
            Mapping of cluster_id -> list of node IDs.
        """
        pass
```

### Example: Degree-Based Partitioning

```python
from npap.interfaces import PartitioningStrategy
import networkx as nx

class DegreePartitioning(PartitioningStrategy):
    """Partition nodes based on their degree."""

    @property
    def required_attributes(self) -> dict[str, list[str]]:
        # No special attributes required
        return {"nodes": [], "edges": []}

    def partition(
        self,
        graph: nx.DiGraph,
        n_clusters: int = 3,
        **kwargs
    ) -> dict[int, list]:
        """Partition by node degree into n_clusters groups.

        Parameters
        ----------
        graph : nx.DiGraph
            Input graph.
        n_clusters : int
            Number of degree-based clusters.

        Returns
        -------
        dict[int, list]
            Cluster mapping.
        """
        import numpy as np

        # Get degrees
        degrees = dict(graph.degree())
        nodes = list(degrees.keys())
        degree_values = np.array(list(degrees.values()))

        # Create clusters based on degree percentiles
        percentiles = np.linspace(0, 100, n_clusters + 1)
        thresholds = np.percentile(degree_values, percentiles)

        # Assign nodes to clusters
        clusters = {i: [] for i in range(n_clusters)}
        for node, degree in degrees.items():
            for i in range(n_clusters):
                if thresholds[i] <= degree <= thresholds[i + 1]:
                    clusters[i].append(node)
                    break

        return clusters
```

### Registering the Strategy

```python
manager = npap.PartitionAggregatorManager()
manager.partitioning_manager.register_strategy("degree", DegreePartitioning())

# Use it
partition = manager.partition("degree", n_clusters=5)
```

### Using the Validation Decorator

NPAP provides a decorator for automatic attribute validation:

```python
from npap.interfaces import PartitioningStrategy
from npap.utils import validate_required_attributes

class MyPartitioning(PartitioningStrategy):
    @property
    def required_attributes(self) -> dict[str, list[str]]:
        return {"nodes": ["weight"], "edges": []}

    @validate_required_attributes
    def partition(self, graph, **kwargs):
        # Validation happens automatically before this runs
        # If 'weight' is missing, raises ValidationError
        ...
```

## Custom Aggregation Strategies

### Node Property Strategy

```python
from npap.interfaces import NodePropertyStrategy
import networkx as nx

class MedianNodeStrategy(NodePropertyStrategy):
    """Aggregate node properties using median."""

    def aggregate_property(
        self,
        graph: nx.DiGraph,
        nodes: list,
        property_name: str
    ):
        """Compute median of property across nodes.

        Parameters
        ----------
        graph : nx.DiGraph
            The original graph.
        nodes : list
            Nodes in the cluster.
        property_name : str
            Name of property to aggregate.

        Returns
        -------
        float
            Median value.
        """
        import numpy as np

        values = [
            graph.nodes[n][property_name]
            for n in nodes
            if property_name in graph.nodes[n]
        ]

        if not values:
            return 0

        return float(np.median(values))
```

### Edge Property Strategy

```python
from npap.interfaces import EdgePropertyStrategy

class MaxEdgeStrategy(EdgePropertyStrategy):
    """Aggregate edge properties using maximum value."""

    def aggregate_property(
        self,
        original_edges: list[dict[str, Any]],
        property_name: str
    ):
        """Return maximum property value across edges.

        Parameters
        ----------
        original_edges : list[dict[str, Any]]
            List of edge attribute dictionaries.
        property_name : str
            Name of property to aggregate.

        Returns
        -------
        float
            Maximum value.
        """
        values = [
            edge[property_name]
            for edge in original_edges
            if property_name in edge
        ]

        if not values:
            return 0

        return max(values)
```

### Registering Aggregation Strategies

```python
manager = npap.PartitionAggregatorManager()

# Register node strategy
manager.aggregation_manager.register_node_strategy("median", MedianNodeStrategy())

# Register edge strategy
manager.aggregation_manager.register_edge_strategy("max", MaxEdgeStrategy())

# Use in profile
from npap import AggregationProfile

profile = AggregationProfile(
    node_properties={
        "load": "median"  # Use our custom strategy
    },
    edge_properties={
        "capacity": "max"  # Use our custom strategy
    }
)

aggregated = manager.aggregate(profile=profile)
```

## Custom Topology Strategy

For custom network reduction approaches:

```python
from npap.interfaces import TopologyStrategy
import networkx as nx

class FullyConnectedTopology(TopologyStrategy):
    """Create fully connected aggregated network."""

    @property
    def can_create_new_edges(self) -> bool:
        """This strategy creates edges that didn't exist."""
        return True

    def create_topology(
        self,
        graph: nx.DiGraph,
        partition_map: dict[int, list]
    ) -> nx.DiGraph:
        """Create fully connected topology.

        Parameters
        ----------
        graph : nx.DiGraph
            Original graph.
        partition_map : dict[int, list]
            Cluster assignments.

        Returns
        -------
        nx.DiGraph
            Fully connected aggregated topology.
        """
        G = nx.DiGraph()

        # Add cluster nodes
        clusters = list(partition_map.keys())
        G.add_nodes_from(clusters)

        # Add all possible edges
        for i in clusters:
            for j in clusters:
                if i != j:
                    G.add_edge(i, j)

        return G
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

### 4. Document Your Strategy

```python
class MyPartitioning(PartitioningStrategy):
    """Short description of the strategy.

    This strategy partitions networks based on [algorithm description].

    Parameters
    ----------
    param1 : type
        Description of parameter.

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

## Complete Example: LMP-Based Partitioning

Here's a complete example of a Locational Marginal Price (LMP) based partitioning strategy:

```python
from npap.interfaces import PartitioningStrategy
from npap.utils import validate_required_attributes, create_partition_map
import networkx as nx
import numpy as np

class LMPPartitioning(PartitioningStrategy):
    """Partition based on Locational Marginal Prices.

    Nodes with similar LMPs are grouped together, as they
    have similar economic value for power injection.

    Required Attributes
    -------------------
    nodes : lmp
        Locational marginal price at each node.
    """

    @property
    def required_attributes(self) -> dict[str, list[str]]:
        return {"nodes": ["lmp"], "edges": []}

    @validate_required_attributes
    def partition(
        self,
        graph: nx.DiGraph,
        n_clusters: int = 10,
        **kwargs
    ) -> dict[int, list]:
        """Partition by LMP using k-means.

        Parameters
        ----------
        graph : nx.DiGraph
            Network with 'lmp' node attribute.
        n_clusters : int
            Number of price zones.

        Returns
        -------
        dict[int, list]
            Cluster mapping.
        """
        from sklearn.cluster import KMeans

        nodes = list(graph.nodes())
        lmps = np.array([
            [graph.nodes[n]["lmp"]]
            for n in nodes
        ])

        # Cluster by LMP
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(lmps)

        return create_partition_map(nodes, labels)


# Usage
manager = npap.PartitionAggregatorManager()
manager.partitioning_manager.register_strategy("lmp", LMPPartitioning())

# Add LMP data to graph
for node in graph.nodes():
    graph.nodes[node]["lmp"] = compute_lmp(node)  # Your LMP calculation

manager.load_data("networkx_direct", graph=graph)
partition = manager.partition("lmp", n_clusters=10)
```

## Testing Custom Strategies

```python
import pytest
import networkx as nx
from my_strategies import MyPartitioning

class TestMyPartitioning:
    def test_basic_partition(self):
        # Create test graph
        G = nx.DiGraph()
        G.add_nodes_from(range(10))
        for i in range(9):
            G.add_edge(i, i + 1)

        strategy = MyPartitioning()
        result = strategy.partition(G, n_clusters=3)

        # Verify result structure
        assert isinstance(result, dict)
        assert len(result) == 3

        # Verify all nodes assigned
        all_nodes = set()
        for nodes in result.values():
            all_nodes.update(nodes)
        assert all_nodes == set(range(10))

    def test_empty_graph(self):
        G = nx.DiGraph()
        strategy = MyPartitioning()
        result = strategy.partition(G, n_clusters=3)
        assert result == {}

    def test_missing_attributes(self):
        G = nx.DiGraph()
        G.add_node(1)  # Missing required attributes

        strategy = MyPartitioning()
        with pytest.raises(ValidationError):
            strategy.partition(G, n_clusters=1)
```
