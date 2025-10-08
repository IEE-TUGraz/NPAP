import functools
import hashlib
import json

import networkx as nx


def validate_required_attributes(func):
    """
    Optimized decorator for attribute validation
    - Early exit on first missing attribute
    - Minimal memory overhead
    - Fast attribute checking
    """

    @functools.wraps(func)
    def wrapper(self, graph: nx.Graph, n_clusters: int, **kwargs):
        required = self.required_attributes

        # Fast early-exit validation for nodes
        if required.get('nodes'):
            required_node_attrs = set(required['nodes'])  # Set lookup is O(1)
            for node_id in graph.nodes():
                node_attrs = graph.nodes[node_id]
                # Use set operations - much faster than individual checks
                missing_attrs = required_node_attrs - node_attrs.keys()
                if missing_attrs:
                    from .exceptions import ValidationError
                    raise ValidationError(
                        f"Node {node_id} missing required attributes",
                        missing_attributes={'nodes': list(missing_attrs)},
                        strategy=self.__class__.__name__
                    )

        # Fast early-exit validation for edges
        if required.get('edges'):
            required_edge_attrs = set(required['edges'])
            for edge in graph.edges():
                edge_attrs = graph.edges[edge]
                missing_attrs = required_edge_attrs - edge_attrs.keys()
                if missing_attrs:
                    from .exceptions import ValidationError
                    raise ValidationError(
                        f"Edge {edge} missing required attributes",
                        missing_attributes={'edges': list(missing_attrs)},
                        strategy=self.__class__.__name__
                    )

        # Only call original function if validation passes
        return func(self, graph, n_clusters, **kwargs)

    return wrapper


def compute_graph_hash(graph: nx.Graph) -> str:
    """
    Compute a hash of the graph structure for validation

    This creates a fingerprint of the graph based on:
    - Number of nodes
    - Number of edges
    - Node IDs (sorted)

    Used to validate that a partition matches the graph it was created from.

    Args:
        graph: NetworkX graph

    Returns:
        Hash string
    """
    # Create a deterministic representation
    graph_data = {
        'n_nodes': len(list(graph.nodes())),
        'n_edges': len(graph.edges()),
        'nodes': sorted([str(n) for n in graph.nodes()]),
    }

    # Convert to JSON and hash
    json_str = json.dumps(graph_data, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]
