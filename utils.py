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
    def wrapper(self, graph: nx.Graph, **kwargs):
        required = self.required_attributes

        # Fast early-exit validation for nodes
        if required.get('nodes'):
            required_node_attrs = set(required['nodes'])  # Set lookup is O(1)
            for node_id in graph.nodes():
                node_attrs = graph.nodes[node_id]
                # Use set operations - much faster than individual checks
                missing_attrs = required_node_attrs - node_attrs.keys()
                if missing_attrs:
                    from exceptions import ValidationError
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
                    from exceptions import ValidationError
                    raise ValidationError(
                        f"Edge {edge} missing required attributes",
                        missing_attributes={'edges': list(missing_attrs)},
                        strategy=self.__class__.__name__
                    )

        # Only call original function if validation passes
        return func(self, graph, **kwargs)

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


def validate_graph_compatibility(partition_result, current_graph_hash: str):
    """
    Validate that a partition result is compatible with the current graph

    Raises GraphCompatibilityError if hashes don't match

    Args:
        partition_result: PartitionResult object
        current_graph_hash: Hash of the current graph
    """
    if partition_result.original_graph_hash != current_graph_hash:
        from exceptions import GraphCompatibilityError
        raise GraphCompatibilityError(
            "Partition was created from a different graph. "
            "Graph structure has changed since partition was created.",
            expected_hash=partition_result.original_graph_hash,
            actual_hash=current_graph_hash
        )


def interactive_plot(graph: nx.Graph, title: str = None):
    """
    Plots a NetworkX graph on an interactive map.

    Args:
        graph: A NetworkX graph with node and edge attributes.
        title (Optional): Title for the plot.
    """
    import plotly.graph_objects as go
    import plotly.io as pio

    # Render in browser by default
    pio.renderers.default = "browser"

    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in graph.edges():
        # Get coordinates for the edge
        x0, y0 = graph.nodes[edge[0]]['lon'], graph.nodes[edge[0]]['lat']
        x1, y1 = graph.nodes[edge[1]]['lon'], graph.nodes[edge[1]]['lat']

        # Add them to the list, with a 'None' to break the line between edges
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scattermapbox(
        lon=edge_x,
        lat=edge_y,
        mode='lines',
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        name='Transmission Lines'
    )

    # Create node traces
    node_x, node_y, node_texts = [], [], []
    for node, data in graph.nodes(data=True):
        node_x.append(data['lon'])
        node_y.append(data['lat'])
        voltage = data.get('base_voltage', 'N/A')
        node_texts.append(f'Node: {node}<br>Base Voltage: {round(voltage, 3)} kV')

    node_trace = go.Scattermapbox(
        lon=node_x,
        lat=node_y,
        mode='markers',
        marker=dict(size=10, color='teal'),
        hoverinfo='text',
        text=node_texts,
        name='Substations'
    )

    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace])

    fig.update_layout(
        title_text=title if title else 'Power Network Graph',
        hovermode='closest',
        showlegend=True,
        mapbox=dict(
            style="carto-positron",
            bearing=0,
            center=dict(lat=50.5, lon=14.0),
            pitch=0,
            zoom=3.5
        ),
        margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )
    fig.show()
