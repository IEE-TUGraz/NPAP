from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import plotly.graph_objects as go
import plotly.io as pio

from npap.interfaces import EdgeType


class PlotStyle(Enum):
    """
    Available visualization styles for power networks.

    SIMPLE: All edges rendered uniformly (fastest, minimal visual complexity)
    VOLTAGE_AWARE: Edges colored by type and voltage level (recommended for power systems)
    CLUSTERED: Nodes colored by cluster assignment (requires prior partitioning)
    """

    SIMPLE = "simple"
    VOLTAGE_AWARE = "voltage_aware"
    CLUSTERED = "clustered"


@dataclass
class PlotConfig:
    """
    Configuration for network visualization.

    This centralized configuration allows users to customize every aspect of the
    visualization without modifying code. All parameters have sensible defaults
    optimized for power system networks.
    """

    # === Display Toggles ===
    show_lines: bool = True
    show_trafos: bool = True
    show_dc_links: bool = True
    show_nodes: bool = True

    # === Voltage Thresholds ===
    # Lines above this voltage (kV) are considered "high voltage" for color differentiation
    line_voltage_threshold: float = 300.0

    # === Color Scheme ===
    line_high_voltage_color: str = "#029E73"  # Green: high voltage transmission
    line_low_voltage_color: str = "#CA9161"  # Brown: lower voltage distribution
    trafo_color: str = "#ECE133"  # Yellow: transformers (voltage change)
    dc_link_color: str = "#CC78BC"  # Pink: DC links (special transmission)
    node_color: str = "#0173B2"  # Blue: buses/substations

    # === Geometry Settings ===
    edge_width: float = 1.5
    node_size: int = 5

    # === Map Configuration ===
    map_style: str = "carto-positron"  # Light theme for readability
    map_center_lat: float = 57.5
    map_center_lon: float = 14.0
    map_zoom: float = 3.7

    # === Figure Settings ===
    title: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None

    # === Cluster Visualization ===
    cluster_colorscale: str = "Viridis"  # Perceptually uniform for cluster coloring


@dataclass
class EdgeGroup:
    """
    Container for aggregated edge coordinates by type/voltage category.

    This grouping strategy is crucial for performance: instead of creating
    separate Plotly traces for each edge (which would be O(n) traces for n edges),
    we create one trace per group (typically 4-5 traces total).

    Plotly renders this faster, and the user experience is
    significantly improved for networks with thousands of edges.
    """

    edge_type: str  # 'line', 'trafo', or 'dc_link'
    voltage_category: str  # 'high', 'low', 'trafo', or 'dc_link'
    lats: List[Optional[float]] = field(default_factory=list)
    lons: List[Optional[float]] = field(default_factory=list)
    count: int = 0  # Number of edges in this group


class EdgeStyleRegistry:
    """
    Centralized registry for edge styling rules.

    This class eliminates code duplication by providing a single source of truth
    for edge colors, display names, and rendering order. Adding new edge categories
    only requires updating this registry, not multiple methods.
    """

    # Display order determines legend and visual layering
    DISPLAY_ORDER = ["line_high", "line_low", "trafo", "dc_link"]

    @staticmethod
    def get_color(group_key: str, config: PlotConfig) -> str:
        """
        Map group keys to colors from configuration.

        Args:
            group_key: Identifier like "line_high", "trafo", etc.
            config: Plot configuration with color definitions

        Returns:
            Hex color string
        """
        color_map = {
            "line_high": config.line_high_voltage_color,
            "line_low": config.line_low_voltage_color,
            "trafo": config.trafo_color,
            "dc_link": config.dc_link_color,
        }
        return color_map.get(group_key, config.line_low_voltage_color)

    @staticmethod
    def get_display_name(group_key: str, config: PlotConfig) -> str:
        """
        Map group keys to human-readable display names for the legend.

        Args:
            group_key: Identifier like "line_high", "trafo", etc.
            config: Plot configuration with threshold values

        Returns:
            Display name string for legend
        """
        threshold = int(config.line_voltage_threshold)
        names = {
            "line_high": f"High Voltage Lines >{threshold}kV",
            "line_low": f"Low Voltage Lines ≤{threshold}kV",
            "trafo": "Transformers",
            "dc_link": "DC Links",
        }
        return names.get(group_key, "Edges")


class NetworkPlotter:
    """
    High-performance interactive visualization for power system networks.

    This class handles the complete visualization pipeline:
    1. Data preparation (coordinate extraction, edge grouping)
    2. Trace generation (edges, nodes)
    3. Figure assembly (layout, styling)
    """

    def __init__(
        self, graph: nx.DiGraph, partition_map: Optional[Dict[int, List[Any]]] = None
    ):
        """
        Initialize plotter with network data.

        Args:
            graph: NetworkX DiGraph with geographical coordinates (lat, lon)
            partition_map: Optional mapping of cluster_id -> [node_ids] for cluster visualization
        """
        self._graph = graph
        self._partition_map = partition_map

        # Pre-compute reverse mapping for O(1) cluster lookups during node coloring
        self._node_to_cluster = (
            self._build_node_to_cluster_map() if partition_map else None
        )

        # Cache all coordinates once for performance (avoids repeated dict lookups)
        self._node_coords = self._extract_node_coordinates()

    def _extract_node_coordinates(self) -> Dict[Any, Tuple[float, float]]:
        """
        Extract and cache all node coordinates for fast lookup.

        This caching strategy is critical for performance: with 6000 nodes and 8000 edges,
        we'd perform 16,000+ coordinate lookups during edge rendering. By caching,
        we reduce this to a single O(n) pass at initialization.

        Returns:
            Dictionary mapping node_id -> (lat, lon) tuple
        """
        coords = {}
        for node, data in self._graph.nodes(data=True):
            lat = data.get("lat")
            lon = data.get("lon")
            # Only cache nodes with valid coordinates (skip nodes without geolocation)
            if lat is not None and lon is not None:
                coords[node] = (lat, lon)
        return coords

    def _build_node_to_cluster_map(self) -> Dict[Any, int]:
        """
        Create reverse mapping from node_id to cluster_id.

        The partition_map is cluster_id -> [node_ids]. For efficient node coloring,
        we need the inverse: node_id -> cluster_id. This O(n) conversion at
        initialization enables O(1) lookups during visualization.

        Returns:
            Dictionary mapping node_id -> cluster_id
        """
        if not self._partition_map:
            return {}

        node_to_cluster = {}
        for cluster_id, nodes in self._partition_map.items():
            for node in nodes:
                node_to_cluster[node] = cluster_id
        return node_to_cluster

    @staticmethod
    def _should_show_edge_type(edge_type: str, config: PlotConfig) -> bool:
        """
        Determine if an edge type should be displayed based on configuration.

        Args:
            edge_type: Type of edge ('line', 'trafo', 'dc_link')
            config: Plot configuration with display toggles

        Returns:
            True if this edge type should be displayed
        """
        visibility_map = {
            EdgeType.LINE.value: config.show_lines,
            EdgeType.TRAFO.value: config.show_trafos,
            EdgeType.DC_LINK.value: config.show_dc_links,
        }
        return visibility_map.get(edge_type, True)

    @staticmethod
    def _categorize_edge(edge_data: dict, config: PlotConfig) -> Tuple[str, str]:
        """
        Determine edge type and voltage category for grouping.

        This method implements the classification logic:
        - Lines are split by voltage threshold (transmission vs distribution)
        - Transformers and DC links have their own categories

        Args:
            edge_data: Edge attributes dictionary
            config: Plot configuration with voltage threshold

        Returns:
            Tuple of (group_key, voltage_category)
        """
        edge_type = edge_data.get("type", EdgeType.LINE.value)

        if edge_type == EdgeType.LINE.value:
            # Classify lines by voltage: high voltage (transmission) vs low voltage
            primary_v = edge_data.get("primary_voltage", 0) or 0
            voltage_category = (
                "high" if primary_v > config.line_voltage_threshold else "low"
            )
            group_key = f"line_{voltage_category}"
        elif edge_type == EdgeType.TRAFO.value:
            voltage_category = "trafo"
            group_key = "trafo"
        else:  # DC Link
            voltage_category = "dc_link"
            group_key = "dc_link"

        return group_key, voltage_category

    def _group_edges_by_type(self, config: PlotConfig) -> Dict[str, EdgeGroup]:
        """
        Group all edges by type and voltage category for efficient rendering.

        This is the core performance optimization: instead of creating one Plotly
        trace per edge (which would create thousands of traces), we group edges
        by category and create one trace per group.

        The grouping uses Plotly's None-separation technique: coordinates are
        arranged as [x1, x2, None, x3, x4, None, ...] where None breaks the line.
        This allows thousands of line segments in a single trace.

        Args:
            config: Plot configuration with display toggles and thresholds

        Returns:
            Dictionary mapping group_key -> EdgeGroup with aggregated coordinates
        """
        groups: Dict[str, EdgeGroup] = {}

        for u, v, data in self._graph.edges(data=True):
            # Skip edges with missing node coordinates (shouldn't happen with valid data)
            if u not in self._node_coords or v not in self._node_coords:
                continue

            edge_type = data.get("type", EdgeType.LINE.value)

            # Respect display toggles from configuration
            if not self._should_show_edge_type(edge_type, config):
                continue

            # Classify edge into a group
            group_key, voltage_category = self._categorize_edge(data, config)

            # Create group on first encounter
            if group_key not in groups:
                groups[group_key] = EdgeGroup(
                    edge_type=edge_type, voltage_category=voltage_category
                )

            # Add edge coordinates to the group using None-separation
            # Format: [lon1, lon2, None, lon3, lon4, None, ...]
            # The None values tell Plotly to break the line, creating separate segments
            lat_u, lon_u = self._node_coords[u]
            lat_v, lon_v = self._node_coords[v]

            group = groups[group_key]
            group.lons.extend([lon_u, lon_v, None])
            group.lats.extend([lat_u, lat_v, None])
            group.count += 1

        return groups

    def _build_edge_traces_voltage_aware(
        self, config: PlotConfig
    ) -> List[go.Scattermapbox]:
        """
        Build edge traces grouped by type and voltage category.

        This method creates separate Plotly traces for each edge group,
        allowing users to toggle edge types in the legend and providing
        visual differentiation by color.

        Args:
            config: Plot configuration with colors and styling

        Returns:
            List of Scattermapbox traces for edges
        """
        groups = self._group_edges_by_type(config)
        traces = []

        # Process groups in defined order for consistent legend appearance
        for group_key in EdgeStyleRegistry.DISPLAY_ORDER:
            if group_key not in groups:
                continue

            group = groups[group_key]
            color = EdgeStyleRegistry.get_color(group_key, config)
            name = EdgeStyleRegistry.get_display_name(group_key, config)

            trace = go.Scattermapbox(
                lon=group.lons,
                lat=group.lats,
                mode="lines",
                line=dict(width=config.edge_width, color=color),
                name=f"{name} ({group.count})",  # Include count for user insight
                hoverinfo="name",
                legendgroup=group_key,
            )
            traces.append(trace)

        return traces

    def _build_edge_traces_simple(self, config: PlotConfig) -> List[go.Scattermapbox]:
        """
        Build a single edge trace with uniform styling.

        This is the fastest rendering option: all edges in one trace with one color.
        Useful for initial network overview or when edge differentiation isn't needed.

        Args:
            config: Plot configuration with basic styling

        Returns:
            List containing single edge trace (or empty if no valid edges)
        """
        lons = []
        lats = []

        # Aggregate all edges into a single coordinate list with None-separation
        for u, v in self._graph.edges():
            if u not in self._node_coords or v not in self._node_coords:
                continue

            lat_u, lon_u = self._node_coords[u]
            lat_v, lon_v = self._node_coords[v]
            lons.extend([lon_u, lon_v, None])
            lats.extend([lat_u, lat_v, None])

        if not lons:
            return []

        trace = go.Scattermapbox(
            lon=lons,
            lat=lats,
            mode="lines",
            line=dict(width=config.edge_width, color="#888"),
            hoverinfo="none",
            name="Transmission Lines",
        )
        return [trace]

    def _build_node_trace(
        self, config: PlotConfig, color_by_cluster: bool = False
    ) -> Optional[go.Scattermapbox]:
        """
        Build node trace with optional cluster-based coloring.

        Nodes are rendered as markers on the map. When color_by_cluster is True,
        each cluster gets a distinct color from the colorscale, making it easy
        to visualize the partitioning results.

        Args:
            config: Plot configuration with node styling
            color_by_cluster: If True, color nodes by cluster assignment

        Returns:
            Scattermapbox trace for nodes, or None if nodes are hidden
        """
        if not config.show_nodes:
            return None

        node_lons = []
        node_lats = []
        hover_texts = []
        colors = []

        for node, data in self._graph.nodes(data=True):
            if node not in self._node_coords:
                continue

            lat, lon = self._node_coords[node]
            node_lons.append(lon)
            node_lats.append(lat)

            # Format voltage information for hover text
            voltage = data.get("voltage") or data.get("base_voltage", "N/A")
            voltage_str = (
                f"{voltage:.2f} kV"
                if isinstance(voltage, (int, float))
                else str(voltage)
            )
            hover_texts.append(f"<b>{node}</b><br>Voltage: {voltage_str}")

            # Assign color based on cluster or use uniform color
            if color_by_cluster and self._node_to_cluster:
                cluster_id = self._node_to_cluster.get(node, -1)
                colors.append(cluster_id)
            else:
                colors.append(config.node_color)

        if not node_lons:
            return None

        # Configure marker appearance based on coloring mode
        if color_by_cluster and self._node_to_cluster:
            # Cluster coloring: use colorscale with legend
            marker = dict(
                size=config.node_size,
                color=colors,
                colorscale=config.cluster_colorscale,
                showscale=True,
                colorbar=dict(
                    title="Clusters",
                    title_font=dict(
                        color="#008080", size=16, family="Arial, sans-serif"
                    ),
                    tickfont=dict(color="#008080", size=14, family="Arial, sans-serif"),
                    x=0.99,
                    xanchor="right",
                    y=0.5,
                    yanchor="middle",
                    len=0.9,
                ),
            )
        else:
            # Uniform coloring: single color for all nodes
            marker = dict(size=config.node_size, color=config.node_color)

        trace = go.Scattermapbox(
            lon=node_lons,
            lat=node_lats,
            mode="markers",
            marker=marker,
            hoverinfo="text",
            text=hover_texts,
            name=f"Buses ({len(node_lons)})",
            legendgroup="nodes",
        )
        return trace

    @staticmethod
    def _create_figure(traces: List[go.Scattermapbox], config: PlotConfig) -> go.Figure:
        """
        Assemble Plotly figure with all traces and layout configuration.

        This method handles the final assembly: combining edge and node traces,
        configuring the mapbox layout, and setting up the legend for optimal
        user interaction.

        Args:
            traces: List of all Scattermapbox traces (edges + nodes)
            config: Plot configuration with layout settings

        Returns:
            Complete Plotly Figure ready for display
        """
        fig = go.Figure(data=traces)
        layout_kwargs = dict(
            # Title configuration
            title_text=config.title or "Power Network",
            title_font=dict(color="white", size=20, family="Arial, sans-serif"),
            title_y=0.994,
            title_x=0.5,
            title_xanchor="center",
            paper_bgcolor="#008080",
            hovermode="closest",
            showlegend=True,
            legend=dict(
                # Position legend in top-left with semi-transparent background
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.95)",
                bordercolor="#008080",
                borderwidth=1,
                font=dict(size=11),
                itemsizing="constant",
                tracegroupgap=5,
            ),
            mapbox=dict(
                style=config.map_style,
                bearing=0,
                center=dict(lat=config.map_center_lat, lon=config.map_center_lon),
                pitch=0,
                zoom=config.map_zoom,
            ),
            margin=dict(r=0, t=30, l=0, b=0),
        )

        # Apply optional dimension overrides
        if config.width:
            layout_kwargs["width"] = config.width
        if config.height:
            layout_kwargs["height"] = config.height

        fig.update_layout(**layout_kwargs)
        return fig

    def _plot(
        self, style: PlotStyle, config: Optional[PlotConfig] = None, show: bool = True
    ) -> go.Figure:
        """
        Centralized plotting method for all visualization styles.

        This method consolidates the logic from plot_simple, plot_voltage_aware,
        and plot_clustered into a single implementation, eliminating code duplication
        while maintaining the distinct visual styles.

        Args:
            style: Visualization style (SIMPLE, VOLTAGE_AWARE, or CLUSTERED)
            config: Optional plot configuration (uses defaults if not provided)
            show: Whether to display the figure immediately in browser

        Returns:
            Plotly Figure object

        Raises:
            ValueError: If CLUSTERED style requested without partition_map
        """
        config = config or PlotConfig()

        # Validate prerequisites for clustered visualization
        if style == PlotStyle.CLUSTERED and not self._partition_map:
            raise ValueError(
                "Cannot create clustered plot without partition_map. "
                "Provide partition_map during NetworkPlotter initialization."
            )

        # Build edge traces based on style
        if style == PlotStyle.SIMPLE:
            edge_traces = self._build_edge_traces_simple(config)
        else:  # VOLTAGE_AWARE or CLUSTERED both use voltage-aware edges
            edge_traces = self._build_edge_traces_voltage_aware(config)

        # Build node trace with optional cluster coloring
        color_by_cluster = style == PlotStyle.CLUSTERED
        node_trace = self._build_node_trace(config, color_by_cluster=color_by_cluster)

        # Assemble all traces
        traces = edge_traces
        if node_trace:
            traces.append(node_trace)

        # Create figure
        fig = self._create_figure(traces, config)

        # Display in browser if requested
        if show:
            pio.renderers.default = "browser"
            fig.show(config={"scrollZoom": True})

        return fig

    # Public API methods - these provide clean interfaces for each style

    def plot_simple(
        self, config: Optional[PlotConfig] = None, show: bool = True
    ) -> go.Figure:
        """
        Create simple visualization with uniform edge styling.

        Best for: Initial network overview, maximum performance

        Args:
            config: Optional plot configuration
            show: Whether to display immediately

        Returns:
            Plotly Figure object
        """
        return self._plot(PlotStyle.SIMPLE, config, show)

    def plot_voltage_aware(
        self, config: Optional[PlotConfig] = None, show: bool = True
    ) -> go.Figure:
        """
        Create voltage-aware visualization with edges colored by type and voltage.

        Best for: Understanding network structure, distinguishing transmission levels

        Args:
            config: Optional plot configuration
            show: Whether to display immediately

        Returns:
            Plotly Figure object
        """
        return self._plot(PlotStyle.VOLTAGE_AWARE, config, show)

    def plot_clustered(
        self, config: Optional[PlotConfig] = None, show: bool = True
    ) -> go.Figure:
        """
        Create clustered visualization with nodes colored by cluster assignment.

        Best for: Visualizing partitioning results, understanding cluster geography

        Args:
            config: Optional plot configuration
            show: Whether to display immediately

        Returns:
            Plotly Figure object

        Raises:
            ValueError: If partition_map was not provided during initialization
        """
        return self._plot(PlotStyle.CLUSTERED, config, show)


def plot_network(
    graph: nx.DiGraph,
    style: str = "simple",
    partition_map: Optional[Dict[int, List[Any]]] = None,
    show: bool = True,
    **kwargs,
) -> go.Figure:
    """
    Convenience function for quick network visualization.

    This function provides a simple interface for one-line plotting without
    needing to instantiate the NetworkPlotter class explicitly.

    Args:
        graph: NetworkX DiGraph with geographical coordinates (lat, lon)
        style: Visualization style ('simple', 'voltage_aware', or 'clustered')
        partition_map: Optional cluster mapping for 'clustered' style
        show: Whether to display the figure immediately
        **kwargs: Additional configuration parameters passed to PlotConfig

    Returns:
        Plotly Figure object

    Raises:
        ValueError: If style is unknown or prerequisites are missing
    """
    config = PlotConfig(**kwargs)
    plotter = NetworkPlotter(graph, partition_map=partition_map)

    # Support both string and enum style specifications
    if style == "simple" or style == PlotStyle.SIMPLE:
        return plotter.plot_simple(config, show=show)
    elif style == "voltage_aware" or style == PlotStyle.VOLTAGE_AWARE:
        return plotter.plot_voltage_aware(config, show=show)
    elif style == "clustered" or style == PlotStyle.CLUSTERED:
        return plotter.plot_clustered(config, show=show)
    else:
        raise ValueError(
            f"Unknown plot style: {style}. "
            f"Valid options: 'simple', 'voltage_aware', 'clustered'"
        )
