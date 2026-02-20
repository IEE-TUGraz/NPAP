from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from npap.interfaces import EdgeType, PartitionResult


class PlotStyle(Enum):
    """
    Define available visualization styles for power networks.

    Attributes
    ----------
    SIMPLE : str
        All edges rendered uniformly (fastest, minimal visual complexity).
    VOLTAGE_AWARE : str
        Edges colored by type and voltage level (recommended for power systems).
    CLUSTERED : str
        Nodes colored by cluster assignment (requires prior partitioning).
    """

    SIMPLE = "simple"
    VOLTAGE_AWARE = "voltage_aware"
    CLUSTERED = "clustered"


class PlotPreset(Enum):
    """
    Preset configurations for quick styling adjustments.

    Attributes
    ----------
    DEFAULT : str
        Balanced defaults for data exploration.
    PRESENTATION : str
        Bigger nodes/edges and a wide canvas for slides or demos.
    DENSE : str
        Higher voltage threshold and compact markers for crowded networks.
    CLUSTER_HIGHLIGHT : str
        Emphasizes cluster coloring with saturated nodes and Turbo colorscale.
    TRANSMISSION_STUDY : str
        Highlights high-voltage corridors with a wide canvas and terrain tiles.
    DISTRIBUTION_STUDY : str
        Focuses on low-voltage, high-density neighborhoods with tighter zoom.
    E_MOBILITY_PLANNING : str
        Accents nodes most relevant for e-mobility rollout with bold markers.
    """

    DEFAULT = "default"
    PRESENTATION = "presentation"
    DENSE = "dense"
    CLUSTER_HIGHLIGHT = "cluster_highlight"
    TRANSMISSION_STUDY = "transmission_study"
    DISTRIBUTION_STUDY = "distribution_study"
    E_MOBILITY_PLANNING = "e_mobility_planning"


_PRESET_OVERRIDES = {
    PlotPreset.DEFAULT: {},
    PlotPreset.PRESENTATION: {
        "edge_width": 2.5,
        "node_size": 7,
        "map_style": "open-street-map",
        "width": 1100,
        "height": 700,
    },
    PlotPreset.DENSE: {
        "line_voltage_threshold": 400.0,
        "edge_width": 1.2,
        "node_size": 4,
        "map_zoom": 4.5,
        "map_style": "carto-darkmatter",
    },
    PlotPreset.CLUSTER_HIGHLIGHT: {
        "cluster_colorscale": "Turbo",
        "node_size": 8,
        "edge_width": 1.8,
        "map_style": "carto-positron",
        "title": "Clustered Network",
    },
    PlotPreset.TRANSMISSION_STUDY: {
        "line_voltage_threshold": 450.0,
        "edge_width": 2.3,
        "node_size": 6,
        "map_style": "stamen-terrain",
        "width": 1200,
        "height": 750,
        "title": "Transmission Study",
    },
    PlotPreset.DISTRIBUTION_STUDY: {
        "line_voltage_threshold": 220.0,
        "edge_width": 1.0,
        "node_size": 9,
        "map_zoom": 8.8,
        "map_style": "open-street-map",
        "cluster_colorscale": "YlOrBr",
        "title": "Distribution Study",
    },
    PlotPreset.E_MOBILITY_PLANNING: {
        "node_color": "#FF6F61",
        "node_size": 10,
        "edge_width": 1.1,
        "map_zoom": 10.5,
        "map_style": "stamen-toner",
        "title": "E-Mobility Planning",
    },
}


def _normalize_plot_preset(preset: PlotPreset | str | None) -> PlotPreset | None:
    """
    Normalize a preset specifier to a PlotPreset enum value.

    Raises
    ------
    ValueError
        If the provided string does not match any preset.
    """
    if preset is None:
        return None

    if isinstance(preset, PlotPreset):
        return preset

    lookup = preset.strip().lower().replace(" ", "_").replace("-", "_")
    for option in PlotPreset:
        if option.value == lookup or option.name.lower() == lookup:
            return option

    raise ValueError(f"Unknown preset: {preset}. Valid options: {[p.value for p in PlotPreset]}")


def _apply_preset_overrides(config: PlotConfig, preset: PlotPreset | str | None) -> PlotConfig:
    """
    Apply preset overrides to the provided PlotConfig.
    """
    preset_enum = _normalize_plot_preset(preset)
    if not preset_enum:
        return config

    overrides = _PRESET_OVERRIDES.get(preset_enum, {})
    if not overrides:
        return config

    return replace(config, **overrides)


def _resolve_partition_map(
    partition_map: dict[int, list[Any]] | PartitionResult | None,
    partition_result: PartitionResult | None,
) -> dict[int, list[Any]] | None:
    """
    Resolve either a partition map or PartitionResult into the final mapping.
    """
    if partition_result:
        return partition_result.mapping

    if isinstance(partition_map, PartitionResult):
        return partition_map.mapping

    return partition_map


@dataclass
class PlotConfig:
    """
    Configure network visualization parameters.

    This centralized configuration allows users to customize every aspect of the
    visualization without modifying code. All parameters have sensible defaults
    optimized for power system networks.

    Attributes
    ----------
    show_lines : bool
        Whether to display transmission lines.
    show_trafos : bool
        Whether to display transformers.
    show_dc_links : bool
        Whether to display DC links.
    show_nodes : bool
        Whether to display nodes/buses.
    line_voltage_threshold : float
        Voltage threshold (kV) for high/low voltage line classification.
    line_high_voltage_color : str
        Hex color for high voltage lines.
    line_low_voltage_color : str
        Hex color for low voltage lines.
    trafo_color : str
        Hex color for transformers.
    dc_link_color : str
        Hex color for DC links.
    node_color : str
        Hex color for nodes/buses.
    edge_width : float
        Width of edge lines in pixels.
    node_size : int
        Size of node markers in pixels.
    map_style : str
        Mapbox style for the base map.
    map_center_lat : float
        Initial map center latitude.
    map_center_lon : float
        Initial map center longitude.
    map_zoom : float
        Initial map zoom level.
    title : str or None
        Figure title.
    width : int or None
        Figure width in pixels.
    height : int or None
        Figure height in pixels.
    cluster_colorscale : str
        Plotly colorscale for cluster coloring.
    """

    show_lines: bool = True
    show_trafos: bool = True
    show_dc_links: bool = True
    show_nodes: bool = True
    line_voltage_threshold: float = 300.0
    line_high_voltage_color: str = "#029E73"
    line_low_voltage_color: str = "#CA9161"
    trafo_color: str = "#ECE133"
    dc_link_color: str = "#CC78BC"
    node_color: str = "#0173B2"
    edge_width: float = 1.5
    node_size: int = 5
    map_style: str = "carto-positron"
    map_center_lat: float = 57.5
    map_center_lon: float = 14.0
    map_zoom: float = 3.7
    title: str | None = None
    width: int | None = None
    height: int | None = None
    cluster_colorscale: str = "Viridis"


@dataclass
class EdgeGroup:
    """
    Store aggregated edge coordinates by type/voltage category.

    This grouping strategy is crucial for performance: instead of creating
    separate Plotly traces for each edge (which would be O(n) traces for n edges),
    we create one trace per group (typically 4-5 traces total).

    Attributes
    ----------
    edge_type : str
        Type of edge ('line', 'trafo', or 'dc_link').
    voltage_category : str
        Voltage category ('high', 'low', 'trafo', or 'dc_link').
    lats : list[float or None]
        List of latitude coordinates with None separators.
    lons : list[float or None]
        List of longitude coordinates with None separators.
    count : int
        Number of edges in this group.
    """

    edge_type: str
    voltage_category: str
    lats: list[float | None] = field(default_factory=list)
    lons: list[float | None] = field(default_factory=list)
    count: int = 0


class EdgeStyleRegistry:
    """
    Provide centralized registry for edge styling rules.

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

        Parameters
        ----------
        group_key : str
            Identifier like "line_high", "trafo", etc.
        config : PlotConfig
            Plot configuration with color definitions.

        Returns
        -------
        str
            Hex color string.
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

        Parameters
        ----------
        group_key : str
            Identifier like "line_high", "trafo", etc.
        config : PlotConfig
            Plot configuration with threshold values.

        Returns
        -------
        str
            Display name string for legend.
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
    Create high-performance interactive visualizations for power networks.

    This class handles the complete visualization pipeline:

    1. Data preparation (coordinate extraction, edge grouping)
    2. Trace generation (edges, nodes)
    3. Figure assembly (layout, styling)

    Parameters
    ----------
    graph : nx.DiGraph
        NetworkX DiGraph with geographical coordinates (lat, lon).
    partition_map : dict[int, list[Any]] or None
        Optional mapping of cluster_id to node_ids for cluster visualization.

    Examples
    --------
    >>> plotter = NetworkPlotter(graph, partition_map=partition.mapping)
    >>> fig = plotter.plot_clustered()
    """

    def __init__(self, graph: nx.DiGraph, partition_map: dict[int, list[Any]] | None = None):
        """
        Initialize plotter with network data.

        Parameters
        ----------
        graph : nx.DiGraph
            NetworkX DiGraph with geographical coordinates (lat, lon).
        partition_map : dict[int, list[Any]] or None
            Optional mapping of cluster_id to node_ids for cluster visualization.
        """
        self._graph = graph
        self._partition_map = partition_map

        # Pre-compute reverse mapping for O(1) cluster lookups during node coloring
        self._node_to_cluster = self._build_node_to_cluster_map() if partition_map else None

        # Cache all coordinates once for performance (avoids repeated dict lookups)
        self._node_coords = self._extract_node_coordinates()

    def _extract_node_coordinates(self) -> dict[Any, tuple[float, float]]:
        """
        Extract and cache all node coordinates for fast lookup.

        This caching strategy is critical for performance: with 6000 nodes and
        8000 edges, we'd perform 16,000+ coordinate lookups during edge rendering.
        By caching, we reduce this to a single O(n) pass at initialization.

        Returns
        -------
        dict[Any, tuple[float, float]]
            Dictionary mapping node_id to (lat, lon) tuple.
        """
        coords = {}
        for node, data in self._graph.nodes(data=True):
            lat = data.get("lat")
            lon = data.get("lon")
            # Only cache nodes with valid coordinates (skip nodes without geolocation)
            if lat is not None and lon is not None:
                coords[node] = (lat, lon)
        return coords

    def _build_node_to_cluster_map(self) -> dict[Any, int]:
        """
        Create reverse mapping from node_id to cluster_id.

        The partition_map is cluster_id -> [node_ids]. For efficient node coloring,
        we need the inverse: node_id -> cluster_id. This O(n) conversion at
        initialization enables O(1) lookups during visualization.

        Returns
        -------
        dict[Any, int]
            Dictionary mapping node_id to cluster_id.
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

        Parameters
        ----------
        edge_type : str
            Type of edge ('line', 'trafo', 'dc_link').
        config : PlotConfig
            Plot configuration with display toggles.

        Returns
        -------
        bool
            True if this edge type should be displayed.
        """
        visibility_map = {
            EdgeType.LINE.value: config.show_lines,
            EdgeType.TRAFO.value: config.show_trafos,
            EdgeType.DC_LINK.value: config.show_dc_links,
        }
        return visibility_map.get(edge_type, True)

    @staticmethod
    def _categorize_edge(edge_data: dict, config: PlotConfig) -> tuple[str, str]:
        """
        Determine edge type and voltage category for grouping.

        This method implements the classification logic:

        - Lines are split by voltage threshold (transmission vs distribution)
        - Transformers and DC links have their own categories

        Parameters
        ----------
        edge_data : dict
            Edge attributes dictionary.
        config : PlotConfig
            Plot configuration with voltage threshold.

        Returns
        -------
        tuple[str, str]
            Tuple of (group_key, voltage_category).
        """
        edge_type = edge_data.get("type", EdgeType.LINE.value)

        if edge_type == EdgeType.LINE.value:
            # Classify lines by voltage: high voltage (transmission) vs low voltage
            primary_v = edge_data.get("primary_voltage", 0) or 0
            voltage_category = "high" if primary_v > config.line_voltage_threshold else "low"
            group_key = f"line_{voltage_category}"
        elif edge_type == EdgeType.TRAFO.value:
            voltage_category = "trafo"
            group_key = "trafo"
        else:  # DC Link
            voltage_category = "dc_link"
            group_key = "dc_link"

        return group_key, voltage_category

    def _group_edges_by_type(self, config: PlotConfig) -> dict[str, EdgeGroup]:
        """
        Group all edges by type and voltage category for efficient rendering.

        This is the core performance optimization: instead of creating one Plotly
        trace per edge (which would create thousands of traces), we group edges
        by category and create one trace per group.

        The grouping uses Plotly's None-separation technique: coordinates are
        arranged as [x1, x2, None, x3, x4, None, ...] where None breaks the line.
        This allows thousands of line segments in a single trace.

        Parameters
        ----------
        config : PlotConfig
            Plot configuration with display toggles and thresholds.

        Returns
        -------
        dict[str, EdgeGroup]
            Dictionary mapping group_key to EdgeGroup with aggregated coordinates.
        """
        groups: dict[str, EdgeGroup] = {}

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

    def _build_edge_traces_voltage_aware(self, config: PlotConfig) -> list[go.Scattermapbox]:
        """
        Build edge traces grouped by type and voltage category.

        This method creates separate Plotly traces for each edge group,
        allowing users to toggle edge types in the legend and providing
        visual differentiation by color.

        Parameters
        ----------
        config : PlotConfig
            Plot configuration with colors and styling.

        Returns
        -------
        list[go.Scattermapbox]
            List of Scattermapbox traces for edges.
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

    def _build_edge_traces_simple(self, config: PlotConfig) -> list[go.Scattermapbox]:
        """
        Build a single edge trace with uniform styling.

        This is the fastest rendering option: all edges in one trace with one color.
        Useful for initial network overview or when edge differentiation isn't needed.

        Parameters
        ----------
        config : PlotConfig
            Plot configuration with basic styling.

        Returns
        -------
        list[go.Scattermapbox]
            List containing single edge trace (or empty if no valid edges).
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
    ) -> go.Scattermapbox | None:
        """
        Build node trace with optional cluster-based coloring.

        Nodes are rendered as markers on the map. When color_by_cluster is True,
        each cluster gets a distinct color from the colorscale, making it easy
        to visualize the partitioning results.

        Parameters
        ----------
        config : PlotConfig
            Plot configuration with node styling.
        color_by_cluster : bool
            If True, color nodes by cluster assignment.

        Returns
        -------
        go.Scattermapbox or None
            Scattermapbox trace for nodes, or None if nodes are hidden.
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
            voltage_str = f"{voltage:.2f} kV" if isinstance(voltage, (int, float)) else str(voltage)
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
                    title_font=dict(color="#008080", size=16, family="Arial, sans-serif"),
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
    def _create_figure(traces: list[go.Scattermapbox], config: PlotConfig) -> go.Figure:
        """
        Assemble Plotly figure with all traces and layout configuration.

        This method handles the final assembly: combining edge and node traces,
        configuring the mapbox layout, and setting up the legend for optimal
        user interaction.

        Parameters
        ----------
        traces : list[go.Scattermapbox]
            List of all Scattermapbox traces (edges + nodes).
        config : PlotConfig
            Plot configuration with layout settings.

        Returns
        -------
        go.Figure
            Complete Plotly Figure ready for display.
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
        self, style: PlotStyle, config: PlotConfig | None = None, show: bool = True
    ) -> go.Figure:
        """
        Execute centralized plotting for all visualization styles.

        This method consolidates the logic from plot_simple, plot_voltage_aware,
        and plot_clustered into a single implementation, eliminating code duplication
        while maintaining the distinct visual styles.

        Parameters
        ----------
        style : PlotStyle
            Visualization style (SIMPLE, VOLTAGE_AWARE, or CLUSTERED).
        config : PlotConfig or None
            Optional plot configuration (uses defaults if not provided).
        show : bool
            Whether to display the figure immediately in browser.

        Returns
        -------
        go.Figure
            Plotly Figure object.

        Raises
        ------
        ValueError
            If CLUSTERED style requested without partition_map.
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

    def plot_simple(self, config: PlotConfig | None = None, show: bool = True) -> go.Figure:
        """
        Create simple visualization with uniform edge styling.

        Best for: Initial network overview, maximum performance.

        Parameters
        ----------
        config : PlotConfig or None
            Optional plot configuration.
        show : bool
            Whether to display immediately.

        Returns
        -------
        go.Figure
            Plotly Figure object.
        """
        return self._plot(PlotStyle.SIMPLE, config, show)

    def plot_voltage_aware(self, config: PlotConfig | None = None, show: bool = True) -> go.Figure:
        """
        Create voltage-aware visualization with edges colored by type and voltage.

        Best for: Understanding network structure, distinguishing transmission levels.

        Parameters
        ----------
        config : PlotConfig or None
            Optional plot configuration.
        show : bool
            Whether to display immediately.

        Returns
        -------
        go.Figure
            Plotly Figure object.
        """
        return self._plot(PlotStyle.VOLTAGE_AWARE, config, show)

    def plot_clustered(self, config: PlotConfig | None = None, show: bool = True) -> go.Figure:
        """
        Create clustered visualization with nodes colored by cluster assignment.

        Best for: Visualizing partitioning results, understanding cluster geography.

        Parameters
        ----------
        config : PlotConfig or None
            Optional plot configuration.
        show : bool
            Whether to display immediately.

        Returns
        -------
        go.Figure
            Plotly Figure object.

        Raises
        ------
        ValueError
            If partition_map was not provided during initialization.
        """
        return self._plot(PlotStyle.CLUSTERED, config, show)

def plot_network(
    graph: nx.DiGraph,
    style: str = "simple",
    partition_map: dict[int, list[Any]] | PartitionResult | None = None,
    partition_result: PartitionResult | None = None,
    show: bool = True,
    preset: PlotPreset | str | None = None,
    config: PlotConfig | None = None,
    **kwargs,
) -> go.Figure:
    """
    Create a quick network visualization.

    This function provides a simple interface for one-line plotting without
    needing to instantiate the NetworkPlotter class explicitly.

    Parameters
    ----------
    graph : nx.DiGraph
        NetworkX DiGraph with geographical coordinates (lat, lon).
    style : str
        Visualization style ('simple', 'voltage_aware', or 'clustered').
    partition_map : dict[int, list[Any]] | PartitionResult or None
        Optional cluster mapping for 'clustered' style (or pass a PartitionResult).
    partition_result : PartitionResult | None
        Alternative place to hand over a PartitionResult directly without
        extracting ``mapping`` manually.
    show : bool
        Whether to display the figure immediately.
    preset : PlotPreset or str, optional
        Named preset that tweaks sizing, map style, and thresholds.
    config : PlotConfig or None
        Optional PlotConfig instance to override defaults. If provided,
        kwargs will further override values from this config.
    **kwargs : dict
        Additional configuration parameters passed to PlotConfig.
        These override both defaults and any provided config values.

    Returns
    -------
    go.Figure
        Plotly Figure object.

    Raises
    ------
    ValueError
        If style is unknown or prerequisites are missing.

    Examples
    --------
    >>> from npap.visualization import plot_network
    >>> fig = plot_network(graph, style="voltage_aware", title="My Network")
    >>> fig = plot_network(graph, style="clustered", partition_map=result.mapping)
    """
    base_config = replace(config) if config else PlotConfig()
    config_with_preset = _apply_preset_overrides(base_config, preset)

    if kwargs:
        effective_config = replace(config_with_preset, **kwargs)
    else:
        effective_config = config_with_preset

    resolved_partition = _resolve_partition_map(partition_map, partition_result)
    plotter = NetworkPlotter(graph, partition_map=resolved_partition)

    # Support both string and enum style specifications
    if style == "simple" or style == PlotStyle.SIMPLE:
        return plotter.plot_simple(effective_config, show=show)
    elif style == "voltage_aware" or style == PlotStyle.VOLTAGE_AWARE:
        return plotter.plot_voltage_aware(effective_config, show=show)
    elif style == "clustered" or style == PlotStyle.CLUSTERED:
        return plotter.plot_clustered(effective_config, show=show)
    else:
        raise ValueError(
            f"Unknown plot style: {style}. Valid options: 'simple', 'voltage_aware', 'clustered'"
        )


def export_figure(
    fig: go.Figure,
    path: str | Path,
    format: str | None = None,
    *,
    scale: float = 1,
    include_plotlyjs: str = "cdn",
    engine: str | None = None,
) -> Path:
    """
    Export a Plotly figure to disk (HTML or static image).

    Parameters
    ----------
    fig : go.Figure
        Figure to export.
    path : str or Path
        Target file path.
    format : str or None
        Optional format override (``"html"``, ``"png"``, ``"svg"``, etc.).
        When ``None``, the extension of ``path`` determines the format
        (defaults to ``html`` when missing).
    scale : float
        Scale factor applied when saving static image formats.
    include_plotlyjs : str
        Plotly.js bundling mode for HTML export (`"cdn"`, `"include"`, or `"relative"`).
    engine : str or None
        Plotly image engine for formats like PNG/SVG (`"kaleido"` by default).

    Returns
    -------
    Path
        Resolved path to the exported file.

    Raises
    ------
    RuntimeError
        If the requested format cannot be generated (e.g., ``kaleido`` missing).
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    resolved_format = (format or target.suffix.lstrip(".")).lower()
    resolved_format = resolved_format or "html"

    if resolved_format == "html":
        fig.write_html(str(target), include_plotlyjs=include_plotlyjs)
        return target

    try:
        fig.write_image(
            str(target),
            format=resolved_format,
            scale=scale,
            engine=engine or "kaleido",
        )
    except ValueError as exc:
        raise RuntimeError(
            "Failed to export figure. Make sure the required image engine "
            "(e.g., kaleido) is installed."
        ) from exc

    return target


def clone_graph(graph: nx.Graph | nx.MultiGraph | nx.MultiDiGraph) -> nx.Graph | nx.MultiGraph | nx.MultiDiGraph:
    """
    Return a deep copy of the supplied graph for safe downstream edits.

    Parameters
    ----------
    graph : nx.Graph or nx.MultiGraph or nx.MultiDiGraph
        Graph to clone.

    Returns
    -------
    nx.Graph or nx.MultiGraph or nx.MultiDiGraph
        Deep copy of the original graph.
    """
    return copy.deepcopy(graph)


def plot_reduced_matrices(
    graph: nx.Graph,
    *,
    matrices: tuple[str, ...] = ("ptdf", "laplacian"),
    show: bool = True,
) -> go.Figure:
    """
    Plot heatmaps for reduced PTDF/laplacian matrices produced during aggregation.

    Parameters
    ----------
    graph : nx.Graph
        Aggregated graph carrying ``reduced_ptdf`` and/or
        ``kron_reduced_laplacian`` in ``graph.graph``.
    matrices : tuple[str, ...]
        Which matrices to visualize; valid values are ``"ptdf"`` and
        ``"laplacian"``.
    show : bool
        Whether to display the figure automatically.

    Returns
    -------
    go.Figure
        Plotly Figure containing the requested heatmaps.
    """
    available = []
    if "ptdf" in matrices:
        ptdf = graph.graph.get("reduced_ptdf")
        if ptdf and isinstance(ptdf.get("matrix"), np.ndarray):
            available.append(("PTDF", ptdf["matrix"], ptdf["nodes"]))
    if "laplacian" in matrices:
        lap = graph.graph.get("kron_reduced_laplacian")
        if isinstance(lap, np.ndarray):
            labels = list(graph.nodes())
            available.append(("Kron Laplacian", lap, labels))

    if not available:
        raise ValueError("No reduced matrices found on the graph.")

    fig = make_subplots(rows=len(available), cols=1, subplot_titles=[name for name, *_ in available])

    for row, (name, matrix, labels) in enumerate(available, start=1):
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                x=[str(label) for label in labels],
                y=[str(label) for label in labels],
                colorbar=dict(title=name),
                colorscale="Viridis",
            ),
            row=row,
            col=1,
        )

    fig.update_layout(
        height=300 * len(available),
        title="Reduced matrices diagnostics",
        xaxis=dict(tickangle=45),
        yaxis=dict(autorange="reversed"),
    )

    if show:
        fig.show()

    return fig
