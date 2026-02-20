"""
Pre-defined aggregation modes for common use cases

Each mode provides a validated combination of:
- Topology strategy
- Physical aggregation strategy (if applicable)
- Statistical property aggregation defaults
"""

from npap.interfaces import AggregationMode, AggregationProfile


def get_mode_profile(mode: AggregationMode, **overrides) -> AggregationProfile:
    """
    Get pre-configured aggregation profile for a given mode.

    Parameters
    ----------
    mode : AggregationMode
        Aggregation mode enum.
    **overrides
        Override any profile parameters.

    Returns
    -------
    AggregationProfile
        AggregationProfile configured for the mode.

    Examples
    --------
    >>> profile = get_mode_profile(AggregationMode.GEOGRAPHICAL)
    """
    if mode == AggregationMode.SIMPLE:
        profile = _simple_mode()
    elif mode == AggregationMode.GEOGRAPHICAL:
        profile = _geographical_mode()
    elif mode == AggregationMode.CUSTOM:
        profile = AggregationProfile(mode=AggregationMode.CUSTOM)
    elif mode == AggregationMode.CONSERVATION:
        profile = _conservation_mode()
    elif mode == AggregationMode.DC_PTDF:
        profile = _ptdf_mode()
    elif mode == AggregationMode.DC_KRON:
        profile = _kron_mode()
    else:
        raise ValueError(f"Unknown aggregation mode: {mode}")

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(profile, key):
            current_value = getattr(profile, key)

            if isinstance(current_value, dict) and isinstance(value, dict):
                # Add new keys or overwrites existing ones
                current_value.update(value)

            elif isinstance(current_value, list) and isinstance(value, list):
                # Add the user's list to the end of the existing list
                current_value.extend(value)

            else:
                # Replace the existing value
                setattr(profile, key, value)

    return profile


def _simple_mode() -> AggregationProfile:
    """
    Simple aggregation mode - basic statistical aggregation

    Use case: Generic graph aggregation without physical constraints

    - Topology: Simple (no new edges)
    - Physical: None
    - Node properties: Sum numerical, first for others
    - Edge properties: Sum numerical, first for others
    """
    return AggregationProfile(
        mode=AggregationMode.SIMPLE,
        topology_strategy="simple",
        physical_strategy=None,
        physical_properties=[],
        node_properties={},  # Will use defaults
        edge_properties={},  # Will use defaults
        default_node_strategy="sum",
        default_edge_strategy="sum",
        warn_on_defaults=False,  # Simple mode uses defaults by design
    )


def _geographical_mode() -> AggregationProfile:
    """
    Geographical aggregation mode

    Use case: Spatial network aggregation based on geographical clustering

    - Topology: Simple
    - Physical: None
    - Node properties: Average coordinates, sum base voltage
    - Edge properties: Sum p_max, average reactance (x)
    """
    return AggregationProfile(
        mode=AggregationMode.GEOGRAPHICAL,
        topology_strategy="simple",
        physical_strategy=None,
        physical_properties=[],
        node_properties={
            "lat": "average",  # similar to middle point
            "lon": "average",  # similar to middle point
            "base_voltage": "average",
        },
        edge_properties={"p_max": "sum", "x": "equivalent_reactance"},
        default_node_strategy="average",
        default_edge_strategy="average",
        warn_on_defaults=True,
    )


def _conservation_mode() -> AggregationProfile:
    """
    Transformer conservation mode.

    Uses electrical topology plus the transformer conservation physical strategy
    to preserve equivalent reactance/resistance for transformer connections.
    """
    return AggregationProfile(
        mode=AggregationMode.CONSERVATION,
        topology_strategy="electrical",
        physical_strategy="transformer_conservation",
        physical_properties=["x", "r"],
        node_properties={
            "lat": "average",
            "lon": "average",
            "voltage": "average",
        },
        edge_properties={"p_max": "sum"},
        default_node_strategy="average",
        default_edge_strategy="sum",
        warn_on_defaults=True,
    )


def _ptdf_mode() -> AggregationProfile:
    """
    PTDF-based aggregation mode for DC networks.

    Builds an electrical topology, applies the PTDF reduction physical strategy,
    and keeps reactance as a physical property so equivalent PTDF-driven
    reactances are propagated before statistical aggregation.
    """
    return AggregationProfile(
        mode=AggregationMode.DC_PTDF,
        topology_strategy="electrical",
        physical_strategy="ptdf_reduction",
        physical_properties=["x"],
        node_properties={
            "lat": "average",
            "lon": "average",
        },
        edge_properties={"p_max": "sum"},
        default_node_strategy="average",
        default_edge_strategy="sum",
        warn_on_defaults=True,
    )


def _kron_mode() -> AggregationProfile:
    """
    Kron reduction mode for DC networks.

    Electrical topology + Kron reduction for reactances that match a reduced
    DC network of cluster representatives.
    """
    return AggregationProfile(
        mode=AggregationMode.DC_KRON,
        topology_strategy="electrical",
        physical_strategy="kron_reduction",
        physical_properties=["x"],
        node_properties={
            "lat": "average",
            "lon": "average",
        },
        edge_properties={"p_max": "sum"},
        default_node_strategy="average",
        default_edge_strategy="sum",
        warn_on_defaults=True,
    )
