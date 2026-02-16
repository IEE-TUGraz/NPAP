"""
Aggregation strategies for network partitioning and aggregation.

This module provides strategies for aggregating partitioned network graphs.

Strategy Categories
-------------------
Topology Strategies
    Define how the graph structure is reduced (SimpleTopologyStrategy,
    ElectricalTopologyStrategy).
Physical Strategies
    Apply electrical laws during aggregation (KronReductionStrategy).
Property Strategies
    Statistical aggregation functions for nodes (SumNodeStrategy,
    AverageNodeStrategy) and edges (SumEdgeStrategy, AverageEdgeStrategy).

Functions
---------
get_mode_profile
    Get pre-defined aggregation profile for a given mode.
"""

from .basic_strategies import (
    AverageEdgeStrategy,
    AverageNodeStrategy,
    ElectricalTopologyStrategy,
    EquivalentReactanceStrategy,
    FirstEdgeStrategy,
    FirstNodeStrategy,
    # Topology strategies
    SimpleTopologyStrategy,
    # Edge property strategies
    SumEdgeStrategy,
    # Node property strategies
    SumNodeStrategy,
    # Typed edge utilities
    build_typed_cluster_edge_map,
)
from .modes import get_mode_profile
from .physical_strategies import KronReductionStrategy

__all__ = [
    "AverageEdgeStrategy",
    "AverageNodeStrategy",
    "ElectricalTopologyStrategy",
    "EquivalentReactanceStrategy",
    "FirstEdgeStrategy",
    "FirstNodeStrategy",
    "KronReductionStrategy",
    "SimpleTopologyStrategy",
    "SumEdgeStrategy",
    "SumNodeStrategy",
    "build_typed_cluster_edge_map",
    "get_mode_profile",
]
