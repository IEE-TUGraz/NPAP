"""
Aggregation strategies for network partitioning and aggregation package.

Separated into:
- Topology strategies: Define graph structure
- Physical strategies: Apply electrical laws
- Property strategies: Statistical aggregation functions for nodes/edges
"""

from .basic_strategies import (
    # Topology strategies
    SimpleTopologyStrategy,
    ElectricalTopologyStrategy,
    # Node property strategies
    SumNodeStrategy,
    AverageNodeStrategy,
    FirstNodeStrategy,
    # Edge property strategies
    SumEdgeStrategy,
    AverageEdgeStrategy,
    FirstEdgeStrategy,
    EquivalentReactanceStrategy,
)
from .modes import get_mode_profile
from .physical_strategies import KronReductionStrategy

__all__ = [
    # Topology strategies
    "SimpleTopologyStrategy",
    "ElectricalTopologyStrategy",
    # Physical strategies
    "KronReductionStrategy",
    # Node property strategies
    "SumNodeStrategy",
    "AverageNodeStrategy",
    "FirstNodeStrategy",
    # Edge property strategies
    "SumEdgeStrategy",
    "AverageEdgeStrategy",
    "FirstEdgeStrategy",
    "EquivalentReactanceStrategy",
    # Mode profiles
    "get_mode_profile",
]
