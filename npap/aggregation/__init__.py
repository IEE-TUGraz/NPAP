"""
Aggregation strategies for network partitioning and aggregation package.

Separated into:
- Topology strategies: Define graph structure
- Physical strategies: Apply electrical laws
- Property strategies: Statistical aggregation functions for nodes/edges
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
    "get_mode_profile",
]
