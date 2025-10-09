"""
Network Partitioning & Aggregation Package (NPAP)

A Python library for partitioning and aggregation of spatial network graph-based data with focus on electrical
power systems. The system operates on NetworkX graphs and implements a strategy pattern throughout, enabling
extensibility for new partitioning algorithms, aggregation methods, and physical constraint handling.
"""

__version__ = "0.1.0"
__author__ = "Marco Antonio Arnaiz Montero"

# Core components
from .managers import PartitionAggregatorManager
from .interfaces import (
    AggregationProfile,
    AggregationMode,
    PartitionResult
)

# Aggregation mode helper
from .aggregation.modes import get_mode_profile

# Exceptions
from .exceptions import (
    NPAPError,
    DataLoadingError,
    PartitioningError,
    AggregationError,
    ElectricalCalculationError,
    ValidationError,
    GraphCompatibilityError,
    StrategyNotFoundError
)

# Main interface
__all__ = [
    'PartitionAggregatorManager',
    'AggregationProfile',
    'AggregationMode',
    'PartitionResult',
    'get_mode_profile',
    'NPAPError',
    'DataLoadingError',
    'PartitioningError',
    'AggregationError',
    'ElectricalCalculationError',
    'ValidationError',
    'GraphCompatibilityError',
    'StrategyNotFoundError',
]


def get_version():
    """Get package version"""
    return __version__
