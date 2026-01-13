"""
Network Partitioning & Aggregation Package (NPAP)

A Python library for partitioning and aggregation of spatial network graph-based data with focus on electrical
power systems. The system operates on NetworkX graphs and implements a strategy pattern throughout, enabling
extensibility for new partitioning algorithms, aggregation methods, and physical constraint handling.
"""

__author__ = "Marco Antonio Arnaiz Montero"

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("npap")
except PackageNotFoundError:
    __version__ = "unknown"

# Core components
from npap.managers import PartitionAggregatorManager
from npap.interfaces import AggregationProfile, AggregationMode, PartitionResult

# Aggregation mode helper
from npap.aggregation import get_mode_profile

# Exceptions
from npap.exceptions import (
    NPAPError,
    DataLoadingError,
    PartitioningError,
    AggregationError,
    ElectricalCalculationError,
    ValidationError,
    GraphCompatibilityError,
    StrategyNotFoundError,
)

# Main interface
__all__ = [
    "PartitionAggregatorManager",
    "AggregationProfile",
    "AggregationMode",
    "PartitionResult",
    "get_mode_profile",
    "NPAPError",
    "DataLoadingError",
    "PartitioningError",
    "AggregationError",
    "ElectricalCalculationError",
    "ValidationError",
    "GraphCompatibilityError",
    "StrategyNotFoundError",
]


def get_version():
    """Get package version"""
    return __version__


def get_author():
    """Get package author"""
    return __author__
