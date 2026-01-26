"""
Network Partitioning & Aggregation Package (NPAP).

A Python library for partitioning and aggregation of spatial network
graph-based data with focus on electrical power systems. The system operates
on NetworkX graphs and implements a strategy pattern throughout, enabling
extensibility for new partitioning algorithms, aggregation methods, and
physical constraint handling.

Main Components
---------------
PartitionAggregatorManager
    Main orchestrator class for the complete workflow.
AggregationProfile
    Configuration for aggregation operations.
AggregationMode
    Pre-defined aggregation modes (SIMPLE, GEOGRAPHICAL, DC_KRON, CUSTOM).
PartitionResult
    Container for partition results with metadata.

Examples
--------
>>> from npap import PartitionAggregatorManager
>>> manager = PartitionAggregatorManager()
>>> manager.load_data("csv_files", node_file="buses.csv", edge_file="lines.csv")
>>> result = manager.partition("geographical_kmeans", n_clusters=10)
>>> aggregated = manager.aggregate()
"""

__author__ = "Marco Anarmo"

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("npap")
except PackageNotFoundError:
    __version__ = "unknown"

# Core components
# Aggregation mode helper
from npap.aggregation import get_mode_profile

# Exceptions
from npap.exceptions import (
    AggregationError,
    DataLoadingError,
    ElectricalCalculationError,
    GraphCompatibilityError,
    NPAPError,
    PartitioningError,
    StrategyNotFoundError,
    ValidationError,
)
from npap.interfaces import AggregationMode, AggregationProfile, PartitionResult
from npap.managers import PartitionAggregatorManager

# Main interface
__all__ = [
    "AggregationError",
    "AggregationMode",
    "AggregationProfile",
    "DataLoadingError",
    "ElectricalCalculationError",
    "GraphCompatibilityError",
    "NPAPError",
    "PartitionAggregatorManager",
    "PartitionResult",
    "PartitioningError",
    "StrategyNotFoundError",
    "ValidationError",
    "get_mode_profile",
]


def get_version():
    """
    Get package version.

    Returns
    -------
    str
        Package version string.
    """
    return __version__


def get_author():
    """
    Get package author.

    Returns
    -------
    str
        Package author name.
    """
    return __author__
