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

import sys as _sys
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("npap")
except PackageNotFoundError:
    __version__ = "unknown"

# Logging helpers.
#
# The implementation module is named `_logging` rather than `logging`: a file
# called `logging.py` inside the package shadows the standard library module
# for any process whose working directory is the package directory, which
# breaks `venv`, `pip` and anything else that imports stdlib logging.
from npap import _logging as _logging_module
from npap._logging import (
    LogCategory,
    configure_logging,
    disable_logging,
    enable_logging,
    get_logger,
)

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

# Backwards-compatible alias so that `from npap.logging import ...` keeps
# working even though no `logging.py` file exists on disk anymore.
_sys.modules.setdefault(__name__ + ".logging", _logging_module)

# Main interface
__all__ = [
    "AggregationError",
    "AggregationMode",
    "AggregationProfile",
    "DataLoadingError",
    "ElectricalCalculationError",
    "GraphCompatibilityError",
    "LogCategory",
    "NPAPError",
    "PartitionAggregatorManager",
    "PartitionResult",
    "PartitioningError",
    "StrategyNotFoundError",
    "ValidationError",
    "configure_logging",
    "disable_logging",
    "enable_logging",
    "get_logger",
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
