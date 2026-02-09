"""
Input data loading strategies for network partitioning and aggregation.

This module provides strategies for loading network data from various sources.

Strategies
----------
CSVFilesStrategy
    Load graph from separate CSV files for nodes and edges.
NetworkXDirectStrategy
    Use an existing NetworkX graph directly.
VoltageAwareStrategy
    Load voltage-aware data with lines, transformers, and DC links.
"""

from .csv_loader import CSVFilesStrategy
from .networkx_loader import NetworkXDirectStrategy
from .va_loader import VoltageAwareStrategy

__all__ = [
    "CSVFilesStrategy",
    "NetworkXDirectStrategy",
    "VoltageAwareStrategy",
]
