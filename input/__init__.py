"""
Input data loading strategies for network partitioning and aggregation.
"""

from .csv_loader import CSVFilesStrategy
from .networkx_loader import NetworkXDirectStrategy

__all__ = [
    'CSVFilesStrategy',
    'NetworkXDirectStrategy',
]