"""
Partitioning strategies for network partitioning and aggregation.

This module provides strategies for partitioning network graphs into clusters.

Strategies
----------
GeographicalPartitioning
    Partition based on geographical coordinates using various algorithms
    (kmeans, kmedoids, dbscan, hdbscan, hierarchical).
ElectricalDistancePartitioning
    Partition based on electrical distance using PTDF analysis.
VAGeographicalPartitioning
    Voltage-aware geographical partitioning respecting voltage level boundaries.
VAElectricalDistancePartitioning
    Voltage-aware electrical distance partitioning with AC island awareness.
"""

from .adjacent import AdjacentNodeAgglomerativePartitioning
from .electrical import ElectricalDistancePartitioning
from .geographical import GeographicalPartitioning
from .graph_theory import CommunityPartitioning, SpectralPartitioning
from .lmp import LMPPartitioning
from .va_electrical import VAElectricalDistancePartitioning
from .va_geographical import VAGeographicalPartitioning

__all__ = [
    "AdjacentNodeAgglomerativePartitioning",
    "CommunityPartitioning",
    "ElectricalDistancePartitioning",
    "GeographicalPartitioning",
    "LMPPartitioning",
    "SpectralPartitioning",
    "VAElectricalDistancePartitioning",
    "VAGeographicalPartitioning",
]
