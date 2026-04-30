from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from npap.exceptions import PartitioningError, ValidationError
from npap.interfaces import EdgeType, PartitioningStrategy
from npap.logging import LogCategory, log_debug, log_info
from npap.utils import (
    create_partition_map,
    validate_partition,
    with_runtime_config,
)


@dataclass
class InputDataCongestionSensitiveConfig:
    """
    Configuration parameters for input data congestion sensitive partitioning.

    Attributes
    ----------
    zero_reactance_replacement : float
        Reactance value used when edge reactance is zero.
    regularization_factor : float
        Small value added to B matrix diagonal for numerical stability.
    infinite_distance : float
        Value used to represent "infinite" distance between AC islands.
    """

    zero_reactance_replacement: float = 1e-5
    regularization_factor: float = 1e-10
    infinite_distance: float = 1e4


class InputDataCongestionSensitivePartitioning(PartitioningStrategy):
    """
    Partition nodes based on congestion sensitivity using PTDF and network data.

    This strategy extends the electrical distance approach by incorporating
    line capacities and nodal generation/demand data to identify areas
    with similar impact on potential network congestion.

    Distance feature basis:
        F^-1 * PTDF * (SUM(max_res_gen + max_disp_gen) - min(demand))

    Where:
        - F: Line capacity (MW)
        - PTDF: Power Transfer Distribution Factor matrix
        - max_res_gen: Installed renewable capacity
        - max_disp_gen: Dispatchable generation capacity
        - demand: Nodal demand

    Configuration can be provided at:
        - Instantiation time (via `config` parameter in __init__)
        - Partition time (via `config` or individual parameters in partition())
    """

    SUPPORTED_ALGORITHMS = ["kmeans", "kmedoids", "hierarchical"]

    # Edge types that participate in AC power flow
    AC_EDGE_TYPES = {EdgeType.LINE.value, EdgeType.TRAFO.value}

    # Config parameter names for runtime override detection
    _CONFIG_PARAMS = {
        "zero_reactance_replacement",
        "regularization_factor",
        "infinite_distance",
    }

    def __init__(
        self,
        algorithm: str = "kmeans",
        slack_bus: Any | None = None,
        ac_island_attr: str = "ac_island",
        config: InputDataCongestionSensitiveConfig | None = None,
    ):
        """
        Initialize the partitioning strategy.

        Parameters
        ----------
        algorithm : str, default='kmeans'
            Clustering algorithm ('kmeans', 'kmedoids').
        slack_bus : Any, optional
            Specific node to use as slack bus, or None for auto-selection.
        ac_island_attr : str, default='ac_island'
            Node attribute name containing AC island ID.
        config : InputDataCongestionSensitiveConfig, optional
            Configuration parameters.
        """
        self.algorithm = algorithm
        self.slack_bus = slack_bus
        self.ac_island_attr = ac_island_attr
        self.config = config or InputDataCongestionSensitiveConfig()

        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm: {algorithm}. "
                f"Supported: {', '.join(self.SUPPORTED_ALGORITHMS)}"
            )

        log_debug(
            f"Initialized InputDataCongestionSensitivePartitioning: algorithm={algorithm}",
            LogCategory.PARTITIONING,
        )

    @property
    def required_attributes(self) -> dict[str, list[str]]:
        """Required attributes for congestion sensitive partitioning."""
        return {
            "nodes": ["renewable_capacity", "dispatchable_capacity", "min_demand"],
            "edges": ["x", "capacity"],
        }

    def _get_strategy_name(self) -> str:
        """Get descriptive strategy name for error messages."""
        return f"input_data_congestion_sensitive_{self.algorithm}"

    @with_runtime_config(InputDataCongestionSensitiveConfig, _CONFIG_PARAMS)
    def partition(self, graph: nx.DiGraph, **kwargs) -> dict[int, list[Any]]:
        """
        Partition nodes based on congestion sensitivity.

        Parameters
        ----------
        graph : nx.DiGraph
            NetworkX DiGraph with required attributes.
        **kwargs : dict
            Additional parameters:
            - n_clusters : Number of clusters (required)
            - random_state : Random seed
            - max_iter : Maximum iterations for clustering

        Returns
        -------
        dict[int, list[Any]]
            Dictionary mapping cluster_id -> list of node_ids.
        """
        try:
            # Placeholder for implementation
            n_clusters = kwargs.get("n_clusters")

            log_info(
                f"Starting input data congestion sensitive partitioning: {self.algorithm}, "
                f"n_clusters={n_clusters}",
                LogCategory.PARTITIONING,
            )

            if n_clusters is None or n_clusters <= 0:
                raise PartitioningError(
                    "Partitioning requires a positive 'n_clusters' parameter.",
                    strategy=self._get_strategy_name(),
                )

            nodes = list(graph.nodes())
            n_nodes = len(nodes)

            # Implementation will go here...
            # 1. Validate island attributes
            # 2. Calculate distance matrix based on the formula
            # 3. Perform clustering

            # For now, return a dummy partition or raise error
            raise NotImplementedError(
                "Congestion sensitive partitioning logic is not yet implemented."
            )

        except Exception as e:
            if isinstance(e, (PartitioningError, ValidationError, NotImplementedError)):
                raise
            raise PartitioningError(
                f"Congestion sensitive partitioning failed: {e}",
                strategy=self._get_strategy_name(),
            ) from e
