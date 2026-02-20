import networkx as nx

from npap.interfaces import AggregationMode, AggregationProfile, PhysicalAggregationStrategy
from npap.managers import AggregationManager


class RecordingPhysicalStrategy(PhysicalAggregationStrategy):
    required_properties = []
    modifies_properties = []

    def __init__(self):
        self.received = {}

    @property
    def can_create_edges(self) -> bool:
        return True

    @property
    def required_topology(self) -> str:
        return "simple"

    def aggregate(
        self,
        original_graph,
        partition_map,
        topology_graph,
        properties,
        parameters=None,
    ):
        self.received["node_to_cluster"] = parameters.get("node_to_cluster")
        self.received["cluster_edge_map"] = parameters.get("cluster_edge_map")
        return topology_graph


def test_aggregation_manager_passes_physical_parameters():
    manager = AggregationManager()
    strategy = RecordingPhysicalStrategy()
    manager.register_physical_strategy("recording", strategy)

    profile = AggregationProfile(
        mode=AggregationMode.CUSTOM,
        topology_strategy="simple",
        physical_strategy="recording",
        physical_properties=[],
        node_properties={},
        edge_properties={},
        default_node_strategy="sum",
        default_edge_strategy="sum",
        warn_on_defaults=False,
    )

    graph = nx.DiGraph()
    graph.add_node("A")
    graph.add_node("B")
    graph.add_edge("A", "B")
    partition_map = {0: ["A"], 1: ["B"]}

    manager.aggregate(graph, partition_map, profile)

    assert strategy.received["node_to_cluster"]
    assert strategy.received["cluster_edge_map"]
