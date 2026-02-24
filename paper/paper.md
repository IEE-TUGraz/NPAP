---
title: 'NPAP: Network Partitioning and Aggregation Python Package for Graph-Based Structures with a focus in Power Systems'
tags:
  - Python
  - network partitioning
  - spatial aggregation
  - power systems
  - clustering
  - energy system modeling
  - NetworkX
authors:
  - name: Marco Anarmo
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Benjamin Stöckl
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Yannick Werner
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Sonja Wogrin
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Institute of Electricity Economics and Energy Innovation, Graz University of Technology, Graz, Austria
    index: 1
date: 24 February 2026
bibliography: paper.bib
---

<!-- TODO: choose dependency references to include.
     Core dependencies:
     - NetworkX (Hagberg et al., 2008) — graph representation foundation
     - scikit-learn (Pedregosa et al., 2011) — clustering algorithms backend

     Temporal aggregation (TSAM analogy):
     - Kotzur et al. (2018), "Impact of different time series aggregation methods
       on optimal energy system design", Renewable Energy
     - Hoffmann et al. (2020), "A Review on Time Series Aggregation Methods for
       Energy System Models", Energies

     Energy system frameworks:
     - Brown et al. (2018), "PyPSA: Python for Power System Analysis", JORS
     - Hörsch et al. (2018), "PyPSA-Eur: An Open Optimisation Model of the
       European Transmission System", Energy Strategy Reviews

     Spatial aggregation in energy systems:
     - Hoffmann et al. (2022), "Advanced Spatial and Technological Aggregation
       Scheme for Energy System Models", Energies
-->

# Summary

NPAP (Network Partitioning and Aggregation Package) is an open-source Python library for
partitioning and aggregating spatial network graphs. Built on NetworkX, it is
designed to work with any graph-based structure while providing specialized support for electrical
power systems.

The process commonly referred to as network "clustering" actually involves two distinct steps that
NPAP makes explicit in its pipeline: *partitioning*, which assigns nodes to regions (clusters) using a
clustering algorithm, and *aggregation*, which reduces the network topology based on that
assignment. This separation is central to the library's design and gives users fine-grained control
over each stage independently.

NPAP provides over 14 pre-registered clustering algorithms including k-means, k-medoids, DBSCAN,
HDBSCAN, and hierarchical clustering applicable to both geographical and electrical distance-based
partitioning. It supports voltage-aware and AC-island-aware clustering for multi-voltage power
networks with DC interconnections. Its strategy pattern architecture allows users to register custom
partitioning and aggregation strategies without modifying the core code.

Analogous to how TSAM consolidated temporal aggregation methods into a single reusable Python
library for energy system modeling, NPAP standardizes spatial network aggregation
into one framework-agnostic package. Although developed with a focus on power systems due to the
funding context, the architecture is general-purpose and applicable to any network representable as
a NetworkX graph.

# Statement of Need

Energy system models have grown significantly in spatial complexity. Modern transmission network
representations can contain thousands of nodes and branches, and running mathematical optimization
problems on such large-scale networks is computationally expensive in both time and resources. To
make these models tractable, researchers reduce their complexity mainly along two dimensions: temporal and
spatial.

Temporal complexity reduction is a well-established field. Extensive literature exists on time series
aggregation methods for energy system models, and tools like TSAM have consolidated multiple
temporal aggregation algorithms into a single, reusable Python library
that has become a standard component in the energy modeling ecosystem. Before TSAM, researchers
implemented these methods individually and ad-hoc for each project.

Spatial complexity reduction, however, lacks an equivalent standalone tool. While there is
significant research on network reduction and bus aggregation methods, the algorithms remain
scattered across individual publications or embedded within specific modeling frameworks. Existing
implementations, such as PyPSA's built-in spatial clustering module, are tightly coupled to
their framework's internal data structures and cannot be reused independently by other tools or
researchers working with different systems.

NPAP fills this gap as a standalone, pip-installable, and extensible package that works with any
NetworkX graph. It has been tested on networks ranging from single-country grids to pan-European
transmission structures like PyPSA-Eur. The target audience includes power systems
researchers, energy modelers, network analysts, and more broadly, anyone working with graph-based
spatial structures that need to be reduced in complexity.

# State of the Field

The most widely used spatial clustering implementation in the open-source energy modeling community
is PyPSA's built-in clustering module. It provides busmap-based spatial aggregation
using methods such as k-means or hierarchical clustering on bus coordinates. However, this
implementation is tightly coupled to PyPSA's internal data structures, specifically its `Network`
object and `buses` DataFrame, making it impractical to reuse outside PyPSA workflows. Researchers
working with other frameworks or custom network representations cannot leverage this code without
significant adaptation.

<!-- TODO: Research which additional packages to include here.
     - FZJ spatial aggregation (Hoffmann et al., 2022): algorithmic contribution in a paper,
       but not released as a reusable open-source library
     - pandapower: power system analysis tool without a dedicated partitioning/aggregation module
     - PowerModels.jl: Julia-based power system optimization
     - Other spatial clustering tools in the energy domain
-->

The analogy with temporal aggregation is instructive. TSAM addressed a similar
fragmentation problem in the temporal dimension: before its release, researchers implemented
time series aggregation methods independently for each project, leading to duplicated effort
and inconsistent implementations. TSAM consolidated these methods (k-means, k-medoids,
hierarchical clustering, and others) into a single reusable library that became a standard tool
in the ecosystem.

NPAP brings this same approach to the spatial dimension. Its unique contributions compared to
existing alternatives are: (1) it is a standalone, pip-installable library decoupled from any
specific framework; (2) it employs a strategy pattern architecture that allows new functionalities to be
added without modifying core code; (3) it provides voltage-aware and AC-island-aware partitioning
for multi-voltage networks; and (4) it makes an explicit conceptual separation between partitioning
and aggregation that no existing tool provides, giving researchers full control over each stage of
the reduction pipeline.

# Software Design

![NPAP pipeline architecture. The full workflow proceeds from data loading through optional pre-processing, partitioning, aggregation, and visualization. Blue regions represent the core pipeline, green regions indicate optional steps, yellow boxes show intermediate data structures, and the green output boxes show final outcomes.\label{fig:pipeline}](figures/npap-pipeline.png)

## Partitioning and Aggregation as Distinct Steps

A key design decision in NPAP is the explicit separation of the network reduction process into
two stages, as shown in \autoref{fig:pipeline}. The process commonly referred to as "clustering" in
the energy systems literature actually involves two fundamentally different operations that are
applicable to any graph-based structure:

1. **Partitioning**: a partitioning strategy applies a machine learning clustering algorithm to the
   network and produces a mapping of each original node to its assigned region or cluster. This step
   only determines group "membership", it does not modify the graph itself.

2. **Aggregation**: the network topology is reduced based on the partitioning result. Nodes within
   each cluster are merged, edges are adjusted, and properties are combined according to
   user-specified rules.

This separation was identified early in the design process as a fundamental distinction. It gives
users fine-grained control: they can swap partitioning algorithms independently of the aggregation
method, apply different aggregation strategies to the same partitioning result, or compose custom
pipelines that combine strategies from different domains.

## Data Pre-processing

Before partitioning, NPAP provides optional pre-processing steps that prepare the network graph.
Parallel edge aggregation merges multiple edges between the same pair of nodes into a single edge,
simplifying the graph topology while preserving aggregate properties. Voltage-level grouping
separates the network into independent sub-graphs per voltage level, enabling voltage-aware
partitioning strategies. These pre-processing steps are orchestrated automatically by the pipeline
when voltage-aware data is loaded. Future extensions include a filtering module that would allow
users to select subsets of the network, for example, filtering by country or region, as an
additional pre-processing step before partitioning.

## Partitioning Strategies

NPAP provides four families of partitioning strategies, each supporting over 14 pre-registered
clustering algorithms including k-means, k-medoids, DBSCAN, HDBSCAN, and hierarchical clustering
with multiple linkage methods — via scikit-learn and specialized libraries.
Geographical distance partitioning uses node coordinates (latitude and longitude) and is suitable for
any geo-referenced network. Electrical distance partitioning computes Power Transfer Distribution
Factors (PTDF) to capture the electrical behavior of the network rather than its physical layout.
Both strategies have voltage-aware variants that partition independently per voltage level.

When networks contain DC interconnections, NPAP automatically detects AC islands and enforces
island boundaries during partitioning. It does so by setting the distance between nodes in different
AC islands to infinity in the distance matrix, ensuring that no clustering algorithm will group nodes
from separate islands into the same cluster. The same mechanism applies to voltage-aware strategies:
distances between nodes at different voltage levels are set to infinity, so that partitioning
respects voltage-level boundaries. This approach is algorithm-agnostic — it works with any
distance-based clustering method without requiring modifications to the algorithm itself.

## Three-Tier Aggregation Pipeline

The aggregation process is decomposed into three sequential steps. *Topology creation* maps nodes to
their assigned clusters and adjusts edges accordingly, producing the reduced graph structure.
*Physical aggregation* is an optional layer that preserves certain physical properties — for example,
Kron reduction creates new edges to maintain electrical equivalence. This layer exists because some
reduction methods require modifications to the graph topology beyond simple node merging.
*Statistical aggregation* applies user-specified functions (sum, average, first, equivalent
reactance, among others) to the remaining node and edge properties, configured through aggregation
profiles or pre-defined aggregation profiles called in the NPAP context aggregation modes.

## Architecture

NPAP follows a strategy pattern with four categories — data loading, partitioning, topology
aggregation, and property aggregation — all orchestrated by manager classes. The data flow proceeds
through optional pre-processing steps (parallel edge aggregation, voltage-level grouping) before
entering the partition-then-aggregate pipeline. New strategies inherit from abstract base classes and
register with their respective managers, allowing users to add custom algorithms without modifying
core code. The library is built on NetworkX for graph representation, NumPy and SciPy for numerical
computation, and Plotly for interactive map-based visualization of results.

# Research Impact Statement

NPAP was developed within the ERC-funded NetZero-Opt project at the Institute of Electricity Economics and Energy
Innovation (IEE) at Graz University of Technology. It is currently being integrated into PyPSA, the most widely
used open-source energy system modeling framework, through an open pull request that introduces NPAP as an alternative
spatial clustering backend.[^1] This integration demonstrates that NPAP's framework-agnostic design,
based on standard NetworkX graphs, allows it to be adopted by existing tools with minimal effort.
The pull request provides three public API entry points for PyPSA users: partitioning only,
aggregation only, and a full clustering pipeline, along with 53+ tests covering both unit and
integration levels.

[^1]: https://github.com/PyPSA/PyPSA/pull/1568

NPAP has been tested on networks of varying scale, from single-country grids to the full PyPSA-Eur
pan-European transmission network, demonstrating its scalability. The library is
pip-installable, fully documented on ReadTheDocs, and includes an automated test suite with
continuous integration across Python 3.10, 3.11, and 3.12.

# Future Steps

Active development directions include dual-based partitioning algorithms using Locational Marginal
Prices (LMP), the ANAC (Adjacent Node Agglomerative Clustering) algorithm, Kron reduction and
PTDF-based equivalent reactance aggregation strategies, gas network support for multi-energy
systems, and filtering modules for pre-processing steps such as country or region selection.

# AI Usage Disclosure

<!-- Required by JOSS -->

# Acknowledgements

The work in this package by the contributors at the Institute of Electricity Economics and Energy
Innovation (IEE) was funded by the European Union (ERC, NetZero-Opt, 101116212). Views and opinions
expressed are however those of the author(s) only and do not necessarily reflect those of the
European Union or the European Research Council. Neither the European Union nor the granting
authority can be held responsible for them.

# References
