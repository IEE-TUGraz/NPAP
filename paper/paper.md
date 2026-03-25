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
    orcid: 0009-0000-3806-7946
    affiliation: "1, 2"
  - name: Benjamin Stöckl
    orcid: 0009-0005-6579-8169
    affiliation: "1, 2"
  - name: Yannick Werner
    orcid: 0000-0002-6674-805X
    affiliation: "1, 2"
  - name: Sonja Wogrin
    orcid: 0000-0002-3889-7197
    affiliation: "1, 2"
affiliations:
  - name: Institute of Electricity Economics and Energy Innovation (IEE), Graz University of Technology, Inffeldgasse 18, Graz, Austria
    index: 1
  - name: Research Center ENERGETIC, Graz University of Technology, Rechbauerstraße 12, Graz, Austria
    index: 2
date: 24 February 2026
bibliography: paper.bib
---

<!-- TODO: choose dependency references to include.
     Spatial aggregation in energy systems:
     - Hoffmann et al. (2022), "Advanced Spatial and Technological Aggregation
       Scheme for Energy System Models", Energies
-->

# Summary

NPAP (Network Partitioning and Aggregation Package) is an open-source Python library for reducing
the spatial complexity of network graphs, with a special focus on physical networks, such as power
grids. Built on NetworkX [@NetworkX], it provides an accessible standalone package that can
be readily integrated with other software and frameworks that work with graph-based structures.
Instead of treating the spatial reduction process as a single action, NPAP explicitly splits it into
two distinct steps: *partitioning*, which assigns vertices (nodes) to groups (clusters), and
*aggregation*, which reduces the network based on a given assignment. NPAP's strategy pattern
architecture allows users to use and register custom partitioning and aggregation strategies
seamlessly without modifying the core code. Currently, NPAP provides 14 different partitioning
strategies combining clustering algorithms such as k-means [], DBSCAN [], and hierarchical
clustering [] with geographical or electrical node distance measures, as well as two pre-defined
aggregation profiles. Although initially developed with a focus on power systems, its architecture
is general-purpose and applicable to any network graph.

# Statement of need

Over the past decades, real-world electricity grids have significantly grown in space and
complexity, potentially covering up to many thousand nodes and branches. As a consequence, energy
system optimization models that represent those networks have become computationally intractable
and very challenging to solve. To regain tractability and computational efficiency, modelers
frequently reduce the model complexity by applying temporal and spatial aggregation
techniques [@Kotzur2021].

Today, temporal complexity reduction, known as time series aggregation, is a well-established field
with extensive research being carried out on different algorithms and how they
impact the outcomes of energy system optimization models [@Kotzur2018;@Hoffmann2020]
[@Teichgraeber2022;@Wogrin2023].
Tools like tsam (Time Series Aggregation Module) [@Hoffmann2022], have consolidated multiple
temporal aggregation algorithms into a single, reusable Python library
that has become a standard component in many energy system optimization modeling frameworks, such as
PyPSA (Python for Power System Analysis) [@Brown2018a]. Before tsam, modelers implemented these
methods individually and ad-hoc for each project, challenging reusability and comparability.

This is exactly where the research on spatial complexity reduction is at: While there is
some research on network reduction [] and aggregation [] methods available, there exists no
standalone tool that brings all methods together and is easy to use, extend, and does not rely
on framework-specific data-structures.

NPAP fills this gap as a standalone, pip-installable, and extensible package that works with any
NetworkX [@NetworkX] graph. The target audience includes power systems researchers, energy modelers,
network analysts, and more broadly, anyone working with graph-based spatial structures who is
interested in reducing spatial complexity.

# State of the field

Within the energy system optimization community, existing implementations of network partitioning and
aggregation methods are usually tightly rooted to the respective frameworks internal data structure
and cannot be reused independently or by other tools and frameworks.
The built-in spatial clustering module of PyPSA [@Brown2018a], for example, provides busmap-based
spatial aggregation using methods such as k-means or hierarchical clustering on geographical
coordinates of buses, based on its `Network` object and `buses` DataFrame.
In ETHOS.FINE [@Kluetz2025]...

Researches working with other frameworks or custom network representations cannot leverage this
code without significant adaptation to the core code structure.

<!-- TODO: Research which additional packages to include here.
     - FZJ spatial aggregation (Hoffmann et al., 2022): algorithmic contribution in a paper,
       but not released as a reusable open-source library
     - pandapower: power system analysis tool without a dedicated partitioning/aggregation module
     - PowerModels.jl: Julia-based power system optimization
     - Other spatial clustering tools in the energy domain
-->

The analogy with temporal complexity reduction and the tsam package [@Hoffmann2022] is instructive.
TSAM addressed a similar fragmentation problem in the temporal dimension: before its release,
researchers implemented time series aggregation methods independently for each project,
leading to duplicated effort and inconsistent implementations. TSAM consolidated these methods into
a single reusable library that became a standard tool in the ecosystem.

NPAP carries this idea to the spatial dimension. Its unique contributions compared to existing
alternatives are: (1) it is a standalone, pip-installable library decoupled from any specific
framework; (2) it employs a strategy pattern architecture that allows new functionalities to be
added without modifying core code; (3) it makes an explicit conceptual separation between
partitioning and aggregation that gives researchers full control over and usability of each stage of
the reduction pipeline; and (4) it provides voltage-aware and AC-island-aware partitioning for
multi-voltage networks.

# Pipeline architecture

![NPAP pipeline architecture. The full workflow proceeds from data loading through optional pre-processing, partitioning, aggregation, and visualization. Blue regions represent the core pipeline, green regions indicate optional steps, yellow boxes show intermediate data structures, and the green output boxes show final outcomes.\label{fig:pipeline}](figures/pipeline-architecture-JOSS.svg)

The full pipeline of NPAP is shown in Figure \ref{fig:pipeline}. Before the actual network
partition and aggregation process, NPAP performs two stages preparing the network graph. In the first
stage data is loaded and a NetworkX graph used for the reduction process is created and validated.In
the second stage, NPAP provides optional pre-processing steps that prepare the
network graph for partitioning, such as the aggregation of parallel edges.

A key design decision in NPAP is the explicit separation of the network reduction process into
two stages, as shown in Figure \ref{fig:pipeline}:

1. **Partitioning**: A partitioning strategy applies, e.g., a machine learning clustering algorithm
   to the network and produces a mapping of each original node to its assigned cluster. This step
   only determines group "membership", it does not modify the graph itself.

2. **Aggregation**: The network topology is reduced based on a partitioning result, i.e., a
   node-to-cluster mapping, by aggregating nodes and edges and their associated properties according
   to user-specified rules.

This separation was identified early in the design process as a fundamental distinction. It gives
users fine-grained control: they can swap partitioning algorithms independently of the aggregation
method, apply different aggregation strategies to the same partitioning result, or compose custom
pipelines that combine strategies from different domains. Ofcourse it is also possible for users
to just use one of them. We now explain the partitioning and aggregation steps in more detail.

The original network as well as the partitioned and aggregated one can be illustrated using the
visualization component, potentially including different voltage levels or direct current (DC)
links.

# Software design

![xxx Label.\label{fig:design}](figures/class-facade-registry-JOSS.svg)
A broad overview of the design of NPAP is shown in Figure \ref{fig:design}. The whole workflow
is orchestrated and accessed by the PartionAggregatorManager. NPAP follows a strategy
pattern with four categories — data loading, partitioning, topology aggregation, and property
aggregation — all orchestrated by three manager classes, shown in the bottom of the Figure.
New strategies inherit from abstract base classes and register with their respective managers,
enabling users to seamlessly add custom strategies without modifying core code. This also
facilitates an easy integration of NPAP into existing energy system modeling frameworks, without
modifying their core code, as we showcase for the PyPSA framework in Section xx.
The whole library is built on well-established python packages, such as NetworkX [@NetworkX] for
graph representation, NumPy [@Harris2020], SciPy [@Virtanen2020], and scikit-learn [@scikit-learn]
for numerical computation, and Plotly [@plotly] for interactive map-based visualization of results,
with the idea of reducing entry barriers for contributing to NPAP. In the following sections, we
introduce the loading and pre-processing of input data, partionining, and the aggregation processes
in more detail.

## Data loading and pre-processing

As NPAP is designed to work with NetworkX graphs, they can be directly passed as input data,
facilitating an easy integration into other framework or software. Apart from that, NPAP currently
supports importing data from CSV files with two main strategies. The first one works with general
node and edge data to create general graph structures. The second one is a domain-dependent strategy
focused on power grids and includes buses, lines, transformers, converters, and DC links.
User-specific data loading strategies can be registered straightforward as shown in
Figure \ref{fig:design}. After loading the data, optional pre-processing steps are carried out by
the PartitionAggregatorManager, such as the aggregation of parallel edges. For power grids,
voltage-level grouping separates the network into independent sub-graphs per voltage level, enabling
voltage-aware partitioning strategies. These pre-processing steps are orchestrated automatically
when nodes (buses) with voltage level attributes, lines, and transformers are loaded.

## Partitioning strategies

NPAP currently provides four families of partitioning strategies combining geographical and
electrical node distance with and without voltage-awareness, which partitions voltage levels
independently. Each family supports 14 different partitioning strategies, such as k-means [],
k-medoids [], DBSCAN [], HDBSCAN [], and hierarchical clustering [] with multiple linkage
methods, via scikit-learn [@scikit-learn] and specialized libraries, such as KMedoids [].
Geographical distance partitioning leverages geographical node coordinates (latitude and longitude)
and is suitable for any geo-referenced network. Electrical distance partitioning computes Power
Transfer Distribution Factors (PTDFs) [] to capture the electrical behavior of an electrical network rather
than its geographical topology.

For power systems in particular, NPAP automatically detects alternating current (AC) islands of the
network that are linked solely through DC interconnections. To preserve the electrical properties of
the grid during reduction, these AC islands are then partitioned independently. This is done by
setting the distance matrix entries of nodes in different AC islands to infinity. The same approach
is followed for nodes located on different voltage levels in the voltage-aware partitioning strategies.
Both approaches are algorithm-agnostic, i.e, they work with any distance-based partitioning method
without requiring modifications to the algorithm itself.

The outcome of the partitioning is a mapping, that assigns nodes of the original network to clusters,
and it is stored in the PartitionResult along with other information, as shown in
Figure \ref{fig:design}.

# Three-tier aggregation strategy pipeline

The aggregation process is decomposed into three sequential steps. The *topology tier* builds a new
network graph based on the mapping result of the partitioning process by creating a representative
node for each cluster and adjusting the edges between them accordingly. Afterwards, to cover use
cases we do not foresee and usability outside the power system domain, we have included an optional
*domain-specific tier*. There, users can modify physical properties of the network graph, e.g.,
adding new edges that have not existed in the original graph or explicitly modifying certain
properties, such as line reactanes. In the *property aggregation tier*, user-specified functions,
e.g, sum, average or equivalent reactances are then applied to the remaining node and edge
properties. These property aggregations strategies can be configured by the user in detail through
the aggregation profiles shown in Figure \ref{fig:design}. At the moment, NPAP provides two
pre-defined aggregation profiles.

# Research impact statement

NPAP is designed as a standalone python package for the partitioning and aggregation of network
graphs. The library is pip-installable, fully documented on ReadTheDocs, and includes an automated
test suite with continuous integration across Python 3.10, 3.11, and 3.12. It has been tested on
small-scale as well as large-scale network graphs with up to a few thousand nodes.

One of the potential applications, which we also had in mind during the creation
process, is to be used in energy system modeling frameworks. For illustration purposes, we have
integrated it into PyPSA-Eur [@Hoersch2018], one of the most utilized frameworks, through an open pull
request that introduces NPAP as an alternative to PyPSA-Eur's spatial clustering backend.[^1] This
integration demonstrates NPAP's framework-agnostic design, based on standard NetworkX graphs,
and shows how it can be integrated in existing frameworks with minimal effort and without
significantly changing the core code. The pull request provides three public API entry points for
PyPSA users: partitioning only, aggregation only, and a full clustering pipeline, along with 53+
tests covering both unit and integration levels.

[^1]: https://github.com/PyPSA/PyPSA/pull/1568

Recently, we have applied the NPAP implementation in PyPSA-Eur [@Hoersch2018] for the partitioning and
aggregation of the full pan-European transmission network [@Xiong2025], which, in its current
state [@Xiong2026], contains around 6800 nodes and 17500 edges, demonstrating its scalability. In
particular, we have used NPAP's voltage-aware partitioning strategy with geographical distance and
k-means clustering [] to analyze the infrastructure investment for power lines on different
voltage levels and for transformers required for the energy transition using PyPSA [@Brown2018a].
The work has been submitted to the International Conference on the European Energy Market 2026.

<!-- Similar to how TSAM consolidated temporal aggregation methods into a reusable Python library for
energy system modeling, NPAP standardizes spatial network partitioning and aggregation within a
single, framework-agnostic package. Although initially developed with a focus on power systems,
its architecture is general-purpose and applicable to any network representable
as a NetworkX graph. -->

Active development directions include the extension of available partitioning
strategies, e.g., the Adjacent Node Agglomerative Clustering (ANAC) algorithm [@Stoeckl2025a],
and aggregation strategies, e.g., based on PTDFs [@Fortenbacher2018], as well as the extension
to other physical infrastructures for different energy carriers, such as hydrogen, which require
distinct partitioning and aggregation strategies. Future extensions include, among other, a filtering
module that would allow users to select subsets of the network, e.g., filtering by country or
region, as an additional pre-processing step before partitioning.

# AI Usage Disclosure

Several artificial intelligence (AI) tools were used during the development of NPAP. Google Gemini
(Pro) was used for non-code-related tasks, such as learning about power system. Anthropic Claude
(Sonnet 4.5, Opus 4.5, and 4.6), accessed through both, the desktop application and the Claude Code
terminal interface, was used for code-related activities, including software design brainstorming,
implementation support, test coverage or writing the documentation. Grammarly was used for
No other code-generation tools (such as GitHub Copilot, Codex, or Cursor) were used. All
AI-generated output was carefully reviewed, tested, and frequently modified before inclusion. The
authors take full responseability for the code.

# Acknowledgements

The work done on the package by the contributors of the Institute of Electricity Economics and Energy
Innovation (IEE) at Graz University of Technology was funded by the European Union
(ERC, NetZero-Opt, 101116212). Views and opinions expressed are however those of the author(s) only
and do not necessarily reflect those of the European Union or the European Research Council. Neither
the European Union nor the granting authority can be held responsible for them.

# References
