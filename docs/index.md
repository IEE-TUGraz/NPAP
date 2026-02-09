---
title: Home
sd_hide_title: true
---

# Home

```{toctree}
:hidden:
:maxdepth: 2

user-guide/index
contributing
api/index
```

<div class="landing-page">

<div class="hero-section">

```{image} assets/NPAP-Banner-dark.svg
:alt: NPAP
:class: only-light hero-banner
:align: center
```

```{image} assets/NPAP-Banner-light.svg
:alt: NPAP
:class: only-dark hero-banner
:align: center
```

<p class="hero-description">
A Python library for <strong>partitioning</strong> and <strong>aggregation</strong> of spatial network graphs, with a special focus in electrical power systems
</p>

<div class="hero-buttons">

```{button-ref} user-guide/index
:color: primary
:class: hero-btn hero-btn-primary

Get Started
```

```{button-link} https://github.com/IEE-TUGraz/NPAP
:color: secondary
:class: hero-btn hero-btn-secondary

GitHub Repository
```
<div>
<div>

<div class="code-block-wrapper">
<div class="code-block-header">
<span class="terminal-dot red"></span>
<span class="terminal-dot yellow"></span>
<span class="terminal-dot green"></span>
<span class="code-block-title">Quick Start ✨</span>
</div>
<div class="code-block-body">

```python
import npap
from npap import AggregationMode

# 1. Initialize the main manager
manager = npap.PartitionAggregatorManager()

# 2. Load data
manager.load_data(
    strategy="networkx_direct",
    graph=your_graph
)

# 2.1. Aggregate parallel edges (if applicable)
manager.aggregate_parallel_edges(
    edge_properties={"p_max": "average", "x": "equivalent_reactance"},
    default_strategy="average",
    warn_on_defaults=False,
)

# 3. Create partition result
partition_result = manager.partition(
    strategy="geographical_kmeans", n_clusters=250
)

# 3.1. Plot the partitioned network
manager.plot_network(
    style="clustered", title="Partitioned Network"
)

# 4. Aggregate the network based on a given AggregationMode
aggregated_network = manager.aggregate(mode=AggregationMode.GEOGRAPHICAL)
```

</div>
</div>

</div>

</div>
