# Examples

Complete, runnable notebooks that take a real network through the whole NPAP
pipeline. Both use the European high-voltage grid — roughly 6,800 buses across
35 countries, derived from OpenStreetMap — which
{py:func}`~npap.datasets.fetch_pypsa_eur` downloads and caches on first use.

Everything here runs after a plain `pip install npap`; no manual data
preparation is needed.

```{toctree}
:hidden:
:maxdepth: 1

examples/getting_started
examples/european_network_pypsa
```

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Getting Started
:link: examples/getting_started
:link-type: doc

The whole pipeline in ten minutes, first on a plain graph and then with the
power-systems features enabled, so the difference between the two is immediate.

+++
[Open in GitHub](https://github.com/IEE-TUGraz/NPAP/blob/main/examples/getting_started.ipynb)
:::

:::{grid-item-card} European High-Voltage Network
:link: examples/european_network_pypsa
:link-type: doc

A deeper walkthrough of the same grid: voltage-level harmonisation, AC-island
handling, and per-edge-type aggregation with the physically correct strategy
for lines, transformers and DC links.

+++
[Open in GitHub](https://github.com/IEE-TUGraz/NPAP/blob/main/examples/european_network_pypsa.ipynb)
:::

::::

## Running them yourself

Clone the repository and launch Jupyter from the `examples/` directory:

```bash
git clone https://github.com/IEE-TUGraz/NPAP.git
cd NPAP
pip install -e ".[dev,test,docs]"
jupyter lab examples/
```

The notebooks are rendered here from their stored outputs, so the figures you
see are the ones produced by the last run against the current release.

:::{admonition} Maintainers: regenerating the stored outputs
:class: dropdown

After changing anything the notebooks exercise, re-run them and commit the
result. `PLOTLY_RENDERER` matters: without it Plotly stores only its own JSON
mime type, which the documentation builder cannot render, and the figures
silently disappear from these pages.

```bash
pip install -e ".[dev,docs]"

PLOTLY_RENDERER=notebook_connected jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 \
    examples/getting_started.ipynb examples/european_network_pypsa.ipynb
```
:::

:::{note}
The first call to {py:func}`~npap.datasets.fetch_pypsa_eur` downloads about
20 MB from [Zenodo](https://doi.org/10.5281/ZENODO.18619025) and caches it, so
subsequent runs start immediately. See {doc}`the datasets API <../api/datasets>`
for where the cache lives and how to clear it.
:::
