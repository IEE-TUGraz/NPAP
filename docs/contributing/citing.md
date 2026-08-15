# Citing NPAP

If NPAP contributes to work that you publish, please cite it. Citations are what
make sustained maintenance of research software possible.

## Citing the software

Every release of NPAP is described by the
[`CITATION.cff`](https://github.com/IEE-TUGraz/NPAP/blob/main/CITATION.cff) file
in the repository root. GitHub renders it as a **"Cite this repository"** button
on the [project page](https://github.com/IEE-TUGraz/NPAP), which exports the
entry in both APA and BibTeX format.

Please cite the **specific version** you used rather than the project in
general, so that your results stay reproducible. You can find the version you
are running with:

```python
import npap

print(npap.__version__)
```

## Citing the paper

NPAP has been submitted to the [Journal of Open Source
Software](https://joss.theoj.org/); the review is
[open and in progress](https://github.com/openjournals/joss-reviews/issues/10557).

:::{note}
Once the paper is accepted, the DOI and the definitive BibTeX entry will be
published here and in `CITATION.cff`. Until then, please cite the software
release as described above.
:::

## Citing the underlying methods

NPAP builds directly on several scientific libraries. If your work depends on a
particular component, consider citing it as well:

| Component | Used for | Reference |
|-----------|----------|-----------|
| [NetworkX](https://networkx.org/) | Graph data structures and algorithms | Hagberg, Schult & Swart (2008) |
| [scikit-learn](https://scikit-learn.org/) | k-means, DBSCAN, hierarchical clustering | Pedregosa et al. (2011) |
| [SciPy](https://scipy.org/) | Sparse linear algebra and distance metrics | Virtanen et al. (2020) |
| [kmedoids](https://github.com/kno10/python-kmedoids) | FasterPAM k-medoids implementation | Schubert & Rousseeuw (2021) |
| [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan) | Density-based clustering | McInnes, Healy & Astels (2017) |

If your networks come from a published dataset — for example the PyPSA-Eur
network used in the [examples](../user-guide/index.md) — please cite that
dataset separately, following its own citation instructions.
