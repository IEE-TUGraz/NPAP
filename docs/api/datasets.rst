Datasets
========

.. currentmodule:: npap.datasets

Example datasets used throughout the documentation. Nothing is shipped with the
package: data is downloaded on first use and cached locally, so every example
runs after a plain ``pip install npap``.

.. code-block:: python

   import npap
   from npap.datasets import fetch_pypsa_eur

   files = fetch_pypsa_eur()          # downloads once, then reads from cache

   manager = npap.PartitionAggregatorManager()
   graph = manager.load_data(
       strategy="csv_files",
       node_file=str(files["buses.csv"]),
       edge_file=str(files["lines.csv"]),
   )

Fetching
--------

.. autofunction:: fetch_pypsa_eur

Cache management
----------------

.. autofunction:: get_data_home

.. autofunction:: clear_data_home

The cache location is resolved in this order:

1. the ``data_dir`` argument, when given;
2. the ``NPAP_DATA_HOME`` environment variable;
3. a per-user cache directory — ``$XDG_CACHE_HOME/npap`` or ``~/.cache/npap``
   on Linux and macOS, under ``%LOCALAPPDATA%`` on Windows.

.. note::

   Cached files **never expire on their own**. They stay on disk until you
   remove them, either with :func:`clear_data_home` or by deleting the
   directory. The full PyPSA-Eur network is roughly 20 MB.

   Because the Zenodo record is pinned in the source, the cache always matches
   the dataset version NPAP expects, which keeps examples reproducible. If that
   pin is ever bumped to a newer record, call :func:`clear_data_home` once so
   the new files are fetched instead of the stale ones.

   To keep the data somewhere else — a scratch disk, or a shared location on a
   cluster — set ``NPAP_DATA_HOME`` or pass ``data_dir`` explicitly.

Data source and licence
-----------------------

The network is the prebuilt PyPSA-Eur grid derived from OpenStreetMap:

    Xiong, B., Fioriti, D., Neumann, F., Riepin, I., & Brown, T. (2026).
    *Prebuilt electricity network for PyPSA-Eur based on OpenStreetMap data.*
    Zenodo. https://doi.org/10.5281/ZENODO.18619025

    Xiong, B., Fioriti, D., Neumann, F., Riepin, I., & Brown, T. (2025).
    *Modelling the high-voltage grid using open data for Europe and beyond.*
    Scientific Data, 12(1). https://doi.org/10.1038/s41597-025-04550-7

The underlying data comes from OpenStreetMap and is licensed under the
`Open Database License <https://opendatacommons.org/licenses/odbl/>`_. Credit
OpenStreetMap contributors and cite the references above when using it.
