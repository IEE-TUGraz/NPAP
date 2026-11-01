Visualization
=============

.. currentmodule:: npap.visualization

Interactive network maps built on Plotly's MapLibre traces.

Every plotting entry point **returns** the :class:`plotly.graph_objects.Figure`
so that it can be post-processed, embedded or exported by the caller. Outside a
Jupyter kernel the figure is additionally written to a standalone HTML file;
inside a notebook nothing is written to disk, so the returned figure renders
inline. Displaying is opt-in through the ``renderer`` argument.

.. code-block:: python

   from npap import PartitionAggregatorManager

   manager = PartitionAggregatorManager()
   manager.load_data("networkx_direct", graph=graph)

   # Returns the figure and writes ./my_grid.html when run as a script
   fig = manager.plot_network(style="simple", title="My grid")

   # Open the interactive map in a browser tab (useful from an IDE)
   manager.plot_network(renderer="browser")

   # Choose where the export lands, or switch it off entirely
   manager.plot_network(output_dir="figures", filename="grid.html")
   manager.plot_network(save_html=False)

Quick plotting
--------------

.. autofunction:: plot_network

Figure output
-------------

.. autofunction:: deliver_figure

.. autofunction:: resolve_html_path

Configuration
-------------

.. autoclass:: PlotConfig
   :show-inheritance:
   :no-index:

.. autoclass:: PlotStyle
   :show-inheritance:
   :no-index:

Plotter
-------

.. autoclass:: NetworkPlotter
   :members:
   :show-inheritance:
   :no-index:
