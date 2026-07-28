NeuroBridge documentation
=========================

NeuroBridge is a research package for controlled neural time-series
simulation and representation learning. It generates an interpretable latent
task process, maps that process into neural population activity, emits sparse
spike counts, and evaluates whether an encoder recovers the known geometry.

The package currently provides circular and linear motor-task simulations,
localized place fields, centered temporal windows, PCA and temporal neural
encoders, structured contrastive objectives, and latent-recovery metrics.

.. note::

   NeuroBridge is an active research prototype. Reference results are useful
   for reproducibility and debugging, but are not yet multi-seed held-out
   benchmarks.

Start here
----------

* :doc:`installation` explains editable and documentation installations.
* :doc:`quickstart` runs the first controlled experiment.
* :doc:`concepts` defines the generative and learning objects.
* :doc:`experiments` presents the four baseline experiments.
* :doc:`demos` links each executable notebook-style script.
* :doc:`api` is generated from the package docstrings.

.. toctree::
   :maxdepth: 2
   :hidden:

   installation
   quickstart
   concepts
   experiments
   demos
   api
   contributing
