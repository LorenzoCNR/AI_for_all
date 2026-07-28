Experiments
===========

Controlled matrix
-----------------

The baseline suite varies task family and latent dimensionality while keeping
the same number of trials, time bins, neurons, window size, and encoder family.

.. list-table::
   :header-rows: 1
   :widths: 24 12 34 30

   * - Experiment
     - Latent
     - Coordinates
     - Neural tuning
   * - Circular essential
     - 3D
     - X, Y, progress
     - direction, progress, mixed
   * - Circular enriched
     - 5D
     - + velocity, context
     - heterogeneous task loadings
   * - Linear essential
     - 2D
     - position, direction
     - + localized place fields
   * - Linear enriched
     - 4D
     - + velocity, context
     - mixed and place-selective units

Reference metrics
-----------------

The current in-sample single-seed RSA Spearman values are:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Task
     - PCA
     - CNN1D
   * - Circular 3D
     - 0.740
     - 0.545
   * - Circular 5D
     - 0.432
     - 0.471
   * - Linear 2D
     - 0.807
     - 0.794
   * - Linear 4D
     - 0.717
     - 0.719

.. important::

   These values establish reproducible reference behavior, not final
   generalization performance. Held-out trials, repeated seeds, objective
   ablations, and confidence intervals are still required.

.. image:: ../../site/assets/circular_3d.png
   :alt: PCA and CNN1D embeddings for the circular 3D experiment
   :width: 100%
