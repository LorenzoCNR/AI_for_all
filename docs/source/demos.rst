Experiment notebooks
====================

The ``.ipynb`` files are the primary documented entry points. Each contains
explanatory Markdown and runnable cells. Matching ``.py`` mirrors are retained
for terminal execution.

The notebooks expose latent generation, neural mapping, spike emission,
windowing, trial splitting, PCA, soft-target construction, CNN training,
encoding, evaluation, and saving as separate cells. They do not call the
all-in-one experiment runner.

Circular task
-------------

``notebooks/experiment_01_circular_3d.ipynb``
   Essential position and progress latent.

``notebooks/experiment_02_circular_5d.ipynb``
   Adds velocity and trial context.

Linear track
------------

``notebooks/experiment_03_linear_position_direction.ipynb``
   Position, direction, and localized place fields.

``notebooks/experiment_04_linear_enriched.ipynb``
   Adds velocity and context while retaining place fields.
