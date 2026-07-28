Demo notebooks
==============

The experiment files use ``# %%`` cells. They can be executed as normal Python
scripts or interactively in VS Code.

Circular task
-------------

``notebooks/experiment_01_circular_3d.py``
   Essential position and progress latent.

``notebooks/experiment_02_circular_5d.py``
   Adds velocity and trial context.

Linear track
------------

``notebooks/experiment_03_linear_2d.py``
   Position, direction, and localized place fields.

``notebooks/experiment_04_linear_4d.py``
   Adds velocity and context while retaining place fields.

Multisubject experiment
-----------------------

``notebooks/First_experiment_24_07.py`` generates the historical multisubject
simulation. ``notebooks/first_experiment_model_24_07.py`` fits its baseline
encoders and preserves that experiment as a separate reproducibility record.
