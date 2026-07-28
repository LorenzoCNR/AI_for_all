Quickstart
==========

Run the essential circular experiment:

.. code-block:: console

   python notebooks/experiment_01_circular_3d.py

This command performs the complete controlled pipeline:

1. generate 160 trials of a three-dimensional circular task latent;
2. map the latent into a 100-neuron population;
3. emit sparse spike counts;
4. construct one centered 10-bin window per trial time;
5. fit PCA and CNN1D;
6. compare both embeddings with the known latent;
7. save models, figures, metrics, and arrays.

Artifacts are written to:

.. code-block:: text

   outputs/experiment_01_circular_3d/
   |-- figures/
   |-- models/
   `-- results.joblib

``outputs`` is intentionally ignored by Git because all artifacts can be
regenerated from the experiment configuration.
