Quickstart
==========

Open the primary documented experiment:

.. code-block:: text

   notebooks/experiment_01_circular_3d.ipynb

Run its cells in order in VS Code or Jupyter. The notebook performs the
complete controlled pipeline without hiding it behind one experiment-wide
function:

1. generate 160 trials of a three-dimensional circular task latent;
2. inspect ``Z`` and the task-state variables;
3. construct ``B`` and inspect the neural tuning mixture;
4. calculate ``u``, ``lambda``, and stochastic counts ``X``;
5. construct one centered 10-bin window per trial time;
6. split complete trials into fitting and test sets;
7. fit PCA directly;
8. inspect one batch-wise soft target matrix;
9. construct and train CNN1D;
10. compare both embeddings with the known latent;
11. save models, figures, metrics, and arrays.

Artifacts are written to:

.. code-block:: text

   outputs/experiment_01_circular_3d/
   |-- figures/
   |-- models/
   `-- results.joblib

``outputs`` is intentionally ignored by Git because all artifacts can be
regenerated from the experiment configuration.

The equivalent non-interactive command is:

.. code-block:: console

   python notebooks/experiment_01_circular_3d.py
