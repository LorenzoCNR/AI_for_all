Installation
============

Requirements
------------

NeuroBridge requires Python 3.11 or newer. A CUDA-capable GPU is optional but
recommended for neural encoder training.

Editable installation
---------------------

From the repository root:

.. code-block:: console

   python -m pip install -e .

The verified Windows numerical stack uses NumPy 1.26.4, SciPy 1.13.1, and
scikit-learn 1.5.2. Binary incompatibilities between NumPy and BLAS/LAPACK
packages can terminate PCA without a normal Python traceback.

Documentation installation
--------------------------

Install the documentation dependencies:

.. code-block:: console

   python -m pip install -e .[docs]

Build the documentation on Windows:

.. code-block:: console

   docs\make.bat html

The generated site is written to ``docs/_build/html/index.html``.
