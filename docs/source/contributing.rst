Contributing
============

Development installation
------------------------

Install the package in editable mode and run the verified test suite:

.. code-block:: console

   python -m pip install -e .
   python -m unittest discover -s tests -v

Experiment contributions
------------------------

New experiments should:

* state the research question before introducing hyperparameters;
* preserve trial boundaries during windowing;
* keep generated outputs outside Git;
* save complete configuration and random seeds;
* report both coordinate and distance-geometry metrics;
* distinguish in-sample recovery from held-out generalization;
* include a targeted test for new generative mechanisms.
