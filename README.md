# NeuroBridge

NeuroBridge is a research-oriented Python package for controlled neural
time-series simulation and representation learning.

The project generates a known task-level latent process, maps it into a neural
population, and tests whether representation-learning methods can recover the
latent geometry from sparse spike counts.

The current repository is a research prototype. Its reproducible experiment
suite compares PCA with a temporal CNN trained using a **soft structured
contrastive loss**. The complete objective is described in
[Learning objectives and loss](docs/03_LEARNING_OBJECTIVES.md).

## Documentation

The complete documentation is readable directly from the repository; the
project website is not required:

- [Documentation index](docs/README.md)
- [Generative model](docs/01_GENERATIVE_MODEL.md)
- [Data and temporal windows](docs/02_DATA_AND_WINDOWS.md)
- [Learning objectives and loss](docs/03_LEARNING_OBJECTIVES.md)
- [PCA and temporal encoders](docs/04_ENCODERS.md)
- [Evaluation, lag, and multiple subjects](docs/05_EVALUATION_AND_MULTISUBJECT.md)
- [Experiments, results, and limitations](docs/06_EXPERIMENTS_AND_LIMITATIONS.md)
- [Cell-by-cell notebook guide](notebooks/EXPERIMENTS.md)

The `.rst` files under `docs/source/` are reStructuredText build sources for
the searchable Sphinx website. They are not required reading: the numbered
Markdown documents above are the canonical scientific explanation.

## Research Questions

- Which properties of a known latent task survive stochastic neural emission?
- Which linear and nonlinear encoders recover that geometry?
- If two subjects perform the same task with a temporal response lag, can the
  shared geometry and the imposed lag be recovered from their embeddings?

## Implemented Components

- Circular and linear latent task generators.
- Subject-specific loading matrices and neural tuning profiles.
- Temporal lag simulation and lag-aware cross-subject alignment.
- Spike-count emission with softplus rates, overdispersion, refractory effects,
  and bursting.
- Centered temporal windows with trial, time, and condition metadata.
- PCA, CNN1D, MLP, LSTM, and Transformer encoders.
- Soft structured contrastive, supervised InfoNCE, and temporal-offset losses.
- Procrustes and representational-similarity metrics.

The validated experiment suite currently uses PCA and CNN1D. Other encoders
are available as research components but are not part of the minimal verified
run.

## Installation

Python 3.11 is recommended.

```bash
git clone https://github.com/LorenzoCNR/AI_for_all.git
cd AI_for_all
python -m pip install -e .
```

The verified Windows numerical stack is:

```text
numpy          1.26.4
scipy          1.13.1
scikit-learn   1.5.2
```

NumPy, SciPy, and scikit-learn should come from mutually compatible binary
builds. An incompatible Windows BLAS/LAPACK stack may fail during PCA without a
Python traceback.

## Quick Start

Open one of the four executable Jupyter notebooks in VS Code or Jupyter:

- `notebooks/experiment_01_circular_3d.ipynb`
- `notebooks/experiment_02_circular_5d.ipynb`
- `notebooks/experiment_03_linear_position_direction.ipynb`
- `notebooks/experiment_04_linear_enriched.ipynb`

Each notebook explains the research question, configuration, generated
matrices, model inputs, recovery metrics, and saved artifacts before executing
the corresponding stage. Use **Run All** for complete reproduction or execute
the cells individually while studying the workflow.

Equivalent `.py` mirrors are retained for terminal execution and automated
checks:

```bash
python notebooks/experiment_01_circular_3d.py
python notebooks/experiment_02_circular_5d.py
python notebooks/experiment_03_linear_2d.py
python notebooks/experiment_04_linear_4d.py
```

See [notebooks/EXPERIMENTS.md](notebooks/EXPERIMENTS.md) for the complete experiment
protocol, output structure, reference results, and interpretation.

### Linear-track place fields

In the linear experiments, a subset of neurons is assigned a preferred
position along the normalized track. Neuron `j` receives a Gaussian drive
centered at its preferred position `mu_j`; its expected firing rate is highest
near `mu_j` and decreases with distance from it. Different preferred positions
make the population cover the route.

This mechanism provides a known neuron-level ground truth. A later
explainability analysis, for example SHAP or another feature-attribution
method, can test whether an encoder or decoder assigns importance to the
place-selective neurons that should be informative at a given track position.
The current notebooks verify that the assigned selectivity is present in the
simulated rates; they do not yet perform the attribution analysis.

### Multi-subject extension

NeuroBridge can map the same task-level latent process into two different
neural populations. The subjects share the task geometry but may have
different loading matrices, neuron counts, baselines, tuning mixtures, and
stochastic spike realizations. One population can also receive a delayed
version of the task process.

After encoding both populations, lag-aware alignment compares candidate
temporal shifts using only overlapping trial-time samples. For each candidate
lag, one embedding is aligned to the other with Procrustes transformation and
an alignment score is computed. In a controlled simulation, the best-scoring
candidate can therefore be compared with the known imposed lag.

This makes shared-geometry and lag recovery testable because the simulator
provides their ground truth. The four notebooks currently validate the
single-population foundations; a complete held-out multi-subject benchmark is
the next experimental stage.

An accompanying static project page is available at
[site/index.html](site/index.html). It can be opened directly without a web
server for local inspection. After this repository is published on `main` and
GitHub Pages is enabled with **GitHub Actions** as its source, the included
workflow publishes:

```text
https://lorenzocnr.github.io/AI_for_all/
```

The visual project page is served at the root URL and the searchable Sphinx
documentation under `/docs/`. The same canonical documents remain accessible
from the repository even when Pages is disabled.

Full searchable package documentation is built with Sphinx:

```bash
python -m pip install -e .[docs]
docs\make.bat html
```

Open `docs/_build/html/index.html` after the build.

## Package Layout

```text
src/neurobridge/
|-- data/
|   |-- dataset.py
|   `-- sim/
|-- eval/
|-- losses/
|-- models/
|-- sampling/
|-- train/
|-- utils/
`-- viz/

notebooks/
|-- experiment_01_circular_3d.ipynb
|-- experiment_02_circular_5d.ipynb
|-- experiment_03_linear_position_direction.ipynb
|-- experiment_04_linear_enriched.ipynb
|-- experiment_01_circular_3d.py
|-- experiment_02_circular_5d.py
|-- experiment_03_linear_2d.py
|-- experiment_04_linear_4d.py
`-- EXPERIMENTS.md

tests/
```

Important modules:

- `data/sim`: task latents, loading matrices, lag, and spike emission.
- `data/dataset.py`: PyTorch temporal-window dataset.
- `models/temporal_cnn.py`: temporal encoders.
- `sampling/batch_similarity.py`: metadata distances and soft similarities.
- `losses/infonce.py`: contrastive objectives.
- `eval/representation.py`: latent-recovery and alignment metrics.

## Tests

```bash
python -m unittest \
  discover -s tests -v
```

The standard-library test run executes 37 tests: 35 pass and 2 optional
interactive Plotly tests are skipped.

## Current Scope

Each controlled experiment generates 160 trials. The split is performed on
complete trials:

```text
128 training trials -> fit PCA and CNN1D
 32 test trials     -> compute held-out recovery metrics
```

Windows from a test trial never enter model fitting. The reported test metrics
therefore measure recovery on new stochastic realizations from the same
simulator and task distribution.

This is meaningful held-out validation, but it is not yet evidence that the
method:

- is stable across many random seeds;
- transfers to a different task, simulator, animal, or recording session;
- generalizes to task conditions absent from training;
- outperforms Poisson-aware or other neural latent-variable baselines;
- reproduces every statistical property of biological spike trains.

Those claims require multi-seed uncertainty estimates, explicit ablations,
additional baselines, and real-data experiments.

## Reproducibility And Outputs

Running a notebook creates:

```text
outputs/<experiment-name>/
|-- metrics.json
|-- results.joblib
|-- models/
`-- figures/
```

`results.joblib` contains the simulated latent state, spike counts, metadata,
train/test trial identifiers, embeddings, and metrics. The model directory
contains fitted PCA and CNN1D parameters; the figure directory contains the
trajectory and diagnostic plots.

These generated files are not stored in Git because they are binary,
relatively large, and reproducible from the versioned notebook and fixed
random seed. Cloning the repository provides the code and configuration;
running the selected notebook reconstructs its complete output directory.
