# NeuroBridge

NeuroBridge is a research-oriented Python package for controlled neural
time-series simulation and representation learning.

The project generates a known task-level latent process, maps it into
subject-specific neural populations, and tests whether representation-learning
methods can recover the shared latent geometry from sparse spike counts.

The current repository is a research prototype. Its first reproducible
experiment compares PCA with a contrastively trained temporal CNN.

## Research Questions

- Which properties of a known latent task survive stochastic neural emission?
- Can different neural populations reveal the same task-level geometry?
- Which linear and nonlinear encoders recover that geometry?
- Can temporal delays between populations be recovered from learned
  representations?

## Implemented Components

- Circular and linear latent task generators.
- Subject-specific loading matrices and neural tuning profiles.
- Temporal lag between a shared task latent and a neural population.
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
git clone <repository-url>
cd Neuro_Bridge
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

The first experiment is intentionally divided into two commands:

```bash
python notebooks/First_experiment_24_07.py
python notebooks/first_experiment_model_24_07.py
```

The first command creates the controlled simulation. The second loads that
simulation, constructs identical windows for PCA and CNN1D, trains the CNN, and
saves metrics, models, and figures.

See [notebooks/README.md](notebooks/README.md) for the complete experiment
protocol, output structure, reference results, and interpretation.

### Controlled task suite

Four independent notebook-style scripts compare essential and enriched latent
spaces:

```bash
python notebooks/experiment_01_circular_3d.py
python notebooks/experiment_02_circular_5d.py
python notebooks/experiment_03_linear_2d.py
python notebooks/experiment_04_linear_4d.py
```

The linear experiments include Gaussian place fields whose preferred
locations are distributed along the track. Each run saves its simulation,
models, embeddings, metrics, and figures under `outputs/<experiment-name>/`.

An accompanying static project page is available at
[site/index.html](site/index.html). It can be opened directly without a web
server.

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
|-- First_experiment_24_07.py
|-- first_experiment_model_24_07.py
|-- experiment_01_circular_3d.py
|-- experiment_02_circular_5d.py
|-- experiment_03_linear_2d.py
|-- experiment_04_linear_4d.py
`-- README.md

experiments/
`-- encoder_baseline_suite.py

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
  tests.test_similarity \
  tests.test_learning_components \
  tests.test_representation_eval \
  tests.test_embedding_plots
```

The current verified result is 32 completed tests: 30 passed and 2 optional
tests skipped.

## Current Scope

The first experiment is an in-sample latent-recovery demonstration. It does not
yet establish:

- generalization to held-out trials or conditions;
- statistical reliability across random seeds;
- recovery of the imposed cross-subject lag;
- superiority over Poisson-aware latent-variable baselines;
- biological realism beyond selected distributional checks.

These are experimental questions, not assumed conclusions.

## Generated Data

`outputs/`, private documentation, and local reference datasets are excluded
from Git. Run the experiment locally to regenerate its artifacts.
