# NeuroBridge Documentation

This directory is the canonical scientific and technical documentation for
NeuroBridge. It is designed to be readable directly on GitHub or from a local
clone: the project website is optional.

## Start Here

Read the documents in this order:

1. [Generative model](01_GENERATIVE_MODEL.md) explains the task latent, neural
   population map, firing rates, spike-count emission, and linear-track place
   fields.
2. [Data and temporal windows](02_DATA_AND_WINDOWS.md) explains the tensors,
   metadata, padding, trial boundaries, train/test split, and what one model
   observation represents.
3. [Learning objectives](03_LEARNING_OBJECTIVES.md) derives the soft structured
   contrastive loss, distinguishes its two temperatures, and states what is
   supervised, self-supervised, and computationally expensive.
4. [Encoders](04_ENCODERS.md) explains PCA, CNN1D, MLP, LSTM, and Transformer
   processing at both the algorithmic and tensor level.
5. [Evaluation and multiple subjects](05_EVALUATION_AND_MULTISUBJECT.md)
   explains RSA, Procrustes alignment, lag recovery, and the limits of causal
   interpretation.
6. [Experiments and evidence](06_EXPERIMENTS_AND_LIMITATIONS.md) documents the
   four controlled notebooks, reference results, generated artifacts, current
   claims, and missing experiments.

For a cell-by-cell reproduction guide, see
[the notebook guide](../notebooks/EXPERIMENTS.md).

## Documentation Layers

NeuroBridge has three documentation layers:

| Location | Purpose | Intended reader |
|---|---|---|
| `docs/*.md` | Canonical scientific and technical explanation | GitHub visitors, collaborators, reviewers |
| `docs/source/*.rst` | Sphinx source used to build the searchable documentation website | Documentation build system |
| `docs/archive/` | Historical drafts retained for traceability | Maintainers only |

Files ending in `.rst` use **reStructuredText**, the markup format consumed by
Sphinx. They play a role similar to Markdown files, but are primarily build
sources. A reader should not need to inspect them to understand the project:
the complete explanation is available in the Markdown documents listed above.

## What Is Implemented

The package currently includes:

- circular and linear controlled motor-task latents;
- heterogeneous neural population mappings;
- Poisson spike-count emission plus optional overdispersion, bursting, and
  refractory mechanisms;
- centered temporal windows that never cross trial boundaries;
- PCA, CNN1D, MLP, LSTM, and Transformer encoders;
- soft structured contrastive, supervised InfoNCE, and temporal-offset
  objectives;
- held-out RSA and Procrustes recovery metrics;
- subject-specific neural mappings, imposed temporal lag, and lag-aware
  alignment utilities.

The four reproducible notebooks currently validate PCA and CNN1D on one
simulated population. Other encoders and multisubject utilities are implemented
research components, but they are not yet covered by the same complete
held-out benchmark.

## Build The Searchable Site

The documentation website is generated from `docs/source/`:

```powershell
python -m pip install -e ".[docs]"
docs\make.bat html
```

The local result is `docs/_build/html/index.html`. Building the website is not
required to read any canonical document in this directory.
