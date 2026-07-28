# Reproducible Experiments

This directory contains four controlled experiments. The `.ipynb` files are
the primary, documented entry points. The matching `.py` files provide the
same configurations for terminal execution and automation.

The notebooks do not call the all-in-one experiment runner. They expose the
generative and learning stages as separate cells so that one stage can be
inspected, changed, and rerun without hiding the scientific choices.

## Experiment Matrix

| Notebook | Task | Known latent coordinates |
|---|---|---|
| `experiment_01_circular_3d.ipynb` | circular reaching | X, Y, progress |
| `experiment_02_circular_5d.ipynb` | circular reaching | X, Y, progress, velocity, context |
| `experiment_03_linear_position_direction.ipynb` | linear track | position, direction |
| `experiment_04_linear_enriched.ipynb` | linear track | position, direction, velocity, context |

All experiments use:

- 160 trials;
- 100 time bins per trial;
- 100 simulated neurons;
- 20 ms bins;
- centered 10-bin windows with stride 1;
- an 80/20 trial-level fit/test split;
- PCA and residual CNN1D encoders;
- 30 CNN training epochs;
- random seed 42.

The controlled matrix changes only the task family and latent-state
dimensionality.

## Running a Notebook

1. Open one `.ipynb` file in VS Code.
2. Select the Python environment in which NeuroBridge is installed.
3. Choose **Run All**, or execute the cells in order.

The notebooks locate the repository automatically when opened from either the
repository root or the `notebooks/` directory.

To run the equivalent scripts:

```bash
python notebooks/experiment_01_circular_3d.py
python notebooks/experiment_02_circular_5d.py
python notebooks/experiment_03_linear_2d.py
python notebooks/experiment_04_linear_4d.py
```

## Pipeline

Each notebook executes the same visible stages:

| Stage | Imported module | Input | Output | Main controls |
|---|---|---|---|---|
| Latent task | `neurobridge.data.sim.LatentTrajectoryGenerator` | experiment config | `Z`, condition, task state | latent dimension, noise, trial count |
| Population map | `build_structured_B` or `build_linear_loading_and_place_fields` | `Z`/task state | `B`, neuron types, place drive | tuning mixture, place fraction, loading scale |
| Spike emission | `drive_to_rate`, `rate_to_spike` | `Z`, `B`, baseline | `u`, `lam`, `X` | nonlinearity, rate scale, `dt` |
| Windowing | `build_windows_and_labels` | `X`, metadata | `TemporalWindowDataset` | window size, stride, padding |
| Split | `split_trials` | trial metadata | train/test masks | training fraction, seed |
| PCA | `sklearn.decomposition.PCA` | flattened windows | PCA embedding | output dimension |
| Soft target | `build_similarity_matrix` | batch metadata | pairwise target `Q` | time/label weights, target temperature |
| CNN1D | `TemporalCNNEncoder` | temporal windows | CNN embedding | channels, layers, kernel, output dimension |
| Loss | `soft_contrastive_loss` | embedding, `Q` | scalar loss | embedding temperature |
| Evaluation | `evaluate_models` | embeddings, known `Z` | held-out metrics | evaluation subset |
| Saving | `joblib`, `torch.save`, plotting module | all named objects | reproducible artifacts | output directory |

The CNN training loop is also visible. The notebook constructs the
`DataLoader`, model, optimizer, target function, and loss function directly.
`train_epoch` hides only repeated PyTorch mechanics for one epoch; it does not
choose the model, metadata geometry, or objective.

`Z` has shape:

```text
(trials, time bins, latent dimensions)
```

`X` has shape:

```text
(trials, time bins, neurons)
```

Centered padding preserves all 100 time bins, so each model produces 16,000
embedding rows before the train/test selection.

## How to Modify an Experiment

- Change the latent process in the configuration and latent-generation cells.
- Change neural tuning in the population-map cell.
- Change the observation process in the explicit `u`, `lam`, and `X` cell.
- Change temporal context in the configuration, then rerun from windowing.
- Change the target geometry through `time_weight`, `label_weight`, and
  `similarity_tau`, or replace the target function.
- Change the objective by replacing `loss_function`.
- Change the neural architecture in the `TemporalCNNEncoder` constructor.
- Add an encoder by following the visible PCA or CNN path and adding its
  embedding to the evaluation dictionary.

## Circular Task

Every trial begins at a common center and progresses toward one of eight
directions.

The essential state contains X position, Y position, and movement progress.
The enriched state adds velocity and trial context.

## Linear Track

Every trial contains two phases:

```text
outbound: position 0 -> 1, direction +1
return:   position 1 -> 0, direction -1
```

These are phases within each trial, not separate trial classes. A subset of
neurons has Gaussian place fields, so different units preferentially respond
near different track positions.

## Metrics

RSA Spearman correlation compares:

1. all pairwise distances among known states in held-out trials;
2. the corresponding pairwise distances among learned embeddings.

It measures preservation of near/far ordering. It is not classification
accuracy and does not guarantee that a plotted trajectory has the correct
visible shape.

Procrustes R^2 measures coordinate agreement after centering, rotation,
reflection, and global rescaling.

For enriched states, the full metric uses every latent coordinate and the
motor-core metric uses only the task-defining coordinates.

## Reference Results

Single-seed held-out RSA Spearman values:

| Experiment | PCA full | CNN1D full | PCA motor core | CNN1D motor core |
|---|---:|---:|---:|---:|
| Circular 3D | 0.893 | 0.927 | 0.893 | 0.927 |
| Circular 5D | 0.682 | 0.619 | 0.899 | 0.904 |
| Linear position + direction | 0.893 | 0.778 | 0.893 | 0.778 |
| Linear enriched | 0.886 | 0.803 | 0.838 | 0.796 |

These values are reproducibility checks, not final comparative claims. Scalar
metrics must be interpreted together with the saved trajectory figures.

## Generated Artifacts

Each run writes to:

```text
outputs/<experiment-name>/
```

The directory contains:

- `results.joblib`: arrays, metadata, embeddings, split, and metrics;
- `metrics.json`: compact metric summary;
- `models/`: fitted PCA and CNN1D parameters;
- `figures/`: latent, embedding, and task-specific diagnostics.

`outputs/` is intentionally excluded from Git because every artifact can be
regenerated from the notebooks.
