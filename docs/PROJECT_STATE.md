# NeuroBridge Project State

This document summarizes the current state of the repository so the project can
be resumed without relying on external notes.

## Research Objective

The current objective is to build a controllable synthetic neural time-series
benchmark for testing representation learning algorithms. The simulator creates
a known low-dimensional latent process and emits spike-count neural activity.
Encoders are then evaluated by how well they recover the latent geometry and
shared structure across simulated subjects.

## Current Simulation Task

Default task:

```text
circular motor task
```

Latent dimensionality:

```text
n_traj_k = 3
```

Main simulator path:

```text
Z_task -> Z_neural_driver -> B -> u -> lambda -> X
```

Where:

- `Z_task` is the shared task/stimulus latent trajectory.
- `Z_neural_driver` is the subject-specific latent driver used for neural
  response generation. It can be temporally lagged relative to `Z_task`.
- `B` maps latent variables to neurons.
- `u` is the pre-nonlinearity neural drive.
- `lambda` is the firing-rate-like intensity.
- `X` is the generated spike-count tensor.

Current default spike-generation features:

```text
nonlinearity: softplus
dt: 0.02
overdispersion: 0.25
refractory_mean_bins: 2
refractory_std_bins: 1
burst_probability: 0.05
burst_size_mean: 1.5
burst_window_bins: 3
```

Multi-subject default:

```text
subject_1 lag: 0 bins
subject_2 lag: 2 bins
```

Important interpretation:

```text
Labels and task latent are shared across subjects.
The lag belongs to neural response generation, not to the screen/task labels.
```

## Current Learning Setup

The current baseline suite is:

```text
experiments/encoder_baseline_suite.py
```

It compares:

```text
PCA
TemporalCNNEncoder
TemporalTransformerEncoder
```

Windowing defaults in the suite:

```text
window_size = 10
stride = 1
padding = center
time_mode = absolute
```

`padding="center"` uses zero padding at trial boundaries and returns one window
centered on each original time bin. With `stride = 1`, the suite produces:

```text
160 trials * 100 bins = 16000 windows per subject
```

This preserves the full length of the original series while supporting
offset/window-based temporal models.

The neural encoders use:

```text
embedding_dim = 3
```

PCA is fitted with:

```text
pca_plot_components = 5
```

The first three PCA components are used for metric comparison; additional
components are saved for diagnostic plots.

## Current Learning Objective

The main training objective is a soft structured contrastive loss.

Batch similarity is built from explicit metadata geometries. The current
circular task uses:

```text
temporal distance
circular label distance
```

Default weights:

```text
time_weight = 0.5
label_weight = 0.5
```

This objective encourages windows close in task time and circular condition to
have nearby embeddings.

The generalized implementation also supports scalar or vector-valued continuous
metadata, for example a 2D position label:

```text
D_total =
    w_time D_time
  + w_direction D_circular(direction)
  + w_position D_euclidean(position_x, position_y)
```

This is not a "single-label vs multi-label" distinction in the classification
sense. The important distinction is the geometry inside each behavioral
variable: categorical, circular, continuous 1D, or continuous nD.

The default baseline configuration is:

```text
loss_mode = soft_structured
```

The suite also supports:

```text
loss_mode = supervised_infonce
```

`supervised_infonce` is a more standard supervised contrastive baseline: samples
with the same condition label are positives, and other samples are negatives.
It does not use temporal distance or circular label distance.

```text
loss_mode = time_offset_infonce
```

`time_offset_infonce` is an unsupervised temporal baseline: positives are
windows from the same trial separated by a chosen temporal offset. It does not
use behavioral labels.

The current `soft_structured` objective is more flexible because it uses a
continuous target similarity matrix, but it can bias the learned representation
toward the chosen metadata weights. A fuller structured sampler with explicit
anchor, positive, and negative sampling remains a planned next baseline.

## Current Evaluation

Latent recovery:

```text
Procrustes R2 against known latent Z
RSA Spearman correlation
RSA Pearson correlation
```

Cross-subject recovery:

```text
lag-aware trial/time alignment
orthogonal Procrustes alignment
best lag selected by maximum Procrustes R2 over candidate lags
```

Current lag scan:

```text
lags = -5, ..., 0, ..., +5
```

Current fast run:

```text
subject_1 pca          proc_r2  0.3360  rsa_s 0.6914
subject_1 cnn          proc_r2  0.5562  rsa_s 0.7124
subject_1 transformer  proc_r2  0.1393  rsa_s 0.4231
subject_2 pca          proc_r2  0.2881  rsa_s 0.6563
subject_2 cnn          proc_r2  0.5247  rsa_s 0.6813
subject_2 transformer  proc_r2 -0.1924  rsa_s 0.2582
```

Lag results:

```text
PCA:         best_lag = 3
CNN:         best_lag = 1
Transformer: best_lag = -5
```

The CNN is currently the only learned encoder behaving sensibly in the fast
run. The Transformer is undertrained (`epochs = 1`) and should not be treated
as a scientific result yet.

## Generated Outputs

The baseline writes:

```text
outputs/baselines/latent_recovery_results.csv
outputs/baselines/cross_subject_lag_alignment.csv
outputs/baselines/figures/*.html
outputs/baselines/figures/png_preview/*.png
outputs/baselines/figures/mat_diagnostics/*.png
outputs/baselines/mat_embeddings/*.mat
```

The output directory is intentionally ignored by Git.

Current `.mat` files:

```text
cnn_subject_embeddings_soft_structured.mat
pca_subject_embeddings_soft_structured.mat
transformer_subject_embeddings_soft_structured.mat
```

Each file contains both subjects. Main fields:

```text
subject_1_embedding / subject_2_embedding
subject_1_X_spikes / subject_2_X_spikes
subject_1_latent_task / subject_2_latent_task
subject_1_latent_neural_driver / subject_2_latent_neural_driver
subject_1_label / subject_2_label
subject_1_trial_id / subject_2_trial_id
subject_1_trial_id_1based / subject_2_trial_id_1based
subject_1_time_id / subject_2_time_id
subject_1_loading_B / subject_2_loading_B
best_lag
best_lag_score
```

## Project Input/Output Layout

For long-lived projects and real datasets, use:

```text
inputs/projects/<project_name>/
outputs/projects/<project_name>/
```

The helper class is:

```text
src/neurobridge/utils/project_store.py
```

It creates input folders for raw/interim/processed data and output folders for
models, embeddings, figures, and logs. Both `inputs/` and `outputs/` are ignored
by Git.

Important plot families:

- original latent 2D/3D plots;
- encoder 2D/3D plots;
- condition-averaged trajectories;
- condition centroids with dispersion;
- PCA pairwise component plots;
- cross-subject best-lag Procrustes alignment plots.

## Main Files To Read First

```text
src/neurobridge/data/sim/Lat_traj_generator.py
src/neurobridge/data/sim/builders.py
src/neurobridge/data/sim/Spikes_generator.py
src/neurobridge/data/dataset.py
src/neurobridge/models/temporal_cnn.py
src/neurobridge/losses/infonce.py
src/neurobridge/sampling/batch_similarity.py
src/neurobridge/eval/representation.py
experiments/encoder_baseline_suite.py
```

## Current Open Technical Questions

- Whether the soft distance overemphasizes condition clustering relative to
  continuous trajectory recovery.
- Whether normalized spherical embeddings should be used only for the loss or
  also for all diagnostics.
- Whether an explicit offset sampler should be added beside the current soft
  similarity matrix.
- Whether cross-subject alignment should eventually include nonlinear methods
  such as CCA/PLS, optimal transport, dynamic time warping, or shared-response
  models.
- Whether the simulator should include additional task families such as linear
  track navigation with place/direction fields.
