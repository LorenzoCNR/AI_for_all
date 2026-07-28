# NeuroBridge Work Recap

This document summarizes the current state of the NeuroBridge synthetic neural
representation-learning pipeline: what was implemented, why it was implemented,
what it produces, and how to interpret the current results.

## 1. Research Goal

The project builds a controlled benchmark for neural representation learning.

The central question is:

```text
Can an encoder recover a known low-dimensional latent task structure from
synthetic neural spike-count time series?
```

The simulator gives us a ground truth latent process. This allows quantitative
evaluation of learned embeddings, instead of only visual inspection.

## 2. Generative Logic

The current simulator follows:

```text
shared task latent Z_task
    -> subject-specific neural driver Z_neural_driver
    -> loading matrix B
    -> neural drive u
    -> firing-rate intensity lambda
    -> spike-count tensor X
```

Main tensors:

```text
Z_task                shared task/stimulus latent, shape (160, 100, 3)
Z_neural_driver       subject-specific possibly lagged neural driver
B                     loading matrix, shape (3, 80)
u                     pre-nonlinearity neural drive
lambda                firing-rate-like intensity
X                     generated spike counts, shape (160, 100, 80)
```

Important conceptual correction:

```text
The labels and the task latent are shared across subjects.
The lag belongs to neural response generation, not to the task labels.
```

So subject 2 does not see a different stimulus. Subject 2 has a delayed neural
response.

## 3. Current Simulation Parameters

```text
seed = 164
n_trials = 160
trial_len = 100
n_neurons = 80
n_conditions = 8
n_traj_k = 3

dt = 0.02
nonlinearity = softplus
directional_scale = 3.0
extra_scale = 0.051

overdispersion = 0.25
refractory_mean_bins = 2
refractory_std_bins = 1
burst_probability = 0.05
burst_size_mean = 1.5
burst_window_bins = 3

subject_1 response lag = 0 bins
subject_2 response lag = 2 bins
```

The spike model is still phenomenological. It simulates observed spike-count
statistics, not the full biophysical mechanism that produces spikes.

## 4. Windowing

The current windowing is designed to preserve the full time-series length for
offset/window-based temporal models.

```text
window_size = 10
stride = 1
padding = center
pad_value = 0.0
time_mode = absolute
```

With centered padding:

```text
160 trials * 100 time bins = 16000 windows per subject
```

Each embedding corresponds to one original time bin, while the encoder still
receives local temporal context.

## 5. Encoders Used In Current Fast Run

The current fast run uses:

```text
PCA
Temporal CNN
Temporal Transformer
```

MLP and LSTM were removed from the current output run to keep the benchmark
focused and computationally manageable.

## 6. Loss Used For Current Embeddings

The current `.mat` embeddings were generated with:

```text
loss_mode = soft_structured
```

This is a custom soft structured contrastive objective:

```text
D_total = 0.5 * D_time + 0.5 * D_label
S_ij = exp(-D_total_ij / similarity_tau)
```

Parameters:

```text
temperature = 0.2
similarity_tau = 0.5
time_weight = 0.5
label_weight = 0.5
batch_size = 256
epochs = 1
learning_rate = 1e-3
weight_decay = 1e-4
embedding_dim = 3
```

Interpretation:

```text
The loss encourages windows close in trial time and/or circular task direction
to be close in embedding space.
```

This loss combines conditional contrastive learning with probability matching
over pairwise similarities.

## 7. Other Losses Implemented

The code also supports:

```text
supervised_infonce
time_offset_infonce
structured_specs
```

Meaning:

```text
supervised_infonce
    positives are samples with the same label.

time_offset_infonce
    unsupervised temporal baseline; positives are windows from the same trial
    separated by a fixed temporal offset.

structured_specs
    generalized soft loss where each behavioral variable has an explicit
    geometry: temporal, circular, categorical, continuous 1D, or continuous nD.
```

Important:

```text
A full structured positive/negative sampler is not implemented yet.
```

What is still missing for a proper temporal-offset contrastive baseline:

```text
anchor sampler
positive sampler from conditional distribution
negative/prior sampler
standard InfoNCE over sampled positives and negatives
```

## 8. Output Files

Main output folder:

```text
outputs/baselines/
```

Current `.mat` files:

```text
outputs/baselines/mat_embeddings/
    cnn_subject_embeddings_soft_structured.mat
    pca_subject_embeddings_soft_structured.mat
    transformer_subject_embeddings_soft_structured.mat
```

Each `.mat` file contains both subjects.

Important fields:

```text
subject_1_embedding
subject_2_embedding
subject_1_X_spikes
subject_2_X_spikes
subject_1_latent_task
subject_2_latent_task
subject_1_latent_neural_driver
subject_2_latent_neural_driver
subject_1_label
subject_2_label
subject_1_trial_id
subject_2_trial_id
subject_1_trial_id_1based
subject_2_trial_id_1based
subject_1_time_id
subject_2_time_id
subject_1_loading_B
subject_2_loading_B
best_lag
best_lag_score
```

Diagnostic plots from the `.mat` files:

```text
outputs/baselines/figures/mat_diagnostics/
```

CSV metrics:

```text
outputs/baselines/latent_recovery_results.csv
outputs/baselines/cross_subject_lag_alignment.csv
```

## 9. Current Metrics

Current fast-run latent recovery:

```text
subject_1 pca          proc_r2  0.3360   rsa_s 0.6914   rsa_p 0.6928
subject_1 cnn          proc_r2  0.5562   rsa_s 0.7124   rsa_p 0.7167
subject_1 transformer  proc_r2  0.1393   rsa_s 0.4231   rsa_p 0.4558

subject_2 pca          proc_r2  0.2881   rsa_s 0.6563   rsa_p 0.6581
subject_2 cnn          proc_r2  0.5247   rsa_s 0.6813   rsa_p 0.6865
subject_2 transformer  proc_r2 -0.1924   rsa_s 0.2582   rsa_p 0.2877
```

Cross-subject lag-aware alignment:

```text
pca          best_lag =  3   best_score =  0.0336
cnn          best_lag =  1   best_score =  0.5457
transformer  best_lag = -5   best_score = -0.4812
```

## 10. Interpretation

PCA recovers some structure but only weakly.

The CNN currently gives the best learned representation:

```text
subject_1 proc_r2 ~= 0.56
subject_2 proc_r2 ~= 0.52
```

This means the CNN embedding recovers a meaningful part of the known latent
geometry.

The Transformer is not reliable in the current run:

```text
epochs = 1
```

So the current Transformer result is a pipeline check, not a scientific
conclusion. It is undertrained.

The lag result is also preliminary. The simulated neural response lag is two
bins for subject 2, but the fast CNN run estimates best lag one. This means the
pipeline is able to perform lag-aware alignment, but the current short training
run is not yet precise enough to claim robust lag recovery.

## 11. Why The Pipeline Was Made Faster

With full-length windowing, each subject has 16000 windows. Some computations
scale poorly:

```text
batch similarity: batch_size x batch_size
RSA distance matrices: n_samples x n_samples
large interactive plots: many points and traces
```

To keep the current run usable:

```text
full embeddings are exported to .mat
RSA/Procrustes metrics are computed on sampled points
diagnostic plots are lightweight PNGs generated from .mat
large Plotly plots are optional
```

## 12. Current Level

Current level:

```text
research prototype
```

Not yet:

```text
final paper-grade benchmark
```

Strengths:

```text
controlled simulator
known latent ground truth
multi-subject setup
response lag
full-length windowing
exportable .mat analysis files
basic metrics and plots
```

Weaknesses:

```text
soft loss is custom and needs stronger standard baselines
Transformer is undertrained
only one seed in current run
no full structured positive/negative sampler yet
no ablation table yet
no real-data validation yet
```

## 13. Next Steps

Priority next steps:

```text
1. Run the same benchmark with supervised_infonce.
2. Run the same benchmark with time_offset_infonce.
3. Add an explicit temporal-offset sampler.
4. Train CNN and Transformer longer.
5. Repeat over multiple seeds.
6. Add ablations over window_size, lag, noise, time_weight, label_weight.
7. Compare all methods in one table.
8. Keep .mat export stable for MATLAB/Python analysis.
```

The immediate scientific comparison should be:

```text
PCA
CNN + soft_structured
CNN + supervised_infonce
CNN + time_offset_infonce
CNN + structured temporal sampler
Transformer variants after proper training
```

## 14. Loading Ablation Result

The 160-trial circular experiment now separates two factors:

```text
informative-neuron probability
loading magnitude on latent coordinates 1--3
```

Raising the probability assigned to direction, progress, and mixed neurons to
0.95 was not sufficient. With the original loading magnitude, CNN 3D recovery
at one epoch remained weak (RSA Spearman approximately 0.14).

Keeping those probabilities fixed and multiplying `B[0:3, :]` by 3 improved
one-epoch RSA to approximately 0.49 and 0.46 for the two subjects. The
start/end condition-spread ratios were 0.11 and 0.25, indicating substantially
better recovery of the shared starting region.

After 50 epochs, RSA increased to approximately 0.53--0.54, but the start/end
ratios increased to 0.90 and 1.04. The current objective therefore improves
global distance agreement while losing the common-origin constraint.

Next controlled comparisons:

```text
circular K=3 versus circular K=5
linear track with direction, position, mixed, and localized place neurons
loss ablations that explicitly preserve common-origin geometry
```
