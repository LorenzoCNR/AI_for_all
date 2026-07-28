# Reproducible Experiments

## Controlled task suite

The new suite isolates task family and latent dimensionality while keeping the
same simulation-to-encoder workflow:

| Notebook | Task | Latent coordinates |
|---|---|---|
| `experiment_01_circular_3d.py` | circular | X, Y, progress |
| `experiment_02_circular_5d.py` | circular | X, Y, progress, velocity, context |
| `experiment_03_linear_2d.py` | linear | position, direction |
| `experiment_04_linear_4d.py` | linear | position, direction, velocity, context |

All four use 160 trials, 100 bins, 100 neurons, centered 10-bin windows, PCA,
and CNN1D. The linear tasks additionally reserve approximately 25% of neurons
for localized Gaussian place fields:

```text
g_j(p) = place_scale * exp(-(p - center_j)^2 / (2 * width^2))
```

Place fields enter the neural drive as a nonlinear position-dependent term.
They are not represented as if they were linear columns of `B`.

Run:

```bash
python notebooks/experiment_01_circular_3d.py
python notebooks/experiment_02_circular_5d.py
python notebooks/experiment_03_linear_2d.py
python notebooks/experiment_04_linear_4d.py
```

Initial RSA Spearman results:

| Task | PCA | CNN1D |
|---|---:|---:|
| Circular 3D | 0.740 | 0.545 |
| Circular 5D | 0.432 | 0.471 |
| Linear 2D | 0.807 | 0.794 |
| Linear 4D | 0.717 | 0.719 |

These are in-sample, single-seed reference values. They support debugging and
controlled comparison, not a final generalization claim.

## Multisubject 24-07 experiment

This directory contains the two stages of the validated NeuroBridge first
experiment.

## Purpose

The experiment asks whether a known task-level latent geometry can be recovered
from two distinct stochastic neural populations.

The two populations:

- perform the same circular motor task;
- share the same task-level latent tensor;
- have different neuron counts, loading matrices, tuning composition, baseline
  activity, emission variability, and temporal lag.

The experiment is split deliberately. Neural-population parameters must not
silently redefine the task latent.

## Pipeline

```text
shared task configuration
        |
        v
Z_task: trials x time x latent dimensions
        |
        +------------------------------+
        |                              |
        v                              v
subject_01 configuration         subject_02 configuration
        |                              |
        v                              v
B_1, c_1, lag_1                 B_2, c_2, lag_2
        |                              |
        v                              v
u_1 -> lambda_1 -> X_1          u_2 -> lambda_2 -> X_2
        |                              |
        +--------------+---------------+
                       |
                       v
              centered neural windows
                       |
                +------+------+
                |             |
                v             v
               PCA         temporal CNN
                |             |
                +------+------+
                       |
                       v
            recovery against known Z_task
```

## Stage 1: Simulation

Run:

```bash
python notebooks/First_experiment_24_07.py
```

Main output:

```text
outputs/first_experiment_24_07/simulation.pkl
```

### Shared task

The current configuration contains:

```text
trials                 80
time bins per trial   100
bin duration           20 ms
conditions              8
latent dimensions       5
```

The latent coordinates are:

1. Position X.
2. Position Y.
3. Movement progress.
4. Velocity.
5. Context.

The five columns are interpretable coordinates, but they should not be
described automatically as five independent degrees of freedom. Progress is
related to radial position, and velocity is derived from temporal progression.

### Subject-specific observation model

For each subject:

```text
Z_neural_driver = temporal_lag(Z_task)
u = Z_neural_driver @ B + c
lambda = softplus(u)
X = stochastic_spike_emission(lambda, dt)
```

`B` has shape:

```text
latent dimensions x neurons
```

Its columns define how individual neurons map latent coordinates into neural
drive. Neuron-type probabilities control the approximate population
composition; tuning scales control loading magnitude. These are different
modelling choices.

The current realized populations contain:

```text
subject_01: 100 neurons, 56 neurons loading on X-Y, lag 0 bins
subject_02: 140 neurons, 86 neurons loading on X-Y, lag 4 bins
```

Four bins at 20 ms correspond to an imposed delay of 80 ms for subject 2.

### Distributional check

The generated spike-count distributions have been compared with the local
Achilles and Gatsby hippocampal reference files:

```text
dataset                mean count/bin    active bins    maximum
Achilles real               0.0438          3.48%          5
Gatsby real                 0.0331          2.61%          6
subject_01 synthetic        0.0279          2.67%          6
subject_02 synthetic        0.0318          2.98%          5
```

This validates selected marginal count statistics. It does not establish that
the real and simulated population covariance structures are identical.

## Stage 2: PCA and CNN1D

Run Stage 1 first, then:

```bash
python notebooks/first_experiment_model_24_07.py
```

The second file creates:

```text
outputs/first_experiment_24_07/
|-- results.pkl
|-- figures/
`-- models/
```

### Common observations

Both models receive the same centered windows:

```text
number of windows x window size x neurons
```

With stride 1 and centered padding, every trial retains all 100 central time
positions. Both populations therefore produce 8,000 windows.

The target associated with each window is the mean `Z_task` state over the
corresponding latent window.

### PCA

PCA flattens each window:

```text
subject_01: 10 x 100 -> 1,000 features
subject_02: 10 x 140 -> 1,400 features
```

It retains five components and is evaluated against all five task coordinates.
PCA is unsupervised: it maximizes explained variance and receives no time or
condition metadata.

### Temporal CNN

External input:

```text
batch x window size x neurons
```

Internal Conv1d input:

```text
batch x neurons x window size
```

The current model uses:

- three Conv1d layers;
- kernel size 3 and same-length padding;
- 64 hidden channels;
- GELU activations;
- global temporal average pooling;
- a final three-dimensional projection;
- L2-normalized embeddings.

The CNN returns one embedding per window, not one embedding per time bin inside
the window.

The package also contains experimental residual components in
`src/neurobridge/models/blocks.py`: an additive skip wrapper, a concatenative
downsampling shortcut, channel normalization, and a squeeze helper. These
components are unit-tested but are **not used** by the current
`TemporalCNNEncoder` or by this experiment. The current CNN has no residual
connections.

Using those components would define a separate residual temporal encoder that
should be compared against the current plain CNN rather than silently replacing
it.

### Soft structured contrastive objective

For each minibatch, NeuroBridge constructs temporal and circular-condition
distance matrices:

```text
D = 0.5 * normalized temporal distance
  + 0.5 * normalized circular condition distance
```

The distance is converted into a soft target similarity:

```text
S_ij = exp(-D_ij / tau)
```

Each row is normalized into a target distribution `Q`. The normalized
embeddings define a cosine-softmax prediction distribution `P`. Training
minimizes the row-wise cross-entropy:

```text
L = -mean_i sum_j Q_ij log(P_ij)
```

Current hyperparameters:

```text
batch size             256
epochs                   50
learning rate          1e-3
weight decay           1e-4
embedding temperature   0.2
metadata tau             0.5
time weight              0.5
condition weight         0.5
```

`Z_task` is not supplied to the CNN during training. It is used afterward for
evaluation.

### Alternative objectives already implemented

The package contains four training modes:

| Mode | Positive structure | Current batch requirement |
|---|---|---|
| `soft_structured` | Soft time and circular-condition geometry | All available pairs in a random minibatch |
| `structured_specs` | Configurable temporal, circular, categorical, or continuous metadata | All available pairs in a random minibatch |
| `supervised_infonce` | Same categorical label | At least two examples of a class in the minibatch |
| `time_offset_infonce` | Same trial and exact temporal offset | The offset pair must occur in the minibatch |

`masked_infonce_loss` is also available as a lower-level objective for an
explicit boolean positive-pair mask.

These objectives define different positive relationships, but the current
suite still uses a standard shuffled `DataLoader`. In particular,
`time_offset_infonce` does not yet have a custom batch sampler that guarantees
the anchor and its offset-positive window are present together. It therefore
defines the correct temporal relation but only observes the relation when both
windows happen to occur in the same random minibatch.

A complete temporal-offset contrastive experiment requires a structured batch
sampler that deliberately selects anchors, positives, and negatives. This is
distinct from the loss function itself.

## Evaluation

Two complementary metric families are reported:

- Procrustes R2: coordinate-level agreement after centering, scaling, and
  orthogonal alignment.
- RSA Spearman/Pearson: agreement between pairwise-distance geometries.

The CNN is three-dimensional and is compared with Position X, Position Y, and
Movement progress. PCA retains five components and is compared with the full
five-coordinate target.

This dimensionality choice is now an explicit ablation rather than an
assumption. A second run uses a five-dimensional CNN output and compares it
with all five coordinates of `Z_task`.

## Verified Reference Results

```text
model   subject      Procrustes R2    RSA Spearman    RSA Pearson
PCA     subject_01      -1.642           0.020           0.020
PCA     subject_02      -1.655           0.019           0.021
CNN1D   subject_01       0.413           0.537           0.524
CNN1D   subject_02       0.418           0.539           0.527
```

## CNN output-dimensionality ablation

The 3D and 5D CNNs use the same simulation, temporal windows, loss, seed, and
training settings. Only the output dimension changes.

For a dimension-independent comparison against the complete 5D latent, RSA is:

```text
subject      CNN 3D RSA Spearman    CNN 5D RSA Spearman
subject_01          0.423                   0.435
subject_02          0.424                   0.482
```

The 5D CNN can also be compared coordinate-wise with the full 5D latent:

```text
subject      Procrustes R2    RSA Spearman    RSA Pearson
subject_01       0.308           0.435           0.474
subject_02       0.395           0.482           0.523
```

### Does the CNN recover the circular clock?

This question is evaluated on the eight condition-averaged endpoints at the
final time bin. It requires several diagnostics:

- planarity: fraction of endpoint variance in the first two singular axes;
- isotropy `S2/S1`: one for equal in-plane spread, zero for line collapse;
- radial coefficient of variation: zero for equal endpoint radii;
- angular-order coherence: one when condition angles occur in the expected
  circular order, allowing a global rotation or reflection;
- circular-distance RSA: agreement between endpoint distances and circular
  label distances.

```text
subject      model    planarity   S2/S1   radial CV   angular order
subject_01   CNN 3D      1.000     0.108      0.342        0.655
subject_01   CNN 5D      0.990     0.952      0.029        1.000
subject_02   CNN 3D      1.000     0.111      0.436        0.637
subject_02   CNN 5D      0.998     0.977      0.013        1.000
```

The 5D CNN therefore recovers the endpoint circular organization much more
faithfully than the 3D bottleneck. It does **not** recover the common center of
the clock. The ratio between initial and final condition-centroid dispersion
is approximately `0.97--0.99` for the CNN, compared with `0.063` for the full
ground truth and less than `0.01` for ground-truth position alone. The learned
condition trajectories are separated from the beginning and are approximately
parallel.

This failure is consistent with the additive target distance:

```text
D = w_time D_time + w_condition D_condition
```

At the start of a trial, `D_condition` still separates different directions
even though their ground-truth positions coincide. The corresponding
exponential affinity factorizes into temporal and condition terms, favoring a
product geometry rather than the interaction
`s(t) [cos(theta), sin(theta)]` required by a radial clock.

The complete 5D trajectory is therefore not recovered: global RSA and
Procrustes remain moderate, and the common-origin test fails.

The circular condition distance is part of the soft contrastive training
target. Consequently, this result shows successful realization of supplied
task geometry from neural observations, not fully unsupervised discovery of a
circle. Held-out trials, shuffled labels, and time-only/condition-only
ablations are required before making a stronger claim.

Detailed metric definitions and machine-readable values are stored in:

```text
outputs/first_experiment_24_07/cnn_5d/METRICS.md
outputs/first_experiment_24_07/cnn_5d/geometry_metrics.csv
outputs/first_experiment_24_07/cnn_5d/geometry_metrics.json
```

## Topological and symmetry-aware extensions

Topological data analysis can add a coordinate-free test of circular structure.
For condition endpoints, a persistent one-dimensional homology class (`H1`)
would support the presence of a loop across a nontrivial range of distance
scales. This complements, but does not replace, RSA and Procrustes.

Persistent homology must not initially be computed indiscriminately over every
trajectory point. Radial trajectories starting from a common origin can fill
a disk-like or spoke-like set whose topology differs from the endpoint circle.
The first analysis should therefore compare:

1. ground-truth and learned condition endpoints;
2. fixed-progress slices;
3. shuffled-condition and noise controls;
4. persistence diagrams across seeds and held-out trials.

The circular task has a discrete `C8` rotational symmetry, approximating
continuous `SO(2)` symmetry. Geometric deep learning could exploit it through:

- a `C8`-equivariant encoder or regularizer;
- group-aware augmentation by circular condition shifts;
- a sparse graph whose nodes are windows and whose edges encode temporal,
  trial, and task neighborhoods;
- graph contrastive learning on those edges instead of all `B x B` pairs.

The graph formulation also addresses the quadratic cost of dense pairwise
similarities. It changes the computation from all pairs to a controlled sparse
edge set.

Symmetry-aware training is a model hypothesis, not neutral evidence. If `C8`
equivariance or an `H1` topological loss is imposed during training, successful
circular recovery cannot be presented as unconstrained discovery. The proper
comparison is unconstrained CNN versus symmetry-aware CNN under identical
held-out evaluation.

The first five PCA components explain approximately 1.09% and 0.82% of the
window variance for subjects 1 and 2.

## Interpreting the PCA Result

The poor PCA result is not caused only by extending the latent space from three
to five coordinates.

A diagnostic PCA applied at different generative stages produced:

```text
input to PCA       subject_01 RSA    subject_02 RSA
linear drive u          0.940             0.859
rate lambda             0.858             0.789
observed counts X       0.020             0.019
```

The latent geometry is present in the neural drive and remains visible after
the rate nonlinearity. It becomes difficult for variance-maximizing linear PCA
to recover from raw sparse stochastic counts.

Similar marginal spike-count distributions do not imply similar population
covariances. PCA depends on the latter.

## Limitations

- Evaluation is in-sample.
- Only one random seed is reported.
- No held-out trial or held-out condition evaluation is included.
- The imposed lag is generated but not recovered in this file.
- PCA is not Poisson-aware and receives untransformed sparse counts.
- The CNN loss uses task metadata and may favor the geometry specified by that
  metadata.
- Hyperparameters have not been selected on an independent validation set.

## Troubleshooting

### Simulation file missing

Run Stage 1 before Stage 2.

### Windows PCA crash without traceback

Verify:

```text
numpy          1.26.4
scipy          1.13.1
scikit-learn   1.5.2
```

Mixed incompatible Conda and pip binary builds can crash inside BLAS/LAPACK.

### Wrong package imported

From the repository root:

```bash
python -m pip install -e . --no-deps
python -c "import neurobridge; print(neurobridge.__file__)"
```

The printed path should point to this repository's `src/neurobridge`.
