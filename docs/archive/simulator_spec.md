# NeuroBridge Spike Simulator Specification

## Goal

The simulator generates observable neural time series from controlled task states.
It is a statistical observation simulator, not a biophysical simulator of ion
channels or membrane dynamics.

Its purpose is to create benchmark data where the ground truth is known:

- task/state variables;
- latent trajectories;
- neuron tuning parameters;
- firing rates;
- spike-count observations.

This allows NeuroBridge to test whether representation learning models recover
temporal, behavioral, and latent structure from neural observations.

## Core Architecture

The simulator is organized in three conceptual layers.

```text
Task/state simulator
    -> neural tuning/rate simulator
    -> spike observation simulator
```

### 1. Task/State Simulator

This layer generates what happens in the task or behavior.

It should return a time-dependent state:

```text
s_i(t)
```

for trial `i` and time bin `t`.

The state can include macro and micro variables.

### Macro Variables

Macro variables describe trial-level or context-level structure:

- condition;
- target;
- global movement direction;
- trial type;
- task context;
- session;
- animal/subject.

Examples:

```text
circular reaching task:
    condition = movement direction among 8 possible directions

linear track task:
    condition = left-to-right or right-to-left traversal
```

### Micro Variables

Micro variables describe the instantaneous state inside the trial:

- time in trial;
- phase;
- position;
- velocity;
- local direction;
- distance from target;
- acceleration;
- latent state.

Example for a rat on a linear track:

```text
position(t)  = position along the track
velocity(t)  = movement speed
direction(t) = +1 or -1
phase(t)     = normalized progression through the trial
```

The key idea is:

```text
the same macro condition can contain many different micro states.
```

For example, a rat moving along a straight track is not exposed to one single
stimulus. Each position and movement phase defines a different behavioral state.

## 2. Neural Tuning / Rate Simulator

This layer maps task state to neural drive and firing rate.

For neuron `j`:

```text
u_j(t) = baseline_j + tuning_j(s(t)) + latent/noise terms
```

Then:

```text
lambda_j(t) = nonlinearity(u_j(t))
```

Typical nonlinearities:

- softplus;
- exponential;
- saturating sigmoid.

### Tuning Families

Different neuron families should be supported.

#### Direction-Tuned Neurons

The neuron responds preferentially to one direction.

Useful for:

- reaching tasks;
- circular motor tasks;
- direction-selective motor or sensory neurons.

Parameters:

- preferred direction;
- directional gain;
- tuning width;
- baseline.

#### Place-Like Neurons

The neuron responds preferentially to one position.

Useful for:

- rat moving on a linear track;
- spatial trajectories;
- position-dependent firing.

Parameters:

- preferred position;
- place-field width;
- gain;
- baseline.

#### Speed-Tuned Neurons

The neuron responds to movement speed.

Parameters:

- preferred speed or monotonic speed gain;
- gain;
- baseline.

#### Conjunctive Neurons

The neuron responds to combinations of state variables.

Examples:

```text
position + direction
position + speed
direction + phase
condition + time
```

Example:

```text
neuron fires near position 0.4 only when movement direction is left-to-right.
```

These neurons are important because real neural populations often encode
mixed selectivity rather than one isolated variable.

## 3. Spike Observation Simulator

This layer maps firing rates to observed spike counts.

Base model:

```text
X_i,t,j ~ Poisson(lambda_i,t,j * dt)
```

where:

- `X_i,t,j` is the spike count;
- `lambda_i,t,j` is the firing rate;
- `dt` is the bin duration.

The output tensor is:

```text
X.shape = (n_trials, trial_len, n_neurons)
```

Each entry is:

```text
X[i, t, j] = spike count of neuron j in time bin t of trial i
```

### Observation Realism

The simulator should support increasing levels of spike-train realism.

#### Poisson Emission

The simplest emission model:

```text
mean = variance
```

Useful for clean benchmarks.

#### Overdispersion

Real spike counts often have:

```text
variance > mean
```

This can be modeled using a stochastic rate multiplier, for example a
Gamma-Poisson mechanism.

#### Refractory Period

After a spike, the neuron has reduced probability of firing again.

In a bin-level simulator, this can be approximated by forcing a number of
following bins to zero.

The refractory duration can be sampled from a normal distribution:

```text
refractory_bins ~ round(N(mean, std))
```

truncated at zero.

#### Bursts / Clusters

Some spikes can trigger short local clusters of extra spikes.

This is a phenomenological way to model bursting without introducing a full
biophysical neuron model.

#### History Dependence

The firing probability can depend on recent spikes:

```text
lambda(t) = f(state(t), spike_history(t))
```

Refractoriness and bursting are special cases of history dependence.

## Required Outputs

A complete simulation should return both observations and ground truth.

Recommended output fields:

```text
X                 spike-count observations
Z                 latent trajectories, if present
u                 neural drive
lambda            firing rates
condition         trial-level condition
time_id           time or phase within trial
trial_id          trial index
position          optional position state
velocity          optional velocity state
direction         optional direction state
phase             optional phase state
neuron_metadata   preferred directions, positions, gains, baselines
config            simulator parameters
```

## Multi-Subject Shared Latent Setting

For NeuroBridge, an important benchmark is the recovery of a shared latent/task
space from different neural observations.

The simulator should therefore support:

```text
shared task state / shared latent Z
    -> subject-specific tuning and emission
    -> X_subject_1, X_subject_2, ...
```

In this setting, subjects perform or observe the same motor task structure, but
their neural populations need not have the same observed coordinates.

Example:

```text
Z(t), condition, phase, position are shared
B_subject_1 and B_subject_2 are different
lambda_subject_1 and lambda_subject_2 are different
X_subject_1 and X_subject_2 are different
```

This creates a controlled test for whether representation learning can recover
a common latent/task geometry from multiple subject-specific spike matrices.

The current circular simulator supports this pattern experimentally by creating
two subject-specific loading matrices and two spike-count tensors from the same
latent trajectory.

### Temporal Lag Between Subjects

In interactive or social motor settings, two subjects may share the same task
structure but show a small temporal offset in neural activity.

Example:

```text
subject 1 acts now
subject 2 observes/responds with a small delay
```

The simulator can model this by applying a subject-specific lag to the latent
trajectory before emission:

```text
Z_subject_s(t) = Z_shared(t - lag_s)
```

The task labels and global task structure remain shared, but the neural
observations are temporally shifted.

This creates an explicit benchmark for models that must recover shared geometry
despite small cross-subject timing offsets.

### Expected Cross-Subject Ambiguities

Even if two encoders recover the same latent/task space, their coordinates need
not be identical. They may differ by:

- translation;
- rotation;
- reflection;
- global scaling;
- anisotropic scaling;
- axis permutation;
- temporal lag;
- mild nonlinear warping;
- nuisance dimensions related to subject/session/firing-rate statistics.

Therefore cross-subject recovery should not be evaluated only by raw coordinate
equality.

### Alignment and Shared-Space Evaluation

Recommended evaluation levels:

```text
Level 1: Centering and scaling
Level 2: Orthogonal Procrustes alignment
Level 3: RSA / distance-matrix correlation
Level 4: CKA / CCA
Level 5: temporal alignment, e.g. lag search or dynamic time warping
Level 6: optimal transport / Gromov-Wasserstein for harder unmatched settings
```

The first benchmark should use conservative alignment:

```text
Procrustes + residual error
RSA correlation between distance matrices
latent recovery after alignment
```

More flexible alignment methods should be added only after simpler geometric
tests are understood, because overly flexible alignment can hide real failures
of the representation model.

## Validation Metrics

The simulator should report whether generated data are plausible and whether the
task structure is recoverable.

Recommended diagnostics:

- mean firing rate;
- spike-count sparsity;
- Fano factor;
- inter-spike interval distribution;
- refractory violation rate;
- burst statistics;
- condition tuning curves;
- place fields, if position is simulated;
- PCA geometry;
- latent-neural distance correlation;
- decoding of condition/state from `X`.

## Simulator Levels

The simulator should be developed in levels.

### Level 1: Clean Latent Poisson

```text
Z(t) -> lambda(t) -> Poisson spike counts
```

Purpose:

- verify the pipeline under favorable conditions;
- test latent recovery.

### Level 2: Realistic Spike Statistics

Adds:

- overdispersion;
- refractory period;
- bursts.

Purpose:

- test robustness to realistic spike-train deviations from Poisson.

### Level 3: Explicit Task State

Adds:

- position;
- velocity;
- phase;
- local direction;
- trial context.

Purpose:

- move from abstract latent variables to motor/task states.

### Level 4: Population and Session Complexity

Adds:

- mixed selectivity;
- neuron-specific baselines and gains;
- shared population noise;
- session drift;
- condition imbalance;
- uninformative neurons.

Purpose:

- stress-test representation learning methods under nuisance variability.

### Level 5: Benchmark Generator

Adds:

- saved datasets;
- metadata;
- diagnostic reports;
- controlled difficulty presets.

Purpose:

- make NeuroBridge usable as a research benchmark framework.

## Connection to NeuroBridge Learning

The simulator should produce the metadata needed to construct structured
similarities between windows:

```text
D_total = w_time D_time
        + w_condition D_condition
        + w_position D_position
        + w_velocity D_velocity
        + ...
```

Then:

```text
S = exp(-D_total / tau)
```

This supports soft contrastive learning where pairs are not only positive or
negative, but similar to different degrees according to task geometry.

## Current Learning Components

The first internal NeuroBridge learning stack is intentionally minimal and
designed to keep all sampling and loss assumptions inspectable.

Implemented components:

```text
TemporalWindowDataset
TemporalCNNEncoder
TemporalMLPEncoder
TemporalLSTMEncoder
TemporalTransformerEncoder
supervised_infonce_loss
soft_contrastive_loss
batch_structured_similarity
train_epoch
encode_windows
```

The default research path is:

```text
X_windows
    -> TemporalWindowDataset
    -> TemporalCNNEncoder
    -> embeddings
    -> batch_structured_similarity(time, label)
    -> soft_contrastive_loss
```

NeuroBridge keeps a small, inspectable implementation so that assumptions about
distances, similarities, and losses can be modified directly.

## Current Controlled Experiments

The four executable notebooks under `notebooks/` compare PCA and CNN1D on
circular and linear tasks with essential and enriched latent states. The CNN
uses the soft structured contrastive objective.

Reported metrics:

```text
Procrustes R2 against latent Z
RSA Spearman correlation
RSA Pearson correlation
cross-subject lag-aware Procrustes alignment
```

Each notebook saves outputs under:

```text
outputs/<experiment-name>/
```

The figures include known latent trajectories, native PCA coordinates, native
CNN coordinates, and task-specific diagnostics such as linear-track place
fields.

Current baseline windowing defaults are:

```text
window_size = 10
stride = 1
centered padding = enabled
```

The absolute time mode is used for lag-aware subject comparison. With the
current synthetic setup, subject 2 is generated with a lag of two bins; the
neural encoders recover this offset by selecting `best_lag = 2` under
trial/time-aware Procrustes alignment.
