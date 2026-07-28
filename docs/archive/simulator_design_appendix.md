# Simulator Design Appendix

This appendix explains why the simulator is built the way it is, and how the
main mathematical objects communicate with each other.

The core goal is not to simulate the biophysical cause of each spike. The goal
is to simulate observable spike-count data generated from a known task/latent
structure, so representation learning methods can be evaluated against ground
truth.

```text
task state -> latent Z -> neural drive u -> firing rate lambda -> spike counts X
```

## 1. What The Simulator Observes

The observed object is:

```text
X[i, t, j]
```

meaning:

```text
spike count of neuron j
in time bin t
of trial i
```

Shape:

```text
X.shape = (n_trials, trial_len, n_neurons)
```

This is not a list of exact spike times. It is a binned spike-count tensor.

**Rationale.** Neural spike trains can be represented as point processes in
continuous time, but most downstream machine-learning pipelines operate on
discretized time bins or windows. The simulator therefore generates binned spike
counts.

**Reference.** Dayan and Abbott (2001), Chapter 1, introduce spike trains as
point processes and firing rates/spike-count rates as summaries derived from
spike events.

**Implementation.** `X` is generated as an integer tensor with shape
`(n_trials, trial_len, n_neurons)`.

**Limitation.** Exact spike timing within each bin is not represented.

## 2. Task State

The task state is the simulated motor/behavioral structure that generates the
neural activity.

For the current circular reaching task:

```text
condition       = trial-level movement direction
phase(t)        = normalized movement progress
position(t)     = 2D position along the reaching trajectory
direction_angle = angle associated with the condition
direction_vector = [cos(theta), sin(theta)]
```

The important distinction is:

```text
macro-state = condition / target / trial context
micro-state = time, phase, position, velocity, local state inside the trial
```

In sensory neuroscience this object is often called a stimulus. In motor tasks,
it is clearer to call it a behavioral or task state.

**Rationale.** In a motor task, the variable modulating neural activity is not
only a discrete label. It can include movement direction, position, phase,
velocity, target, or task context.

**Reference.** Dayan and Abbott (2001) discuss stimulus-response relationships
and tuning curves. For motor settings, the analogous object is a time-dependent
behavioral/task state rather than only a sensory stimulus.

**Implementation.** The current circular simulator stores `phase`, `position`,
`direction_angle`, and `direction_vector` in `task_state`.

**Limitation.** The current task state does not yet include explicit cue, delay,
hold, velocity, or reaction-time periods.

## 3. Why Use The Movement Profile s(t)?

The current movement profile is:

```text
s(t) = 10 t^3 - 15 t^4 + 6 t^5,   t in [0, 1]
```

Equivalently:

```text
s(t) = 6 t^5 - 15 t^4 + 10 t^3
```

This is a quintic smooth transition from 0 to 1.

It has useful boundary properties:

```text
s(0) = 0
s(1) = 1
s'(0) = 0
s'(1) = 0
s''(0) = 0
s''(1) = 0
```

Interpretation:

- starts at position 0;
- ends at position 1;
- starts and ends with zero velocity;
- starts and ends with zero acceleration;
- creates a smooth bell-shaped velocity profile.

This is useful for motor tasks because reaching movements are often modeled as
smooth point-to-point trajectories. The same polynomial appears in minimum-jerk
trajectory models and in smooth interpolation functions.

In this simulator, `s(t)` is not claiming to be the full biology of movement.
It is a controlled way to create a smooth movement phase.

**Rationale.** A reaching trajectory should not jump abruptly from start to end.
The quintic profile creates smooth onset and offset, which is more plausible
than a linear ramp for point-to-point movement.

**Reference.** Flash and Hogan (1985) proposed that human arm movements can be
modeled by minimizing jerk, leading to smooth point-to-point trajectories with
bell-shaped velocity profiles. The polynomial used here is the standard
minimum-jerk/smootherstep profile satisfying zero velocity and acceleration at
the boundaries.

**Implementation.** `_build_movement_profile` in
`Lat_traj_generator.py` returns `t` and `s`.

**Limitation.** This is a stylized movement profile. Real movement can include
reaction time, corrections, pauses, variable speed, and task-specific kinematic
features.

## 4. Circular Task Geometry

For condition `c`, define:

```text
theta = 2 pi c / n_conditions
```

Then:

```text
direction_vector = [cos(theta), sin(theta)]
```

The deterministic 2D position is:

```text
position(t) = s(t) * direction_vector
```

So each trial moves smoothly from the origin toward a direction-specific target
on a circle.

This makes the circular motor geometry explicit.

**Rationale.** If the task is directional reaching, the natural geometry of
conditions is circular. Encoding directions with sine and cosine avoids treating
adjacent directions as unrelated categories.

**Reference.** Direction tuning in motor cortex is often represented using
preferred directions and cosine-like tuning. Dayan and Abbott (2001) introduce
tuning curves as mappings from task/stimulus variables to average firing rate.

**Implementation.** `deterministic_builder` uses `cos(theta)` and `sin(theta)`
for the first two latent coordinates.

**Limitation.** This assumes circular geometry. It should not be used unchanged
for qualitative categorical labels or linear-track position.

## 5. Latent Trajectory Z

The latent trajectory is:

```text
Z_i(t) = m_i(t) + eta_i(t)
```

where:

```text
m_i(t)   = deterministic task trajectory
eta_i(t) = stochastic trial-to-trial variability
```

Current shape:

```text
Z.shape = (n_trials, trial_len, n_traj_k)
```

For the current circular task:

```text
Z[:, :, 0] relates to phase * cos(theta)
Z[:, :, 1] relates to phase * sin(theta)
Z[:, :, 2] relates to movement phase/progress
```

Thus, `Z` is a low-dimensional latent state containing both directional and
temporal structure.

**Rationale.** The latent state is the ground truth representation we want
algorithms to recover from neural observations. It gives the simulator a known
answer.

**Reference.** Latent neural dynamics are central in neural population modeling
and representation learning.

**Implementation.** `LatentTrajectoryGenerator.generate_latent` returns `Z`,
condition labels, and optional `task_state`.

**Limitation.** The current latent state is designed by hand. It is useful for
controlled benchmarks, but it is not inferred from real data.

## 6. Stochastic Latent Variability

The noise term is AR(1)-like:

```text
eta(t) = phi eta(t-1) + epsilon(t)
```

where:

```text
epsilon(t) ~ Normal(0, noise_scale^2)
```

Why use AR(1)?

- neural and behavioral states are temporally smooth;
- consecutive bins should not be independent white noise;
- `phi` controls temporal persistence;
- `noise_scale` controls trial-to-trial difficulty.

If `noise_scale` increases, recovering the true latent/task geometry becomes
harder.

**Rationale.** Neural and behavioral trajectories are temporally correlated.
AR(1) noise creates smooth variability across time rather than independent
noise at every bin.

**Reference.** Autoregressive and state-space processes are standard tools for
time-series and latent-dynamics modeling.

**Implementation.** `stochastic_builder` in `builders.py` creates the AR(1)
component.

**Limitation.** AR(1) is simple. It does not model richer latent dynamics,
switching states, or long-memory processes.

## 7. Loading Matrix B

The loading matrix maps the latent state into neural population space:

```text
u = Z @ B + c
```

Shape:

```text
B.shape = (n_traj_k, n_neurons)
```

Each column of `B` corresponds to one neuron.

B is how the latent space becomes a neural population observation.

```text
B[:, j] = how neuron j reads the latent state
```

### Directional Scale

For the first two latent dimensions:

```text
B[0, j] = directional_scale * cos(phi_j)
B[1, j] = directional_scale * sin(phi_j)
```

`directional_scale` controls how strongly neurons encode the circular motor
geometry.

Large `directional_scale`:

- clearer direction tuning;
- easier latent recovery;
- stronger condition separation.

Small `directional_scale`:

- weaker task signal;
- harder recovery.

### Extra Scale

For extra latent dimensions:

```text
B[2:, :] = extra_scale * random_normal
```

`extra_scale` controls how much additional latent variability enters the neural
population.

Large `extra_scale`:

- more non-directional structure;
- more difficult embedding problem.

Small `extra_scale`:

- cleaner circular geometry.

**Rationale.** A low-dimensional latent state is not directly observed. The
loading matrix determines how each neuron samples or reads that latent state.

**Reference.** Linear loading matrices are standard in latent variable models
such as factor analysis, PCA-like observation models, and many neural population
models. Preferred-direction structure connects to classical tuning-curve ideas.

**Implementation.** `build_structured_B` creates a structured loading matrix
with directionally tuned first two rows and random extra-dimensional loadings.

**Limitation.** Current `B` is imposed, not learned. It also assumes a linear
mapping from latent state to drive before the rate nonlinearity.

## 8. Neural Drive u

The neural drive is:

```text
u[i, t, j] = Z[i, t, :] dot B[:, j] + c[j]
```

This is not yet a firing rate. It can be negative and unconstrained.

`c[j]` is the baseline drive of neuron `j`.

Currently:

```text
c = ones(n_neurons)
```

Future versions should allow neuron-specific baselines.

**Rationale.** The drive combines task-related latent modulation with baseline
activity. This separates structured task signal from background firing tendency.

**Reference.** Generalized linear models for spiking often combine stimulus or
covariate filters with baseline terms before applying a rate nonlinearity.

**Implementation.** `latent_to_drive` computes `u = Z @ B + c`.

**Limitation.** Current baseline is identical across neurons unless manually
changed.

## 9. Firing Rate lambda

The firing rate must be positive:

```text
lambda = nonlinearity(u)
```

Currently supported nonlinearities:

```text
softplus
exponential
```

### Softplus

```text
softplus(u) = log(1 + exp(u))
```

Why softplus?

- always positive;
- smooth;
- more numerically stable than exponential;
- does not explode as aggressively for large positive values.

This is the current default.

### Exponential

```text
lambda = exp(u)
```

This is common in Poisson GLMs, but it can produce very large rates if the drive
is large.

For the current simulator, `softplus` is safer.

**Rationale.** Firing rates must be non-negative. A nonlinearity transforms the
unconstrained drive into a valid rate.

**Reference.** Poisson GLMs often use exponential nonlinearities. Softplus is a
smooth positive alternative that is numerically less explosive.

**Implementation.** `drive_to_rate` supports `softplus` and `exponential`.

**Limitation.** Current nonlinearities are not saturating. Real neurons have
physiological limits on firing rate.

## 10. Spike Count Observation X

Base model:

```text
X[i, t, j] ~ Poisson(lambda[i, t, j] * dt)
```

where:

```text
dt = duration of one time bin
```

If `lambda` is in spikes/second, then `dt` is in seconds.

Example:

```text
lambda = 10 Hz
dt = 0.02 s
lambda * dt = 0.2 expected spikes per bin
```

Thus `dt` controls the time scale of the observation.

**Rationale.** The Poisson spike generator links firing rate to spike count
probability over a finite interval.

**Reference.** Dayan and Abbott (2001), Poisson Spike Generator section:
for a small interval `dt`, the probability of a spike is approximately
`r(t) dt`. The binned-count version is
`X ~ Poisson(lambda dt)`.

**Implementation.** `rate_to_spike` samples from `np.random.poisson(lam * dt)`.

**Limitation.** Pure Poisson assumes conditional independence given the rate and
implies equal mean and variance.

## 11. Observation Realism

The base Poisson model is extended with optional phenomenological corrections.

### Overdispersion

Poisson implies:

```text
variance = mean
```

Real spike counts can have:

```text
variance > mean
```

The simulator can multiply the rate by a Gamma random factor with mean 1.

This creates extra trial/bin variability.

**Rationale.** Real spike counts often show variability larger than the pure
Poisson prediction.

**Reference.** Dayan and Abbott (2001) discuss spike-count variability, Fano
factor, and deviations from the pure Poisson model.

**Implementation.** `rate_to_spike` can multiply the rate by a Gamma random
gain before Poisson sampling.

**Limitation.** This is a phenomenological correction, not a mechanistic source
of variability.

### Refractory Period

After a spike, the neuron may be unable or less likely to fire again for a short
time.

The simulator approximates this by forcing following bins to zero for a sampled
duration:

```text
refractory_bins ~ round(N(mean, std))
```

**Rationale.** Immediately after a spike, neurons can be unable or less likely
to spike again. This reduces unrealistically short inter-spike intervals.

**Reference.** Dayan and Abbott (2001) discuss refractory effects as an
important deviation from the simple Poisson model.

**Implementation.** `_apply_refractory_period` zeros following bins for a
sampled refractory duration.

**Limitation.** Bin-level refractory correction is approximate. It does not
model membrane dynamics.

### Bursts / Clusters

Some spike bins can trigger extra nearby spikes.

This is a simple phenomenological model of local spike clustering.

**Rationale.** Some neurons produce local clusters of spikes rather than
isolated independent events.

**Reference.** Dayan and Abbott (2001) discuss bursting and Cox-like processes
as ways to go beyond simple Poisson firing.

**Implementation.** `_add_bursts` adds extra spikes in a short local temporal
window.

**Limitation.** The burst model is deliberately simple and does not model
burst-generating ion-channel mechanisms.

## 12. Multi-Subject Shared Latent Simulation

The simulator can generate two or more subjects from the same task/latent
structure.

Core idea:

```text
shared Z and task state
subject-specific B, lambda, X
```

For subject `s`:

```text
u_s = Z_s @ B_s + c_s
lambda_s = nonlinearity(u_s)
X_s ~ Poisson(lambda_s * dt)
```

This creates:

```text
X_subjects["subject_1"]
X_subjects["subject_2"]
```

The scientific question is:

```text
Can representation learning recover the same latent/task space from different
subject-specific spike matrices?
```

**Rationale.** The simulator should test shared latent recovery, not only
single-subject decoding. Two subjects can share task geometry while their neural
populations have different observed coordinates.

**Reference.** Cross-session and cross-subject latent alignment is an active
problem in neural population analysis. Gallego et al. (2023) discuss alignment
of latent neural representations.

**Implementation.** The script creates subject-specific `B`, `lambda`, and `X`
from shared task structure.

**Limitation.** Current subjects are simulated with the same number of neurons
and same task trials. Future benchmarks should relax this.

## 13. Temporal Lag Between Subjects

For interactive motor settings, one subject may be slightly delayed relative to
another.

The simulator supports:

```text
Z_subject_s(t) = Z_shared(t - lag_s)
```

Example:

```text
subject_1 lag = 0 bins
subject_2 lag = 2 bins
```

This creates a benchmark for lag-aware shared latent recovery.

**Rationale.** In acting/observing or interactive motor settings, neural
activity may be slightly shifted between subjects.

**Reference.** Temporal alignment problems are commonly treated with lag search
or dynamic time warping; see Sakoe and Chiba (1978) for DTW.

**Implementation.** `apply_temporal_lag` shifts `Z` for each subject without
wrapping across the trial.

**Limitation.** Current lag is fixed per subject. Real interaction may require
trial-specific or time-varying lags.

## 14. Windowing

The model observes windows rather than isolated time points.

Windowing produces:

```text
X_windows
time_id
global_time_id
trial_id
labels_windows
```

Interpretation:

```text
X_windows      = local neural time-series segments
time_id        = phase/time within trial
global_time_id = absolute time in concatenated data
trial_id       = which trial produced the window
labels_windows = condition of the trial/window
```

Window size is a scientific hyperparameter:

- too small: loses temporal context;
- too large: mixes states and may blur dynamics.

**Rationale.** Single time bins can be too sparse or noisy. Windows provide
local temporal context and match the receptive-field logic of temporal CNNs.

**Reference.** Dayan and Abbott (2001) discuss estimating firing rates with
temporal windows and kernels. Local temporal context can be represented through
offset or windowed neural input.

**Implementation.** `build_windows` creates trial-aware windows and prevents
windows from crossing trial boundaries.

**Limitation.** Window size and stride are hyperparameters and should be
validated empirically.

## 15. Distances and Similarity

The current similarity structure is built from distances between windows.

Example:

```text
D_total = w_time D_time + w_label D_label
```

Then:

```text
S = exp(-D_total / tau)
```

This means pairs are not simply positive/negative. They have graded similarity.

This is central to NeuroBridge:

```text
representation learning should preserve task/latent geometry, not only classify
labels.
```

**Rationale.** The learning signal should encode graded task geometry, not only
hard positive/negative pairs.

**Reference.** Contrastive learning commonly uses structured sampling of
positive and negative examples. NeuroBridge generalizes this toward a soft
similarity matrix based on task distances.

**Implementation.** `combine_distances` builds a weighted distance and
`dist_to_simi` converts it to similarity.

**Limitation.** Full pairwise matrices are O(n^2). Training should eventually
compute similarities batch-wise or sparsely.

## 16. Circular Label Distance

For circular motor conditions, labels have geometry.

Direction 1 and direction 8 are close on the circle.

Therefore:

```text
circular_distance(label_i, label_j)
```

is more appropriate than ordinary categorical mismatch.

This is task-specific. For purely qualitative labels, circular distance would
not be appropriate.

**Rationale.** Direction labels have circular topology. Treating them as
ordinary categories loses adjacency information.

**Reference.** Circular variables are standard in directional statistics and
motor tuning analyses.

**Implementation.** `circular_distance` computes wrap-around label distance.

**Limitation.** This distance is valid for circular tasks, not arbitrary labels.

## 17. Alignment Of Learned Latent Spaces

If two encoders recover the same latent space, their coordinates may differ by:

- rotation;
- reflection;
- translation;
- scaling;
- axis permutation;
- lag;
- mild warping.

Therefore, comparison should use alignment and geometry-preserving metrics:

```text
Procrustes alignment
RSA / distance-matrix correlation
CKA
CCA
lag search
DTW
optimal transport / Gromov-Wasserstein for harder cases
```

Initial benchmark:

```text
Procrustes + RSA + latent recovery
```

**Rationale.** Learned latent coordinates are not identifiable exactly. Two
representations can be equivalent up to rotation, reflection, scale, or lag.

**Reference.** Orthogonal Procrustes alignment (Schoenemann, 1966),
representational similarity analysis (Kriegeskorte et al., 2008),
hyperalignment (Haxby et al., 2011), and CKA (Kornblith et al., 2019) provide
ways to compare representations beyond raw coordinate equality.

**Implementation.** These metrics are not yet implemented in NeuroBridge, but
they should become the first evaluation module after encoder training.

**Limitation.** Very flexible alignment can hide model failure. Start with
conservative alignment.

## 18. What Must Be Recovered

The encoder sees:

```text
X_windows
```

It does not see `Z` during unsupervised/self-supervised representation learning.

But because this is a simulator, we can evaluate against:

```text
Z
phase
position
condition
subject lag
```

The goal is:

```text
embedding geometry approximately matches latent/task geometry
```

not merely:

```text
embedding predicts labels
```

## 19. Current Encoder And Loss Stack

The first NeuroBridge learning stack is intentionally small and inspectable.

Implemented encoders:

```text
TemporalCNNEncoder
TemporalMLPEncoder
TemporalLSTMEncoder
TemporalTransformerEncoder
```

Default encoder:

```text
TemporalCNNEncoder
```

because it captures local temporal context while remaining simple enough to
inspect.

Implemented losses:

```text
supervised_infonce_loss
soft_contrastive_loss
```

The soft contrastive loss is the NeuroBridge-specific direction:

```text
target similarity S_ij is graded, not binary
```

**Rationale.** Structured positive/negative sampling is a standard contrastive
strategy. NeuroBridge instead expresses task geometry as a soft batch-wise
similarity matrix.

**Implementation.** The current learning path is:

```text
TemporalWindowDataset
-> TemporalCNNEncoder
-> batch_structured_similarity
-> soft_contrastive_loss
-> train_epoch
```

**Limitation.** The controlled experiments currently validate PCA and CNN1D.
Systematic comparisons with the other available time-series encoders remain
future experiments.

The reproducible entry points are the four notebooks under `notebooks/`.
They report Procrustes R2 and RSA distance-geometry correlations against the
known latent state and save trial-averaged trajectory diagnostics.
PCA component pairs
cross-subject best-lag Procrustes alignment
```

These plots are intended to visually check whether the learned embedding
recovers the directional "clock" structure and whether subject-specific
embeddings align after temporal offset correction.

## 20. Current Default Configuration

Current working defaults:

```text
task: circular
condition_type: balanced
nonlinearity: softplus
dt: 0.02
directional_scale: 3
extra_scale: 0.051
overdispersion: 0.25
refractory_mean_bins: 2
refractory_std_bins: 1
burst_probability: 0.05
burst_size_mean: 1.5
burst_window_bins: 3
n_subjects: 2
subject_1 lag: 0
subject_2 lag: 2
window_size: 10
stride: 2
time_mode: absolute
```

The absolute time mode is important for lag search: relative time in `[0, 1]`
is useful as a behavioral coordinate, but trial-local bin indices are needed
to detect a two-bin simulated offset.

## 21. References

- Dayan, P. and Abbott, L. F. (2001). *Theoretical Neuroscience*. MIT Press.
- Flash, T. and Hogan, N. (1985). The coordination of arm movements: an
  experimentally confirmed mathematical model. *Journal of Neuroscience*.
  https://doi.org/10.1523/JNEUROSCI.05-07-01688.1985
- Schneider, S., Lee, J. H., and Mathis, M. W. (2023). Learnable latent
  embeddings for joint behavioural and neural analysis. *Nature*.
  https://doi.org/10.1038/s41586-023-06031-6
- Gallego, J. A., et al. (2023). Aligning latent representations of neural
  activity. *Nature Biomedical Engineering*.
  https://doi.org/10.1038/s41551-022-00962-7
- Schoenemann, P. H. (1966). A generalized solution of the orthogonal
  Procrustes problem. *Psychometrika*.
  https://doi.org/10.1007/BF02289451
- Kriegeskorte, N., Mur, M., and Bandettini, P. (2008). Representational
  similarity analysis. *Frontiers in Systems Neuroscience*.
  https://doi.org/10.3389/neuro.06.004.2008
- Haxby, J. V., et al. (2011). A common, high-dimensional model of
  representational space. *Neuron*.
  https://doi.org/10.1016/j.neuron.2011.08.026
- Kornblith, S., Norouzi, M., Lee, H., and Hinton, G. (2019). Similarity of
  neural network representations revisited. *ICML*.
  https://arxiv.org/abs/1905.00414
- Sakoe, H. and Chiba, S. (1978). Dynamic programming algorithm optimization
  for spoken word recognition. *IEEE Transactions on Acoustics, Speech, and
  Signal Processing*.
- Peyre, G. and Cuturi, M. (2019). Computational Optimal Transport.
  *Foundations and Trends in Machine Learning*.
  https://doi.org/10.1561/2200000073
