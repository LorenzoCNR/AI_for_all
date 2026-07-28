# Generative Model

## Purpose And Scope

NeuroBridge generates neural observations from a known low-dimensional task
state. The known state provides ground truth for testing representation
learning:

```text
task state -> latent trajectory Z -> neural drive u
           -> firing rate lambda -> spike counts X
```

This is an observation-level statistical simulator. It does not simulate ion
channels, membrane voltage, or the biophysical cause of individual action
potentials.

## Tensor Definitions

Let:

- `R` be the number of trials;
- `T` be the number of time bins per trial;
- `K` be the latent dimension;
- `N` be the number of simulated neurons.

The main tensors are:

```text
Z       : (R, T, K)  known task-level latent state
B       : (K, N)     latent-to-neuron loading matrix
c       : (N,)       neuron baseline
u       : (R, T, N)  neural drive
lambda  : (R, T, N)  firing rate in spikes/second
X       : (R, T, N)  observed spike counts
```

For the controlled notebooks, `R=160`, `T=100`, and `N=100`.

## Deterministic Task State And Trial Variability

For trial `r`, the latent trajectory is

```text
Z_r(t) = m_r(t) + eta_r(t),
```

where `m_r(t)` is the deterministic task trajectory and `eta_r(t)` is
temporally correlated trial variability:

```text
eta_r(t) = phi * eta_r(t-1) + epsilon_r(t),
epsilon_r(t) ~ Normal(0, noise_scale^2 I).
```

The default controlled configuration uses `phi=0.4` and
`noise_scale=0.05`. This creates smooth trial-to-trial perturbations without
changing the known task family.

## Circular Reaching Task

Each trial starts from a common center and reaches one of eight targets. For
condition angle `theta`, the planar trajectory is

```text
x(t) = s(t) cos(theta)
y(t) = s(t) sin(theta),
```

where the smooth movement profile is

```text
s(t) = 10 t^3 - 15 t^4 + 6 t^5,  t in [0,1].
```

This profile starts and ends with zero velocity and acceleration. It is a
controlled minimum-jerk-like interpolation, not a complete model of biological
reaching.

The essential circular latent is:

```text
Z(t) = [x(t), y(t), progress(t)].
```

The enriched version adds velocity and trial context:

```text
Z(t) = [x(t), y(t), progress(t), velocity(t), context].
```

## Linear Track Task

One trial contains both movement phases:

```text
outbound: position 0 -> 1, direction +1
return:   position 1 -> 0, direction -1
```

The physical behavior is one-dimensional, but the essential latent is
two-dimensional:

```text
Z(t) = [position(t), direction(t)].
```

This distinction separates two visits to the same position made in opposite
directions. In the position-direction plane, the outbound and return branches
form a cycle. The enriched state adds velocity and context.

## Neural Population Map

The linear part of the population observation model is

```text
u_r(t) = Z_r(t) B + c + g_r(t),
```

where `g_r(t)` is an optional nonlinear task-specific drive. Each column of
`B` describes how one neuron weights the latent coordinates.

The population is heterogeneous. Neurons can emphasize direction, progress,
position, velocity, context, or mixtures. The
`first_coordinates_multiplier` parameter increases loading magnitude for the
task-defining coordinates; the controlled default is `3.0`.

This is an experimental control, not a discovered biological law. Increasing
it raises the signal-to-noise ratio of the motor core and can make recovery
easier.

## Linear-Track Place Fields

A subset of linear-track neurons receives a localized Gaussian drive:

```text
g_j(p) = a_j exp(-(p-mu_j)^2 / (2 sigma_j^2)).
```

Here:

- `mu_j` is neuron `j`'s preferred track position;
- `sigma_j` is its field width;
- `a_j` is its gain.

The controlled defaults are `place_fraction=0.25`,
`place_width=0.10`, and `place_scale=3.0`.

The place term remains separate from `ZB` because a localized Gaussian field
is not linear in position. It also provides neuron-level ground truth for a
future explainability test: an attribution method should identify the neurons
whose preferred fields cover the queried position. That XAI analysis is not
yet part of the validated suite.

## From Drive To Firing Rate

The controlled notebooks use

```text
lambda = rate_scale * softplus(u),
softplus(u) = log(1 + exp(u)).
```

`softplus` guarantees positive rates while remaining smooth. The default
`rate_scale=10.0` gives `lambda` units of spikes/second.

The baseline vector is sampled around `baseline_mean=1.0` with
`baseline_std=0.10`. Changing these quantities changes the overall rate regime
and therefore the difficulty of recovery.

## From Rate To Spike Counts

For bin width `dt`, the base observation model is

```text
X_r,t,j ~ Poisson(lambda_r,t,j * dt).
```

The controlled suite uses `dt=0.02` seconds, so one trial contains 100 bins
covering two seconds. `X` contains integer counts per bin, not exact spike
times.

Optional mechanisms are implemented:

- **Overdispersion:** a Gamma-distributed multiplicative rate gain is sampled
  before the Poisson count, producing variance above the Poisson mean.
- **Bursting:** a nonzero count can trigger additional short-lived counts with
  configurable probability and intensity.
- **Refractory suppression:** after a nonzero bin, subsequent bins can be
  suppressed for a normally distributed number of bins.

These mechanisms are phenomenological. They make observations statistically
richer but do not make the simulator biophysical. Importantly, the four
controlled notebooks leave them disabled and therefore use pure Poisson
emission unless the call parameters are changed.

## Subject-Specific Populations And Lag

The same task generator can produce a shared task state for two subjects while
using different loading matrices, baselines, neuron counts, and stochastic
emissions. Thus the task geometry is common but the neural observations are
not identical.

A positive temporal lag shifts one subject's latent trajectory later within
each trial. Boundary values are padded and samples never wrap into another
trial. If the lag is 10 bins and `dt=0.02`, the imposed delay is 200 ms.

This creates controlled ground truth for asking whether learned representations
recover both shared geometry and temporal asymmetry.

## Main Assumptions And Limits

- The latent state is specified by the researcher rather than inferred from
  biology.
- The linear map `B` is deliberately interpretable but simplified.
- Pure Poisson emission has Fano factor near one and lacks history dependence.
- Optional refractory and burst mechanisms operate at bin resolution.
- Results depend on rate scale, neuron tuning mixture, latent noise, bin width,
  and trial count.
- Good recovery can demonstrate estimator behavior under known conditions; it
  cannot establish that the same latent process generated real neural data.
