# Hasson Pipeline Review

This note summarizes the local `Hasson_pipeline` folder and how it relates to
NeuroBridge.

## Data Contract

Each trial structure is expected to contain:

- `Spikes`: neural activity, shaped as neurons/channels by time bins.
- `Manifold`: latent/manifold representation, shaped as components by time bins.
- `trialTypeDir`: direction label.
- `trialTypeCond`: task/interaction condition label.

The `README.txt` and main scripts use `trialTypeDir` and `trialTypeCond`.
`assert_valid_trial.m` currently checks `trialTypeD` and `trialTypeCon`, so that
validator is inconsistent with the rest of the pipeline and should be corrected
before relying on it.

## Main Flow

The core entry point is `compute_LL_matrix.m`.

1. Select trials for a direction and condition with `filter_data.m`.
2. Convert selected trials into lag/block representations with
   `convert_to_lag_struct.m`.
3. Depending on `config.corr_obj`, compute one of three lag-lag matrices:
   - `manifold`: correlate the two subjects' manifold representations directly.
   - `hat-obs`: decode neural activity from each subject's manifold, then
     correlate predicted activity from one subject with observed activity from
     the other.
   - `hat-hat`: decode neural activity from both subjects' manifolds, then
     correlate the two predicted neural profiles.

The output is a lag-lag correlation matrix. Rows correspond to `Y_subject`
lags and columns correspond to `X_subject` lags.

## Windowing And Lag Construction

`convert_to_lag_struct.m` defines the temporal units used by the analysis.

- `block_size` defines the lag/window size.
- `block_stride` is a fraction of `block_size`, then converted to an absolute
  stride.
- `sub_block_size` defines a local averaging window inside each block.
- `sub_block_stride` controls overlap between sub-blocks.

For each block and sub-block, the script averages the manifold and neural data.
This increases the number of observations per selected trial when sub-blocks
are smaller than blocks.

## Decoder Layer

`decoders.m` supports:

- Ridge regression, with lambda selected by GCV over `ridge_lambdas`.
- KNN regression, with neighbor prediction on standardized manifold features.

The decoder is fit separately for each lag/block and each neuron/channel.
Predictions are returned as cells indexed by lag and neuron/channel.

Important limitation: the current decoder predictions are effectively in-sample.
The shuffle baseline is useful, but it does not replace held-out trial-level
cross-validation. For publication-level claims, this needs a stricter validation
scheme.

## Correlation Layer

`f_corr.m` converts lag/neuron cells to arrays, averages over the second
dimension, and computes a correlation matrix between lag profiles.

Interpretation: the resulting matrix is a population/component-averaged
lag-lag similarity measure. It is not yet a neuron-resolved or causal estimate.

## Directional Summary

`LL_directional_stats.m` and the loop script summarize the lag-lag matrix as:

- upper triangle: one subject leading the other;
- lower triangle: the opposite lead direction;
- diagonal: synchronous similarity.

The scripts label these as `S -> K`, `K -> S`, and synchrony. This is useful,
but should be described carefully as lag-asymmetry in representational/neural
similarity, not direct evidence of information transfer by itself.

## Strengths

- Clear separation between manifold-only and decoder-mediated analyses.
- Explicit direction and task-condition filtering.
- Lag-lag representation is aligned with the dyadic interaction question.
- Ridge and KNN provide interpretable baselines.
- Shuffle metrics are already present.

## Main Risks

- Decoder evaluation is in-sample unless cross-validation is added.
- Trial autocorrelation and overlapping windows can inflate similarity.
- A shared task stimulus can create apparent cross-subject alignment even
  without subject-to-subject coupling.
- Averaging over neurons/components may hide structure and subject-specific
  differences.
- Direct manifold correlation assumes comparable axes across subjects; this may
  require Procrustes, CCA, PLS, or RSA-style alignment.
- The validator field names are inconsistent with the actual data contract.

## NeuroBridge Integration Plan

For NeuroBridge, the useful abstraction is:

1. Generate or load two-subject data with shared task labels.
2. Produce subject-specific neural activity and/or learned embeddings.
3. Build lagged/block representations from each subject.
4. Compute:
   - manifold-vs-manifold lag-lag matrices;
   - embedding-vs-embedding lag-lag matrices;
   - decoder-mediated `hat-obs` and `hat-hat` matrices.
5. Report diagonal, lead-lag asymmetry, shuffle baselines, and held-out
   decoding metrics.

The next scientific upgrade is to make the validation stricter:

- split by trials, not by windows;
- compare true labels against direction/condition shuffles;
- include common-stimulus controls;
- report confidence intervals or permutation p-values for lag asymmetry.

## Repository Hygiene

Local MAT data should not be committed. The `.gitignore` excludes
`Hasson_pipeline/*.mat` and MATLAB autosave files
`Hasson_pipeline/*.asv`.
