# Learning Objectives

## Objective Of Representation Learning

For each neural window `x_i`, an encoder produces an embedding `z_i`. Training
does not directly regress `z_i` onto the simulator latent. Instead, it asks the
geometry among embeddings to reproduce a target geometry derived from
window metadata.

The controlled CNN1D experiment uses a **soft structured contrastive loss**.
Two other implemented alternatives are supervised InfoNCE and temporal-offset
InfoNCE.

## Metadata Distance

For minibatch elements `i` and `j`, define normalized temporal distance
`T_ij`, categorical condition distance

```text
C_ij = 1[label_i != label_j],
```

and movement progress `s_i`. The controlled target uses

```text
D_ij =
    (w_time T_ij + w_condition sqrt(s_i s_j) C_ij)
    / (w_time + w_condition).
```

Default weights are `w_time=0.5` and `w_condition=0.5`.

The progress gate has a specific purpose. Different reaching conditions share
the same origin, so the condition penalty should be weak at movement onset and
stronger after trajectories diverge.

The package also supports temporal, circular, categorical, and continuous
metadata geometries through a general specification API. The exact controlled
notebook target above uses categorical condition separation plus time; it does
not use circular adjacency among directions.

## Soft Target Distribution

Distance becomes unnormalized affinity:

```text
S_ij = exp(-D_ij / tau_metadata).
```

The diagonal is removed and each row is normalized:

```text
Q_ij = S_ij / sum(k != i) S_ik.
```

`Q_i` is the desired probability distribution over all other observations in
the minibatch. It replaces a binary positive/negative decision with graded
relationships. The controlled default is `tau_metadata=0.5`.

Small `tau_metadata` concentrates probability on the closest metadata
neighbors. Large values make the target flatter.

## Distribution Predicted By The Embedding

Embeddings are L2-normalized. Their logits are cosine similarities:

```text
ell_ij = cosine(z_i,z_j) / tau_embedding.
```

After removing the diagonal:

```text
P_ij = exp(ell_ij) / sum(k != i) exp(ell_ik).
```

The default embedding temperature is `tau_embedding=0.1`. It controls how
strongly the encoder distribution concentrates around its nearest embedding
neighbors.

The two temperatures are not interchangeable:

- `tau_metadata` shapes the desired neighborhood distribution `Q`;
- `tau_embedding` shapes the predicted neighborhood distribution `P`.

## Cross-Entropy Loss

For minibatch size `B`, training minimizes:

```text
L = -(1/B) sum_i sum(j != i) Q_ij log P_ij.
```

This is the cross-entropy `CE(Q,P)`. Since `Q` is fixed with respect to model
parameters, minimizing it is equivalent to minimizing `KL(Q || P)` up to the
constant entropy of `Q`.

The loss does not minimize Euclidean distance directly. It minimizes
disagreement between two row-wise probability distributions: one built from
metadata and one built from the learned embedding.

## Is It Supervised?

The answer depends on the metadata:

- using task condition labels makes the controlled objective task-informed or
  weakly supervised;
- using only time and trial identity produces a self-supervised temporal
  objective;
- using explicit class identity as a hard positive mask is supervised
  InfoNCE.

Calling every version unsupervised would be incorrect.

## Other Implemented Objectives

### Supervised InfoNCE

Observations with the same label are positives. All other non-self batch
members act as negatives. This gives a hard class-based target and can collapse
within-condition temporal structure if used alone.

### Temporal-Offset InfoNCE

Positive pairs belong to the same trial and differ by an exact time offset.
No behavioral label is required. This is self-supervised, but the offset is a
researcher-chosen temporal prior.

## Why Batch Shuffling Does Not Destroy Time

The DataLoader may shuffle windows between minibatches. The sample metadata
travels with each window, so `time_id`, `trial_id`, condition, and progress
remain correct. The loss reconstructs pairwise relationships after the batch
has been assembled.

What shuffling does change is the set of candidate neighbors visible in that
step. This creates batch-context dependence and is one reason batch size and
sampling strategy matter.

## Computational Cost

Both `Q` and `P` are dense `B x B` matrices. Time and memory are therefore
quadratic in minibatch size:

```text
complexity approximately O(B^2).
```

Possible future directions include sparse neighborhoods, blockwise
computation, memory banks, approximate nearest neighbors, and symmetry-aware
reuse. None is part of the validated implementation.

## Scientific Risks And Required Ablations

The objective intentionally injects a target geometry. Good recovery is not
surprising by itself; the scientific question is which aspects are recovered,
under what noise and observation mappings, and whether the prior improves
generalization.

Required comparisons include:

- time-only target;
- condition-only target;
- shuffled condition labels;
- hard supervised InfoNCE;
- temporal-offset self-supervision;
- PCA and untrained-network baselines;
- multiple values of window size, weights, and both temperatures;
- multiple random seeds and confidence intervals.

Without these controls, the target can become partly tautological: the method
may reproduce the geometry it was explicitly told to prefer.
