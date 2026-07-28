# Learning Objectives

## Objective Of Representation Learning

For each neural window `x_i`, an encoder produces an embedding `z_i`. Training
does not directly regress `z_i` onto the simulator latent. Instead, it asks the
geometry among embeddings to reproduce a target geometry derived from
window metadata.

The controlled CNN1D experiment uses a **soft structured contrastive loss**.
Two other implemented alternatives are supervised InfoNCE and temporal-offset
InfoNCE.

It is contrastive because each anchor window is compared with all other
windows in the minibatch. It is soft because those candidate pairs are not
reduced to a binary positive/negative split: each pair can receive a different
target weight. It is structured because the weights are constructed from
temporal and task metadata. The loss then asks the embedding-space
neighborhoods to reproduce that target structure.

## The Idea Before The Equations

Suppose a minibatch contains many neural windows. Pick one window $i$, called
the **anchor**. Every other window $j$ is a possible neighbor.

NeuroBridge constructs two answers to the same question:

1. **Target answer $Q$:** according to time and task metadata, how much
   probability should anchor $i$ assign to every candidate $j$?
2. **Encoder answer $P$:** according to the learned neural embeddings, how
   much probability does anchor $i$ actually assign to every candidate?

The loss trains the encoder by making $P$ resemble $Q$. There is no single
hard positive and no single hard negative: all non-self pairs can receive a
different target weight.

### Notation Used Below

- Indices $i,j,k$ identify windows in the current minibatch.
- $B$ is minibatch size.
- $x_i$ is the neural window and $z_i=f_\theta(x_i)$ its embedding.
- $D_{ij}$ is a desired distance; smaller means "should be closer."
- $Q_{ij}$ is the target neighbor probability.
- $P_{ij}$ is the neighbor probability predicted by the encoder.
- A sum over $k\ne i$ normalizes across every candidate except the anchor
  itself.

## Metadata Distance

The metadata distance is a **designed training target**. It is computed from
known time and task descriptors, not from neural activity. The encoder never
uses it as an input feature; it sees neural windows and is penalized when their
embedding neighborhoods disagree with this target.

### Step 1: Temporal Separation

For windows $i$ and $j$, first compute absolute temporal separation:

```math
\Delta t_{ij}=|t_i-t_j|.
```

Let $\Delta t_{\max}$ be the largest pairwise temporal separation in the
current minibatch. Normalize:

```math
T_{ij}=\frac{\Delta t_{ij}}{\Delta t_{\max}}.
```

Therefore $T_{ij}$ lies between zero and one. Equal trial times give zero. The
most temporally separated pair present in that minibatch gives one. Because the
normalization is batch-wise, the same raw time difference can receive a
slightly different normalized value in a differently composed batch.

### Step 2: Condition Separation

Define a categorical distance:

- $C_{ij}=0$ when $c_i=c_j$;
- $C_{ij}=1$ when $c_i\ne c_j$.

This controlled objective treats all different conditions as equally
different. It does not encode circular adjacency: directions 1 and 2 receive
the same categorical penalty as directions 1 and 5.

### Step 3: Progress Gate

Define:

```math
G_{ij}=\sqrt{s_i s_j},
```

where $s_i,s_j\in[0,1]$ are movement progress.

- if both windows are at movement onset, $G_{ij}\approx0$;
- if both are near arrival, $G_{ij}\approx1$;
- if either window is near onset, the gate remains small.

This matters because all circular-reaching conditions share the same physical
origin. A condition label differs from the start, but the corresponding
positions have not separated yet.

### Step 4: Weighted Combination

The complete metadata distance is:

```math
D_{ij} =
\frac{
w_{\mathrm{time}}T_{ij}
+
w_{\mathrm{condition}}G_{ij}C_{ij}
}{
w_{\mathrm{time}}+w_{\mathrm{condition}}
}.
```

The symbols mean:

- $c_i$ is the task condition of window $i$;
- $w_{\mathrm{time}}$ and $w_{\mathrm{condition}}$ are nonnegative weights;
- $D_{ij}$ is small for desired neighbors and large for undesired neighbors.

Default weights are `w_time=0.5` and `w_condition=0.5`.

### Numerical Examples

Use equal weights, so the denominator is one.

| Pair | $T_{ij}$ | $C_{ij}$ | Progress | $G_{ij}$ | $D_{ij}$ |
|---|---:|---:|---:|---:|---:|
| Same time, same condition | 0.0 | 0 | 0.5 and 0.5 | 0.5 | 0.00 |
| Half-trial time difference, same condition | 0.5 | 0 | 0.5 and 0.5 | 0.5 | 0.25 |
| Same time, different conditions, near origin | 0.0 | 1 | 0.1 and 0.1 | 0.1 | 0.05 |
| Same time, different conditions, near arrival | 0.0 | 1 | 0.9 and 0.9 | 0.9 | 0.45 |
| Half-trial difference, different late conditions | 0.5 | 1 | 0.9 and 0.9 | 0.9 | 0.70 |

The interpretation is always relative:

- lower $D_{ij}$ means the pair should receive more target probability;
- higher $D_{ij}$ means it should receive less;
- $D_{ij}$ is not an observed spike distance;
- $D_{ij}$ does not say that every same-condition pair must collapse to one
  point, because temporal separation still contributes.

The package also supports temporal, circular, categorical, and continuous
metadata geometries through a general specification API. The exact controlled
notebook target above uses categorical condition separation plus time; it does
not use circular adjacency among directions.

## Soft Target Distribution

Distance becomes a positive affinity:

```math
S_{ij}=\exp\left(-\frac{D_{ij}}{\tau_{\mathrm{metadata}}}\right).
```

Small distance gives affinity near one; large distance gives affinity near
zero. The diagonal $S_{ii}$ is removed because a window must not select
itself. Each row is then normalized:

```math
Q_{ij}
=
\frac{S_{ij}}
{\sum_{k\ne i}S_{ik}}
=
\frac{\exp(-D_{ij}/\tau_{\mathrm{metadata}})}
{\sum_{k\ne i}\exp(-D_{ik}/\tau_{\mathrm{metadata}})}.
```

`Q_i` is the desired probability distribution over all other observations in
the minibatch. It replaces a binary positive/negative decision with graded
relationships. The controlled default is `tau_metadata=0.5`.

Small `tau_metadata` concentrates probability on the closest metadata
neighbors. Large values make the target flatter.

## Distribution Predicted By The Embedding

Embeddings are L2-normalized. For nonzero vectors, cosine similarity is:

```math
\cos(z_i,z_j)
=
\frac{z_i^\top z_j}{\| z_i\|_2\| z_j\|_2}.
```

It is near one for vectors pointing in the same direction, near zero for
orthogonal vectors, and near minus one for opposite directions.

The encoder's neighbor distribution is:

```math
P_{ij}
=
\frac{
\exp\left(\cos(z_i,z_j)/
\tau_{\mathrm{embedding}}\right)
}{
\sum_{k\ne i}
\exp\left(\cos(z_i,z_k)/
\tau_{\mathrm{embedding}}\right)
}.
```

For each anchor $i$, the row sums to one. A candidate with greater cosine
similarity receives greater predicted probability.

The default embedding temperature is `tau_embedding=0.1`. It controls how
strongly the encoder distribution concentrates around its nearest embedding
neighbors.

The two temperatures are not interchangeable:

- `tau_metadata` shapes the desired neighborhood distribution `Q`;
- `tau_embedding` shapes the predicted neighborhood distribution `P`.

## Cross-Entropy Loss

For minibatch size $B$, training minimizes:

```math
\mathcal{L}
=
-\frac{1}{B}
\sum_{i=1}^{B}
\sum_{j\ne i}
Q_{ij}\log P_{ij}.
```

This is the cross-entropy `CE(Q,P)`. Since `Q` is fixed with respect to model
parameters, minimizing it is equivalent to minimizing `KL(Q || P)` up to the
constant entropy of `Q`.

The logarithm makes confident mistakes costly. If $Q_{ij}$ is large but
$P_{ij}$ is tiny, then $-Q_{ij}\log P_{ij}$ is large. If the encoder assigns
probability in the same pattern requested by $Q$, the loss decreases.

The loss does not minimize Euclidean distance directly. It minimizes
disagreement between two row-wise probability distributions: one built from
metadata and one built from the learned embedding.

### Dimensions In A Real Minibatch

With `batch_size=256` and `embedding_dim=3`:

```text
neural windows x   : (256, 10, 100)
embeddings z       : (256, 3)
metadata distance D: (256, 256)
target Q           : (256, 256)
prediction P       : (256, 256)
loss L             : one scalar
```

Row $i$ of $Q$ and row $i$ of $P$ both describe the 255 possible
non-self neighbors of anchor $i$.

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
