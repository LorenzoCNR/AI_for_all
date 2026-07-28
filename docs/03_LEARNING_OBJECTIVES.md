# Learning Objectives

## Objective Of Representation Learning

For each neural window `x_i`, an encoder produces an embedding `z_i`. Training
does not directly regress `z_i` onto the simulator latent. Instead, it asks the
geometry among embeddings to reproduce a target geometry derived from
window metadata.

The controlled CNN1D experiment uses a **soft structured contrastive loss**.
Two other implemented alternatives are supervised InfoNCE and temporal-offset
InfoNCE.

## The Idea Before The Equations

Suppose a minibatch contains many neural windows. Pick one window \(i\), called
the **anchor**. Every other window \(j\) is a possible neighbor.

NeuroBridge constructs two answers to the same question:

1. **Target answer \(Q\):** according to time and task metadata, how much
   probability should anchor \(i\) assign to every candidate \(j\)?
2. **Encoder answer \(P\):** according to the learned neural embeddings, how
   much probability does anchor \(i\) actually assign to every candidate?

The loss trains the encoder by making \(P\) resemble \(Q\). There is no single
hard positive and no single hard negative: all non-self pairs can receive a
different target weight.

### Notation Used Below

- Indices \(i,j,k\) identify windows in the current minibatch.
- \(B\) is minibatch size.
- \(x_i\) is the neural window and \(z_i=f_\theta(x_i)\) its embedding.
- \(D_{ij}\) is a desired distance; smaller means "should be closer."
- \(Q_{ij}\) is the target neighbor probability.
- \(P_{ij}\) is the neighbor probability predicted by the encoder.
- A sum over \(k\ne i\) normalizes across every candidate except the anchor
  itself.

## Metadata Distance

For minibatch windows \(i\) and \(j\), define:

$$
T_{ij}
=
\frac{|t_i-t_j|}
{\max_{a,b\in\mathrm{batch}}|t_a-t_b|}.
$$

Here \(t_i\) is time within the trial. Therefore \(T_{ij}\in[0,1]\): zero
means equal trial time and one is the largest temporal separation represented
in that minibatch.

Condition distance is:

$$
C_{ij} =
\begin{cases}
0, & \text{if } c_i=c_j,\\
1, & \text{if } c_i\ne c_j.
\end{cases}
$$

The controlled distance combines time, condition, and movement progress:

$$
D_{ij} =
\frac{
w_{\mathrm{time}}T_{ij}
+
w_{\mathrm{condition}}\sqrt{s_i s_j}\,C_{ij}
}{
w_{\mathrm{time}}+w_{\mathrm{condition}}
}.
$$

The symbols mean:

- \(c_i\) is the task condition of window \(i\);
- \(s_i\in[0,1]\) is its movement progress;
- \(w_{\mathrm{time}}\) and \(w_{\mathrm{condition}}\) are nonnegative weights;
- \(D_{ij}\) is small for desired neighbors and large for undesired neighbors.

Default weights are `w_time=0.5` and `w_condition=0.5`.

The progress gate \(\sqrt{s_i s_j}\) has a specific purpose. If both windows
are near movement onset, then \(s_i\approx s_j\approx0\), so different
directions receive almost no condition penalty: they truly share the center.
Near the end of movement the gate approaches one, so different directions are
separated.

Three concrete cases clarify the distance:

- same time and same condition: both terms are zero, hence \(D_{ij}=0\);
- different time but same condition: only temporal distance contributes;
- same late time but different conditions: the condition penalty contributes
  almost at full strength.

The package also supports temporal, circular, categorical, and continuous
metadata geometries through a general specification API. The exact controlled
notebook target above uses categorical condition separation plus time; it does
not use circular adjacency among directions.

## Soft Target Distribution

Distance becomes a positive affinity:

$$
S_{ij}=\exp\left(-\frac{D_{ij}}{\tau_{\mathrm{metadata}}}\right).
$$

Small distance gives affinity near one; large distance gives affinity near
zero. The diagonal \(S_{ii}\) is removed because a window must not select
itself. Each row is then normalized:

$$
Q_{ij}
=
\frac{S_{ij}}
{\sum_{k\ne i}S_{ik}}
=
\frac{\exp(-D_{ij}/\tau_{\mathrm{metadata}})}
{\sum_{k\ne i}\exp(-D_{ik}/\tau_{\mathrm{metadata}})}.
$$

`Q_i` is the desired probability distribution over all other observations in
the minibatch. It replaces a binary positive/negative decision with graded
relationships. The controlled default is `tau_metadata=0.5`.

Small `tau_metadata` concentrates probability on the closest metadata
neighbors. Large values make the target flatter.

## Distribution Predicted By The Embedding

Embeddings are L2-normalized. For nonzero vectors, cosine similarity is:

$$
\operatorname{cos}(z_i,z_j)
=
\frac{z_i^\top z_j}{\lVert z_i\rVert_2\lVert z_j\rVert_2}.
$$

It is near one for vectors pointing in the same direction, near zero for
orthogonal vectors, and near minus one for opposite directions.

The encoder's neighbor distribution is:

$$
P_{ij}
=
\frac{
\exp\left(\operatorname{cos}(z_i,z_j)/
\tau_{\mathrm{embedding}}\right)
}{
\sum_{k\ne i}
\exp\left(\operatorname{cos}(z_i,z_k)/
\tau_{\mathrm{embedding}}\right)
}.
$$

For each anchor \(i\), the row sums to one. A candidate with greater cosine
similarity receives greater predicted probability.

The default embedding temperature is `tau_embedding=0.1`. It controls how
strongly the encoder distribution concentrates around its nearest embedding
neighbors.

The two temperatures are not interchangeable:

- `tau_metadata` shapes the desired neighborhood distribution `Q`;
- `tau_embedding` shapes the predicted neighborhood distribution `P`.

## Cross-Entropy Loss

For minibatch size \(B\), training minimizes:

$$
\mathcal{L}
=
-\frac{1}{B}
\sum_{i=1}^{B}
\sum_{j\ne i}
Q_{ij}\log P_{ij}.
$$

This is the cross-entropy `CE(Q,P)`. Since `Q` is fixed with respect to model
parameters, minimizing it is equivalent to minimizing `KL(Q || P)` up to the
constant entropy of `Q`.

The logarithm makes confident mistakes costly. If \(Q_{ij}\) is large but
\(P_{ij}\) is tiny, then \(-Q_{ij}\log P_{ij}\) is large. If the encoder assigns
probability in the same pattern requested by \(Q\), the loss decreases.

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

Row \(i\) of \(Q\) and row \(i\) of \(P\) both describe the 255 possible
non-self neighbors of anchor \(i\).

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
