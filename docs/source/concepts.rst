Core concepts
=============

Task latent
-----------

The known latent tensor has shape:

.. math::

   Z \in \mathbb{R}^{N_{\mathrm{trial}} \times T \times K}.

For a circular task, the essential coordinates are planar position and
movement progress. Velocity and context can be added as fourth and fifth
coordinates. For a linear task, position and movement direction form the
essential two-dimensional state; velocity and context produce the enriched
four-dimensional state.

The linear-track behavior and latent state must not be conflated. Normalized
physical position follows:

.. math::

   p(t): 0 \longrightarrow 1 \longrightarrow 0.

Movement phase is represented by signed direction:

.. math::

   d(t) =
   \begin{cases}
   +1, & \text{outbound movement from 0 to 1},\\
   -1, & \text{return movement from 1 to 0}.
   \end{cases}

Thus the essential linear ground truth is ``[p(t), d(t)]``. A behavior plot
shows one trajectory along a physical line. A latent-state plot shows two
position branches distinguished by direction. The output metadata calls this
variable ``movement_phase``; numeric values 0 and 1 are retained internally
only for batching and the contrastive objective.

Population observation model
----------------------------

The linear neural drive is:

.. math::

   u = ZB + c,

where ``B`` maps latent coordinates to neurons and ``c`` is baseline activity.
The positive firing rate is:

.. math::

   \lambda = r_{\mathrm{scale}}\operatorname{softplus}(u).

The controlled suite uses ``r_scale = 10`` and ``dt = 0.02`` seconds. Spike
counts are sampled from the resulting rate over 20 ms bins. The scale is
explicit because a dimensionless softplus output should not silently be
interpreted as a realistic firing rate.

Localized place fields
----------------------

Linear-track place neurons receive an additional nonlinear drive:

.. math::

   g_j(p) =
   a_j \exp\left[-\frac{(p-\mu_j)^2}{2\sigma_j^2}\right].

The preferred position ``mu_j`` maps a neuron to one region of the track;
``sigma_j`` determines field width. This term is kept separate from ``B``
because a localized Gaussian field is not a linear function of position.

In plain terms, each place-selective neuron prefers one location between 0
and 1. Its simulated rate is highest near that location and lower farther
away. Different neurons receive different preferred locations, so the
population covers the track. The saved place-field heatmap places one neuron
on each row; a bright diagonal means preferred locations progress from the
start to the end of the track.

Temporal windows
----------------

Both PCA and CNN1D receive the same centered neural windows with shape:

.. code-block:: text

   observations x window size x neurons

One window is associated with one target time bin and never crosses a trial
boundary.

PCA flattens each window before fitting a linear variance-maximizing
projection. CNN1D transposes it to ``batch x neurons x time`` and applies
residual temporal convolutions followed by global temporal pooling. Both
produce one embedding vector per window. Neither model produces ten separate
embedding vectors from a ten-bin window.

Structured contrastive target
-----------------------------

Metadata distances define a soft target distribution over pairs. Let ``s``
denote movement progress and let ``I(c_i != c_j)`` be a categorical condition
distance:

.. math::

   D_{ij} =
   \frac{
   w_t D^{(t)}_{ij} +
   w_c \sqrt{s_i s_j}\,\mathbb{I}(c_i \ne c_j)
   }{w_t + w_c}.

.. math::

   Q_{ij} \propto \exp(-D_{ij}/\tau).

The progress gate makes the condition term vanish at the common movement
origin and increase as the trajectories separate. Without it, the objective
would force different directions apart even when their known latent states
are identical at movement onset.

The CNN embedding defines a cosine-softmax distribution ``P`` over the other
samples in a minibatch. Training minimizes:

.. math::

   \mathcal{L} = -\frac{1}{B}
   \sum_i \sum_{j \ne i} Q_{ij}\log P_{ij},

where ``B`` is the minibatch size. Positives and negatives are therefore not
hard binary sets: each pair receives a target probability determined by
metadata.

Evaluation
----------

Aggregate Procrustes :math:`R^2` measures coordinate recovery after centered
orthogonal alignment:

.. math::

   R^2_{\mathrm{Proc}} =
   1 -
   \frac{\lVert Z-\widehat{E} \rVert_F^2}
        {\lVert Z-\bar{Z} \rVert_F^2}.

Representational similarity analysis measures agreement between vectorized
pairwise-distance matrices and is invariant to coordinate orientation.
Neither metric replaces trajectory inspection. The primary plots average
windows across trials at each time point; all-window scatter plots are kept
only as diagnostics.

Concretely, RSA is computed as follows:

1. For every pair of test windows, compute the Euclidean distance between
   their known latent states.
2. For the same pairs, compute the Euclidean distance between their learned
   embedding vectors.
3. Compute Spearman correlation between the two lists of distances.

A reported value of ``0.927`` means that pairs ranked as relatively near or
far in the true state tend to retain that ordering in the embedding. It does
not mean 92.7% classification accuracy. It also does not prove that a visible
trajectory, center, or loop has the correct shape; those properties require
the trajectory plots and targeted diagnostics.

``Full`` RSA uses every simulated latent coordinate. ``Motor-core`` RSA uses
X, Y, and progress for circular reaching, or position and movement direction
for the linear track. In an essential experiment the full state and motor core
are identical. In an enriched experiment, the comparison reveals whether a
model recovered the task-defining coordinates without necessarily recovering
velocity and context.
