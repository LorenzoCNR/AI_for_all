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

Population observation model
----------------------------

The linear neural drive is:

.. math::

   u = ZB + c,

where ``B`` maps latent coordinates to neurons and ``c`` is baseline activity.
The positive firing rate is:

.. math::

   \lambda = \operatorname{softplus}(u).

Spike counts are sampled from the resulting rate over bins of duration
``dt``.

Localized place fields
----------------------

Linear-track place neurons receive an additional nonlinear drive:

.. math::

   g_j(p) =
   a_j \exp\left[-\frac{(p-\mu_j)^2}{2\sigma_j^2}\right].

The preferred position ``mu_j`` maps a neuron to one region of the track;
``sigma_j`` determines field width. This term is kept separate from ``B``
because a localized Gaussian field is not a linear function of position.

Temporal windows
----------------

Both PCA and CNN1D receive the same centered neural windows with shape:

.. code-block:: text

   observations x window size x neurons

One window is associated with one target time bin and never crosses a trial
boundary.

Structured contrastive target
-----------------------------

Metadata distances define a soft target distribution over pairs:

.. math::

   D_{ij} = w_t D^{(t)}_{ij} + w_c D^{(c)}_{ij},

.. math::

   Q_{ij} \propto \exp(-D_{ij}/\tau).

The CNN embedding defines a cosine-softmax distribution ``P``. Training
minimizes cross-entropy between ``Q`` and ``P``.

Evaluation
----------

Procrustes R² measures coordinate recovery after centered orthogonal
alignment. Representational similarity analysis measures agreement between
pairwise-distance geometries and can compare spaces with different coordinate
orientations.
