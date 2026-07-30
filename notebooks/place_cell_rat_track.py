# %%
"""Didactic linear-track population with identifiable neuron ground truth."""

# %% [markdown]
# # Rat on a linear track: identifiable neural tuning
#
# This notebook constructs a simple synthetic population with three primary
# neuron classes:
#
# - `positional`: a preferred spatial bin, independent of direction;
# - `directional`: outbound or return preference, independent of position;
# - `mixed`: a preferred spatial bin and a preferred direction.
#
# The optional `gradient` class is disabled by default. The assigned class and
# tuning parameters are retained as ground truth. Spike counts remain random:
# the tuning controls the conditional rate and Poisson sampling generates `X`.

# %%
from pathlib import Path
import csv
import os
import random
import sys

import joblib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
import numpy as np

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path.cwd().resolve()

if not (PROJECT_ROOT / "src" / "neurobridge").exists():
    PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "src" / "neurobridge").exists():
    raise FileNotFoundError("Run this notebook from the Neuro_Bridge repository.")

os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from neurobridge.data.sim import LatentTrajectoryGenerator
from neurobridge.data.sim.builders import drive_to_rate, rate_to_spike
from neurobridge.experiments import (
    SyntheticTaskConfig,
    build_linear_loading_and_place_fields,
)

print("Project root:", PROJECT_ROOT)

# %% [markdown]
# ## 1. Configuration
#
# The track is split into `n_position_bins`. A spatially tuned neuron chooses
# one preferred bin once, at population construction time. Its expected rate is
# maximal in that bin and decreases over neighboring bins. The observed count
#  is sampled from a Poisson distribution.
#
# `place_fraction` is the name of the fraction of purely positional neurons. With one third positional neurons and no gradients, the remaining population is split
# equally between directional and mixed neurons.
#

# %%
CONFIG = SyntheticTaskConfig(
    name="place_cell_rat_track",
    condition_mode="linear",
    latent_dim=2,
    n_trials=100,
    trial_length=300,
    n_neurons=100,
    n_position_bins=5,
    # fraction of purely positional neurons (place cells) in the population
    place_fraction=1.0 / 3.0,
    place_width=0.075,
    place_scale=3.0,
    gradient_fraction=0.0,
    # what happens when the rat is moving in the nonpreferred direction?
    # The place cell will still fire, but at a lower rate.
    nonpreferred_direction_gain=0.10,
    baseline_mean=1.0,
    baseline_std=0.10,
    rate_scale=0.8,
    dt=0.025,
    noise_scale=0.10,
    random_state=42,
)

random.seed(CONFIG.random_state)
np.random.seed(CONFIG.random_state)

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / CONFIG.name
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
CONFIG

# A mildly overdispersed emission model produces realistic count variability
# while leaving the assigned positional and directional tuning unchanged.
EMISSION_CONFIG = {
    "overdispersion": 0.5,
    "burst_probability": 0.4,
    "burst_size_mean": 0.5,
    "burst_window_bins": 1,
}
EMISSION_CONFIG

# %% [markdown]
# ## 2. Linear trajectory
#
# Position follows `0 -> 1 -> 0`. Direction code is binary:
# outbound is `1` and return is `0`. For the linear loading matrix only, this
# code is centered to `+1/-1` so opposite directional preferences are symmetric.

# %%
generator = LatentTrajectoryGenerator(
    n_trials=CONFIG.n_trials,
    L=CONFIG.trial_length,
    k=CONFIG.latent_dim,
    phi=CONFIG.phi,
    condition_mode=CONFIG.condition_mode,
    n_conditions=CONFIG.n_conditions,
    noise_scale=CONFIG.noise_scale,
    condition_type="balanced",
)
Z_generator, condition, state = generator.generate_latent(return_state=True)

position_clean = state["position"][:, :, 0]
turn_indices = np.argmax(position_clean, axis=1)
direction_binary = np.zeros(position_clean.shape, dtype=int)
for trial, turn in enumerate(turn_indices):
    direction_binary[trial, : turn + 1] = 1

Z = np.stack([Z_generator[:, :, 0], direction_binary], axis=-1)
Z_neural_driver = Z.copy()
Z_neural_driver[:, :, 1] = 2.0 * direction_binary - 1.0

assert Z.shape == (CONFIG.n_trials, CONFIG.trial_length, 2)
assert condition is None
assert np.array_equal(np.unique(direction_binary), np.array([0, 1]))

trial = 0
time_bin = np.arange(CONFIG.trial_length)
fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
axes[0].plot(time_bin, position_clean[trial], linewidth=2.5)
axes[0].set_ylabel("Track position")
axes[1].step(time_bin, direction_binary[trial], where="post", linewidth=2.5)
axes[1].set_yticks([0, 1], labels=["return", "outbound"])
axes[1].set_ylabel("Direction")
axes[1].set_xlabel("Time of trial (bins)")
for axis in axes:
    axis.axvline(turn_indices[trial], color="black", linestyle="--")
    axis.grid(alpha=0.2)
fig.suptitle("Complete Trial, way go and return", fontsize=16)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Identifiable neural population
#
# `preferred_bin=-1` means that a neuron has no localized spatial preference.
# `preferred_direction=-1` means no direction preference; otherwise `1` is
# outbound and `0` is return. These arrays are simulator ground truth.

# %%
(
    B,
    neuron_types,
    ##explain
    preferred_centers,
    spatial_drive,
    neuron_ground_truth,
) = build_linear_loading_and_place_fields(
    k=CONFIG.latent_dim,
    n_neurons=CONFIG.n_neurons,
    position=position_clean,
    direction=direction_binary,
    place_fraction=CONFIG.place_fraction,
    place_width=CONFIG.place_width,
    place_scale=CONFIG.place_scale,
    first_coordinates_multiplier=CONFIG.first_coordinates_multiplier,
    random_state=CONFIG.random_state + 1,
    n_position_bins=CONFIG.n_position_bins,
    gradient_fraction=CONFIG.gradient_fraction,
    nonpreferred_direction_gain=CONFIG.nonpreferred_direction_gain,
    return_metadata=True,
)

preferred_bins = neuron_ground_truth["preferred_bin"]
preferred_directions = neuron_ground_truth["preferred_direction"]
gradient_sign = neuron_ground_truth["gradient_sign"]
bin_edges = neuron_ground_truth["position_bin_edges"]
bin_centers = neuron_ground_truth["position_bin_centers"]

assert np.array_equal(neuron_ground_truth["neuron_id"], np.arange(CONFIG.n_neurons))
assert set(np.unique(neuron_types)).issubset(
    {"positional", "directional", "mixed", "gradient"}
)

type_order = ["positional", "directional", "mixed", "gradient"]
print("Neuron classes:")
for neuron_type in type_order:
    count = int(np.sum(neuron_types == neuron_type))
    if count:
        print(f"  {neuron_type:>11}: {count}")

print("\nFirst 15 ground-truth rows:")
print("id | type        | preferred bin | preferred direction")
for neuron in range(min(15, CONFIG.n_neurons)):
    direction_name = {-1: "none", 0: "return", 1: "outbound"}[
        int(preferred_directions[neuron])
    ]
    print(
        f"{neuron:2d} | {neuron_types[neuron]:11s} | "
        f"{preferred_bins[neuron]:13d} | {direction_name}"
    )

# %%
ground_truth_path = OUTPUT_ROOT / "neuron_ground_truth.csv"
with ground_truth_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "neuron_id",
            "neuron_type",
            "preferred_bin",
            "preferred_direction",
            "preferred_direction_name",
            "preferred_center",
            "gradient_sign",
        ],
    )
    writer.writeheader()
    for neuron in range(CONFIG.n_neurons):
        writer.writerow(
            {
                "neuron_id": neuron,
                "neuron_type": neuron_types[neuron],
                "preferred_bin": preferred_bins[neuron],
                "preferred_direction": preferred_directions[neuron],
                "preferred_direction_name": {-1: "none", 0: "return", 1: "outbound"}[
                    int(preferred_directions[neuron])
                ],
                "preferred_center": preferred_centers[neuron],
                "gradient_sign": gradient_sign[neuron],
            }
        )
print("Saved:", ground_truth_path)

# %% [markdown]
# ## 4. Ground-truth population map
#
# Each point is one neuron. Spatially tuned neurons are placed at their
# preferred bin. Directional neurons appear in the separate direction panel.

# %%
colors = {
    "positional": "tab:blue",
    "directional": "tab:orange",
    "mixed": "tab:green",
    "gradient": "tab:purple",
}
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

counts = [np.sum(neuron_types == name) for name in type_order]
active_types = [name for name, count in zip(type_order, counts) if count]
active_counts = [count for count in counts if count]
axes[0].bar(active_types, active_counts, color=[colors[name] for name in active_types])
axes[0].tick_params(axis="x", rotation=30)
axes[0].set_ylabel("Number of neurons")
axes[0].set_title("Ground-truth classes")

for neuron_type in ["positional", "mixed"]:
    mask = neuron_types == neuron_type
    axes[1].scatter(
        preferred_bins[mask],
        np.flatnonzero(mask),
        label=neuron_type,
        color=colors[neuron_type],
        alpha=0.8,
    )
axes[1].set_xlabel("Preferred position bin")
axes[1].set_ylabel("Neuron ID")
axes[1].set_title("Spatial preferences")
axes[1].legend()

for direction_value, direction_name, marker in [(0, "return", "<"), (1, "outbound", ">")]:
    mask = (preferred_directions == direction_value)
    axes[2].scatter(
        np.full(mask.sum(), direction_value),
        np.flatnonzero(mask),
        marker=marker,
        label=direction_name,
        alpha=0.75,
    )
axes[2].set_xticks([0, 1], labels=["return", "outbound"])
axes[2].set_ylabel("Neuron ID")
axes[2].set_title("Directional and mixed preferences")
axes[2].legend()
for axis in axes:
    axis.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Conditional rates and Poisson spike counts

# %%
baseline_rng = np.random.default_rng(CONFIG.random_state + 2)
baseline = baseline_rng.normal(
    CONFIG.baseline_mean,
    CONFIG.baseline_std,
    size=CONFIG.n_neurons,
)

linear_drive = Z_neural_driver @ B
u = linear_drive + baseline + spatial_drive
lam = CONFIG.rate_scale * drive_to_rate(u, "softplus")

np.random.seed(CONFIG.random_state + 4)
X = rate_to_spike(lam, CONFIG.dt, **EMISSION_CONFIG)

assert X.shape == (CONFIG.n_trials, CONFIG.trial_length, CONFIG.n_neurons)
assert np.issubdtype(X.dtype, np.integer)
assert np.all(X >= 0)

flat_counts = X.reshape(-1)
mean_count = float(flat_counts.mean())
mean_rate = mean_count / CONFIG.dt
zero_fraction = float(np.mean(flat_counts == 0))
variance = float(flat_counts.var())

print(f"Mean conditional rate: {lam.mean():.3f} Hz")
print(f"Mean rate from X:      {mean_rate:.3f} Hz")
print(f"Mean count/bin:        {mean_count:.5f}")
print(f"Zero fraction:         {zero_fraction:.3%}")
print(f"Variance/mean:         {variance / mean_count:.3f}")
print(f"Maximum count/bin:     {flat_counts.max()}")

dataset_path = OUTPUT_ROOT / "synthetic_binned_population.joblib"
joblib.dump(
    {
        "spikes": X,
        "position": position_clean,
        "direction": direction_binary,
        "neuron_ground_truth": neuron_ground_truth,
        "neuron_types": neuron_types,
        "config": CONFIG,
        "emission_config": EMISSION_CONFIG,
    },
    dataset_path,
    compress=3,
)
print("Saved:", dataset_path)

values, frequencies = np.unique(flat_counts, return_counts=True)
probabilities = frequencies / frequencies.sum()
print("\nUnique spike-count values:")
print("count | observations | probability")
for value, frequency, probability in zip(values, frequencies, probabilities):
    print(f"{int(value):5d} | {int(frequency):12d} | {probability:11.7f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].bar(values, probabilities, color="slateblue")
axes[0].set_yscale("log")
axes[0].set_xlabel("Spike count in one bin")
axes[0].set_ylabel("Probability")
axes[0].set_title("Empirical spike-count distribution")

mean_rate_by_neuron = X.mean(axis=(0, 1)) / CONFIG.dt
rate_groups = [mean_rate_by_neuron[neuron_types == name] for name in active_types]
axes[1].boxplot(rate_groups, tick_labels=active_types, showfliers=True)
axes[1].tick_params(axis="x", rotation=30)
axes[1].set_ylabel("Mean firing rate (Hz)")
axes[1].set_title("Rate by ground-truth class")
for axis in axes:
    axis.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Spike-count distribution interpretation
#
# The frequencies should decrease rapidly as the count increases. Conditional on one rate,
# counts are Poisson, `P(X=k)=exp(-lambda*dt)*(lambda*dt)^k/k!`; the factorial
# makes the log-probability curve bend downward rather than form a perfect
# straight line. Pooling neurons and time points also produces a mixture of
# different Poisson rates. A monotonic, rapidly decreasing tail on the log-scale
# plot is therefore the appropriate sanity check.

# %% [markdown]
# ## 6. Empirical tuning maps from spikes
#
# These maps estimate firing rates from `X` for each spatial bin, direction, and neuron.

# %%
position_bins = np.digitize(position_clean, bin_edges[1:-1], right=False)
position_bins = np.clip(position_bins, 0, CONFIG.n_position_bins - 1)

empirical_rate_map = np.full(
    (2, CONFIG.n_position_bins, CONFIG.n_neurons),
    np.nan,
)
conditional_rate_map = np.full_like(empirical_rate_map, np.nan)
for direction_value in (0, 1):
    for spatial_bin in range(CONFIG.n_position_bins):
        mask = (
            (direction_binary == direction_value)
            & (position_bins == spatial_bin)
        )
        if np.any(mask):
            empirical_rate_map[direction_value, spatial_bin] = (
                X[mask].mean(axis=0) / CONFIG.dt
            )
            conditional_rate_map[direction_value, spatial_bin] = lam[mask].mean(axis=0)

selected_neurons = []
for neuron_type in active_types:
    candidates = np.flatnonzero(neuron_types == neuron_type)
    selected_neurons.extend(candidates[:2].tolist())

fig, axes = plt.subplots(
    len(selected_neurons),
    1,
    figsize=(10, 2.4 * len(selected_neurons)),
    sharex=True,
    squeeze=False,
)
for row, neuron in enumerate(selected_neurons):
    axis = axes[row, 0]
    axis.plot(bin_centers, empirical_rate_map[0, :, neuron], label="return", marker="o")
    axis.plot(bin_centers, empirical_rate_map[1, :, neuron], label="outbound", marker="o")
    axis.set_ylabel("Hz")
    axis.set_title(
        f"Neuron {neuron}: {neuron_types[neuron]} | "
        f"bin={preferred_bins[neuron]}, dir={preferred_directions[neuron]}"
    )
    axis.grid(alpha=0.2)
axes[0, 0].legend()
axes[-1, 0].set_xlabel("Track position")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Dynamic view of one trial
#
# The upper panel shows the rat. The lower panel shows the conditional rates of
# a small set of identified neurons at the same instant.

# %%
animation_neurons = np.asarray(selected_neurons[: min(8, len(selected_neurons))])
fig, (track_axis, rate_axis) = plt.subplots(2, 1, figsize=(11, 6))
track_axis.plot([0, 1], [0, 0], color="black", linewidth=5)
rat_dot, = track_axis.plot([], [], "o", color="crimson", markersize=13)
track_axis.set_xlim(-0.05, 1.05)
track_axis.set_ylim(-0.2, 0.2)
track_axis.set_yticks([])
track_axis.set_title("Rat position")

bar_colors = [colors[neuron_types[n]] for n in animation_neurons]
bars = rate_axis.bar(
    np.arange(len(animation_neurons)),
    np.zeros(len(animation_neurons)),
    color=bar_colors,
)
rate_axis.set_xticks(
    np.arange(len(animation_neurons)),
    labels=[f"n{n}\n{neuron_types[n]}" for n in animation_neurons],
)
rate_axis.set_ylabel("Conditional rate (Hz)")
rate_axis.set_ylim(0, max(1.0, 1.1 * lam[trial, :, animation_neurons].max()))
status = rate_axis.text(0.01, 0.95, "", transform=rate_axis.transAxes, va="top")

def update_animation(frame):
    rat_dot.set_data([position_clean[trial, frame]], [0])
    rates = lam[trial, frame, animation_neurons]
    for bar, rate in zip(bars, rates):
        bar.set_height(rate)
    direction_name = "outbound" if direction_binary[trial, frame] == 1 else "return"
    status.set_text(
        f"time={frame}, bin={position_bins[trial, frame]}, direction={direction_name}"
    )
    return rat_dot, *bars, status

animation = FuncAnimation(
    fig,
    update_animation,
    frames=CONFIG.trial_length,
    interval=60,
    blit=False,
)
plt.close(fig)
display(HTML(animation.to_jshtml(fps=18, default_mode="loop")))
