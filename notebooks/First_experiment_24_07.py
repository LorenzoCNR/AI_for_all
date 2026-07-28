## -*- coding: utf-8 -*-

# %% [markdown]
# # Multi-subject neural simulation
#
# Generate one common latent motor task and multiple subject-specific
# neural populations.
#
# The experiment is deliberately organized into separate stages:
#
# 1. define the task and generate the shared latent tensor
#    Z_task: trials x time bins x latent coordinates;
# 2. define subject-specific population characteristics without changing
#    Z_task;
# 3. construct each loading matrix B and baseline vector c;
# 4. apply the subject-specific temporal lag to the neural driver;
# 5. map the driver through u = ZB + c, a positive rate nonlinearity, and
#    the stochastic spike-count emission model;
# 6. validate and save the complete simulation for downstream encoders.
#
# The two populations therefore share the controlled task geometry but differ
# in neural readout, neuron count, temporal lag, and emission variability.

# %%
# ====
# IMPORTS (moduli)
# ===

from math import pi
from pathlib import Path
import os
import sys

import numpy as np
import matplotlib.pyplot as plt


# ## Project source folder
#
# When this file is executed as a script, the repository root is the parent
# of ``notebooks``. In a VS Code interactive window ``__file__`` may be
# unavailable, so the current working directory remains the fallback.
try:
    project_root = Path(__file__).resolve().parents[1]
except NameError:
    project_root = Path.cwd().resolve()

src_path = project_root / "src"

if not (src_path / "neurobridge").exists():
    project_root = Path.cwd().resolve()
    src_path = project_root / "src"

if not (src_path / "neurobridge").exists():
    raise FileNotFoundError(
        "The NeuroBridge package folder was not found. "
        f"Expected path: {src_path / 'neurobridge'}"
    )

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

print("Project root:", project_root)
print("Source path:", src_path)

# ===============
#  IMPORTS (project)
# ==============

from neurobridge.data.sim.Lat_traj_generator import (
    LatentTrajectoryGenerator,
)

from neurobridge.data.sim.Spikes_generator import (
    SpikeEmissionGenerator,
)

from neurobridge.data.sim.builders import (
    apply_temporal_lag,
    build_direction_dominant_B,
    build_structured_B,)

print("Imports ok!")


# %% 
# =========================
# SHARED TASK PARAMETERS
#  TAsk Can be linear (i.e. rats) or circular (i.e. monkey)
# =========================

CONDITION_MODE = "circular"
N_CONDITIONS = 8
CONDITIONS = np.arange(N_CONDITIONS)

N_TRIALS = 160
TRIAL_LENGTH = 100
K = 5
LOADING_MODE = os.environ.get(
    "NEUROBRIDGE_LOADING_MODE",
    "heterogeneous",
).strip().lower()
DIRECTION_DOMINANT_SCALE = 3.0
DIRECTION_DOMINANT_EXTRA_SCALE = 0.051
FIRST3_LOADING_MULTIPLIER = 3.0
FIRST3_ENRICHED_PROBABILITIES = {
    "direction": 0.45,
    "position_or_progress": 0.20,
    "velocity": 0.02,
    "context": 0.02,
    "mixed": 0.30,
    "none": 0.01,
}

PHI = 0.40
NOISE_SCALE = 0.05
CONDITION_TYPE = "balanced"
SPEED_RANGE = (0.8, 1.2)
CONTEXT_RANGE = (0.7, 1.3)
TASK_RANDOM_STATE = 42

# bin size 
DT = 0.02 # (20ms)
NONLINEARITY = "softplus"

# =========================
# LATENT-DIMENSION METADATA
# =========================
#
# The number of latent dimensions K and the number of neuron types are
# different quantities and must not be forced to be equal.
#
# These names describe the columns actually constructed by
# LatentTrajectoryGenerator for the selected task mode.

def build_latent_dim_names(condition_mode, k):
    if condition_mode == "circular":
        canonical_names = [
            "Position X",
            "Position Y",
            "Movement progress",
            "Velocity",
            "Context",
        ]
    elif condition_mode == "linear":
        canonical_names = [
            "Position",
            "Movement direction",
            "Velocity",
            "Context",
        ]
    else:
        raise ValueError(
            f"Unsupported condition_mode: {condition_mode}"
        )

    latent_dim_names = canonical_names[:k]

    # Dimensions beyond the explicitly modelled task coordinates contain
    # only the stochastic latent component in the current generator.
    if k > len(canonical_names):
        latent_dim_names.extend(
            [
                f"Stochastic latent {dim_index + 1}"
                for dim_index in range(
                    len(canonical_names),
                    k,
                )
            ]
        )

    if len(latent_dim_names) != k:
        raise ValueError(
            "The number of latent dimension names must match K: "
            f"{len(latent_dim_names)} != {k}"
        )

    return latent_dim_names


LATENT_DIM_NAMES = build_latent_dim_names(
    condition_mode=CONDITION_MODE,
    k=K,
)


# %% [markdown]
# ##  Subject-specific configurations

subject_configs = [
    { "subject_name": "subject_01",
        "n_neurons": 100,
        "lag_bins": 0,

        "directional_scale": 1.0,
        "position_scale": 1.0,
        "velocity_scale": 1.0,
        "context_scale": 1.0,

        "neuron_type_probabilities": {
            # In circular mode, direction and mixed neurons load on the
            # first two latent coordinates (Position X and Position Y).
            "direction": 0.30,
            "position_or_progress": 0.15,
            "velocity": 0.10,
            "context": 0.10,
            "mixed": 0.30,
            "none": 0.05,
        },

        "baseline_mean": 1.0,
        "baseline_std": 0.10,
        "overdispersion": 0.20,
        "refractory_mean_bins": 2,
        "refractory_std_bins": 1.0,

        "burst_probability": 0.05,
        "burst_size_mean": 1.5,
        "burst_window_bins": 3,
        "random_state": 43,
    },

    {   "subject_name": "subject_02",
        "n_neurons": 140,
        "lag_bins": 4,

        "directional_scale": 0.8,
        "position_scale": 1.2,
        "velocity_scale": 1.5,
        "context_scale": 0.6,
        "neuron_type_probabilities": {
            # The second population remains distinct while still allocating
            # most neurons to the shared spatial coordinates.
            "direction": 0.30,
            "position_or_progress": 0.15,
            "velocity": 0.15,
            "context": 0.10,
            "mixed": 0.25,
            "none": 0.05,
        },

        "baseline_mean": 1.3,
        "baseline_std": 0.20,
        "overdispersion": 0.35,
        "refractory_mean_bins": 3,
        "refractory_std_bins": 1.5,
        "burst_probability": 0.10,
        "burst_size_mean": 2.0,
        "burst_window_bins": 5,
        "random_state": 44,
    },
]


# %% [markdown]
# ## Check and Validate subject(s) configurations

required_subject_keys = [
    "subject_name",
    "n_neurons",
    "lag_bins",
    "directional_scale",
    "position_scale",
    "velocity_scale",
    "context_scale",
    "neuron_type_probabilities",
    "baseline_mean",
    "baseline_std",
    "overdispersion",
    "refractory_mean_bins",
    "refractory_std_bins",
    "burst_probability",
    "burst_size_mean",
    "burst_window_bins",
    "random_state",
]

N_SUBJECTS = len(subject_configs)

if not subject_configs:
    raise ValueError(
        "subject_configs must contain at least one subject"
    )

subject_names = []

for subject in subject_configs:

    missing_keys = [
        key
        for key in required_subject_keys
        if key not in subject
    ]

    if missing_keys:
        raise ValueError(
            "Missing configuration keys: "
            + ", ".join(missing_keys)
        )

    subject_name = subject["subject_name"]

    if subject_name in subject_names:
        raise ValueError(
            f"Duplicated subject name: {subject_name}"
        )

    subject_names.append(subject_name)

# VALIDATE NUMBER OF NEURONS

    if subject["n_neurons"] <= 0:
        raise ValueError(
            f"{subject_name}: n_neurons must be strictly positive"
        )

#  NORMALIZE THE TEMPORAL LAG TYPE

    subject["lag_bins"] = int(subject["lag_bins"])

    # — VALIDATE TUNING SCALES

    tuning_scale_keys = [
        "directional_scale",
        "position_scale",
        "velocity_scale",
        "context_scale",
    ]

    for scale_key in tuning_scale_keys:
        if subject[scale_key] < 0:
            raise ValueError(
                f"{subject_name}: {scale_key} must be non-negative"
            )

    #  — VALIDATE BASELINE VARIABILITY

    if subject["baseline_std"] < 0:
        raise ValueError(
            f"{subject_name}: baseline_std must be non-negative"
        )

    #— VALIDATE OVERDISPERSION

    if subject["overdispersion"] < 0:
        raise ValueError(
            f"{subject_name}: overdispersion must be non-negative"
        )

    # — VALIDATE REFRACTORY PARAMETERS

    refractory_mean = subject["refractory_mean_bins"]

    if refractory_mean is not None and refractory_mean < 0:
        raise ValueError(
            f"{subject_name}: refractory_mean_bins "
            "must be None or non-negative"
        )

    if subject["refractory_std_bins"] < 0:
        raise ValueError(
            f"{subject_name}: refractory_std_bins "
            "must be non-negative"
        )

    # — VALIDATE BURST PROBABILITY

    if not (0 <= subject["burst_probability"] <= 1):
        raise ValueError(
            f"{subject_name}: burst_probability must be in [0, 1]"
        )

    #  — VALIDATE BURST SIZE AND WINDOW

    if subject["burst_size_mean"] < 0:
        raise ValueError(
            f"{subject_name}: burst_size_mean must be non-negative"
        )

    if subject["burst_window_bins"] <= 0:
        raise ValueError(
            f"{subject_name}: burst_window_bins must be positive"
        )

    #  — EXTRACT NEURON-TYPE PROBABILITIES

    probabilities = subject["neuron_type_probabilities"]

    required_neuron_types = [
        "direction",
        "position_or_progress",
        "velocity",
        "context",
        "mixed",
        "none",
    ]

    #  — FIND MISSING NEURON TYPES

    missing_neuron_types = [
        neuron_type
        for neuron_type in required_neuron_types
        if neuron_type not in probabilities
    ]

    if missing_neuron_types:
        raise ValueError(
            f"{subject_name}: missing neuron types: "
            + ", ".join(missing_neuron_types)
        )

    #  — CONVERT PROBABILITIES TO A NUMPY ARRAY

    probability_values = np.array(
        [
            probabilities[neuron_type]
            for neuron_type in required_neuron_types
        ],
        dtype=float,
    )

    # — VALIDATE PROBABILITY RANGE

    if np.any(
        (probability_values < 0)
        | (probability_values > 1)
    ):
        raise ValueError(
            f"{subject_name}: neuron-type probabilities "
            "must be between 0 and 1"
        )

    #— VALIDATE PROBABILITY SUM

    if not np.isclose(
        probability_values.sum(),
        1.0,
    ):
        raise ValueError(
            f"{subject_name}: neuron-type probabilities "
            "must sum to 1"
        )

# — VALIDATE COMPATIBILITY BETWEEN ACTIVE NEURON TYPES AND K
    #
    # Check whether a neuron type with positive
    # probability has the latent coordinate that it needs.

    if CONDITION_MODE == "circular":
        velocity_is_available = K > 3
        context_is_available = K > 4
    else:
        velocity_is_available = K > 2
        context_is_available = K > 3

    if (
        probabilities["velocity"] > 0
        and not velocity_is_available
    ):
        raise ValueError(
            f"{subject_name}: velocity neurons have positive "
            f"probability, but K={K} does not include a "
            f"velocity coordinate in {CONDITION_MODE} mode"
        )

    if (
        probabilities["context"] > 0
        and not context_is_available
    ):
        raise ValueError(
            f"{subject_name}: context neurons have positive "
            f"probability, but K={K} does not include a "
            f"context coordinate in {CONDITION_MODE} mode"
        )

# %%
# =========================
# VALIDATION SUMMARY
# =========================

print("Number of subjects:", N_SUBJECTS)
print("Subject names:", subject_names)
print("All subject configurations are valid")

#THE RANDOM SEED
np.random.seed(TASK_RANDOM_STATE)


#CREATE THE LATENT TRAJECTORY GENERATOR
# conditions should be CONDITIONS only for a circular task.
# For a linear task, use None.

trajectory_generator = LatentTrajectoryGenerator(
    n_trials=N_TRIALS,
    L=TRIAL_LENGTH,
    k=K,
    phi=PHI,
    conditions=CONDITIONS,
    condition_mode=CONDITION_MODE,
    n_conditions=N_CONDITIONS,
    noise_scale=NOISE_SCALE,
    condition_type=CONDITION_TYPE,
    speed_range=SPEED_RANGE,
    context_range=CONTEXT_RANGE,
)


# GENERATE THE LATENT DATA

Z_task, cond, state = trajectory_generator.generate_latent(return_state=True)


#CHECK THE LATENT SHAPE
latent_dim=Z_task.shape
assert latent_dim ==(N_TRIALS, TRIAL_LENGTH, K)
print("Z_task shape:",latent_dim)
print("cond shape:", cond.shape)
print("State keys:", list(state.keys()))

# %%
# ===============================
# Empty dictionary that will store all subjects.

subjects = {}

for subject in subject_configs:
    # Extract the current subject name, neuron count and random seed.
    subject_name = subject['subject_name']
    n_neurons = subject["n_neurons"]
    subject_seed = subject["random_state"]
    #subject-specific NumPy random generator.

    subject_rng =  np.random.default_rng(subject_seed)

    # Build the subject-specific loading matrix.
    #     B
    subject_effective_config = dict(subject)

    if LOADING_MODE in {
        "heterogeneous",
        "first3_enriched",
        "first3_enriched_strong",
    }:
        effective_probabilities = (
            subject["neuron_type_probabilities"]
            if LOADING_MODE == "heterogeneous"
            else FIRST3_ENRICHED_PROBABILITIES
        )
        subject_effective_config["neuron_type_probabilities"] = dict(
            effective_probabilities
        )
        B, neuron_types = build_structured_B(
            k=K,
            n_neurons=n_neurons,
            conditions=(
                CONDITIONS
                if CONDITION_MODE == "circular"
                else None
            ),
            n_conditions=N_CONDITIONS,
            condition_mode=CONDITION_MODE,
            directional_scale=subject["directional_scale"],
            position_scale=subject["position_scale"],
            velocity_scale=subject["velocity_scale"],
            context_scale=subject["context_scale"],
            neuron_type_probabilities=effective_probabilities,
            random_state=subject_seed,
            return_neuron_types=True,
        )
        if LOADING_MODE == "first3_enriched_strong":
            B[:3, :] *= FIRST3_LOADING_MULTIPLIER
    elif LOADING_MODE == "direction_dominant":
        B = build_direction_dominant_B(
            k=K,
            n_neurons=n_neurons,
            conditions=CONDITIONS,
            n_conditions=N_CONDITIONS,
            directional_scale=DIRECTION_DOMINANT_SCALE,
            extra_scale=DIRECTION_DOMINANT_EXTRA_SCALE,
            random_state=subject_seed,
        )
        neuron_types = np.full(
            n_neurons,
            "direction_dominant",
            dtype=object,
        )
    else:
        raise ValueError(
            "NEUROBRIDGE_LOADING_MODE must be "
            "'heterogeneous', 'first3_enriched', "
            "'first3_enriched_strong', "
            "or 'direction_dominant'"
        )

    # Generate the subject-specific baseline vector.
    # Distribution:
    #     normal
    # Parameters:
    #     baseline_mean
    #     baseline_std
    c = subject_rng.normal(
    loc=subject["baseline_mean"],
    scale=subject["baseline_std"],
    size=n_neurons,)

    # Apply the subject-specific temporal lag to Z_task.
    # Store the result in Z_neural_driver.
    # Z_taskhave to remain unchanged.

    Z_neural_driver = apply_temporal_lag(
    Z_task,
    lag_bins=subject["lag_bins"],)
    # ===========================
    #  CREATE THE SPIKE EMITTER
    # ===========================
    
    emitter = SpikeEmissionGenerator(B=B,c=c, dt=DT, nonlinearity=NONLINEARITY,overdispersion=subject["overdispersion"],
        refractory_mean_bins=subject["refractory_mean_bins"], refractory_std_bins=subject["refractory_std_bins"],
        burst_probability=subject["burst_probability"], burst_size_mean=subject["burst_size_mean"],
         burst_window_bins=subject["burst_window_bins"])
    
    
    # ===============================
    #  — GENERATE NEURAL ACTIVITY
    # =========================
    np.random.seed(subject_seed)
    u, lam, X = emitter.generate_spikes(
        Z_neural_driver)
    
    subjects[subject_name] = {
            "config": subject_effective_config,
            "Z_neural_driver": Z_neural_driver,
            "B": B,
            "c": c,
            "neuron_types": neuron_types,
            "u": u,
            "lam": lam,
            "X": X,
        }
    
        # CHECK CURRENT SUBJECT SHAPES
        
    expected_neural_shape = (
        N_TRIALS,
        TRIAL_LENGTH,
        n_neurons,)

    assert B.shape == (K, n_neurons)
    assert c.shape == (n_neurons,)
    assert Z_neural_driver.shape == Z_task.shape
    assert u.shape == expected_neural_shape
    assert lam.shape == expected_neural_shape
    assert X.shape == expected_neural_shape


# ## . Latent-space diagnostics
# Verify the geometry and temporal structure of the shared latent task
# Select one valid trial index.
import random
selected_trial = random.randrange(N_TRIALS)

Z_trial = Z_task[selected_trial, :, :]
trial_condition = cond[selected_trial]

assert Z_trial.shape== (TRIAL_LENGTH, K)

# %% — PLOT THE FIRST TWO LATENT COORDINATES

#     a marker for the starting point
fig, ax = plt.subplots(figsize=(6, 6))

time_bins = np.arange(TRIAL_LENGTH)

scatter = ax.scatter(
    Z_trial[:, 0],
    Z_trial[:, 1],
    c=time_bins,
    label="Latent states",
)

ax.plot(
    Z_trial[:, 0],
    Z_trial[:, 1],
    alpha=0.4,
    label="Temporal path",
)

ax.scatter(
    Z_trial[0, 0],
    Z_trial[0, 1],
    marker="*",
    s=200,
    zorder=5,
    label="Start",
)

ax.scatter(
    Z_trial[-1, 0],
    Z_trial[-1, 1],
    marker="x",
    s=100,
    zorder=5,
    label="End",
)

ax.set_xlabel("Latent coordinate 0")
ax.set_ylabel("Latent coordinate 1")
ax.set_title(
    f"Trial {selected_trial} — condition {trial_condition}"
)
ax.axis("equal")
ax.legend()

fig.colorbar(
    scatter,
    ax=ax,
    label="Time bin",
)

fig.tight_layout()
plt.show()

#
# %% — PLOT MEAN LATENT COORDINATES FOR ONE CONDITION

time_bins = np.arange(TRIAL_LENGTH)

latent_names = [
    "Position X",
    "Position Y",
    "Movement progress",
    "Velocity",
    "Context",
]

selected_condition = trial_condition

condition_mask = cond == selected_condition

n_cols = 2
n_rows = int(np.ceil(K / n_cols))

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(10, 3 * n_rows),
)

axes_flat = np.atleast_1d(axes).flatten()

for i in range(K):
    ax = axes_flat[i]

    mean_latent = np.mean(
        Z_task[condition_mask, :, i],
        axis=0,
    )

    ax.plot(
        time_bins,
        mean_latent,
    )

    coordinate_name = (
        latent_names[i]
        if i < len(latent_names)
        else f"Latent coordinate {i}"
    )

    ax.set_xlabel("Time bin")
    ax.set_ylabel("Latent value")
    ax.set_title(
        f"{coordinate_name} — condition {selected_condition}"
    )

for i in range(K, len(axes_flat)):
    axes_flat[i].set_visible(False)

fig.suptitle(
    f"Mean latent dynamics for condition {selected_condition}"
)

fig.tight_layout()
plt.show()


# %% CONDITION BALANCE
#
# Compute the unique condition labels and the number of trials
# associated with each condition.
#
# Expected outputs:
#
#     condition_labels
#     condition_counts
#
# Both should be one-dimensional NumPy arrays.

condition_labels, condition_counts = np.unique(cond, return_counts=True)

assert condition_labels.shape == condition_counts.shape

print("Condition labels:", condition_labels)
print("Condition counts:", condition_counts)

# Spazio latente raggiera diciamo

fig, ax = plt.subplots(figsize=(7, 7))

for condition in condition_labels:
    condition_mask = cond == condition

    mean_trajectory = np.mean(
        Z_task[condition_mask],
        axis=0,
    )

    ax.plot(
        mean_trajectory[:, 0],
        mean_trajectory[:, 1],
        label=f"Condition {condition}",
    )

    # Mark the endpoint of each mean trajectory.
    ax.scatter(
        mean_trajectory[-1, 0],
        mean_trajectory[-1, 1],
        s=60,
    )

ax.scatter(
    0,
    0,
    marker="*",
    s=180,
    label="Origin",
)

ax.set_xlabel("Position X")
ax.set_ylabel("Position Y")
ax.set_title("Mean latent trajectories by condition")
ax.axis("equal")
ax.legend()
fig.tight_layout()
plt.show()

# %%
# ================
# VISUALIZE ONE SUBJECT-SPECIFIC TEMPORAL LAG
# ==================
#
# Use the same trial and coordinate for both curves.
selected_subject = "subject_02"
selected_coordinate = 2

Z_driver_trial = subjects[selected_subject][
    "Z_neural_driver"
][selected_trial]

assert Z_driver_trial.shape == Z_trial.shape

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    time_bins,
    Z_trial[:, selected_coordinate],
    label="Original latent",
)

ax.plot(
    time_bins,
    Z_driver_trial[:, selected_coordinate],
    label="Lagged neural driver",
)

ax.set_xlabel("Time bin")
ax.set_ylabel("Latent value")
ax.set_title(
    f"Temporal lag — {selected_subject}, "
    f"lag={subjects[selected_subject]['config']['lag_bins']} bins"
)
ax.legend()

fig.tight_layout()
plt.show()

# %%
# ==================================
# SUBJECT-SPECIFIC NEURAL DIAGNOSTICS
# =================================

assert len(subjects) == N_SUBJECTS

print("Generated subjects:", list(subjects.keys()))
print("Number of generated subjects:", len(subjects))

subject_diagnostics = {}

for subject_name, subject_data in subjects.items():

    X_subject = subject_data["X"]
    u_subject = subject_data["u"]
    lam_subject = subject_data["lam"]
    neuron_types_subject = subject_data["neuron_types"]

    type_names, type_counts = np.unique(
        neuron_types_subject,
        return_counts=True,
    )

    subject_diagnostics[subject_name] = {
        "n_neurons": subject_data["config"]["n_neurons"],
        "mean_drive": np.mean(u_subject),
        "min_rate": np.min(lam_subject),
        "max_rate": np.max(lam_subject),
        "mean_rate": np.mean(lam_subject),
        "mean_spike_count": np.mean(X_subject),
        "active_fraction": np.mean(X_subject > 0),
        "neuron_type_counts": dict(
            zip(type_names, type_counts)
        ),
    }


for subject_name, diagnostics in subject_diagnostics.items():

    print(subject_name)
    print("  neurons:", diagnostics["n_neurons"])
    print("  mean drive:", diagnostics["mean_drive"])
    print("  minimum rate:", diagnostics["min_rate"])
    print("  maximum rate:", diagnostics["max_rate"])
    print("  mean rate:", diagnostics["mean_rate"])
    print(
        "  mean spike count:",
        diagnostics["mean_spike_count"],
    )
    print(
        "  active fraction:",
        diagnostics["active_fraction"],
    )
    print(
        "  neuron types:",
        diagnostics["neuron_type_counts"],
    )
    print()

# %%
# ===============
# SPIKE RASTER
# ===================

raster_subject = "subject_02"
raster_trial = selected_trial

X_trial = subjects[raster_subject]["X"][raster_trial]

expected_raster_shape = (
    TRIAL_LENGTH,
    subjects[raster_subject]["config"]["n_neurons"],
)

assert X_trial.shape == expected_raster_shape

fig, ax = plt.subplots(figsize=(10, 5))

image = ax.imshow(
    X_trial.T,
    aspect="auto",
    origin="lower",
    interpolation="nearest",
)

ax.set_xlabel("Time bin")
ax.set_ylabel("Neuron")
ax.set_title(
    f"Spike raster — {raster_subject}, "
    f"trial {raster_trial}"
)

fig.colorbar(
    image,
    ax=ax,
    label="Spike count",
)

fig.tight_layout()
plt.show()

## counts and frequencies of spikes (per bin)
X_subject = subjects["subject_02"]["X"]

unique_X, counts_X = np.unique(
    X_subject,
    return_counts=True,
)

print("Spike values:", unique_X)
print("Spike counts:", counts_X)

# %%
# ========================
# SAVE COMPLETE SIMULATION
# =======================

import pickle

# =========================
# SAVE COMPLETE SIMULATION
# =========================

experiment_names = {
    "heterogeneous": "experiment_01_circular_task_160_trials",
    "direction_dominant": (
        "experiment_02_circular_task_160_trials_direction_dominant_B"
    ),
    "first3_enriched": (
        "experiment_03_circular_task_160_trials_first3_enriched"
    ),
    "first3_enriched_strong": (
        "experiment_04_circular_task_160_trials_first3_enriched_strong"
    ),
}
experiment_name = experiment_names[LOADING_MODE]
output_root = (
    project_root
    / "outputs"
    / experiment_name
    / "simulation"
)
output_root.mkdir(parents=True, exist_ok=True)

simulation_path = output_root / "simulation.pkl"

simulation_data = {
    "task_config": {
        "condition_mode": CONDITION_MODE,
        "n_conditions": N_CONDITIONS,
        "conditions": CONDITIONS,
        "n_trials": N_TRIALS,
        "trial_length": TRIAL_LENGTH,
        "k": K,
        "phi": PHI,
        "noise_scale": NOISE_SCALE,
        "condition_type": CONDITION_TYPE,
        "speed_range": SPEED_RANGE,
        "context_range": CONTEXT_RANGE,
        "task_random_state": TASK_RANDOM_STATE,
        "dt": DT,
        "nonlinearity": NONLINEARITY,
        "loading_mode": LOADING_MODE,
        "direction_dominant_scale": DIRECTION_DOMINANT_SCALE,
        "direction_dominant_extra_scale": DIRECTION_DOMINANT_EXTRA_SCALE,
        "first3_enriched_probabilities": FIRST3_ENRICHED_PROBABILITIES,
        "first3_loading_multiplier": FIRST3_LOADING_MULTIPLIER,
        "latent_dim_names": [
            "Position X",
            "Position Y",
            "Movement progress",
            "Velocity",
            "Context",
        ],
    },
    "Z_task": Z_task,
    "cond": cond,
    "state": state,
    "subjects": subjects,
    "subject_diagnostics": subject_diagnostics,
}

with open(simulation_path, "wb") as file:
    pickle.dump(simulation_data, file)

assert simulation_path.exists()

print("Simulation saved to:", simulation_path)
