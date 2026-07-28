"""Controlled circular and linear synthetic-task experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from neurobridge.data.dataset import TemporalWindowDataset
from neurobridge.data.sim import LatentTrajectoryGenerator, build_structured_B
from neurobridge.data.sim.builders import drive_to_rate, rate_to_spike
from neurobridge.eval.representation import evaluate_latent_recovery
from neurobridge.losses.infonce import soft_contrastive_loss
from neurobridge.models.temporal_cnn import TemporalCNNEncoder
from neurobridge.sampling.batch_similarity import (
    batch_structured_similarity_from_specs,
)
from neurobridge.sampling.f_windows import build_windows
from neurobridge.train.loop import encode_windows, train_epoch


@dataclass(frozen=True)
class SyntheticTaskConfig:
    """Complete configuration for one controlled baseline experiment."""

    name: str
    condition_mode: str
    latent_dim: int
    n_trials: int = 160
    trial_length: int = 100
    n_neurons: int = 100
    n_conditions: int = 8
    window_size: int = 10
    stride: int = 1
    cnn_epochs: int = 10
    cnn_embedding_dim: int | None = None
    batch_size: int = 256
    learning_rate: float = 1e-3
    temperature: float = 0.1
    similarity_tau: float = 0.5
    time_weight: float = 0.5
    label_weight: float = 0.5
    phi: float = 0.4
    noise_scale: float = 0.05
    dt: float = 0.02
    baseline_mean: float = 1.0
    baseline_std: float = 0.10
    first_coordinates_multiplier: float = 3.0
    place_fraction: float = 0.25
    place_width: float = 0.10
    place_scale: float = 3.0
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.condition_mode not in {"circular", "linear"}:
            raise ValueError("condition_mode must be circular or linear.")
        minimum_dim = 3 if self.condition_mode == "circular" else 2
        if self.latent_dim < minimum_dim:
            raise ValueError(
                f"{self.condition_mode} requires latent_dim >= {minimum_dim}."
            )
        if not 0.0 <= self.place_fraction < 1.0:
            raise ValueError("place_fraction must satisfy 0 <= value < 1.")
        if self.place_width <= 0 or self.place_scale < 0:
            raise ValueError("Place-field width and scale must be positive.")

    @property
    def embedding_dim(self) -> int:
        return self.latent_dim if self.cnn_embedding_dim is None else (
            self.cnn_embedding_dim
        )


def _circular_probabilities(k: int) -> dict[str, float]:
    if k == 3:
        return {
            "direction": 0.45,
            "position_or_progress": 0.20,
            "velocity": 0.0,
            "context": 0.0,
            "mixed": 0.34,
            "none": 0.01,
        }
    return {
        "direction": 0.45,
        "position_or_progress": 0.20,
        "velocity": 0.02,
        "context": 0.02,
        "mixed": 0.30,
        "none": 0.01,
    }


def build_linear_loading_and_place_fields(
    *,
    k: int,
    n_neurons: int,
    position: np.ndarray,
    place_fraction: float,
    place_width: float,
    place_scale: float,
    first_coordinates_multiplier: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build linear-task loadings and localized position-dependent drive.

    Place neurons are not represented by a linear column of ``B``. Their
    contribution is a Gaussian field evaluated along the track:

        g_j(p) = a_j exp(-(p - mu_j)^2 / (2 sigma_j^2)).
    """
    if k < 2:
        raise ValueError("Linear loading requires k >= 2.")
    if position.ndim != 2:
        raise ValueError("position must have shape (n_trials, trial_length).")

    rng = np.random.default_rng(random_state)
    B = np.zeros((k, n_neurons), dtype=float)

    if k == 2:
        names = np.array(["direction", "position", "mixed", "place", "none"])
        remaining = 1.0 - place_fraction
        probabilities = remaining * np.array([
            0.333,
            0.333,
            0.320,
            0.0,
            0.014,
        ])
        probabilities[3] = place_fraction
    else:
        names = np.array([
            "direction",
            "position",
            "velocity",
            "context",
            "mixed",
            "place",
            "none",
        ])
        remaining = 1.0 - place_fraction
        probabilities = remaining * np.array([
            0.27,
            0.27,
            0.04,
            0.04,
            0.37,
            0.0,
            0.01,
        ])
        probabilities[5] = place_fraction

    probabilities = probabilities / probabilities.sum()
    neuron_types = rng.choice(names, size=n_neurons, p=probabilities)

    for neuron, neuron_type in enumerate(neuron_types):
        if neuron_type == "direction":
            B[1, neuron] = rng.choice([-1.0, 1.0])
        elif neuron_type == "position":
            B[0, neuron] = rng.choice([-1.0, 1.0])
        elif neuron_type == "velocity" and k > 2:
            B[2, neuron] = rng.choice([-1.0, 1.0])
        elif neuron_type == "context" and k > 3:
            B[3, neuron] = rng.choice([-1.0, 1.0])
        elif neuron_type == "mixed":
            available = np.arange(k)
            selected = rng.choice(
                available,
                size=min(max(2, k // 2), k),
                replace=False,
            )
            B[selected, neuron] = rng.choice(
                [-1.0, 1.0],
                size=len(selected),
            )
            B[:, neuron] /= np.linalg.norm(B[:, neuron])

    B[:2, :] *= first_coordinates_multiplier

    centers = np.full(n_neurons, np.nan)
    place_indices = np.flatnonzero(neuron_types == "place")
    centers[place_indices] = rng.uniform(0.05, 0.95, size=len(place_indices))

    place_drive = np.zeros((*position.shape, n_neurons), dtype=float)
    for neuron in place_indices:
        centered_position = position - centers[neuron]
        place_drive[:, :, neuron] = place_scale * np.exp(
            -(centered_position**2) / (2.0 * place_width**2)
        )

    return B, neuron_types, centers, place_drive


def _build_windows_and_labels(
    X: np.ndarray,
    condition: np.ndarray | None,
    state: dict[str, np.ndarray],
    config: SyntheticTaskConfig,
) -> tuple[TemporalWindowDataset, dict[str, np.ndarray]]:
    X_flat = X.reshape(-1, X.shape[-1])
    labels = condition if config.condition_mode == "circular" else None

    (
        X_windows,
        time_id,
        global_time_id,
        trial_id,
        labels_windows,
    ) = build_windows(
        X_flat,
        config.window_size,
        config.stride,
        labels=labels,
        trial_len=config.trial_length,
        time_mode="absolute",
        padding="center",
    )

    if config.condition_mode == "linear":
        local_time = np.asarray(time_id, dtype=int)
        velocity = state["velocity"][trial_id, local_time]
        labels_windows = (velocity < 0).astype(int)

    dataset = TemporalWindowDataset(
        X_windows,
        time_id,
        global_time_id,
        trial_id,
        labels_windows,
    )
    metadata = {
        "time_id": np.asarray(time_id),
        "global_time_id": np.asarray(global_time_id),
        "trial_id": np.asarray(trial_id),
        "labels": np.asarray(labels_windows),
    }
    return dataset, metadata


def _similarity_builder(
    batch: dict[str, torch.Tensor],
    config: SyntheticTaskConfig,
) -> torch.Tensor:
    label_geometry = (
        "circular" if config.condition_mode == "circular" else "categorical"
    )
    label_spec: dict[str, object] = {
        "key": "label",
        "geometry": label_geometry,
        "weight": config.label_weight,
    }
    if label_geometry == "circular":
        label_spec["num_labels"] = config.n_conditions

    return batch_structured_similarity_from_specs(
        batch,
        [
            {
                "key": "time_id",
                "geometry": "temporal",
                "weight": config.time_weight,
            },
            label_spec,
        ],
        tau=config.similarity_tau,
        normalize=True,
    )


def _fit_models(
    dataset: TemporalWindowDataset,
    config: SyntheticTaskConfig,
    device: torch.device,
) -> dict[str, object]:
    X_windows = dataset.X_windows.numpy()
    flattened = X_windows.reshape(len(dataset), -1)
    pca_dim = min(config.latent_dim, flattened.shape[1])
    pca = PCA(n_components=pca_dim, random_state=config.random_state)
    pca_embedding = pca.fit_transform(flattened)

    model = TemporalCNNEncoder(
        n_features=X_windows.shape[-1],
        embedding_dim=config.embedding_dim,
        hidden_dim=64,
        kernel_size=3,
        n_layers=3,
        normalize=True,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=1e-4,
    )
    training_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    losses: list[float] = []
    for _ in range(config.cnn_epochs):
        loss = train_epoch(
            model,
            training_loader,
            optimizer,
            lambda z, similarity: soft_contrastive_loss(
                z,
                similarity,
                temperature=config.temperature,
            ),
            device=device,
            similarity_builder=lambda batch: _similarity_builder(
                batch,
                config,
            ),
        )
        losses.append(float(loss))

    ordered_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    cnn_embedding, _ = encode_windows(model, ordered_loader, device=device)
    return {
        "pca_model": pca,
        "pca_embedding": pca_embedding,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_,
        "cnn_model": model,
        "cnn_embedding": cnn_embedding.numpy(),
        "cnn_losses": np.asarray(losses),
    }


def _evaluate_models(
    *,
    Z: np.ndarray,
    metadata: dict[str, np.ndarray],
    fitted: dict[str, object],
    max_samples: int = 2000,
) -> dict[str, dict[str, float]]:
    trial_id = metadata["trial_id"].astype(int)
    time_id = metadata["time_id"].astype(int)
    latent_target = Z[trial_id, time_id]

    n_samples = len(latent_target)
    evaluation_indices = np.linspace(
        0,
        n_samples - 1,
        min(max_samples, n_samples),
        dtype=int,
    )

    metrics: dict[str, dict[str, float]] = {}
    for model_name, embedding_key in [
        ("pca", "pca_embedding"),
        ("cnn1d", "cnn_embedding"),
    ]:
        embedding = np.asarray(fitted[embedding_key])
        metrics[model_name] = evaluate_latent_recovery(
            embedding[evaluation_indices],
            latent_target[evaluation_indices],
        )

    return metrics


def _save_overview(
    results: dict[str, object],
    output_root: Path,
) -> None:
    metadata = results["metadata"]
    labels = metadata["labels"]
    time_id = metadata["time_id"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, key, title in [
        (axes[0], "pca_embedding", "PCA"),
        (axes[1], "cnn_embedding", "CNN1D"),
    ]:
        embedding = results[key]
        scatter = axis.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=labels,
            s=4,
            alpha=0.35,
            cmap="hsv",
        )
        axis.set_title(f"{title}: {results['config']['name']}")
        axis.set_xlabel("dimension 1")
        axis.set_ylabel("dimension 2")
        fig.colorbar(scatter, ax=axis, label="condition/direction")

    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_root / "embedding_overview.png", dpi=180)
    plt.close(fig)

    if results["config"]["condition_mode"] == "linear":
        fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        embedding = results["cnn_embedding"]
        scatter = axis.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=time_id,
            s=4,
            alpha=0.35,
            cmap="viridis",
        )
        axis.set_title("CNN1D embedding colored by track time")
        axis.set_xlabel("dimension 1")
        axis.set_ylabel("dimension 2")
        fig.colorbar(scatter, ax=axis, label="time bin")
        fig.savefig(figure_root / "cnn_embedding_by_track_time.png", dpi=180)
        plt.close(fig)


def run_synthetic_task_experiment(
    config: SyntheticTaskConfig,
    *,
    project_root: str | Path | None = None,
) -> dict[str, object]:
    """Generate spikes, fit PCA/CNN, save artifacts, and return results."""
    random.seed(config.random_state)
    np.random.seed(config.random_state)
    torch.manual_seed(config.random_state)

    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path.cwd().resolve()
    )
    output_root = root / "outputs" / config.name
    output_root.mkdir(parents=True, exist_ok=True)

    conditions = (
        np.arange(config.n_conditions)
        if config.condition_mode == "circular"
        else None
    )
    generator = LatentTrajectoryGenerator(
        config.n_trials,
        config.trial_length,
        config.latent_dim,
        config.phi,
        conditions=conditions,
        condition_mode=config.condition_mode,
        n_conditions=config.n_conditions,
        noise_scale=config.noise_scale,
        condition_type="balanced",
    )
    Z, condition, state = generator.generate_latent(return_state=True)

    if config.condition_mode == "circular":
        B, neuron_types = build_structured_B(
            k=config.latent_dim,
            n_neurons=config.n_neurons,
            conditions=conditions,
            n_conditions=config.n_conditions,
            condition_mode="circular",
            directional_scale=1.0,
            position_scale=1.0,
            velocity_scale=1.0,
            context_scale=1.0,
            neuron_type_probabilities=_circular_probabilities(
                config.latent_dim
            ),
            random_state=config.random_state + 1,
            return_neuron_types=True,
        )
        B[:3, :] *= config.first_coordinates_multiplier
        place_centers = np.full(config.n_neurons, np.nan)
        place_drive = np.zeros(
            (config.n_trials, config.trial_length, config.n_neurons)
        )
    else:
        (
            B,
            neuron_types,
            place_centers,
            place_drive,
        ) = build_linear_loading_and_place_fields(
            k=config.latent_dim,
            n_neurons=config.n_neurons,
            position=state["position"][:, :, 0],
            place_fraction=config.place_fraction,
            place_width=config.place_width,
            place_scale=config.place_scale,
            first_coordinates_multiplier=config.first_coordinates_multiplier,
            random_state=config.random_state + 1,
        )

    rng = np.random.default_rng(config.random_state + 2)
    baseline = rng.normal(
        config.baseline_mean,
        config.baseline_std,
        size=config.n_neurons,
    )
    u = Z @ B + baseline + place_drive
    lam = drive_to_rate(u, "softplus")
    X = rate_to_spike(lam, config.dt)

    dataset, metadata = _build_windows_and_labels(
        X,
        condition,
        state,
        config,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fitted = _fit_models(dataset, config, device)
    metrics = _evaluate_models(
        Z=Z,
        metadata=metadata,
        fitted=fitted,
    )

    results: dict[str, object] = {
        "config": asdict(config),
        "device": str(device),
        "Z": Z,
        "condition": condition,
        "state": state,
        "B": B,
        "baseline": baseline,
        "neuron_types": neuron_types,
        "place_centers": place_centers,
        "u": u,
        "lam": lam,
        "X": X,
        "metadata": metadata,
        "metrics": metrics,
        **fitted,
    }

    model_root = output_root / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted["pca_model"], model_root / "pca.joblib")
    torch.save(
        fitted["cnn_model"].state_dict(),
        model_root / "cnn1d_state_dict.pt",
    )

    serializable_results = {
        key: value
        for key, value in results.items()
        if key not in {"pca_model", "cnn_model"}
    }
    joblib.dump(serializable_results, output_root / "results.joblib")
    _save_overview(results, output_root)
    return results
