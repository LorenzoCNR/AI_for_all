# -*- coding: utf-8 -*-
"""
Conference-oriented baseline suite for NeuroBridge.

This script generates synthetic shared-latent spike data, windows it, trains
multiple temporal encoders with the NeuroBridge soft contrastive loss, and
evaluates latent recovery against the known simulator ground truth.
"""
# %%
from pathlib import Path
import os
import sys
import random
import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from torch.utils.data import DataLoader
from scipy.io import savemat
from sklearn.decomposition import PCA

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path.cwd().resolve()

if not (PROJECT_ROOT / "src" / "neurobridge").exists():
    PROJECT_ROOT = Path.cwd().resolve()

if not (PROJECT_ROOT / "src" / "neurobridge").exists():
    raise FileNotFoundError(f"Cannot find Neuro_Bridge project root from {PROJECT_ROOT}")

os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# %%
from neurobridge.data.dataset import TemporalWindowDataset
from neurobridge.data.sim import LatentTrajectoryGenerator, SpikeEmissionGenerator, build_structured_B
from neurobridge.data.sim.builders import apply_temporal_lag
from neurobridge.eval.representation import (
    evaluate_latent_recovery,
    lagged_alignment_by_trial_time,
    lagged_alignment_scores,
    procrustes_align,
)
from neurobridge.losses import soft_contrastive_loss, supervised_infonce_loss, time_offset_infonce_loss
from neurobridge.models import (
    TemporalCNNEncoder,
    TemporalLSTMEncoder,
    TemporalMLPEncoder,
    TemporalTransformerEncoder,
)
from neurobridge.sampling.batch_similarity import (
    batch_structured_similarity,
    batch_structured_similarity_from_specs,
)
from neurobridge.sampling.f_windows import build_windows
from neurobridge.train.loop import encode_windows, train_epoch
from neurobridge.viz.manifold_plots import (
    plot_condition_centroids_2d,
    plot_condition_trajectories_2d,
    plot_condition_trajectories_sphere,
    plot_embedding_2d,
    plot_embedding_sphere,
)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def make_default_config():
    return {
        "seed": 164,
        "n_trials": 160,
        "trial_len": 100,
        "n_neurons": 80,
        "n_conditions": 8,
        "n_traj_k": 3,
        "phi": 0.4,
        "condition_mode": "circular",
        "condition_type": "balanced",
        "noise_scale": 0.15,
        "directional_scale": 3.0,
        "extra_scale": 0.051,
        "dt": 0.02,
        "nonlinearity": "softplus",
        "overdispersion": 0.25,
        "refractory_mean_bins": 2,
        "refractory_std_bins": 1,
        "burst_probability": 0.05,
        "burst_size_mean": 1.5,
        "burst_window_bins": 3,
        "subject_lags_bins": {"subject_1": 0, "subject_2": 2},
        "window_size": 10,
        "stride": 1,
        "time_mode": "absolute",
        "window_padding": "center",
        "pad_value": 0.0,
        "embedding_dim": 3,
        "pca_plot_components": 5,
        "batch_size": 256,
        "epochs": 1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "temperature": 0.2,
        "loss_mode": "soft_structured",
        "positive_time_offset": 10,
        "similarity_tau": 0.5,
        "time_weight": 0.5,
        "label_weight": 0.5,
        "similarity_specs": [
            {"key": "time_id", "geometry": "temporal", "weight": 0.5},
            {"key": "label", "geometry": "circular", "num_labels": 8, "weight": 0.5},
        ],
        "encoders": ["cnn", "transformer"],
        "metric_max_samples": 2500,
        "save_all_plots": False,
    }


def make_synthetic_shared_latent_data(config):
    # Circular condition identifiers are zero based throughout NeuroBridge.
    condition = list(range(config["n_conditions"]))

    generator = LatentTrajectoryGenerator(
        config["n_trials"],
        config["trial_len"],
        config["n_traj_k"],
        config["phi"],
        condition,
        condition_mode=config["condition_mode"],
        n_conditions=config["n_conditions"],
        noise_scale=config["noise_scale"],
        condition_type=config["condition_type"],
    )
    Z_task, C, task_state = generator.generate_latent(return_state=True)

    subject_data = {}
    for subject_name, lag_bins in config["subject_lags_bins"].items():
        B = build_structured_B(
            config["n_traj_k"],
            config["n_neurons"],
            condition,
            config["n_conditions"],
            condition_mode=config["condition_mode"],
            directional_scale=config["directional_scale"],
            extra_scale=config["extra_scale"],
        )
        c = np.ones(config["n_neurons"])
        emitter = SpikeEmissionGenerator(
            B,
            c,
            dt=config["dt"],
            nonlinearity=config["nonlinearity"],
            overdispersion=config["overdispersion"],
            refractory_mean_bins=config["refractory_mean_bins"],                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
            refractory_std_bins=config["refractory_std_bins"],
            burst_probability=config["burst_probability"],
            burst_size_mean=config["burst_size_mean"],
            burst_window_bins=config["burst_window_bins"],
        )
        Z_neural_driver = apply_temporal_lag(Z_task, lag_bins=lag_bins)
        u, lam, X = emitter.generate_spikes(Z_neural_driver)
        subject_data[subject_name] = {
            "Z_task": Z_task,
            "Z_neural_driver": Z_neural_driver,
            "response_lag_bins": lag_bins,
            "B": B,
            "u": u,
            "lam": lam,
            "X": X,
        }

    return Z_task, C, task_state, subject_data


def make_window_dataset(X, Z_task, labels, config):
    X_reshaped = X.reshape(-1, X.shape[2])
    Z_reshaped = Z_task.reshape(-1, Z_task.shape[2])

    X_windows, time_id, global_time_id, trial_id, labels_windows = build_windows(
        X_reshaped,
        config["window_size"],
        config["stride"],
        labels=labels,
        trial_len=config["trial_len"],
        time_mode=config["time_mode"],
        padding=config["window_padding"],
        pad_value=config["pad_value"],
    )

    Z_windows, _, _, _, _ = build_windows(
        Z_reshaped,
        config["window_size"],
        config["stride"],
        labels=labels,
        trial_len=config["trial_len"],
        time_mode=config["time_mode"],
        padding=config["window_padding"],
        pad_value=config["pad_value"],
    )
    Z_target = Z_windows.mean(axis=1)

    dataset = TemporalWindowDataset(
        X_windows,
        time_id,
        global_time_id,
        trial_id,
        labels_windows,
    )

    return dataset, Z_target


def make_encoder(name, window_size, n_features, embedding_dim):
    if name == "cnn":
        return TemporalCNNEncoder(n_features=n_features, embedding_dim=embedding_dim, hidden_dim=64, n_layers=3)
    if name == "mlp":
        return TemporalMLPEncoder(window_size=window_size, n_features=n_features, embedding_dim=embedding_dim)
    if name == "lstm":
        return TemporalLSTMEncoder(n_features=n_features, embedding_dim=embedding_dim, hidden_dim=64)
    if name == "transformer":
        return TemporalTransformerEncoder(
            n_features=n_features,
            embedding_dim=embedding_dim,
            model_dim=64,
            n_heads=4,
            n_layers=2,
        )
    raise ValueError(f"Unknown encoder: {name}")


def evaluate_latent_recovery_sampled(embedding, latent, max_samples=None, seed=0):
    if max_samples is None or len(embedding) <= max_samples:
        return evaluate_latent_recovery(embedding, latent)

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(embedding), size=max_samples, replace=False))
    return evaluate_latent_recovery(embedding[idx], latent[idx])


def train_encoder(name, dataset, config, device):
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
    )

    sample = dataset[0]["x"]
    window_size, n_features = sample.shape
    model = make_encoder(name, window_size, n_features, config["embedding_dim"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    losses = []
    for _ in range(config["epochs"]):
        batch_loss = False
        if config["loss_mode"] == "soft_structured":
            loss_fn = lambda z, S: soft_contrastive_loss(z, S, temperature=config["temperature"])
            similarity_builder = lambda batch: batch_structured_similarity(
                batch,
                time_weight=config["time_weight"],
                label_weight=config["label_weight"],
                tau=config["similarity_tau"],
                num_labels=config["n_conditions"],
            )
        elif config["loss_mode"] == "structured_specs":
            loss_fn = lambda z, S: soft_contrastive_loss(z, S, temperature=config["temperature"])
            similarity_builder = lambda batch: batch_structured_similarity_from_specs(
                batch,
                config["similarity_specs"],
                tau=config["similarity_tau"],
            )
        elif config["loss_mode"] == "supervised_infonce":
            loss_fn = lambda z, labels: supervised_infonce_loss(z, labels, temperature=config["temperature"])
            similarity_builder = None
        elif config["loss_mode"] == "time_offset_infonce":
            loss_fn = lambda z, batch: time_offset_infonce_loss(
                z,
                batch["trial_id"],
                batch["time_id"],
                offset=config["positive_time_offset"],
                temperature=config["temperature"],
            )
            similarity_builder = None
            batch_loss = True
        else:
            raise ValueError(
                "loss_mode must be 'soft_structured', 'structured_specs', "
                "'supervised_infonce', or 'time_offset_infonce'"
            )

        loss = train_epoch(
            model,
            loader,
            optimizer,
            loss_fn,
            device=device,
            similarity_builder=similarity_builder,
            batch_loss=batch_loss,
        )
        losses.append(loss)

    eval_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
    embeddings, metadata = encode_windows(model, eval_loader, device=device)
    return embeddings.numpy(), losses, metadata


def save_embedding_mat_files(
        embeddings_by_subject,
        latent_by_subject,
        metadata_by_subject,
        subject_data,
        trial_labels,
        task_state,
        results,
        lag_results,
        config,
        output_dir):
    """Save one MATLAB file per method with both subject embeddings and metadata."""
    mat_dir = output_dir / "mat_embeddings"
    mat_dir.mkdir(parents=True, exist_ok=True)

    results_by_key = {
        (row["subject"], row["method"]): row
        for row in results
    }
    lag_by_method = {
        row["method"]: row
        for row in lag_results
    }
    methods = sorted({method for _, method in embeddings_by_subject.keys()})

    saved_paths = []
    for method in methods:
        payload = {
            "method": np.array(method, dtype=object),
            "loss_mode": np.array(config["loss_mode"], dtype=object),
            "simulation_interpretation": np.array(
                "Labels and task latent are shared across subjects; response lag affects neural activity generation.",
                dtype=object,
            ),
            "window_size": np.array(config["window_size"]),
            "stride": np.array(config["stride"]),
            "time_mode": np.array(config["time_mode"], dtype=object),
            "embedding_dim": np.array(config["embedding_dim"]),
            "dt": np.array(config["dt"]),
            "n_trials": np.array(config["n_trials"]),
            "trial_len": np.array(config["trial_len"]),
            "n_neurons": np.array(config["n_neurons"]),
            "n_conditions": np.array(config["n_conditions"]),
            "window_padding": np.array(config["window_padding"], dtype=object),
            "pad_value": np.array(config["pad_value"]),
            "trial_condition": np.asarray(trial_labels),
            "shared_latent_task": subject_data["subject_1"]["Z_task"],
        }

        if method in lag_by_method:
            payload["best_lag"] = np.array(lag_by_method[method]["best_lag"])
            payload["best_lag_score"] = np.array(lag_by_method[method]["best_score"])
            for key, value in lag_by_method[method].items():
                if key.startswith("lag_"):
                    payload[key] = np.array(value)

        for key, value in task_state.items():
            if isinstance(value, np.ndarray):
                payload[f"task_state_{key}"] = value

        for subject_name in sorted(metadata_by_subject.keys()):
            metadata = metadata_by_subject[subject_name]
            scores = results_by_key[(subject_name, method)]
            data = subject_data[subject_name]
            prefix = subject_name
            payload[f"{prefix}_embedding"] = embeddings_by_subject[(subject_name, method)]
            payload[f"{prefix}_latent_target"] = latent_by_subject[subject_name]
            payload[f"{prefix}_latent_task"] = data["Z_task"]
            payload[f"{prefix}_latent_neural_driver"] = data["Z_neural_driver"]
            payload[f"{prefix}_response_lag_bins"] = np.array(data["response_lag_bins"])
            payload[f"{prefix}_X_spikes"] = data["X"]
            payload[f"{prefix}_neural_drive_u"] = data["u"]
            payload[f"{prefix}_firing_rate_lambda"] = data["lam"]
            payload[f"{prefix}_loading_B"] = data["B"]
            payload[f"{prefix}_label"] = metadata["label"]
            payload[f"{prefix}_trial_condition"] = np.asarray(trial_labels)
            payload[f"{prefix}_trial_id"] = metadata["trial_id"]
            payload[f"{prefix}_trial_id_1based"] = metadata["trial_id"] + 1
            payload[f"{prefix}_time_id"] = metadata["time_id"]
            payload[f"{prefix}_global_time_id"] = metadata["global_time_id"]
            payload[f"{prefix}_n_windows"] = np.array(len(metadata["trial_id"]))
            payload[f"{prefix}_n_trials"] = np.array(config["n_trials"])
            payload[f"{prefix}_trial_len"] = np.array(config["trial_len"])
            payload[f"{prefix}_n_timepoints"] = np.array(data["X"].shape[0] * data["X"].shape[1])
            payload[f"{prefix}_procrustes_r2"] = np.array(scores["procrustes_r2"])
            payload[f"{prefix}_rsa_spearman"] = np.array(scores["rsa_spearman"])
            payload[f"{prefix}_rsa_pearson"] = np.array(scores["rsa_pearson"])

        path = mat_dir / f"{method}_subject_embeddings_{config['loss_mode']}.mat"
        savemat(path, payload)
        saved_paths.append(path)

    return saved_paths


def condition_color(label_index, n_labels):
    cmap = plt.get_cmap("hsv")
    color = cmap(label_index / max(n_labels, 1))
    return f"rgb({color[0]*255:.0f},{color[1]*255:.0f},{color[2]*255:.0f})"


def matched_labels_for_lag(labels_ref, trial_id_ref, time_id_ref, trial_id_other, time_id_other, lag):
    other_keys = {(int(trial), int(time)) for trial, time in zip(trial_id_other, time_id_other)}
    matched_labels = []
    for label, trial, time in zip(labels_ref, trial_id_ref, time_id_ref):
        if (int(trial), int(time) + int(lag)) in other_keys:
            matched_labels.append(label)
    return np.asarray(matched_labels)


def plot_cross_subject_alignment_2d(
        subject_1_embedding,
        subject_2_aligned_embedding,
        labels,
        output_folder,
        name,
        title,
        dims=(0, 1)):
    labels = np.asarray(labels).reshape(-1)
    os.makedirs(output_folder, exist_ok=True)
    fig = go.Figure()
    unique_labels = np.unique(labels)

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        color = condition_color(idx, len(unique_labels))
        fig.add_trace(go.Scatter(
            x=subject_1_embedding[mask, dims[0]],
            y=subject_1_embedding[mask, dims[1]],
            mode="markers",
            marker=dict(color=color, size=6, symbol="circle", opacity=0.75),
            name=f"s1 cond {label}",
        ))
        fig.add_trace(go.Scatter(
            x=subject_2_aligned_embedding[mask, dims[0]],
            y=subject_2_aligned_embedding[mask, dims[1]],
            mode="markers",
            marker=dict(color=color, size=7, symbol="x", opacity=0.85),
            name=f"s2 aligned cond {label}",
        ))

    fig.update_layout(
        title=title,
        xaxis=dict(title=f"dim {dims[0] + 1}", scaleanchor="y", scaleratio=1),
        yaxis=dict(title=f"dim {dims[1] + 1}"),
        legend=dict(title="Subject / condition"),
    )
    fig.write_html(os.path.join(output_folder, name))
    return fig


def plot_cross_subject_alignment_3d(
        subject_1_embedding,
        subject_2_aligned_embedding,
        labels,
        output_folder,
        name,
        title):
    labels = np.asarray(labels).reshape(-1)
    os.makedirs(output_folder, exist_ok=True)
    fig = go.Figure()
    unique_labels = np.unique(labels)

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        color = condition_color(idx, len(unique_labels))
        fig.add_trace(go.Scatter3d(
            x=subject_1_embedding[mask, 0],
            y=subject_1_embedding[mask, 1],
            z=subject_1_embedding[mask, 2],
            mode="markers",
            marker=dict(color=color, size=3, symbol="circle", opacity=0.65),
            name=f"s1 cond {label}",
        ))
        fig.add_trace(go.Scatter3d(
            x=subject_2_aligned_embedding[mask, 0],
            y=subject_2_aligned_embedding[mask, 1],
            z=subject_2_aligned_embedding[mask, 2],
            mode="markers",
            marker=dict(color=color, size=4, symbol="x", opacity=0.85),
            name=f"s2 aligned cond {label}",
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="dim 1"),
            yaxis=dict(title="dim 2"),
            zaxis=dict(title="dim 3"),
            aspectmode="cube",
        ),
        legend=dict(title="Subject / condition"),
    )
    fig.write_html(os.path.join(output_folder, name))
    return fig


def run_suite():
    config = {
        "seed": 164,
        "n_trials": 160,
        "trial_len": 100,
        "n_neurons": 80,
        "n_conditions": 8,
        "n_traj_k": 3,
        "phi": 0.4,
        "condition_mode": "circular",
        "condition_type": "balanced",
        "noise_scale": 0.15,
        "directional_scale": 3.0,
        "extra_scale": 0.051,
        "dt": 0.02,
        "nonlinearity": "softplus",
        "overdispersion": 0.25,
        "refractory_mean_bins": 2,
        "refractory_std_bins": 1,
        "burst_probability": 0.05,
        "burst_size_mean": 1.5,
        "burst_window_bins": 3,
        "subject_lags_bins": {"subject_1": 0, "subject_2": 2},
        "window_size": 10,
        "stride": 1,
        "time_mode": "absolute",
        "window_padding": "center",
        "pad_value": 0.0,
        "embedding_dim": 3,
        "pca_plot_components": 5,
        "batch_size": 256,
        "epochs": 1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "temperature": 0.2,
        "loss_mode": "soft_structured",
        "positive_time_offset": 10,
        "similarity_tau": 0.5,
        "time_weight": 0.5,
        "label_weight": 0.5,
        "similarity_specs": [
            {"key": "time_id", "geometry": "temporal", "weight": 0.5},
            {"key": "label", "geometry": "circular", "num_labels": 8, "weight": 0.5},
        ],
        "encoders": ["cnn", "transformer"],
        "metric_max_samples": 2500,
        "save_all_plots": False,
    }

    set_seed(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, C, task_state, subject_data = make_synthetic_shared_latent_data(config)

    results = []
    embeddings_by_subject = {}
    pca_full_by_subject = {}
    latent_by_subject = {}
    metadata_by_subject = {}
    for subject_name, data in subject_data.items():
        print(f"\nPreparing dataset for {subject_name}", flush=True)
        dataset, Z_target = make_window_dataset(data["X"], data["Z_task"], C, config)
        latent_by_subject[subject_name] = Z_target
        metadata_by_subject[subject_name] = {
            "label": dataset.labels_windows.numpy(),
            "trial_id": dataset.trial_id.numpy(),
            "time_id": dataset.time_id.numpy(),
            "global_time_id": dataset.global_time_id.numpy(),
        }

        print(f"Running PCA for {subject_name} on {len(dataset)} windows", flush=True)
        X_flat_windows = dataset.X_windows.numpy().reshape(len(dataset), -1)
        pca_full = PCA(n_components=config["pca_plot_components"]).fit_transform(X_flat_windows)
        pca_embedding = pca_full[:, :config["embedding_dim"]]
        pca_scores = evaluate_latent_recovery_sampled(
            pca_embedding,
            Z_target,
            max_samples=config["metric_max_samples"],
            seed=config["seed"],
        )
        results.append({
            "subject": subject_name,
            "method": "pca",
            "final_loss": np.nan,
            **pca_scores,
        })
        embeddings_by_subject[(subject_name, "pca")] = pca_embedding
        pca_full_by_subject[subject_name] = pca_full

        for encoder_name in config["encoders"]:
            print(f"Training {encoder_name} for {subject_name}", flush=True)
            embedding, losses, _ = train_encoder(encoder_name, dataset, config, device)
            print(f"Evaluating {encoder_name} for {subject_name}", flush=True)
            scores = evaluate_latent_recovery_sampled(
                embedding,
                Z_target,
                max_samples=config["metric_max_samples"],
                seed=config["seed"],
            )
            results.append({
                "subject": subject_name,
                "method": encoder_name,
                "final_loss": losses[-1],
                **scores,
            })
            embeddings_by_subject[(subject_name, encoder_name)] = embedding

    print("\n=== Latent Recovery Baseline Suite ===")
    header = f"{'subject':<10} {'method':<12} {'loss':>9} {'proc_r2':>9} {'rsa_s':>9} {'rsa_p':>9}"
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{row['subject']:<10} {row['method']:<12} "
            f"{row['final_loss']:>9.4f} {row['procrustes_r2']:>9.4f} "
            f"{row['rsa_spearman']:>9.4f} {row['rsa_pearson']:>9.4f}"
        )

    print("\n=== Cross-subject lag-aware alignment ===")
    lag_results = []
    cross_subject_aligned = {}
    for method in ["pca", *config["encoders"]]:
        emb_1 = embeddings_by_subject[("subject_1", method)]
        emb_2 = embeddings_by_subject[("subject_2", method)]
        meta_1 = metadata_by_subject["subject_1"]
        meta_2 = metadata_by_subject["subject_2"]
        best_lag, scores, aligned_pairs = lagged_alignment_by_trial_time(
            emb_1,
            emb_2,
            meta_1["trial_id"],
            meta_1["time_id"],
            meta_2["trial_id"],
            meta_2["time_id"],
            lags=range(-5, 6),
        )
        ref_points, other_points = aligned_pairs[best_lag]
        if len(ref_points) >= 3:
            other_aligned, _ = procrustes_align(other_points, ref_points)
            matched_labels = matched_labels_for_lag(
                meta_1["label"],
                meta_1["trial_id"],
                meta_1["time_id"],
                meta_2["trial_id"],
                meta_2["time_id"],
                best_lag,
            )
            cross_subject_aligned[method] = {
                "subject_1": ref_points,
                "subject_2_aligned": other_aligned,
                "labels": matched_labels,
                "best_lag": best_lag,
            }
        lag_results.append({
            "method": method,
            "best_lag": best_lag,
            "best_score": scores[best_lag],
            **{f"lag_{lag}": score for lag, score in scores.items()},
        })
        print(f"{method:<12} best_lag={best_lag:>3} best_score={scores[best_lag]:.4f}")

    output_dir = PROJECT_ROOT / "outputs" / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    for subject_name, Z_target in (latent_by_subject.items() if config["save_all_plots"] else []):
        metadata = metadata_by_subject[subject_name]
        plot_embedding_sphere(
            Z_target,
            metadata["label"],
            figure_dir,
            f"{subject_name}_original_latent_sphere_scatter.html",
            title=f"{subject_name} - original latent Z by condition",
            show=False,
        )
        plot_condition_trajectories_sphere(
            Z_target,
            metadata["label"],
            metadata["trial_id"],
            metadata["time_id"],
            figure_dir,
            f"{subject_name}_original_latent_condition_trajectories.html",
            title=f"{subject_name} - original latent Z condition trajectories",
            show=False,
        )
        plot_embedding_2d(
            Z_target,
            metadata["label"],
            figure_dir,
            f"{subject_name}_original_latent_2d_scatter.html",
            title=f"{subject_name} - original latent Z 2D by condition",
            show=False,
        )
        plot_condition_trajectories_2d(
            Z_target,
            metadata["label"],
            metadata["trial_id"],
            metadata["time_id"],
            figure_dir,
            f"{subject_name}_original_latent_2d_condition_trajectories.html",
            title=f"{subject_name} - original latent Z 2D condition trajectories",
            show=False,
        )
        plot_condition_centroids_2d(
            Z_target,
            metadata["label"],
            figure_dir,
            f"{subject_name}_original_latent_2d_condition_centroids.html",
            title=f"{subject_name} - original latent Z 2D condition centroids",
            show=False,
        )

    for (subject_name, method), embedding in (embeddings_by_subject.items() if config["save_all_plots"] else []):
        metadata = metadata_by_subject[subject_name]
        plot_embedding_sphere(
            embedding,
            metadata["label"],
            figure_dir,
            f"{subject_name}_{method}_sphere_scatter.html",
            title=f"{subject_name} - {method} embedding by condition",
            show=False,
        )
        plot_condition_trajectories_sphere(
            embedding,
            metadata["label"],
            metadata["trial_id"],
            metadata["time_id"],
            figure_dir,
            f"{subject_name}_{method}_condition_trajectories.html",
            title=f"{subject_name} - {method} condition-averaged trajectories",
            show=False,
        )
        plot_embedding_2d(
            embedding,
            metadata["label"],
            figure_dir,
            f"{subject_name}_{method}_2d_scatter.html",
            title=f"{subject_name} - {method} 2D embedding by condition",
            show=False,
        )
        plot_condition_trajectories_2d(
            embedding,
            metadata["label"],
            metadata["trial_id"],
            metadata["time_id"],
            figure_dir,
            f"{subject_name}_{method}_2d_condition_trajectories.html",
            title=f"{subject_name} - {method} 2D condition-averaged trajectories",
            show=False,
        )
        plot_condition_centroids_2d(
            embedding,
            metadata["label"],
            figure_dir,
            f"{subject_name}_{method}_2d_condition_centroids.html",
            title=f"{subject_name} - {method} 2D condition centroids",
            show=False,
        )

    pca_plot_dims = [
        ((0, 1), "pc1_pc2"),
        ((0, 2), "pc1_pc3"),
        ((1, 2), "pc2_pc3"),
        ((2, 3), "pc3_pc4"),
        ((0, 3), "pc1_pc4"),
    ]
    for subject_name, pca_full in (pca_full_by_subject.items() if config["save_all_plots"] else []):
        metadata = metadata_by_subject[subject_name]
        for dims, suffix in pca_plot_dims:
            if max(dims) >= pca_full.shape[1]:
                continue
            plot_embedding_2d(
                pca_full,
                metadata["label"],
                figure_dir,
                f"{subject_name}_pca_{suffix}_scatter.html",
                title=f"{subject_name} - PCA {suffix.upper()} by condition",
                dims=dims,
                show=False,
            )
            plot_condition_trajectories_2d(
                pca_full,
                metadata["label"],
                metadata["trial_id"],
                metadata["time_id"],
                figure_dir,
                f"{subject_name}_pca_{suffix}_condition_trajectories.html",
                title=f"{subject_name} - PCA {suffix.upper()} condition trajectories",
                dims=dims,
                show=False,
            )
            plot_condition_centroids_2d(
                pca_full,
                metadata["label"],
                figure_dir,
                f"{subject_name}_pca_{suffix}_condition_centroids.html",
                title=f"{subject_name} - PCA {suffix.upper()} condition centroids",
                dims=dims,
                show=False,
            )

    for method, alignment in (cross_subject_aligned.items() if config["save_all_plots"] else []):
        best_lag = alignment["best_lag"]
        plot_cross_subject_alignment_2d(
            alignment["subject_1"],
            alignment["subject_2_aligned"],
            alignment["labels"],
            figure_dir,
            f"cross_subject_{method}_bestlag_{best_lag}_procrustes_2d.html",
            title=f"Cross-subject {method}: subject 2 aligned to subject 1, best lag {best_lag}",
        )
        plot_cross_subject_alignment_3d(
            alignment["subject_1"],
            alignment["subject_2_aligned"],
            alignment["labels"],
            figure_dir,
            f"cross_subject_{method}_bestlag_{best_lag}_procrustes_3d.html",
            title=f"Cross-subject {method}: subject 2 aligned to subject 1, best lag {best_lag}",
        )

    results_path = output_dir / "latent_recovery_results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    lag_path = output_dir / "cross_subject_lag_alignment.csv"
    with lag_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(lag_results[0].keys()))
        writer.writeheader()
        writer.writerows(lag_results)

    mat_paths = save_embedding_mat_files(
        embeddings_by_subject,
        latent_by_subject,
        metadata_by_subject,
        subject_data,
        C,
        task_state,
        results,
        lag_results,
        config,
        output_dir,
    )

    print(f"\nSaved: {results_path}")
    print(f"Saved: {lag_path}")
    print("Saved MATLAB embedding files:")
    for path in mat_paths:
        print(f"- {path}")
    print(f"Saved figures in: {figure_dir}")


if __name__ == "__main__":
    run_suite()

# %%
