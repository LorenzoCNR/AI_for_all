from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score


def center_scale(X, eps=1e-8):
    X = np.asarray(X, dtype=float)
    X = X - X.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(X)
    if scale <= eps:
        return X
    return X / scale


def procrustes_align(source, target):
    """
    Align source to target using orthogonal Procrustes.

    source and target must have the same shape.
    """
    source = center_scale(source)
    target = center_scale(target)

    if source.shape != target.shape:
        raise ValueError("source and target must have the same shape")

    # Solve min_R ||source @ R - target||_F with R.T @ R = I.
    #
    # This is the same SVD solution used by orthogonal Procrustes:
    # source.T @ target = U S V.T, then R = U V.T.
    # Keeping the small decomposition explicit avoids a SciPy LAPACK
    # deadlock observed on some Windows scientific-Python environments.
    cross_covariance = source.T @ target
    left_vectors, _, right_vectors_t = np.linalg.svd(
        cross_covariance,
        full_matrices=False,
    )
    R = left_vectors @ right_vectors_t
    aligned = source @ R
    return aligned, R


def procrustes_r2(embedding, latent):
    embedding = np.asarray(embedding)
    latent = np.asarray(latent)

    if embedding.shape[1] != latent.shape[1]:
        n_components = latent.shape[1]
        embedding = PCA(n_components=n_components).fit_transform(embedding)

    aligned, _ = procrustes_align(embedding, latent)
    latent_scaled = center_scale(latent)
    return float(r2_score(latent_scaled, aligned, multioutput="uniform_average"))


def distance_geometry_correlation(embedding, latent, metric="euclidean", method="spearman"):
    embedding = np.asarray(embedding)
    latent = np.asarray(latent)

    D_embedding = squareform(pdist(embedding, metric=metric))
    D_latent = squareform(pdist(latent, metric=metric))

    idx = np.triu_indices_from(D_embedding, k=1)
    x = D_embedding[idx]
    y = D_latent[idx]

    if method == "spearman":
        corr, _ = spearmanr(x, y)
    elif method == "pearson":
        corr, _ = pearsonr(x, y)
    else:
        raise ValueError("method must be 'spearman' or 'pearson'")

    return float(corr)


def evaluate_latent_recovery(embedding, latent):
    return {
        "procrustes_r2": procrustes_r2(embedding, latent),
        "rsa_spearman": distance_geometry_correlation(embedding, latent, method="spearman"),
        "rsa_pearson": distance_geometry_correlation(embedding, latent, method="pearson"),
    }


def lagged_alignment_scores(embedding_ref, embedding_other, lags):
    """
    Evaluate Procrustes R2 between two embeddings over candidate temporal lags.
    Positive lag compares ref[t] to other[t + lag].
    """
    scores = {}
    embedding_ref = np.asarray(embedding_ref)
    embedding_other = np.asarray(embedding_other)

    for lag in lags:
        lag = int(lag)
        if lag > 0:
            ref = embedding_ref[:-lag]
            other = embedding_other[lag:]
        elif lag < 0:
            ref = embedding_ref[-lag:]
            other = embedding_other[:lag]
        else:
            ref = embedding_ref
            other = embedding_other

        if len(ref) < 3:
            scores[lag] = np.nan
            continue

        scores[lag] = procrustes_r2(other, ref)

    best_lag = max(scores, key=lambda key: -np.inf if np.isnan(scores[key]) else scores[key])
    return best_lag, scores


def lagged_alignment_by_trial_time(
        embedding_ref,
        embedding_other,
        trial_id_ref,
        time_id_ref,
        trial_id_other,
        time_id_other,
        lags):
    """
    Evaluate Procrustes R2 over candidate lags using trial/time metadata.

    Positive lag compares ref(trial, time) with other(trial, time + lag).
    This is safer than shifting the flattened array because windows are grouped
    by trial and should not wrap across trial boundaries.
    """
    embedding_ref = np.asarray(embedding_ref)
    embedding_other = np.asarray(embedding_other)
    trial_id_ref = np.asarray(trial_id_ref).reshape(-1)
    time_id_ref = np.asarray(time_id_ref).reshape(-1)
    trial_id_other = np.asarray(trial_id_other).reshape(-1)
    time_id_other = np.asarray(time_id_other).reshape(-1)

    if len(embedding_ref) != len(trial_id_ref) or len(embedding_ref) != len(time_id_ref):
        raise ValueError("reference metadata must match reference embedding length")
    if len(embedding_other) != len(trial_id_other) or len(embedding_other) != len(time_id_other):
        raise ValueError("other metadata must match other embedding length")

    other_lookup = {
        (int(trial), int(time)): embedding_other[idx]
        for idx, (trial, time) in enumerate(zip(trial_id_other, time_id_other))
    }

    scores = {}
    aligned_pairs = {}
    for lag in lags:
        ref_points = []
        other_points = []
        for idx, (trial, time) in enumerate(zip(trial_id_ref, time_id_ref)):
            key = (int(trial), int(time) + int(lag))
            if key not in other_lookup:
                continue
            ref_points.append(embedding_ref[idx])
            other_points.append(other_lookup[key])

        if len(ref_points) < 3:
            scores[int(lag)] = np.nan
            aligned_pairs[int(lag)] = (np.empty((0, embedding_ref.shape[1])), np.empty((0, embedding_other.shape[1])))
            continue

        ref_points = np.asarray(ref_points)
        other_points = np.asarray(other_points)
        scores[int(lag)] = procrustes_r2(other_points, ref_points)
        aligned_pairs[int(lag)] = (ref_points, other_points)

    best_lag = max(scores, key=lambda key: -np.inf if np.isnan(scores[key]) else scores[key])
    return best_lag, scores, aligned_pairs
