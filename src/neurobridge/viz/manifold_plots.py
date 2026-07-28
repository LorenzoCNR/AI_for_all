"""
Plotting utilities for Neuro_Bridge (interactive 3D embeddings).

Funzioni:
- plot_direction_averaged_embedding: traiettorie medie dell’embedding normalizzate
  su sfera unitaria con Plotly, rispettando l’ordine originale delle label
  e mostrando (opzionale) la mappatura label_originale → label_usata_in_training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Dict
import os

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def _unit_normalize_rows(X, eps=1e-8):
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def _condition_color(label_index, n_labels):
    """
    Return a stable color for circular task labels.

    `plt.get_cmap("hsv", n)` samples both endpoints. With eight labels this
    makes the first and last condition nearly identical, which is misleading
    for inspection. Sampling manually with endpoint=False keeps the circular
    structure without visually collapsing condition 1 and condition 8.
    """
    cmap = plt.get_cmap("hsv")
    color = cmap(label_index / max(n_labels, 1))
    return f"rgb({color[0]*255:.0f},{color[1]*255:.0f},{color[2]*255:.0f})"


def _make_unit_sphere_trace(opacity=0.08):
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    return go.Surface(
        x=x,
        y=y,
        z=z,
        colorscale="Blues",
        opacity=opacity,
        showscale=False,
        name="unit sphere",
    )


def plot_embedding_sphere(
        embedding,
        labels,
        output_folder,
        name,
        title=None,
        normalize=True,
        marker_size=3,
        show=False,
        write_html=True):
    """
    Plot a 3D embedding on a unit sphere, colored by direction/condition labels.

    This is the windowed-embedding counterpart of
    `plot_direction_averaged_embedding`. It does not assume fixed trial reshape;
    it simply visualizes one point per window.
    """
    embedding = np.asarray(embedding)
    labels = np.asarray(labels).reshape(-1)

    if embedding.ndim != 2 or embedding.shape[1] != 3:
        raise ValueError("embedding must have shape (n_samples, 3)")
    if labels.shape[0] != embedding.shape[0]:
        raise ValueError("labels must have one value per embedding sample")

    if normalize:
        embedding = _unit_normalize_rows(embedding)

    os.makedirs(output_folder, exist_ok=True)
    fig = go.Figure()
    fig.add_trace(_make_unit_sphere_trace())

    unique_labels = np.unique(labels)

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        hex_color = _condition_color(idx, len(unique_labels))
        fig.add_trace(go.Scatter3d(
            x=embedding[mask, 0],
            y=embedding[mask, 1],
            z=embedding[mask, 2],
            mode="markers",
            marker=dict(color=hex_color, size=marker_size, opacity=0.75),
            name=f"cond {label}",
        ))

    fig.update_layout(
        title=title or name,
        scene=dict(
            xaxis=dict(title="z1"),
            yaxis=dict(title="z2"),
            zaxis=dict(title="z3"),
            aspectmode="cube",
        ),
        legend=dict(title="Condition"),
    )

    output_html = os.path.join(output_folder, name)
    if write_html:
        fig.write_html(output_html)
    if show:
        fig.show(renderer="browser")
    return fig


def plot_condition_trajectories_sphere(
        embedding,
        labels,
        trial_id,
        time_id,
        output_folder,
        name,
        title=None,
        normalize=True,
        show=False,
        write_html=True):
    """
    Plot condition-averaged trajectories on the unit sphere.

    For each condition and time value, the function averages embeddings across
    trials, then draws the resulting trajectory.
    """
    embedding = np.asarray(embedding)
    labels = np.asarray(labels).reshape(-1)
    trial_id = np.asarray(trial_id).reshape(-1)
    time_id = np.asarray(time_id).reshape(-1)

    if embedding.ndim != 2 or embedding.shape[1] != 3:
        raise ValueError("embedding must have shape (n_samples, 3)")
    n_samples = embedding.shape[0]
    if labels.shape[0] != n_samples or trial_id.shape[0] != n_samples or time_id.shape[0] != n_samples:
        raise ValueError("labels, trial_id, and time_id must match embedding length")

    if normalize:
        embedding = _unit_normalize_rows(embedding)

    os.makedirs(output_folder, exist_ok=True)
    fig = go.Figure()
    fig.add_trace(_make_unit_sphere_trace())

    unique_labels = np.unique(labels)
    unique_times = np.unique(time_id)

    for idx, label in enumerate(unique_labels):
        trajectory = []
        for time_value in unique_times:
            mask = (labels == label) & (time_id == time_value)
            if np.any(mask):
                trajectory.append(embedding[mask].mean(axis=0))

        if len(trajectory) < 2:
            continue

        trajectory = np.asarray(trajectory)
        if normalize:
            trajectory = _unit_normalize_rows(trajectory)

        hex_color = _condition_color(idx, len(unique_labels))
        fig.add_trace(go.Scatter3d(
            x=trajectory[:, 0],
            y=trajectory[:, 1],
            z=trajectory[:, 2],
            mode="lines+markers",
            line=dict(color=hex_color, width=4),
            marker=dict(size=3, color=hex_color),
            name=f"cond {label}",
        ))
        fig.add_trace(go.Scatter3d(
            x=[trajectory[0, 0]],
            y=[trajectory[0, 1]],
            z=[trajectory[0, 2]],
            mode="markers",
            marker=dict(size=5, color=hex_color, symbol="circle"),
            showlegend=False,
        ))
        fig.add_trace(go.Scatter3d(
            x=[trajectory[-1, 0]],
            y=[trajectory[-1, 1]],
            z=[trajectory[-1, 2]],
            mode="markers",
            marker=dict(size=6, color="black", symbol="x"),
            showlegend=False,
        ))

    fig.update_layout(
        title=title or name,
        scene=dict(
            xaxis=dict(title="z1"),
            yaxis=dict(title="z2"),
            zaxis=dict(title="z3"),
            aspectmode="cube",
        ),
        legend=dict(title="Condition"),
    )

    output_html = os.path.join(output_folder, name)
    if write_html:
        fig.write_html(output_html)
    if show:
        fig.show(renderer="browser")
    return fig


def plot_embedding_2d(
        embedding,
        labels,
        output_folder,
        name,
        title=None,
        dims=(0, 1),
        marker_size=4,
        show=False,
        write_html=True):
    """
    Plot a 2D view of an embedding, colored by condition labels.

    This is useful for the original latent variables and for PCA, where forcing
    points onto a sphere can hide the geometry we actually want to inspect.
    """
    embedding = np.asarray(embedding)
    labels = np.asarray(labels).reshape(-1)

    if embedding.ndim != 2:
        raise ValueError("embedding must be a 2D array")
    if labels.shape[0] != embedding.shape[0]:
        raise ValueError("labels must have one value per embedding sample")
    if max(dims) >= embedding.shape[1]:
        raise ValueError("requested dims exceed embedding dimensionality")

    os.makedirs(output_folder, exist_ok=True)
    fig = go.Figure()

    unique_labels = np.unique(labels)
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        hex_color = _condition_color(idx, len(unique_labels))
        fig.add_trace(go.Scatter(
            x=embedding[mask, dims[0]],
            y=embedding[mask, dims[1]],
            mode="markers",
            marker=dict(color=hex_color, size=marker_size, opacity=0.75),
            name=f"cond {label}",
        ))

    fig.update_layout(
        title=title or name,
        xaxis=dict(title=f"dim {dims[0] + 1}", scaleanchor="y", scaleratio=1),
        yaxis=dict(title=f"dim {dims[1] + 1}"),
        legend=dict(title="Condition"),
    )

    output_html = os.path.join(output_folder, name)
    if write_html:
        fig.write_html(output_html)
    if show:
        fig.show(renderer="browser")
    return fig


def plot_condition_trajectories_2d(
        embedding,
        labels,
        trial_id,
        time_id,
        output_folder,
        name,
        title=None,
        dims=(0, 1),
        show=False,
        write_html=True):
    """
    Plot condition-averaged trajectories in a 2D embedding view.
    """
    embedding = np.asarray(embedding)
    labels = np.asarray(labels).reshape(-1)
    trial_id = np.asarray(trial_id).reshape(-1)
    time_id = np.asarray(time_id).reshape(-1)

    if embedding.ndim != 2:
        raise ValueError("embedding must be a 2D array")
    n_samples = embedding.shape[0]
    if labels.shape[0] != n_samples or trial_id.shape[0] != n_samples or time_id.shape[0] != n_samples:
        raise ValueError("labels, trial_id, and time_id must match embedding length")
    if max(dims) >= embedding.shape[1]:
        raise ValueError("requested dims exceed embedding dimensionality")

    os.makedirs(output_folder, exist_ok=True)
    fig = go.Figure()

    unique_labels = np.unique(labels)
    unique_times = np.unique(time_id)

    for idx, label in enumerate(unique_labels):
        trajectory = []
        for time_value in unique_times:
            mask = (labels == label) & (time_id == time_value)
            if np.any(mask):
                trajectory.append(embedding[mask].mean(axis=0))

        if len(trajectory) < 2:
            continue

        trajectory = np.asarray(trajectory)
        hex_color = _condition_color(idx, len(unique_labels))
        fig.add_trace(go.Scatter(
            x=trajectory[:, dims[0]],
            y=trajectory[:, dims[1]],
            mode="lines+markers",
            line=dict(color=hex_color, width=3),
            marker=dict(size=4, color=hex_color),
            name=f"cond {label}",
        ))
        fig.add_trace(go.Scatter(
            x=[trajectory[0, dims[0]]],
            y=[trajectory[0, dims[1]]],
            mode="markers",
            marker=dict(size=8, color=hex_color, symbol="circle"),
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[trajectory[-1, dims[0]]],
            y=[trajectory[-1, dims[1]]],
            mode="markers",
            marker=dict(size=9, color="black", symbol="x"),
            showlegend=False,
        ))

    fig.update_layout(
        title=title or name,
        xaxis=dict(title=f"dim {dims[0] + 1}", scaleanchor="y", scaleratio=1),
        yaxis=dict(title=f"dim {dims[1] + 1}"),
        legend=dict(title="Condition"),
    )

    output_html = os.path.join(output_folder, name)
    if write_html:
        fig.write_html(output_html)
    if show:
        fig.show(renderer="browser")
    return fig


def plot_condition_centroids_2d(
        embedding,
        labels,
        output_folder,
        name,
        title=None,
        dims=(0, 1),
        show_dispersion=True,
        show=False,
        write_html=True):
    """
    Plot one centroid per condition with optional dispersion circles.

    The centroid summarizes where each condition lives in the embedding.
    The dispersion circle uses the root mean squared distance from points in
    the condition to their centroid, projected on the requested two dimensions.
    """
    embedding = np.asarray(embedding)
    labels = np.asarray(labels).reshape(-1)

    if embedding.ndim != 2:
        raise ValueError("embedding must be a 2D array")
    if labels.shape[0] != embedding.shape[0]:
        raise ValueError("labels must have one value per embedding sample")
    if max(dims) >= embedding.shape[1]:
        raise ValueError("requested dims exceed embedding dimensionality")

    os.makedirs(output_folder, exist_ok=True)
    fig = go.Figure()

    unique_labels = np.unique(labels)
    centroid_x = []
    centroid_y = []
    centroid_text = []

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        points = embedding[mask][:, dims]
        centroid = points.mean(axis=0)
        centered = points - centroid
        dispersion = float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1))))
        hex_color = _condition_color(idx, len(unique_labels))

        centroid_x.append(centroid[0])
        centroid_y.append(centroid[1])
        centroid_text.append(str(label))

        if show_dispersion and dispersion > 0:
            theta = np.linspace(0, 2 * np.pi, 80)
            fig.add_trace(go.Scatter(
                x=centroid[0] + dispersion * np.cos(theta),
                y=centroid[1] + dispersion * np.sin(theta),
                mode="lines",
                line=dict(color=hex_color, width=1, dash="dot"),
                opacity=0.45,
                showlegend=False,
            ))

        fig.add_trace(go.Scatter(
            x=[centroid[0]],
            y=[centroid[1]],
            mode="markers+text",
            marker=dict(color=hex_color, size=14, line=dict(color="black", width=1)),
            text=[f"{label}"],
            textposition="top center",
            name=f"cond {label}",
        ))

    if len(centroid_x) > 2:
        fig.add_trace(go.Scatter(
            x=[*centroid_x, centroid_x[0]],
            y=[*centroid_y, centroid_y[0]],
            mode="lines",
            line=dict(color="black", width=1),
            name="centroid cycle",
        ))

    fig.update_layout(
        title=title or name,
        xaxis=dict(title=f"dim {dims[0] + 1}", scaleanchor="y", scaleratio=1),
        yaxis=dict(title=f"dim {dims[1] + 1}"),
        legend=dict(title="Condition"),
    )

    output_html = os.path.join(output_folder, name)
    if write_html:
        fig.write_html(output_html)
    if show:
        fig.show(renderer="browser")
    return fig


def plot_direction_averaged_embedding(z_, l_dir_, original_label_order, c_s,
                                      output_folder, name, trial_length=None,
                                      quiescent_length=None,
                                      constant_length=True, ww=10,
                                      label_swap_info=None):
    """
      Plot averaged neural embedding trajectories normalized on a unit sphere,
      preserving the original label order and optionally displaying a mapping
      between modified and original labels in the legend.

      Args:
          z_: ndarray - Neural embedding (time x dim)
          l_dir_: ndarray - Labels used during training (possibly remapped)
          original_label_order: list[int] - Order in which to plot original directions
          c_s: str - Color used for "start" markers
          output_folder: str - Path where output will be saved
          name: str - Name of the HTML file to export
          trial_length: int - Length of each trial (required for reshaping)
          constant_length: bool - If True, assumes trial lengths are uniform
          ww: int - Model receptive window size to subtract
          label_swap_info: dict - Mapping {original_label: label_used_in_training}
    """
      #  Check inputs
    if trial_length is None:
        raise ValueError("trial_length must be provided.")
    if quiescent_length is None:
        raise ValueError("quiescent_length must be provided.")
    movement_length = trial_length - quiescent_length


    # create unit sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    ### coordinate x y z della sferea
    x = np.outer(np.cos(u), np.sin(v))    
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    fig = go.Figure()
    ### creo un oggetto plotly vuoto
    ##♀ aggiungo sfera trasparente
    fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale="Blues", opacity=0.1, 
                             showscale=False))
    
    
    #   "Start" and "End" markers (legend)
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode="markers",
        marker=dict(color=c_s, size=5, symbol="circle"),
        name="Start"
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode="markers",
        marker=dict(color="black", size=5, symbol="x"),
        name="End"
    ))
    # Unique directions 
    unique_dirs = np.unique(l_dir_)
    ### Trova tutte le direzioni presenti nei dati (l_dir_).
    n_colors = len(unique_dirs)

    # Setup colormap
    #Conta quante direzioni distinte ci sono → quanti colori servono.
    cmap = plt.get_cmap('hsv', n_colors)
    ## Normalizzatore da usare per mappare gli indici ai colori.
    norm = mcolors.Normalize(vmin=0, vmax=n_colors - 1)

    # Loop over trajectories (labels)
    for idx, original_label in enumerate(original_label_order):

    # Map the original label to the actual one used during training
        used_label = label_swap_info.get(original_label, original_label) if label_swap_info else original_label
        
        ###verifica
        # Selezione indici temporali in cui l'etichetta è quella data...quindi con mask prendo tutti i 
        # segmenti di una traiettoria associati a label i-esima
        mask = (l_dir_ == original_label)
        print(mask.shape)
        if not np.any(mask):
            print(f" No data found for original label {original_label} (used {used_label})")
            continue
        ### Calcolo punti per trial:
        # quanti timepoint per trial aspettarsi, differenziando tra quiete e movimento.
        if original_label == 0:
            points_per_trial = quiescent_length - ww
        else:
            points_per_trial = movement_length - ww
        try:
            ###Raggruppa tutti i segmenti con la stessa etichetta in array 3D:
            #(n_trial, points_per_trial, 3), poi fa la media lungo gli n_trial.
            trial_avg = z_[mask].reshape(-1, points_per_trial, 3).mean(axis=0)
        except ValueError as e:
            print(f"Errore di reshape per label {original_label}: {e}")
            continue
# Normalize each vector (aveg point x,y,z) (to lie on the unit sphere)

        trial_avg_normed = trial_avg / np.linalg.norm(trial_avg, axis=1, keepdims=True)


       
        print("Trial average shape:", trial_avg.shape)
        #trial_avg -= trial_avg.mean(axis=0)  # Sottraggo la media di ogni coordinata
        #trial_avg_normed = trial_avg/np.linalg.norm(trial_avg, axis=1)[:,None]
    #     #trial_avg /= np.linalg.norm(trial_avg, axis=2, keepdims=True)  # Normalizzazione
    #    # trial_avg_normed = trial_avg.mean(axis=0) 
    #     #trial_avg = z_[direction_trial, :].reshape(-1, trial_length - ww, 3).mean(axis=0)
    #     #trial_avg_normed = trial_avg / np.linalg.norm(trial_avg, axis=1)[:, None]
    
        # Colore della traiettoria con Matplotlib colormap
        #color =cmap(idx)   # Stesso colore di Matplotlib
        
        
        # Map color to the current trajectory
        if original_label == 0:
           hex_color = 'lightgray'
        else:
            color = cmap(norm(idx))
            hex_color = f'rgb({color[0]*255:.0f},{color[1]*255:.0f},{color[2]*255:.0f})'

                # Construct label for the legend
        if label_swap_info and original_label in label_swap_info:
            label_display = f"{original_label} → {label_swap_info[original_label]}"
        else:
            label_display = f"{original_label}"
        
                
        # Aggiunta della traiettoria
        fig.add_trace(go.Scatter3d(
            x=trial_avg_normed[:, 0],
            y=trial_avg_normed[:, 1],
            z=trial_avg_normed[:, 2],
            mode="lines",
            line=dict(color=hex_color, width=3),
            # direction label
            name=label_display

        ))
    
        # Aggiunta dei marker Start e End per OGNI traiettoria
        fig.add_trace(go.Scatter3d(
            x=[trial_avg_normed[0, 0]],
            y=[trial_avg_normed[0, 1]],
            z=[trial_avg_normed[0, 2]],
            mode="markers+text",
            marker=dict(color=c_s, size=3, symbol="circle"),
            text=[f"s {original_label}"],
            textposition="top right",
            showlegend=False   
        ))
    
        fig.add_trace(go.Scatter3d(
            x=[trial_avg_normed[-1, 0]],
            y=[trial_avg_normed[-1, 1]],
            z=[trial_avg_normed[-1, 2]],
            mode="markers+text",
            marker=dict(color="black", size=3, symbol="x"),
            text=[f"e {original_label}"],
            textposition="top right",
            showlegend=False  
        ))
    
    # Griglia e legenda
    fig.update_layout(
        title=dict(
            ## chekc the name
        text=name,  
        x=0.5,  #
        xanchor='center',  
    ),
        scene=dict(
            xaxis=dict(showgrid=True, gridcolor="gray", title='x1'),
            yaxis=dict(showgrid=True, gridcolor="gray", title='x2'),
            zaxis=dict(showgrid=True, gridcolor="gray", title='x3'),
        ),
        legend=dict(x=1.1, y=1, font=dict(size=10), title ='Direction of Movement'),
    )
    
    # Salvataggio del file HTML e PNG
    #output_folder = "output_plots"
    #os.makedirs(output_folder, exist_ok=True)
    output_html = os.path.join(output_folder, name)
    #output_png = os.path.join(output_folder, "plot_interattivo.png")
    
    fig.write_html(output_html)
    #fig.write_image(output_png, scale=2)
    fig.show(renderer="browser")

 
################################ Function to plot multiple maifolds G%B RATS
def plot_datasets_in_groups(dataset_dict, label, group_size,title=None):
    """
    Plot multiple 3D embeddings (datasets) side by side in groups.

    Each dataset is visualized as a 3D scatter plot.
    Points are colored according to a provided label matrix, typically
    representing left/right conditions or trial categories.

    IMPORTANT:
     - `label` can be EITHER:
        (a) a single ndarray (N x M) applied to all datasets (backward-compatible), OR
        (b) a dict {dataset_name: ndarray (Ni x M)} with per-dataset labels.
     - Column convention is unchanged:
        label[:, 2] == 1 -> "left"
        label[:, 1] == 1 -> "right"
        label[:, 0] provides the numeric values for coloring.
    - No filenames are saved; figures are shown and also returned.

    Args:
        dataset_dict (dict): {dataset_name: embedding ndarray (Ni x 3)}.
        label (ndarray or dict): global label matrix or per-dataset label dict.
        group_size (int): number of datasets per figure (row of subplots).

    Returns:
        list: matplotlib Figure objects (one per group).
    """

    # Get dataset names and number of datasets
    dataset_names = list(dataset_dict.keys())
    num_datasets = len(dataset_names)
    figures = []

    # Iterate through datasets in groups of 'group_size'
    for i in range(0, num_datasets, group_size):
        actual_group_size = min(group_size, num_datasets - i)

        # Create subplot row (3D projection)
        fig, axs = plt.subplots(1, actual_group_size, figsize=(24, 6),
                                subplot_kw={'projection': '3d'})
        if actual_group_size == 1:
            axs = [axs]
        if title:
            fig.suptitle(title, fontsize=16)

        # Loop through the datasets in this group
        for j in range(actual_group_size):
            dataset_name = dataset_names[i + j]
            emb = dataset_dict[dataset_name]

            # Select correct label matrix (either from dict or shared)
            if isinstance(label, dict):
                if dataset_name not in label:
                    raise KeyError(f"Missing labels for dataset '{dataset_name}' in label dict.")
                lab = label[dataset_name]
            else:
                lab = label  # same label matrix for all

            # Check consistency
            if lab.shape[0] != emb.shape[0]:
                raise ValueError(
                    f"Label length mismatch for '{dataset_name}': "
                    f"labels={lab.shape[0]} vs emb={emb.shape[0]}"
                )
            if lab.ndim != 2 or lab.shape[1] < 3:
                raise ValueError(
                    f"Label matrix for '{dataset_name}' must be 2D with >= 3 columns."
                )


            # Example: assuming label[:,2] = left, label[:,1] = right
            idx_left = lab[:, 2] == 1
            idx_right = lab[:, 1] == 1

            # Plot left and right points with different colormaps
            scatter_left = axs[j].scatter(
                emb[idx_left, 0], emb[idx_left, 1], emb[idx_left, 2],
                c=lab[idx_left, 0], cmap="cool_r", s=0.5)
            scatter_right = axs[j].scatter(
                emb[idx_right, 0], emb[idx_right, 1], emb[idx_right, 2],
                c=lab[idx_right, 0], cmap="summer_r", s=0.5)

            axs[j].axis("off")
            axs[j].set_title(dataset_name, fontsize=12)

        # Add shared colorbars
        cbar_left = fig.colorbar(scatter_left, ax=axs, pad=0.02, fraction=0.02,
                                 location='bottom', shrink=0.5)
        cbar_left.set_label('Left')

        cbar_right = fig.colorbar(scatter_right, ax=axs, pad=0.02, fraction=0.02,
                                  location='bottom', shrink=0.5)
        cbar_right.set_label('Right')

        # Adjust colorbar positions for neat layout
        cbar_left.ax.set_position([0.24, 0.05, 0.35, 0.03])
        cbar_right.ax.set_position([0.41, 0.05, 0.35, 0.03])

        # Adjust subplot spacing and display
        plt.subplots_adjust(left=0.1, wspace=0.01)
        plt.show()

        figures.append(fig)

    return figures
