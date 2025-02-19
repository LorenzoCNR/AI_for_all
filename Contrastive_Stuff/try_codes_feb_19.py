# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 20:05:35 2025

@author: loren
"""



import numpy as np
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpld3
#fig = plt.figure(figsize=(8, 6))
n_traj=8
trial_length
output_folder = default_output_dir
#os.makedirs(output_folder, exist_ok=True)


# Creazione della sfera unitaria
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))

# Crea la superficie della sfera
fig = go.Figure()

fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale="Blues", opacity=0.1, 
                         showscale=False))


# Aggiunta dei marker "Start" e "End" come PRIMI elementi nella legenda
fig.add_trace(go.Scatter3d(
    x=[None], y=[None], z=[None],
    mode="markers",
    marker=dict(color="red", size=5, symbol="circle"),
    name="Start"
))

fig.add_trace(go.Scatter3d(
    x=[None], y=[None], z=[None],
    mode="markers",
    marker=dict(color="blue", size=5, symbol="x"),
    name="End"
))

# Loop sulle traiettorie con i colori di Matplotlib
for i in range(n_traj):
    direction_trial = (l_dir == i)
    trial_avg = z_[direction_trial, :].reshape(-1, trial_length - ww, 3).mean(axis=0)
    trial_avg_normed = trial_avg / np.linalg.norm(trial_avg, axis=1)[:, None]

    # Colore della traiettoria con Matplotlib colormap
    color = plt.cm.hsv(i / n_traj)  # Stesso colore di Matplotlib
    hex_color = f'rgb({color[0]*255},{color[1]*255},{color[2]*255})'  # Convertito in RGB per Plotly

    # Aggiunta della traiettoria
    fig.add_trace(go.Scatter3d(
        x=trial_avg_normed[:, 0],
        y=trial_avg_normed[:, 1],
        z=trial_avg_normed[:, 2],
        mode="lines",
        line=dict(color=hex_color, width=3),
        name=f"Dir {i}"  # Etichetta direzione
    ))

    # Aggiunta dei marker Start e End per OGNI traiettoria
    fig.add_trace(go.Scatter3d(
        x=[trial_avg_normed[0, 0]],
        y=[trial_avg_normed[0, 1]],
        z=[trial_avg_normed[0, 2]],
        mode="markers",
        marker=dict(color="red", size=5, symbol="circle"),
        showlegend=False  # Non aggiunge duplicati in legenda
    ))

    fig.add_trace(go.Scatter3d(
        x=[trial_avg_normed[-1, 0]],
        y=[trial_avg_normed[-1, 1]],
        z=[trial_avg_normed[-1, 2]],
        mode="markers",
        marker=dict(color="blue", size=5, symbol="x"),
        showlegend=False  # Non aggiunge duplicati in legenda
    ))

# Personalizzazione della griglia e della legenda
fig.update_layout(
    title="direction-averaged embedding",
    scene=dict(
        xaxis=dict(showgrid=True, gridcolor="gray"),
        yaxis=dict(showgrid=True, gridcolor="gray"),
        zaxis=dict(showgrid=True, gridcolor="gray"),
    ),
    legend=dict(x=1.1, y=1, font=dict(size=10)),
)

# Salvataggio del file HTML e PNG
#output_folder = "output_plots"
#os.makedirs(output_folder, exist_ok=True)
output_html = os.path.join(output_folder, "plot_interattivo.html")
output_png = os.path.join(output_folder, "plot_interattivo.png")

fig.write_html(output_html)
#fig.write_image(output_png, scale=2)
fig.show()