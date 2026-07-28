# %% [markdown]
# # Linear track: essential 2D latent space
#
# Latent coordinates: track position and movement direction. A subset of
# neurons has localized Gaussian place fields along the track.

# %%
from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "src" / "neurobridge").exists():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from neurobridge.experiments import (
    SyntheticTaskConfig,
    run_synthetic_task_experiment,
)

# %%
CONFIG = SyntheticTaskConfig(
    name="experiment_03_linear_2d",
    condition_mode="linear",
    latent_dim=2,
    place_fraction=0.25,
    place_width=0.10,
    cnn_epochs=30,
)
CONFIG

# %%
results = run_synthetic_task_experiment(
    CONFIG,
    project_root=PROJECT_ROOT,
)

# %%
print("Z:", results["Z"].shape)
print("X:", results["X"].shape)
print("Place neurons:", sum(results["neuron_types"] == "place"))
print("PCA:", results["pca_embedding"].shape)
print("CNN:", results["cnn_embedding"].shape)
print("Metrics:", results["metrics"])
