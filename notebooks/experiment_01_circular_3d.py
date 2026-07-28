# %% [markdown]
# # Circular task: essential 3D latent space
#
# Latent coordinates: position X, position Y, and movement progress.

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
    name="experiment_01_circular_3d",
    condition_mode="circular",
    latent_dim=3,
    cnn_epochs=10,
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
print("PCA:", results["pca_embedding"].shape)
print("CNN:", results["cnn_embedding"].shape)
print("CNN final loss:", results["cnn_losses"][-1])
print("Metrics:", results["metrics"])
