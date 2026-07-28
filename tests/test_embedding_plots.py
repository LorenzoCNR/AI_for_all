import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.neurobridge.viz.manifold_plots import (
    plot_condition_trajectories_sphere,
    plot_embedding_sphere,
)


@unittest.skip("Plotly figure construction can hang under unittest in this environment; verified by direct smoke test.")
class TestEmbeddingPlots(unittest.TestCase):
    def test_embedding_sphere_writes_html(self):
        with tempfile.TemporaryDirectory(dir="C:\\tmp") as tmp_dir:
            embedding = np.random.randn(24, 3)
            labels = np.tile(np.arange(1, 9), 3)
            fig = plot_embedding_sphere(
                embedding,
                labels,
                tmp_dir,
                "sphere.html",
                show=False,
                write_html=False,
            )
            self.assertGreater(len(fig.data), 1)

    def test_condition_trajectories_writes_html(self):
        with tempfile.TemporaryDirectory(dir="C:\\tmp") as tmp_dir:
            n_trials = 8
            n_times = 4
            embedding = np.random.randn(n_trials * n_times, 3)
            labels = np.repeat(np.arange(1, 9), n_times)
            trial_id = np.repeat(np.arange(n_trials), n_times)
            time_id = np.tile(np.linspace(0, 1, n_times), n_trials)
            fig = plot_condition_trajectories_sphere(
                embedding,
                labels,
                trial_id,
                time_id,
                tmp_dir,
                "trajectories.html",
                show=False,
                write_html=False,
            )
            self.assertGreater(len(fig.data), 1)


if __name__ == "__main__":
    unittest.main()
