import unittest

import numpy as np

from src.neurobridge.eval.representation import (
    distance_geometry_correlation,
    evaluate_latent_recovery,
    lagged_alignment_scores,
    procrustes_align,
    procrustes_r2,
)


class TestRepresentationEval(unittest.TestCase):
    def test_procrustes_handles_rotation(self):
        theta = np.pi / 4
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ])
        X = np.random.randn(50, 2)
        Y = X @ R
        aligned, _ = procrustes_align(Y, X)
        self.assertEqual(aligned.shape, X.shape)
        self.assertGreater(procrustes_r2(Y, X), 0.95)

    def test_distance_geometry_correlation(self):
        X = np.random.randn(30, 3)
        corr = distance_geometry_correlation(X, X)
        self.assertGreater(corr, 0.99)

    def test_evaluate_latent_recovery_keys(self):
        X = np.random.randn(30, 3)
        scores = evaluate_latent_recovery(X, X)
        self.assertIn("procrustes_r2", scores)
        self.assertIn("rsa_spearman", scores)
        self.assertIn("rsa_pearson", scores)

    def test_lagged_alignment_scores(self):
        X = np.random.randn(40, 3)
        Y = np.vstack([X[:1], X[:-1]])
        best_lag, scores = lagged_alignment_scores(X, Y, lags=range(-2, 3))
        self.assertIn(best_lag, scores)


if __name__ == "__main__":
    unittest.main()
