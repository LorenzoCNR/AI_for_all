import numpy as np
import unittest

from src.neurobridge.sampling.similarity_ import dist_to_simi


class TestDistToSimi(unittest.TestCase):
    def test_expected_values_and_monotonicity(self):
        D_total = np.array([
            [0.0, 0.5, 1.0],
            [0.5, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ])

        W = dist_to_simi(D_total, tau=1.0)

        self.assertEqual(W.shape, D_total.shape)
        self.assertTrue(np.allclose(np.diag(W), 1.0))
        self.assertTrue(np.isclose(W[0, 1], np.exp(-0.5)))
        self.assertTrue(np.isclose(W[0, 2], np.exp(-1.0)))
        self.assertGreater(W[0, 0], W[0, 1])
        self.assertGreater(W[0, 1], W[0, 2])

    def test_requires_2d_matrix(self):
        with self.assertRaisesRegex(ValueError, "2D matrix"):
            dist_to_simi(np.array([0.0, 1.0]), tau=1.0)

    def test_requires_square_matrix(self):
        with self.assertRaisesRegex(ValueError, "square"):
            dist_to_simi(np.zeros((2, 3)), tau=1.0)

    def test_requires_positive_tau(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            dist_to_simi(np.eye(2), tau=0)

    def test_requires_non_negative_distances(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            dist_to_simi(np.array([[0.0, -1.0], [1.0, 0.0]]), tau=1.0)


if __name__ == "__main__":
    unittest.main()
