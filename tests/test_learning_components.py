# -*- coding: utf-8 -*-
"""Tests for datasets, temporal encoders, CEBRA blocks, losses, and sampling."""

import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.neurobridge.data.dataset import TemporalWindowDataset
from src.neurobridge.data.sim.builders import build_structured_B
from src.neurobridge.losses.infonce import (
    soft_contrastive_loss,
    supervised_infonce_loss,
    time_offset_infonce_loss,
)
from src.neurobridge.models.blocks import _MeanAndConv, _Norm, _Skip, Squeeze
from src.neurobridge.models.temporal_cnn import (
    SinusoidalPositionalEncoding,
    TemporalCNNEncoder,
    TemporalLSTMEncoder,
    TemporalMLPEncoder,
    TemporalTransformerEncoder,
)
from src.neurobridge.sampling.batch_similarity import (
    batch_structured_similarity,
    batch_structured_similarity_from_specs,
)
from src.neurobridge.sampling.f_windows import build_windows


class TestLearningComponents(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        self.X_windows = rng.poisson(0.2, size=(32, 10, 8)).astype(np.float32)
        self.time_id = np.linspace(0, 1, 32, dtype=np.float32)
        self.global_time_id = np.arange(32, dtype=np.int64)
        self.trial_id = np.repeat(np.arange(8), 4)
        self.labels = np.tile(np.arange(1, 9), 4)

    def test_dataset_and_dataloader_shapes(self) -> None:
        dataset = TemporalWindowDataset(
            self.X_windows,
            self.time_id,
            self.global_time_id,
            self.trial_id,
            self.labels,
        )
        self.assertEqual(dataset[0]["x"].shape, torch.Size([10, 8]))

        loader = DataLoader(dataset, batch_size=16)
        batch = next(iter(loader))
        self.assertEqual(batch["x"].shape, torch.Size([16, 10, 8]))
        self.assertEqual(batch["label"].shape, torch.Size([16]))

    def test_encoders_return_normalized_embeddings(self) -> None:
        torch.manual_seed(42)
        x = torch.randn(16, 10, 8)
        encoders = [
            TemporalCNNEncoder(n_features=8, embedding_dim=5),
            TemporalMLPEncoder(
                window_size=10,
                n_features=8,
                embedding_dim=5,
            ),
            TemporalLSTMEncoder(n_features=8, embedding_dim=5),
            TemporalTransformerEncoder(
                n_features=8,
                embedding_dim=5,
                model_dim=16,
                n_heads=4,
                n_layers=1,
                max_length=32,
            ),
        ]

        for encoder in encoders:
            encoder.eval()
            with torch.no_grad():
                z = encoder(x)
            self.assertEqual(z.shape, torch.Size([16, 5]))
            norms = torch.linalg.vector_norm(z, dim=-1)
            self.assertTrue(
                torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
            )

    def test_cnn_rejects_even_kernel_size(self) -> None:
        with self.assertRaises(ValueError):
            TemporalCNNEncoder(n_features=8, kernel_size=4)

    def test_encoder_rejects_wrong_feature_dimension(self) -> None:
        model = TemporalCNNEncoder(n_features=8)
        wrong_x = torch.randn(4, 10, 7)
        with self.assertRaises(ValueError):
            model(wrong_x)

    def test_mlp_rejects_wrong_window_size(self) -> None:
        model = TemporalMLPEncoder(window_size=10, n_features=8)
        wrong_x = torch.randn(4, 9, 8)
        with self.assertRaises(ValueError):
            model(wrong_x)

    def test_transformer_requires_divisible_model_dimension(self) -> None:
        with self.assertRaises(ValueError):
            TemporalTransformerEncoder(
                n_features=8,
                model_dim=30,
                n_heads=4,
            )

    def test_positional_encoding_preserves_shape_and_changes_values(self) -> None:
        encoding = SinusoidalPositionalEncoding(model_dim=8, max_length=16)
        x = torch.zeros(2, 10, 8)
        y = encoding(x)

        self.assertEqual(y.shape, x.shape)
        self.assertFalse(torch.allclose(y[:, 0], y[:, 1]))

    def test_positional_encoding_rejects_long_sequence(self) -> None:
        encoding = SinusoidalPositionalEncoding(model_dim=8, max_length=4)
        with self.assertRaises(ValueError):
            encoding(torch.zeros(2, 5, 8))

    def test_skip_preserves_expected_shape(self) -> None:
        block = _Skip(
            nn.Conv1d(8, 8, kernel_size=3, padding=1),
            nn.GELU(),
            crop=(1, 1),
        )
        x = torch.randn(4, 8, 20)
        y = block(x)
        self.assertEqual(y.shape, torch.Size([4, 8, 18]))

    def test_skip_rejects_mismatched_residual_shape(self) -> None:
        block = _Skip(
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            crop=(1, 1),
        )
        with self.assertRaises(ValueError):
            block(torch.randn(4, 8, 20))

    def test_squeeze_removes_only_singleton_third_dimension(self) -> None:
        block = Squeeze()
        x = torch.randn(4, 8, 1, 20)
        y = block(x)
        self.assertEqual(y.shape, torch.Size([4, 8, 20]))

        unchanged = torch.randn(4, 8, 3, 20)
        self.assertEqual(block(unchanged).shape, unchanged.shape)

    def test_norm_normalizes_channel_dimension(self) -> None:
        block = _Norm()
        x = torch.randn(4, 8, 20)
        y = block(x)
        norms = torch.linalg.vector_norm(y, dim=1)
        self.assertTrue(
            torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        )

    def test_norm_handles_zero_vectors(self) -> None:
        block = _Norm()
        y = block(torch.zeros(2, 4, 6))
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(torch.equal(y, torch.zeros_like(y)))

    def test_mean_and_conv_output_shape(self) -> None:
        block = _MeanAndConv(
            inp=8,
            output=16,
            kernel=4,
            stride=2,
        )
        x = torch.randn(4, 8, 20)
        y = block(x)

        expected_time = block.layer(x).shape[-1]
        self.assertEqual(y.shape, torch.Size([4, 24, expected_time]))

    def test_losses_are_finite_scalars(self) -> None:
        torch.manual_seed(42)
        z = torch.randn(16, 5, requires_grad=True)
        labels = torch.tensor(
            np.tile(np.arange(1, 9), 2),
            dtype=torch.long,
        )

        supervised_loss = supervised_infonce_loss(
            z,
            labels,
            temperature=0.2,
        )
        self.assertEqual(supervised_loss.ndim, 0)
        self.assertTrue(torch.isfinite(supervised_loss))

        similarity = torch.exp(-torch.rand(16, 16))
        soft_loss = soft_contrastive_loss(
            z,
            similarity,
            temperature=0.2,
        )
        self.assertEqual(soft_loss.ndim, 0)
        self.assertTrue(torch.isfinite(soft_loss))

    def test_supervised_loss_supports_backward(self) -> None:
        z = torch.randn(16, 5, requires_grad=True)
        labels = torch.tensor(np.tile(np.arange(4), 4), dtype=torch.long)
        loss = supervised_infonce_loss(z, labels, temperature=0.2)
        loss.backward()
        self.assertIsNotNone(z.grad)
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_batch_similarity_shape(self) -> None:
        dataset = TemporalWindowDataset(
            self.X_windows,
            self.time_id,
            self.global_time_id,
            self.trial_id,
            self.labels,
        )
        batch = next(iter(DataLoader(dataset, batch_size=16)))
        similarity = batch_structured_similarity(
            batch,
            tau=0.5,
            num_labels=8,
        )
        self.assertEqual(similarity.shape, torch.Size([16, 16]))
        self.assertTrue(torch.all(similarity > 0))
        self.assertTrue(torch.all(similarity <= 1))

    def test_vector_label_similarity_shape(self) -> None:
        position_labels = np.column_stack(
            [
                np.linspace(0, 1, 32),
                np.linspace(1, 0, 32),
            ]
        ).astype(np.float32)
        dataset = TemporalWindowDataset(
            self.X_windows,
            self.time_id,
            self.global_time_id,
            self.trial_id,
            position_labels,
        )
        batch = next(iter(DataLoader(dataset, batch_size=16)))
        specs = [
            {"key": "time_id", "geometry": "temporal", "weight": 0.5},
            {"key": "label", "geometry": "euclidean", "weight": 0.5},
        ]
        similarity = batch_structured_similarity_from_specs(
            batch,
            specs,
            tau=0.5,
        )
        self.assertEqual(batch["label"].shape, torch.Size([16, 2]))
        self.assertEqual(similarity.shape, torch.Size([16, 16]))
        self.assertTrue(torch.all(similarity > 0))
        self.assertTrue(torch.all(similarity <= 1))

    def test_time_offset_infonce_is_finite_scalar(self) -> None:
        z = torch.randn(16, 5)
        trial_id = torch.repeat_interleave(torch.arange(4), 4)
        time_id = torch.tile(torch.arange(4), (4,))
        loss = time_offset_infonce_loss(
            z,
            trial_id,
            time_id,
            offset=1,
            temperature=0.2,
        )
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

    def test_center_padding_preserves_trial_length(self) -> None:
        X = np.arange(10, dtype=np.float32).reshape(10, 1)
        labels = np.array([1, 2])
        (
            X_windows,
            time_id,
            global_time_id,
            trial_id,
            labels_windows,
        ) = build_windows(
            X,
            window_size=3,
            stride=1,
            labels=labels,
            trial_len=5,
            time_mode="absolute",
            padding="center",
            pad_value=0.0,
        )

        self.assertEqual(X_windows.shape, (10, 3, 1))
        self.assertTrue(np.array_equal(time_id[:5], np.arange(5)))
        self.assertTrue(np.array_equal(global_time_id, np.arange(10)))
        self.assertTrue(np.array_equal(trial_id, np.repeat([0, 1], 5)))
        self.assertTrue(np.array_equal(labels_windows, np.repeat(labels, 5)))
        self.assertTrue(
            np.array_equal(
                X_windows[0, :, 0],
                np.array([0, 0, 1], dtype=np.float32),
            )
        )
        self.assertTrue(
            np.array_equal(
                X_windows[4, :, 0],
                np.array([3, 4, 0], dtype=np.float32),
            )
        )

    def test_structured_B_circular_and_linear_shapes(self) -> None:
        B_circular = build_structured_B(
            3,
            16,
            list(range(8)),
            8,
            condition_mode="circular",
        )
        self.assertEqual(B_circular.shape, (3, 16))

        B_linear = build_structured_B(
            3,
            16,
            list(range(1, 11)),
            10,
            condition_mode="linear",
            random_state=24,
        )
        self.assertEqual(B_linear.shape, (3, 16))
        self.assertTrue(np.all(np.isfinite(B_linear)))

        B_linear_repeated = build_structured_B(
            3,
            16,
            list(range(1, 11)),
            10,
            condition_mode="linear",
            random_state=24,
        )
        self.assertTrue(np.array_equal(B_linear, B_linear_repeated))


if __name__ == "__main__":
    unittest.main()
