import unittest

import numpy as np

from neurobridge.data.dataset import TemporalWindowDataset
from neurobridge.data.sim import build_linear_loading_and_place_fields
from neurobridge.experiments import (
    SyntheticTaskConfig,
)


class TestSyntheticTaskSuite(unittest.TestCase):
    def test_expected_minimum_latent_dimensions(self):
        SyntheticTaskConfig("circular", "circular", 3)
        SyntheticTaskConfig("linear", "linear", 2)

    def test_similarity_distance_requires_a_positive_weight(self):
        with self.assertRaisesRegex(
            ValueError,
            "At least one distance weight",
        ):
            SyntheticTaskConfig(
                "invalid",
                "linear",
                2,
                time_weight=0.0,
                label_weight=0.0,
            )

    def test_linear_spatial_fields_are_binned_and_identifiable(self):
        outbound = np.linspace(0.0, 1.0, 50)
        returning = np.linspace(1.0, 0.0, 50)
        position = np.tile(
            np.concatenate([outbound, returning]),
            (4, 1),
        )
        direction = np.tile(
            np.concatenate([np.ones(50), np.zeros(50)]),
            (4, 1),
        )
        B, neuron_types, centers, drive, metadata = (
            build_linear_loading_and_place_fields(
                k=2,
                n_neurons=80,
                position=position,
                direction=direction,
                place_fraction=1.0 / 3.0,
                place_width=0.08,
                place_scale=3.0,
                first_coordinates_multiplier=3.0,
                random_state=7,
                n_position_bins=20,
                return_metadata=True,
            )
        )

        self.assertEqual(B.shape, (2, 80))
        self.assertEqual(drive.shape, (4, 100, 80))

        spatial_indices = np.flatnonzero(
            np.isin(neuron_types, ["positional", "mixed"])
        )
        self.assertGreater(len(spatial_indices), 0)
        neuron = spatial_indices[0]
        peak_position = position[0, np.argmax(drive[0, :, neuron])]
        self.assertLess(abs(peak_position - centers[neuron]), 0.06)
        self.assertEqual(np.count_nonzero(B[:, neuron]), 0)
        self.assertEqual(
            metadata["preferred_bin"].shape,
            (80,),
        )
        self.assertTrue(
            np.all(metadata["preferred_bin"][spatial_indices] >= 0)
        )

        directional_indices = np.flatnonzero(neuron_types == "directional")
        self.assertTrue(
            np.all(metadata["preferred_bin"][directional_indices] == -1)
        )
        self.assertTrue(
            np.all(
                np.isin(
                    metadata["preferred_direction"][directional_indices],
                    [0, 1],
                )
            )
        )

        mixed_indices = np.flatnonzero(neuron_types == "mixed")
        self.assertGreater(len(mixed_indices), 0)
        mixed = mixed_indices[0]
        preferred_position = centers[mixed]
        outbound_index = np.argmin(abs(outbound - preferred_position))
        return_index = 50 + np.argmin(abs(returning - preferred_position))
        preferred_direction = metadata["preferred_direction"][mixed]
        preferred_index, other_index = (
            (outbound_index, return_index)
            if preferred_direction == 1
            else (return_index, outbound_index)
        )
        self.assertGreater(
            drive[0, preferred_index, mixed],
            drive[0, other_index, mixed],
        )

    def test_temporal_dataset_returns_extra_metadata(self):
        dataset = TemporalWindowDataset(
            np.zeros((3, 4, 2)),
            np.arange(3),
            np.arange(3),
            np.zeros(3),
            np.zeros(3),
            extra_metadata={"progress": np.array([0.0, 0.5, 1.0])},
        )

        self.assertEqual(float(dataset[1]["progress"]), 0.5)


if __name__ == "__main__":
    unittest.main()
