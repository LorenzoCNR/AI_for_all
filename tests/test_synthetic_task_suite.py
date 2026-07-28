import unittest

import numpy as np

from neurobridge.data.dataset import TemporalWindowDataset
from neurobridge.experiments import (
    SyntheticTaskConfig,
    build_linear_loading_and_place_fields,
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

    def test_linear_place_fields_are_localized(self):
        position = np.tile(np.linspace(0.0, 1.0, 100), (4, 1))
        B, neuron_types, centers, drive = (
            build_linear_loading_and_place_fields(
                k=2,
                n_neurons=80,
                position=position,
                place_fraction=0.25,
                place_width=0.08,
                place_scale=3.0,
                first_coordinates_multiplier=3.0,
                random_state=7,
            )
        )

        self.assertEqual(B.shape, (2, 80))
        self.assertEqual(drive.shape, (4, 100, 80))

        place_indices = np.flatnonzero(neuron_types == "place")
        self.assertGreater(len(place_indices), 0)

        neuron = place_indices[0]
        peak_position = position[0, np.argmax(drive[0, :, neuron])]
        self.assertLess(abs(peak_position - centers[neuron]), 0.03)
        self.assertEqual(np.count_nonzero(B[:, neuron]), 0)

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
