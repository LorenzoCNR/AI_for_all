import numpy as np

from neurobridge.experiments import (
    SyntheticTaskConfig,
    build_linear_loading_and_place_fields,
)


def test_expected_minimum_latent_dimensions():
    SyntheticTaskConfig("circular", "circular", 3)
    SyntheticTaskConfig("linear", "linear", 2)


def test_linear_place_fields_are_localized():
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

    assert B.shape == (2, 80)
    assert drive.shape == (4, 100, 80)

    place_indices = np.flatnonzero(neuron_types == "place")
    assert len(place_indices) > 0

    neuron = place_indices[0]
    peak_position = position[0, np.argmax(drive[0, :, neuron])]
    assert abs(peak_position - centers[neuron]) < 0.03
    assert np.count_nonzero(B[:, neuron]) == 0
