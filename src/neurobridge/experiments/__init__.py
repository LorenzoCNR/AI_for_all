"""Reproducible experiment runners built on the NeuroBridge package."""

from .synthetic_task_suite import (
    SyntheticTaskConfig,
    build_similarity_matrix,
    build_windows_and_labels,
    build_linear_loading_and_place_fields,
    circular_neuron_type_probabilities,
    evaluate_models,
    save_experiment_figures,
    split_trials,
    run_synthetic_task_experiment,
)

__all__ = [
    "SyntheticTaskConfig",
    "build_similarity_matrix",
    "build_windows_and_labels",
    "build_linear_loading_and_place_fields",
    "circular_neuron_type_probabilities",
    "evaluate_models",
    "save_experiment_figures",
    "split_trials",
    "run_synthetic_task_experiment",
]
