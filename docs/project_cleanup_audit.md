# Repository Cleanup Audit

Date: 2026-06-03

This audit records the current public repository structure and the cleanup
choices made before publishing the project.

## Public Project Files

The public project is organized around:

```text
README.md
configs/
docs/
experiments/
src/
tests/
Tree_Module_packages.txt
setup_dirs.bat
```

The main runnable scripts are:

```text
experiments/Spike_simulator.py
experiments/encoder_baseline_suite.py
```

The main library code is under:

```text
src/neurobridge/
```

## Ignored Or Private Files

The following are intentionally excluded from Git:

```text
_private_archive/
outputs/
__pycache__/
.pytest_cache/
*.log
LaTeX auxiliary files
```

`outputs/` contains generated CSVs, HTML plots, and PNG previews. These are
reproducible artifacts and should be regenerated locally.

`_private_archive/` contains personal material, legacy scripts, old notebooks,
intermediate experiments, and files that should not be part of the public code
surface.

## Files Moved To Private Archive

Moved into:

```text
_private_archive/repo_cleanup_2026_06_03/
```

Examples:

```text
generative model.py
Commenti e Spiegazioni File in NeurobridgeUtils.txt
Idee_da_sviluppare_neurobridge.txt
Tasks_Neuro_bridge.py
experiments/Try_encoder_distance_.py
experiments/try_plot.py
src/neurobridge/models/temporal_cnn - Copia (*.py)
src/neurobridge/models/ch10_handson_with_pytorch.py
src/neurobridge/models/pca_model.py
src/neurobridge/models/neural_generator.py
src/neurobridge/encoders/temporal_cnn.py
src/neurobridge/eval/latent_recovery.py
src/neurobridge/eval/selection.py
src/neurobridge/sampling/f_windows
```

Rationale:

- duplicate or scratch files should not sit beside production modules;
- personal notes should not be mixed with public documentation;
- `src/` should contain importable package code only;
- generated outputs should not be versioned.

## Public Documentation

Current official documentation:

```text
docs/PROJECT_STATE.md
docs/simulator_spec.md
docs/simulator_design_appendix.md
docs/spike_simulator_generative_model.tex
docs/spike_simulator_generative_model.pdf
docs/implementation_file_map.md
```

`docs/PROJECT_STATE.md` is the fastest entry point for resuming the project.

## Remaining Review Candidates

These files remain public but should be reviewed later:

```text
src/neurobridge/models/blocks.py
src/neurobridge/eval/decoding_Eval.py
src/neurobridge/eval/knn_decoder.py
src/neurobridge/viz/plots.py
```

They may be useful, but they are less central than the simulator and baseline
suite. They should be audited before a formal release.

## Validation

The current core test command is:

```bash
python -m unittest tests.test_similarity tests.test_learning_components tests.test_representation_eval tests.test_embedding_plots
```

The baseline command is:

```bash
python experiments/encoder_baseline_suite.py
```
