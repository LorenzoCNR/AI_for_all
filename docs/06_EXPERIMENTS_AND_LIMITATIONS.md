# Experiments, Results, And Current Evidence

## Controlled Experiment Matrix

The primary reproducible entry points are:

| Notebook | Task | Known latent coordinates |
|---|---|---|
| `experiment_01_circular_3d.ipynb` | center-out reaching | X, Y, progress |
| `experiment_02_circular_5d.ipynb` | center-out reaching | X, Y, progress, velocity, context |
| `experiment_03_linear_position_direction.ipynb` | linear track | position, direction |
| `experiment_04_linear_enriched.ipynb` | linear track | position, direction, velocity, context |

The notebooks expose generation, windowing, split, PCA, CNN training, loss,
evaluation, and saving in separate cells. They do not hide the complete
experiment behind a single runner function.

## Fixed Reference Configuration

```text
trials                         160
time bins per trial            100
neurons                        100
bin width                      0.02 seconds
window size                    10 bins
window stride                  1
training fraction              0.8 by complete trial
CNN epochs                     30
minibatch size                 256
learning rate                  0.001
metadata temperature           0.5
embedding temperature          0.1
time / condition weights       0.5 / 0.5
random seed                    42
```

These are reproducibility defaults, not optimized scientific conclusions.

## What Is Fit And What Is Held Out

The split contains 128 training and 32 test trials. PCA and CNN1D are fitted
only on training windows. Metrics are computed on windows from complete unseen
trials.

The known latent remains available for evaluation but is not directly used as
a regression target for CNN training. Its task metadata influences the soft
structured target as described in
[Learning objectives](03_LEARNING_OBJECTIVES.md).

## Reference Held-Out Results

Single-seed RSA Spearman values are:

| Experiment | PCA full | CNN1D full | PCA motor core | CNN1D motor core |
|---|---:|---:|---:|---:|
| Circular 3D | 0.893 | 0.927 | 0.893 | 0.927 |
| Circular 5D | 0.682 | 0.619 | 0.899 | 0.904 |
| Linear position + direction | 0.893 | 0.778 | 0.893 | 0.778 |
| Linear enriched | 0.886 | 0.803 | 0.838 | 0.796 |

These values are regression checks for the current code and seed. They do not
establish model superiority. In particular, circular 5D shows that a model can
preserve the motor core strongly while agreeing less with the complete
velocity-context state.

## Reading The Figures

The main trajectory panels average embeddings across trials within condition
at each target time. For circular reaching, trajectories should depart from a
common center toward condition-specific targets. For the linear task, the
position-direction state has outbound and return branches.

The linear expected-state panel is not a second physical track. Its horizontal
axis is position and its vertical axis is direction. The right edge marks the
turnaround at position one; the left dotted connection indicates closure
between the end of one trial and the start of the next when visualizing the
state cycle.

Scatter plots of all windows can look like dense clouds because they include
trial noise and all time points. They are diagnostics and should not replace
condition-averaged trajectory plots.

## Generated Artifacts

Each notebook creates:

```text
outputs/<experiment-name>/
|-- metrics.json
|-- results.joblib
|-- models/
`-- figures/
```

`results.joblib` contains the latent state, spike counts, metadata, trial split,
embeddings, and evaluation values. `models/` contains fitted PCA and CNN1D
parameters. `figures/` contains task, population, trajectory, and diagnostic
plots.

Generated outputs are excluded from Git because they are binary and
reproducible from versioned notebooks and seeds. The exclusion does not mean
the experiment is undocumented: configuration and computation remain in the
repository.

## Test Status

The standard test command is:

```powershell
python -m unittest discover -s tests -v
```

The current suite executes 37 tests: 35 pass and two optional interactive
Plotly tests are skipped.

## Claims Supported Today

The current controlled evidence supports the following narrow statements:

- the package can generate known circular and linear task states and map them
  to stochastic neural populations;
- PCA and CNN1D can be trained and evaluated without trial leakage;
- held-out embeddings preserve part of the simulated latent distance geometry;
- motor-core and full-state recovery can differ;
- visual trajectory recovery and scalar RSA are complementary diagnostics.

## Claims Not Yet Supported

The current evidence does not establish:

- statistical stability across random seeds;
- superiority of CNN1D over alternative encoders;
- transfer to unseen conditions, sessions, subjects, or real datasets;
- biological adequacy of the spike simulator;
- causal communication between subjects;
- robust recovery of an imposed cross-subject lag;
- successful neuron-level XAI recovery of place fields.

## Experiments Required Next

1. Repeat every experiment over multiple seeds and report uncertainty.
2. Ablate time, condition, progress gate, window size, and both temperatures.
3. Compare soft structured loss with supervised and temporal-offset InfoNCE.
4. Add MLP, LSTM, Transformer, and neural latent-variable baselines under the
   same trial split.
5. Validate multisubject lag recovery with known delays and null controls.
6. Test whether feature attribution recovers simulated place-selective neurons.
7. Evaluate real neural data only after the controlled failure modes are
   characterized.

## Reproduction

Use the notebook guide:

[Reproducible experiments](../notebooks/EXPERIMENTS.md)

The guide identifies each stage, imported module, tensor shape, configurable
parameter, and saved output.
