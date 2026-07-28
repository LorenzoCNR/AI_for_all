# NeuroBridge Control Room

Questa e la mappa pratica per riprendere il controllo della piattaforma.
NeuroBridge oggi e un prototipo scientifico funzionante: simula attivita
neurale da una dinamica latente nota, allena encoder su finestre temporali e
misura quanto gli embedding recuperano la geometria latente.

## 1. Idea centrale

Pipeline concettuale:

```text
Z_task -> Z_neural_driver -> B -> u -> lambda -> X -> windows -> encoder -> embedding -> metrics
```

Significato:

- `Z_task`: traiettoria latente condivisa, cioe il ground truth del compito.
- `Z_neural_driver`: latente usato per generare l'attivita neurale; puo avere lag tra soggetti.
- `B`: loading matrix che mappa latenti su neuroni.
- `u`: drive neurale prima della non-linearita.
- `lambda`: rate/intensita di firing.
- `X`: spike counts simulati.
- `windows`: finestre temporali date agli encoder.
- `embedding`: rappresentazione appresa.
- `metrics`: confronto quantitativo con il ground truth.

## 2. File che comandano la piattaforma

```text
experiments/encoder_baseline_suite.py
```

Il file principale. Fa tutto:

- definisce la configurazione;
- genera dati sintetici multi-soggetto;
- costruisce finestre temporali;
- allena PCA, CNN e Transformer;
- calcola metriche;
- salva CSV, figure e `.mat`.

```text
src/neurobridge/data/sim/Lat_traj_generator.py
src/neurobridge/data/sim/builders.py
src/neurobridge/data/sim/Spikes_generator.py
```

Il simulatore. Qui vivono traiettorie latenti, lag, loading matrix, rate e spike
emission.

```text
src/neurobridge/data/dataset.py
src/neurobridge/sampling/f_windows.py
```

Costruzione dataset e finestre temporali.

```text
src/neurobridge/models/temporal_cnn.py
```

Encoder neurali: CNN, MLP, LSTM, Transformer.

```text
src/neurobridge/losses/infonce.py
src/neurobridge/sampling/batch_similarity.py
```

Loss contrastive e geometria delle similarita nel batch.

```text
src/neurobridge/train/loop.py
```

Training loop minimo.

```text
src/neurobridge/eval/representation.py
```

Metriche: Procrustes R2, RSA, allineamento cross-subject con lag.

## 3. Come lanciare le cose

Test rapidi:

```bash
python -m unittest tests.test_similarity tests.test_learning_components tests.test_representation_eval tests.test_embedding_plots
```

Suite baseline:

```bash
python experiments/encoder_baseline_suite.py
```

Output generati:

```text
outputs/baselines/
```

`outputs/` e ignorata da Git. Si puo rigenerare.

## 4. Le leve scientifiche principali

Nel file:

```text
experiments/encoder_baseline_suite.py
```

cerca `config = { ... }` dentro `run_suite()`.

Le leve piu importanti:

```text
seed
n_trials
trial_len
n_neurons
n_conditions
n_traj_k
subject_lags_bins
window_size
stride
embedding_dim
batch_size
epochs
loss_mode
encoders
metric_max_samples
```

Loss disponibili:

```text
soft_structured
structured_specs
supervised_infonce
time_offset_infonce
```

Encoder disponibili:

```text
pca
cnn
mlp
lstm
transformer
```

Nota: PCA viene sempre calcolata come baseline classica. Gli encoder neurali
sono controllati da `encoders`.

## 5. Stato attuale del progetto

Quello che funziona:

- simulatore sintetico controllato;
- multi-soggetto con lag di risposta;
- windowing trial-safe;
- CNN, Transformer e PCA;
- loss contrastive multiple;
- metriche quantitative;
- export `.mat` per analisi MATLAB/Python;
- test unitari di base.

Quello che e ancora troppo prototipo:

- `encoder_baseline_suite.py` e troppo grande;
- la config e duplicata tra `make_default_config()` e `run_suite()`;
- mancano esperimenti multi-seed sistematici;
- manca un vero sampler temporale strutturato;
- mancano tabelle aggregate con intervalli di confidenza;
- mancano script CLI puliti per riprodurre ogni figura;
- i risultati attuali sono fast-run, non paper-grade.

## 6. Primo refactor per avere controllo

Ordine consigliato:

1. Separare la configurazione in YAML.
2. Trasformare `encoder_baseline_suite.py` in orchestratore leggero.
3. Creare funzioni dedicate:
   - `simulate_dataset(config)`
   - `build_subject_windows(data, config)`
   - `fit_baseline(method, dataset, config)`
   - `evaluate_embeddings(...)`
   - `save_results(...)`
4. Aggiungere CLI:

```bash
python -m neurobridge.cli.run_baseline --config configs/baseline_soft.yaml
python -m neurobridge.cli.run_baseline --config configs/baseline_time.yaml
python -m neurobridge.cli.run_baseline --config configs/baseline_supervised.yaml
```

## 7. Roadmap Nature-level

Livello 1: repo controllabile

- struttura pulita;
- test verdi;
- config versionate;
- output ignorati;
- primo commit con solo package, esperimenti, docs e test.

Livello 2: benchmark riproducibile

- multi-seed;
- sweep su rumore, lag, window size, neuroni, trial;
- confronto PCA / CNN / Transformer / supervised / time-offset;
- tabelle CSV aggregate;
- figure generate da script.

Livello 3: solidita scientifica

- metriche con confidence interval;
- ablation study;
- failure modes;
- controllo su bias della loss;
- validazione cross-subject piu robusta.

Livello 4: contributo forte

- simulatore motivato e ben descritto;
- benchmark standardizzato;
- baseline temporale con sampling esplicito;
- confronto con metodo esterno reale;
- paper narrative chiara: cosa NeuroBridge misura che gli altri benchmark non
  misurano.

## 8. Regola pratica

Quando ti perdi:

1. Parti da `experiments/encoder_baseline_suite.py`.
2. Guarda `config`.
3. Segui la pipeline in ordine:

```text
make_synthetic_shared_latent_data
make_window_dataset
train_encoder
evaluate_latent_recovery_sampled
lagged_alignment_by_trial_time
save_embedding_mat_files
```

Questo e il filo rosso della piattaforma.
