# protein-secondary-structure-npe

Inference of protein secondary structure motifs using a two-state Hidden Markov Model and neural posterior estimation (BayesFlow).

## Project Overview

We simulate amino acid sequences using an HMM with two states (alpha-helix / other) and given transition/emission probabilities. The Forward-Backward algorithm computes true posterior state probabilities. A BayesFlow neural network is trained to approximate these posterior probabilities from amino acid sequences alone.

## Task Breakdown

| Member | Role | Tasks |
|--------|------|-------|
| **1** | **Simulator & Data Lead** | Write HMM simulator (states, transitions, emission tables); run Forward-Backward to generate sequence-posterior pairs; preprocess and format human insulin 1A7F as real-world validation data |
| **2** | **Architecture Lead** | Design sequence-processing summary network (LSTM/Transformer); implement padding/masking for variable-length batches; configure the BayesFlow adapter and inference (coupling) network |
| **3** | **Optimization Lead** | Build the training pipeline; conduct hyperparameter tuning (optimizers, learning rate schedulers, regularization); output training diagnostics and loss trajectory curves |
| **4** | **Diagnostics & Inference Lead** | Execute Simulation-Based Calibration (SBC histograms, ECDFs); generate parameter recovery and z-score/contraction plots; run final inference on human insulin and compare to ground truth |

Slides are prepared together as a team.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Environment

Set up the virtual environment and install dependencies as shown in [Setup](#setup).
`requirements.txt` already includes `biopython`, which is needed to build the insulin
validation file (`insulin.py`).

### Generate a training dataset

Simulate sequences and label each position with its exact alpha-helix posterior, then
save the result to a single `.npz` file:

```python
from simulator import get_rng
from forward_backward import generate_dataset, save_dataset

rng = get_rng(0)
sequences, posteriors, states = generate_dataset(1000, rng, min_len=10, max_len=60)
save_dataset("dataset/train.npz", sequences, posteriors, states)
```

`generate_dataset` returns three index-aligned lists of variable-length arrays; each
`min_len`/`max_len` bounds the per-sequence length (inclusive).

### Load a dataset

```python
from forward_backward import load_dataset

sequences, posteriors, states = load_dataset("dataset/train.npz")
```

The return value is three lists, aligned by index:

- `sequences[i]` — integer-encoded amino acids (values 0–19),
- `posteriors[i]` — per-position `P(state = alpha | sequence)` in `[0, 1]`,
- `states[i]` — the true hidden state path (0 = alpha, 1 = other).

Sequences have different lengths, so they are not stored as a single rectangular array.
Instead, `save_dataset` concatenates all sequences end-to-end and keeps a `lengths`
array; `load_dataset` uses `lengths` to slice the concatenated arrays back into per-
sequence pieces. This avoids object arrays and pickling.

### Build the insulin validation file

```bash
python insulin.py
```

This downloads the mmCIF for PDB entry 1A7F, extracts both chains, and writes
`dataset/insulin_1A7F.npz` in the same format as a training dataset (chain A at index 0,
chain B at index 1). Requires `biopython`.

### Run the checks

```bash
python sanity_check.py
python check_forward_backward.py
python check_architecture.py
python check_training_smoke.py
python check_training_pipeline.py
```

`sanity_check.py` confirms the simulator's empirical start, transition, and emission
frequencies match the HMM parameters. `check_forward_backward.py` verifies the posteriors
are valid probabilities, agree with an independent Forward-Backward implementation, and
round-trip exactly through `save_dataset`/`load_dataset`. `check_architecture.py` and
`check_training_smoke.py` verify the padding/masking/BiLSTM architecture and BayesFlow
component configuration. `check_training_pipeline.py` is a fast end-to-end smoke test of
the training pipeline.

### Train the BayesFlow posterior estimator

```bash
python train.py
```

Builds a real-sized simulated dataset, pretrains the BiLSTM summary network
(`architecture.SequenceSummaryNetwork`), freezes it, runs a small automated
hyperparameter search over the BayesFlow coupling flow, retrains the best
configuration for longer, and saves everything needed downstream:

- `checkpoints/summary_network.pt`, `checkpoints/model.keras`,
  `checkpoints/manifest.json` — trained weights + the exact configuration used;
- `plots/pretrain_loss.png`, `plots/hyperparameter_search.png`,
  `plots/bayesflow_loss.png` — loss-trajectory diagnostics;
- `results/hyperparameter_search.csv` — every searched configuration's final loss.

This takes 40-60 minutes on CPU. See `SECTION3_NOTES.md` for the full pipeline
design.