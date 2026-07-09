# Section 3 — Training pipeline notes

Handoff notes for the Task 3 (Optimization Lead) training pipeline: dataset
padding, BiLSTM pretraining, BayesFlow coupling-flow training, hyperparameter
search, diagnostics, and saved artifacts.

## Modules

- **`training.py`** implements the pipeline: fixed-length dataset padding,
  BiLSTM pretraining, freezing + BayesFlow offline-data preparation,
  coupling-flow training, hyperparameter search, loss-curve plots, and
  artifact saving.
- **`train.py`** is the runnable entry point: builds a real-sized dataset,
  runs the full pipeline, and saves everything under `checkpoints/`,
  `plots/`, and `results/`.
- **`check_training_pipeline.py`** is a fast (~10s) smoke test covering every
  stage on tiny data — the correctness gate for this module, mirroring
  `check_architecture.py`/`check_forward_backward.py` from earlier sections.

## Pipeline design

Training happens in two stages, matching how `architecture.py`'s
`prepare_bayesflow_batch` computes `sequence_summary` under
`torch.no_grad()` — the summary network is not meant to be differentiated
through BayesFlow's flow loss:

1. **Pretrain** `SequenceSummaryNetwork` (`pretrain_summary_network`) on the
   masked BCE objective against Forward-Backward posteriors, plus a small
   auxiliary loss (predicting each sequence's mean alpha-helix fraction from
   the pooled `summary` vector) so both of the network's heads — the
   per-position `position_head` and the pooled `summary_head` — end up
   trained. Uses AdamW, gradient clipping, `ReduceLROnPlateau`, and early
   stopping with best-weight restoration.
2. **Freeze** it and compute the BayesFlow condition for the whole dataset
   (`compute_sequence_summaries`): the pooled `summary_head` vector
   concatenated with the per-position logits (masked to zero on padding).
   The pooled vector alone isn't enough — masked mean pooling collapses the
   whole sequence into one vector, so it can encode "how alpha-helix-heavy
   is this sequence overall" but not *which* position is alpha-helix.
   Concatenating the per-position logits restores that positional signal.
   Raw logits are used rather than `sigmoid(logits)` probabilities, since
   `CouplingFlow`'s base distribution is an unconstrained Gaussian and a
   second `[0, 1]`-bounded signal in the condition destabilizes training.
3. **Train** BayesFlow's `CouplingFlow` via `BasicWorkflow.fit_offline` on
   `{alpha_posteriors, sequence_summary}`, where `sequence_summary` is a
   fixed, non-differentiable condition.

The automated hyperparameter search (`run_hyperparameter_search`) targets
stage 3 — varying optimizer choice (Adam/AdamW), initial learning rate,
coupling-flow depth, and weight decay (regularization) across a curated
~6-config list. Configs are ranked by **masked accuracy against
Forward-Backward on real (non-padded) positions** (`masked_point_accuracy`),
not by the flow's own `val_loss`: every padded target has ~44% of its
entries at an exact-zero point mass (position 0, plus padding past each
sequence's real length), which a continuous density can't represent — a
config can drive `val_loss` arbitrarily negative by overfitting that point
mass without being any more accurate. `val_loss` is still recorded in the
results table for reference but isn't trusted for selection. For the same
reason, checkpoint saving (`FlowConfig.save_best_only`) and early stopping
both default to off for the final retrain: training runs for a fixed epoch
budget, and the resulting weights are evaluated afterward with
`masked_point_accuracy` on the full validation set rather than trusting
Keras's notion of "best."

## Global fixed padding length — important for Task 4

`architecture.pad_sequences`/`pad_targets` pad to *each batch's own* maximum
length. That's fine for Task 2's shape checks, but not for this
pipeline: `CouplingFlow`'s parameter dimensionality is fixed the first time
it is built, so every batch must have the same width for the life of the
trained model. `training.py` therefore adds `pad_to_fixed_length`/
`pad_targets_to_fixed_length`, which always pad to one chosen `max_len`
(default **60**, comfortably covering insulin's 21/29-aa chains), not to the
batch's own maximum.

**Task 4 must pad any new sequence to the same `max_len`** (read it from
`checkpoints/manifest.json`) before feeding it through the trained summary
network and coupling flow — `predict_alpha_posterior` already does this
correctly, so use it rather than hand-rolling the padding.

## Results

Measured with `masked_point_accuracy` on the full 400-example validation
set (`nanmedian` aggregation over 300 posterior samples per sequence,
restricted to each sequence's real positions):

| Evaluation | MAE | Correlation |
|---|---|---|
| Sequence-blind marginal baseline (ignores the input) | 0.162 | 0.200 |
| **Trained model — full validation set** | **0.061** | **0.798** |
| Trained model — held-out simulated (fresh seed) | 0.060 | 0.789 |

## Insulin: chain A vs. chain B

The trained model matches Forward-Backward almost exactly on both insulin
chains (correlation 0.96 on chain A, 0.99 on chain B, network vs. FB
directly) - it has faithfully learned its training target. But against the
*true experimental* labels, chain B is a good match (correlation 0.85) while
chain A is not (correlation 0.43). This is not a network generalization gap:
Forward-Backward itself, run directly on the real sequence, gets chain A
about as wrong as the network does (correlation 0.40 vs. true label). The
network is correctly reproducing Forward-Backward, including where
Forward-Backward is wrong.

The mechanism is the HMM's stationary distribution:
`alpha_stationary = 0.05 / (0.05 + 0.10) = 1/3`. Chain B is 34.5% true helix
— right at that prior, so the HMM does fine. Chain A is 61.9% true helix —
far above it — and the emission tables aren't informative enough to pull the
posterior away from the prior over a 21-residue sequence.

This is a genuine result worth including in the report: comparing
predictions to Forward-Backward tests whether the network learned its
target (it did); comparing to real insulin tests whether the two-state HMM
is an adequate model of real secondary structure (for chain A, it is not).
These are two different ground truths answering two different questions.

## What BayesFlow contributes here

Worth stating explicitly for the writeup, not left implicit: `position_head`
— a plain BiLSTM + linear head trained with ordinary binary cross-entropy,
no BayesFlow involved — reaches ~0.998-1.000 correlation with the true
Forward-Backward posterior on its own. The BayesFlow coupling flow,
conditioned in part on that same signal, returns something less accurate
(masked correlation ~0.80). This follows from `SECTION1_NOTES.md`'s
observation that Forward-Backward posteriors are a deterministic function of
the input sequence under the fixed HMM: there's no genuine parameter
uncertainty for a density estimator to recover, so it has structurally
nothing to add over a direct point-prediction network on accuracy alone.
What BayesFlow does add is the amortized-inference framework itself and
samples usable for calibration/SBC-style diagnostics (Task 4) in a way a
plain sigmoid output isn't. Worth framing deliberately as a group rather
than implying BayesFlow improved on a simpler baseline.

## Saved artifacts (`checkpoints/` after running `train.py`)

- `summary_network.pt` — `SequenceSummaryNetwork.state_dict()`.
- `model.keras` — the trained `ContinuousApproximator`, saved automatically
  by `BasicWorkflow`'s `checkpoint_filepath` during `fit_offline`.
- `manifest.json` — `max_len`, `min_len`, dataset sizes, the pretraining and
  final coupling-flow hyperparameters used, final losses, and
  `final_masked_mae`/`final_masked_corr` (the metric that matters,
  evaluated on the full validation set after training).
- `results/hyperparameter_search.csv` — every searched config's `val_loss`
  and `masked_mae`/`masked_corr` (the latter selected the winner).
- `plots/pretrain_loss.png`, `plots/hyperparameter_search.png` (masked MAE
  per config), `plots/bayesflow_loss.png`.

## For Task 4: loading the model and running inference

Don't hand-roll the reload — use `training.load_trained_pipeline` and
`training.predict_alpha_posterior`:

```python
from training import load_trained_pipeline, predict_alpha_posterior
from forward_backward import load_dataset
import numpy as np

summary_model, approximator, manifest = load_trained_pipeline("checkpoints")
sequences, true_labels, states = load_dataset("dataset/insulin_1A7F.npz")
predictions = predict_alpha_posterior(
    list(sequences), summary_model, approximator, max_len=manifest["max_len"], num_samples=300
)
# predictions[i] has shape (num_samples, len(sequences[i])).
# Point estimate: np.nanmedian(predictions[i], axis=0) -- NOT .mean(axis=0).
# Non-finite draws are preserved as NaN rather than hidden; a meaningful
# fraction of draws can be extreme even on a well-behaved sequence, so check
# any printed warnings before treating raw samples as a clean posterior for
# SBC/ECDF/z-score/contraction work.
```

Gotchas `load_trained_pipeline` already handles, in case you need to do this
by hand: (1) `bayesflow` must be imported *before* `keras.saving.load_model`,
or Keras can't find `ContinuousApproximator`/`CouplingFlow`/`Adapter` and
raises a confusing `TypeError`; (2) any new sequence must be padded to
exactly `manifest["max_len"]` (not its own length); (3) `sequence_summary`
must be built the same way used in training (pooled summary + per-position
logits, see `compute_sequence_summaries`).

## Running it

```bash
python check_training_pipeline.py   # fast (~10s) correctness check
python train.py                     # full pipeline (~40-60 min on CPU)
```

## Known limitations

- The flow's loss still directly optimizes an unmasked NLL over targets with
  a large exact-zero point mass. Hyperparameter/final-model *selection* is
  robust to this (ranked by masked accuracy, not raw loss), but individual
  training runs can still produce occasional numerically extreme samples.
  Check `manifest.json`'s `final_masked_mae`/`final_masked_corr` — and
  ideally rerun `masked_point_accuracy` yourself — before trusting any
  specific checkpoint, including the one committed here.
- See "What BayesFlow contributes here" above: this needs to be part of the
  team's framing for the report, not just this file.
- No explicit sequence-length feature is fed to the flow beyond what the
  per-position logits/pooled summary implicitly carry.
