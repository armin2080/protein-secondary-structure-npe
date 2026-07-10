# Section 4 — Diagnostics & Inference notes

Handoff notes for the Task 4 (Diagnostics & Inference Lead) evaluation:
synthetic diagnostics on held-out data, SBC calibration checks, z-score and
contraction analysis, and real-protein validation on human insulin 1A7F.

## Modules

- **`setup_section4.py`** — One-time setup script. Generates insulin validation
  data, a small training dataset, and runs a quick training session that produces
  a usable checkpoint. Run once before using the other scripts:
  ```bash
  python setup_section4.py   # ~2-3 min, produces checkpoints/ + dataset/
  ```
- **`diagnostics.py`** — Synthetic evaluation on held-out simulated sequences.
  Loads the trained pipeline, runs inference with both BayesFlow and BiLSTM
  baseline, and produces seven diagnostic plots in `plots/`. See the table below
  for what each plot shows.
- **`validate_insulin.py`** — Real-protein validation on human insulin (PDB:
  1A7F). Loads the trained pipeline, runs inference on both chains, and compares
  predicted P(alpha) to experimental HELIX labels. Produces two plots and prints
  per-chain correlation/accuracy.

## How to load the trained model

Do not hand-roll the reload — use the helpers Member 3 already built:

```python
from training import load_trained_pipeline, predict_alpha_posterior
from forward_backward import load_dataset

summary_model, approximator, manifest = load_trained_pipeline("checkpoints")
sequences, true_labels, _ = load_dataset("dataset/insulin_1A7F.npz")

predictions = predict_alpha_posterior(
    list(sequences), summary_model, approximator,
    max_len=manifest["max_len"],  # must match training width (default 60)
    num_samples=300,
)
# predictions[i] has shape (num_samples, len(sequences[i])).
# Point estimate: np.nanmedian(predictions[i], axis=0) — NOT .mean(axis=0).
```

## Diagnostic plots produced by `diagnostics.py`

| Plot | What it shows | How to interpret |
|------|--------------|-----------------|
| `calibration.png` | Reliability diagram: binned predicted P(alpha) vs observed frequency, BayesFlow + BiLSTM side by side | Points on the diagonal = calibrated. Bars weighted by bin count. |
| `residuals.png` | Residual histograms + QQ plots for both models | Centered at 0 = unbiased. QQ close to diagonal = approximately normal errors. |
| `ecdf.png` | ECDF of per-position absolute error for both models | Curves further left = better (more positions with small error). |
| `scatter.png` | Hexbin scatter: predicted vs true P(alpha), both models | Tight diagonal = good recovery. Spread = uncertainty. |
| `sbc.png` | SBC rank histogram (fraction of posterior samples below the true value) + PIT ECDF | Uniform histogram = calibrated posterior. KS statistic quantifies deviation. |
| `zscore_coverage.png` | Z-score distribution (histogram + QQ vs N(0,1)), z-score vs true P(alpha) scatter, credible interval coverage (empirical vs nominal) | Z-scores ~N(0,1) = well-calibrated per-position uncertainty. Coverage on diagonal = honest credible intervals. |
| `contraction.png` | For 4 example sequences: true P(alpha), BayesFlow median ± 90% CI, BiLSTM point estimate | Blue band covering the black line = posterior captures truth. Wide band = high uncertainty. |

## Insulin validation plots produced by `validate_insulin.py`

| Plot | What it shows |
|------|--------------|
| `insulin_predictions.png` | Per-residue plot for chain A (21 aa) and chain B (29 aa): true alpha-helix label as shaded background, BayesFlow predicted P(alpha) (blue), BiLSTM prediction (red). Amino acid letters on x-axis. |
| `insulin_scatter.png` | Scatter of predicted P(alpha) vs experimental 0/1 label, both chains pooled. Reports per-model correlation, MAE, and accuracy at 0.5 threshold. |

## Results summary (quick-training checkpoint, n=800 train, n=200 val)

### Synthetic held-out evaluation (200 sequences, 6,764 positions)

| Metric | BayesFlow | BiLSTM (baseline) |
|--------|-----------|-------------------|
| MAE | 0.11 | **0.02** |
| Pearson correlation | 0.56 | **0.99** |

The BiLSTM far outperforms BayesFlow on synthetic data. This is expected:
the Forward-Backward posteriors are a **deterministic function** of the input
sequence under fixed HMM parameters, so a direct point-prediction network
learns the mapping perfectly. A density estimator (CouplingFlow) must model
a distribution over residuals and loses accuracy. See "Reporting guidance"
below for how to frame this in the presentation.

### Insulin 1A7F validation

| Chain | True helix % | BayesFlow corr | BiLSTM corr | Key insight |
|-------|-------------|---------------|-------------|-------------|
| A (21 aa) | 61.9% | 0.17 | 0.24 | Both models fail — the HMM prior (π=33%) is too far from 62% true helix, and emission tables aren't discriminative enough on 21 residues |
| B (29 aa) | 34.5% | **0.86** | 0.84 | Both models succeed — chain B's true helix fraction matches the HMM's stationary distribution, so the prior doesn't fight the evidence |

### SBC & calibration diagnostics

- **SBC ranks** show deviation from uniformity (KS statistic reported on the
  plot). This is expected: the HMM parameters are fixed (not inferred), and
  the target is deterministic, so the usual "recover a random parameter draw"
  SBC framing does not directly apply. The rank histogram should be understood
  as testing whether the BayesFlow posterior *envelopes* the true FB value
  with correct frequency, not as a classical prior-predictive SBC check.
- **Z-scores** are centered near 0 (unbiased) but overdispersed (std > 1),
  indicating the posterior is wider than it should be. This is consistent
  with the CouplingFlow's difficulty modeling the exact-zero point mass at
  position 0 (deterministic start in `other`).
- **Credible interval coverage** is plotted as empirical vs nominal coverage.
  Deviation from the diagonal indicates miscalibration — the posterior is
  typically conservative (empirical coverage > nominal), meaning the credible
  intervals are wider than needed.

## Important caveats for the report

### 1. Two ground truths, two different questions
- **Forward-Backward comparison** answers: "Did the network learn its training
  target?" → Yes, the BiLSTM nearly perfectly (corr 0.99), BayesFlow reasonably
  well (corr 0.56).
- **Insulin experimental comparison** answers: "Is the two-state HMM an
  adequate model of real secondary structure?" → For chain B, yes. For chain A,
  no — the HMM prior is too strong, and the emission tables lack information to
  override it. This is a **model limitation**, not a network failure.

### 2. BayesFlow does not improve accuracy — it adds uncertainty quantification
The `position_head` (BiLSTM + linear layer trained with binary cross-entropy,
no CouplingFlow) reaches ~0.99 correlation with Forward-Backward on its own.
BayesFlow adds:
- Posterior samples with credible intervals (see `contraction.png`).
- Calibration diagnostics (SBC, z-score, coverage) that assess how trustworthy
  the predictions are.
- A framework that would generalize to problems with genuine parameter
  uncertainty, even if here the target is deterministic.

State this honestly in the report rather than implying BayesFlow improved
accuracy. The project spec requires BayesFlow, and it delivers what it's
designed for — amortized inference and uncertainty quantification.

### 3. Sample quality issues with the CouplingFlow
A fraction of posterior draws (1–34% per sequence, depending on the sequence)
fall outside [0, 1] or are non-finite. This is a known consequence of:
- The CouplingFlow's base distribution being an unconstrained Gaussian.
- The HMM target containing an exact-zero point mass at position 0 of every
  sequence (deterministic start in `other`).
- The flow's NLL training objective on padded targets with ~44% exact zeros.

This does **not** break the project — point estimates via `np.nanmedian` are
robust and produce meaningful results. For SBC/z-score/coverage work we clip
finite draws to [0, 1] and drop NaN/Inf draws per-position. The warnings
printed by `predict_alpha_posterior` are informative, not errors.

### 4. Why chain A fails and chain B succeeds
The HMM stationary distribution is π(alpha) = 0.05 / (0.05 + 0.10) = **1/3**:
- Chain B is 34.5% true helix → matches the prior → model works.
- Chain A is 61.9% true helix → far from the prior → the 21-residue sequence
  isn't long enough for the emission evidence to overcome the prior bias.

This is a genuine scientific result: the two-state HMM with fixed empirical
emission tables is insufficient to model all protein secondary structure,
especially short, helix-rich sequences. Forward-Backward itself performs
equally poorly on chain A (correlation ~0.40), confirming this is a model
limitation, not a BayesFlow limitation.

## Running the scripts

```bash
python setup_section4.py      # one-time: generate data + train quick model
python diagnostics.py         # synthetic evaluation → 7 plots in plots/
python validate_insulin.py    # insulin validation → 2 plots in plots/
```

All plots are saved to `plots/` with non-interactive Matplotlib backend.
All scripts auto-generate their needed data if it is missing (test sequences,
insulin dataset) and give clear error messages if checkpoints are missing.

## For the report: suggested figures to include

| Figure | File | Caption idea |
|--------|------|-------------|
| Calibration comparison | `calibration.png` | "Reliability diagram comparing BayesFlow and BiLSTM predictions against true Forward-Backward posteriors on held-out simulated data." |
| SBC diagnostics | `sbc.png` | "Adapted Simulation-Based Calibration: rank histogram and PIT ECDF of BayesFlow posterior samples vs. deterministic Forward-Backward target." |
| Posterior contraction | `contraction.png` | "Per-sequence posterior contraction: BayesFlow median with 90% credible interval vs. true P(alpha) and BiLSTM point estimate on example sequences." |
| Insulin per-residue | `insulin_predictions.png` | "Per-residue predicted P(alpha-helix) for human insulin chains A and B (PDB: 1A7F), compared to experimental HELIX annotation." |
| Insulin scatter | `insulin_scatter.png` | "Scatter of predicted vs. experimental alpha-helix labels for both insulin chains, with per-model correlation and accuracy." |

The remaining diagnostic plots (residuals, ECDF, scatter, z-score coverage)
can be referenced in the text or placed in an appendix depending on
presentation time constraints.
