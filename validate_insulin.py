"""Section 4 — Insulin 1A7F validation.

Run:  python validate_insulin.py

Loads the trained pipeline and evaluates it on human insulin (PDB: 1A7F).
Produces two plots in the ``plots/`` directory:
  1. insulin_predictions.png  — per-residue predicted P(alpha) vs. experimental
                                 HELIX label for chain A and chain B
  2. insulin_scatter.png      — scatter of predictions vs. labels across residues

Also prints per-chain metrics (correlation, accuracy at 0.5 threshold).
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr

from training import (
    CHECKPOINT_DIR,
    load_trained_pipeline,
    predict_alpha_posterior,
)
from forward_backward import load_dataset
from encoding import IDX_TO_AA

PLOTS_DIR = "plots"
INSULIN_PATH = "dataset/insulin_1A7F.npz"
CHAIN_NAMES = ["Chain A (21 aa, GIVEQCCTSICSLYQLENYCN)",
               "Chain B (29 aa, FVNQHLCGSHLVEALELVCGERGGFYTPK)"]


def _sequence_to_labels(sequence, aa_labels=None):
    """Convert integer-encoded sequence to amino-acid letter labels."""
    if aa_labels is None:
        return [IDX_TO_AA[int(i)] for i in sequence]
    return [f"{IDX_TO_AA[int(i)]}\n({aa_labels[i]:.0f})" for i in sequence]


def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load trained pipeline
    # ------------------------------------------------------------------
    print("Loading trained pipeline ...")
    try:
        summary_model, approximator, manifest = load_trained_pipeline(CHECKPOINT_DIR)
        max_len = manifest["max_len"]
    except FileNotFoundError:
        print("ERROR: No checkpoints found. Run setup_section4.py first.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Load insulin data
    # ------------------------------------------------------------------
    if not os.path.exists(INSULIN_PATH):
        print(f"ERROR: {INSULIN_PATH} not found. Run setup_section4.py or insulin.py.")
        sys.exit(1)

    print(f"Loading insulin data from {INSULIN_PATH} ...")
    sequences, true_labels, _states = load_dataset(INSULIN_PATH)

    # ------------------------------------------------------------------
    # 3. Run inference
    # ------------------------------------------------------------------
    print("Running BayesFlow inference on insulin ...")
    from architecture import pad_sequences, predict_alpha_probabilities
    bf_samples = predict_alpha_posterior(
        sequences, summary_model, approximator, max_len=max_len, num_samples=300, seed=0
    )
    bf_point = [np.nanmedian(s, axis=0) for s in bf_samples]

    # BiLSTM baseline
    summary_model.eval()
    padded, mask, lengths = pad_sequences(sequences)
    with torch.no_grad():
        _summary, logits = summary_model(padded, mask=mask, lengths=lengths)
        bilstm_probs = predict_alpha_probabilities(logits, mask)
    bilstm_point = [bilstm_probs[i, :int(lengths[i])].cpu().numpy() for i in range(len(sequences))]

    # ------------------------------------------------------------------
    # 4. Per-chain plots
    # ------------------------------------------------------------------
    print("Generating per-residue prediction plots ...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    for chain_idx, (ax, name) in enumerate(zip(axes, CHAIN_NAMES)):
        seq = sequences[chain_idx]
        true = true_labels[chain_idx]
        bf_pred = bf_point[chain_idx]
        bilstm_pred = bilstm_point[chain_idx]

        positions = np.arange(len(seq))
        aa_labels = _sequence_to_labels(seq)

        # True label as shaded background
        ax.fill_between(positions, 0, 1, where=(true > 0.5),
                        color="tab:orange", alpha=0.15, label="True alpha-helix (experimental)")
        ax.fill_between(positions, 0, 1, where=(true <= 0.5),
                        color="tab:gray", alpha=0.08, label="True other (experimental)")

        ax.plot(positions, bf_pred, "o-", color="tab:blue", markersize=5, label="BayesFlow predicted P(alpha)")
        ax.plot(positions, bilstm_pred, "s--", color="tab:red", markersize=4, label="BiLSTM predicted P(alpha)")

        ax.set_xticks(positions)
        ax.set_xticklabels(aa_labels, fontsize=7)
        ax.set_ylabel("P(alpha-helix)")
        ax.set_title(name)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylim(-0.05, 1.15)
        ax.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "insulin_predictions.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5. Scatter plot: predicted vs true across both chains
    # ------------------------------------------------------------------
    print("Generating insulin scatter plot ...")
    all_true = np.concatenate([np.asarray(t) for t in true_labels])
    all_bf = np.concatenate([np.asarray(p) for p in bf_point])
    all_bilstm = np.concatenate([np.asarray(p) for p in bilstm_point])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, pred, label, color in zip(
        axes,
        [all_bf, all_bilstm],
        ["BayesFlow", "BiLSTM"],
        ["tab:blue", "tab:red"],
    ):
        # Jitter binary labels slightly for visibility
        jitter = np.random.default_rng(0).uniform(-0.02, 0.02, size=len(all_true))
        ax.scatter(all_true + jitter, pred, s=15, alpha=0.5, color=color,
                   edgecolors="white", linewidth=0.2)

        corr, _ = pearsonr(all_true, pred)
        mae = np.mean(np.abs(pred - all_true))
        acc = np.mean((pred >= 0.5) == (all_true >= 0.5))
        ax.set_xlabel("Experimental alpha-helix label (0/1)")
        ax.set_ylabel("Predicted P(alpha)")
        ax.set_title(f"{label}\ncorr={corr:.3f}, MAE={mae:.3f}, acc={acc:.3f}")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.05, 1.15)
        ax.grid(True, alpha=0.3)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "insulin_scatter.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Per-chain metrics
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("  Insulin Validation Results")
    print("=" * 50)
    for chain_idx, name in enumerate(["Chain A", "Chain B"]):
        true = true_labels[chain_idx]
        bf_p = bf_point[chain_idx]
        bilstm_p = bilstm_point[chain_idx]

        bf_corr, _ = pearsonr(true, bf_p)
        bilstm_corr, _ = pearsonr(true, bilstm_p)
        bf_acc = np.mean((bf_p >= 0.5) == (true >= 0.5))
        bilstm_acc = np.mean((bilstm_p >= 0.5) == (true >= 0.5))

        print(f"\n  {name} ({len(true)} residues):")
        print(f"    BayesFlow  corr={bf_corr:.4f}  acc={bf_acc:.4f}")
        print(f"    BiLSTM     corr={bilstm_corr:.4f}  acc={bilstm_acc:.4f}")

    print(f"\n  Plots saved to {PLOTS_DIR}/")
    print(f"    insulin_predictions.png, insulin_scatter.png")


if __name__ == "__main__":
    main()
