"""Section 4 — Synthetic diagnostics for the trained BayesFlow model.

Run:  python diagnostics.py

Loads the trained pipeline and evaluates it on held-out simulated sequences.
Produces four diagnostic plots in the ``plots/`` directory:
  1. calibration.png       — reliability diagram: predicted P(alpha) vs observed frequency
  2. residuals.png         — per-position residuals (predicted - true) histogram + QQ
  3. ecdf.png              — ECDF of absolute errors for BayesFlow vs. BiLSTM baseline
  4. scatter.png           — predicted vs true P(alpha) scatter with hexbin density
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch

from training import (
    CHECKPOINT_DIR,
    DEFAULT_MAX_LEN,
    load_trained_pipeline,
    predict_alpha_posterior,
)
from forward_backward import generate_dataset, load_dataset
from simulator import get_rng
from architecture import (
    SequenceSummaryNetwork,
    pad_sequences,
    pad_targets,
    masked_bce_loss,
    predict_alpha_probabilities,
)

PLOTS_DIR = "plots"


def load_or_generate_test_sequences(
    n: int = 200,
    min_len: int = 10,
    max_len: int = DEFAULT_MAX_LEN,
    seed: int = 999,
):
    """Return (sequences, posteriors) for held-out test data.

    Tries ``dataset/test_quick.npz`` first; generates fresh data if missing.
    """
    test_path = "dataset/test_quick.npz"
    if os.path.exists(test_path):
        print(f"Loading test data from {test_path}")
        seqs, posts, _states = load_dataset(test_path)
        return seqs, posts
    print(f"Generating fresh test data (n={n}, seed={seed}) ...")
    rng = get_rng(seed)
    seqs, posts, _states = generate_dataset(n, rng, min_len=min_len, max_len=max_len)
    from forward_backward import save_dataset
    os.makedirs("dataset", exist_ok=True)
    save_dataset(test_path, seqs, posts, _states)
    print(f"Saved test data to {test_path}")
    return seqs, posts


def _predict_bilstm(
    sequences,
    summary_model: SequenceSummaryNetwork,
):
    """Get BiLSTM point predictions (no BayesFlow).

    Returns list of arrays (one per sequence), each length L_i.
    """
    padded, mask, lengths = pad_sequences(sequences)
    summary_model.eval()
    with torch.no_grad():
        _summary, logits = summary_model(padded, mask=mask, lengths=lengths)
        probs = predict_alpha_probabilities(logits, mask)
    results = []
    for i, length in enumerate(lengths.tolist()):
        results.append(probs[i, :int(length)].cpu().numpy())
    return results


def _reliability_diagram(
    true_all: np.ndarray,
    pred_all: np.ndarray,
    n_bins: int = 10,
):
    """Compute calibration curve: binned predicted prob vs observed fraction."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    observed = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    for b in range(n_bins):
        mask = (pred_all >= bins[b]) & (pred_all < bins[b + 1])
        counts[b] = mask.sum()
        if counts[b] > 0:
            observed[b] = true_all[mask].mean()

    return bin_centers, observed, counts


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
    # 2. Load / generate test data
    # ------------------------------------------------------------------
    print("Preparing test data ...")
    test_seqs, test_posts = load_or_generate_test_sequences(n=200, seed=999)

    # ------------------------------------------------------------------
    # 3. Run inference — BayesFlow
    # ------------------------------------------------------------------
    print("Running BayesFlow inference ...")
    bf_samples = predict_alpha_posterior(
        test_seqs, summary_model, approximator, max_len=max_len, num_samples=200, seed=0
    )
    bf_point = np.array([np.nanmedian(s, axis=0) for s in bf_samples], dtype=object)

    # ------------------------------------------------------------------
    # 4. Run inference — BiLSTM baseline
    # ------------------------------------------------------------------
    print("Running BiLSTM baseline inference ...")
    bilstm_point = _predict_bilstm(test_seqs, summary_model)

    # ------------------------------------------------------------------
    # 5. Flatten everything to flat arrays of real positions
    # ------------------------------------------------------------------
    true_flat = np.concatenate([np.asarray(p) for p in test_posts])
    bf_flat = np.concatenate([np.asarray(p) for p in bf_point])
    bilstm_flat = np.concatenate([np.asarray(p) for p in bilstm_point])

    n_positions = len(true_flat)
    print(f"Total real positions evaluated: {n_positions}")

    # ------------------------------------------------------------------
    # Figure 1 — Calibration (reliability diagram)
    # ------------------------------------------------------------------
    print("Generating calibration plot ...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # BayesFlow calibration
    bins, obs, cnts = _reliability_diagram(true_flat, bf_flat)
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
    ax.scatter(bins, obs, s=cnts * 2, alpha=0.7, label="BayesFlow", color="tab:blue")
    ax.set_xlabel("Predicted P(alpha)")
    ax.set_ylabel("Observed frequency")
    ax.set_title("BayesFlow — Reliability Diagram")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # BiLSTM calibration
    bins, obs, cnts = _reliability_diagram(true_flat, bilstm_flat)
    ax = axes[1]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
    ax.scatter(bins, obs, s=cnts * 2, alpha=0.7, label="BiLSTM", color="tab:orange")
    ax.set_xlabel("Predicted P(alpha)")
    ax.set_ylabel("Observed frequency")
    ax.set_title("BiLSTM — Reliability Diagram")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "calibration.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2 — Residuals
    # ------------------------------------------------------------------
    print("Generating residuals plot ...")
    residuals_bf = bf_flat - true_flat
    residuals_bilstm = bilstm_flat - true_flat

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # BayesFlow residual histogram
    axes[0, 0].hist(residuals_bf, bins=80, alpha=0.7, color="tab:blue", edgecolor="white")
    axes[0, 0].axvline(0, color="k", linestyle="--", alpha=0.3)
    axes[0, 0].set_xlabel("Residual (predicted - true)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title(f"BayesFlow Residuals\nMAE={np.mean(np.abs(residuals_bf)):.4f}")

    # BiLSTM residual histogram
    axes[0, 1].hist(residuals_bilstm, bins=80, alpha=0.7, color="tab:orange", edgecolor="white")
    axes[0, 1].axvline(0, color="k", linestyle="--", alpha=0.3)
    axes[0, 1].set_xlabel("Residual (predicted - true)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title(f"BiLSTM Residuals\nMAE={np.mean(np.abs(residuals_bilstm)):.4f}")

    # QQ plot — BayesFlow
    from scipy import stats as scipy_stats
    sorted_res = np.sort(residuals_bf)
    theoretical = scipy_stats.norm.ppf(
        (np.arange(len(sorted_res)) + 0.5) / len(sorted_res),
        loc=np.mean(residuals_bf),
        scale=np.std(residuals_bf),
    )
    axes[1, 0].scatter(theoretical, sorted_res, s=1, alpha=0.5, color="tab:blue")
    axes[1, 0].plot(theoretical, theoretical, "k--", alpha=0.3)
    axes[1, 0].set_xlabel("Theoretical quantiles")
    axes[1, 0].set_ylabel("Observed quantiles")
    axes[1, 0].set_title("BayesFlow — QQ Plot")

    # QQ plot — BiLSTM
    sorted_res = np.sort(residuals_bilstm)
    theoretical = scipy_stats.norm.ppf(
        (np.arange(len(sorted_res)) + 0.5) / len(sorted_res),
        loc=np.mean(residuals_bilstm),
        scale=np.std(residuals_bilstm),
    )
    axes[1, 1].scatter(theoretical, sorted_res, s=1, alpha=0.5, color="tab:orange")
    axes[1, 1].plot(theoretical, theoretical, "k--", alpha=0.3)
    axes[1, 1].set_xlabel("Theoretical quantiles")
    axes[1, 1].set_ylabel("Observed quantiles")
    axes[1, 1].set_title("BiLSTM — QQ Plot")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "residuals.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 3 — ECDF of absolute errors
    # ------------------------------------------------------------------
    print("Generating ECDF plot ...")
    abs_err_bf = np.abs(residuals_bf)
    abs_err_bilstm = np.abs(residuals_bilstm)

    fig, ax = plt.subplots(figsize=(8, 5))
    for data, label, color in [
        (abs_err_bf, "BayesFlow", "tab:blue"),
        (abs_err_bilstm, "BiLSTM", "tab:orange"),
    ]:
        sorted_data = np.sort(data)
        y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax.plot(sorted_data, y, label=label, color=color, linewidth=1.5)

    ax.set_xlabel("Absolute error |predicted - true|")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("ECDF of Per-Position Absolute Errors")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, None)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "ecdf.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 4 — Predicted vs True scatter (hexbin)
    # ------------------------------------------------------------------
    print("Generating scatter plot ...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (pred, label, color) in zip(
        axes,
        [(bf_flat, "BayesFlow", "Blues"), (bilstm_flat, "BiLSTM", "Oranges")],
    ):
        hb = ax.hexbin(true_flat, pred, gridsize=60, cmap=color, mincnt=1)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("True P(alpha)")
        ax.set_ylabel("Predicted P(alpha)")
        ax.set_title(label)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        plt.colorbar(hb, ax=ax, label="Count")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "scatter.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 5 — Simulation-Based Calibration (SBC) rank histogram
    # ------------------------------------------------------------------
    print("Generating SBC rank histogram ...")
    # For each position, compute the rank of the true value among the
    # posterior samples (fraction of samples below the true value).
    # A well-calibrated posterior yields uniformly distributed ranks.
    ranks = []
    for seq_idx in range(len(test_seqs)):
        samples = bf_samples[seq_idx]  # (n_samples, length)
        true_vals = np.asarray(test_posts[seq_idx])
        length = len(true_vals)
        for pos in range(length):
            post_samples = samples[:, pos]
            # Filter NaN samples
            valid = np.isfinite(post_samples)
            if valid.sum() < 10:
                continue
            valid_samples = post_samples[valid]
            # Fraction of posterior samples below the true value
            rank = np.mean(valid_samples < true_vals[pos])
            ranks.append(rank)

    ranks = np.array(ranks)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Rank histogram — uniform if calibrated
    axes[0].hist(ranks, bins=20, range=(0, 1), color="tab:blue", alpha=0.7,
                 edgecolor="white")
    axes[0].axhline(y=len(ranks) / 20, color="k", linestyle="--", alpha=0.3,
                    label="Expected (uniform)")
    axes[0].set_xlabel("Rank (fraction of samples below true value)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"SBC Rank Histogram\nn={len(ranks)} positions")
    axes[0].legend(fontsize=8)

    # PIT (Probability Integral Transform) ECDF
    sorted_ranks = np.sort(ranks)
    axes[1].plot(sorted_ranks, np.linspace(0, 1, len(sorted_ranks)),
                 color="tab:blue", linewidth=1.5, label="Empirical CDF of ranks")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.3, label="Uniform CDF")
    axes[1].fill_between(
        np.linspace(0, 1, 100),
        np.linspace(0, 1, 100) - 0.1,
        np.linspace(0, 1, 100) + 0.1,
        alpha=0.1, color="gray",
    )
    axes[1].set_xlabel("Rank")
    axes[1].set_ylabel("Cumulative fraction")
    axes[1].set_title("SBC — PIT ECDF")
    axes[1].legend(fontsize=8)

    # Compute Kolmogorov-Smirnov statistic vs uniform
    ks_stat = np.max(np.abs(sorted_ranks - np.linspace(0, 1, len(sorted_ranks))))
    axes[1].text(0.05, 0.95, f"KS = {ks_stat:.3f}", transform=axes[1].transAxes,
                 fontsize=9, verticalalignment="top")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "sbc.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 6 — Z-score analysis (per-position calibration)
    # ------------------------------------------------------------------
    print("Generating z-score analysis ...")
    z_scores = []
    z_positions_total = 0
    z_positions_skipped_boundary = 0
    for seq_idx in range(len(test_seqs)):
        samples = bf_samples[seq_idx]  # (n_samples, length)
        true_vals = np.asarray(test_posts[seq_idx])
        length = len(true_vals)
        for pos in range(length):
            z_positions_total += 1
            # Skip degenerate point-mass positions (padding & boundaries)
            # where the HMM truth is exactly zero — these break the z-score
            # formula because the flow's predicted std collapses to near-zero.
            if true_vals[pos] < 1e-10:
                z_positions_skipped_boundary += 1
                continue
            post_samples = samples[:, pos]
            valid = np.isfinite(post_samples)
            if valid.sum() < 10:
                continue
            valid_samples = post_samples[valid]
            pred_mean = np.nanmean(valid_samples)
            pred_std = np.nanstd(valid_samples)
            if pred_std < 1e-4:  # degenerate posterior — variance collapsed
                continue
            z = (true_vals[pos] - pred_mean) / pred_std
            if np.isfinite(z):
                z_scores.append(z)

    z_scores = np.array(z_scores)
    print(f"  Z-scores computed: {len(z_scores)} (skipped {z_positions_skipped_boundary} boundary/padding, "
          f"dropped {z_positions_total - len(z_scores) - z_positions_skipped_boundary} non-finite/low-std)")

    # -- Z-score distribution (standalone wide figure) --
    from scipy import stats as scipy_stats
    z_display = z_scores[(z_scores > -4) & (z_scores < 4)]
    n_clipped = len(z_scores) - len(z_display)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.hist(z_display, bins=50, density=True, alpha=0.7, color="tab:blue",
            edgecolor="white", linewidth=0.3)
    x_grid = np.linspace(-4, 4, 300)
    ax.plot(x_grid, scipy_stats.norm.pdf(x_grid, 0, 1), "k--",
            linewidth=1.5, label="N(0,1)")
    ax.set_xlabel("Z-score")
    ax.set_ylabel("Density")
    ax.set_title(f"Z-score Distribution (|z| < 4)  —  mean={np.mean(z_display):.3f}, std={np.std(z_display):.3f}"
                 + (f"  [{n_clipped} clipped]" if n_clipped else ""))
    ax.legend(fontsize=10, loc="upper right", framealpha=0.7, edgecolor="gray")
    ax.set_xlim(-4, 4)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "zscore_distribution.png"), dpi=180)
    plt.close(fig)

    # -- Credible interval coverage (standalone wide figure) --
    credible_levels = np.arange(0.05, 1.0, 0.05)
    empirical_coverage = []
    for level in credible_levels:
        alpha_tail = (1 - level) / 2
        covered = 0
        total = 0
        for seq_idx in range(len(test_seqs)):
            samples = bf_samples[seq_idx]
            true_vals = np.asarray(test_posts[seq_idx])
            length = len(true_vals)
            for pos in range(length):
                if true_vals[pos] < 1e-10:  # skip boundary/padding
                    continue
                valid = np.isfinite(samples[:, pos])
                if valid.sum() < 10:
                    continue
                valid_samples = samples[valid, pos]
                lo = np.nanpercentile(valid_samples, 100 * alpha_tail)
                hi = np.nanpercentile(valid_samples, 100 * (1 - alpha_tail))
                if lo <= true_vals[pos] <= hi:
                    covered += 1
                total += 1
        empirical_coverage.append(covered / total if total > 0 else 0)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Ideal")
    ax.plot(credible_levels, empirical_coverage, "o-", color="tab:blue",
            linewidth=1.5, markersize=5, label="BayesFlow")
    ax.set_xlabel("Nominal credible level")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Credible Interval Coverage")
    ax.legend(fontsize=10, loc="lower right", framealpha=0.7, edgecolor="gray")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "zscore_coverage.png"), dpi=180)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 7 — Posterior contraction
    # ------------------------------------------------------------------
    print("Generating posterior contraction plot ...")
    # Show how BayesFlow's posterior varies with input vs. BiLSTM point estimate.
    # For a few example sequences, plot the point estimate ± credible interval.
    n_examples = min(4, len(test_seqs))
    fig, axes = plt.subplots(n_examples, 1, figsize=(12, 3 * n_examples), squeeze=False)
    axes = axes.flatten()

    for ex in range(n_examples):
        ax = axes[ex]
        seq = test_seqs[ex]
        true_vals = np.asarray(test_posts[ex])
        samples = bf_samples[ex]  # (n_samples, length)
        length = len(true_vals)
        positions = np.arange(length)

        # Compute posterior quantiles per position
        pred_median = np.array([np.nanmedian(samples[:, pos]) for pos in range(length)])
        pred_lo = np.array([np.nanpercentile(samples[:, pos], 5) for pos in range(length)])
        pred_hi = np.array([np.nanpercentile(samples[:, pos], 95) for pos in range(length)])

        # True values
        ax.plot(positions, true_vals, "o-", color="k", markersize=4, linewidth=1.5,
                label="True P(alpha)")
        # BayesFlow prediction with credible interval
        ax.fill_between(positions, pred_lo, pred_hi, alpha=0.25, color="tab:blue",
                        label="90% CI (BayesFlow)")
        ax.plot(positions, pred_median, "s-", color="tab:blue", markersize=3,
                linewidth=1, label="BayesFlow median")
        # BiLSTM point estimate
        bilstm_p = bilstm_point[ex]
        ax.plot(positions, bilstm_p, "x--", color="tab:red", markersize=3,
                linewidth=1, label="BiLSTM point")

        ax.set_xlabel("Position")
        ax.set_ylabel("P(alpha)")
        ax.set_title(f"Sequence {ex} (length={length}) — Posterior Contraction")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylim(-0.05, 1.15)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "contraction.png"), dpi=150)
    plt.close(fig)

    # Also save individual sequence plots for slides (with legends)
    for ex in [0, 2]:  # sequences used on slide 11
        seq = test_seqs[ex]
        true_vals = np.asarray(test_posts[ex])
        samples = bf_samples[ex]
        length = len(true_vals)
        positions = np.arange(length)

        pred_median = np.array([np.nanmedian(samples[:, pos]) for pos in range(length)])
        pred_lo = np.array([np.nanpercentile(samples[:, pos], 5) for pos in range(length)])
        pred_hi = np.array([np.nanpercentile(samples[:, pos], 95) for pos in range(length)])

        fig, ax = plt.subplots(1, 1, figsize=(8, 3.5))
        ax.plot(positions, true_vals, "o-", color="k", markersize=5, linewidth=1.8,
                label="True $P(\\alpha)$ (FB)")
        ax.fill_between(positions, pred_lo, pred_hi, alpha=0.25, color="tab:blue",
                        label="BayesFlow 90\\% CI")
        ax.plot(positions, pred_median, "s-", color="tab:blue", markersize=4,
                linewidth=1.5, label="BayesFlow median")
        ax.plot(positions, bilstm_point[ex], "x--", color="tab:red", markersize=4,
                linewidth=1.5, label="BiLSTM point")
        ax.set_xlabel("Position")
        ax.set_ylabel("$P(\\alpha)$")
        ax.set_title(f"Sequence {ex} (length={length}) — Posterior Contraction")
        ax.legend(fontsize=9, loc="upper left", framealpha=0.8, edgecolor="gray")
        ax.set_ylim(-0.05, 1.15)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, f"contraction_seq{ex}.png"), dpi=180)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    from scipy.stats import pearsonr
    bf_corr, _ = pearsonr(true_flat, bf_flat)
    bilstm_corr, _ = pearsonr(true_flat, bilstm_flat)

    print("\n" + "=" * 50)
    print("  Diagnostic Summary")
    print("=" * 50)
    print(f"  Positions evaluated:        {n_positions}")
    print(f"  BayesFlow  MAE:             {np.mean(abs_err_bf):.4f}")
    print(f"  BiLSTM     MAE:             {np.mean(abs_err_bilstm):.4f}")
    print(f"  BayesFlow  correlation:     {bf_corr:.4f}")
    print(f"  BiLSTM     correlation:     {bilstm_corr:.4f}")
    print(f"\n  Plots saved to {PLOTS_DIR}/")
    print(f"    calibration.png, residuals.png, ecdf.png, scatter.png")
    print(f"    sbc.png, zscore_coverage.png, contraction.png")


if __name__ == "__main__":
    main()
