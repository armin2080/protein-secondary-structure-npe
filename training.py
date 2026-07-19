"""Training pipeline for the BayesFlow neural posterior estimator (Task 3).

This module handles these tasks:
- build a fixed-size, padded train/validation dataset from the Task 1 simulator
  and Forward-Backward ground truth;
- pretrain the Task 2 ``SequenceSummaryNetwork`` (BiLSTM) on the masked BCE
  objective, with an optimizer, an LR scheduler, and regularization;
- freeze that network and use it to compute fixed-size sequence summaries;
- train the Task 2 BayesFlow coupling-flow inference network on those frozen
  summaries via ``BasicWorkflow.fit_offline``;
- run a small automated hyperparameter search over the coupling-flow
  optimizer/learning-rate/depth/weight-decay, then retrain the best config for
  longer;
- produce loss-trajectory diagnostics plots and save reusable artifacts.


- **Two-stage training, not end-to-end.** ``prepare_bayesflow_batch`` in
  ``architecture.py`` computes ``sequence_summary`` under ``torch.no_grad()``,
  so the BiLSTM is not meant to be differentiated through BayesFlow's loss.
  We pretrain it first on the direct per-position objective, freeze it, then
  train the coupling flow on top of its frozen output.
- **Fixed global padding length.** ``CouplingFlow``'s parameter dimensionality
  is fixed the first time it is built, so every batch fed to it must have the
  same width. All sequences (train, val, and later insulin in Task 4) must be
  padded to one shared ``max_len`` chosen up front -- this module always pads
  to a fixed length rather than to each batch's own maximum, unlike
  ``architecture.pad_sequences``/``pad_targets``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from architecture import (
    PAD_IDX,
    BAYESFLOW_INFERENCE_CONDITIONS,
    BAYESFLOW_INFERENCE_VARIABLES,
    SequenceSummaryNetwork,
    _configure_bayesflow_environment,
    create_bayesflow_adapter,
    create_bayesflow_coupling_network,
    masked_bce_loss,
)
from forward_backward import generate_dataset
from simulator import get_rng

DEFAULT_MIN_LEN = 10
DEFAULT_MAX_LEN = 60  # global padding length; Task 4 must pad insulin to the same width

CHECKPOINT_DIR = "checkpoints"
PLOTS_DIR = "plots"
RESULTS_DIR = "results"


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# --- Fixed-length padding -----------------------------------------------------
# architecture.pad_sequences/pad_targets pad to the given batch's own maximum
# length, which is fine for Task 2's shape checks but not for this module: the
# coupling flow needs the *same* width across every batch and across datasets
# built at different times (training vs. later insulin inference).


@dataclass
class PaddedDataset:
    """A dataset padded to one fixed length, ready for mini-batching."""

    sequences: torch.Tensor  # (N, max_len) long
    mask: torch.Tensor  # (N, max_len) bool
    lengths: torch.Tensor  # (N,) long
    targets: torch.Tensor  # (N, max_len) float32, Forward-Backward alpha posteriors
    max_len: int

    @property
    def n(self) -> int:
        return self.sequences.shape[0]


def pad_to_fixed_length(
    sequences,
    max_len: int,
    pad_value: int = PAD_IDX,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad every sequence to exactly ``max_len`` (not the batch's own max)."""
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    if torch.any(lengths > max_len):
        raise ValueError(f"a sequence has length {int(lengths.max())} > max_len={max_len}")
    if torch.any(lengths <= 0):
        raise ValueError("all sequences must have positive length")

    padded = torch.full((len(sequences), max_len), pad_value, dtype=torch.long)
    mask = torch.zeros((len(sequences), max_len), dtype=torch.bool)
    for i, sequence in enumerate(sequences):
        length = int(lengths[i].item())
        padded[i, :length] = torch.as_tensor(sequence, dtype=torch.long)
        mask[i, :length] = True
    return padded, mask, lengths


def pad_targets_to_fixed_length(targets, max_len: int, pad_value: float = 0.0) -> torch.Tensor:
    """Pad per-position targets to exactly ``max_len``."""
    padded = torch.full((len(targets), max_len), pad_value, dtype=torch.float32)
    for i, target in enumerate(targets):
        target_tensor = torch.as_tensor(target, dtype=torch.float32)
        padded[i, : len(target_tensor)] = target_tensor
    return padded


def build_datasets(
    n_train: int = 2000,
    n_val: int = 400,
    min_len: int = DEFAULT_MIN_LEN,
    max_len: int = DEFAULT_MAX_LEN,
    seed_train: int = 10,
    seed_val: int = 11,
) -> tuple[PaddedDataset, PaddedDataset]:
    """Simulate train/val sequence-posterior pairs and pad both to ``max_len``."""
    train_sequences, train_posteriors, _ = generate_dataset(
        n_train, get_rng(seed_train), min_len=min_len, max_len=max_len
    )
    val_sequences, val_posteriors, _ = generate_dataset(
        n_val, get_rng(seed_val), min_len=min_len, max_len=max_len
    )

    train_padded, train_mask, train_lengths = pad_to_fixed_length(train_sequences, max_len)
    train_targets = pad_targets_to_fixed_length(train_posteriors, max_len)
    val_padded, val_mask, val_lengths = pad_to_fixed_length(val_sequences, max_len)
    val_targets = pad_targets_to_fixed_length(val_posteriors, max_len)

    train_data = PaddedDataset(train_padded, train_mask, train_lengths, train_targets, max_len)
    val_data = PaddedDataset(val_padded, val_mask, val_lengths, val_targets, max_len)
    return train_data, val_data


# --- Stage 1: pretrain the BiLSTM summary network -----------------------------


@dataclass
class PretrainConfig:
    embedding_dim: int = 32
    hidden_dim: int = 64
    summary_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 60
    batch_size: int = 64
    patience: int = 8
    grad_clip: float = 1.0
    aux_weight: float = 0.5
    seed: int = 0


def _evaluate_pretrain(model: SequenceSummaryNetwork, data: PaddedDataset, device: str) -> float:
    model.eval()
    with torch.no_grad():
        _summary, logits = model(
            data.sequences.to(device), mask=data.mask.to(device), lengths=data.lengths.to(device)
        )
        loss = masked_bce_loss(logits, data.targets.to(device), data.mask.to(device))
    return loss.item()


def pretrain_summary_network(
    train: PaddedDataset,
    val: PaddedDataset,
    cfg: PretrainConfig = PretrainConfig(),
    device: str | None = None,
    verbose: bool = True,
) -> tuple[SequenceSummaryNetwork, dict[str, list[float]]]:
    """Train the BiLSTM on masked BCE against Forward-Backward posteriors.

    Uses AdamW (weight decay = regularization), gradient clipping,
    ReduceLROnPlateau (LR scheduling), and early stopping on validation loss
    (restoring the best-validation-loss weights at the end).

    Also trains ``summary_head`` (previously dead weight -- its output was
    never used by the loss, so ``summary_head[0].weight.grad`` was literally
    ``None`` after every backward call; verified directly). A throwaway
    auxiliary head predicts each sequence's mean alpha-helix fraction over
    real positions from the pooled ``summary`` vector; its MSE loss (weighted
    by ``cfg.aux_weight``) is added to the main masked BCE loss. The auxiliary
    head itself is discarded after training - only its gradient contribution
    to ``summary_head`` matters, since ``summary_head``'s output is what later
    becomes part of the BayesFlow condition (see ``compute_sequence_summaries``).
    """
    device = device or get_device()
    torch.manual_seed(cfg.seed)

    model = SequenceSummaryNetwork(
        embedding_dim=cfg.embedding_dim,
        hidden_dim=cfg.hidden_dim,
        summary_dim=cfg.summary_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        bidirectional=True,
    ).to(device)
    aux_head = nn.Linear(cfg.summary_dim, 1).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(aux_head.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(train.n)
        running_loss = 0.0
        for start in range(0, train.n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            optimizer.zero_grad()
            batch_mask = train.mask[idx].to(device)
            summary, logits = model(
                train.sequences[idx].to(device),
                mask=batch_mask,
                lengths=train.lengths[idx].to(device),
            )
            batch_targets = train.targets[idx].to(device)
            bce = masked_bce_loss(logits, batch_targets, batch_mask)

            mask_float = batch_mask.to(batch_targets.dtype)
            true_fraction = (batch_targets * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1.0)
            pred_fraction = aux_head(summary).squeeze(-1)
            aux = nn.functional.mse_loss(pred_fraction, true_fraction)

            loss = bce + cfg.aux_weight * aux
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.parameters()) + list(aux_head.parameters()), cfg.grad_clip)
            optimizer.step()
            running_loss += bce.item() * len(idx)
        train_loss = running_loss / train.n

        val_loss = _evaluate_pretrain(model, val, device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose:
            print(
                f"  [pretrain] epoch {epoch + 1:3d}/{cfg.epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  lr={current_lr:.2e}"
            )

        if epochs_without_improvement >= cfg.patience:
            if verbose:
                print(f"  [pretrain] early stopping at epoch {epoch + 1} (patience={cfg.patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


def compute_sequence_summaries(
    model: SequenceSummaryNetwork, data: PaddedDataset, device: str | None = None
) -> np.ndarray:
    """Frozen forward pass producing the BayesFlow conditioning vector.

    Returns the pooled ``summary_head`` vector concatenated with the
    per-position logits (masked to zero on padding), not the pooled vector
    alone. This matters: masked mean pooling collapses the whole sequence
    into one vector and is positionally destructive - it can encode
    "how alpha-helix-heavy is this sequence overall" but not "*which*
    position is alpha-helix", because averaging discards where in the
    sequence each contribution came from. Conditioning the coupling flow on
    the pooled vector alone was verified empirically to perform *worse* than
    a baseline that ignores the sequence entirely (pooled correlation with
    the true Forward-Backward posterior ~0.17 on held-out data, vs ~0.20 for
    a sequence-blind marginal-average baseline). The per-position logits
    (from the same BiLSTM, via ``masked_bce_loss`` pretraining) are
    near-perfect on their own (~0.999 correlation) precisely because they
    never pass through the pooling bottleneck; concatenating them restores
    positional information to the flow's condition and reaches ~0.81
    correlation on held-out data.

    Raw logits are used here, not ``sigmoid(logits)`` probabilities:
    ``CouplingFlow``'s base distribution is an unconstrained Gaussian, and
    concatenating a *second* signal confined to [0, 1] alongside the
    already-[0, 1]-bounded ``alpha_posteriors`` target caused severe
    training instability (samples with magnitude in the tens on held-out
    data). See ``SECTION3_NOTES.md`` for the full investigation.

    The pooled half of this vector is produced by ``summary_head``, which
    ``pretrain_summary_network`` now trains via an auxiliary mean-alpha-fraction
    loss (see its docstring) -- previously ``summary_head`` received no
    gradient anywhere in the pipeline (Stage 1 only backpropagated through
    ``position_head``; Stage 2 runs this function under ``torch.no_grad()``),
    so the pooled half of the condition was a random, untrained projection.
    """
    device = device or get_device()
    return _bayesflow_condition(model, data.sequences, data.mask, data.lengths, device)


def _bayesflow_condition(
    model: SequenceSummaryNetwork,
    padded_sequences: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    device: str,
) -> np.ndarray:
    """Shared implementation behind ``compute_sequence_summaries`` and
    ``predict_alpha_posterior``, so both build the condition identically."""
    model.eval()
    with torch.no_grad():
        pooled, logits = model(
            padded_sequences.to(device),
            mask=mask.to(device),
            lengths=lengths.to(device),
            return_position_logits=True,
        )
        logits = logits * mask.to(device).to(logits.dtype)
    combined = torch.cat([pooled, logits], dim=-1)
    return combined.cpu().numpy().astype(np.float32)


def build_bayesflow_data(
    model: SequenceSummaryNetwork, data: PaddedDataset, device: str | None = None
) -> dict[str, np.ndarray]:
    """Offline data dict for ``BasicWorkflow.fit_offline``: frozen summaries + targets."""
    summary = compute_sequence_summaries(model, data, device=device)
    alpha_posteriors = data.targets.numpy().astype(np.float32)
    return {"alpha_posteriors": alpha_posteriors, "sequence_summary": summary}


# --- Stage 2: train the BayesFlow coupling flow --------------------------------


@dataclass
class FlowConfig:
    depth: int = 6
    hidden_widths: tuple[int, ...] = (128, 128)
    optimizer_name: str = "adamw"  # "adam" or "adamw"
    initial_learning_rate: float = 5e-4
    weight_decay: float = 1e-3  # used only when optimizer_name == "adamw"
    epochs: int = 60
    batch_size: int = 64
    early_stopping_patience: int | None = None
    checkpoint_filepath: str | None = None
    # False, not True: Keras's "best" checkpoint (and EarlyStopping's
    # restore_best_weights) both monitor val_loss, which is not trustworthy for
    # this problem - see masked_point_accuracy. save_best_only=True would
    # preferentially keep whichever epoch happened to have the most negative
    # (possibly numerically unstable) val_loss rather than the most accurate one.
    # Always saves the last epoch's weights; evaluate with masked_point_accuracy
    # after training, not by trusting which checkpoint Keras thought was "best".
    save_best_only: bool = False
    seed: int = 0


def _build_flow_optimizer(cfg: FlowConfig, n_train: int) -> Any:
    """Adam/AdamW with a cosine-decay-with-warmup schedule (mirrors BayesFlow's
    own default, but with a configurable optimizer choice and weight decay)."""
    _configure_bayesflow_environment()
    import keras

    num_batches = max(1, n_train // cfg.batch_size)
    total_steps = max(1, cfg.epochs * num_batches)
    warmup_steps = max(1, int(0.05 * total_steps))
    decay_steps = max(1, total_steps - warmup_steps)

    schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.1 * cfg.initial_learning_rate,
        warmup_target=cfg.initial_learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        alpha=0.0,
    )
    if cfg.optimizer_name == "adam":
        return keras.optimizers.Adam(schedule, clipnorm=1.5)
    if cfg.optimizer_name == "adamw":
        return keras.optimizers.AdamW(schedule, weight_decay=cfg.weight_decay, clipnorm=1.5)
    raise ValueError(f"unknown optimizer_name {cfg.optimizer_name!r}")


def train_bayesflow_workflow(
    train_data: dict[str, np.ndarray],
    val_data: dict[str, np.ndarray] | None,
    cfg: FlowConfig = FlowConfig(),
    verbose: bool = True,
) -> tuple[Any, Any]:
    """Build and fit a BayesFlow ``BasicWorkflow`` on precomputed offline data.

    ``train_data``/``val_data`` are dicts with ``alpha_posteriors`` and
    ``sequence_summary`` keys, as produced by ``build_bayesflow_data``.
    """
    _configure_bayesflow_environment()
    import bayesflow as bf
    import keras

    keras.utils.set_random_seed(cfg.seed)

    if cfg.checkpoint_filepath is not None:
        os.makedirs(cfg.checkpoint_filepath, exist_ok=True)

    n_train = train_data["alpha_posteriors"].shape[0]
    workflow = bf.BasicWorkflow(
        simulator=None,
        adapter=create_bayesflow_adapter(),
        inference_network=create_bayesflow_coupling_network(depth=cfg.depth, hidden_widths=cfg.hidden_widths),
        summary_network=None,
        optimizer=_build_flow_optimizer(cfg, n_train),
        checkpoint_filepath=cfg.checkpoint_filepath,
        checkpoint_name="model",
        save_best_only=cfg.save_best_only,
        inference_variables=BAYESFLOW_INFERENCE_VARIABLES,
        inference_conditions=BAYESFLOW_INFERENCE_CONDITIONS,
    )

    # No ReduceLROnPlateau here: the optimizer's learning rate is a CosineDecay
    # *schedule* (see _build_flow_optimizer), and Keras 3 cannot overwrite a
    # schedule-based learning rate with a plain float, which is what
    # ReduceLROnPlateau tries to do the moment it detects a plateau. The
    # warmup+cosine-decay schedule is this stage's LR scheduler.
    monitor = "val_loss" if val_data is not None else "loss"
    callbacks = []
    if cfg.early_stopping_patience is not None:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=monitor, patience=cfg.early_stopping_patience, restore_best_weights=True
            )
        )

    history = workflow.fit_offline(
        data=train_data,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        validation_data=val_data,
        callbacks=callbacks,
        verbose=1 if verbose else 0,
    )
    return workflow, history


# --- Masked accuracy: the real ranking/selection metric --------------------------


def masked_point_accuracy(
    workflow: Any,
    val_dataset: "PaddedDataset",
    val_bf: dict[str, np.ndarray],
    num_samples: int = 50,
    max_eval: int = 80,
    seed: int = 0,
) -> tuple[float, float]:
    """Masked MAE/correlation against true Forward-Backward posteriors, on real
    (non-padded) positions only. This -- not the flow's own ``val_loss`` -- is
    what hyperparameter search and final-config selection use.

    Why not ``val_loss``: every padded ``alpha_posteriors`` target has position 0
    exactly 0.0 (sequences deterministically start in "other") plus however many
    padding positions past the sequence's real length -- about 44% of every
    training target on average. ``CouplingFlow``'s continuous density cannot
    represent that exact point mass, so its negative log-likelihood is unbounded
    below: a config can drive ``val_loss`` toward -inf simply by learning an
    extremely (and numerically unstable) peaked density around the padding
    zeros, which has nothing to do with accuracy on real positions. Verified
    empirically: at the search's own epoch budget, the config with the most
    negative ``val_loss`` produced 0.35% NaN / 0.09% Inf samples (magnitudes up
    to 1e37) and much worse real-position error than a config ``val_loss`` ranked
    far behind it. ``val_data.targets`` is already the exact Forward-Backward
    ground truth from Section 1 (no extra simulation needed here).

    Uses the median across samples per position, not the mean: the flow can
    still occasionally produce an extreme individual draw even from a
    well-behaved config (see ``predict_alpha_posterior``), and the median is far
    more robust to that than the mean.
    """
    n = min(max_eval, val_dataset.n)
    conditions = {"sequence_summary": val_bf["sequence_summary"][:n]}
    samples = np.asarray(
        workflow.approximator.sample(num_samples=num_samples, conditions=conditions, seed=seed)["alpha_posteriors"]
    )  # (n, num_samples, max_len)

    all_true, all_pred = [], []
    for i in range(n):
        length = int(val_dataset.lengths[i])
        real = samples[i, :, :length]
        pred = np.nanmedian(np.where(np.isfinite(real), real, np.nan), axis=0)
        all_true.append(val_dataset.targets[i, :length].numpy())
        all_pred.append(np.clip(pred, 0.0, 1.0))
    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)
    mae = float(np.abs(all_true - all_pred).mean())
    corr = float(np.corrcoef(all_true, all_pred)[0, 1])
    return mae, corr


# --- Hyperparameter search ------------------------------------------------------


@dataclass
class HPSearchResult:
    results: pd.DataFrame
    best_config: FlowConfig
    best_masked_mae: float
    best_masked_corr: float
    histories: dict[str, dict[str, list[float]]] = field(default_factory=dict)


def default_hp_search_space() -> list[dict[str, Any]]:
    """A curated (not exhaustive) set of configs spanning optimizer choice,
    learning rate, coupling-flow depth, and weight decay (regularization)."""
    return [
        {"name": "baseline", "optimizer_name": "adamw", "initial_learning_rate": 5e-4, "weight_decay": 1e-3, "depth": 6},
        {"name": "higher_lr", "optimizer_name": "adamw", "initial_learning_rate": 1e-3, "weight_decay": 1e-3, "depth": 6},
        {"name": "lower_lr", "optimizer_name": "adamw", "initial_learning_rate": 2e-4, "weight_decay": 1e-3, "depth": 6},
        {"name": "deeper_flow", "optimizer_name": "adamw", "initial_learning_rate": 5e-4, "weight_decay": 1e-3, "depth": 10},
        {"name": "stronger_reg", "optimizer_name": "adamw", "initial_learning_rate": 5e-4, "weight_decay": 5e-3, "depth": 6},
        {"name": "adam_no_decay", "optimizer_name": "adam", "initial_learning_rate": 5e-4, "weight_decay": 0.0, "depth": 6},
    ]


def run_hyperparameter_search(
    train_data: dict[str, np.ndarray],
    val_data: dict[str, np.ndarray],
    val_dataset: "PaddedDataset",
    search_epochs: int = 15,
    batch_size: int = 64,
    search_space: list[dict[str, Any]] | None = None,
    accuracy_num_samples: int = 50,
    accuracy_max_eval: int = 80,
    verbose: bool = True,
) -> HPSearchResult:
    """Train each config in ``search_space`` briefly and rank by masked accuracy
    against true Forward-Backward posteriors on real positions (``val_dataset``),
    not by the flow's own ``val_loss``. See ``masked_point_accuracy`` for why
    ``val_loss`` is not a trustworthy ranking signal for this problem: a config
    can score a very negative ``val_loss`` by overfitting the ~44%-of-every-target
    padding point mass, which is unrelated to (and was measured to be inversely
    related to) real predictive accuracy.
    """
    search_space = search_space if search_space is not None else default_hp_search_space()

    rows = []
    histories: dict[str, dict[str, list[float]]] = {}
    best_mae = float("inf")
    best_corr = float("nan")
    best_config: FlowConfig | None = None

    for spec in search_space:
        name = spec["name"]
        cfg = FlowConfig(
            depth=spec["depth"],
            hidden_widths=spec.get("hidden_widths", (128, 128)),
            optimizer_name=spec["optimizer_name"],
            initial_learning_rate=spec["initial_learning_rate"],
            weight_decay=spec["weight_decay"],
            epochs=search_epochs,
            batch_size=batch_size,
        )
        if verbose:
            print(f"[hp-search] '{name}': {spec}")
        workflow, history = train_bayesflow_workflow(train_data, val_data, cfg, verbose=False)
        h = history.history
        final_val_loss = h["val_loss"][-1] if "val_loss" in h else h["loss"][-1]
        masked_mae, masked_corr = masked_point_accuracy(
            workflow, val_dataset, val_data, num_samples=accuracy_num_samples, max_eval=accuracy_max_eval
        )

        rows.append(
            {
                "name": name,
                "optimizer_name": cfg.optimizer_name,
                "initial_learning_rate": cfg.initial_learning_rate,
                "weight_decay": cfg.weight_decay,
                "depth": cfg.depth,
                "final_train_loss": h["loss"][-1],
                "final_val_loss": final_val_loss,
                "masked_mae": masked_mae,
                "masked_corr": masked_corr,
            }
        )
        histories[name] = h
        if verbose:
            print(
                f"           final_val_loss={final_val_loss:.4f}  "
                f"masked_mae={masked_mae:.4f}  masked_corr={masked_corr:.4f}"
            )

        if masked_mae < best_mae:
            best_mae = masked_mae
            best_corr = masked_corr
            best_config = cfg

    results = pd.DataFrame(rows).sort_values("masked_mae").reset_index(drop=True)
    assert best_config is not None
    return HPSearchResult(
        results=results, best_config=best_config, best_masked_mae=best_mae, best_masked_corr=best_corr,
        histories=histories,
    )


# --- Diagnostics: loss-trajectory plots -----------------------------------------


def plot_pretrain_loss(history: dict[str, list[float]], out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="train")
    ax.plot(epochs, history["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("masked BCE loss")
    ax.set_title("Summary network pretraining loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_bayesflow_loss(history: Any, out_path: str) -> None:
    """Save the BayesFlow coupling-flow train/val loss trajectory.

    ``history`` is the ``keras.callbacks.History`` returned by
    ``train_bayesflow_workflow``/``workflow.fit_offline``.
    """
    _configure_bayesflow_environment()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from bayesflow.diagnostics import plots as bf_plots

    fig = bf_plots.loss(history)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_hyperparameter_search(results: pd.DataFrame, out_path: str) -> None:
    """Bar chart of masked MAE per config (the selection metric) -- not
    final_val_loss, which is retained in the results table for transparency but
    is not trustworthy for ranking (see masked_point_accuracy)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(results["name"], results["masked_mae"], color="#4C72B0")
    ax.set_ylabel("masked MAE vs. true Forward-Backward")
    ax.set_title("Coupling-flow hyperparameter search (lower is better)")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --- Saving artifacts for Task 4 -------------------------------------------------


def save_artifacts(
    summary_model: SequenceSummaryNetwork,
    manifest: dict[str, Any],
    checkpoint_dir: str = CHECKPOINT_DIR,
) -> None:
    """Save the frozen BiLSTM weights and a manifest describing the run.

    The BayesFlow approximator itself is saved separately by
    ``BasicWorkflow``'s own ``checkpoint_filepath``/``ModelCheckpoint`` during
    ``fit_offline`` (as ``<checkpoint_dir>/model.keras``), so only the BiLSTM
    weights and the manifest need saving here. Task 4 should load
    ``model.keras`` with ``keras.saving.load_model`` and ``summary_network.pt``
    with ``SequenceSummaryNetwork`` + ``load_state_dict``, using the recorded
    ``max_len``/hyperparameters in ``manifest.json`` to reconstruct the exact
    architecture before loading weights.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(summary_model.state_dict(), os.path.join(checkpoint_dir, "summary_network.pt"))
    with open(os.path.join(checkpoint_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)


# --- Loading + inference for Task 4 ---------------------------------------------


def load_trained_pipeline(checkpoint_dir: str = CHECKPOINT_DIR) -> tuple[SequenceSummaryNetwork, Any, dict[str, Any]]:
    """Load the frozen BiLSTM and trained BayesFlow approximator for inference.

    Reconstructs ``SequenceSummaryNetwork`` from ``manifest.json``'s
    ``pretrain_config`` and loads its weights, then loads the trained
    BayesFlow approximator from ``model.keras``. Returns
    ``(summary_model, approximator, manifest)``.

    Gotcha handled here: ``bayesflow`` must be imported *before*
    ``keras.saving.load_model``, or Keras cannot find BayesFlow's custom
    classes (``ContinuousApproximator``, ``CouplingFlow``, ``Adapter``, ...)
    and raises a confusing ``TypeError: Could not locate class ...``.
    """
    _configure_bayesflow_environment()
    import keras
    import bayesflow  # noqa: F401  (import needed so keras.saving can find bayesflow's classes)

    with open(os.path.join(checkpoint_dir, "manifest.json")) as f:
        manifest = json.load(f)
    pc = manifest["pretrain_config"]

    summary_model = SequenceSummaryNetwork(
        embedding_dim=pc["embedding_dim"],
        hidden_dim=pc["hidden_dim"],
        summary_dim=pc["summary_dim"],
        num_layers=pc["num_layers"],
        dropout=pc["dropout"],
        bidirectional=True,
    )
    summary_model.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, "summary_network.pt"), weights_only=True)
    )
    summary_model.eval()

    approximator = keras.saving.load_model(os.path.join(checkpoint_dir, "model.keras"))
    return summary_model, approximator, manifest


def predict_alpha_posterior(
    sequences,
    summary_model: SequenceSummaryNetwork,
    approximator: Any,
    max_len: int,
    num_samples: int = 300,
    device: str | None = None,
    seed: int | None = 0,
    warn_threshold: float = 0.05,
) -> list[np.ndarray]:
    """Predict P(alpha) samples for new sequences using the trained pipeline.

    ``sequences`` is a list of variable-length integer-encoded amino-acid
    arrays (e.g. from ``forward_backward.load_dataset`` or ``insulin.py``).
    ``max_len`` must match the value the model was trained with
    (``manifest["max_len"]``) - the coupling flow's dimensionality is fixed
    at that width, so every sequence must have length <= max_len, including
    insulin's 21/29-aa chains.

    Returns one array per input sequence, shape ``(num_samples, length)``.
    **Use ``np.nanmedian(pred, axis=0)`` for a point estimate, not ``.mean()``.**
    The coupling flow's base distribution is an unconstrained Gaussian, and it
    can occasionally produce individual draws far outside a valid probability.
    Measured directly on the shipped checkpoint (not a hypothetical): on
    held-out sequences, out-of-``[0, 1]``-but-finite draws ranged from ~1% up to
    34% of samples for a single sequence, with finite magnitudes as large as
    ~1e37; on one sequence, 1.9% of draws were outright NaN. This is *not* rare
    tail behavior - it is common enough that a caller using ``.mean()``, or
    treating raw samples as a clean posterior for SBC/ECDF/z-score/contraction
    diagnostics without checking this first, will get badly misleading results.

    To keep this usable without silently hiding the problem: NaN/Inf draws are
    preserved as ``np.nan`` (not collapsed into fake 0/1 atoms), and finite
    draws are clipped to ``[0, 1]``. If more than ``warn_threshold`` (default
    5%) of a sequence's draws were non-finite or needed clipping, a warning is
    printed identifying which sequence and the exact fraction - do not ignore
    it. If you need genuinely clean samples for calibration work, drop the
    NaNs (``pred[:, np.isfinite(pred).all(axis=0)]`` is wrong; you likely want
    to discard individual NaN draws per-position, e.g. via ``np.nanmedian``/
    ``np.nanpercentile`` throughout rather than plain ``np.mean``/``np.std``).
    """
    device = device or get_device()
    padded, mask, lengths = pad_to_fixed_length(sequences, max_len)
    condition = _bayesflow_condition(summary_model, padded, mask, lengths, device)

    raw = np.asarray(
        approximator.sample(num_samples=num_samples, conditions={"sequence_summary": condition}, seed=seed)[
            "alpha_posteriors"
        ]
    )  # (n_sequences, num_samples, max_len)

    predictions = []
    for i in range(len(sequences)):
        length = int(lengths[i])
        real = raw[i, :, :length]
        nonfinite = ~np.isfinite(real)
        clipped_needed = np.isfinite(real) & ((real < 0.0) | (real > 1.0))
        bad_frac = (nonfinite | clipped_needed).mean()
        if bad_frac > warn_threshold:
            print(
                f"WARNING predict_alpha_posterior: sequence {i} (length {length}) has "
                f"{100 * nonfinite.mean():.1f}% non-finite and {100 * clipped_needed.mean():.1f}% "
                f"out-of-[0,1]-but-finite samples out of {num_samples}. Point estimates via "
                f"np.nanmedian are still reasonable; raw samples are not a clean posterior for "
                f"calibration diagnostics without accounting for this."
            )
        pred = np.where(nonfinite, np.nan, np.clip(real, 0.0, 1.0))
        predictions.append(pred)
    return predictions
