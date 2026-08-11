"""Joint (end-to-end) training pipeline for the BayesFlow neural posterior estimator.

Alternative to ``training.py``'s frozen two-stage design: BayesFlow calls the
(unfrozen) BiLSTM itself as a real ``bf.networks.SummaryNetwork`` (see
``architecture.create_joint_summary_network``), so gradients from the flow's
loss reach it directly, every step. Purely additive - reuses ``training.py``'s
dataset/padding/plotting helpers unchanged and doesn't touch the staged
pipeline's behavior.

Key differences from the staged pipeline:
- **One stage, one optimizer.** No pretrain phase, no precomputed
  ``sequence_summary`` array -- ``train_joint_workflow`` feeds raw
  ``observables`` straight to ``fit_offline``. One combined Keras model means
  one optimizer (``_build_joint_optimizer``), not the staged design's two
  independently-tuned ones.
- **Model selection still avoids the flow's own val_loss**, same reason as
  ``training.masked_point_accuracy`` (~44% of every target is an exact-zero
  point mass a flow can chase toward ``-inf`` loss). ``masked_point_accuracy_joint``
  mirrors that logic, sourcing its condition from raw ``observables``.
- **Smaller HP search space** than the staged pipeline's 6 configs: joint
  training pays the BiLSTM's cost on every step of every trial, whereas the
  staged search reuses cached summaries and pays it once.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from architecture import (
    BAYESFLOW_INFERENCE_VARIABLES,
    SequenceSummaryNetwork,
    _configure_bayesflow_environment,
    create_bayesflow_coupling_network,
    create_bayesflow_joint_adapter,
    create_joint_summary_network,
)
from training import (
    CHECKPOINT_DIR,
    PaddedDataset,
    _postprocess_alpha_samples,
    pad_to_fixed_length,
)

# --- Joint training config -------------------------------------------------------


@dataclass
class JointConfig:
    # BiLSTM architecture (same defaults as training.PretrainConfig).
    embedding_dim: int = 32
    hidden_dim: int = 64
    summary_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    # Coupling-flow architecture (same defaults as training.FlowConfig).
    depth: int = 6
    hidden_widths: tuple[int, ...] = (128, 128)
    # Single shared optimizer -- see _build_joint_optimizer's docstring for why
    # there is only one, unlike the staged design's two independently-tuned ones.
    optimizer_name: str = "adamw"  # "adam" or "adamw"
    initial_learning_rate: float = 5e-4
    weight_decay: float = 1e-3  # used only when optimizer_name == "adamw"
    epochs: int = 60
    batch_size: int = 64
    early_stopping_patience: int | None = None
    checkpoint_filepath: str | None = None
    save_best_only: bool = False
    seed: int = 0


def _build_joint_optimizer(cfg: JointConfig, n_train: int) -> Any:
    """AdamW/Adam with cosine-decay-warmup, mirroring
    ``training._build_flow_optimizer``. One optimizer for the whole joint
    model: ``fit_offline`` manages a single optimizer per Keras model, so
    BiLSTM and flow necessarily share one schedule here, unlike the staged
    design's two independently-tuned ones.
    """
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


def train_joint_workflow(
    train_data: PaddedDataset,
    val_data: PaddedDataset | None,
    cfg: JointConfig = JointConfig(),
    verbose: bool = True,
) -> tuple[Any, Any]:
    """Build and fit a BayesFlow ``BasicWorkflow`` end-to-end: BiLSTM and flow
    train together from raw padded sequences in one ``fit_offline`` call.
    Compare ``training.train_bayesflow_workflow``, which trains only the flow
    on frozen, precomputed summaries.
    """
    _configure_bayesflow_environment()
    import bayesflow as bf
    import keras

    keras.utils.set_random_seed(cfg.seed)

    if cfg.checkpoint_filepath is not None:
        os.makedirs(cfg.checkpoint_filepath, exist_ok=True)

    train_dict = {
        "observables": train_data.sequences.numpy(),
        "alpha_posteriors": train_data.targets.numpy(),
    }
    val_dict = None
    if val_data is not None:
        val_dict = {
            "observables": val_data.sequences.numpy(),
            "alpha_posteriors": val_data.targets.numpy(),
        }

    summary_network = create_joint_summary_network(
        embedding_dim=cfg.embedding_dim,
        hidden_dim=cfg.hidden_dim,
        summary_dim=cfg.summary_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )
    workflow = bf.BasicWorkflow(
        simulator=None,
        adapter=create_bayesflow_joint_adapter(),
        inference_network=create_bayesflow_coupling_network(depth=cfg.depth, hidden_widths=cfg.hidden_widths),
        summary_network=summary_network,
        optimizer=_build_joint_optimizer(cfg, train_data.n),
        checkpoint_filepath=cfg.checkpoint_filepath,
        checkpoint_name="model",
        save_best_only=cfg.save_best_only,
        inference_variables=BAYESFLOW_INFERENCE_VARIABLES,
    )

    # No ReduceLROnPlateau: the LR is a CosineDecay schedule, which Keras 3
    # can't overwrite with a plain float (same as train_bayesflow_workflow).
    monitor = "val_loss" if val_dict is not None else "loss"
    callbacks = []
    if cfg.early_stopping_patience is not None:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=monitor, patience=cfg.early_stopping_patience, restore_best_weights=True
            )
        )

    history = workflow.fit_offline(
        data=train_dict,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        validation_data=val_dict,
        callbacks=callbacks,
        verbose=1 if verbose else 0,
    )
    return workflow, history


# --- Masked accuracy: same rationale as training.masked_point_accuracy -----------


def masked_point_accuracy_joint(
    workflow: Any,
    val_dataset: PaddedDataset,
    num_samples: int = 50,
    max_eval: int = 80,
    seed: int = 0,
) -> tuple[float, float]:
    """Masked MAE/correlation vs. true Forward-Backward posteriors, real
    positions only -- joint counterpart of ``training.masked_point_accuracy``
    (see there for why the flow's own ``val_loss`` isn't trustworthy). Same
    logic, condition sourced from raw ``observables`` instead of a precomputed
    ``sequence_summary``.
    """
    n = min(max_eval, val_dataset.n)
    conditions = {"observables": val_dataset.sequences[:n].numpy()}
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


# --- Hyperparameter search --------------------------------------------------------


@dataclass
class JointHPSearchResult:
    results: pd.DataFrame
    best_config: JointConfig
    best_masked_mae: float
    best_masked_corr: float
    histories: dict[str, dict[str, list[float]]] = field(default_factory=dict)


def default_joint_hp_search_space() -> list[dict[str, Any]]:
    """Smaller than ``training.default_hp_search_space``'s 6 configs -- see
    module docstring: joint training pays LSTM cost every step of every
    trial, not once.
    """
    return [
        {"name": "baseline", "optimizer_name": "adamw", "initial_learning_rate": 5e-4, "weight_decay": 1e-3, "depth": 6},
        {"name": "higher_lr", "optimizer_name": "adamw", "initial_learning_rate": 1e-3, "weight_decay": 1e-3, "depth": 6},
        {"name": "deeper_flow", "optimizer_name": "adamw", "initial_learning_rate": 5e-4, "weight_decay": 1e-3, "depth": 10},
        {"name": "stronger_reg", "optimizer_name": "adamw", "initial_learning_rate": 5e-4, "weight_decay": 5e-3, "depth": 6},
    ]


def run_joint_hyperparameter_search(
    train_data: PaddedDataset,
    val_data: PaddedDataset,
    search_epochs: int = 15,
    batch_size: int = 64,
    search_space: list[dict[str, Any]] | None = None,
    accuracy_num_samples: int = 50,
    accuracy_max_eval: int = 80,
    verbose: bool = True,
) -> JointHPSearchResult:
    """Train each config in ``search_space`` briefly and rank by masked accuracy
    against true Forward-Backward posteriors, same rationale as
    ``training.run_hyperparameter_search``.
    """
    search_space = search_space if search_space is not None else default_joint_hp_search_space()

    rows = []
    histories: dict[str, dict[str, list[float]]] = {}
    best_mae = float("inf")
    best_corr = float("nan")
    best_config: JointConfig | None = None

    for spec in search_space:
        name = spec["name"]
        cfg = JointConfig(
            depth=spec["depth"],
            hidden_widths=spec.get("hidden_widths", (128, 128)),
            optimizer_name=spec["optimizer_name"],
            initial_learning_rate=spec["initial_learning_rate"],
            weight_decay=spec["weight_decay"],
            epochs=search_epochs,
            batch_size=batch_size,
        )
        if verbose:
            print(f"[joint-hp-search] '{name}': {spec}")
        workflow, history = train_joint_workflow(train_data, val_data, cfg, verbose=False)
        h = history.history
        final_val_loss = h["val_loss"][-1] if "val_loss" in h else h["loss"][-1]
        masked_mae, masked_corr = masked_point_accuracy_joint(
            workflow, val_data, num_samples=accuracy_num_samples, max_eval=accuracy_max_eval
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
    return JointHPSearchResult(
        results=results, best_config=best_config, best_masked_mae=best_mae, best_masked_corr=best_corr,
        histories=histories,
    )


# --- Saving / loading artifacts ---------------------------------------------------


def save_joint_artifacts(manifest: dict[str, Any], checkpoint_dir: str = CHECKPOINT_DIR) -> None:
    """Save the joint-training manifest. Unlike ``training.save_artifacts``,
    no separate BiLSTM ``state_dict``: its weights live in ``model.keras``,
    written by ``BasicWorkflow``'s own checkpointing during ``fit_offline``.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def load_joint_pipeline(checkpoint_dir: str) -> tuple[SequenceSummaryNetwork, Any, dict[str, Any]]:
    """Load a jointly-trained checkpoint for inference.

    Returns ``(summary_model, approximator, manifest)`` -- same shape as
    ``training.load_trained_pipeline``, so existing callers (``diagnostics.py``,
    ``validate_insulin.py``) need almost no changes. ``summary_model`` is the
    trained inner ``SequenceSummaryNetwork`` pulled out of the loaded
    ``JointSequenceSummaryNetwork``.
    """
    _configure_bayesflow_environment()
    import keras
    import bayesflow  # noqa: F401  (import needed so keras.saving can find bayesflow's classes)

    create_joint_summary_network()  # registers JointSequenceSummaryNetwork with keras.saving; instance discarded

    with open(os.path.join(checkpoint_dir, "manifest.json")) as f:
        manifest = json.load(f)

    approximator = keras.saving.load_model(os.path.join(checkpoint_dir, "model.keras"))
    summary_model = approximator.summary_network.wrapped.module
    summary_model.eval()
    return summary_model, approximator, manifest


def predict_alpha_posterior_joint(
    sequences,
    approximator: Any,
    max_len: int,
    num_samples: int = 300,
    seed: int | None = 0,
    warn_threshold: float = 0.05,
) -> list[np.ndarray]:
    """Joint counterpart of ``training.predict_alpha_posterior`` -- same
    NaN/Inf/clip handling (see its docstring). Condition is raw padded
    ``observables``; the approximator runs its own trained summary network.
    """
    padded, _mask, lengths = pad_to_fixed_length(sequences, max_len)

    raw = np.asarray(
        approximator.sample(num_samples=num_samples, conditions={"observables": padded.numpy()}, seed=seed)[
            "alpha_posteriors"
        ]
    )  # (n_sequences, num_samples, max_len)

    return _postprocess_alpha_samples(raw, lengths, warn_threshold, caller_name="predict_alpha_posterior_joint")
