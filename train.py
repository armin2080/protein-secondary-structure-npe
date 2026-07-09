"""End-to-end Part 3 training pipeline.

Run with:
    python train.py

Builds a real-sized dataset, pretrains the BiLSTM summary network, runs an
automated hyperparameter search over the BayesFlow coupling flow, retrains the
best configuration for longer, and saves diagnostics plots, a hyperparameter
results table, and reusable checkpoints/manifest for Task 4.

This is not a quick check -- see ``check_training_pipeline.py`` for that. On a
CPU this run takes tens of minutes.
"""

import os
import time
from dataclasses import asdict, replace

from training import (
    CHECKPOINT_DIR,
    DEFAULT_MAX_LEN,
    DEFAULT_MIN_LEN,
    PLOTS_DIR,
    RESULTS_DIR,
    FlowConfig,
    PretrainConfig,
    build_bayesflow_data,
    build_datasets,
    masked_point_accuracy,
    plot_bayesflow_loss,
    plot_hyperparameter_search,
    plot_pretrain_loss,
    pretrain_summary_network,
    run_hyperparameter_search,
    save_artifacts,
    train_bayesflow_workflow,
)


def main() -> None:
    start = time.time()
    print("=== Part 3: training pipeline ===")

    print("\n[1/5] Building datasets...")
    train_data, val_data = build_datasets(
        n_train=2000, n_val=400, min_len=DEFAULT_MIN_LEN, max_len=DEFAULT_MAX_LEN
    )
    print(
        f"  train: {train_data.n} sequences, val: {val_data.n} sequences, "
        f"max_len={DEFAULT_MAX_LEN} (min_len={DEFAULT_MIN_LEN})"
    )

    print("\n[2/5] Pretraining summary network (BiLSTM) on masked BCE loss...")
    pretrain_cfg = PretrainConfig()
    summary_model, pretrain_history = pretrain_summary_network(train_data, val_data, pretrain_cfg)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_pretrain_loss(pretrain_history, os.path.join(PLOTS_DIR, "pretrain_loss.png"))
    print(
        f"  final train loss={pretrain_history['train_loss'][-1]:.4f}  "
        f"final val loss={pretrain_history['val_loss'][-1]:.4f}"
    )

    print("\n[3/5] Freezing summary network, preparing BayesFlow offline data...")
    train_bf_data = build_bayesflow_data(summary_model, train_data)
    val_bf_data = build_bayesflow_data(summary_model, val_data)

    print("\n[4/5] Hyperparameter search over the BayesFlow coupling flow "
          "(ranked by masked accuracy against Forward-Backward, not val_loss)...")
    hp_result = run_hyperparameter_search(train_bf_data, val_bf_data, val_data, search_epochs=15, batch_size=64)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    hp_result.results.to_csv(os.path.join(RESULTS_DIR, "hyperparameter_search.csv"), index=False)
    plot_hyperparameter_search(hp_result.results, os.path.join(PLOTS_DIR, "hyperparameter_search.png"))
    print(hp_result.results.to_string(index=False))
    print(
        f"  best config: {hp_result.best_config}  "
        f"(masked_mae={hp_result.best_masked_mae:.4f}, masked_corr={hp_result.best_masked_corr:.4f})"
    )

    print("\n[5/5] Retraining best config for longer + saving artifacts...")
    # No early_stopping_patience here: EarlyStopping's restore_best_weights would
    # monitor the same untrustworthy val_loss (see masked_point_accuracy). Train
    # for the fixed budget and evaluate the actual (last-epoch) weights below.
    final_cfg = replace(hp_result.best_config, epochs=100, checkpoint_filepath=CHECKPOINT_DIR)
    final_workflow, final_history = train_bayesflow_workflow(train_bf_data, val_bf_data, final_cfg)
    plot_bayesflow_loss(final_history, os.path.join(PLOTS_DIR, "bayesflow_loss.png"))

    final_mae, final_corr = masked_point_accuracy(
        final_workflow, val_data, val_bf_data, num_samples=300, max_eval=val_data.n
    )
    print(f"  final model masked accuracy on full validation set: MAE={final_mae:.4f} corr={final_corr:.4f}")

    final_h = final_history.history
    manifest = {
        "max_len": DEFAULT_MAX_LEN,
        "min_len": DEFAULT_MIN_LEN,
        "n_train": train_data.n,
        "n_val": val_data.n,
        "pretrain_config": asdict(pretrain_cfg),
        "final_flow_config": asdict(final_cfg),
        "final_pretrain_train_loss": pretrain_history["train_loss"][-1],
        "final_pretrain_val_loss": pretrain_history["val_loss"][-1],
        "final_bayesflow_train_loss": final_h["loss"][-1],
        "final_bayesflow_val_loss": final_h.get("val_loss", [None])[-1],
        "final_masked_mae": final_mae,
        "final_masked_corr": final_corr,
        "hyperparameter_search_best_masked_mae": hp_result.best_masked_mae,
        "hyperparameter_search_best_masked_corr": hp_result.best_masked_corr,
    }
    save_artifacts(summary_model, manifest, checkpoint_dir=CHECKPOINT_DIR)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s.")
    print(f"  checkpoints -> {CHECKPOINT_DIR}/ (summary_network.pt, model.keras, manifest.json)")
    print(f"  plots       -> {PLOTS_DIR}/ (pretrain_loss.png, hyperparameter_search.png, bayesflow_loss.png)")
    print(f"  HP results  -> {RESULTS_DIR}/hyperparameter_search.csv")


if __name__ == "__main__":
    main()
