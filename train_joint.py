"""End-to-end joint (non-frozen) training pipeline - alternative to train.py.

Run with:
    python train_joint.py

Builds the same real-sized dataset as ``train.py`` (identical seeds/sizes,
for a fair comparison), searches over the joint model (BiLSTM + coupling flow
trained together, single optimizer), retrains the best config for longer, and
saves diagnostics plots, a results table, and checkpoints/manifest for Task 4.

This is not a quick check -- see ``check_joint_training_pipeline.py``. Expect
at least as long as ``train.py``, likely longer: the HP search runs the full
BiLSTM forward/backward every trial step, unlike the staged search's cheap
precomputed summaries (see ``joint_training.py``).
"""

import os
import time
from dataclasses import asdict, replace

from training import (
    DEFAULT_MAX_LEN,
    DEFAULT_MIN_LEN,
    PLOTS_DIR,
    build_datasets,
    plot_bayesflow_loss,
    plot_hyperparameter_search,
)
from joint_training import (
    JointConfig,
    masked_point_accuracy_joint,
    run_joint_hyperparameter_search,
    save_joint_artifacts,
    train_joint_workflow,
)

CHECKPOINT_DIR_JOINT = "checkpoints_joint"
RESULTS_DIR_JOINT = "results_joint"


def main() -> None:
    start = time.time()
    print("=== Joint (end-to-end) training pipeline ===")

    print("\n[1/4] Building datasets (same seeds/sizes as train.py)...")
    train_data, val_data = build_datasets(
        n_train=2000, n_val=400, min_len=DEFAULT_MIN_LEN, max_len=DEFAULT_MAX_LEN
    )
    print(
        f"  train: {train_data.n} sequences, val: {val_data.n} sequences, "
        f"max_len={DEFAULT_MAX_LEN} (min_len={DEFAULT_MIN_LEN})"
    )

    print(
        "\n[2/4] Hyperparameter search over the joint model (BiLSTM + coupling flow "
        "trained together; ranked by masked accuracy against Forward-Backward, not val_loss)..."
    )
    hp_result = run_joint_hyperparameter_search(train_data, val_data, search_epochs=10, batch_size=64)
    os.makedirs(RESULTS_DIR_JOINT, exist_ok=True)
    hp_result.results.to_csv(os.path.join(RESULTS_DIR_JOINT, "hyperparameter_search.csv"), index=False)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_hyperparameter_search(hp_result.results, os.path.join(PLOTS_DIR, "hyperparameter_search_joint.png"))
    print(hp_result.results.to_string(index=False))
    print(
        f"  best config: {hp_result.best_config}  "
        f"(masked_mae={hp_result.best_masked_mae:.4f}, masked_corr={hp_result.best_masked_corr:.4f})"
    )

    print("\n[3/4] Retraining best config for longer + saving artifacts...")
    # No early_stopping_patience: same reason as train.py -- EarlyStopping's
    # restore_best_weights would monitor the same untrustworthy val_loss.
    final_cfg = replace(hp_result.best_config, epochs=100, checkpoint_filepath=CHECKPOINT_DIR_JOINT)
    final_workflow, final_history = train_joint_workflow(train_data, val_data, final_cfg)
    plot_bayesflow_loss(final_history, os.path.join(PLOTS_DIR, "joint_loss.png"))

    final_mae, final_corr = masked_point_accuracy_joint(
        final_workflow, val_data, num_samples=300, max_eval=val_data.n
    )
    print(f"  final model masked accuracy on full validation set: MAE={final_mae:.4f} corr={final_corr:.4f}")

    elapsed = time.time() - start

    print("\n[4/4] Saving manifest...")
    final_h = final_history.history
    manifest = {
        "max_len": DEFAULT_MAX_LEN,
        "min_len": DEFAULT_MIN_LEN,
        "n_train": train_data.n,
        "n_val": val_data.n,
        "joint_config": asdict(final_cfg),
        "final_joint_train_loss": final_h["loss"][-1],
        "final_joint_val_loss": final_h.get("val_loss", [None])[-1],
        "final_masked_mae": final_mae,
        "final_masked_corr": final_corr,
        "hyperparameter_search_best_masked_mae": hp_result.best_masked_mae,
        "hyperparameter_search_best_masked_corr": hp_result.best_masked_corr,
        "elapsed_seconds": elapsed,
    }
    save_joint_artifacts(manifest, checkpoint_dir=CHECKPOINT_DIR_JOINT)

    print(f"\nDone in {elapsed:.1f}s.")
    print(f"  checkpoints -> {CHECKPOINT_DIR_JOINT}/ (model.keras, manifest.json)")
    print(f"  plots       -> {PLOTS_DIR}/ (joint_loss.png, hyperparameter_search_joint.png)")
    print(f"  HP results  -> {RESULTS_DIR_JOINT}/hyperparameter_search.csv")


if __name__ == "__main__":
    main()
