"""Setup script for Section 4 (Diagnostics & Inference Lead).

Run once to generate everything your diagnostics/validation scripts need:
  python setup_section4.py

This will:
  1. Download insulin 1A7F structure and build the validation dataset.
  2. Generate a small training dataset.
  3. Run a quick training session that produces a usable checkpoint
     (~5-10 min on CPU, much faster than the full `train.py`).
"""

import os
import sys
import time


def step(message: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {message}")
    print(f"{'=' * 60}")


def main() -> None:
    start = time.time()

    # ------------------------------------------------------------------
    # Step 1: download + build insulin validation data
    # ------------------------------------------------------------------
    step("1/3  Building insulin 1A7F validation dataset")
    import insulin as ins

    os.makedirs(ins.DATA_DIR, exist_ok=True)
    ins.download_structure()
    sequences, labels, states, residues = ins.build_insulin_dataset()
    ins.save_dataset(ins.DATASET_PATH, sequences, labels, states)
    print(f"  saved  -> {ins.DATASET_PATH}  (chain A: {len(sequences[0])} aa, chain B: {len(sequences[1])} aa)")

    # ------------------------------------------------------------------
    # Step 2: generate training data (small, so training is fast)
    # ------------------------------------------------------------------
    step("2/3  Generating small training & validation datasets")
    from simulator import get_rng
    from forward_backward import generate_dataset, save_dataset

    rng = get_rng(42)
    train_seqs, train_posts, train_states = generate_dataset(800, rng, min_len=10, max_len=60)
    save_dataset("dataset/train_quick.npz", train_seqs, train_posts, train_states)
    print(f"  saved  -> dataset/train_quick.npz  ({len(train_seqs)} sequences)")

    rng = get_rng(99)
    val_seqs, val_posts, val_states = generate_dataset(200, rng, min_len=10, max_len=60)
    save_dataset("dataset/val_quick.npz", val_seqs, val_posts, val_states)
    print(f"  saved  -> dataset/val_quick.npz  ({len(val_seqs)} sequences)")

    # ------------------------------------------------------------------
    # Step 3: quick training → checkpoint
    # ------------------------------------------------------------------
    step("3/3  Running quick training (produces checkpoints/)")
    from dataclasses import replace
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
        pretrain_summary_network,
        save_artifacts,
        train_bayesflow_workflow,
    )

    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("  -> Building padded datasets ...")
    train_data, val_data = build_datasets(
        n_train=800, n_val=200, min_len=DEFAULT_MIN_LEN, max_len=DEFAULT_MAX_LEN,
        seed_train=42, seed_val=99,
    )

    print("  -> Pretraining BiLSTM summary network (12 epochs) ...")
    pretrain_cfg = PretrainConfig(epochs=12, batch_size=64, patience=6)
    summary_model, history = pretrain_summary_network(train_data, val_data, pretrain_cfg, verbose=True)

    print(f"      final train loss: {history['train_loss'][-1]:.4f}")
    print(f"      final val   loss: {history['val_loss'][-1]:.4f}")

    print("  -> Freezing summary network, building BayesFlow data ...")
    train_bf = build_bayesflow_data(summary_model, train_data)
    val_bf = build_bayesflow_data(summary_model, val_data)

    print("  -> Training BayesFlow coupling flow (25 epochs) ...")
    flow_cfg = FlowConfig(epochs=25, initial_learning_rate=1e-3, checkpoint_filepath=CHECKPOINT_DIR)
    _workflow, flow_history = train_bayesflow_workflow(train_bf, val_bf, flow_cfg)

    print(f"      final flow loss: {flow_history.history['loss'][-1]:.4f}")

    print("  -> Saving artifacts ...")
    manifest = {
        "max_len": DEFAULT_MAX_LEN,
        "min_len": DEFAULT_MIN_LEN,
        "n_train": train_data.n,
        "n_val": val_data.n,
        "pretrain_config": {
            "embedding_dim": pretrain_cfg.embedding_dim,
            "hidden_dim": pretrain_cfg.hidden_dim,
            "summary_dim": pretrain_cfg.summary_dim,
            "num_layers": pretrain_cfg.num_layers,
            "dropout": pretrain_cfg.dropout,
        },
        "final_flow_config": {
            "epochs": flow_cfg.epochs,
            "initial_learning_rate": flow_cfg.initial_learning_rate,
        },
        "final_pretrain_train_loss": history["train_loss"][-1],
        "final_pretrain_val_loss": history["val_loss"][-1],
        "final_bayesflow_train_loss": flow_history.history["loss"][-1],
    }
    save_artifacts(summary_model, manifest, checkpoint_dir=CHECKPOINT_DIR)

    elapsed = time.time() - start
    print(f"\n  ✓ Setup complete in {elapsed:.1f}s")
    print(f"    checkpoints/  -> summary_network.pt, model.keras, manifest.json")
    print(f"    dataset/      -> insulin_1A7F.npz, train_quick.npz, val_quick.npz")


if __name__ == "__main__":
    main()
