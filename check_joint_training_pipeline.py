"""Quick correctness check for the joint (end-to-end) training pipeline.

Run with:
    python check_joint_training_pipeline.py

This is not the real training run (see ``train_joint.py``). Tiny data, few
epochs; verifies dataset padding, fit_offline with a live (unfrozen) BiLSTM,
gradient flow into it, the hyperparameter search, and artifact save/reload --
mirrors ``check_training_pipeline.py``'s structure for the staged pipeline.
"""

import os
import shutil

import numpy as np
import torch

from dataclasses import asdict

from architecture import (
    BAYESFLOW_INFERENCE_VARIABLES,
    _configure_bayesflow_environment,
    create_bayesflow_coupling_network,
    create_bayesflow_joint_adapter,
    create_joint_summary_network,
)
from forward_backward import generate_dataset
from simulator import get_rng
from training import build_datasets, plot_bayesflow_loss, plot_hyperparameter_search
from joint_training import (
    JointConfig,
    load_joint_pipeline,
    predict_alpha_posterior_joint,
    run_joint_hyperparameter_search,
    save_joint_artifacts,
    train_joint_workflow,
)

CHECK_DIR = "_check_joint_training_pipeline_artifacts"


def main() -> None:
    if os.path.exists(CHECK_DIR):
        shutil.rmtree(CHECK_DIR)
    plots_dir = os.path.join(CHECK_DIR, "plots")
    checkpoint_dir = os.path.join(CHECK_DIR, "checkpoints")
    max_len = 20

    # [1] Dataset padding: fixed max_len regardless of the batch's own max.
    train_data, val_data = build_datasets(
        n_train=48, n_val=16, min_len=8, max_len=max_len, seed_train=100, seed_val=101
    )
    assert train_data.sequences.shape == (48, max_len)
    assert val_data.mask.shape == (16, max_len)

    # [2] Gradient-flow check: build the pieces directly (not via
    # train_joint_workflow) to snapshot the BiLSTM's embedding weight before
    # vs. after a short fit_offline call, on the same instance.
    _configure_bayesflow_environment()
    import bayesflow as bf

    probe_summary_network = create_joint_summary_network(embedding_dim=8, hidden_dim=8, summary_dim=8, num_layers=1, dropout=0.0)
    probe_workflow = bf.BasicWorkflow(
        adapter=create_bayesflow_joint_adapter(),
        inference_network=create_bayesflow_coupling_network(depth=2, hidden_widths=(16, 16)),
        summary_network=probe_summary_network,
        inference_variables=BAYESFLOW_INFERENCE_VARIABLES,
    )
    embedding_before = probe_summary_network.wrapped.module.embedding.weight.detach().clone()
    probe_workflow.fit_offline(
        data={"observables": train_data.sequences.numpy(), "alpha_posteriors": train_data.targets.numpy()},
        epochs=2, batch_size=16, verbose=0,
    )
    embedding_after = probe_summary_network.wrapped.module.embedding.weight.detach()
    assert not torch.allclose(embedding_before, embedding_after), (
        "gradient did not reach the (unfrozen) BiLSTM's embedding weights"
    )

    # [3] Joint fit via the real train_joint_workflow entry point: the flow's
    # loss must be finite.
    joint_cfg = JointConfig(
        embedding_dim=8, hidden_dim=8, summary_dim=8, num_layers=1, dropout=0.0,
        depth=2, hidden_widths=(16, 16), epochs=4, batch_size=16,
    )
    workflow, history = train_joint_workflow(train_data, val_data, joint_cfg, verbose=False)
    train_losses = history.history["loss"]
    assert all(np.isfinite(x) for x in train_losses)

    plot_bayesflow_loss(history, os.path.join(plots_dir, "joint_loss.png"))
    assert os.path.exists(os.path.join(plots_dir, "joint_loss.png"))

    # [4] Hyperparameter search over a tiny curated space, ranked by masked
    # accuracy against true Forward-Backward on real positions (see
    # masked_point_accuracy_joint for why val_loss alone is untrustworthy).
    tiny_search_space = [
        {"name": "a", "optimizer_name": "adamw", "initial_learning_rate": 1e-3, "weight_decay": 1e-3, "depth": 2},
        {"name": "b", "optimizer_name": "adam", "initial_learning_rate": 5e-4, "weight_decay": 0.0, "depth": 2},
    ]
    hp_result = run_joint_hyperparameter_search(
        train_data, val_data, search_epochs=3, batch_size=16, search_space=tiny_search_space, verbose=False
    )
    assert len(hp_result.results) == 2
    assert hp_result.best_config is not None
    assert np.isfinite(hp_result.best_masked_mae)
    assert "masked_mae" in hp_result.results.columns
    plot_hyperparameter_search(hp_result.results, os.path.join(plots_dir, "hyperparameter_search_joint.png"))
    assert os.path.exists(os.path.join(plots_dir, "hyperparameter_search_joint.png"))

    # [5] Fit the winning config with checkpointing on, so model.keras is written.
    final_cfg = JointConfig(
        embedding_dim=8, hidden_dim=8, summary_dim=8, num_layers=1, dropout=0.0,
        depth=hp_result.best_config.depth,
        hidden_widths=hp_result.best_config.hidden_widths,
        optimizer_name=hp_result.best_config.optimizer_name,
        initial_learning_rate=hp_result.best_config.initial_learning_rate,
        weight_decay=hp_result.best_config.weight_decay,
        epochs=4,
        batch_size=16,
        checkpoint_filepath=checkpoint_dir,
    )
    final_workflow, final_history = train_joint_workflow(train_data, val_data, final_cfg, verbose=False)
    assert all(np.isfinite(x) for x in final_history.history["loss"])
    assert os.path.exists(os.path.join(checkpoint_dir, "model.keras"))

    # [6] Save the manifest (no separate BiLSTM .pt file -- see save_joint_artifacts).
    manifest = {"max_len": max_len, "joint_config": asdict(final_cfg), "note": "check_joint_training_pipeline smoke test"}
    save_joint_artifacts(manifest, checkpoint_dir=checkpoint_dir)
    assert os.path.exists(os.path.join(checkpoint_dir, "manifest.json"))
    assert not os.path.exists(os.path.join(checkpoint_dir, "summary_network.pt"))

    # [7] Sample before reloading, to compare against a fresh reload below --
    # same seed should give identical samples if save/reload is exact.
    probe_seqs, _probe_posts, _ = generate_dataset(3, get_rng(555), min_len=5, max_len=max_len)
    padded_probe = torch.full((3, max_len), 20, dtype=torch.long)
    for i, s in enumerate(probe_seqs):
        padded_probe[i, : len(s)] = torch.as_tensor(s, dtype=torch.long)
    samples_before = np.asarray(
        final_workflow.approximator.sample(
            num_samples=10, conditions={"observables": padded_probe.numpy()}, seed=0
        )["alpha_posteriors"]
    )

    # [8] Reload from disk (no in-memory state assumed) and confirm it
    # reproduces identical samples -- what Task 4 will rely on.
    loaded_summary_model, loaded_approximator, loaded_manifest = load_joint_pipeline(checkpoint_dir)
    assert loaded_manifest["max_len"] == max_len
    assert loaded_summary_model.embedding.weight.shape == (21, 8)  # real, inspectable BiLSTM

    samples_after = np.asarray(
        loaded_approximator.sample(
            num_samples=10, conditions={"observables": padded_probe.numpy()}, seed=0
        )["alpha_posteriors"]
    )
    assert np.allclose(samples_before, samples_after), "reload did not reproduce identical samples"

    predictions = predict_alpha_posterior_joint(
        probe_seqs, loaded_approximator, max_len=max_len, num_samples=20
    )
    assert len(predictions) == 3
    for seq, pred in zip(probe_seqs, predictions):
        assert pred.shape == (20, len(seq))
        finite = pred[np.isfinite(pred)]
        assert np.all(finite >= 0.0) and np.all(finite <= 1.0)

    shutil.rmtree(CHECK_DIR)

    print("Joint training pipeline check passed.")
    print(f"joint fit loss   : {train_losses[0]:.4f} -> {train_losses[-1]:.4f}")
    print(f"hp search        : {len(hp_result.results)} configs, best = {hp_result.results.iloc[0]['name']}")
    print("Gradient reaches the (unfrozen) BiLSTM embedding: verified.")
    print("Save/reload round-trip reproduces identical samples: verified.")
    print("Fresh reload + predict_alpha_posterior_joint() verified (this is what Task 4 will use).")


if __name__ == "__main__":
    main()
