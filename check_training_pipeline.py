"""Quick correctness check for the Task 3 training pipeline.

Run with:
    python check_training_pipeline.py

This is not the real training run (see ``train.py`` for that). It uses tiny
data and few epochs to verify every stage of the pipeline runs end-to-end:
dataset padding, BiLSTM pretraining, freezing + BayesFlow offline data
preparation, ``fit_offline`` training, the hyperparameter search, diagnostics
plotting, and artifact save/reload.
"""

import os
import shutil

import numpy as np
import torch

from dataclasses import asdict

from architecture import SequenceSummaryNetwork
from forward_backward import generate_dataset
from simulator import get_rng
from training import (
    FlowConfig,
    PretrainConfig,
    build_bayesflow_data,
    build_datasets,
    load_trained_pipeline,
    plot_bayesflow_loss,
    plot_hyperparameter_search,
    plot_pretrain_loss,
    predict_alpha_posterior,
    pretrain_summary_network,
    run_hyperparameter_search,
    save_artifacts,
    train_bayesflow_workflow,
)

CHECK_DIR = "_check_training_pipeline_artifacts"


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
    assert torch.all(train_data.sequences[~train_data.mask] == 20)  # PAD_IDX

    # [2] BiLSTM pretraining: loss should drop, val loss must be finite.
    pretrain_cfg = PretrainConfig(
        embedding_dim=8, hidden_dim=8, summary_dim=8, num_layers=1, dropout=0.0, epochs=6, batch_size=16, patience=6
    )
    model, history = pretrain_summary_network(train_data, val_data, pretrain_cfg, verbose=False)
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert np.isfinite(history["val_loss"][-1])
    plot_pretrain_loss(history, os.path.join(plots_dir, "pretrain_loss.png"))
    assert os.path.exists(os.path.join(plots_dir, "pretrain_loss.png"))

    # [3] Freeze + prepare BayesFlow offline data. The condition is the (now
    # auxiliary-loss-trained) pooled summary (dim 8) concatenated with
    # per-position logits (dim max_len): see compute_sequence_summaries.
    train_bf = build_bayesflow_data(model, train_data)
    val_bf = build_bayesflow_data(model, val_data)
    assert train_bf["sequence_summary"].shape == (48, 8 + max_len)
    assert train_bf["alpha_posteriors"].shape == (48, max_len)

    # [4] Hyperparameter search over a tiny curated space. Ranked by masked
    # accuracy against true Forward-Backward on real positions, not val_loss
    # (see masked_point_accuracy for why val_loss alone is untrustworthy here).
    tiny_search_space = [
        {"name": "a", "optimizer_name": "adamw", "initial_learning_rate": 1e-3, "weight_decay": 1e-3, "depth": 2},
        {"name": "b", "optimizer_name": "adam", "initial_learning_rate": 5e-4, "weight_decay": 0.0, "depth": 2},
    ]
    hp_result = run_hyperparameter_search(
        train_bf, val_bf, val_data, search_epochs=3, batch_size=16, search_space=tiny_search_space, verbose=False
    )
    assert len(hp_result.results) == 2
    assert hp_result.best_config is not None
    assert np.isfinite(hp_result.best_masked_mae)
    assert "masked_mae" in hp_result.results.columns
    plot_hyperparameter_search(hp_result.results, os.path.join(plots_dir, "hyperparameter_search.png"))
    assert os.path.exists(os.path.join(plots_dir, "hyperparameter_search.png"))

    # [5] Fit the winning config, plot the BayesFlow loss curve.
    final_cfg = FlowConfig(
        depth=hp_result.best_config.depth,
        hidden_widths=hp_result.best_config.hidden_widths,
        optimizer_name=hp_result.best_config.optimizer_name,
        initial_learning_rate=hp_result.best_config.initial_learning_rate,
        weight_decay=hp_result.best_config.weight_decay,
        epochs=4,
        batch_size=16,
        checkpoint_filepath=checkpoint_dir,
    )
    workflow, bf_history = train_bayesflow_workflow(train_bf, val_bf, final_cfg, verbose=False)
    train_losses = bf_history.history["loss"]
    assert all(np.isfinite(x) for x in train_losses)
    plot_bayesflow_loss(bf_history, os.path.join(plots_dir, "bayesflow_loss.png"))
    assert os.path.exists(os.path.join(plots_dir, "bayesflow_loss.png"))
    assert os.path.exists(os.path.join(checkpoint_dir, "model.keras"))

    # [6] Save artifacts and confirm the BiLSTM weights reload identically.
    manifest = {"max_len": max_len, "pretrain_config": asdict(pretrain_cfg), "note": "check_training_pipeline smoke test"}
    save_artifacts(model, manifest, checkpoint_dir=checkpoint_dir)
    assert os.path.exists(os.path.join(checkpoint_dir, "summary_network.pt"))
    assert os.path.exists(os.path.join(checkpoint_dir, "manifest.json"))

    reloaded = SequenceSummaryNetwork(embedding_dim=8, hidden_dim=8, summary_dim=8, num_layers=1, dropout=0.0)
    reloaded.load_state_dict(torch.load(os.path.join(checkpoint_dir, "summary_network.pt"), weights_only=True))
    reloaded.eval()
    with torch.no_grad():
        original_summary = model(
            train_data.sequences, mask=train_data.mask, lengths=train_data.lengths, return_position_logits=False
        )
        reloaded_summary = reloaded(
            train_data.sequences, mask=train_data.mask, lengths=train_data.lengths, return_position_logits=False
        )
    assert torch.allclose(original_summary, reloaded_summary)

    # [7] Reload from disk in a way that doesn't assume
    # anything already in memory, then run inference on a couple of new sequences.
    loaded_summary_model, loaded_approximator, loaded_manifest = load_trained_pipeline(checkpoint_dir)
    assert loaded_manifest["max_len"] == max_len
    new_seqs, _new_posts, _ = generate_dataset(3, get_rng(555), min_len=5, max_len=max_len)
    predictions = predict_alpha_posterior(
        new_seqs, loaded_summary_model, loaded_approximator, max_len=max_len, num_samples=20
    )
    assert len(predictions) == 3
    for seq, pred in zip(new_seqs, predictions):
        assert pred.shape == (20, len(seq))
        # Finite entries must be valid probabilities; NaN (non-finite draws, preserved
        # rather than silently laundered into fake 0/1 atoms) is allowed -- see
        # predict_alpha_posterior's docstring.
        finite = pred[np.isfinite(pred)]
        assert np.all(finite >= 0.0) and np.all(finite <= 1.0)

    shutil.rmtree(CHECK_DIR)

    print("Training pipeline check passed.")
    print(f"pretrain loss    : {history['train_loss'][0]:.4f} -> {history['train_loss'][-1]:.4f}")
    print(f"hp search        : {len(hp_result.results)} configs, best = {hp_result.results.iloc[0]['name']}")
    print(f"bayesflow loss   : {train_losses[0]:.4f} -> {train_losses[-1]:.4f}")
    print("Diagnostics plots created and artifacts saved/reloaded successfully.")
    print("Fresh reload + predict_alpha_posterior() verified (this is what Task 4 will use).")


if __name__ == "__main__":
    main()
