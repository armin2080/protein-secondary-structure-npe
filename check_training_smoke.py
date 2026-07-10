"""Small end-to-end smoke test for the architecture.

Run with:
    python check_training_smoke.py

This is intentionally not the final training pipeline. It is a quick check that:
- simulated HMM data can be converted into padded neural-network batches;
- the BiLSTM architecture can reduce the masked posterior-prediction loss;
- BayesFlow adapter + CouplingFlow components can be instantiated.
"""

import torch

from architecture import (
    SequenceSummaryNetwork,
    create_bayesflow_components,
    masked_bce_loss,
    pad_sequences,
    pad_targets,
)
from forward_backward import generate_dataset
from simulator import get_rng


def evaluate(model, sequences, targets):
    """Compute masked BCE loss without updating the model."""
    padded_sequences, mask, lengths = pad_sequences(sequences)
    padded_targets = pad_targets(targets)

    model.eval()
    with torch.no_grad():
        _summary, logits = model(padded_sequences, mask=mask, lengths=lengths)
        loss = masked_bce_loss(logits, padded_targets, mask)
    return loss.item()


def main() -> None:
    torch.manual_seed(0)

    train_sequences, train_posteriors, _ = generate_dataset(
        n=160,
        rng=get_rng(1),
        min_len=10,
        max_len=50,
    )
    val_sequences, val_posteriors, _ = generate_dataset(
        n=40,
        rng=get_rng(2),
        min_len=10,
        max_len=50,
    )

    padded_train, train_mask, train_lengths = pad_sequences(train_sequences)
    train_targets = pad_targets(train_posteriors)

    model = SequenceSummaryNetwork(
        embedding_dim=16,
        hidden_dim=32,
        summary_dim=32,
        bidirectional=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    initial_train_loss = evaluate(model, train_sequences, train_posteriors)
    initial_val_loss = evaluate(model, val_sequences, val_posteriors)

    # A short full-batch loop is enough to show that gradients flow and the
    # architecture can learn the Forward-Backward posterior target.
    for _epoch in range(12):
        model.train()
        optimizer.zero_grad()
        _summary, logits = model(
            padded_train,
            mask=train_mask,
            lengths=train_lengths,
        )
        loss = masked_bce_loss(logits, train_targets, train_mask)
        loss.backward()
        optimizer.step()

    final_train_loss = evaluate(model, train_sequences, train_posteriors)
    final_val_loss = evaluate(model, val_sequences, val_posteriors)

    components = create_bayesflow_components()
    adapter_name = type(components["adapter"]).__name__
    inference_name = type(components["inference_network"]).__name__

    print("Training smoke test passed.")
    print(f"train examples       : {len(train_sequences)}")
    print(f"validation examples  : {len(val_sequences)}")
    print(f"initial train loss   : {initial_train_loss:.4f}")
    print(f"final train loss     : {final_train_loss:.4f}")
    print(f"initial val loss     : {initial_val_loss:.4f}")
    print(f"final val loss       : {final_val_loss:.4f}")
    print(f"BayesFlow config     : {adapter_name} + {inference_name}")

    assert final_train_loss < initial_train_loss
    assert torch.isfinite(torch.tensor(final_train_loss))
    assert adapter_name == "Adapter"
    assert inference_name == "CouplingFlow"


if __name__ == "__main__":
    main()
