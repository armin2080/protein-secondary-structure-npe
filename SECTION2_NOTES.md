# Section 2 - Architecture notes

Handoff notes for the sequence architecture, padding/masking, and BayesFlow
configuration.

## Module

- **`architecture.py`** implements the architecture layer for variable-length
  protein sequences.
- **`check_architecture.py`** verifies shapes, padding, masking, forward pass,
  masked loss, and BayesFlow component creation.
- **`check_training_smoke.py`** runs a small training smoke test to confirm that
  the BiLSTM can reduce loss on Forward-Backward posterior targets.

## Padding and masking

The data pipeline stores sequences as lists with different lengths. Neural
networks need rectangular batches, so `pad_sequences` pads every sequence to the
maximum length in the batch.

- Real amino acids use indices `0..19`.
- Padding uses `PAD_IDX = 20`.
- `mask` has the same shape as the padded batch:
  - `True` = real amino-acid position,
  - `False` = padding.

Targets are padded with `pad_targets`. Padding values in targets are ignored by
`masked_bce_loss`, so they do not affect training.

## Sequence network

`SequenceSummaryNetwork` is a PyTorch model:

```text
padded amino-acid ids
-> embedding layer
-> bidirectional LSTM
-> masked mean pooling
-> sequence summary
```

It also has a position head:

```text
BiLSTM outputs -> per-position logits -> sigmoid -> P(alpha)
```

Outputs:

- `sequence_summary`: fixed-size vector per sequence, used as BayesFlow
  inference condition.
- `position_logits`: one logit per sequence position, useful for predicting
  Forward-Backward alpha posterior probabilities.

## BayesFlow configuration

BayesFlow-related helpers in `architecture.py`:

- `prepare_bayesflow_batch(...)` creates named tensors:
  - `observables`
  - `mask`
  - `lengths`
  - `alpha_posteriors`
  - `sequence_summary`
- `create_bayesflow_adapter()` creates the BayesFlow adapter.
- `create_bayesflow_coupling_network()` creates the `CouplingFlow` inference
  network.
- `create_bayesflow_components()` returns adapter, inference network, and
  variable names.
- `create_bayesflow_workflow()` creates a configured `BasicWorkflow` shell.

The training loop itself is not part of Section 2; it belongs to the optimization
task. Section 2 prepares the architecture and BayesFlow components needed for it.

## Checks

Run:

```powershell
.\.venv\Scripts\python.exe check_architecture.py
.\.venv\Scripts\python.exe check_training_smoke.py
```

Expected key outputs:

```text
Architecture check passed.
BayesFlow config : Adapter + CouplingFlow
```

and for the smoke test:

```text
Training smoke test passed.
initial train loss   : ...
final train loss     : ...
initial val loss     : ...
final val loss       : ...
BayesFlow config     : Adapter + CouplingFlow
```

The loss should decrease in the smoke test. This confirms that gradients flow,
the model can learn from Forward-Backward posterior targets, and the BayesFlow
components can be instantiated.

## Presentation summary

Section 2 converts variable-length amino-acid sequences into neural-network
batches using padding and masks. A BiLSTM processes the padded sequences and
produces both fixed-size sequence summaries and per-position alpha-helix logits.
The masked loss ignores padding. Finally, the BayesFlow adapter and CouplingFlow
inference network are configured so the architecture can be used in the later
simulation-based inference training pipeline.
