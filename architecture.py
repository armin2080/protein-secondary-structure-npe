"""Sequence architecture utilities for variable-length protein data.

This module handles the Task 2 responsibilities:
- padding amino-acid sequences into rectangular batches;
- creating masks so padding is ignored;
- defining a small BiLSTM sequence model;
- computing a masked per-position binary loss.

Amino acids use indices 0..19. The padding token is 20.

The important idea is that the data module keeps sequences as Python lists of
different lengths, while neural networks usually expect rectangular tensors.
This file is the bridge between those two representations.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from encoding import N_SYMBOLS


# The amino-acid alphabet has 20 real symbols, indexed 0..19.
# We reserve index 20 for padding. This is safe because index 20 is not a real
# amino acid in this project.
PAD_IDX = N_SYMBOLS

BAYESFLOW_INFERENCE_VARIABLES = ["alpha_posteriors"]
BAYESFLOW_INFERENCE_CONDITIONS = ["sequence_summary"]


def _configure_bayesflow_environment() -> None:
    """Set local defaults before importing BayesFlow/Keras."""
    os.environ.setdefault("KERAS_BACKEND", "torch")

    # Matplotlib is imported by BayesFlow diagnostics. Keeping its cache inside
    # the project avoids permission warnings on locked-down Windows setups.
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))


def pad_sequences(
    sequences: Sequence[Sequence[int]],
    pad_value: int = PAD_IDX,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length amino-acid sequences.

    Args:
        sequences: Non-empty list of 1-D arrays/lists with amino-acid indices.
        pad_value: Token used for padded positions.

    Returns:
        padded: LongTensor with shape ``(batch_size, max_len)``.
        mask: BoolTensor with shape ``(batch_size, max_len)``, true for real tokens.
        lengths: LongTensor with shape ``(batch_size,)``.

    Example:
        [[4, 2, 1], [7, 8, 9, 3]] becomes
        padded = [[4, 2, 1, 20], [7, 8, 9, 3]]
        mask   = [[T, T, T,  F], [T, T, T, T]]
    """
    # A batch with no sequences is almost always a bug. It would also make
    # max length undefined, so we fail early with a clear message.
    if len(sequences) == 0:
        raise ValueError("sequences must contain at least one sequence")

    # Store the original lengths before padding. The LSTM uses these lengths to
    # skip padded tokens efficiently, and the mask is built from the same idea.
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    if torch.any(lengths <= 0):
        raise ValueError("all sequences must have positive length")

    batch_size = len(sequences)
    max_len = int(lengths.max().item())

    # Start with a full rectangle of padding tokens. Then each real sequence is
    # copied into the left part of its row.
    padded = torch.full((batch_size, max_len), pad_value, dtype=torch.long)

    # The mask says which entries are real amino acids. It has the same shape as
    # padded, but contains booleans instead of token ids.
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, sequence in enumerate(sequences):
        sequence_tensor = torch.as_tensor(sequence, dtype=torch.long)
        length = int(lengths[i].item())

        # Fill only the real part of row i. Everything after length remains
        # equal to PAD_IDX and has mask value False.
        padded[i, :length] = sequence_tensor
        mask[i, :length] = True

    return padded, mask, lengths


def pad_targets(
    targets: Sequence[Sequence[float]],
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Pad variable-length per-position targets.

    Targets are usually Forward-Backward alpha posterior probabilities.
    They must be padded to the same rectangular shape as the input sequences.
    The numeric value used for target padding does not matter if the loss is
    masked correctly, because padded positions are ignored.
    """
    if len(targets) == 0:
        raise ValueError("targets must contain at least one target sequence")

    lengths = [len(target) for target in targets]
    if min(lengths) <= 0:
        raise ValueError("all target sequences must have positive length")

    # Targets are probabilities, so they should be floating point tensors.
    padded = torch.full((len(targets), max(lengths)), pad_value, dtype=torch.float32)

    for i, target in enumerate(targets):
        target_tensor = torch.as_tensor(target, dtype=torch.float32)
        padded[i, : len(target_tensor)] = target_tensor

    return padded


def masked_mean_pooling(hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool sequence representations while ignoring padded positions.

    hidden_states has shape (batch_size, max_len, hidden_dim).
    mask has shape (batch_size, max_len).

    The result has shape (batch_size, hidden_dim): one vector per sequence.
    """
    # Add a final dimension so the mask can multiply every hidden feature at a
    # given position: (batch, max_len) -> (batch, max_len, 1).
    mask_float = mask.unsqueeze(-1).to(hidden_states.dtype)

    # Padded positions become zero before summing, so they do not contribute to
    # the sequence summary.
    summed = (hidden_states * mask_float).sum(dim=1)

    # Divide by the number of real amino acids, not by max_len. clamp avoids a
    # division by zero if a malformed all-padding sequence ever appears.
    counts = mask_float.sum(dim=1).clamp(min=1.0)
    return summed / counts


class SequenceSummaryNetwork(nn.Module):
    """BiLSTM network for amino-acid sequence summaries and position logits.

    The model has two outputs:
    - a sequence-level summary vector, useful if the group plugs this into a
      BayesFlow-style architecture later;
    - per-position logits, useful for predicting P(alpha | sequence position).

    A logit is an unconstrained number. Applying sigmoid(logit) converts it into
    a probability between 0 and 1.
    """

    def __init__(
        self,
        vocab_size: int = N_SYMBOLS + 1,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        summary_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = True,
        pad_idx: int = PAD_IDX,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx

        # The embedding layer maps integer amino-acid ids to dense vectors.
        # padding_idx tells PyTorch that PAD_IDX is special padding, not a real
        # amino acid token.
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )

        # The LSTM reads the sequence of embeddings. With bidirectional=True,
        # each position gets information from both left-to-right and
        # right-to-left context.
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # A bidirectional LSTM concatenates forward and backward hidden states,
        # so its output dimension doubles.
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        # This head turns the pooled sequence representation into one compact
        # vector per protein sequence.
        self.summary_head = nn.Sequential(
            nn.Linear(lstm_out_dim, summary_dim),
            nn.ReLU(),
        )

        # This head is applied at every position. It outputs one logit per amino
        # acid position: positive logits mean higher predicted alpha probability.
        self.position_head = nn.Linear(lstm_out_dim, 1)

    def forward(
        self,
        sequences: torch.Tensor,
        mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        return_position_logits: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run the sequence model.

        Args:
            sequences: LongTensor with shape ``(batch_size, max_len)``.
            mask: Optional BoolTensor with true values at real positions.
            lengths: Optional original sequence lengths.
            return_position_logits: Whether to return per-position logits.

        Returns:
            ``summary`` if ``return_position_logits`` is false; otherwise
            ``(summary, position_logits)``.
        """
        # If the caller does not provide a mask, infer it from PAD_IDX. This
        # keeps the model usable even when only padded sequences are passed in.
        if mask is None:
            mask = sequences != self.pad_idx
        if lengths is None:
            lengths = mask.sum(dim=1)

        # Shape: (batch, max_len) -> (batch, max_len, embedding_dim).
        embedded = self.embedding(sequences)

        # pack_padded_sequence tells the LSTM the real lengths. This prevents
        # PyTorch from wasting recurrent computation on padding tokens.
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.lstm(packed)

        # Convert the packed LSTM output back to a normal rectangular tensor.
        # total_length keeps the output aligned with the original padded input.
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=sequences.shape[1],
        )

        # Pool over positions to obtain one global vector per sequence.
        summary = self.summary_head(masked_mean_pooling(outputs, mask))
        if not return_position_logits:
            return summary

        # Apply the position head to every LSTM output. squeeze(-1) changes the
        # shape from (batch, max_len, 1) to (batch, max_len).
        position_logits = self.position_head(outputs).squeeze(-1)
        return summary, position_logits


def masked_bce_loss(
    position_logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Binary cross-entropy over real sequence positions only.

    position_logits and targets both have shape (batch_size, max_len).
    mask has the same shape and decides which entries count in the loss.
    """
    # reduction="none" keeps one loss value per position. We need that because
    # padded positions must be removed before averaging.
    loss = F.binary_cross_entropy_with_logits(
        position_logits,
        targets,
        reduction="none",
    )

    # Convert True/False to 1.0/0.0, multiply away padding losses, then average
    # over only the real amino-acid positions.
    mask_float = mask.to(loss.dtype)
    return (loss * mask_float).sum() / mask_float.sum().clamp(min=1.0)


def predict_alpha_probabilities(
    position_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert logits to alpha probabilities, zeroing padding if a mask is given."""
    # sigmoid maps any real number to the interval [0, 1], so logits become
    # interpretable probabilities.
    probabilities = torch.sigmoid(position_logits)
    if mask is not None:
        # Padding positions are not real predictions. Zeroing them makes printed
        # or saved probability tensors easier to inspect.
        probabilities = probabilities * mask.to(probabilities.dtype)
    return probabilities


@dataclass(frozen=True)
class BayesFlowBatch:
    """Fixed-shape arrays for a BayesFlow-style workflow.

    BayesFlow inference networks expect named arrays with consistent shapes.
    The raw project data is ragged, so this small container records the padded
    arrays that can be passed into an adapter/workflow.

    Fields:
        observables: Padded amino-acid ids, shape (batch, max_len).
        mask: Boolean mask, shape (batch, max_len), true for real positions.
        lengths: Original sequence lengths, shape (batch,).
        inference_variables: Padded alpha posterior targets, shape
            (batch, max_len), or None if no targets were provided.
        inference_conditions: Fixed-size sequence summaries, shape
            (batch, summary_dim), or None if no summary network was provided.
    """

    observables: torch.Tensor
    mask: torch.Tensor
    lengths: torch.Tensor
    inference_variables: torch.Tensor | None = None
    inference_conditions: torch.Tensor | None = None

    def as_dict(self) -> dict[str, torch.Tensor]:
        """Return the batch using explicit BayesFlow-style variable names."""
        data = {
            "observables": self.observables,
            "mask": self.mask,
            "lengths": self.lengths,
        }
        if self.inference_variables is not None:
            data["alpha_posteriors"] = self.inference_variables
        if self.inference_conditions is not None:
            data["sequence_summary"] = self.inference_conditions
        return data


def prepare_bayesflow_batch(
    sequences: Sequence[Sequence[int]],
    posteriors: Sequence[Sequence[float]] | None = None,
    summary_network: SequenceSummaryNetwork | None = None,
) -> BayesFlowBatch:
    """Prepare project data for a BayesFlow-style adapter/workflow.

    The adapter-facing names are:
    - ``observables``: padded amino-acid ids;
    - ``mask`` and ``lengths``: padding metadata;
    - ``alpha_posteriors``: padded per-position target vector, if available;
    - ``sequence_summary``: fixed-size BiLSTM summary, if a network is supplied.

    This function deliberately does not train anything. It only converts the
    variable-length data contract of this repository into fixed-shape tensors.
    """
    padded_sequences, mask, lengths = pad_sequences(sequences)
    padded_posteriors = pad_targets(posteriors) if posteriors is not None else None

    summary = None
    if summary_network is not None:
        was_training = summary_network.training
        summary_network.eval()
        with torch.no_grad():
            summary = summary_network(
                padded_sequences,
                mask=mask,
                lengths=lengths,
                return_position_logits=False,
            )
        if was_training:
            summary_network.train()

    return BayesFlowBatch(
        observables=padded_sequences,
        mask=mask,
        lengths=lengths,
        inference_variables=padded_posteriors,
        inference_conditions=summary,
    )


def create_bayesflow_adapter() -> Any:
    """Create a BayesFlow adapter for the project variable names.

    This uses BayesFlow's default adapter machinery to declare:
    - ``alpha_posteriors`` as inference variables;
    - ``sequence_summary`` as inference conditions.

    No ``summary_variables`` are declared here. This project computes
    ``sequence_summary`` *outside* BayesFlow (see ``prepare_bayesflow_batch``,
    which runs ``SequenceSummaryNetwork`` under ``torch.no_grad()``), so there
    is no BayesFlow-managed ``summary_network`` for raw ``observables``/
    ``mask``/``lengths`` to flow through. Declaring them as ``summary_variables``
    without a ``summary_network`` breaks in two ways: the adapter's
    concatenation step fails immediately because ``lengths`` is 1-D while
    ``observables``/``mask`` are 2-D, and even with matching shapes BayesFlow's
    ``ConditionBuilder.resolve`` raises "Cannot use summary_variables without a
    summary network." Verified working without them via ``workflow.fit_offline``.

    BayesFlow is optional for the lightweight checks in this repository. If it
    is not installed, this function raises a clear dependency error instead of
    breaking imports of ``architecture.py``.
    """
    _configure_bayesflow_environment()
    try:
        import bayesflow as bf
    except ImportError as exc:
        raise ImportError(
            "BayesFlow is not installed. Install it with "
            "`python -m pip install \"bayesflow>=2.0\"` to build the adapter."
        ) from exc

    return bf.BasicWorkflow.default_adapter(
        inference_variables=BAYESFLOW_INFERENCE_VARIABLES,
        inference_conditions=BAYESFLOW_INFERENCE_CONDITIONS,
        summary_variables=None,
    )


def create_bayesflow_coupling_network(
    depth: int = 6,
    hidden_widths: Sequence[int] = (128, 128),
    transform: str = "affine",
    permutation: str = "random",
    use_actnorm: bool = True,
) -> Any:
    """Create BayesFlow's coupling-flow inference network.

    The coupling flow is the normalizing-flow part of BayesFlow: it learns a
    conditional distribution over inference variables. In this project, the
    intended inference variables are padded alpha-posterior vectors, conditioned
    on fixed-size sequence summaries.
    """
    _configure_bayesflow_environment()
    try:
        import bayesflow as bf
    except ImportError as exc:
        raise ImportError(
            "BayesFlow is not installed. Install it with "
            "`python -m pip install \"bayesflow>=2.0\"` to build the flow."
        ) from exc

    return bf.networks.CouplingFlow(
        depth=depth,
        transform=transform,
        permutation=permutation,
        use_actnorm=use_actnorm,
        subnet="mlp",
        subnet_kwargs={"widths": list(hidden_widths)},
    )


def create_bayesflow_components() -> dict[str, Any]:
    """Create the adapter and inference network needed by a BayesFlow workflow.

    The return value is intentionally a plain dictionary so later training code
    can pass these pieces into ``bf.BasicWorkflow`` without guessing names.
    """
    return {
        "adapter": create_bayesflow_adapter(),
        "inference_network": create_bayesflow_coupling_network(),
        "inference_variables": BAYESFLOW_INFERENCE_VARIABLES,
        "inference_conditions": BAYESFLOW_INFERENCE_CONDITIONS,
    }


def create_bayesflow_workflow() -> Any:
    """Create a configured BayesFlow ``BasicWorkflow`` shell.

    This is the Bayesian inference entry point for the project. It wires the
    adapter and the coupling-flow inference network together, but does not yet
    define the training loop or simulator object. Those belong to the later
    optimization/training task.
    """
    _configure_bayesflow_environment()
    try:
        import bayesflow as bf
    except ImportError as exc:
        raise ImportError(
            "BayesFlow is not installed. Install it with "
            "`python -m pip install \"bayesflow>=2.0\"` to build the workflow."
        ) from exc

    components = create_bayesflow_components()
    return bf.BasicWorkflow(
        simulator=None,
        adapter=components["adapter"],
        inference_network=components["inference_network"],
        summary_network=None,
        inference_variables=components["inference_variables"],
        inference_conditions=components["inference_conditions"],
    )
