"""Forward-Backward ground-truth generator (SPEC.md section 4).

The HMM parameters are *known*, so we build an `hmmlearn.CategoricalHMM`, set its
distributions directly from `encoding.py`, and never call `.fit()`. Running the exact
Forward-Backward algorithm (via `predict_proba`) over simulated sequences yields the
per-position posterior `P(state = alpha | sequence)` — the training target for the
BayesFlow network.

Everything is integer-encoded (amino acids 0..19, states 0=alpha / 1=other) in the
canonical ordering; constants come from `encoding.py` and sequences from `simulator.py`.
Sequences stay variable-length here — padding/masking is Member 2's job.
"""

import numpy as np
from hmmlearn.hmm import CategoricalHMM

from encoding import ALPHA, EMISSION, N_STATES, N_SYMBOLS, START_PROB, TRANSITION
from simulator import simulate_batch


def build_hmm():
    """Return a CategoricalHMM with the fixed, known parameters set manually.

    Parameters are given (not learned), so we assign `startprob_`, `transmat_`, and
    `emissionprob_` directly and never call `.fit()`. `encoding.py` already stores them in
    hmmlearn's layout: `TRANSITION[i][j]` is from-state i to-state j, and `EMISSION[s]` is
    the emission distribution of state s over the 20 amino acids.
    """
    model = CategoricalHMM(n_components=N_STATES)
    model.startprob_ = START_PROB.copy()
    model.transmat_ = TRANSITION.copy()
    model.emissionprob_ = EMISSION.copy()
    # Pin the alphabet size. Otherwise hmmlearn infers n_features from the data's max
    # symbol + 1, which is wrong for short sequences that happen to omit high-index amino
    # acids (e.g. V=19) and would mismatch the (2, 20) emission matrix.
    model.n_features = N_SYMBOLS
    return model


def alpha_posterior(sequences, model=None):
    """Per-position posterior `P(state = alpha | sequence)` via Forward-Backward.

    Args:
        sequences: a single integer-encoded sequence (1-D array) OR a list/batch of them.
        model: an optional prebuilt HMM (see `build_hmm`); one is built if omitted.

    Returns:
        For a single sequence, a 1-D float array of length L. For a batch, a list of such
        arrays (one per input sequence, same order). Each value is in [0, 1].
    """
    if model is None:
        model = build_hmm()

    # Normalize to a list of 1-D int arrays; remember whether the caller passed one.
    single = _is_single_sequence(sequences)
    seqs = [np.asarray(sequences, dtype=np.int64)] if single else [
        np.asarray(s, dtype=np.int64) for s in sequences
    ]

    # hmmlearn wants all observations stacked as (n_obs, 1) with a `lengths` array telling
    # it where each sequence ends; it runs Forward-Backward independently per sequence.
    lengths = [len(s) for s in seqs]
    X = np.concatenate(seqs).reshape(-1, 1)
    gamma = model.predict_proba(X, lengths=lengths)  # (n_obs, N_STATES)

    alpha = gamma[:, ALPHA]
    parts = np.split(alpha, np.cumsum(lengths)[:-1])
    return parts[0] if single else parts


def generate_dataset(n, rng, min_len=10, max_len=60, model=None):
    """Simulate `n` sequences and label each with its alpha posterior.

    Args:
        n: number of sequences.
        rng: a numpy Generator (see `simulator.get_rng`).
        min_len, max_len: inclusive bounds on the per-sequence length (SPEC includes short
            sequences so the model generalizes to insulin's 21/30-aa chains).
        model: optional prebuilt HMM; built once if omitted.

    Returns:
        (sequences, posteriors, states): three lists of variable-length arrays, aligned by
        index. `states` is the true hidden path (kept for Member 4's diagnostics).
    """
    if model is None:
        model = build_hmm()

    batch = simulate_batch(n, rng, min_len=min_len, max_len=max_len)
    states = [s for s, _ in batch]
    sequences = [seq for _, seq in batch]
    posteriors = alpha_posterior(sequences, model=model)
    return sequences, posteriors, states


def save_dataset(path, sequences, posteriors, states):
    """Save a variable-length dataset to a single `.npz` (concatenate + lengths).

    Ragged arrays are stored by concatenating all sequences end-to-end and keeping a
    `lengths` array to slice them back — no `dtype=object`, no pickling. This mirrors the
    same `lengths` convention `predict_proba` uses.
    """
    lengths = np.array([len(s) for s in sequences], dtype=np.int64)
    np.savez(
        path,
        lengths=lengths,
        sequences=np.concatenate(sequences).astype(np.int64),
        posteriors=np.concatenate(posteriors).astype(np.float64),
        states=np.concatenate(states).astype(np.int64),
    )


def load_dataset(path):
    """Inverse of `save_dataset`: return (sequences, posteriors, states) as lists.

    Splits each concatenated array back into per-sequence pieces using `lengths`.
    """
    data = np.load(path)
    bounds = np.cumsum(data["lengths"])[:-1]  # split points between sequences
    sequences = np.split(data["sequences"], bounds)
    posteriors = np.split(data["posteriors"], bounds)
    states = np.split(data["states"], bounds)
    return sequences, posteriors, states


def _is_single_sequence(sequences):
    """True if `sequences` is one sequence rather than a batch of them.

    A single sequence is a 1-D array/list of ints; a batch is a list of such arrays.
    """
    if isinstance(sequences, np.ndarray):
        return sequences.ndim == 1
    # A list/tuple: it's a single sequence only if its first element is a scalar.
    return len(sequences) == 0 or np.ndim(sequences[0]) == 0


if __name__ == "__main__":
    from simulator import get_rng

    rng = get_rng(0)
    sequences, posteriors, states = generate_dataset(3, rng, min_len=5, max_len=12)
    print(f"generated {len(sequences)} sequences (lengths {[len(s) for s in sequences]})")
    seq, post = sequences[0], posteriors[0]
    print("sequence      :", seq)
    print("alpha posterior:", np.round(post, 3))
