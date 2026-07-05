"""Verification for the Forward-Backward ground-truth generator.

Three checks (run directly:  python check_forward_backward.py):
  1. Posteriors are valid probabilities: every value in [0, 1], the two state columns sum
     to 1 per position, and `alpha_posterior` returns exactly the alpha column.
  2. Correctness: an independent pure-numpy Forward-Backward matches hmmlearn on tiny
     sequences. This proves the transition orientation and emission indexing are wired
     right (a swapped `transmat_` or mis-indexed emission would diverge here).
  3. Reproducibility: same seed -> identical dataset; save -> load round-trips exactly.
"""

import os
import tempfile

import numpy as np

from encoding import ALPHA, EMISSION, N_STATES, START_PROB, TRANSITION
from forward_backward import (
    alpha_posterior,
    build_hmm,
    generate_dataset,
    load_dataset,
    save_dataset,
)
from simulator import get_rng

SEED = 20260705


def _reference_fb(obs):
    """Independent, from-scratch Forward-Backward posterior over hidden states.

    Classic recursion in probability space (fine for the tiny sequences used here). Uses
    the same convention as hmmlearn: TRANSITION[i, j] is P(state j at t | state i at t-1)
    and EMISSION[s, o] is P(symbol o | state s). Returns gamma of shape (L, N_STATES).
    """
    obs = np.asarray(obs, dtype=int)
    L = len(obs)

    # Forward: alpha[t, j] = P(o_0..o_t, state_t = j).
    alpha = np.zeros((L, N_STATES))
    alpha[0] = START_PROB * EMISSION[:, obs[0]]
    for t in range(1, L):
        for j in range(N_STATES):
            alpha[t, j] = (alpha[t - 1] * TRANSITION[:, j]).sum() * EMISSION[j, obs[t]]

    # Backward: beta[t, i] = P(o_{t+1}..o_{L-1} | state_t = i).
    beta = np.zeros((L, N_STATES))
    beta[-1] = 1.0
    for t in range(L - 2, -1, -1):
        for i in range(N_STATES):
            beta[t, i] = (TRANSITION[i, :] * EMISSION[:, obs[t + 1]] * beta[t + 1]).sum()

    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma


def check_valid_probabilities():
    """Posteriors are valid: in [0, 1], columns sum to 1, position 0 alpha == 0."""
    model = build_hmm()
    rng = get_rng(SEED)
    sequences, _, _ = generate_dataset(200, rng, min_len=5, max_len=40, model=model)

    # Full (n_obs, 2) posterior matrix over the whole batch.
    lengths = [len(s) for s in sequences]
    X = np.concatenate(sequences).reshape(-1, 1)
    gamma = model.predict_proba(X, lengths=lengths)

    assert gamma.min() >= 0.0 and gamma.max() <= 1.0, "posteriors must lie in [0, 1]"
    assert np.allclose(gamma.sum(axis=1), 1.0), "the two state columns must sum to 1"

    # `alpha_posterior` must return exactly the alpha column, per sequence.
    alpha_cols = alpha_posterior(sequences, model=model)
    assert np.allclose(np.concatenate(alpha_cols), gamma[:, ALPHA]), \
        "alpha_posterior must equal the alpha column of predict_proba"

    # Start is deterministically `other`, so every sequence's position-0 alpha is exactly 0.
    starts = np.array([post[0] for post in alpha_cols])
    assert np.allclose(starts, 0.0), "position-0 alpha posterior must be 0 (starts in other)"

    print(f"[1/3] valid probabilities: {len(sequences)} sequences, "
          f"gamma in [{gamma.min():.3f}, {gamma.max():.3f}], columns sum to 1  OK")


def check_correctness():
    """Independent Forward-Backward matches hmmlearn on tiny sequences."""
    model = build_hmm()

    # Worked length-2 example, obs = [G=7, A=0], computed by hand from encoding.py:
    #   forward : alpha[0] = [0, 0.09];  alpha[1] = [0.00054, 0.00513]
    #   backward: beta[0]  = [0.114, 0.063];  beta[1] = [1, 1]
    #   gamma alpha column = [0.0, 0.00054 / 0.00567] = [0.0, 0.095238...]
    obs = [7, 0]
    expected = np.array([0.0, 0.00054 / 0.00567])
    got = alpha_posterior(np.array(obs), model=model)
    assert np.allclose(got, expected), f"hand-computed length-2 case: {got} vs {expected}"
    assert np.allclose(got, _reference_fb(obs)[:, ALPHA]), "reference FB disagrees on [7, 0]"

    # A handful of tiny sequences (length 2 and 3) checked against the reference recursion.
    tiny = [[7, 0], [0, 19], [3, 3], [10, 5, 14], [7, 7, 7], [19, 0, 12]]
    for obs in tiny:
        ref = _reference_fb(obs)                       # full (L, 2) gamma
        hmm = model.predict_proba(np.asarray(obs).reshape(-1, 1))
        assert np.allclose(ref, hmm), f"reference FB vs hmmlearn mismatch on {obs}"

    print(f"[2/3] correctness: hand-computed case + {len(tiny)} tiny sequences match "
          f"the independent Forward-Backward  OK")


def check_reproducibility():
    """Same seed -> identical dataset; save/load round-trips exactly."""
    model = build_hmm()

    seqs_a, post_a, states_a = generate_dataset(50, get_rng(SEED), model=model)
    seqs_b, post_b, states_b = generate_dataset(50, get_rng(SEED), model=model)

    assert len(seqs_a) == len(seqs_b)
    for a, b in zip(seqs_a, seqs_b):
        assert np.array_equal(a, b), "sequences differ across identical seeds"
    for a, b in zip(post_a, post_b):
        assert np.array_equal(a, b), "posteriors differ across identical seeds"
    for a, b in zip(states_a, states_b):
        assert np.array_equal(a, b), "states differ across identical seeds"

    # Save -> load round-trip.
    fd, path = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    try:
        save_dataset(path, seqs_a, post_a, states_a)
        seqs_l, post_l, states_l = load_dataset(path)
    finally:
        os.remove(path)

    assert len(seqs_l) == len(seqs_a)
    for orig, loaded in zip(seqs_a, seqs_l):
        assert np.array_equal(orig, loaded), "sequence changed through save/load"
    for orig, loaded in zip(post_a, post_l):
        assert np.array_equal(orig, loaded), "posterior changed through save/load"
    for orig, loaded in zip(states_a, states_l):
        assert np.array_equal(orig, loaded), "states changed through save/load"

    print("[3/3] reproducibility: identical seeds match, save/load round-trips exact  OK")


def main():
    check_valid_probabilities()
    check_correctness()
    check_reproducibility()
    print("\nAll Forward-Backward checks passed.")


if __name__ == "__main__":
    main()
