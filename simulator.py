"""HMM simulator: sample hidden state paths and amino-acid sequences.

Implements the generative process from SPEC.md section 3 using the fixed constants in
`encoding.py`. Sequences are integer-encoded (0..19) in the canonical amino-acid order.
All randomness flows through a numpy Generator so every draw is seedable/reproducible.
"""

import numpy as np

from encoding import EMISSION, N_STATES, N_SYMBOLS, OTHER, START_PROB, TRANSITION


def get_rng(seed=None):
    """Return a seeded numpy Generator. Accepts an int seed, None, or a Generator."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def simulate_sequence(length, rng):
    """Sample one (states, sequence) pair of the given length.

    Args:
        length: number of positions L (>= 1).
        rng: a numpy Generator (see `get_rng`).

    Returns:
        (states, sequence): two int arrays of shape (length,). `states` holds hidden
        states (0=alpha, 1=other); `sequence` holds amino-acid indices (0..19).
    """
    if length < 1:
        raise ValueError(f"length must be >= 1, got {length}")

    # Hidden state path: a Markov chain, so state[t] depends on state[t-1] and the
    # loop is inherently sequential. Start is deterministic (`other`, SPEC section 3).
    states = np.empty(length, dtype=np.int64)
    states[0] = OTHER
    for t in range(1, length):
        states[t] = rng.choice(N_STATES, p=TRANSITION[states[t - 1]])

    # Emissions: independent given the states, so draw them vectorized — one call per
    # state fills all of that state's positions at once (2 draws total, independent of L).
    sequence = np.empty(length, dtype=np.int64)
    for s in range(N_STATES):
        mask = states == s
        n = int(mask.sum())
        if n:
            sequence[mask] = rng.choice(N_SYMBOLS, size=n, p=EMISSION[s])

    return states, sequence


def simulate_batch(n, rng, min_len=10, max_len=60):
    """Sample `n` sequences with random lengths drawn uniformly in [min_len, max_len].

    Short lengths are included on purpose: insulin's chains are 21/30 aa, so the model
    must generalize to short sequences (SPEC.md section 5). Lengths vary per sequence, so
    the result is a list of variable-length pairs; padding/`.npz` packaging for training
    is handled later by the Forward-Backward / ground-truth step, not here.

    Args:
        n: number of sequences.
        rng: a numpy Generator (see `get_rng`).
        min_len, max_len: inclusive bounds on the uniform length distribution.

    Returns:
        list of (states, sequence) pairs, each as returned by `simulate_sequence`.
    """
    if min_len < 1 or max_len < min_len:
        raise ValueError(f"require 1 <= min_len <= max_len, got ({min_len}, {max_len})")

    lengths = rng.integers(min_len, max_len + 1, size=n)
    return [simulate_sequence(int(length), rng) for length in lengths]


if __name__ == "__main__":
    # Tiny demo: reproducible, starts in `other`, values in range.
    rng = get_rng(0)
    states, sequence = simulate_sequence(30, rng)
    print("states  :", states)
    print("sequence:", sequence)
