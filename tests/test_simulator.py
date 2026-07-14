"""Tests for the HMM simulator (simulator.py)."""

import numpy as np
import pytest

from encoding import (
    EMISSION,
    N_STATES,
    N_SYMBOLS,
    OTHER,
    TRANSITION,
)
from simulator import get_rng, simulate_batch, simulate_sequence


def test_simulate_sequence_shapes_and_dtypes():
    length = 25
    states, sequence = simulate_sequence(length, get_rng(0))

    assert states.shape == (length,)
    assert sequence.shape == (length,)
    assert states.dtype == np.int64
    assert sequence.dtype == np.int64


def test_simulate_sequence_value_ranges():
    states, sequence = simulate_sequence(50, get_rng(1))

    assert set(np.unique(states)).issubset({0, 1})
    assert states.min() >= 0 and states.max() < N_STATES
    assert sequence.min() >= 0 and sequence.max() < N_SYMBOLS


def test_simulate_sequence_always_starts_in_other():
    for seed in range(10):
        states, _ = simulate_sequence(15, get_rng(seed))
        assert states[0] == OTHER


def test_same_seed_gives_identical_output():
    states_a, seq_a = simulate_sequence(40, get_rng(123))
    states_b, seq_b = simulate_sequence(40, get_rng(123))

    assert np.array_equal(states_a, states_b)
    assert np.array_equal(seq_a, seq_b)


def test_get_rng_passes_through_a_generator():
    g = get_rng(7)
    assert get_rng(g) is g


def test_get_rng_returns_generator_for_int_and_none():
    assert isinstance(get_rng(0), np.random.Generator)
    assert isinstance(get_rng(None), np.random.Generator)


def test_simulate_sequence_rejects_nonpositive_length():
    with pytest.raises(ValueError):
        simulate_sequence(0, get_rng(0))
    with pytest.raises(ValueError):
        simulate_sequence(-3, get_rng(0))


def test_simulate_batch_returns_list_of_n_pairs():
    n = 12
    batch = simulate_batch(n, get_rng(0), min_len=5, max_len=9)

    assert isinstance(batch, list)
    assert len(batch) == n
    for states, sequence in batch:
        assert states.shape == sequence.shape
        assert states.ndim == 1


def test_simulate_batch_lengths_within_bounds_inclusive():
    min_len, max_len = 4, 8
    batch = simulate_batch(200, get_rng(42), min_len=min_len, max_len=max_len)

    lengths = [len(seq) for _, seq in batch]
    assert min(lengths) >= min_len
    assert max(lengths) <= max_len
    # The bounds are inclusive; over 200 draws in a 5-wide window both ends should appear.
    assert min(lengths) == min_len
    assert max(lengths) == max_len


def test_simulate_batch_sequences_start_in_other():
    batch = simulate_batch(20, get_rng(3), min_len=6, max_len=10)
    for states, _ in batch:
        assert states[0] == OTHER


def test_simulate_batch_rejects_bad_bounds():
    with pytest.raises(ValueError):
        simulate_batch(5, get_rng(0), min_len=5, max_len=3)
    with pytest.raises(ValueError):
        simulate_batch(5, get_rng(0), min_len=0, max_len=4)


@pytest.mark.slow
def test_empirical_frequencies_match_spec():
    """Shrunken, seeded port of sanity_check.py: empirical freqs vs the spec constants.

    Small enough to run in a fraction of a second, with a tolerance far above the
    Monte-Carlo sampling error at this sample size.
    """
    n_seq, seq_len, tol = 100, 200, 0.03
    rng = get_rng(12345)

    trans_counts = np.zeros((N_STATES, N_STATES))
    emit_counts = np.zeros((N_STATES, N_SYMBOLS))
    n_start_other = 0

    for _ in range(n_seq):
        states, sequence = simulate_sequence(seq_len, rng)
        n_start_other += int(states[0] == OTHER)
        np.add.at(trans_counts, (states[:-1], states[1:]), 1)
        np.add.at(emit_counts, (states, sequence), 1)

    # Every sequence must start in `other`.
    assert n_start_other == n_seq

    emp_trans = trans_counts / trans_counts.sum(axis=1, keepdims=True)
    emp_emit = emit_counts / emit_counts.sum(axis=1, keepdims=True)

    assert np.abs(TRANSITION - emp_trans).max() < tol
    assert np.abs(EMISSION - emp_emit).max() < tol
