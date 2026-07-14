"""Tests for the shared amino-acid encoding and HMM constants (encoding.py)."""

import numpy as np

from encoding import (
    AA_TO_IDX,
    ALPHA,
    AMINO_ACIDS,
    EMISSION,
    IDX_TO_AA,
    N_STATES,
    N_SYMBOLS,
    OTHER,
    START_PROB,
    TRANSITION,
)


def test_canonical_order_is_20_unique_uppercase_letters():
    assert len(AMINO_ACIDS) == 20
    assert len(set(AMINO_ACIDS)) == 20  # all unique
    assert all(c.isupper() and c.isalpha() for c in AMINO_ACIDS)


def test_symbol_and_state_constants():
    assert N_SYMBOLS == 20
    assert N_STATES == 2
    assert ALPHA == 0
    assert OTHER == 1


def test_start_prob_shape_and_sums_to_one():
    assert START_PROB.shape == (2,)
    assert np.isclose(START_PROB.sum(), 1.0)


def test_transition_shape_and_rows_sum_to_one():
    assert TRANSITION.shape == (2, 2)
    # Rows are from-state distributions, so each row (axis=1) sums to 1.
    assert np.allclose(TRANSITION.sum(axis=1), 1.0)


def test_emission_shape_and_rows_sum_to_one():
    assert EMISSION.shape == (N_STATES, N_SYMBOLS) == (2, 20)
    assert np.allclose(EMISSION.sum(axis=1), 1.0)


def test_aa_index_maps_have_20_entries():
    assert len(AA_TO_IDX) == 20
    assert len(IDX_TO_AA) == 20


def test_aa_index_roundtrip_both_directions():
    for aa in AMINO_ACIDS:
        assert IDX_TO_AA[AA_TO_IDX[aa]] == aa
    for i in range(N_SYMBOLS):
        assert AA_TO_IDX[IDX_TO_AA[i]] == i


def test_index_map_matches_canonical_order():
    for i, aa in enumerate(AMINO_ACIDS):
        assert AA_TO_IDX[aa] == i
    # Spot-check the endpoints of the canonical ordering.
    assert AA_TO_IDX["A"] == 0
    assert AA_TO_IDX["V"] == 19
