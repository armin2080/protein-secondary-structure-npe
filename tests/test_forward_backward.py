"""Tests for the Forward-Backward ground-truth generator (forward_backward.py).

The independent reference recursion is reused from check_forward_backward (importing
that module is side-effect-free), rather than being reimplemented here.
"""

import numpy as np
import pytest

from check_forward_backward import _reference_fb
from encoding import ALPHA, EMISSION, N_STATES, N_SYMBOLS, START_PROB, TRANSITION
from forward_backward import (
    alpha_posterior,
    build_hmm,
    generate_dataset,
    load_dataset,
    save_dataset,
)
from simulator import get_rng

SEED = 20260705

# Tiny sequences (lengths 2 and 3) exercised against the reference recursion. Kept in
# sync with the worked cases in check_forward_backward.py.
TINY = [[7, 0], [0, 19], [3, 3], [10, 5, 14], [7, 7, 7], [19, 0, 12]]


def test_build_hmm_sets_params_from_encoding():
    model = build_hmm()

    assert model.n_components == N_STATES
    assert np.allclose(model.startprob_, START_PROB)
    assert np.allclose(model.transmat_, TRANSITION)
    assert np.allclose(model.emissionprob_, EMISSION)


def test_build_hmm_pins_n_features_to_20():
    model = build_hmm()
    assert model.n_features == N_SYMBOLS == 20


def test_alpha_posterior_single_sequence_is_valid():
    model = build_hmm()
    obs = np.array([7, 0, 12, 3])
    post = alpha_posterior(obs, model=model)

    assert isinstance(post, np.ndarray)
    assert post.shape == (len(obs),)
    assert post.min() >= 0.0 and post.max() <= 1.0


def test_position_zero_alpha_is_zero():
    model = build_hmm()
    for obs in TINY:
        post = alpha_posterior(np.array(obs), model=model)
        assert post[0] == pytest.approx(0.0, abs=1e-12)


def test_per_position_columns_sum_to_one():
    model = build_hmm()
    obs = np.array([7, 0, 12, 3, 9])
    gamma = model.predict_proba(obs.reshape(-1, 1))  # (L, N_STATES)

    assert gamma.min() >= 0.0 and gamma.max() <= 1.0
    assert np.allclose(gamma.sum(axis=1), 1.0)


def test_hand_computed_length_two_case():
    model = build_hmm()
    obs = [7, 0]  # G, A
    expected = np.array([0.0, 0.00054 / 0.00567])
    got = alpha_posterior(np.array(obs), model=model)
    assert np.allclose(got, expected)


def test_matches_reference_on_tiny_sequences():
    model = build_hmm()
    for obs in TINY:
        ref = _reference_fb(obs)  # full (L, N_STATES) gamma
        hmm = model.predict_proba(np.asarray(obs).reshape(-1, 1))
        assert np.allclose(ref, hmm), f"reference FB vs hmmlearn mismatch on {obs}"

        got = alpha_posterior(np.array(obs), model=model)
        assert np.allclose(got, ref[:, ALPHA]), f"alpha_posterior vs reference on {obs}"


def test_reference_rows_sum_to_one():
    for obs in TINY:
        gamma = _reference_fb(obs)
        assert np.allclose(gamma.sum(axis=1), 1.0)


def test_alpha_posterior_batch_returns_list():
    model = build_hmm()
    batch = [np.array([7, 0]), np.array([3, 3, 3])]
    out = alpha_posterior(batch, model=model)

    assert isinstance(out, list)
    assert len(out) == len(batch)
    for post, seq in zip(out, batch):
        assert post.shape == (len(seq),)


def test_alpha_posterior_single_returns_array():
    model = build_hmm()
    out = alpha_posterior(np.array([7, 0, 3]), model=model)
    assert isinstance(out, np.ndarray)


def test_generate_dataset_is_reproducible():
    model = build_hmm()
    seqs_a, post_a, states_a = generate_dataset(8, get_rng(SEED), model=model)
    seqs_b, post_b, states_b = generate_dataset(8, get_rng(SEED), model=model)

    assert len(seqs_a) == len(seqs_b) == 8
    for a, b in zip(seqs_a, seqs_b):
        assert np.array_equal(a, b)
    for a, b in zip(post_a, post_b):
        assert np.array_equal(a, b)
    for a, b in zip(states_a, states_b):
        assert np.array_equal(a, b)


def test_save_load_roundtrip_is_exact(tmp_path):
    seqs, post, states = generate_dataset(5, get_rng(SEED), min_len=5, max_len=12)

    path = tmp_path / "dataset.npz"
    save_dataset(str(path), seqs, post, states)
    seqs_l, post_l, states_l = load_dataset(str(path))

    assert len(seqs_l) == len(seqs)
    for orig, loaded in zip(seqs, seqs_l):
        assert np.array_equal(orig, loaded)
        assert loaded.dtype == np.int64
    for orig, loaded in zip(post, post_l):
        assert np.array_equal(orig, loaded)
        assert loaded.dtype == np.float64
    for orig, loaded in zip(states, states_l):
        assert np.array_equal(orig, loaded)
        assert loaded.dtype == np.int64
