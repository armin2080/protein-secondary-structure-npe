"""Tests for the insulin 1A7F preprocessing (insulin.py).

The pure functions are tested directly with synthetic inputs. The full parse of the
deposited structure is an integration test that reads the cached mmCIF and skips (rather
than downloading) when it is absent, so the suite never depends on a live network.
"""

import os

import numpy as np
import pytest

import insulin
from encoding import AMINO_ACIDS
from insulin import (
    THREE_TO_ONE,
    WT_INSULIN_A,
    _diff_vs_wildtype,
    alpha_labels,
    build_insulin_dataset,
    helix_ranges,
)

# Repo root is the parent of this tests/ directory; the cached structure lives under it.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHED_CIF = os.path.join(REPO_ROOT, "dataset", "1A7F.cif")


# --- alpha_labels -------------------------------------------------------------

def test_alpha_labels_marks_residues_in_ranges_inclusive():
    residues = [(2, "G"), (3, "I"), (5, "V"), (10, "L")]
    ranges = [(2, 7), (13, 19)]
    labels = alpha_labels(residues, ranges)

    assert np.array_equal(labels, np.array([1, 1, 1, 0]))
    assert labels.dtype == np.int64
    assert len(labels) == len(residues)


def test_alpha_labels_boundaries_are_inclusive():
    residues = [(1, "A"), (2, "A"), (7, "A"), (8, "A")]
    labels = alpha_labels(residues, [(2, 7)])
    assert np.array_equal(labels, np.array([0, 1, 1, 0]))


def test_alpha_labels_empty_ranges_gives_all_zero():
    residues = [(2, "G"), (3, "I"), (10, "L")]
    labels = alpha_labels(residues, [])
    assert np.array_equal(labels, np.zeros(3, dtype=np.int64))


# --- THREE_TO_ONE -------------------------------------------------------------

def test_three_to_one_covers_20_standard_amino_acids():
    assert len(THREE_TO_ONE) == 20
    # Every one-letter value is a single letter, and together they cover the canonical set.
    assert set(THREE_TO_ONE.values()) == set(AMINO_ACIDS)
    assert all(len(one) == 1 for one in THREE_TO_ONE.values())


def test_three_to_one_keys_are_uppercase_three_letter():
    for key in THREE_TO_ONE:
        assert len(key) == 3
        assert key.isupper() and key.isalpha()


def test_three_to_one_spot_checks():
    assert THREE_TO_ONE["ALA"] == "A"
    assert THREE_TO_ONE["TYR"] == "Y"
    assert THREE_TO_ONE["VAL"] == "V"


# --- helix_ranges -------------------------------------------------------------

def test_helix_ranges_list_valued_filters_chain_and_class():
    cif_dict = {
        "_struct_conf.beg_auth_asym_id": ["A", "A", "B"],
        "_struct_conf.beg_auth_seq_id": ["2", "13", "9"],
        "_struct_conf.end_auth_seq_id": ["7", "19", "18"],
        "_struct_conf.pdbx_PDB_helix_class": ["1", "1", "1"],
    }
    assert helix_ranges(cif_dict, "A") == [(2, 7), (13, 19)]
    assert helix_ranges(cif_dict, "B") == [(9, 18)]


def test_helix_ranges_scalar_valued_uses_as_list_branch():
    # A single helix may come back as scalar strings rather than lists.
    cif_dict = {
        "_struct_conf.beg_auth_asym_id": "A",
        "_struct_conf.beg_auth_seq_id": "2",
        "_struct_conf.end_auth_seq_id": "7",
        "_struct_conf.pdbx_PDB_helix_class": "1",
    }
    assert helix_ranges(cif_dict, "A") == [(2, 7)]


def test_helix_ranges_missing_key_returns_empty():
    assert helix_ranges({}, "A") == []


def test_helix_ranges_excludes_non_alpha_class():
    cif_dict = {
        "_struct_conf.beg_auth_asym_id": ["A", "A"],
        "_struct_conf.beg_auth_seq_id": ["2", "20"],
        "_struct_conf.end_auth_seq_id": ["7", "25"],
        "_struct_conf.pdbx_PDB_helix_class": ["1", "5"],  # 5 = not right-handed alpha
    }
    assert helix_ranges(cif_dict, "A") == [(2, 7)]


# --- _diff_vs_wildtype --------------------------------------------------------

def test_diff_vs_wildtype_reports_mutations_and_truncation():
    diffs = _diff_vs_wildtype(insulin.MUTANT_1A7F_B, insulin.WT_INSULIN_B)
    assert diffs == ["B16 Y->E", "B24 F->G", "des-B30 (T removed)"]


def test_diff_vs_wildtype_identical_has_no_diffs():
    assert _diff_vs_wildtype(WT_INSULIN_A, WT_INSULIN_A) == []


# --- integration: parse the cached structure ----------------------------------

@pytest.mark.integration
def test_build_insulin_dataset_from_cached_cif():
    if not os.path.exists(CACHED_CIF):
        pytest.skip(f"cached structure not present: {CACHED_CIF}")

    sequences, posteriors, states, residues_by_chain = build_insulin_dataset(
        cif_path=CACHED_CIF
    )

    seqA, seqB = sequences
    strA = "".join(insulin.IDX_TO_AA[i] for i in seqA)
    strB = "".join(insulin.IDX_TO_AA[i] for i in seqB)

    # Sequence identity (SPEC section 5.4).
    assert len(seqA) == 21
    assert strA == insulin.WT_INSULIN_A
    assert len(seqB) == 29
    assert strB == insulin.MUTANT_1A7F_B

    # Label invariants.
    for labels, seq, st in zip(posteriors, sequences, states):
        assert set(np.unique(labels)).issubset({0.0, 1.0})
        assert len(labels) == len(seq)
        assert np.array_equal(st, 1 - labels.astype(np.int64))

    # Alpha positions match the deposited HELIX ranges: A2-A7, A13-A19 and B9-B18.
    alpha_nums = [
        {num for (num, _), lab in zip(residues, labels) if lab == 1}
        for residues, labels in zip(residues_by_chain, posteriors)
    ]
    assert alpha_nums[0] == set(range(2, 8)) | set(range(13, 20))
    assert alpha_nums[1] == set(range(9, 19))
