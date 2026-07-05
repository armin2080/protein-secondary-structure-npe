"""Shared amino-acid encoding and fixed two-state HMM constants.

Every module (simulator, Forward-Backward generator, insulin preprocessing) imports
its encoding and HMM parameters from here so the canonical ordering is defined once.
All numbers are taken verbatim from SPEC.md section 2 — do not redefine them elsewhere.

States:
    0 = alpha-helix   (ALPHA)
    1 = other         (OTHER: beta-sheets + random coils)
"""

import numpy as np

# --- Amino-acid encoding (canonical order, SPEC section 2) -------------------
AMINO_ACIDS = "ARNDCEQGHILKMFPSTWYV"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
IDX_TO_AA = {i: aa for i, aa in enumerate(AMINO_ACIDS)}

N_SYMBOLS = len(AMINO_ACIDS)  # 20

# --- Hidden states -----------------------------------------------------------
ALPHA = 0
OTHER = 1
N_STATES = 2

# --- Start distribution: the sequence always starts in `other` ---------------
# [P(alpha), P(other)]
START_PROB = np.array([0.00, 1.00])

# --- Transition probabilities (row = from-state, col = to-state) -------------
#            to:  alpha  other
TRANSITION = np.array([
    [0.90, 0.10],   # from alpha
    [0.05, 0.95],   # from other
])

# --- Emission probabilities --------------------------------------------------
# Percentages from the SPEC section 2 table, in canonical amino-acid order.
# Each column (state) is a distribution over the 20 amino acids and sums to 100%.
_EMISSION_ALPHA_PCT = [12, 6, 3, 5, 1, 9, 5, 4, 2, 7, 12, 6, 3, 4, 2, 5, 4, 1, 3, 6]
_EMISSION_OTHER_PCT = [6, 5, 5, 6, 2, 5, 3, 9, 3, 5, 8, 6, 2, 4, 6, 7, 6, 1, 4, 7]

emission_alpha = np.array(_EMISSION_ALPHA_PCT, dtype=float) / 100.0
emission_other = np.array(_EMISSION_OTHER_PCT, dtype=float) / 100.0

# Row-per-state emission matrix, shape (N_STATES, N_SYMBOLS); index with EMISSION[state].
EMISSION = np.array([emission_alpha, emission_other])

# --- Sanity assertions: every probability vector must sum to 1 ---------------
assert np.isclose(START_PROB.sum(), 1.0), "START_PROB must sum to 1"
assert np.allclose(TRANSITION.sum(axis=1), 1.0), "each TRANSITION row must sum to 1"
assert np.allclose(EMISSION.sum(axis=1), 1.0), "each EMISSION row must sum to 1"
assert EMISSION.shape == (N_STATES, N_SYMBOLS)
