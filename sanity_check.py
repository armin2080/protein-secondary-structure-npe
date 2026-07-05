"""Sanity check: empirical simulator frequencies vs the spec constants.

Simulates many long sequences and confirms that the empirical start, transition, and
emission frequencies match the fixed HMM parameters in `encoding.py` within Monte-Carlo
error. Run directly:  python sanity_check.py
"""

import numpy as np

from encoding import (
    EMISSION,
    N_STATES,
    N_SYMBOLS,
    OTHER,
    TRANSITION,
)
from simulator import get_rng, simulate_sequence

N_SEQ = 2000       # number of sequences
SEQ_LEN = 500      # length of each (long -> plenty of both states in steady state)
SEED = 12345
TOL = 0.01         # Monte-Carlo tolerance (well above the ~0.001 sampling std error)


def collect(n_seq, seq_len, rng):
    """Simulate sequences and accumulate transition / emission / start counts."""
    trans_counts = np.zeros((N_STATES, N_STATES))     # [from, to]
    emit_counts = np.zeros((N_STATES, N_SYMBOLS))     # [state, amino acid]
    n_start_other = 0

    for _ in range(n_seq):
        states, sequence = simulate_sequence(seq_len, rng)

        # Value-range guards.
        assert states.min() >= 0 and states.max() < N_STATES
        assert sequence.min() >= 0 and sequence.max() < N_SYMBOLS

        n_start_other += int(states[0] == OTHER)

        # Transition counts: pair each state with its successor.
        np.add.at(trans_counts, (states[:-1], states[1:]), 1)
        # Emission counts: pair each state with its emitted amino acid.
        np.add.at(emit_counts, (states, sequence), 1)

    return trans_counts, emit_counts, n_start_other


def _report(name, expected, empirical):
    """Print an expected-vs-empirical table and return the max abs deviation."""
    dev = np.abs(expected - empirical)
    print(f"\n=== {name} (max abs deviation {dev.max():.4f}) ===")
    for i, (e_row, m_row) in enumerate(zip(expected, empirical)):
        exp_str = " ".join(f"{v:5.3f}" for v in e_row)
        emp_str = " ".join(f"{v:5.3f}" for v in m_row)
        print(f"  row {i}: expected [{exp_str}]")
        print(f"         empirical[{emp_str}]")
    return dev.max()


def main():
    rng = get_rng(SEED)
    trans_counts, emit_counts, n_start_other = collect(N_SEQ, SEQ_LEN, rng)

    # Start distribution: every sequence must start in `other`.
    start_frac_other = n_start_other / N_SEQ
    print(f"start-in-other fraction: {start_frac_other:.4f} (expected 1.0000)")
    assert start_frac_other == 1.0, "every sequence must start in `other`"

    # Row-normalize counts into empirical probabilities.
    emp_trans = trans_counts / trans_counts.sum(axis=1, keepdims=True)
    emp_emit = emit_counts / emit_counts.sum(axis=1, keepdims=True)

    max_trans_dev = _report("Transition matrix", TRANSITION, emp_trans)
    max_emit_dev = _report("Emission matrix", EMISSION, emp_emit)

    print(f"\nTolerance: {TOL}")
    assert max_trans_dev < TOL, f"transition deviation {max_trans_dev:.4f} exceeds {TOL}"
    assert max_emit_dev < TOL, f"emission deviation {max_emit_dev:.4f} exceeds {TOL}"
    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    main()
