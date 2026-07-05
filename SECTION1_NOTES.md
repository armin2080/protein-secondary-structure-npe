# Section 1 — Simulator & Data notes

Handoff notes for the simulator, Forward-Backward ground-truth generator, and insulin
validation data. This covers what each module provides, the conventions everything shares,
the on-disk data contract, and a few things to watch out for when building on top.

## Modules

- **`encoding.py`** — The single source of truth for the amino-acid encoding and the fixed
  two-state HMM parameters (start distribution, transition matrix, emission tables). Every
  other module imports its constants from here, so the canonical ordering and probabilities
  are defined exactly once.
- **`simulator.py`** — The generative HMM sampler. `simulate_sequence(length, rng)` draws one
  `(states, sequence)` pair; `simulate_batch(n, rng, min_len, max_len)` draws a list of them
  with random lengths. All randomness flows through a seeded NumPy `Generator`
  (`get_rng(seed)`), so every draw is reproducible.
- **`forward_backward.py`** — Computes the exact per-position posterior
  `P(state = alpha | sequence)` by running the Forward-Backward algorithm over the known HMM
  (via `hmmlearn.CategoricalHMM` with parameters set manually — never `.fit()`).
  `generate_dataset` pairs simulated sequences with these posteriors, and
  `save_dataset`/`load_dataset` handle on-disk storage.
- **`insulin.py`** — Turns PDB entry 1A7F (human insulin, solution NMR) into the real-world
  validation set. For both chains it produces an integer-encoded sequence and a per-position
  binary alpha-helix label taken from the structure's deposited HELIX annotation, saved in the
  same format as a simulated dataset.

## Shared conventions

- **Amino-acid encoding.** Canonical order is `ARNDCEQGHILKMFPSTWYV` — 20 amino acids, integer
  encoded `A = 0`, `R = 1`, …, `V = 19`. Use `AA_TO_IDX` / `IDX_TO_AA` from `encoding.py`; do
  not redefine the order anywhere else.
- **State convention.** `ALPHA = 0` (alpha-helix), `OTHER = 1` (everything else: beta-sheets and
  coils). These are exported from `encoding.py` as `ALPHA` and `OTHER`.

## Data contract

`save_dataset(path, sequences, posteriors, states)` writes a single `.npz` file with four
arrays:

- `lengths` — `int64`, one entry per sequence, giving each sequence's length.
- `sequences` — `int64`, all sequences concatenated end-to-end (amino-acid indices 0–19).
- `posteriors` — `float64`, all posteriors concatenated (per-position `P(alpha)` in `[0, 1]`).
- `states` — `int64`, all hidden-state paths concatenated (0 = alpha, 1 = other).

Sequences are **variable-length**, so there is no rectangular array and no padding on disk.
The three payload arrays are concatenations; `lengths` records where each sequence ends. This
avoids object arrays and pickling.

`load_dataset(path)` inverts this and returns `(sequences, posteriors, states)` as three lists
aligned by index. For the `i`-th example:

- `sequences[i]` — shape `(L_i,)`, `int` amino-acid indices,
- `posteriors[i]` — shape `(L_i,)`, `float` `P(alpha)` in `[0, 1]`,
- `states[i]` — shape `(L_i,)`, `int` hidden states in `{0, 1}`,

where `L_i` varies from one example to the next.

## What to check / gotchas

- **Padding and masking are not done here.** The dataset stores ragged, variable-length
  sequences. Batching them into padded tensors and masking the padding is the summary/
  architecture step's responsibility (Member 2), not something the data layer handles.
- **`n_features` is pinned on purpose.** `build_hmm` in `forward_backward.py` explicitly sets
  the alphabet size to 20. Without this, `hmmlearn` infers the number of symbols from the
  maximum index it sees in the data, which is wrong for short sequences that happen to omit a
  high-index amino acid (e.g. `V = 19`) and would mismatch the `(2, 20)` emission matrix. Keep
  it pinned.
- **Insulin chain order.** In the insulin dataset, index 0 is chain A and index 1 is chain B.

## Two caveats to keep in mind

1. **Insulin labels are experimental ground truth, not a training target.** For 1A7F the
   "posteriors" slot holds hard `0/1` alpha-vs-other labels read directly from the deposited
   structure — not Forward-Backward posteriors. They exist to be compared against the trained
   network's predicted `P(alpha)` at validation time. Do not train on them.
2. **The network is effectively distilling Forward-Backward.** The HMM parameters are fixed, so
   the Forward-Backward posteriors are a deterministic function of the input sequence — there is
   no observation noise or parameter uncertainty in the labels. The network is learning to
   reproduce a deterministic map, not to invert a stochastic generative model. This matters for
   calibration and SBC (Member 4): the usual "recover a random draw of the parameter" framing
   does not apply directly here, so calibration/SBC setup should account for the fact that the
   target is a fixed function of the input.
