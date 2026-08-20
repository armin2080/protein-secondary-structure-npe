# Report changes — 12 Aug 2026 (for Persons 1–3)

Branch: `joint-pipeline`.

This lists the changes I (Armin / Person 4) made **today in your sections**,
relative to the older commit `729c863` in this branch. Everything below is
**wording / layout only — no numbers, tables, figures, or code identifiers were
changed.** Please spot-check your part.

Quick way to check the diff yourself:

```bash
git diff 729c863 -- "Report Template/template/report.tex"
```

---

## Kiana: Data (§2) & Statistical model (§3)

| What I changed | Why |
|:---------------|:----|
| Compressed the prose in §2 (opening, canonical representation, simulated data, `.npz` handoff, insulin parsing, hard-labels paragraph) and in §3 (HMM intro, transition paragraph, emission paragraph, `hmmlearn` paragraph, verification, interpretation) | The report text was over the **10-page limit** — pure wording trims |
| Merged the "Generative process" subsection into §3.1, and merged "Independent correctness check" + "Verification and reproducibility" into one "Verification" subsection | Same content, fewer headings → saves space |
| Fixed an overflowing line: `{\_struct\_conf}` → `\texttt{struct\_\allowbreak conf}` (§2) | LaTeX line stuck out past the margin |
| Added `\label{sec:stat}` and made §2 reference Section 3 | Needed for a new cross-reference |
| **Please double-check:** I removed the claim of "45 tests" in §3 — I could not verify that count (the check scripts contain 59 asserts). Re-add the real number if you have it | Correctness: no unverifiable number in the report |
| Added one explicit "prior" sentence at the end of §3.1 (no prior over parameters, but an explicit proper stationary prior over the hidden states, ≈ 33% α-helix) | Grader rubric explicitly grades "Explicit proper priors" — the sentence lets them check it off |
| Data-layer table (§2): second column changed to ragged-right (`>{\raggedright\arraybackslash}p{...}`) | Fixed an Underfull \hbox warning |

**Please check these are intact:** amino-acid order, length range 10–60, 2000/400
split, insulin sequences + B16E/B24G/des-B30, helix ranges, 61.9% / 34.5%, HMM
matrices, run lengths 10/20, μ = (1/3, 2/3), verification numbers (0.0004 / 0.0007,
the worked (G,A) → 0.0952 example, seed 12345).

---

## Geri: Approximator (§4)

| What I changed | Why |
|:---------------|:----|
| Compressed the architecture prose (summary-network bullet, `position_head` / padding sentence, Stage 1 & 2 items, raw-logits sentence, joint subsection) | Page-limit trim — wording only |
| Moved the architecture chain into a display equation | The inline line overflowed the margin; content identical |
| Added `\allowbreak` inside `pack_padded_sequence` and `keras.layers.TorchModuleWrapper` | Long identifiers overflowed the margin |

**Please check:** embedding 21→32, 2 BiLSTM layers (dim 64), dropout 0.2, width-128
subnets, actnorm, depth-by-search, padding token 20, `BasicWorkflow.fit_offline` —
all unchanged.

---

## Samaneh: Training (§5)

| What I changed | Why |
|:---------------|:----|
| Compressed the prose (two-stage design, Stage 1, Stage 2, Convergence, Joint training) | Page-limit trim — wording only |
| Removed the "Calibration and coverage" subsection and replaced it with one pointer sentence to Section 6 | It repeated §6 almost verbatim; all its numbers are still in §6 |
| Re-sized the two plots of Figure 3 (the joint-pipeline figure) so they align vertically | The wide loss curve and the bar chart had different heights and looked misaligned |
| `fit_offline` got an `\allowbreak` | Long identifier overflowed the margin |

**Please check:** 2000/400 sequences, length 10–60 padded to 60, all hyperparameters,
~44% exact-zero targets, 6-config search → depth 10, entropy floor 0.4621 → 0.4623,
joint 4-config search → lr 1e-3 / wd 1e-3 / depth 6, 33.5 min / ~17 min — all unchanged.

---
