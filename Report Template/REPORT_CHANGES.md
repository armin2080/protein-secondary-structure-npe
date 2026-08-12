# Report Change Log — joint-pipeline branch

This file documents what changed in `Report Template/template/report.tex` for each
person's section, before committing to the `joint-pipeline` branch. It exists so each
member can review "their" part before the changes are committed.

Branch: `joint-pipeline`
Report: `Report Template/template/report.tex`
Compiled PDF: `Report Template/template/report.pdf` (19 pages)

---

## Global / structural changes (apply to everyone)

| Change | Why |
|:-------|:----|
| Title page: real names filled in | Was `Student 2/3/4 name, matriculation number`. Now: Kiana (284599), Samaneh (`[Matriculation number]` placeholder), Armin (**268884**), Geri (`[Matriculation number]` placeholder). **Samaneh & Geri: fill in your matriculation numbers.** |
| `\paragraph{...}` → `\subsection{...}` (6 in §5) | Template requires nothing deeper than subsections. |
| Removed all forced float placement, then re-added standard `[ht]` | Without placement hints LaTeX dumped all figures onto pages 11–17, far from their text. `[ht]` (not `[h]`) is the sane standard and keeps each figure near its section. |
| `\nicefrac{1}{3}` → `$\tfrac{1}{3}$` | `\nicefrac` package was never loaded — this was a compile error. |
| Intro: new Figure 1 (pipeline schematic) | The assignment instructions explicitly require an "illustrative figure" in the Introduction. |
| 0 undefined references, compiles cleanly | Verified with 2-pass pdflatex. |

---

## Kiana — Data (§2) & Statistical model (§3)

**Content verified correct, no substantive changes.**

- Emission table matches `encoding.py` exactly; both rows sum to 100. ✓
- Insulin sequences, mutant B (B16E, B24G, des-B30), helix ranges A2–A7 / A13–A19 /
  B9–B18, fractions 61.9% / 34.5% — all match `insulin.py`. ✓
- HMM matrices, stationary distribution, expected run lengths, and the worked
  Forward-Backward example (G,A) → 0.095238 verified numerically. ✓

**⚠️ One claim to double-check:** §3 "Verification and reproducibility" says the
automated test suite involves **"45 tests"**. The total `assert` statements across
the five check scripts is 59, and the number of check functions doesn't obviously sum
to 45. **Please confirm the exact test count** before submission.

---

## Geri — Approximator (§4)

**Expanded explicit architecture details** (the assignment asks for "how many layers,
activation functions, regularization, etc."):

- `Embedding(21 → 32)` — previously just `Embedding(21)`; the embedding *dimension* is 32.
- Added **2 LSTM layers, dropout 0.2** (regularization — was missing entirely).
- Added CouplingFlow subnet detail: **two-layer hidden subnets of width 128**, affine
  coupling + actnorm, depth selected by search.

No numbers changed; this is purely making the architecture description complete.

---

## Samaneh — Training & model selection (§5)

**Text kept, headings restructured, numbers re-verified.**

- 6 `\paragraph{...}` → `\subsection{...}` (see global changes).
- Numbers confirmed against the branch's actual diagnostics run:
  - Stage-1 pretrain: AdamW lr 1e-3, wd 1e-4, batch 64, grad-clip 1.0,
    ReduceLROnPlateau, ≤60 epochs, patience 8. ✓
  - Stage-2 flow: AdamW lr 5e-4, wd 1e-3, batch 64, 100 epochs, depth 10 via
    6-config search, ~44% exact-zero targets, entropy floor 0.4621 → reached 0.4623. ✓
  - Joint: 4-config search, lr 1e-3, wd 1e-3, depth 6; 33.5 min retrain + ~17 min
    search (≈ 3010 s in manifest). ✓
- §5 "Calibration and coverage" numbers (mean rank 0.528, KS 0.110, 63 clipped,
  z 0.13/0.71, coverage 96.1%) are **consistent with §6** and with the branch's
  current `diagnostics.py` output.

---