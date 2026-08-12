# Report Change Log — joint-pipeline branch

This file documents what changed in `Report Template/template/report.tex` for each
person's section, before committing to the `joint-pipeline` branch. It exists so each
member can review "their" part before the changes are committed.

Branch: `joint-pipeline`
Report: `Report Template/template/report.tex`
Compiled PDF: `Report Template/template/report.pdf` (18 pages)

> **Second pass — page-limit trim (below).** The first pass (sections further
> down) filled in names, fixed compile errors, and made the content template
> compliant. This second pass shortens the prose to meet the **≤ 10 pages of
> text** requirement (title, figures, tables, and references are excluded from
> the count).

---

## Second pass — page-limit trim (everyone)

**Goal and outcome.** The report was ~6,300 words ≈ ~14 text pages. It is now
**~4,250 body words ≈ ~9.4 text pages** (compiled PDF: 18 total pages, of which
1 title + 1 references). This is comfortably within the 10-page text limit.

**The most important guarantee: no numbers, no tables, no figures, and no code
identifiers were changed anywhere.** Only prose was compressed/reworded, and the
wording-only edits below are listed so each person can re-verify their section
in seconds.

### Global (all sections)

| Change | What to check |
|:-------|:--------------|
| Added `\label{sec:stat}` (§3) and `\label{sec:diag}` (§6) | Needed so new cross-references (`Section~\ref{sec:stat}` in §2, `Section~\ref{sec:diag}` in §5) resolve. Verified: 0 undefined references. |
| BiLSTM architecture chain moved from inline math into a `multline*` display equation (§4) | Pure formatting — the chain `Embedding(21→32) → BiLSTM(64, 2 layers, dropout 0.2) → masked mean pooling → Linear+ReLU → summary(64)` is unchanged. Fixes an overfull line. |
| `\texttt{fit\_offline}` and `\texttt{pack\_padded\_sequence}` now contain `\allowbreak` (§4, §5) | Wording-only; fixes two overfull `\hbox` warnings. |
| Intro roadmap paragraph compressed | 5 sentences → 1 sentence. Content identical, just tighter. |

### Kiana — Data (§2)

| Where | What changed | What to check |
|:------|:-------------|:--------------|
| Canonical representation | Removed the separate paragraph about `hmmlearn` compatibility; the "same representation for simulation and insulin" point is now one sentence. | Alphabet order, 20-symbol claim, `encoding.py` reference — unchanged. |
| Simulated training data | The 4-sentence generation description → 2 sentences ("draws a state path … emits one amino acid per position"). Removed the clause "Dataset size is not intrinsic to the simulator". State-path sentence shortened. | **Numbers unchanged:** length range 10–60, 2000 train / 400 validation, seeded NumPy `Generator`. |
| Insulin 1A7F parsing | mmCIF/Biopython paragraph compressed (dropped the per-step "three-letter → one-letter → integer" description). Hard-labels paragraph compressed. | **Sequences, B16E/B24G/des-B30 substitutions, helix ranges A2–A7 / A13–A19 / B9–B18, fractions 61.9% / 34.5% — unchanged.** |

### Kiana — Statistical model (§3)

| Where | What changed | What to check |
|:------|:-------------|:--------------|
| Independent correctness check | The worked (G,A) example was three display equations; now inline in one paragraph. | **All numbers preserved:** `f₂(0)=0.09×0.05×0.12=0.00054`, `f₂(1)=0.09×0.95×0.06=0.00513`, posterior `0.0952…`, `P(S₁=0)=0` from the deterministic start, six extra sequences. |
| Verification and reproducibility | Two paragraphs → two sentences. | **All numbers preserved:** 2000 seq × length 500, seed 12345, deviations 0.0004 / 0.0007, threshold 0.01, 200 FB sequences, `[0,1]` / sums-to-one / zero-α-at-pos-0 checks, round-trip reproducibility. |
| ⚠️ **"45 tests" claim removed** | Old text said the automated suite involves *"45 tests"*. We could not verify that count (the check scripts contain 59 asserts, and no function count sums to 45), so the sentence now reads "the automated test suite (Section 2–3) all passes". | **Please confirm the real test count and re-add it if you have it** (or leave the general phrasing). |
| Interpretation and limitations | Compressed; deterministic-function point, collapsed β/coil state, non-position-dependent emissions, deterministic start, hard vs. soft labels — all still stated. | No numbers involved. |

### Geri — Approximator (§4)

| What changed | What to check |
|:-------------|:--------------|
| Architecture chain → display equation; `pack_padded_sequence` sentence lightly reworded with `\allowbreak`. | Architecture details (21→32 embedding, 2 LSTM layers, dim 64, dropout 0.2, width-128 subnets, actnorm, depth by search) — **unchanged**. |
| Joint subsection closing: "Everything else is identical (dataset, …)" — dropped the repeated `(Section~\ref{sec:training})` and reworded slightly. | No content removed. |

### Samaneh — Training (§5)

| Where | What changed | What to check |
|:------|:-------------|:--------------|
| Two-stage design | Removed the explanatory clause "the summary network has a direct supervised signal to fit and by the time the flow trains, that signal is no longer shifting underneath it" (the point is implied by the preceding sentence). | **All numbers unchanged:** 2000/400 sequences, length 10–60 padded to 60. |
| Stage 2 | "One subtlety worth flagging" → "One subtlety"; "Picking the best configuration by this loss would just reward …" → "selecting by it would reward …". | **All numbers unchanged:** ~44% exact-zero targets, six-config search, depth 10. |
| Convergence | "which looks like stalling but it isn't" → "which is not stalling"; trimmed two redundant phrases. | **All numbers unchanged:** plateau ≈0.46, floor 0.4621, reached 0.4623, gap 0.0003. |
| Calibration and coverage | **Heavily compressed (~150 → ~90 words).** Removed the detailed z-score-clipping description and the boundary-effect explanation — both are fully covered in §6 — and now point to `Section~\ref{sec:diag}`. | **All numbers kept:** 200 sequences, 6764 positions, mean rank 0.528, KS 0.110, z 0.13 / 0.71, 96.1% coverage at nominal 90%. |
| Joint training | `fit_offline` got an `\allowbreak` (formatting only). | Numbers unchanged (4-config search, lr 1e-3, wd 1e-3, depth 6, 33.5 min / ~17 min). |

### Armin — Diagnostics (§6) & Discussion (§8)

| Where | What changed | What to check |
|:------|:-------------|:--------------|
| §6 header | Added `\label{sec:diag}` (no visible change). | — |
| SBC | Wording trims only ("Both histograms are close to uniform" → "Both are close to uniform"; dropped "of the flow", "of the interval"). | **All numbers unchanged:** 0.528 / 44.2% / 0.110 and 0.519 / 45.0% / 0.132, ~3% spike, KS comparison. |
| Z-score paragraph | Removed "Inspection shows", "the Forward--Backward truth", "so the posterior systematically underestimates the true value there", and the final "the flow no longer collapses…" clause. | **All numbers unchanged:** 200 of 6764 excluded, 63 of 6564 (~1%, max \|z\|≈3137), 83% at norm. pos. ≥ 0.85, P(α)∈[0.3,0.8], z 0.129 / 0.709; joint 18 positions, median 0.83, max \|z\|=10.2, z 0.105 / 0.761. |
| §8 "What BayesFlow contributes" | The 3-item bullet list was merged into one paragraph. Removed "for a density estimator to recover", "enabling uncertainty quantification at each residue", and the example "such as inferring the HMM parameters themselves". | **All numbers unchanged:** 0.9997 vs 0.778, 0.987, chain A ≈0.4, chain B ≈0.85, Table refs. |
| §8 "Potential improvements" | Several sentences compressed (e.g. "Several directions follow directly…" → "The most impactful improvement…"). | All four improvement ideas still present (richer simulator, transformer encoder, online simulation, protein panel). |
| §8 "Reflection on own learnings" | Compressed ~150 → ~110 words. | All five points preserved (SBI idea, staged-vs-joint lesson, hand-built diagnostics, reproducibility, insulin ceiling). |

**How to verify the page count yourself:** compile with `pdflatex` (2 passes), or
in the PDF: 1 title page + 1 references page + 16 content pages = 18 total.
The prose itself is ~9.4 pages of the 10-page allowance.

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