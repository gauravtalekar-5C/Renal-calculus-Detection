# Project review — Renal Calculus Detection & Measurement

**Purpose of this document:** to let you review the pipeline and the code against
what you asked for, and decide whether it meets your needs. Structured as:
what you asked → what exists → where the code is → whether it is proven.

Date: 2026-08-05 · Latest run: `run_full44/` (37 studies)

---

## 1. Your requirements vs what exists

Your original brief, item by item.

| # | you asked for | status | proven? | code |
|---|---|---|---|---|
| 1 | detect / segment calculi in the kidney | **working** | ✅ 79% sens / 92% spec | `utils/detect_stones.py` |
| 2 | count them | **produces a number** | ❌ **never validated** | same, 3D connected components |
| 3 | measure size / volume | **produces a number** | ⚠️ phantom only, not on patients | same, `fwhm_measure()` |
| 4 | location (upper / middle / lower calyx) | **produces a label** | ❌ never validated | same, `kidney_frame()` |
| 5 | HU values | **working** | ✅ direct measurement | same |
| 6 | ureteric stones + distance from UVJ | **not built** | — | deferred, see §7 |
| 7 | fat stranding | **not built** | — | deferred, see §7 |

**Read the "proven?" column carefully.** Items 2, 3 and 4 produce output that looks
authoritative in a CSV but has never been checked against a human. That is not a
statement about whether the code is correct — it is a statement that nobody has
verified it on real patients. Item 3 is phantom-tested, which proves the
arithmetic but not the object selection.

**What is genuinely trustworthy today: items 1 and 5.** Presence, side, and
density.

---

## 2. Pipeline, stage by stage, with code to review

Run with `CALCULUS_RUN=run_full44 ./run_part1.sh`. Every stage is resumable.

### Stage 1 · Series triage — `utils/triage_series.py`

Picks one series out of the ~8 in each study zip.

**Review these decisions:**
- `THIN_MM = 1.5` — slice thickness limit. Justified by phantom test: a 2 mm /
  300 HU stone is invisible at 3 mm. Matches Kambadakone et al.
- Plain only, no contrast. Iodine in urine exceeds 130 HU, same as stone.
- `assign_phase()` precedence: name_plain → name_contrast → header → aorta
  relative to vertebra → aorta absolute. **Review the ordering** — absolute
  aorta HU is last because it is least reliable.

**Output:** `csv/triage_series.csv` (every series, with verdict),
`csv/triage_study.csv` (chosen series per study).

**Result:** 37/44 usable. 7 rejected: 2 at 5 mm, 2 at 2-3 mm, 3 with no usable
series.

### Stage 2 · Patient gate — `utils/patient_gate.py`

**Rule:** keep `PatientAge > 18`.

**Review:** whether excluding all under-18s is acceptable for your use case. It
costs us one good study (a 17-year-old with normal adult kidneys) in exchange for
a cohort where the segmentation model is valid. Data behind it in §5.

**Output:** `csv/patient_gate.csv` — 3 excluded, with reasons.

### Stage 3 · DICOM → NIfTI — `utils/extract_series.py`

**Review the affine construction (lines ~70-95).** This is the code that had the
patient left/right swap. The current version:

```python
affine[:3, 0] = row_dir * dx     # dx = PixelSpacing[1]
affine[:3, 1] = col_dir * dy     # dy = PixelSpacing[0]
affine[:3, 2] = slice_dir * abs(dz)
affine[:3, 3] = origin
affine = np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine
```

Note `dx` comes from `PixelSpacing[1]` and `dy` from `PixelSpacing[0]` — that
transpose is deliberate and is the part that was wrong.

**How to verify it yourself, independent of the code:** open any
`run_full44/kidney_masks/<id>/coronal.png`. The liver must be on the same side as
the **green** (right) kidney. Do not rely on `nib.aff2axcodes()` — it returned the
correct-looking answer throughout the period the volume was mirrored.

### Stage 4 · Kidney segmentation — `utils/run_anatomy.py`

TotalSegmentator, 14 structures. ~2 min/study.

**Review:** the `--roi_subset` list. We request only L1, L5, sacrum and hips for
bone — **not ribs or L2-L4**. That gap is why bone exclusion is done by size
instead (Stage 6). If you would rather exclude bone by label, this list is what
to change.

### Stage 5 · QC + mask overlays — `utils/kidney_qc.py`, `experiment_1/render_kidney_masks.py`

**This is the stage you should exercise first**, because everything downstream
inherits the masks.

Open `run_full44/kidney_masks/<study_id>/coronal.png` for all 37. Roughly 10
minutes of your time, and it is the highest-value review you can do.

QC thresholds to review: volume 90-220 mL, length 80-140 mm, HU 15-55,
asymmetry <2×.

**Currently flagged for your judgement:**

| study | issue | question |
|---|---|---|
| 8379961 | 543 mL total | leakage into liver/bowel, or genuinely large? |
| 8631025 | 454 mL total, right kidney 275 mL | same |
| 8193874 | left kidney 0.7 mL | report says non-visualized left kidney — is our 0.7 mL correct behaviour, or should this study be excluded as unanalysable? |

### Stage 6 · Detection + measurement — `utils/detect_stones.py`

The core. ~4 min/study. **Review these in order of impact:**

**a) The ROI (`build_roi`, ~line 300).** A morphological closing at 15 mm plus a
3 mm cuff. Fills the renal sinus without expanding outward. A plain dilation was
tried and reached into fat, bowel and rib.

**b) The thresholds.**
```python
GROW_HU = 130     # stone extent
SEED_HU = 200     # required dense core  ← see §6, we recommend changing this
```

**c) The denoiser (`denoise_ct`, ~line 250).** SimpleITK curvature anisotropic
diffusion, repeated until components < 200. Verified scale-invariant.

**d) Region growing on RAW, seed-check on denoised** (~line 450). If you reverse
this, small stones are lost — we measured 3 stones → 0 in one study.

**e) Bone rejection (~line 400 and `split_bone_bridges`).** Bone = any dense
(≥300 HU) component ≥3000 mm³. **Review the 3000 mm³ figure** — it is the number
separating "large stone" from "bone", and a staghorn approaching it would be
misclassified.

**f) The phase gate (~line 445).** Refuses enhanced/excretory studies based on
kidney parenchyma HU. **Review the HU window** — this gate has both wrongly
accepted (before it existed) and wrongly rejected (an earlier version) real
cases.

**g) Measurement (`fwhm_measure`, `max_diameter_mm`, `partial_volume_mm3`).**
FWHM boundary + partial-volume integration. Phantom-validated to +0.12 mm /
−2% at 0.7 mm voxels.

**Outputs:**
- `csv/baseline_stones.csv` — accepted stones, one row each
- `csv/candidates.csv` — **every** candidate including rejects, with reason. This
  is the most useful file for review: it shows what the software threw away.
- `csv/baseline_summary.csv` — one row per study

### Stage 7 · Overlays — `utils/render_overlays.py`

`_coronal_mip.png` is the one to open first — a maximum-intensity projection
showing every stone in the kidney in one image, with no slice selection.

### Stage 8 · Scoring — `utils/compare_reports.py`, `utils/seed_sweep.py`

**Review the ground-truth logic (lines ~190-210).** Primary truth is the
spreadsheet's `calculus_type`, negation-corrected; our text regex is a
cross-check. They agree on 35/37.

**Review `NEGATION_RE`** — added because a report saying a stone *"is not
visualized on the present study"* was being scored as a stone present.

**Review `CLAUSE_SPLIT_RE`** — splitting report text on `.` also split decimal
numbers, so "30.2 mm" parsed as "2 mm". Fixed today; it had inflated the size
error to 11.8 mm.

---

## 3. Current results

37 studies analysed, 31 scoreable (3 dye-enhanced, 3 paediatric excluded).

| metric | value | 95% CI | note |
|---|---|---|---|
| sensitivity | **79%** (15/19) | 57-91% | **84% available — see §6** |
| specificity | **92%** (11/12) | 65-99% | |
| side correct | **100%** (15/15) | — | |
| largest renal stone size | median **+1.7 mm**, mean abs **3.2 mm** | | 2/6 within 2 mm |
| stones detected | 57 across the cohort | | |

**The confidence intervals are the honest headline.** 19 positive studies is a
small sample; the true sensitivity is somewhere in 57-91%. State this as "roughly
80% on a small sample", not "79%".

**Remaining size disagreements** (only 6 studies have a renal size in the report):

| study | model | report | comment |
|---|---|---|---|
| 8619669 | 33.3 mm | 30.2 mm | +3.1 mm. Likely a real staghorn; our phantom validation did not cover 30 mm objects, so FWHM may over-grow at this size |
| 8542501 | 4.3 mm | 13.0 mm | largest disagreement — needs review |
| 8440711 | 9.4 mm | 6.0 mm | |

---

## 4. Effect of the four fixes made this week

| fix | effect |
|---|---|
| bone bridging — split instead of discard | fired on 2 studies, **+3 stones** (8440711: 6 → 9). Modest, but they were silent false negatives before |
| negation handling | 8610030 no longer counted as a miss |
| renal-only size comparison | mean abs error 6.3 mm → (11.8 via a bug) → **3.2 mm** |
| decimal-split bug | found while verifying the above. Had made the size metric worse than what it replaced |
| `largest_mm` counting rejects | one study previously showed "0 stones, largest 16 mm"; another 20.19 mm for a real 2.69 mm stone |

**Note on the third and fourth rows:** my renal-only size fix initially made the
metric *worse* (11.8 mm) because it inherited a pre-existing decimal-splitting
bug. That is worth recording as a process point — the fix appeared to fail, and
the reason was a different bug it had exposed rather than caused.

---

## 5. The evidence behind the paediatric exclusion

For review, since it discards data.

| body width at kidney level | kidney volume produced | age | correct? |
|---|---|---|---|
| 199 cm² | 96.7 mL | 7y | ❌ |
| 217 cm² | 35.1 mL | 7y | ❌ |
| 221 cm² | 30.9 mL, 3 fragments | 18y | ❌ |
| 298-841 cm² (34 studies) | all plausible | adults | ✅ |

3/3 below 230 cm² failed. 0/34 above 298 cm² failed. No paediatric task exists in
TotalSegmentator (all 53 checked).

**Consequence if we did not exclude:** study 8591756 (the 31 mL case) reported
**5 stones including one at 20.3 mm** — a clinically significant finding, entirely
fabricated from three fragments of non-kidney tissue.

---

## 6. Decision needed from you: lower `SEED_HU` from 200 to 175

`SEED_HU` is the "a real stone must contain at least one voxel this bright"
threshold. The sweep replays every accept/reject decision at six settings from
the recorded candidate data — exact, not simulated.

| SEED_HU | sensitivity | specificity | false candidates per negative scan |
|---|---|---|---|
| 130 | 84% (16/19) | 83% (10/12) | 0.17 |
| 150 | 84% (16/19) | 83% (10/12) | 0.17 |
| **175** | **84% (16/19)** | **92% (11/12)** | **0.08** |
| 200 ← current | 79% (15/19) | 92% (11/12) | 0.08 |
| 250 | 74% (14/19) | 92% (11/12) | 0.08 |
| 300 | 74% (14/19) | 100% (12/12) | 0.00 |

**175 is strictly better than our current 200: +5 points of sensitivity at
identical specificity and identical false-candidate rate.** There is no trade-off
at that step — 200 is simply the wrong side of a cliff.

**Caveats before you accept it:**
- This optimises *presence per study* on 19 positives. It does not measure
  per-stone accuracy, which needs annotation.
- Choosing a threshold on the same data you evaluate it on is mild overfitting.
  The honest version is "175 looks better and there is no visible cost", not
  "175 is 84% accurate".
- 300 reaching 100% specificity is a 12-study artifact, not a real finding.

**My recommendation:** change to 175, and record that it was chosen on 31
studies and should be re-checked when the cohort grows.

`csv/seed_sweep.csv` has the full table.

---

## 7. What is deliberately not built

| item | why | data ready? |
|---|---|---|
| ureteric stones, UVJ distance | a straight-line ureteric corridor produced ~70 false positives per scan. Needs a learned corridor | yes — 33 of 44 studies are ureteric-tier, and bladder + iliac arteries are already segmented |
| fat stranding | not started | yes — kidney masks give the perinephric shell |
| false-positive classifier | needs annotated candidates | blocked |

---

## 8. What I would want you to check

In priority order, with realistic time cost.

**1. Kidney masks — 10 minutes, highest value.**
`run_full44/kidney_masks/*/coronal.png`, all 37. You are checking one thing: does
the outline look like a kidney, in the right place, on the right side. Everything
in this project rests on it.

**2. Two specific detections — 5 minutes.**
- `overlays/8376148/` — our only false positive. A 2.45 mm / 256 HU stone in the
  left lower pole; the report mentions no renal stone. **If it is real, our
  specificity is 12/12 not 11/12**, and it means reports under-report small
  incidental stones — which biases every specificity figure we quote.
- `overlays/8619669/` — 33.3 mm detection. Genuine staghorn, or several stones
  merged?

**3. Two kidney masks — 5 minutes.**
`kidney_masks/8379961/boundary.png` and `8631025/boundary.png`. Outline-only view.
Is the mask spilling into liver or bowel?

**4. The `SEED_HU` decision — §6 above.**

**5. Sanity-check `candidates.csv` against your expectations.** 25 candidates were
rejected as "not bright enough" and 27 as bone. If your clinical judgement is
that low-density stones matter more than we have assumed, that changes §6.

---

## 9. Honest summary

**What works and is proven:** finding out whether a kidney contains a stone, and
which side, on plain thin-slice adult CT. Roughly 80% sensitivity, 90%
specificity, on a small sample. HU measurement is direct and trustworthy.

**What produces numbers nobody has checked:** stone count, stone volume, and
calyx location. The measurement engine is accurate on synthetic phantoms to a
fifth of a millimetre — but that proves the arithmetic, not that the right objects
were found in a real patient.

**What limits us now.** Not the code. Two things:

1. **Sample size.** 19 positive studies gives sensitivity a 34-point-wide
   confidence interval. No code change narrows that.
2. **Annotation.** 30-50 studies with each stone marked. This is the only route
   to validating count and volume, and it simultaneously unblocks the
   false-positive classifier. The current candidate pool is **106 candidates
   across 22 studies** — small enough to be a checklist rather than a project.

**If you need count and volume to be trustworthy, annotation is the next step and
nothing else substitutes for it.**

---

## 10. Files to review

```
run_part1.sh                    the whole pipeline, 8 stages, ~90 lines
utils/detect_stones.py          the core — start here (~780 lines)
utils/triage_series.py          series selection rules
utils/extract_series.py         affine construction (the L/R bug lived here)
utils/patient_gate.py           paediatric exclusion
utils/compare_reports.py        scoring + report text parsing
utils/seed_sweep.py             offline threshold replay
utils/test_measurement.py       phantom validation — run this to check the maths
experiment_1/render_kidney_masks.py   mask overlays

METHODS.md                      how everything works and why
LINEAR_ISSUES.md                issue-by-issue breakdown for tracking
REVIEW.md                       this file
```

**To reproduce everything from the downloaded data:**

```bash
CALCULUS_RUN=run_full44 ./run_part1.sh
```

**To check the measurement maths independently:**

```bash
./venv/bin/python utils/test_measurement.py
```
