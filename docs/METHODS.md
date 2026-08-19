# Renal Calculus Detection & Measurement — Part 1 (kidney only)

End-to-end record of what was built, why each decision was made, and what is
and is not validated. Written to be copied into project documentation.

Project root: `/root/Gaurav/kindey_calculus_measurement/`
Date of this record: 2026-08-04

---

## 1. Objective and scope

**Full brief:** detect and segment calculi on CT, count them, measure volume,
give the location (upper / middle / lower calyx), report HU values, and for
ureteric stones give the distance from the UVJ. Plus perinephric fat stranding.

**Part 1, delivered here:** everything above *inside the kidney only*.

Deliberately deferred to Part 2:

| deferred | reason |
|---|---|
| ureteric stones, UVJ distance | a straight-line ureteric corridor cuts through bowel and iliac vessels and produced ~70 false positives per scan. Needs a learned corridor, not a stand-in. |
| fat stranding | separate task, not started |
| CNN false-positive filter | needs annotated data (see §8) |

---

## 2. Data

**Source:** internal API, `https://api.5cnetwork.com/dicom/download/{study_iuid}`.
Cohort defined in `calculus.xlsx` (24,774 report rows), which carries
`calculus_flag`, `calculus_line` (the report text) and `calculus_type`
(location label: `renal`, `ureteric`, `VUJ`, `other`, or combinations).

**Downloaded:** 44 studies, 70,643 DICOM files, 14.6 GB.

| tier (from report text) | studies |
|---|---|
| ureteric | 33 |
| urography | 10 |
| renal | 1 |

**A hard constraint discovered empirically: the API has a ~34-day retention
window.** Studies older than that fail with HTTP 504 after the gateway's 300 s
timeout. This was initially misdiagnosed as a concurrency problem; a clean step
function (28 Jun fails / 29 Jun succeeds) disproved that. Consequences baked
into the downloader:

- read timeout set to 330 s, *above* the server's 300 s limit — a shorter
  timeout masks the real 504 as a client-side error
- no retry on 502/503/504, because cold-storage studies fail identically every
  time. Retrying wasted ~19 min per study.
- `probe_api.py` aborts each request after 4 MB, so the current cutoff can be
  re-measured cheaply before any bulk pull

Filenames use `study_id`, never the server's `Content-Disposition`, which
contains the patient name (PHI).

---

## 3. Pipeline

One command, resumable, every stage skipping already-finished studies:

```bash
CALCULUS_RUN=run_full44 ./run_part1.sh
```

Results go under `$CALCULUS_RUN`; `nifti/` and `seg/` are shared at the project
root because they are expensive to rebuild and identical across analyses.

### Stage 1 — Series triage (`utils/triage_series.py`)

A study zip contains many series (plain, contrast, delayed, thick, scout). Pick
the one usable for measurement.

**Rules:** plain (non-contrast), axial, slice spacing ≤ 1.5 mm, coverage
≥ 250 mm, ≥ 40 slices.

**Why ≤ 1.5 mm:** partial-volume averaging on thick slices dilutes a small
stone's peak below any detection threshold. Our phantom test showed a
**2 mm / 300 HU stone is missed entirely at 3 mm slice thickness**. This
independently reproduces the thin-slice recommendation in Kambadakone et al.

**Why plain, not contrast:** iodinated contrast in the collecting system exceeds
130 HU and is indistinguishable from a calculus by density.

Phase detection precedence: series description (plain) → series description
(contrast) → header agent → aorta HU relative to vertebral body → absolute
aorta HU. Absolute aorta HU alone is unreliable, hence the ordering.

**Result on our data:** 37 of 44 measurable; 2 too thick (5 mm), 2 detect-only
(2–3 mm), 3 unusable. Of the 37, 14 are finer than 0.75 mm and 23 fall in
0.75–1.5 mm.

### Stage 2 — Patient gate (`utils/patient_gate.py`)

**Keep age > 18 y**, read from the DICOM `PatientAge` header.

**Why:** TotalSegmentator is trained on an overwhelmingly adult cohort and fails
on children. Measured:

| body cross-section at kidney level | kidney volume produced | age |
|---|---|---|
| 199 cm² | 96.7 mL ❌ | 7 y |
| 217 cm² | 35.1 mL ❌ | 7 y |
| 221 cm² | 30.9 mL ❌ (3 disconnected fragments) | 18 y, small stature |
| 298 – 841 cm² | all plausible ✅ | adults |

3 of 3 below 230 cm² failed; 0 of 34 above 298 cm² failed. There is **no
paediatric task in TotalSegmentator** (all 53 tasks checked), so these cannot be
rescued by switching models.

Age alone catches all three failures, so age is the only exclusion criterion.
Body cross-section is still measured and recorded because it explains *why* the
model fails, but it does not decide anything. An earlier version excluded on
both and wrongly dropped a 17 y patient with a 519 cm² adult abdomen and a
normal 232 mL kidney volume.

**This runs before extraction and segmentation**, so an excluded study costs no
GPU time.

2 of 44 studies have no age in the header (one contains the placeholder
`000Y`); those are kept and flagged rather than dropped.

### Stage 3 — DICOM → NIfTI (`utils/extract_series.py`)

Stack the chosen series into a volume in Hounsfield units.

**The affine is the subtle part and was wrong for a long time.** The NIfTI
affine is built from `ImageOrientationPatient` (row and column direction
cosines), their cross product for the slice direction, and
`ImagePositionPatient` for the origin, then converted LPS → RAS.

The original bug assigned `row_dir` and `col_dir` to the wrong affine columns,
which **silently swapped the patient's left and right.** The header still read
`('L','P','S')`, so no automated check caught it. It was found visually: the
liver appeared on the image-left (proving image-left = patient right) while the
blue `kidney_left` contour was drawn on the same side.

**Lesson worth documenting: never trust `aff2axcodes` alone. Render one overlay
and confirm the liver is on the patient's right.**

Array layout after extraction: axis 0 → patient LEFT, axis 1 → POSTERIOR,
axis 2 → SUPERIOR.

### Stage 4 — Kidney segmentation (`utils/run_anatomy.py`)

TotalSegmentator, `--roi_subset` of 14 structures: `kidney_left`,
`kidney_right`, `kidney_cyst_left/right`, `urinary_bladder`, `aorta`,
`inferior_vena_cava`, `iliac_artery_left/right`, `vertebrae_L1`, `vertebrae_L5`,
`sacrum`, `hip_left`, `hip_right`.

~2 min per study on an A100 (longer when the GPU is shared).

### Stage 5 — Segmentation QC and mask overlays

`utils/kidney_qc.py` flags per kidney: volume outside 90–220 mL, length outside
80–140 mm, median HU outside 15–55, or >2× asymmetry.

`experiment_1/render_kidney_masks.py` renders 5 images per study:

| image | question it answers |
|---|---|
| `coronal.png` | are both kidneys found, right shape, right place? (**check first**) |
| `sagittal.png` | are the poles included, or clipped? |
| `boundary.png` | outline only — does the edge follow the capsule, or leak? |
| `axial_grid.png` | 12 slices: poles + biggest mask-area jumps (failure-prone) |
| `axial_grid_even.png` | 12 evenly spaced, for a fair overview |

Left kidney blue, right green, radiological orientation.

**Rationale for doing this as its own stage:** every downstream number inherits
these masks, so they are verified on their own before any stone result is
believed. This is what caught the affine bug and the paediatric failures.

### Stage 6 — Stone detection and measurement (`utils/detect_stones.py`)

Follows Elton et al. 2022 (`github.com/rsummers11/Renal-Calculi`, PMC10407943),
steps 2–9, with our own additions.

**6a. Region of interest — a morphological closing, not a dilation.**

```
dilate kidney mask by 15 mm  →  erode back by 15 mm   (= closing)
plus a 3 mm cuff around the kidney itself
```

A closing fills the concave renal sinus / pelvis — where stones actually sit —
without expanding outward. A plain 12 mm dilation was tried first and swallowed
perinephric fat, bowel and rib, producing detections outside the kidney.

**6b. Denoising — gradient (curvature) anisotropic diffusion.**

SimpleITK `CurvatureAnisotropicDiffusionImageFilter`, applied repeatedly until
the number of connected components above 130 HU drops below 200 (Elton's
criterion), max 10 rounds. Falls back to `TimeStep = 0.03125` (the 3D CFL limit)
if the default is rejected.

Verified properties: conductance is scale-invariant
(`denoise(10·I) == 10·denoise(I)` to 0.000%), so `ConductanceParameter = 3.0`
means "3× this image's own typical gradient". On a 5³ phantom, voxels ≥ 130 HU
went 207 → 125 (= 5³ exactly) while the stone peak stayed at 800.0 across all
rounds — noise removed, signal untouched.

A hand-written Perona–Malik implementation was tried first and **amplified**
noise (375 → 180,419 noise voxels) because of a sign error in the divergence
(`flux[i-1] − flux[i]` instead of `flux[i] − flux[i-1]`), which is backward
diffusion. Replaced entirely with the library filter.

**6c. Candidate generation — hysteresis threshold.**

```
seed  200 HU   every real stone must contain at least one voxel this bright
grow  130 HU   defines the stone's full extent
```

At a bare 130 HU (what Elton uses, because their CNN cleans up afterwards) we
measured **81 false stones per scan**, median peak 139 HU. Hysteresis reduced
that to 9. **The 200 HU seed is a crutch for the missing CNN, and it costs
sensitivity** — see §8.

**Region growing runs on the RAW volume, not the denoised one.** Growing extent
on denoised data lost real 2 mm calculi (8584188: 3 stones → 0). The denoised
volume is used only to decide *where* candidates are; the raw volume decides how
big they are.

**6d. Rejection rules** (our stand-in for Elton's CNN; every rejected candidate
is kept in `candidates.csv` with its reason, because that is the labelling pool
Part 2 needs):

| rule | threshold | why |
|---|---|---|
| `no_dense_core` | seed peak < 200 HU | noise, low-density non-stone |
| `bone_partial_volume` | >50% of voxels within 2 mm of bone | rib/vertebra cortex read as 15–45 mm "stones" |
| `vascular_calcification` | >50% within 3 mm of aorta/IVC/iliac | plaque |
| `below_min_diameter` | < 1.5 mm | below resolution |
| `below_min_volume` | < 0.25 mm³ | noise |

Bone is identified as **any dense (≥300 HU) connected component ≥ 3000 mm³**,
not from segmentation labels. Reason: only L1/L5/sacrum/hips were requested from
TotalSegmentator, so L2–L4, ribs and femurs were invisible and their cortex was
reported as stones. Size is the robust discriminator — even a large staghorn is
a few hundred mm³.

**Bone bridging fix.** At 130 HU a stone lying against a rib is joined to it by
partial-volume voxels, making one component that is majority bone. Rejecting the
component discarded the stone — a *silent* false negative. Now the component is
**split**: voxels near bone become one candidate (still rejected, preserving the
audit trail), and the remainder is re-labelled into pieces judged on their own
merits. No voxel is lost, so all-bone components behave exactly as before.

**6e. Phase gate.** If median kidney parenchyma HU is outside the unenhanced
range, the study is refused with
`"enhanced or excretory phase - not analysable for stones"`.

Gated on **parenchymal** HU, not aortic HU: in the excretory phase the aorta has
washed out while the collecting system fills, so aortic HU looks normal. Study
8285221 (kidney HU 96) was reported as 9 stones before this gate existed. An
earlier version also used a dense-volume criterion and wrongly rejected a real
staghorn case (8261985, 4435 mm³ of genuine stone); the volume criterion was
dropped.

**6f. Measurement.**

- **Boundary:** full-width-half-maximum between the stone's peak (95th
  percentile) and the local background.
- **Volume:** partial-volume integration — each voxel in a 1.0 mm shell
  contributes `clip((HU − bg) / (peak − bg), 0, 1)` rather than 0 or 1.
- **Diameter:** maximum extent plus `0.5 ×` mean in-plane spacing.
- **Location:** a kidney-local coordinate frame oriented superiorly;
  `t ≈ 0` → lower pole, `t > 2/3` → upper pole, else interpolar.
- **HU:** max, mean, SD, p90 over the refined extent.

**Phantom validation** (synthetic spheres with supersampled partial volume, PSF
blur and noise):

| voxel size | diameter error | volume error |
|---|---|---|
| 0.7 mm isotropic | +0.12 mm (abs 0.22) | −2% (abs 6%) |
| 0.8 × 0.8 × 1.25 mm | +0.19 mm | +0% (abs 9%) |

Performance note: an early version was 10× too slow because
`binary_dilation` ran over the full volume inside the per-candidate loop.
Replaced with one `distance_transform_edt` per structure plus per-stone bounding
-box cropping: 10 min → 4 min per study.

### Stage 7 — Stone overlays (`utils/render_overlays.py`)

| image | slice chosen |
|---|---|
| `_coronal_mip.png` | **no slice** — maximum-intensity projection, so every stone in the kidney appears in one image |
| `_kidney_axials.png` | 12 slices: half chosen by mask-area jump + poles, half evenly spaced |
| `stone_NN.png` | the axial slice through that stone's centroid |
| `_contact_sheet.png` | all of the above together |

**On slice selection generally: detection itself uses every slice containing
kidney (typically 100–200).** A single slice cannot work — a 3 mm stone spans
~5 of 150 slices, counting requires 3D 26-connectivity, and volume requires all
voxels. Slices are picked only for display.

### Stage 8 — Scoring against reports (`utils/compare_reports.py`, `utils/seed_sweep.py`)

Ground truth is the spreadsheet's `calculus_type` (negation-corrected), with a
regex over `calculus_line` kept as an independent second opinion in a separate
column. They agree on 35 of 37 studies.

**Negation handling** was required. Study 8610030 reads *"The previously
described 5 mm calculus in the interpolar calyx of the right kidney **is not
visualized on the present study**"* — the stone has passed. We correctly found
nothing and were scored as a false negative. The spreadsheet's `calculus_type`
has the same blind spot (it says `"ureteric, renal"`), consistent with both
being derived from the same prose. `NEGATION_RE` covers "not visualized", "no
longer seen", "has passed", "resolution of", "no residual calculus".

**Size comparison is restricted to renal-only sizes.** Comparing our kidney
measurement against the largest size anywhere in the report was meaningless —
8506983's biggest number is a 13 mm **gallbladder** stone, and several studies'
largest is ureteric. That inflated mean absolute error to 6.3 mm while the
median was −1.5 mm; the spread was contamination, not measurement error.

`seed_sweep.py` replays the whole accept/reject chain offline from
`candidates.csv` at SEED_HU = 130 / 150 / 175 / 200 / 250 / 300. This is exact,
not approximate, because SEED_HU is used in exactly one comparison and every
candidate ever generated is recorded with the peak that comparison uses
(`seed_peak_hu`). Six settings measured without re-running detection once.

---

## 4. Key parameters

```python
# triage
THIN_MM = 1.5;  THICK_MM = 3.0;  MIN_COVERAGE_MM = 250;  MIN_SLICES = 40

# patient gate
MIN_AGE_Y = 18.0

# ROI
SINUS_FILL_MM = 15      # closing radius, fills the renal sinus
CAPSULE_CUFF_MM = 3

# thresholding
GROW_HU = 130           # stone extent
SEED_HU = 200           # required dense core
MIN_VOXELS = 3;  MIN_DIAM_MM = 1.5;  MIN_CANDIDATE_MM3 = 0.25

# rejection
BONE_HU = 300;  BONE_MIN_VOL_MM3 = 3000;  BONE_MARGIN_MM = 2.0
VESSEL_MARGIN_MM = 3.0
MAX_STONE_DIAM_MM = 30  # above this, flag for review rather than report

# denoising
DENOISE_ITERS = 1;  DENOISE_MAX_ROUNDS = 10;  DENOISE_TARGET_CC = 200
CLIP_LOW, CLIP_HIGH = -200.0, 1000.0
```

---

## 5. Results

On 31 analysable adult studies (44 downloaded − 7 unusable series − 3 contrast
phase − 3 paediatric):

| metric | value | 95% CI |
|---|---|---|
| **sensitivity** (presence of an intrarenal calculus) | **79%** (15/19) | 57–91% |
| **specificity** | **92%** (11/12) | 65–99% |
| **side correct** | **100%** (15/15) | — |
| largest-stone size error | median −1.5 mm | — |

**The confidence intervals are the important part.** With 19 positive studies the
true sensitivity could be anywhere from 57% to 91%. Quote this as "roughly 80%
on a small sample", not as an established figure. The same rates on 100 studies
would narrow to 70–86% and 85–96%: the cohort, not the algorithm, is now the
limiting factor.

These numbers predate the bone-bridging and negation fixes; a re-run is in
progress.

---

## 6. Failure modes found and fixed

Documented because each one was silent — the pipeline reported plausible numbers
while being wrong.

| # | failure | how it was caught | fix |
|---|---|---|---|
| 1 | patient left/right swapped in the affine | liver on image-left while the blue `kidney_left` contour was on the same side | correct row/col direction cosines; verify visually, not from the header |
| 2 | 81 false stones per scan | median candidate peak was 139 HU | hysteresis 200/130 |
| 3 | rib and vertebra cortex reported as 8–45 mm stones | overlay review | dense component ≥ 3000 mm³ = bone |
| 4 | excretory-phase contrast reported as 9 stones | kidney HU 96 in study 8285221 | phase gate on parenchymal HU |
| 5 | that gate then rejected a real staghorn | 8261985 had 4435 mm³ of genuine stone | gate on parenchymal HU only; drop the dense-volume criterion |
| 6 | detections outside the kidney | overlay review of 8591756 | closing instead of dilation, + 3 mm cuff |
| 7 | denoiser amplified noise | 375 → 180,419 noise voxels | sign error in the divergence; replaced with SimpleITK |
| 8 | real 2 mm calculi lost after denoising | 8584188: 3 stones → 0 | grow extent on RAW, seed-check on denoised |
| 9 | `largest_mm` / `total_volume_mm3` included **rejected** candidates | 0 stones reported alongside a 16 mm "largest" | aggregate accepted stones only |
| 10 | paediatric kidneys segmented as 30 mL fragments | kidney QC volume flags | age > 18 gate before segmentation |
| 11 | stones fused to bone discarded whole | a candidate with `bone_frac 0.84` — 16% *not* bone | split the component, judge the pieces |
| 12 | resolved stones scored as false negatives | 8610030 "is not visualized" | negation handling in both truth sources |
| 13 | size scored against a gallbladder stone | mean abs error 6.3 mm vs median −1.5 mm | renal-only size extraction |

Process mistakes worth noting: `pkill -f detect_stones.py` killed the shell
running it, because the pattern matched its own command line; and `pgrep -f
render_kidney_masks` matched the *waiter* script whose command line contained
that path. Use `pgrep -f "[d]etect_stones"` and check `/proc/<pid>/cwd` before
concluding anything about what is running.

---

## 7. What is NOT validated

| claim | status |
|---|---|
| stone present / absent | ✅ 79% / 92%, small sample |
| which side | ✅ 15/15 |
| **stone count** | ❌ **never checked** |
| **stone volume** | ⚠️ phantom-validated arithmetic only |
| calyx location | ❌ never checked |
| HU value | ✅ direct measurement |

Count and volume cannot be validated against reports, because reports say
*"a few tiny calculi"* and never *"4 calculi totalling 112 mm³"*. The phantom
test proves the arithmetic is right; it says nothing about whether the right
objects were found.

Also: the 79% applies to **plain, thin-slice, adult** CT. 13 of 44 studies were
excluded, correctly, and the figure does not transfer to arbitrary scans.

---

## 8. Known limitations and next steps

**The 200 HU seed costs sensitivity.** In study 8379961 — which the spreadsheet
says contains renal calculi — candidates peaking at 176 and 183 HU were rejected
as `no_dense_core`. Uric acid stones sit at roughly 200–400 HU. The seed exists
only because there is no false-positive classifier to absorb a lower threshold.
`seed_sweep.py` quantifies the trade.

**The missing piece is Elton's stage 4: a CNN false-positive filter.** Its role
is *not* mainly to remove false positives — it is to make an aggressive
threshold usable, buying back the faint stones a conservative seed discards.
Recipe from the paper: 13-layer 3D CNN, patch from the *original* CT, clip
−200…1000 HU, zero-mean/unit-SD, dropout 0.65, rotation/flip/jitter, rectified
Adam, batch 8, positives reweighted 50:50.

One correction to carry over: **define the patch in millimetres (24 mm) then
resample to 24³**, not 24 voxels. At our 0.5–1.5 mm spacing, 24 voxels spans
12–36 mm, so the network would see the same stone at wildly different physical
scales.

**With only 30–50 studies, try gradient boosting on hand-crafted features
first** (HU statistics, volume, diameter, elongation, sphericity, distance to
sinus, distance to bone, local background). It will likely beat a 3D CNN on that
much data, is faster to build, and reveals *which* features matter. Renal-sinus
arterial plaque was ~50% of Elton's false positives, and it is separable by
shape: plaque is tubular along the vessel, a stone is compact. That cue exists
only in 3D — in a single 2D slice a tube cut crosswise looks like a dot.

**The binding constraint is no longer code. It is annotation.** 30–50 studies
with each stone marked unblocks count validation, volume validation, *and* the
Part 2 classifier simultaneously. Nothing else on the list is worth much first.

**On public datasets:** a `.nii.gz` file with annotations is the right shape but
not sufficient. Check (a) values are real HU — `int16`, range ≈ −1024…3000, with
fat near −100 and muscle near +50; an 8-bit windowed export destroys HU
irreversibly and in a soft-tissue window 12 of our 15 stones saturate to pure
white; (b) slice thickness ≤ 1.5 mm; (c) *all* stones annotated, not just the
reported one — otherwise unlabelled stones become training negatives; (d)
non-contrast; (e) mask affine matches the image affine; (f) orientation verified
visually.

---

## 9. Files

```
kindey_calculus_measurement/
├── run_part1.sh              8-stage pipeline, resumable
├── calculus.xlsx             cohort + report text + calculus_type
├── METHODS.md                this document
├── README.md                 API retention cliff, phase precedence, what failed
├── utils/                    15 scripts
│   ├── paths.py              single place deciding where results go
│   ├── build_worklist.py     cohort sampling by tier/variant
│   ├── probe_api.py          cheap retention-cutoff measurement
│   ├── download_dicoms.py    resumable; --iuid for a single study
│   ├── triage_series.py      stage 1
│   ├── patient_gate.py       stage 2
│   ├── extract_series.py     stage 3
│   ├── run_anatomy.py        stage 4
│   ├── kidney_qc.py          stage 5
│   ├── detect_stones.py      stage 6
│   ├── render_overlays.py    stage 7
│   ├── summarize.py          joins report text
│   ├── compare_reports.py    stage 8
│   ├── seed_sweep.py         offline SEED_HU replay
│   └── test_measurement.py   phantom validation
├── experiment_1/
│   ├── render_kidney_masks.py   mask overlays (5 views/study)
│   └── show_denoising.py        before/after denoising figure
├── dicoms/zips/              44 studies, 14.6 GB
├── nifti/                    37 volumes, 5.0 GB   (shared)
├── seg/                      37 segmentations     (shared)
└── run_full44/               results for this run
    ├── csv/                  17 tables
    ├── kidney_masks/         37 studies × 5 images
    └── overlays/             37 studies
```

~3,800 lines of Python.

**Key output tables:**

| file | contents |
|---|---|
| `baseline_stones.csv` | one row per accepted stone: size, volume, HU, side, pole |
| `baseline_summary.csv` | one row per study: count, largest, total volume, QC |
| `candidates.csv` | **every** candidate incl. rejects + reason — the Part 2 labelling pool |
| `baseline_vs_report.csv` | model vs report, per study |
| `kidney_qc.csv` | per-kidney volume, length, HU, flags |
| `patient_gate.csv` | who was excluded and why |
| `seed_sweep.csv` | sensitivity/specificity vs SEED_HU |
| `triage_series.csv` | every series in every zip, with verdict |
