# Candidate judge — what it is, what it is worth, and why it is not wired in

## What was built
A per-candidate classifier (`judge_model.joblib`, HistGradientBoosting) that
scores every bright spot the cascade proposes: stone, or mimic. Trained on the
cohort's own candidates with labels derived from the radiologist reports.

Applied **after** the two fill-fraction rejections, not instead of them. Those
remove 68% of candidates at no sensitivity cost, and leaving them in made the
model spend its capacity re-learning "streaks are not stones".

## Labels, and their weakness
There is no per-candidate ground truth, only a report per study.

- **negative (trustworthy)** — every candidate in a study whose report names no
  calculus: 367 after filtering
- **positive (weaker)** — a candidate in a positive study whose compartment
  matches one the report names AND whose size or density is close to a number
  the report quotes: 371
- **unlabelled (excluded from training, still scored)** — 299

A real stone the radiologist missed is labelled negative here, and there is no
way to find those in this data.

## Measured, out-of-fold, grouped by study
Folds split **studies**, never candidates: candidates from one study share a
patient, a scanner and a kernel, and splitting them lets the model memorise the
study instead of the finding.

    per-candidate      AUC 0.893   AP 0.896
    per compartment    renal 0.795   ureteric 0.718   bladder 0.861

Study level, 173 positive / 153 negative:

    threshold   sensitivity   false positive
      (rules only)   96.5%        62.1%
        0.02         94.2%        52.3%
        0.05         91.9%        49.7%
        0.10         88.4%        45.1%
        0.20         85.0%        36.6%

For comparison, the starting point was **sensitivity 97.6%, FP 87.4%**.

## Why it is NOT wired into the pipeline
At matched sensitivity the judge adds almost nothing over the rules alone —
94.2%/52.3% against 94.1%/52.4%. A learned component trained on 326 studies with
report-derived labels is not worth that, and it would be a new thing to validate,
version and explain for no measured gain.

The arithmetic of why: study-level FP needs EVERY candidate in a clean study to
score low. After the rules a clean study still carries ~2.4 candidates, so at a
threshold catching 90% of stones, per-candidate specificity near 0.65 compounds
to roughly 55% study-level FP. Getting study FP to 20% at 90% sensitivity needs
per-candidate AUC around 0.96. We have 0.893.

## What it did establish, which was the point
1. **The measurements carry real signal** — AUC 0.893 — so the earlier
   univariate failures were a limitation of single thresholds, not of the data.
2. **The ureter is the ceiling.** AUC 0.718, and on its own it flags 71.6% of
   clean studies. Renal is 21.1%, bladder 5.3%. Disabling ureteric alone takes
   FP from 80% to 24% and sensitivity from 95% to 80%.
3. **Aggregating to a study-level model is worse** than taking the maximum
   candidate score (AUC 0.831 vs the max rule), which 326 studies against 19
   aggregate features should have predicted.

## The next thing worth doing
Point an image model at ureteric candidates specifically. The 35 numbers are a
lossy summary; a phlebolith and a ureteric stone differ in things the summary
throws away — the vessel they sit in, the wall around them, the rim, the
neighbourhood. That is a patch CNN, and it needs the NIfTIs, which the cohort
janitor deleted for the first 340 studies. Retention is fixed going forward.

The other half is not a model at all: `off_path_mm` is ~16 mm for real stones and
for mimics alike, meaning the ureter corridor does not localise a 3–5 mm tube.
Every position feature is measured against that line. Fixing the corridor may be
worth more than any classifier trained on features derived from it.
