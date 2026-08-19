# calculus

Renal and ureteric calculus detection and measurement on **non-contrast CT**.

On a plain CT a stone is the brightest thing in a region that should be dark, so
detection is a thresholding problem rather than a recognition problem. Almost all
of the difficulty is establishing *where you are*: the same 400 HU dot is a stone
in the renal pelvis, a phlebolith in the pelvic sidewall, and cortical bone on the
sacrum. Nine of the eleven pipeline stages exist to establish location; two are
about density.

No stone model is trained. The only learned component is
[TotalSegmentator](https://github.com/wasserth/TotalSegmentator), used to outline
organs — every published renal-calculus model we could source is licensed
non-commercially.

## What it produces

Per intrarenal stone: count, maximum diameter, three axes, volume, peak and mean
HU, calyceal third, anterior/posterior. Per ureteric stone: side, zone
(near PUJ / mid ureter / near VUJ), size, density, and arc-length distance from
the vesico-ureteric junction.

## Where it stands

| | Recall | Precision | Specificity |
|---|---|---|---|
| Kidney, per study (n=121) | **95.8%** | 74.2% | 51.0%* |
| Kidney, per kidney (n=242) | **95.6%** | 72.7% | 68.0% |
| Ureter, per study (n=84) | **94.6%** | 53.8% | 36.2% |

Size agreement against reported measurements: median absolute error **1.4 mm**
overall, ~1 mm for stones under 15 mm, and 5.7 mm above that — branched
(staghorn) stones fragment into components and are under-measured. Against
phantoms, where ground truth is exact: **0.11 mm** at 0.7 mm voxels, 0.55 mm at
3 mm slices.

*Two scoring paths currently disagree on kidney specificity (51% vs 83%) because
they use different report-label sources. Unresolved — see `docs/OPEN.md`. Neither
number should be quoted until it is.

**Recall is the strong number everywhere; precision is the open problem**, and it
is worst in the ureter. Radiologists themselves overcall distal ureteric stones
19.7% of the time; we are at 46%.

## Install

```bash
python -m venv venv && . venv/bin/activate
pip install -e .            # analysis code only
pip install -e ".[seg]"     # adds TotalSegmentator (pulls a CUDA torch build)
```

## Use

One study, end to end:

```bash
calculus-study 8231547                 # a study id, a zip, or a DICOM folder
```

A cohort:

```bash
export CALCULUS_RUN=my_run              # where results go; never overwrites another run
python -m calculus.common.triage_series     # pick the measurable series
python -m calculus.common.extract_series    # -> nifti/
python -m calculus.common.run_anatomy       # -> seg/   organ masks
python -m calculus.kidney.detect_stones --workers 3
python -m calculus.ureter.detect_ureteric --workers 3
python -m calculus.report.make_report_full
python -m calculus.evaluate.kidney_metrics
```

`CALCULUS_RUN`, `CALCULUS_NIFTI`, `CALCULUS_SEG` and `CALCULUS_ZIPS` point a run
at its own directories, so a second cohort never mixes with the first.

## Layout

```
calculus/
  common/     paths, DICOM ingest, series triage, patient gate, organ masks
  kidney/     Part 1: detection, QC, overlays, 3D surfaces
  ureter/     Part 2: anatomical corridor, detection, review sheets, experiments
  report/     clinical report tables, project PDF
  evaluate/   scoring against report text
  pipeline/   single-study entry points
tests/        geometry and report-parsing tests; no patient data required
docs/         methodology and the open questions
```

## Patient data

Nothing in `.gitignore`'s data section may ever be committed. Downloads are named
by `study_id` and the server's `Content-Disposition` header is ignored, because it
carries the patient name. Derived NIfTI files were checked field by field and
carry no identifiers; DICOM files do and must not leave the machine.

## Two things worth knowing before trusting a number

**130 HU is a hard floor.** Nothing below it can become a candidate. Reported
stones in our cohort go down to 90 HU, so roughly 11% are unreachable by
construction, not by failure.

**A report is not an annotation.** It records what was clinically worth writing.
Small calyceal stones are routinely omitted, which flatters recall and penalises
specificity. Every metric here inherits that.
