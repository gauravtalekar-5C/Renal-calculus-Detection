# Open questions

Kept in the repository because a known unknown is cheaper than a rediscovery.

## 1. Kidney specificity: 51% or 83%?

`evaluate/kidney_metrics.py` scores against the compartment field of
`report_vs_model.csv` and gets 51%. `evaluate/score_run.py` scores against
`report_labels.csv` (parsed per-kidney paragraph) and gets 83%. Different
denominators too: 49 negatives versus 53. Until one label source is shown correct,
neither figure is quotable.

## 2. Ureteric precision is 53.8%

Radiologists overcall distal ureteric stones 19.7% of the time. We are at 46%.
Density and geometry are exhausted: tightening the corridor costs more recall than
it saves precision (measured), and a solidity filter buys 47%->56% at the cost of
two real stones. The two untried levers are a classifier over the 19 candidate
features already computed (5,592 labelled negatives exist, needs ~150 positives),
and hydronephrosis as corroborating evidence.

## 3. The UVJ landmark has never been validated

Every distance we print rests on a geometric rule nobody has compared to a
radiologist's click. The error is unbounded, not merely unknown. ~40 studies at
two clicks each would settle it.

## 4. Contrast studies report "0 calculi" instead of abstaining

Detection is impossible on a post-contrast study and the QC gate catches them,
but the report still prints a zero, which reads as a clean scan. One-line fix in
`report/make_report.py`, not yet applied.

## 5. Not implemented, though in scope

Hydronephrosis, perinephric fat stranding, stent, bladder calculi, comet-tail
phlebolith sign, appendicolith and calcified-node suppression, metal streak
handling at the VUJ.
