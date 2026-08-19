"""Sensitivity and specificity for a run, against the full-report labels.

Reads:
    csv/report_labels.csv     ground truth from parse_reports.py (full report
                              body, HISTORY dropped, per-kidney paragraphs)
    csv/baseline_summary.csv  or csv/per_study/*_summary.csv if the run is
                              still going
    csv/kidney_qc.csv         to exclude studies whose kidney mask is unusable

EXCLUSIONS, and why each one is defensible
------------------------------------------
  not analysable   the phase gate refused the study (enhanced / excretory CT).
                   Contrast in the collecting system is indistinguishable from
                   stone, so neither a hit nor a miss means anything.
  bad kidney mask  volume outside 120-500 mL against a ~220 mL median. If the
                   mask is wrong the search region is wrong, so the detector
                   was never given a fair chance.
  age-gate leaks   paediatric studies the gate excluded but whose NIfTI/seg
                   survived from an earlier run. The adult model does not work
                   on them.

All three are reported explicitly rather than quietly dropped -- a shrinking
denominator is the easiest way to make a metric look better than it is.

Usage:
    CALCULUS_RUN=run_v5 ./venv/bin/python utils/score_run.py
    CALCULUS_RUN=run_v5 ./venv/bin/python utils/score_run.py --compare run_v4
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# package root -> project root: calculus/<sub>/x.py is two levels down
ROOT = os.path.dirname(os.path.dirname(HERE))
from calculus.common.paths import CSV                              # noqa: E402

# paediatric studies the age gate excluded, whose nifti/seg predate the gate
LEAKS = {8416497, 8511166, 8591756}
QC_LO, QC_HI = 120, 500          # mL; cohort median is ~220


def wilson(k, n, z=1.96):
    """95% interval for a proportion.

    Wilson, not the textbook normal approximation: at n=10 the simple formula
    puts the upper limit above 100%, which is nonsense and was how the old
    "specificity 80%" looked more solid than it was.
    """
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def load_summary(csv_dir):
    p = os.path.join(csv_dir, "baseline_summary.csv")
    if os.path.exists(p):
        return pd.read_csv(p), "combined"
    fs = sorted([f for f in glob.glob(os.path.join(csv_dir, "per_study",
                                          "*_summary.csv"))
         if "_ureter_" not in os.path.basename(f)])
    if not fs:
        sys.exit(f"no results in {csv_dir}")
    return (pd.concat([pd.read_csv(f) for f in fs], ignore_index=True),
            f"per_study ({len(fs)} studies)")


def score(csv_dir, label):
    summ, src = load_summary(csv_dir)
    lp = os.path.join(csv_dir, "report_labels.csv")
    if not os.path.exists(lp):                 # labels are run-independent
        lp = os.path.join(ROOT, "run_v4", "csv", "report_labels.csv")
    lab = pd.read_csv(lp)

    d = summ.merge(lab[["study_id", "report_renal_calculus"]], on="study_id")
    n_all = len(d)
    # phase gate / segmentation failures carry an `error` string
    d = d[d.get("error").isna() | (d.get("error") == "")] if "error" in d else d
    n_analysable = len(d)

    qp = os.path.join(csv_dir, "kidney_qc.csv")
    if not os.path.exists(qp):
        qp = os.path.join(ROOT, "run_v4", "csv", "kidney_qc.csv")
    bad = set()
    if os.path.exists(qp):
        q = pd.read_csv(qp)
        qcol = "verdict" if "verdict" in q.columns else (
            "qc" if "qc" in q.columns else None)
        # kidney_qc.py names this column "qc", not "verdict". Looking only
        # for "verdict" fell through to the stale QC_LO/QC_HI volume window
        # SILENTLY, so the recalibrated verdicts were never applied.
        if qcol:
            # Use kidney_qc's own verdict rather than a second, stricter volume
            # window kept here. The old QC_LO/QC_HI of 120-500 mL predates the
            # recalibration and excluded studies the current QC calls usable --
            # two different definitions of "bad mask" in one pipeline, with this
            # one silently shrinking the denominator.
            bad = set(q[q[qcol].isin(["fail", "cannot_assess",
                                      "contrast"])].study_id)
        else:
            bad = set(q[(q.total_ml < QC_LO) | (q.total_ml > QC_HI)].study_id)
    d = d[~d.study_id.isin(bad | LEAKS)]

    det = d.n_stones.fillna(0) > 0
    pos = d.report_renal_calculus == True       # noqa: E712
    neg = ~pos
    tp = int((pos & det).sum()); fn = int((pos & ~det).sum())
    tn = int((neg & ~det).sum()); fp = int((neg & det).sum())
    se = 100 * tp / max(tp + fn, 1)
    sp = 100 * tn / max(tn + fp, 1)
    l1, h1 = wilson(tp, tp + fn); l2, h2 = wilson(tn, tn + fp)

    print(f"\n{'='*74}\n{label}   (source: {src})\n{'='*74}")
    print(f"  studies with results        {n_all}")
    print(f"  analysable (phase gate ok)  {n_analysable}")
    print(f"  excluded: bad kidney mask {len(bad & set(summ.study_id))}, "
          f"age-gate leaks {len(LEAKS & set(summ.study_id))}")
    print(f"  SCORED                      {len(d)}   "
          f"({tp+fn} positive / {tn+fp} negative)\n")
    print(f"  sensitivity  {tp:3d}/{tp+fn:<3d}  {se:5.1f}%   95% CI {l1:.0f}-{h1:.0f}")
    print(f"  specificity  {tn:3d}/{tn+fp:<3d}  {sp:5.1f}%   95% CI {l2:.0f}-{h2:.0f}")
    print(f"  Youden       {se + sp - 100:5.1f}")
    print(f"  TP {tp}   FN {fn}   TN {tn}   FP {fp}")
    if len(d):
        st = d.n_stones.fillna(0)
        print(f"\n  stones found: {int(st.sum())} across {int((st>0).sum())} studies")
    return {"label": label, "sens": se, "spec": sp, "n": len(d),
            "tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "fn_ids": sorted(d[pos & ~det].study_id),
            "fp_ids": sorted(d[neg & det].study_id)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", default=None,
                    help="another run directory to score alongside, e.g. run_v4")
    args = ap.parse_args()

    cur = score(CSV, os.path.basename(os.path.dirname(CSV)) or "current")
    if args.compare:
        other = os.path.join(ROOT, args.compare, "csv")
        if os.path.isdir(other):
            prev = score(other, args.compare)
            print(f"\n{'='*74}\nCHANGE\n{'='*74}")
            print(f"  sensitivity {prev['sens']:5.1f}% -> {cur['sens']:5.1f}%  "
                  f"({cur['sens']-prev['sens']:+.1f})")
            print(f"  specificity {prev['spec']:5.1f}% -> {cur['spec']:5.1f}%  "
                  f"({cur['spec']-prev['spec']:+.1f})")

    print(f"\nfalse negatives ({len(cur['fn_ids'])}): "
          f"{', '.join(str(s) for s in cur['fn_ids'])}")
    print(f"false positives ({len(cur['fp_ids'])}): "
          f"{', '.join(str(s) for s in cur['fp_ids'])}")
    print("\nCAVEAT: measured against radiologist REPORT TEXT, not annotation, "
          "and per STUDY\nnot per stone. Reports omit small calyceal stones, "
          "so this flatters sensitivity.\nNot comparable to published numbers "
          "measured on other cohorts.")


if __name__ == "__main__":
    main()
