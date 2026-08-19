"""Does the object contain as much substance as a lump its size should?

IDEA 2 of the ureteric improvement plan, tested rather than assumed.

A stone is a solid lump. Many of our false positives are thin smears along a
bone edge or a vessel wall: "18 mm wide" but containing 42 mm3, which is 1 % of
what an 18 mm lump holds. Our current tests only ask how WIDE an object is, so a
smear passes as a large stone.

    solidity = measured volume / volume of a ball of the measured diameter

Real stones score 0.3-1.0. The smears score 0.01-0.02 -- a 30-fold separation,
and it is INDEPENDENT of density, which matters because the false positives all
sit just above our density floor where no HU threshold can reach them.

WHY THIS IS SCORED PER STUDY AND NOT PER CANDIDATE
--------------------------------------------------
Counting candidates removed is not evidence: a filter that deletes 40 % of
everything looks impressive while quietly deleting the one stone the report
describes. So each cut-off is scored the way a radiologist would:

    recall     of the studies whose report states a ureteric stone, in how many
               do we still report one on the correct side
    FP studies of the studies whose report states none, in how many do we still
               report one

The same trap caught my corridor-tightening idea, which looked good on candidate
counts and turned out to cost 37 points of sensitivity.

Nothing is re-detected and nothing is changed: this reads ureter_candidates.csv
and prints a table. detect_ureteric.py is untouched.

    csv/ureter_solidity.csv    every cut-off scored

Usage:
    CALCULUS_RUN=stone_analysis ./venv/bin/python utils/ureter_solidity.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# package root -> project root: calculus/<sub>/x.py is two levels down
ROOT = os.path.dirname(os.path.dirname(HERE))
from calculus.common.paths import CSV                              # noqa: E402
from calculus.evaluate.compare_reports import SIDE_RE                # noqa: E402

CUTS = [0.0, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
TOPK = 2                      # as the detector currently reports


def main():
    c = pd.read_csv(os.path.join(CSV, "ureter_candidates.csv"))
    c["study_id"] = c.study_id.astype(str)
    searched = set(c.study_id)

    g = pd.read_csv(os.path.join(ROOT, "csv", "ureteric_stone_studies.csv"))
    g["study_id"] = g.study_id.astype(str)
    gt = g[g.study_id.isin(searched)].set_index("study_id").sentence.to_dict()

    def side_of(t):
        h = [k for k, rx in SIDE_RE.items() if rx.search(str(t))]
        return h[0] if len(h) == 1 else ""

    pos = {k: side_of(v) for k, v in gt.items()}
    pos = {k: v for k, v in pos.items() if v}          # side must be known
    neg = sorted(searched - set(gt))
    print(f"{len(pos)} studies with a stated stone and side; "
          f"{len(neg)} studies with no ureteric stone in the report\n")

    a = c[c.is_stone.astype(bool)].copy()
    ball = (4.0 / 3.0) * np.pi * (a.max_diameter_mm / 2.0) ** 3
    a["solidity"] = a.volume_mm3 / ball.replace(0, np.nan)

    rows = []
    for cut in CUTS:
        k = a[a.solidity >= cut].copy()
        k["rank"] = (k.groupby(["study_id", "side"]).hu_max
                      .rank(ascending=False, method="first"))
        rep = k[k["rank"] <= TOPK]
        hit = sum(1 for sid, sd in pos.items()
                  if sd in set(rep[rep.study_id == sid].side))
        fp = sum(1 for sid in neg if len(rep[rep.study_id == sid]))
        rows.append({"solidity_cut": cut,
                     "recall_pct": round(100 * hit / max(len(pos), 1), 1),
                     "hits": f"{hit}/{len(pos)}",
                     "fp_studies": fp,
                     "fp_rate_pct": round(100 * fp / max(len(neg), 1), 1),
                     "precision_pct": round(100 * hit / max(hit + fp, 1), 1),
                     "stones_reported": len(rep)})
    d = pd.DataFrame(rows)
    d["gain"] = (d.recall_pct - d.fp_rate_pct).round(1)
    d.to_csv(os.path.join(CSV, "ureter_solidity.csv"), index=False)

    print(f"{'cut':>5} {'recall':>7} {'hits':>7} {'FP studies':>11} "
          f"{'FP rate':>8} {'precision':>10} {'stones':>7} {'gain':>6}")
    for r in d.itertuples():
        mark = "  <- now" if r.solidity_cut == 0.0 else ""
        print(f"{r.solidity_cut:5.2f} {r.recall_pct:6.1f}% {r.hits:>7} "
              f"{r.fp_studies:11} {r.fp_rate_pct:7.1f}% {r.precision_pct:9.1f}% "
              f"{r.stones_reported:7}{mark}")
    best = d.sort_values("gain", ascending=False).iloc[0]
    print(f"\nbest by recall-minus-FP-rate: cut {best.solidity_cut:.2f}  "
          f"(recall {best.recall_pct}%, precision {best.precision_pct}%)")
    print("Reported, not applied: changing a threshold is a decision, not a "
          "script's call.")


if __name__ == "__main__":
    main()
