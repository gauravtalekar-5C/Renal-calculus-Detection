"""What a tighter corridor and a smaller top-K would have cost, replayed.

STEP 1 of the ureteric improvement plan. Two knobs are currently set by my
judgement rather than by evidence:

    CORRIDOR_MM = 20     how far off the interpolated ureteric course a dense
                         object may sit and still be called ureteric
    TOP_K = 2            how many stones per side reach the report

The 37-study validation showed the count is roughly 3x too high, and that
accepted detections cluster 11-18 mm off a 20 mm centreline -- so the corridor is
barely discriminating. This sweeps both knobs.

NO RE-DETECTION IS NEEDED, which is the point. `off_path_mm` is already recorded
for every candidate, so "what if the corridor had been 12 mm" is answered by
filtering the CSV: a candidate 15 mm off the path would simply never have been
generated. Every other test in the chain (bone, vessel, HU floor, phlebolith) is
independent of the radius and its outcome is already recorded per row. The
replay is therefore exact, not an approximation -- and it costs seconds instead
of the ~17 min/study a real re-run costs.

The one thing it CANNOT do is widen the corridor past 20 mm: those candidates
were never generated, so there is nothing to replay.

HOW IT IS SCORED, with no annotation
------------------------------------
    positives   studies whose report states a ureteric calculus, and the side
                it states. A hit = at least one reported detection on that side.
    negatives   studies searched whose report states no ureteric calculus.
                Any detection at all is a false positive.

Both label sets come from report text. Reports omit findings, so a "false
positive" here may be a real stone nobody wrote down -- the number is a ceiling
on our error, not a proof of it.

    csv/ureter_sweep.csv    every (radius, K) combination scored

Usage:
    CALCULUS_RUN=stone_analysis ./venv/bin/python utils/ureter_sweep.py
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# package root -> project root: calculus/<sub>/x.py is two levels down
ROOT = os.path.dirname(os.path.dirname(HERE))
from calculus.common.paths import CSV                                  # noqa: E402
from calculus.evaluate.compare_reports import SIDE_RE                    # noqa: E402

RADII = [8, 10, 12, 14, 16, 18, 20]
TOPK = [1, 2, 3]


def report_side(text):
    hits = [k for k, rx in SIDE_RE.items() if rx.search(str(text or ""))]
    return hits[0] if len(hits) == 1 else ("both" if len(hits) == 2 else "")


def main():
    c = pd.read_csv(os.path.join(CSV, "ureter_candidates.csv"))
    c["study_id"] = c.study_id.astype(str)
    searched = set(c.study_id)

    g = pd.read_csv(os.path.join(ROOT, "csv", "ureteric_stone_studies.csv"))
    g["study_id"] = g.study_id.astype(str)
    gt = g[g.study_id.isin(searched)].set_index("study_id").sentence.to_dict()

    pos_side = {k: report_side(v) for k, v in gt.items()}
    pos_known = {k: v for k, v in pos_side.items() if v in ("left", "right")}
    negatives = sorted(searched - set(gt))
    print(f"searched {len(searched)} studies\n"
          f"  positives with a stated side : {len(pos_known)}\n"
          f"  positives, side not parsable : {len(gt) - len(pos_known)}\n"
          f"  negatives (no ureteric mention in the report) : {len(negatives)}\n")

    acc = c[c.is_stone.astype(bool)].copy()
    rows = []
    for R in RADII:
        inside = acc[acc.off_path_mm <= R]
        for K in TOPK:
            # re-rank by density within each study+side, exactly as the detector
            # does, then keep the top K
            r = inside.copy()
            r["rank"] = (r.groupby(["study_id", "side"]).hu_max
                          .rank(ascending=False, method="first"))
            rep = r[r["rank"] <= K]

            hit = sum(1 for sid, side in pos_known.items()
                      if side in set(rep[rep.study_id == sid].side))
            fp_studies = sum(1 for sid in negatives
                             if len(rep[rep.study_id == sid]))
            per_study = len(rep) / max(len(searched), 1)
            bil = rep.groupby("study_id").side.nunique()
            rows.append({
                "corridor_mm": R, "top_k": K,
                "sensitivity_pct": round(100 * hit / max(len(pos_known), 1), 1),
                "side_hits": f"{hit}/{len(pos_known)}",
                "fp_studies": fp_studies,
                "fp_rate_pct": round(100 * fp_studies / max(len(negatives), 1), 1),
                "stones_per_study": round(per_study, 2),
                "bilateral_studies": int((bil == 2).sum()),
                "total_reported": len(rep),
            })
    d = pd.DataFrame(rows)
    dest = os.path.join(CSV, "ureter_sweep.csv")
    d.to_csv(dest, index=False)

    print(f"{'corridor':>9} {'K':>2} {'sens':>6} {'hits':>7} {'FP studies':>11} "
          f"{'FP rate':>8} {'stones/study':>13} {'bilateral':>10}")
    for r in d.itertuples():
        print(f"{r.corridor_mm:9} {r.top_k:2} {r.sensitivity_pct:5.1f}% "
              f"{r.side_hits:>7} {r.fp_studies:11} {r.fp_rate_pct:7.1f}% "
              f"{r.stones_per_study:13.2f} {r.bilateral_studies:10}")

    # Youden-style pick: sensitivity minus false-positive rate. Reported rather
    # than applied -- changing a constant is a decision, not a script's call.
    d["youden"] = d.sensitivity_pct - d.fp_rate_pct
    best = d.sort_values(["youden", "stones_per_study"],
                         ascending=[False, True]).iloc[0]
    cur = d[(d.corridor_mm == 20) & (d.top_k == 2)].iloc[0]
    print(f"\ncurrent setting  corridor 20 mm, K=2 : "
          f"sens {cur.sensitivity_pct}%, FP rate {cur.fp_rate_pct}%, "
          f"{cur.stones_per_study} stones/study")
    print(f"best by sens-minus-FP : corridor {best.corridor_mm} mm, "
          f"K={best.top_k} : sens {best.sensitivity_pct}%, "
          f"FP rate {best.fp_rate_pct}%, {best.stones_per_study} stones/study")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
