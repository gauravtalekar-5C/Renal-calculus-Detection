"""Recall, precision and side accuracy for ureteric detection, against reports.

Every study in this cohort was selected BECAUSE its report states a ureteric
calculus, so this set has no negatives -- which means it measures RECALL and
side accuracy honestly, and cannot measure precision or specificity at all.
That is stated in the output rather than papered over: a precision computed on
an all-positive set is meaningless, and quoting one would be dishonest.

Specificity still comes from the main cohort's 47 report-negative studies.

WHAT COUNTS AS A HIT
    the report names a side -> we must report a stone on THAT side
    the report names no side -> any ureteric detection counts

WHAT THE GROUND TRUTH IS NOT
    A report is not an annotation. It states what was worth writing: a size, a
    side, sometimes an HU, rarely a coordinate. So a "hit" means we found a
    stone on the right side of the right patient -- not that we found the same
    object the radiologist measured. Per-object agreement needs one click per
    stone, which is the 20-minute annotation task, not a 50-minute tracing one.

    csv/eval_ureteric.csv        one row per study: report vs us
    csv/eval_ureteric_size.csv   reported size beside our measurement

Usage:
    CALCULUS_RUN=ureteric_whole_stone_data ./venv/bin/python utils/eval_ureteric.py
"""
import os
import re
import sys
from math import sqrt

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# package root -> project root: calculus/<sub>/x.py is two levels down
ROOT = os.path.dirname(os.path.dirname(HERE))
from calculus.common.paths import CSV, RUN                                   # noqa: E402
from calculus.evaluate.compare_measurements import calculus_sizes              # noqa: E402
from calculus.evaluate.compare_reports import SIDE_RE                          # noqa: E402

HU_RE = re.compile(r"(\d{2,4})\s*(?:HU|hu)\b|(?:HU|hu)\s*[:~=]?\s*(\d{2,4})")


def wilson(k, n):
    if n == 0:
        return (0.0, 0.0)
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


def side_of(t):
    h = [k for k, rx in SIDE_RE.items() if rx.search(str(t))]
    return h[0] if len(h) == 1 else ("both" if len(h) == 2 else "")


def main():
    cp = os.path.join(CSV, "ureter_candidates.csv")
    if not os.path.exists(cp):
        sys.exit(f"no {cp} -- run detect_ureteric.py for this cohort first")
    c = pd.read_csv(cp)
    c["study_id"] = c.study_id.astype(str)
    acc = c[c.is_stone.astype(bool)]
    rep = acc[acc.report_this.astype(bool)] if "report_this" in acc else acc

    # the worklist carries the report sentence that put each study in this cohort
    wl = pd.read_csv(os.path.join(RUN, "worklist.csv"))
    wl["study_id"] = wl.study_id.astype(str)
    sent = wl.set_index("study_id").sentence.to_dict()

    searched = sorted(set(c.study_id))
    rows = []
    for sid in searched:
        s = str(sent.get(sid, ""))
        gside = side_of(s)
        ours = rep[rep.study_id == sid]
        our_sides = set(ours.side.dropna())
        hit = (bool(our_sides) if gside in ("", "both")
               else gside in our_sides)
        rsz = calculus_sizes(s)
        rsizes = rsz["ureteric"] + rsz["renal"]
        hus = [float(m.group(1) or m.group(2)) for m in HU_RE.finditer(s)]
        rows.append({
            "study_id": sid,
            "report_side": gside,
            "our_sides": ",".join(sorted(our_sides)),
            "hit": hit,
            "n_reported_by_us": len(ours),
            "report_sizes_mm": ";".join(f"{v:g}" for v in sorted(rsizes, reverse=True)),
            "our_largest_mm": (round(float(ours.max_diameter_mm.max()), 1)
                               if len(ours) else ""),
            "report_hu": ";".join(f"{v:g}" for v in sorted(hus, reverse=True)),
            "our_hu_max": int(ours.hu_max.max()) if len(ours) else "",
            "our_zones": ",".join(sorted(set(ours.zone.dropna()))) if len(ours) else "",
            "min_dist_to_uvj_mm": (round(float(ours.dist_to_uvj_along_mm.min()), 1)
                                   if len(ours) and ours.dist_to_uvj_along_mm.notna().any()
                                   else ""),
            "report_says": s[:220],
        })
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(CSV, "eval_ureteric.csv"), index=False)

    n = len(d)
    hits = int(d.hit.sum())
    lo, hi = wilson(hits, n)
    sided = d[d.report_side.isin(["left", "right"])]
    sh = int(sided.hit.sum())
    slo, shi = wilson(sh, len(sided))
    found_any = int((d.n_reported_by_us > 0).sum())

    print(f"\n{'='*62}\nURETERIC EVALUATION -- {n} studies, every one report-positive")
    print(f"{'='*62}")
    print(f"  detected a stone at all      {found_any}/{n}  "
          f"{100*found_any/max(n,1):.1f}%")
    print(f"  RECALL (correct side)        {sh}/{len(sided)}  "
          f"{100*sh/max(len(sided),1):.1f}%   95% CI {slo:.0f}-{shi:.0f}")
    print(f"  recall (any side counted)    {hits}/{n}  "
          f"{100*hits/max(n,1):.1f}%   95% CI {lo:.0f}-{hi:.0f}")
    print(f"  stones reported per study    {len(rep)/max(n,1):.2f}")
    print(f"  studies we called bilateral  "
          f"{int((d.our_sides == 'left,right').sum())}")
    # ---- precision and specificity, borrowing the negatives ---------------
    # This cohort is all-positive, so precision cannot come from it alone. The
    # main cohort has 47 studies whose reports state no ureteric calculus; those
    # supply the false positives. Both were produced by the same detector on the
    # same kind of scan, so combining them is sound -- but PRECISION DEPENDS ON
    # PREVALENCE, and the mix here (many positives, few negatives) is an artefact
    # of how the cohorts were built, not the rate in a real worklist. So
    # specificity is the transferable number and precision is quoted with the
    # prevalence it was computed at.
    negp = os.path.join(ROOT, "stone_analysis", "csv", "ureter_candidates.csv")
    gtp = os.path.join(ROOT, "csv", "ureteric_stone_studies.csv")
    if os.path.exists(negp) and os.path.exists(gtp):
        nc = pd.read_csv(negp); nc["study_id"] = nc.study_id.astype(str)
        gg = pd.read_csv(gtp); gg["study_id"] = gg.study_id.astype(str)
        main_searched = set(nc.study_id)
        negatives = sorted(main_searched - set(gg.study_id) - set(searched))
        na = nc[nc.is_stone.astype(bool)]
        nrep = na[na.report_this.astype(bool)] if "report_this" in na else na
        fp = sum(1 for s_ in negatives if len(nrep[nrep.study_id == s_]))
        tn = len(negatives) - fp
        spec = 100 * tn / max(len(negatives), 1)
        prec = 100 * sh / max(sh + fp, 1)
        slo2, shi2 = wilson(tn, len(negatives))
        plo, phi = wilson(sh, sh + fp)
        print(f"\n  --- with the main cohort's {len(negatives)} report-negative "
              f"studies as negatives ---")
        print(f"  SPECIFICITY                  {tn}/{len(negatives)}  "
              f"{spec:.1f}%   95% CI {slo2:.0f}-{shi2:.0f}")
        print(f"  false positives              {fp} studies")
        print(f"  PRECISION                    {sh}/{sh+fp}  {prec:.1f}%   "
              f"95% CI {plo:.0f}-{phi:.0f}")
        print(f"  (prevalence here is {100*len(sided)/max(len(sided)+len(negatives),1):.0f}% "
              f"positive -- an artefact of cohort construction, so read")
        print(f"   specificity as the transferable number, not precision)")
    else:
        print("\n  PRECISION NOT COMPUTED: this cohort is all-positive and the "
              "main cohort's\n  candidates were not found for negatives.")

    miss = d[~d.hit]
    if len(miss):
        print(f"\n  missed ({len(miss)}):")
        for r in miss.itertuples():
            print(f"    {r.study_id}  report {r.report_side or '?'}  "
                  f"ours '{r.our_sides or 'none'}'  {r.report_says[:90]}")

    # size agreement, where the report states one
    sz = []
    for r in d.itertuples():
        if not r.report_sizes_mm or r.our_largest_mm == "":
            continue
        rm = max(float(v) for v in r.report_sizes_mm.split(";"))
        sz.append({"study_id": r.study_id, "report_mm": rm,
                   "our_mm": float(r.our_largest_mm),
                   "diff_mm": round(rm - float(r.our_largest_mm), 2)})
    if sz:
        s = pd.DataFrame(sz)
        s.to_csv(os.path.join(CSV, "eval_ureteric_size.csv"), index=False)
        a = s.diff_mm.abs()
        print(f"\n{'='*62}\nSIZE ERROR ANALYSIS  (n={len(s)} studies where the "
              f"report states a size)\n{'='*62}")
        print(f"  bias (report - ours)   {s.diff_mm.mean():+.2f} mm   "
              f"median {s.diff_mm.median():+.2f} mm")
        print(f"  mean absolute error    {a.mean():.2f} mm   "
              f"median {a.median():.2f} mm")
        print(f"  spread (SD of error)   {s.diff_mm.std():.2f} mm")
        for t in (1, 2, 3, 5):
            print(f"  within {t} mm            {100*(a <= t).mean():.0f}%")
        # error by size band: partial volume dominates the small end, so a single
        # average hides the only regime that matters clinically (the 5 mm
        # threshold for spontaneous passage)
        print(f"\n  by reported size:")
        bands = [(0, 3), (3, 5), (5, 8), (8, 15), (15, 999)]
        for lo_, hi_ in bands:
            b = s[(s.report_mm >= lo_) & (s.report_mm < hi_)]
            if not len(b):
                continue
            print(f"    {lo_:2}-{hi_ if hi_ < 999 else '+':>3} mm  n={len(b):3}  "
                  f"bias {b.diff_mm.mean():+6.2f}  "
                  f"mean abs {b.diff_mm.abs().mean():5.2f}  "
                  f"within 2 mm {100*(b.diff_mm.abs() <= 2).mean():3.0f}%")
        print(f"\n  we UNDER-measure in {int((s.diff_mm > 0).sum())} studies, "
              f"OVER-measure in {int((s.diff_mm < 0).sum())}")
        worst = s.reindex(a.sort_values(ascending=False).index).head(6)
        print(f"\n  largest disagreements:")
        for r in worst.itertuples():
            print(f"    {r.study_id}  report {r.report_mm:5.1f} mm   "
                  f"ours {r.our_mm:5.1f} mm   diff {r.diff_mm:+6.1f} mm")
        print(f"\n  CAVEAT: 'ours' is the largest stone WE report and 'report' the "
              f"largest the\n  radiologist wrote -- in a multi-stone study those "
              f"can be different stones.\n  A per-object figure needs one click "
              f"per stone.")
    print(f"\nwrote {os.path.join(CSV, 'eval_ureteric.csv')}")


if __name__ == "__main__":
    main()
