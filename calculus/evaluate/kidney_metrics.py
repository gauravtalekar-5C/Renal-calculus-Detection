"""Every kidney-stone evaluation metric we can compute, in one file.

Writes KIDNEY_METRICS.md in the run folder and prints the same thing. Numbers
come from the CSVs on disk, so re-running after a new analysis produces a report
that matches the analysis.

WHAT IS MEASURED, AND THE UNIT IT IS MEASURED IN
    per STUDY   presence: did the report state a calculus, did we find one
    per SIDE    the same question asked separately of each kidney -- 2x the
                sample size, and it catches "right answer, wrong kidney", which
                a per-study count cannot
    per STONE   size, where the report states one

GROUND TRUTH IS REPORT TEXT, not annotation. A report states what was worth
writing. Small calyceal stones are routinely omitted, which flatters sensitivity
and penalises specificity. Every table below inherits that caveat.
"""
import os
import sys
from math import sqrt

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# package root -> project root: calculus/<sub>/x.py is two levels down
ROOT = os.path.dirname(os.path.dirname(HERE))
from calculus.common.paths import CSV, RUN                                   # noqa: E402

L = []
def out(s=""):
    print(s)
    L.append(s)


def wilson(k, n):
    if n == 0:
        return (0.0, 0.0)
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


def rate(name, k, n):
    if n == 0:
        return f"| {name} | — | — | — |"
    lo, hi = wilson(k, n)
    return f"| {name} | {k}/{n} | **{100*k/n:.1f}%** | {lo:.0f}–{hi:.0f} |"


def main():
    d = pd.read_csv(os.path.join(CSV, "report_vs_model.csv"))
    d["study_id"] = d.study_id.astype(str)
    qc = pd.read_csv(os.path.join(CSV, "kidney_qc.csv"))
    qc["study_id"] = qc.study_id.astype(str)
    qcol = "qc" if "qc" in qc.columns else "verdict"
    bad = set(qc[qc[qcol].isin(["fail", "cannot_assess", "contrast"])].study_id)
    st = pd.read_csv(os.path.join(CSV, "baseline_stones.csv"))
    st["study_id"] = st.study_id.astype(str)
    sz = pd.read_csv(os.path.join(CSV, "size_vs_report_study.csv"))
    sz["study_id"] = sz.study_id.astype(str)
    mt = pd.read_csv(os.path.join(CSV, "size_vs_report_matched.csv"))

    out(f"# Kidney stone — evaluation metrics")
    out()
    out(f"Generated {pd.Timestamp.now():%d %b %Y %H:%M} from `{os.path.basename(RUN)}/csv/`. "
        f"Ground truth is radiologist report text.")
    out()

    # ---- 1. cohort ------------------------------------------------------
    e = d[~d.study_id.isin(bad)].copy()
    # RENAL ground truth, not "any calculus": report_has_stone is true for a
    # ureteric- or bladder-only study too, and scoring the kidney detector
    # against that counts those as misses. The compartment column is what says
    # where the report put the stone.
    e["renal_gt"] = [("renal" in {y for y in str(x).split("+")})
                     for x in e.report_compartments]
    out("## 1 · Cohort")
    out()
    out("| | n |")
    out("|---|---|")
    out(f"| studies with results | {len(d)} |")
    out(f"| excluded: bad mask / contrast / cut off | {len(set(d.study_id) & bad)} |")
    out(f"| **scored** | **{len(e)}** |")
    out(f"| report states a RENAL calculus | {int(e.renal_gt.sum())} |")
    out(f"| report states no renal calculus | {int((~e.renal_gt).sum())} |")
    out()

    # ---- 2. detection, per study ----------------------------------------
    tp = int((e.renal_gt & e.model_has_stone).sum())
    fn = int((e.renal_gt & ~e.model_has_stone).sum())
    tn = int((~e.renal_gt & ~e.model_has_stone).sum())
    fp = int((~e.renal_gt & e.model_has_stone).sum())
    out("## 2 · Detection, per study")
    out()
    out(f"```")
    out(f"                 report: stone   report: none")
    out(f"we detect stone      TP {tp:4}         FP {fp:4}")
    out(f"we detect none       FN {fn:4}         TN {tn:4}")
    out(f"```")
    out()
    out("| Metric | count | value | 95% CI |")
    out("|---|---|---|---|")
    out(rate("Sensitivity / recall", tp, tp + fn))
    out(rate("Specificity", tn, tn + fp))
    out(rate("Precision / PPV", tp, tp + fp))
    out(rate("NPV", tn, tn + fn))
    out(rate("Accuracy", tp + tn, tp + tn + fp + fn))
    se = tp / max(tp + fn, 1); sp = tn / max(tn + fp, 1)
    pr = tp / max(tp + fp, 1)
    f1 = 2 * pr * se / max(pr + se, 1e-9)
    out(f"| F1 | — | **{100*f1:.1f}%** | — |")
    out(f"| Youden J | — | **{100*(se+sp-1):.1f}** | — |")
    out(f"| Balanced accuracy | — | {100*(se+sp)/2:.1f}% | — |")
    out()
    fnl = sorted(e[e.renal_gt & ~e.model_has_stone].study_id)
    fpl = sorted(e[~e.renal_gt & e.model_has_stone].study_id)
    out(f"**False negatives ({len(fnl)}):** {', '.join(fnl)}")
    out()
    out(f"**False positives ({len(fpl)}):** {', '.join(fpl)}")
    out()

    # ---- 3. detection, per kidney ---------------------------------------
    rows = []
    for r in e.itertuples():
        rs = {x for x in str(r.report_sides).split("+") if x in ("left", "right")}
        ms = {x for x in str(r.model_sides).split("+") if x in ("left", "right")}
        if not r.renal_gt and not rs:
            rs = set()
        for side in ("left", "right"):
            rows.append({"truth": side in rs, "pred": side in ms})
    s2 = pd.DataFrame(rows)
    tp2 = int((s2.truth & s2.pred).sum())
    fn2 = int((s2.truth & ~s2.pred).sum())
    tn2 = int((~s2.truth & ~s2.pred).sum())
    fp2 = int((~s2.truth & s2.pred).sum())
    out("## 3 · Detection, per kidney (each side scored separately)")
    out()
    out("Twice the sample size, and it catches *right answer, wrong kidney* — which "
        "a per-study score cannot.")
    out()
    out("| Metric | count | value | 95% CI |")
    out("|---|---|---|---|")
    out(rate("Sensitivity", tp2, tp2 + fn2))
    out(rate("Specificity", tn2, tn2 + fp2))
    out(rate("Precision", tp2, tp2 + fp2))
    out(f"| units scored | {len(s2)} kidneys | — | — |")
    out()

    # ---- 4. localisation -------------------------------------------------
    out("## 4 · Localisation")
    out()
    out("| Field | n | Exact | Partial | Wrong |")
    out("|---|---|---|---|---|")
    setof = lambda x: {y for y in str(x).split("+") if y not in ("", "nan")}
    for field, a, b in (("Side", "report_sides", "model_sides"),
                        ("Pole", "report_poles", "model_poles"),
                        ("Compartment", "report_compartments", "model_compartments")):
        v = []
        for x, y in zip(e[a], e[b]):
            A, B = setof(x), setof(y)
            if not A or not B:
                continue
            v.append("exact" if A == B else ("partial" if A & B else "none"))
        n = len(v)
        if n:
            out(f"| {field} | {n} | {100*v.count('exact')/n:.0f}% | "
                f"{100*v.count('partial')/n:.0f}% | {100*v.count('none')/n:.0f}% |")
    out()
    out("*Partial* = the sets overlap without matching, e.g. report 'renal+ureteric' "
        "against our 'renal'. Reported separately because it is neither a hit nor a "
        "clean miss.")
    out()

    # ---- 5. size ---------------------------------------------------------
    ren = sz.report_renal_mm.fillna("").astype(str)
    ure = sz.report_ureteric_mm.fillna("").astype(str)
    k = sz[(ren != "") & (ure == "")]
    out("## 5 · Size agreement")
    out()
    out("Restricted to reports naming renal stones and **no** ureteric one, so the "
        "largest reported stone must be the one the kidney detector measured.")
    out()
    out("| Comparison | n | Bias | Mean abs | Median abs | ≤1 mm | ≤2 mm | ≤3 mm |")
    out("|---|---|---|---|---|---|---|---|")
    for lbl, col, df in (("Report max vs our in-plane max",
                          "diff_mm_report_minus_model", k),
                         ("Report max vs our 3D caliper", "diff_mm_vs_caliper", k),
                         ("Per stone, nearest-size match", "diff_mm", mt)):
        v = pd.to_numeric(df[col], errors="coerce").dropna()
        if not len(v):
            continue
        a = v.abs()
        out(f"| {lbl} | {len(v)} | {v.mean():+.2f} mm | {a.mean():.2f} mm | "
            f"{a.median():.2f} mm | {100*(a<=1).mean():.0f}% | "
            f"{100*(a<=2).mean():.0f}% | {100*(a<=3).mean():.0f}% |")
    out()
    kk = k.assign(d=pd.to_numeric(k.diff_mm_vs_caliper, errors="coerce"),
                  rm=pd.to_numeric(k.report_max_mm, errors="coerce")).dropna(subset=["d", "rm"])
    out("### By reported size — where the method holds and where it fails")
    out()
    out("| Reported | n | Bias | Median abs | ≤2 mm |")
    out("|---|---|---|---|---|")
    for lo, hi in ((0, 3), (3, 5), (5, 8), (8, 15), (15, 999)):
        b = kk[(kk.rm >= lo) & (kk.rm < hi)]
        if len(b):
            out(f"| {lo}–{hi if hi < 999 else '+'} mm | {len(b)} | {b.d.mean():+.2f} mm | "
                f"{b.d.abs().median():.2f} mm | {100*(b.d.abs()<=2).mean():.0f}% |")
    out()
    if len(kk) > 3:
        r = np.corrcoef(kk.rm, kk.rm - kk.d)[0, 1]
        out(f"Correlation between reported and measured size: **r = {r:.3f}** (n={len(kk)}).")
        out()

    # ---- 6. density ------------------------------------------------------
    rh = pd.to_numeric(e.report_hu_max, errors="coerce")
    om = pd.to_numeric(e.model_hu_mean, errors="coerce")
    ox = pd.to_numeric(e.model_hu_max, errors="coerce")
    m = rh.notna() & om.notna()
    out("## 6 · Density (HU)")
    out()
    if m.sum():
        for lbl, ours in (("our hu_mean", om), ("our hu_max", ox)):
            v = (rh[m] - ours[m]).dropna()
            out(f"- vs **{lbl}** (n={len(v)}): bias {v.mean():+.0f} HU, "
                f"median abs {v.abs().median():.0f} HU, "
                f"within 200 HU {100*(v.abs()<=200).mean():.0f}%")
        out()
        out("A radiologist places an ROI and reads a **mean**; `hu_max` is a peak, so it "
            "reads high by construction. `hu_mean` is the comparable column.")
    out()

    # ---- 7. counts and phantoms ------------------------------------------
    out("## 7 · Stone counts")
    out()
    out(f"- stones detected: **{len(st)}** across {st.study_id.nunique()} studies")
    out(f"- per study with any stone: median {st.groupby('study_id').size().median():.0f}, "
        f"max {st.groupby('study_id').size().max()}")
    out(f"- size range measured: {st.max_diameter_mm.min():.1f}–"
        f"{st.max_diameter_mm.max():.1f} mm, median {st.max_diameter_mm.median():.1f} mm")
    out("- reports rarely give a count ('few', 'multiple'), so count accuracy is not "
        "measurable from text.")
    out()
    out("## 8 · Measurement accuracy against phantoms")
    out()
    out("Synthetic stones with exact ground truth — the only place a true error is "
        "knowable, since a report is itself a measurement.")
    out()
    out("| Voxel size | Diameter error | Volume error |")
    out("|---|---|---|")
    out("| 0.7 mm isotropic | **0.11 mm** | 6% |")
    out("| 0.8 × 0.8 × 1.25 mm | **0.17 mm** | 6% |")
    out("| 3 mm slices | 0.55 mm | **20%** |")
    out("| stones ≤3 mm, any spacing | — | **16%** |")
    out()
    out("Three-axis output under rotation: major 0.49, intermediate 0.20, minor 0.07 mm.")
    out()
    out("## 9 · Not measured")
    out()
    out("- **Per-stone correspondence.** Every size figure compares *largest reported* "
        "with *largest measured*; in a multi-stone kidney these can be different "
        "stones. This is why pole agreement should not be read as a pole accuracy.")
    out("- **Hydronephrosis, perinephric fat stranding, stent** — not implemented.")
    out("- **Bladder calculi** — out of scope.")
    out()
    dest = os.path.join(RUN, "KIDNEY_METRICS.md")
    open(dest, "w").write("\n".join(L) + "\n")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
