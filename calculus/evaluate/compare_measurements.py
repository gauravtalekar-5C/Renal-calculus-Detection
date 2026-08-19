"""Predicted stone sizes beside the sizes the radiologist wrote, in one CSV.

Two files, because two different questions get asked:

  csv/size_vs_report_study.csv    ONE ROW PER STUDY
      every size the report mentions, next to every size we measured, so a
      study can be eyeballed as a whole.

  csv/size_vs_report_matched.csv  ONE ROW PER MATCHED STONE
      each reported size paired with the measured stone closest to it in size
      on the same side. This is the per-stone comparison that has been missing:
      the study-level "largest vs largest" figure quietly compares DIFFERENT
      stones in a multi-stone kidney, which is why the old 51 % category
      agreement was never a real number.

WHICH MEASURED DIMENSION IS COMPARED, AND WHY IT MATTERS
-------------------------------------------------------
A radiologist quotes what they read off the axial and coronal images -- an
IN-PLANE dimension. Our max_diameter_mm is the 3D maximum caliper, which on an
oblique stone is systematically larger. Comparing those two is not like for
like: it puts a +1.81 mm bias into the numbers that is pure convention, not
error. Both are written to the CSV:

    model_largest_caliper_mm   the 3D maximum caliper  (our best geometry)
    model_largest_inplane_mm   max(dim_tr, dim_ap)     (comparable to a report)

and the summary is computed on the in-plane column.

WHAT THE GROUND TRUTH IS, AND WHAT IT IS NOT
--------------------------------------------
`calculus_line` from the cohort spreadsheet -- the sentences the reporting
radiologist wrote. Free and honest, but: sizes are rounded, "few calculi" is not
a count, and a report states what was worth saying rather than everything
present. A disagreement here is not automatically our error.

Usage:
    CALCULUS_RUN=run_v6 ./venv/bin/python utils/compare_measurements.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# package root -> project root: calculus/<sub>/x.py is two levels down
ROOT = os.path.dirname(os.path.dirname(HERE))
from calculus.common.paths import CSV                       # noqa: E402
import re                                    # noqa: E402
from calculus.evaluate.compare_reports import SIDE_RE, clauses, negated   # noqa: E402

# A size only counts as ground truth if its own clause is ABOUT a calculus.
# compare_reports.renal_sizes_mm filters on the clause mentioning the kidney,
# which is not the same thing: "Right kidney: Measures 11.2 x 4.7 cm" passes
# that test and contributed a 116 mm "stone". Same for "20 x 20 mm cortical
# cyst". Both were in the first run of this script and inflated mean absolute
# error to 6.0 mm while the median sat at 1.8.
CALC_RE = re.compile(r"calcul|stone|microlith|staghorn|nephrolith", re.I)
# Structures that carry their OWN measurement and would be mistaken for a stone.
# "hydronephro" and "hydrourete" were in this list and had to come out: an
# OBSTRUCTING ureteric stone is almost always described in the same clause as the
# dilatation it causes ("Mild hydronephrosis of approximately 9 x 6 mm sized
# calculus noted in left upper ureter"). Excluding those clauses silently dropped
# the most clinically important ureteric stones from both the case count and the
# size comparison. Dilatation is rarely given a millimetre size, so leaving it in
# costs little; excluding it cost a lot.
NOT_CALC_RE = re.compile(
    r"cyst|mass|lesion|angiomyolipom|tumou?r|abscess|collection|"
    r"prostat|cortical thickness|parenchymal thickness",
    re.I)
BLADDER_RE = re.compile(r"vesical|bladder", re.I)
URETER_RE = re.compile(r"ureter|uvj|vuj|pelvi-?uretic|pelviureteric|puj", re.I)

# One or more numbers joined by x / - followed by a unit: handles "6 mm",
# "1-2mm", "4.6 x 3.1 mm" and "4.6 x 3.1 x 5.7 mm". The pair-only regex in
# compare_reports drops the FIRST value of a three-axis measurement.
DIMS_RE = re.compile(
    r"((?:\d+(?:\.\d+)?\s*[x×\-]\s*)*\d+(?:\.\d+)?)\s*(mm|cm)\b", re.I)


def sizes_in(text):
    """Every size in one clause, in mm, all axes of a multi-axis measurement."""
    out = []
    for m in DIMS_RE.finditer(text or ""):
        mult = 10.0 if m.group(2).lower() == "cm" else 1.0
        for part in re.split(r"[x×\-]", m.group(1)):
            part = part.strip()
            if part:
                try:
                    out.append(float(part) * mult)
                except ValueError:
                    pass
    return out


def calculus_sizes(text):
    """{'renal': [...], 'ureteric': [...], 'bladder': [...]} from a report line.

    Split by compartment because our detector answers per compartment: a
    36 mm bladder stone is not a miss of ours, it is out of scope, and scoring
    it as a 31 mm error is simply wrong.
    """
    out = {"renal": [], "ureteric": [], "bladder": []}
    for part in clauses(text):
        if not CALC_RE.search(part) or negated(part):
            continue
        if NOT_CALC_RE.search(part):
            continue                     # the size belongs to a cyst, not a stone
        vals = sizes_in(part)
        if not vals:
            continue
        key = ("bladder" if BLADDER_RE.search(part) else
               "ureteric" if URETER_RE.search(part) else "renal")
        out[key].extend(vals)
    return out
# WHAT THIS FUNCTION DOES: pulls out only the measurements the report attaches to
# an actual stone, and files each one under the kidney, the ureter or the
# bladder, so like is compared with like.

XLSX = os.path.join(ROOT, "calculus_with_report.xlsx")
SHEET = "jun-jul-2026"
FLAG_MM = 5.0            # the difference worth a human look, as in the spine
                         # project's sheet: >=5 mm is not a rounding argument


def report_side(text):
    hits = [k for k, rx in SIDE_RE.items() if rx.search(str(text or ""))]
    return hits[0] if len(hits) == 1 else ("both" if len(hits) == 2 else "")


def inplane(row):
    """The larger of the two in-plane axes -- what a report would quote."""
    vals = [row.get("dim_tr_mm"), row.get("dim_ap_mm")]
    vals = [float(v) for v in vals if pd.notna(v)]
    return max(vals) if vals else float(row.get("max_diameter_mm") or np.nan)


def main():
    stones_p = os.path.join(CSV, "baseline_stones.csv")
    if not os.path.exists(stones_p):
        sys.exit(f"missing {stones_p} -- run detect_stones.py first")
    stones = pd.read_csv(stones_p)
    stones["study_id"] = stones.study_id.astype(str)
    stones["inplane_mm"] = stones.apply(inplane, axis=1)

    ur_p = os.path.join(CSV, "ureter_candidates.csv")
    ur = None
    if os.path.exists(ur_p):
        ur = pd.read_csv(ur_p)
        ur["study_id"] = ur.study_id.astype(str)
        ur = ur[ur.is_stone.astype(bool)]
        if "report_this" in ur.columns:
            ur = ur[ur.report_this.astype(bool)]
        ur["inplane_mm"] = ur.apply(inplane, axis=1)

    rep = pd.read_excel(XLSX, sheet_name=SHEET)
    rep["study_id"] = rep.study_id.astype(str)
    rep = rep.set_index("study_id")

    ids = sorted(set(stones.study_id) | (set(ur.study_id) if ur is not None
                                         else set()))
    study_rows, pair_rows = [], []
    for sid in ids:
        line = str(rep.calculus_line.get(sid, "") or "")
        rs = calculus_sizes(line)
        # our detectors cover kidney + ureter; bladder stones are recorded in
        # their own column and excluded from the error statistics
        r_sizes = sorted(rs["renal"] + rs["ureteric"], reverse=True)
        k = stones[stones.study_id == sid]
        u = ur[ur.study_id == sid] if ur is not None else pd.DataFrame()
        m_sizes = sorted(list(k.inplane_mm.dropna())
                         + list(u.inplane_mm.dropna() if len(u) else []),
                         reverse=True)
        r_max = max(r_sizes) if r_sizes else np.nan
        m_max = max(m_sizes) if m_sizes else np.nan
        cal = list(k.max_diameter_mm.dropna()) + (
            list(u.max_diameter_mm.dropna()) if len(u) else [])
        m_cal = max(cal) if cal else np.nan
        diff = (r_max - m_max) if (r_sizes and m_sizes) else np.nan
        study_rows.append({
            "study_id": sid,
            "report_sizes_mm": ";".join(f"{v:g}" for v in r_sizes),
            "report_max_mm": round(r_max, 1) if r_sizes else "",
            "report_renal_mm": ";".join(f"{v:g}" for v in sorted(rs["renal"], reverse=True)),
            "report_ureteric_mm": ";".join(f"{v:g}" for v in sorted(rs["ureteric"], reverse=True)),
            "report_bladder_mm_excluded": ";".join(f"{v:g}" for v in sorted(rs["bladder"], reverse=True)),
            "report_side": report_side(line),
            "n_report_sizes": len(r_sizes),
            "model_sizes_inplane_mm": ";".join(f"{v:.1f}" for v in m_sizes),
            "model_largest_inplane_mm": round(m_max, 1) if m_sizes else "",
            "model_largest_caliper_mm": round(m_cal, 1) if pd.notna(m_cal) else "",
            "n_model_kidney": len(k),
            "n_model_ureteric": len(u),
            "diff_mm_report_minus_model": (round(diff, 2)
                                           if pd.notna(diff) else ""),
            "abs_diff_mm": round(abs(diff), 2) if pd.notna(diff) else "",
            # Reports are inconsistent about WHICH axis they quote: some give a
            # single in-plane figure, others "15 x 11.4 x 26 mm (AP x TR x CC)"
            # whose maximum is the craniocaudal axis. Comparing a report max
            # against our in-plane max then reads as a 10 mm error that is pure
            # convention. Both differences are given so the reader can see which
            # one a disagreement lives in.
            "diff_mm_vs_caliper": (round(r_max - m_cal, 2)
                                   if (r_sizes and pd.notna(m_cal)) else ""),
            "abs_diff_vs_caliper_mm": (round(abs(r_max - m_cal), 2)
                                       if (r_sizes and pd.notna(m_cal)) else ""),
            f"diff_ge_{int(FLAG_MM)}mm": ("yes" if pd.notna(diff)
                                          and abs(diff) >= FLAG_MM else "no"),
            "report_line": line[:300],
        })

        # ---- per-stone matching: each reported size to its nearest measured
        # stone, greedily, largest reported first. Greedy on size (not position)
        # because a report gives no coordinates -- so this is an ASSIGNMENT, not
        # a proven correspondence, and is labelled as such in the CSV.
        pool = ([(v, "kidney", s) for v, s in zip(k.inplane_mm, k.side)]
                + [(v, "ureteric", s) for v, s in
                   zip(u.inplane_mm, u.side)] if len(u) else
                [(v, "kidney", s) for v, s in zip(k.inplane_mm, k.side)])
        pool = [p for p in pool if pd.notna(p[0])]
        rside = report_side(line)
        for rv in r_sizes:
            cand = [p for p in pool
                    if rside in ("", "both") or p[2] == rside] or pool
            if not cand:
                pair_rows.append({"study_id": sid, "report_size_mm": rv,
                                  "matched_model_mm": "", "compartment": "",
                                  "side": "", "diff_mm": "", "abs_diff_mm": "",
                                  "note": "no measured stone to match"})
                continue
            best = min(cand, key=lambda p: abs(p[0] - rv))
            pool.remove(best)
            pair_rows.append({
                "study_id": sid,
                "report_size_mm": round(rv, 1),
                "matched_model_mm": round(best[0], 2),
                "compartment": best[1],
                "side": best[2],
                "diff_mm": round(rv - best[0], 2),
                "abs_diff_mm": round(abs(rv - best[0]), 2),
                "note": "nearest-size assignment, same side where stated",
            })

    sdf = pd.DataFrame(study_rows)
    pdf = pd.DataFrame(pair_rows)
    sdf.to_csv(os.path.join(CSV, "size_vs_report_study.csv"), index=False)
    pdf.to_csv(os.path.join(CSV, "size_vs_report_matched.csv"), index=False)

    print(f"wrote {os.path.join(CSV, 'size_vs_report_study.csv')}  "
          f"({len(sdf)} studies)")
    print(f"wrote {os.path.join(CSV, 'size_vs_report_matched.csv')}  "
          f"({len(pdf)} matched pairs)")

    def stats(v, label):
        v = pd.to_numeric(v, errors="coerce").dropna()
        if not len(v):
            print(f"\n{label}: nothing comparable")
            return
        a = v.abs()
        print(f"\n{label}  (n={len(v)})")
        print(f"  bias (report - model)   {v.mean():+.2f} mm   "
              f"median {v.median():+.2f} mm")
        print(f"  mean abs error          {a.mean():.2f} mm   "
              f"median {a.median():.2f} mm")
        for t in (1, 2, 3, 5):
            print(f"  within {t} mm             {100*(a <= t).mean():.0f}%")

    stats(sdf.diff_mm_report_minus_model,
          "STUDY LEVEL - report max vs our IN-PLANE max")
    stats(sdf.diff_mm_vs_caliper,
          "STUDY LEVEL - report max vs our 3D CALIPER max")
    stats(pdf.diff_mm, "PER STONE - nearest-size assignment")
    n_flag = (sdf[f"diff_ge_{int(FLAG_MM)}mm"] == "yes").sum()
    print(f"\n{n_flag} study(ies) differ by >= {FLAG_MM:g} mm -- these are the "
          f"ones worth opening the overlay for.")


if __name__ == "__main__":
    main()
