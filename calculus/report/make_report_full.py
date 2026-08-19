"""The whole urology report for one study, as ONE csv — kidney and ureter together.

make_report.py writes two files per study (findings, calculi) because they are
two different table shapes. That is convenient for analysis and wrong for
reading: the printed report is a single page with a findings table, a per-side
calculus table with total counts, a bladder table, and numbered impressions.
This writes exactly that, as one CSV per study, with a `section` column so it
opens readably in Excel and still parses in one line of pandas.

    reports/<study_id>_report.csv     one study, every section
    reports/all_reports.csv           every study, same shape

SECTIONS, in the order the printed report uses:
    HEADER            study id, study type, kidney/bladder volumes
    FINDINGS          Side | Size : Volume | HUN/HN | Calculus | PFS | Stent
    CALCULUS_RIGHT    Organ | Size | Density | Location | A/P   + total count
    CALCULUS_LEFT     same
    CALCULUS_BLADDER  same
    IMPRESSION        numbered sentences

URETERIC LOCATION WORDING
------------------------
The detector's zones are geometric thirds of the interpolated ureteric course.
They are written out in the clinical phrasing a urologist reads:

    upper -> "near PUJ (proximal ureter)"
    mid   -> "mid ureter"
    lower -> "distal ureter (near VUJ)"
    vuj   -> "at the VUJ"

plus the arc-length distance from the UVJ. The distance is stated as
"approx." on purpose: the UVJ landmark is geometric and has never been checked
against a radiologist's click, so a spuriously precise number would overclaim.

STILL NOT MEASURED, and shown as "-" rather than omitted:
hydronephrosis (HUN/HN), perinephric fat stranding (PFS), stent, and bladder
calculi. Dropping the column would read as "absent"; "-" reads as "not
assessed".

Usage:
    CALCULUS_RUN=run_v6 ./venv/bin/python utils/make_report_full.py
    CALCULUS_RUN=run_v6 ./venv/bin/python utils/make_report_full.py --study 8231547
"""
import argparse
import glob
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# package root -> project root: calculus/<sub>/x.py is two levels down
ROOT = os.path.dirname(os.path.dirname(HERE))
from calculus.common.paths import CSV, RUN, SEG                              # noqa: E402
from calculus.report.make_report import (NA, ZONE, ap_label, fmt_size,        # noqa: E402
                         mask_dims_mm)

# one folder for every per-study table, not two
OUT = os.path.join(RUN, "reports")

# the clinical phrasing the user's report template uses
# Organ column: exactly the sample report's wording, "Ureter (lower)".
UR_SHORT = {"upper": "upper", "mid": "mid", "lower": "lower", "vuj": "VUJ"}
# Location column: the anatomical landmark the user asked for, then the
# distance. Separated by " - " and not a comma, so the field stays one cell in
# any viewer that does not honour CSV quoting.
UR_ZONE = {"upper": "Near PUJ", "mid": "Mid ureter",
           "lower": "Near VUJ", "vuj": "At VUJ"}


def kidney_block(sid, stones):
    """Findings rows plus the per-side volumes, from the segmentation masks."""
    rows, vols = [], {}
    for side, name in (("Right", "kidney_right"), ("Left", "kidney_left")):
        p = os.path.join(SEG, sid, f"{name}.nii.gz")
        if not os.path.exists(p):
            continue
        n = nib.load(p)
        sp = tuple(float(v) for v in n.header.get_zooms()[:3])
        m = np.asanyarray(n.dataobj) > 0
        if not m.any():
            continue
        L, W, H = mask_dims_mm(m, sp)
        cc = m.sum() * float(np.prod(sp)) / 1000.0
        vols[side] = cc
        n_st = int((stones.side == side.lower()).sum()) if len(stones) else 0
        rows.append([f"{side} Kidney", f"{fmt_size(L, W, H)} : {cc:.0f} cc",
                     NA, n_st, NA, NA])
    p = os.path.join(SEG, sid, "urinary_bladder.nii.gz")
    if os.path.exists(p):
        n = nib.load(p)
        sp = tuple(float(v) for v in n.header.get_zooms()[:3])
        m = np.asanyarray(n.dataobj) > 0
        if m.any():
            cc = m.sum() * float(np.prod(sp)) / 1000.0
            vols["Bladder"] = cc
            rows.append(["Bladder", f"{cc:.0f} cc", NA, NA, NA, NA])
    return rows, vols


def stone_lines(sid, stones, ureter):
    """{'right': [...], 'left': [...]} of calculus table rows, both compartments."""
    out = {"right": [], "left": []}
    if len(stones):
        ap = ap_label(stones, sid)
        for r, a in zip(stones.itertuples(), ap):
            side = str(r.side or "").lower()
            if side not in out:
                continue
            have3d = all(pd.notna(getattr(r, k, None))
                         for k in ("dim_tr_mm", "dim_ap_mm", "dim_cc_mm"))
            size = (fmt_size(r.dim_tr_mm, r.dim_ap_mm, r.dim_cc_mm) if have3d
                    else f"{r.max_diameter_mm:.1f}")
            organ = ("Kidney" if r.compartment == "kidney"
                     else str(r.compartment).replace("_", " ").title()
                     if pd.notna(r.compartment) else "Kidney")
            out[side].append([organ, size, int(r.hu_max),
                              ZONE.get(r.location, r.location or NA), a])
    if ureter is not None and len(ureter):
        for r in ureter.itertuples():
            side = str(r.side or "").lower()
            if side not in out:
                continue
            have3d = all(pd.notna(getattr(r, k, None))
                         for k in ("dim_tr_mm", "dim_ap_mm", "dim_cc_mm"))
            size = (fmt_size(r.dim_tr_mm, r.dim_ap_mm, r.dim_cc_mm) if have3d
                    else f"{r.max_diameter_mm:.1f}")
            zone = str(r.zone or "")
            loc = UR_ZONE.get(zone, "Ureter")
            if pd.notna(getattr(r, "dist_to_uvj_along_mm", None)):
                loc += f" - Distance from UVJ: {r.dist_to_uvj_along_mm:.1f} mm"
            out[side].append([f"Ureter ({UR_SHORT.get(zone, zone)})", size,
                              int(r.hu_max), loc, NA])
    return out


def impressions(sid, stones, ureter):
    """Numbered sentences, in the wording the printed report uses.

    Grouped by side and location and reporting the largest of each group,
    because that is how the sample report reads ("Multiple calculi are present
    in right upper calyx, largest measuring ..."). One line per stone would run
    to twenty lines on a staghorn kidney and read as a data dump.
    """
    out = []
    if len(stones):
        for side in ("right", "left"):
            s = stones[stones.side == side]
            if not len(s):
                continue
            for loc, grp in s.groupby(s.location.fillna("")):
                name = ZONE.get(loc, loc or "kidney")
                big = grp.loc[grp.max_diameter_mm.idxmax()]
                have3d = all(pd.notna(big.get(k)) for k in
                             ("dim_tr_mm", "dim_ap_mm", "dim_cc_mm"))
                size = (fmt_size(big.dim_tr_mm, big.dim_ap_mm, big.dim_cc_mm)
                        if have3d else f"{big.max_diameter_mm:.1f}")
                if len(grp) > 1:
                    out.append(f"Multiple calculi are present in {side} {name}, "
                               f"largest measuring {size} mm "
                               f"({int(big.hu_max)} HU).")
                else:
                    out.append(f"A {size} mm calculus ({int(big.hu_max)} HU) is "
                               f"present in the {side} {name}.")
    if ureter is not None and len(ureter):
        for r in ureter.itertuples():
            have3d = all(pd.notna(getattr(r, k, None))
                         for k in ("dim_tr_mm", "dim_ap_mm", "dim_cc_mm"))
            size = (fmt_size(r.dim_tr_mm, r.dim_ap_mm, r.dim_cc_mm) if have3d
                    else f"{r.max_diameter_mm:.1f}")
            where = UR_ZONE.get(str(r.zone or ""), "ureter").lower()
            d = (f" at approximately {r.dist_to_uvj_along_mm:.1f} mm from the UVJ"
                 if pd.notna(getattr(r, "dist_to_uvj_along_mm", None)) else "")
            out.append(f"{size} mm calculus ({int(r.hu_max)} HU) is present in "
                       f"the {r.side} ureter {where}{d}.")
    if not out:
        out.append("No calculus detected in either kidney or ureter.")
    # the honest footer -- these are absent from the model, not absent in the
    # patient, and a reader of the CSV alone must not infer otherwise
    out.append("NOT ASSESSED by this model: hydronephrosis, perinephric fat "
               "stranding, stent, bladder calculi.")
    if ureter is not None and len(ureter):
        out.append("Ureteric distances are approximate: the UVJ reference point "
                   "is geometric and not yet validated against a radiologist.")
    return out


def build(sid, stones, ureter):
    k = stones[stones.study_id.astype(str) == sid] if len(stones) else stones
    u = (ureter[ureter.study_id.astype(str) == sid]
         if ureter is not None and len(ureter) else None)
    find, vols = kidney_block(sid, k)
    lines = stone_lines(sid, k, u)

    rows = [["HEADER", "Study ID", sid, "", "", "", ""],
            ["HEADER", "Study", "CT KUB PLAIN", "", "", "", ""],
            ["HEADER", "Total calculi", len(k) + (len(u) if u is not None else 0),
             "", "", "", ""],
            ["FINDINGS", "Side", "Size (in mm) : Volume (in cc)", "HUN/HN",
             "Calculus", "PFS", "Stent"]]
    for r in find:
        rows.append(["FINDINGS"] + list(r))
    for side in ("right", "left"):
        tag = f"CALCULUS_{side.upper()}"
        rows.append([tag, f"{side.title()}: Total Counts", len(lines[side]),
                     "", "", "", ""])
        rows.append([tag, "Organ", "Size (in mm)", "Density (HU)", "Location",
                     "A/P", ""])
        for r in lines[side]:
            rows.append([tag] + list(r) + [""])
    rows.append(["CALCULUS_BLADDER", "Bladder: Total Counts", NA,
                 "", "", "", ""])
    rows.append(["CALCULUS_BLADDER", "Organ", "Size (in mm)", "Density (HU)",
                 "Location", "A/P", ""])
    rows.append(["CALCULUS_BLADDER", NA, NA, NA, NA, NA, ""])
    for i, line in enumerate(impressions(sid, k, u), 1):
        rows.append(["IMPRESSION", i, line, "", "", "", ""])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default=None)
    a = ap.parse_args()

    sp = os.path.join(CSV, "baseline_stones.csv")
    if not os.path.exists(sp):
        sys.exit(f"missing {sp} -- run detect_stones.py first")
    stones = pd.read_csv(sp)
    stones["study_id"] = stones.study_id.astype(str)

    up = os.path.join(CSV, "ureter_candidates.csv")
    ureter = None
    if os.path.exists(up):
        d = pd.read_csv(up)
        d["study_id"] = d.study_id.astype(str)
        d = d[d.is_stone.astype(bool)]
        ureter = d[d.report_this.astype(bool)] if "report_this" in d else d
    print("ureteric: " + (f"{len(ureter)} reportable stones"
                          if ureter is not None else "not available yet"))

    if a.study:
        ids = [a.study]
    else:
        ids = sorted({os.path.basename(f).split("_summary")[0]
                      for f in glob.glob(os.path.join(CSV, "per_study",
                                                      "*_summary.csv"))
                      if "_ureter_" not in os.path.basename(f)})
    os.makedirs(OUT, exist_ok=True)
    cols = ["section", "col1", "col2", "col3", "col4", "col5", "col6"]
    allr = []
    for sid in ids:
        rows = build(sid, stones, ureter)
        pd.DataFrame(rows, columns=cols).to_csv(
            os.path.join(OUT, f"{sid}_report.csv"), index=False)
        allr += [[sid] + r for r in rows]
    pd.DataFrame(allr, columns=["study_id"] + cols).to_csv(
        os.path.join(OUT, "all_reports.csv"), index=False)
    print(f"wrote {len(ids)} report(s) -> {OUT}")


if __name__ == "__main__":
    main()
