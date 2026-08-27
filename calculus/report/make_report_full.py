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
from calculus.report.make_report import (NA, ZONE, kidney_location, ap_label, fmt_size,        # noqa: E402
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


def _assessable(sid):
    """Whether detect_stones actually examined this study. See
    make_report.kidney_assessable for the reasoning; duplicated as a small local
    reader rather than importing, to keep the two report writers independent."""
    p = os.path.join(CSV, "per_study", f"{sid}_summary.csv")
    if not os.path.exists(p):
        return True, ""
    try:
        d = pd.read_csv(p)
    except Exception:
        return True, ""
    if not len(d):
        return True, ""
    raw = d.iloc[0].get("error", "")
    # pandas reads an empty CSV cell back as NaN, and str(NaN) is the STRING
    # "nan" -- truthy and non-empty. This printed "NOT ASSESSED for renal
    # calculi (nan)" into the impression of a study that had been analysed
    # perfectly well. Third instance of this same NaN confusion today, after a
    # crash in make_report and a wrong status in the API.
    err = "" if pd.isna(raw) else str(raw).strip()
    if err.lower() == "nan":
        err = ""
    if not err:
        return True, ""
    low = err.lower()
    if "enhanced" in low or "excretory" in low:
        return False, "intravenous contrast present"
    if "no segmentation" in low:
        return False, "kidneys not segmented"
    return False, err[:60]


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
                              kidney_location(r), a])
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
            # VERTEBRAL LEVEL FIRST. This was added to make_report (which writes
            # the calculi table) and NOT here, so the structured report -- and
            # therefore the API, which reads this file -- still printed only the
            # UVJ distance. That distance rests on a landmark measured 49 mm out
            # on one distended bladder; the level is read off the vertebral masks
            # and matched the radiologists' own wording ("at L5-S1 level") on 2
            # of the 3 cohort cases that state one. Two report writers, one of
            # them updated: the localisation improvement was invisible to every
            # consumer of the structured report.
            lvl = str(getattr(r, "vertebral_level", "") or "").strip()
            bits = []
            if lvl and lvl.lower() != "nan":
                bits.append(f"{lvl} level")
            zt = UR_ZONE.get(zone, "Ureter")
            if zt:
                bits.append(zt)
            if pd.notna(getattr(r, "dist_to_uvj_along_mm", None)):
                pass    # distance withheld -- see make_report.ZONE_UR
            loc = " - ".join(bits) if bits else "Ureter"
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
    # BLADDER. impressions() was written when only the kidney and ureter were
    # searched, so a bladder-only finding produced no sentence at all -- the
    # report read "No calculus detected in either kidney or ureter" while its own
    # CALCULUS_BLADDER section listed a stone. A report that contradicts itself
    # in two adjacent sections is worse than one that is merely incomplete.
    for b in _bladder_lines(sid):
        out.append(f"A {b[1]} mm calculus ({b[2]} HU) is present in the "
                   f"{b[3]}.")

    if not out:
        # "No calculus detected" is only honest when we actually looked. On a
        # contrast or excretory-phase scan the detector abstains, and this line
        # would otherwise turn an abstention into a negative finding -- which is
        # what happened to 6 of the 10 renal misses in the 54-study audit.
        ok, note = _assessable(sid)
        if ok:
            out.append("No calculus detected in the kidneys, ureters "
                       "or bladder.")
        else:
            out.append(f"NOT ASSESSED for renal calculi ({note}). This study "
                       "was not evaluated; absence of a finding here does NOT "
                       "mean absence of a calculus.")
    # the honest footer -- these are absent from the model, not absent in the
    # patient, and a reader of the CSV alone must not infer otherwise
    # Bladder calculi were on this list until the bladder detector existed.
    # Leaving them would tell a reader we had not looked, on a study where we
    # had looked and found one.
    out.append("NOT ASSESSED by this model: hydronephrosis, perinephric fat "
               "stranding, ureteric stent.")
    if ureter is not None and len(ureter):
        out.append("Ureteric distances are approximate: the UVJ reference point "
                   "is geometric and not yet validated against a radiologist.")
    return out


def _bladder_lines(sid):
    """Bladder calculi as report rows.

    Read from the bladder detector's own CSV rather than from `stones`: the
    kidney table does not contain them, and my first attempt at this section
    wrote `(k or [])` where k is a DataFrame, which raises
    "The truth value of a DataFrame is ambiguous". That error was invisible
    because the caller redirected stderr to /dev/null, so the report simply did
    not exist and the API returned report: null.
    """
    p = os.path.join(CSV, "per_study", f"{sid}_bladder_candidates.csv")
    if not os.path.exists(p):
        return []
    try:
        d = pd.read_csv(p)
    except Exception:
        return []
    if not len(d) or "is_stone" not in d.columns:
        return []
    d = d[d.is_stone.astype(bool)]
    out = []
    for r in d.itertuples():
        dep = getattr(r, "dependent_frac", None)
        loc = "bladder lumen"
        if dep is not None and pd.notna(dep):
            loc += (" (dependent)" if float(dep) >= 0.6
                    else " (non-dependent)" if float(dep) <= 0.3 else "")
        def _f(name):
            v = getattr(r, name, float("nan"))
            return f"{float(v):.1f}" if pd.notna(v) else "?"
        out.append(["Bladder",
                    f"{_f('dim_ap_mm')} x {_f('dim_tr_mm')} x "
                    f"{_f('dim_cc_mm')} (AP x TR x CC)",
                    int(r.hu_max) if pd.notna(r.hu_max) else NA, loc, NA])
    return out


def build(sid, stones, ureter):
    k = stones[stones.study_id.astype(str) == sid] if len(stones) else stones
    u = (ureter[ureter.study_id.astype(str) == sid]
         if ureter is not None and len(ureter) else None)
    find, vols = kidney_block(sid, k)
    lines = stone_lines(sid, k, u)

    # The header count omitted the bladder, so 8583083 -- seven ureteric and
    # five vesical calculi -- carried "Total calculi 7" above a table listing
    # twelve. Counted from the same _bladder_lines the CALCULUS_BLADDER section
    # below uses, so the two cannot disagree.
    n_bl = len(_bladder_lines(sid))
    rows = [["HEADER", "Study ID", sid, "", "", "", ""],
            ["HEADER", "Study", "CT KUB PLAIN", "", "", "", ""],
            ["HEADER", "Total calculi",
             len(k) + (len(u) if u is not None else 0) + n_bl,
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
    # BLADDER. This printed a row of dashes and a count of "-" because the
    # bladder was never searched. It is searched now, and a study with a detected
    # vesical calculus still showed "-" here while the header said
    # "Total calculi: 1" -- a report contradicting itself.
    bl = _bladder_lines(sid)
    rows.append(["CALCULUS_BLADDER", "Bladder: Total Counts", len(bl),
                 "", "", "", ""])
    rows.append(["CALCULUS_BLADDER", "Organ", "Size (in mm)", "Density (HU)",
                 "Location", "A/P", ""])
    if bl:
        for r in bl:
            rows.append(["CALCULUS_BLADDER"] + list(r) + [""])
    else:
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
