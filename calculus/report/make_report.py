"""Per-study clinical report tables, in the layout the urology report uses.

Produces, for every study:

    reports/<study_id>_findings.csv   the top table, one row per side + bladder
        Side | Size (in mm) : Volume (in cc) | HUN/HN | Calculus | PFS | Stent

    reports/<study_id>_calculi.csv    one row per stone
        Organ | Side | Size (in mm) | Density (HU) | Location | A/P

and the same concatenated across studies as reports/all_findings.csv /
reports/all_calculi.csv.

COLUMNS MATCH THE PRINTED REPORT EXACTLY -- nothing extra. Analysis fields
(volume in mm3, principal axes, elongation, location confidence, reject
reasons) are deliberately NOT here; they are all in csv/baseline_stones.csv and
csv/candidates.csv. Mixing the two makes the tables hard to read side by side
against a real report, which is the only job these files have.

DELIBERATELY EMPTY COLUMNS
--------------------------
hun_hn (hydronephrosis), pfs (perinephric fat stranding), stent, and the
ureteric rows with their distance from the UVJ are Part 2. They are emitted as
"-" rather than dropped, so the table lines up with the target layout and it is
obvious that the value is not-yet-measured rather than measured-as-absent.
Silently omitting a column reads as "no stent"; "-" reads as "not assessed".

WHERE EACH NUMBER COMES FROM
----------------------------
size_mm for a KIDNEY   principal axes of the segmentation mask (SVD), so the
                       long axis is the kidney's own, not the scanner's. A
                       tilted kidney still measures its true length.
volume_cc              voxel count x voxel volume. This is a true volumetric
                       measure, not the ellipsoid approximation from LxWxH.
size_mm for a STONE    dim_tr x dim_ap x dim_cc from shape_metrics() -- the
                       SCANNER axes, because that is what a radiologist reads
                       off axial and coronal images and therefore what the
                       report is comparable against. The stone's own principal
                       axes are in the CSV too (axis_major/intermediate/minor)
                       and are the better choice for tracking a stone between
                       scans, but they are not what a report quotes.
density_hu             peak HU inside the FWHM boundary.
location               our pole thirds, renamed to the calyceal wording the
                       report uses. NOTE this is a rename, not a finer
                       measurement: individual calyces are not resolvable on
                       non-contrast CT unless dilated, so "upper calyx" here
                       means "upper third of the kidney".
ap                     anterior or posterior, from the stone's position along
                       the AP axis relative to its own kidney's centroid.

Usage:
    CALCULUS_RUN=run_v5 ./venv/bin/python utils/make_report.py
    CALCULUS_RUN=run_v5 ./venv/bin/python utils/make_report.py --study 8231547
"""
import argparse
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# package root -> project root: calculus/<sub>/x.py is two levels down
ROOT = os.path.dirname(os.path.dirname(HERE))
from calculus.common.paths import CSV, RUN, SEG                    # noqa: E402

OUT = os.path.join(RUN, "reports")
NA = "-"                                   # not assessed, distinct from zero

# Our thirds -> the wording the urology report uses.
# WHAT WE ACTUALLY MEASURE, named honestly. The label comes from projecting the
# stone onto the kidney's long axis and taking a third -- we have no calyx
# segmentation (TotalSegmentator gives the whole kidney, not its collecting
# system), so calling a geometric third an "upper calyx" claimed anatomy we never
# resolved. A radiologist reading "lower calyx" assumes we identified the cup the
# stone sits in; we identified which third of the kidney it is in.
ZONE = {"upper_pole": "upper third",
        "interpolar": "mid third",
        "lower_pole": "lower third"}

# Stones outside the kidney tissue get no third, because a stone in the renal
# pelvis is not in any of them. The compartment is already known, so print it
# instead of leaving the cell blank -- an empty cell reads as "we do not know",
# which is a different and worse statement than "it is in the pelvis". Pelvis and
# perirenal are not separated by the detector, so the label names both rather
# than picking one.
COMPARTMENT_LOC = {
    "renal_pelvis_or_perirenal": "renal pelvis / perirenal",
    "bladder": "bladder",
    "ureter": "ureter",
}


# The SAME thirds, named the way a radiologist names them when the stone is in
# the collecting system. A stone in the sinus at the mid third is a mid-pole
# CALYCEAL calculus -- that is not a re-interpretation, it is what the anatomy
# at that location is. A stone in parenchyma at the mid third is not, so the two
# get different words.
ZONE_CALYX = {"upper_pole": "upper pole calyx",
              "interpolar": "mid pole calyx",
              "lower_pole": "lower pole calyx"}


def kidney_location(row):
    """Location cell for a kidney-side stone.

    Preference order: calyx (third + collecting system) > third > compartment.

    Until the sinus-closing fix landed in detect_stones.sinus_closed, calyceal
    stones reached here with NO zone at all -- the compartment test read the
    unclosed parenchyma mask, so a stone in a calyx fell outside the kidney and
    never had a third computed. Every such stone printed
    "renal pelvis / perirenal". That is why this project had never produced a
    report naming a calyx, despite calyceal stones being the commonest finding
    in the cohort.
    """
    loc = getattr(row, "location", None)
    if loc is not None and pd.notna(loc) and str(loc).strip():
        cs = getattr(row, "in_collecting_system", False)
        if cs is not None and pd.notna(cs) and bool(cs):
            return ZONE_CALYX.get(str(loc), ZONE.get(str(loc), str(loc)))
        return ZONE.get(str(loc), str(loc))
    comp = getattr(row, "compartment", None)
    if comp is not None and pd.notna(comp):
        return COMPARTMENT_LOC.get(str(comp), str(comp).replace("_", " "))
    return NA


def mask_dims_mm(mask, spacing):
    """Length x width x height of a mask along ITS OWN principal axes.

    A kidney sits at an angle in the body, so a bounding box measured along the
    scanner axes reports the box, not the organ -- it over-reads length on a
    tilted kidney. SVD finds the organ's own axes first.
    """
    idx = np.array(np.nonzero(mask), float).T
    if len(idx) < 4:
        return (0.0, 0.0, 0.0)
    pts = idx * np.array(spacing)
    centred = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    proj = centred @ vt.T
    ext = np.sort(proj.max(axis=0) - proj.min(axis=0))[::-1]
    return tuple(float(v) for v in ext)


def fmt_size(tr, ap, cc):
    """Size string in the REPORT's axis order, labelled.

    Radiologists measure with a caliper on one axial slice, so they quote two
    in-plane axes and 328 reports in the cohort state the order outright:
    "6.8 x 5.5 mm (APxTR)". We measure the 3D voxel extent, so we always have a
    third, cranio-caudal axis they never took -- and for a ureteric stone that
    third axis is frequently the LARGEST, because the stone is elongated along
    a tube running head-to-foot.

    Two consequences, both of which bit us on 8677121:

      - Emitting TR first while every report writes AP first meant a reader
        comparing our string to a report was silently comparing different axes.
      - Comparing our largest axis to their largest made us look like we
        over-measure, when their largest is in-plane and ours often is not.
        Ours read "7.2" against their "4.2" for a stone whose in-plane axes
        were 5.5 x 4.4 against their 4.2 x 3.7.

    So: report order, and say which axis is which. The suffix carries no digits,
    so anything parsing the numbers out is unaffected.
    """
    return f"{ap:.1f} x {tr:.1f} x {cc:.1f} (AP x TR x CC)"


# Bladder calculi. Until calculus.bladder.detect_bladder existed, the Calculus
# cell for the bladder printed "-" with the comment "0 here would be a claim we
# have not tested" -- which was right. It is now tested, so the cell can carry a
# number, and a real vesical calculus can appear in the stone table.
def _side_str(v):
    """Side as a string, treating NaN and None as unknown.

    `(r.side or "")` looks safe and is not: NaN is TRUTHY, so `NaN or ""`
    returns NaN and NaN.title() raises AttributeError. A kidney stone whose
    compartment is "bladder" or "unknown" is written with an empty side, which
    round-trips through CSV as NaN -- so make_report crashed on precisely the
    rows that have no side. The sweep script sent its output to /dev/null, so
    the failure showed up only as two fewer matched findings.
    """
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v)


def bladder_rows(sid):
    """Bladder calculi as report rows, and the count for the findings table."""
    p = os.path.join(CSV, "per_study", f"{sid}_bladder_candidates.csv")
    if not os.path.exists(p):
        return [], None                     # not run -> the cell stays "-"
    try:
        d = pd.read_csv(p)
    except Exception:
        return [], None
    if not len(d) or "is_stone" not in d.columns:
        return [], 0
    d = d[d.is_stone.astype(bool)]
    rows = []
    for r in d.itertuples():
        # A/P has no meaning in the bladder: it is a within-kidney reference.
        # dependent_frac is the analogous cue here (1 = at the bladder floor)
        # and is reported in the Location cell, because where a stone sits in
        # the bladder is what a urologist asks before planning cystolitholapaxy.
        dep = getattr(r, "dependent_frac", None)
        loc = "bladder lumen"
        if dep is not None and pd.notna(dep):
            loc += (" (dependent)" if float(dep) >= 0.6 else
                    " (non-dependent)" if float(dep) <= 0.3 else "")
        rows.append({
            "study_id": sid, "Organ": "Bladder", "Side": NA,
            "Size (in mm)": fmt_size(getattr(r, "dim_tr_mm", float("nan")),
                                     getattr(r, "dim_ap_mm", float("nan")),
                                     getattr(r, "dim_cc_mm", float("nan"))),
            "Density (HU)": int(r.hu_max), "Location": loc, "A/P": NA,
        })
    return rows, len(d)


# --- NEAR MISSES: what we found and were not confident enough to claim -------
#
# WHY THIS SECTION EXISTS, AND WHY IT IS NOT A LOWER THRESHOLD.
#
# On validation case 8633709 the report describes a calculus in the right mid
# calyx, 2.3 x 2.0 mm, ~170 HU. We detected it, measured it at 2.44 mm and
# 164 HU -- 0.14 mm and 3.5% from the radiologist -- and then deleted it,
# because the seed test reads its raw peak at 154 HU against a threshold of 200.
#
# The sweep priced the alternative: SEED_HU 200 -> 150 recovers that stone and
# adds 4 unmatched detections across 18 studies. But the entire benefit is ONE
# stone at 154 HU, and setting the threshold to 150 because of it is the
# MAX_DIAM_MM = 22 error inverted -- a bound fitted to the extreme of a tiny
# sample, which fails on the first case outside it. The next microlith at 140 HU
# would be missed and nothing would have been learned.
#
# So the threshold does not move. Instead the near-miss is REPORTED, in its own
# section, where a radiologist can overrule it in one glance. That fixes the
# actual defect -- silently dropping a stone we measured accurately -- without
# fitting a constant to a single case, and without putting anything unproven
# into the clinical table.
#
# WHAT QUALIFIES. Only rejections that were LINE-CALLS AGAINST A THRESHOLD:
#
#     no_dense_core        the seed peak fell short of SEED_HU
#     below_hu_floor       the density fell short of HU_FLOOR
#     below_min_diameter   the size fell short of MIN_DIAM_MM
#     too_small
#
# NOT categorical judgements -- bone, vascular_calcification, phlebolith_likely,
# extraureteric_calcification, tubular_not_stone, not_calculus_density. Those say
# "this is a different kind of object", not "this nearly cleared a line". Listing
# them would bury the useful cases in noise and invite exactly the over-reading
# this section exists to prevent.
#
# THE MARGIN DIFFERS BY COMPARTMENT, because the PRIOR does.
#
# First attempt used one generous margin (0.75) everywhere. On 8633709 that
# produced 15 near-misses of which 14 were ureteric objects at 225-289 HU in the
# upper corridor, and the one entry that mattered -- the 154 HU calyceal stone
# the radiologist reported -- came last. A review list that buries its own
# signal under noise is worse than no list: it trains the reader to skip it.
#
# The asymmetry is real and was measured when HU_FLOOR was set. Inside a kidney
# the search region is the organ itself, tightly bounded, so an object just under
# the seed is plausibly a stone. Inside the ureteric corridor -- a 20 mm tube
# that also contains bowel, vessel wall and partial-volume bone edges -- objects
# between 130 and 300 HU were measured at 44 FALSE POSITIVES PER STUDY, which is
# exactly why the floor is there. Surfacing them at a 0.75 margin re-imports the
# noise the floor exists to remove.
#
# So: generous in the kidney and the bladder, strict in the ureter.
NEAR_MISS_FRAC = {
    "no_dense_core":      0.75,   # kidney: organ-bounded, a near miss is plausible
    "below_hu_floor":     0.95,   # ureter: only the very closest are worth a look
    "below_min_diameter": 0.75,
    "too_small":          0.75,
}

THRESHOLD_REJECTIONS = {
    "no_dense_core":      ("seed_peak_hu", "SEED_HU"),
    "below_hu_floor":     ("hu_max", "HU_FLOOR"),
    "below_min_diameter": ("max_diameter_mm", "MIN_DIAM_MM"),
    "too_small":          ("max_diameter_mm", "MIN_DIAM_MM"),
}


def _threshold_for(name):
    from calculus.kidney import detect_stones as _ds
    from calculus.ureter import detect_ureteric as _du
    return {"SEED_HU": _ds.SEED_HU, "HU_FLOOR": _du.HU_FLOOR,
            "MIN_DIAM_MM": _ds.MIN_DIAM_MM}.get(name)


def near_miss_rows(sid):
    """Candidates rejected on a threshold, close enough to be worth a look."""
    out = []
    sources = (("kidney", f"{sid}_candidates.csv"),
               ("ureteric", f"{sid}_ureter_candidates.csv"),
               ("bladder", f"{sid}_bladder_candidates.csv"))
    for where, name in sources:
        p = os.path.join(CSV, "per_study", name)
        if not os.path.exists(p):
            continue
        try:
            d = pd.read_csv(p)
        except Exception:
            continue
        if not len(d) or "reject_reason" not in d.columns:
            continue
        rr = d.reject_reason.fillna("").astype(str).str.strip()
        for reason, (col, tname) in THRESHOLD_REJECTIONS.items():
            sub = d[rr == reason]
            if not len(sub) or col not in sub.columns:
                continue
            thr = _threshold_for(tname)
            if thr is None:
                continue
            for r in sub.itertuples():
                v = getattr(r, col, None)
                frac = NEAR_MISS_FRAC.get(reason, 0.75)
                if v is None or pd.isna(v) or float(v) < frac * thr:
                    continue
                loc = (kidney_location(r) if where == "kidney"
                       else (str(getattr(r, "vertebral_level", "") or "")
                             + " " + str(getattr(r, "zone", "") or "")).strip()
                       if where == "ureteric" else "bladder lumen")
                out.append({
                    "study_id": sid, "where": where,
                    "Side": _side_str(getattr(r, "side", "")).title(),
                    "Size (in mm)": round(float(getattr(r, "max_diameter_mm",
                                                        float("nan"))), 1),
                    "Density (HU)": (int(r.hu_max) if hasattr(r, "hu_max")
                                     and pd.notna(r.hu_max) else NA),
                    "Location": loc or NA,
                    "why_not_reported": reason,
                    "measured": round(float(v), 1),
                    "threshold": f"{tname} = {thr:g}",
                    # how close it came, as a fraction of its OWN threshold.
                    # Raw values are not comparable across thresholds: a 154 HU
                    # seed peak and a 289 HU density fail different lines, and
                    # sorting by the raw number put the useful entry last.
                    "closeness": round(float(v) / thr, 3),
                })
    # Closest to its OWN threshold first -- the most likely to be real.
    out.sort(key=lambda r: -float(r["closeness"]))
    return out


def kidney_rows(sid, stones):
    """One row per kidney plus a bladder row, matching the report's top table."""
    rows = []
    spacing = None
    for side, name in (("Right", "kidney_right"), ("Left", "kidney_left")):
        p = os.path.join(SEG, sid, f"{name}.nii.gz")
        if not os.path.exists(p):
            continue
        n = nib.load(p)
        spacing = tuple(float(v) for v in n.header.get_zooms()[:3])
        m = np.asanyarray(n.dataobj) > 0
        if not m.any():
            continue
        L, W, H = mask_dims_mm(m, spacing)
        vol_cc = m.sum() * float(np.prod(spacing)) / 1000.0
        n_st = int((stones.side == side.lower()).sum()) if len(stones) else 0
        # An abstention must not read as a negative -- see kidney_assessable.
        ok, note = kidney_assessable(sid)
        if not ok:
            n_st = f"NOT ASSESSED ({note})" if note else "NOT ASSESSED"
        rows.append({
            "study_id": sid,
            "Side": f"{side} Kidney",
            # size and volume are ONE column in the report: "L x W x H : V cc"
            "Size (in mm) : Volume (in cc)":
                f"{fmt_size(L, W, H)} : {vol_cc:.0f} cc",
            "HUN/HN": NA,                  # Part 2
            "Calculus": n_st,
            "PFS": NA,                     # Part 2
            "Stent": NA,                   # Part 2
        })
    # bladder: volume only, which is all the target layout shows
    p = os.path.join(SEG, sid, "urinary_bladder.nii.gz")
    if os.path.exists(p):
        n = nib.load(p)
        sp = tuple(float(v) for v in n.header.get_zooms()[:3])
        m = np.asanyarray(n.dataobj) > 0
        if m.any():
            rows.append({
                "study_id": sid, "Side": "Bladder",
                # the report shows volume only for the bladder
                "Size (in mm) : Volume (in cc)":
                    f"{m.sum() * float(np.prod(sp)) / 1000.0:.0f} cc",
                "HUN/HN": NA,
                # A number now, when the bladder detector has run for this
                # study; "-" when it has not. An unrun compartment and an empty
                # one are different statements and must not share a symbol.
                "Calculus": (NA if bladder_rows(sid)[1] is None
                             else bladder_rows(sid)[1]),
                "PFS": NA, "Stent": NA,
            })
    return rows


def kidney_assessable(sid):
    """Did the detector actually examine this study's kidneys?

    Returns (assessable, note).

    WHY THIS MATTERS MORE THAN ANY DETECTION FIX. detect_stones abstains on an
    enhanced or excretory-phase scan, and it is right to: iodinated contrast in
    the collecting system reads 300-1400 HU, indistinguishable from calculus, so
    anything reported would be guesswork. But the report then printed
    "Calculus: 0" -- a confident negative for a question nobody asked.

    Measured on the 54-study audit cohort: 6 of the 10 renal misses were
    contrast studies where the detector correctly declined and the report said
    zero. A silent zero is worse than a visible gap, because a clinician cannot
    know to look again.

    So an abstention is now stated. The same principle governs the bladder
    column, which has always printed '-' rather than 0 because the bladder is
    outside the search region.
    """
    p = os.path.join(CSV, "per_study", f"{sid}_summary.csv")
    if not os.path.exists(p):
        return True, ""
    try:
        d = pd.read_csv(p)
    except Exception:
        return True, ""
    if not len(d):
        return True, ""
    r = d.iloc[0]
    raw = r.get("error", "")
    # str(NaN) is the STRING "nan", which is truthy. This printed
    # "NOT ASSESSED (nan)" into the Calculus cell of a study that had been
    # analysed perfectly well. Fourth instance of this same confusion today --
    # after a crash here, a wrong API status, and a wrong impression line.
    err = "" if pd.isna(raw) else str(raw).strip()
    if err.lower() == "nan":
        err = ""
    if err:
        if "enhanced" in err.lower() or "excretory" in err.lower():
            med = r.get("kidney_median_hu")
            note = ("contrast: kidney parenchyma median "
                    f"{med:.0f} HU" if pd.notna(med) else "contrast")
            return False, note
        if "no segmentation" in err.lower():
            return False, "kidneys not segmented"
        return False, err[:60]
    return True, ""
# WHAT THIS FUNCTION DOES: reports whether the kidneys were actually examined,
# so the Calculus column can say NOT ASSESSED instead of 0 when they were not.


def ap_label(stones, sid):
    """Anterior or posterior, per stone.

    extract_series.py lays the array out as (left, posterior, superior), so
    axis 1 INCREASES towards the back. A stone whose axis-1 coordinate is
    greater than its own kidney's centroid is therefore posterior.

    Compared against the stone's OWN kidney rather than the midline, because
    the two kidneys sit at different depths and a single body-wide cut-off
    would label a whole side wrongly.
    """
    out = []
    centro = {}
    for side, name in (("left", "kidney_left"), ("right", "kidney_right")):
        p = os.path.join(SEG, sid, f"{name}.nii.gz")
        if os.path.exists(p):
            m = np.asanyarray(nib.load(p).dataobj) > 0
            if m.any():
                centro[side] = float(np.nonzero(m)[1].mean())
    for r in stones.itertuples():
        try:
            y = float(eval(r.centroid_vox)[1]) if isinstance(r.centroid_vox, str) \
                else float(r.centroid_vox[1])
        except Exception:
            out.append(NA); continue
        c = centro.get(r.side)
        out.append(NA if c is None else ("Posterior" if y > c else "Anterior"))
    return out


# The two detectors overlap at the PELVIURETERIC JUNCTION, and one stone can be
# reported twice.
#
# The kidney detector searches the sinus-closed kidney plus a 3 mm capsular cuff.
# The ureteric detector searches a corridor that begins at the PUJ and excludes
# only `binary_dilation(kidney, iterations=2)` -- about 1.7 mm. Between 1.7 mm
# and 3 mm outside the parenchyma, both regions contain the same voxels.
#
# Seen on validation case 8675742: a stone reported as
#     Renal Pelvis Or Perirenal, Right, 3.2 x 6.2 x 6.8, 1242 HU
# and again as
#     Ureter (upper), Right, 5.6 x 5.9 x 6.4, 1242 HU
# Identical density on the same side -- one physical calculus, counted twice.
# The report's own text describes ONE stone "approximately 0.5 mm from the
# pelviureteric junction".
#
# De-duplicated HERE rather than by widening the ureteric exclusion, because a
# stone genuinely at the PUJ must remain findable by BOTH detectors -- whichever
# reaches it should still find it. Only the REPORT must not count it twice.
#
# The kidney row is kept: at the PUJ the anatomical statement "renal pelvis" is
# more specific than "upper ureter", and the kidney measurement is made inside a
# region bounded by the kidney rather than by a corridor that also contains
# bowel and vessels. The dropped row's vertebral level is carried across, since
# that is information the kidney detector does not compute.
DUPLICATE_MM = 8.0


def _centroid_mm(row, spacing):
    """centroid_vox is stored as 'i,j,k' or '[i, j, k]'; return mm."""
    v = getattr(row, "centroid_vox", None)
    if v is None or not str(v).strip() or str(v).lower() == "nan":
        return None
    txt = str(v).replace("[", "").replace("]", "")
    try:
        idx = [float(x) for x in txt.split(",")]
    except ValueError:
        return None
    if len(idx) != 3:
        return None
    return np.array(idx) * np.asarray(spacing, float)


def drop_puj_duplicates(kidney_c, ureteric_c, spacing):
    """Remove ureteric rows that describe the same object as a kidney row.

    Returns (ureteric_rows_kept, n_dropped, levels_by_kidney_index).
    """
    if not kidney_c or not ureteric_c or spacing is None:
        return ureteric_c, 0, {}
    kcen = [(_i, _centroid_mm(r["_row"], spacing))
            for _i, r in enumerate(kidney_c) if r.get("_row") is not None]
    kept, dropped, levels = [], 0, {}
    for u in ureteric_c:
        ucen = _centroid_mm(u["_row"], spacing) if u.get("_row") is not None else None
        hit = None
        if ucen is not None:
            for ki, kc in kcen:
                if kc is None:
                    continue
                if float(np.linalg.norm(kc - ucen)) <= DUPLICATE_MM:
                    hit = ki
                    break
        if hit is None:
            kept.append(u)
        else:
            dropped += 1
            lvl = u.get("_level", "")
            if lvl:
                levels[hit] = lvl
    return kept, dropped, levels
# WHAT THIS FUNCTION DOES: stops one stone at the pelviureteric junction being
# counted once by each detector, keeping the kidney row and carrying over the
# vertebral level that only the ureteric detector knows.


def calculi_rows(sid, stones):
    if not len(stones):
        return []
    ap = ap_label(stones, sid)
    rows = []
    for r, a in zip(stones.itertuples(), ap):
        have3d = all(hasattr(r, k) and pd.notna(getattr(r, k))
                     for k in ("dim_tr_mm", "dim_ap_mm", "dim_cc_mm"))
        size = (fmt_size(r.dim_tr_mm, r.dim_ap_mm, r.dim_cc_mm) if have3d
                else f"{r.max_diameter_mm:.1f}")     # mesh unavailable
        rows.append({
            "study_id": sid,
            # compartment can be absent on a row that came from elsewhere;
            # crashing the whole report over one field is the wrong trade
            "Organ": ("Kidney" if r.compartment == "kidney" else
                      str(r.compartment).replace("_", " ").title()
                      if pd.notna(r.compartment) else "Kidney"),
            "Side": _side_str(r.side).title(),
            "Size (in mm)": size,
            "Density (HU)": int(r.hu_max),
            "Location": kidney_location(r),
            "A/P": a,
            # private, stripped before writing -- see drop_puj_duplicates
            "_row": r,
        })
    return rows


ZONE_UR = {"upper": "upper", "mid": "mid", "lower": "lower", "vuj": "VUJ"}
# Anatomical wording for the Location column. A bare "Distance from UVJ: 230 mm"
# is precise but unreadable at a glance -- a reporting radiologist wants "near
# the PUJ", which is the phrasing the urology report uses. The number stays,
# because the zone alone loses the position within the zone.
ZONE_TEXT = {"upper": "Near PUJ", "mid": "Mid ureter",
             "lower": "Near VUJ", "vuj": "At VUJ"}


def ureteric_rows(sid, ucand):
    """Ureteric calculi, as the report writes them: an Organ of
    'Ureter (lower)' and a Location of 'Distance from UVJ: X mm'.

    Only rows the detector marked `report_this` are used. That flag now means
    "every accepted stone" -- the top-2-per-side cap was removed once fixes to
    the bone masks, the HU floor and the touching-stone splitter took 8677121
    from 2 stones + 1 false positive to 3 true stones + 0 false positives, at
    which point the cap was withholding a real 428 HU calculus.

    OPEN RISK, stated because it has not been retested. The cap was originally
    justified by a measurement: on the 37-study validation cohort the full
    accepted set over-counted at a median of 3 per study against reports
    describing about one. Fixes 1-3 removed the mechanisms behind that
    over-count, but that has only been DEMONSTRATED on one study. Until the
    37-study cohort is re-scored, an uncapped table may over-count elsewhere.
    Re-cap via detect_ureteric.TOP_K_REPORTED if it does.

    Distance is reported as the ARC LENGTH along the interpolated ureteric
    course, not the straight line, because that is what "6 cm from the UVJ"
    means to a urologist planning access. Both are in the CSV.
    """
    if ucand is None or not len(ucand):
        return []
    u = ucand[(ucand.study_id.astype(str) == str(sid))
              & ucand.is_stone.astype(bool)]
    if "report_this" in u.columns:
        u = u[u.report_this.astype(bool)]
    rows = []
    for r in u.itertuples():
        have3d = all(hasattr(r, k) and pd.notna(getattr(r, k))
                     for k in ("dim_tr_mm", "dim_ap_mm", "dim_cc_mm"))
        size = (fmt_size(r.dim_tr_mm, r.dim_ap_mm, r.dim_cc_mm) if have3d
                else f"{r.max_diameter_mm:.1f}")
        zone = ZONE_UR.get(str(r.zone), str(r.zone or ""))
        d = getattr(r, "dist_to_uvj_along_mm", None)
        zt = ZONE_TEXT.get(str(r.zone), "")
        # VERTEBRAL LEVEL FIRST, because it is the reference the reports
        # themselves use ("at L3-L4 vertebral level", "at the L5-S1 level",
        # "at L4 level") and because it is read off the vertebral masks rather
        # than derived from a landmark we guessed. On 8676809 the UVJ landmark
        # sat 49 mm from the truth on a heavily distended bladder and this row
        # read "Mid ureter - 54.9 mm from UVJ" for a stone the radiologist put
        # at the VUJ. The level matched the report exactly on 2 of the 3 cases
        # that state one, and was one half-level out on the third.
        lvl = getattr(r, "vertebral_level", None)
        lvl = str(lvl).strip() if lvl is not None and pd.notna(lvl) else ""
        bits = []
        if lvl:
            bits.append(f"{lvl} level")
        if zt:
            bits.append(zt)
        if d is not None and pd.notna(d):
            # kept, but last and explicitly marked approximate -- it is the
            # right quantity for planning retrograde access and will become
            # trustworthy once the landmark is validated against a click
            bits.append(f"~{float(d):.0f} mm from UVJ")
        loc = " - ".join(bits) if bits else NA
        rows.append({
            "study_id": sid,
            "Organ": f"Ureter ({zone})" if zone else "Ureter",
            "Side": _side_str(r.side).title(),
            "Size (in mm)": size,
            "Density (HU)": int(r.hu_max),
            "Location": loc,
            # A/P is a within-kidney reference and has no meaning for a stone
            # in the ureter, so it is left not-assessed rather than invented
            "A/P": NA,
            # private, stripped before writing -- see drop_puj_duplicates
            "_row": r,
            "_level": lvl,
        })
    return rows
# WHAT THIS FUNCTION DOES: turns the ureteric detections into the same table
# rows the printed report uses, giving each stone its side, size, density, which
# third of the ureter it sits in, and how far it is from the bladder junction.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default=None)
    args = ap.parse_args()

    # Prefer the combined CSVs, but fall back to the per-study folder so this
    # can run WHILE detection is still going. detect_stones.py writes one file
    # per study as it finishes and only combines them at the very end, so
    # without this fallback the report tables would not exist until the whole
    # run completed -- which for 137 studies is several hours.
    import glob
    path = os.path.join(CSV, "baseline_stones.csv")
    per = os.path.join(CSV, "per_study")
    if os.path.exists(path):
        stones = pd.read_csv(path)
        summ = pd.read_csv(os.path.join(CSV, "baseline_summary.csv"))
        source = "combined"
    else:
        cf = sorted(glob.glob(os.path.join(per, "*_candidates.csv")))
        sf = sorted(glob.glob(os.path.join(per, "*_summary.csv")))
        if not sf:
            sys.exit(f"no results yet -- neither {path} nor {per}/*.csv exist")
        cand = (pd.concat([pd.read_csv(f) for f in cf], ignore_index=True)
                if cf else pd.DataFrame(columns=["reject_reason"]))
        # baseline_stones is the ACCEPTED subset; reproduce that filter here
        stones = (cand[cand.reject_reason.isna() | (cand.reject_reason == "")]
                  if len(cand) else cand)
        summ = pd.concat([pd.read_csv(f) for f in sf], ignore_index=True)
        source = f"per_study ({len(sf)} studies so far -- run still in progress)"
    print(f"reading from: {source}")

    # ureteric detections, when Part 2 has run for this run directory
    up = os.path.join(CSV, "ureter_candidates.csv")
    ucand = pd.read_csv(up) if os.path.exists(up) else None
    print("ureteric: " + (f"{int(ucand.is_stone.astype(bool).sum())} accepted "
                          f"across {ucand.study_id.nunique()} studies"
                          if ucand is not None else
                          "no ureter_candidates.csv -- kidney rows only"))

    ids = [args.study] if args.study else [str(s) for s in summ.study_id]

    os.makedirs(OUT, exist_ok=True)
    all_k, all_c = [], []
    for sid in ids:
        st = stones[stones.study_id.astype(str) == str(sid)] if len(stones) \
            else pd.DataFrame()
        k = kidney_rows(str(sid), st)
        # voxel spacing, for the PUJ duplicate test which compares centroids in
        # millimetres. Read from the kidney mask, which every study that got
        # this far has; None disables the test rather than guessing a spacing.
        spacing = None
        for _kk in ("kidney_left", "kidney_right"):
            _pp = os.path.join(SEG, str(sid), f"{_kk}.nii.gz")
            if os.path.exists(_pp):
                spacing = tuple(float(v) for v in
                                nib.load(_pp).header.get_zooms()[:3])
                break
        kc = calculi_rows(str(sid), st)
        uc_rows = ureteric_rows(str(sid), ucand)
        # One stone at the PUJ can be found by BOTH detectors; count it once.
        uc_rows, n_dup, dup_levels = drop_puj_duplicates(kc, uc_rows, spacing)
        if n_dup:
            print(f"  {sid}: dropped {n_dup} ureteric row(s) duplicating a "
                  f"kidney row at the PUJ", flush=True)
        # carry the vertebral level from the dropped row onto the kept one --
        # the kidney detector does not compute it
        for ki, lvl in dup_levels.items():
            if lvl and 0 <= ki < len(kc):
                loc = str(kc[ki].get("Location", "")).strip()
                kc[ki]["Location"] = f"{lvl} level - {loc}" if loc else f"{lvl} level"
        bl_rows, _n_bl = bladder_rows(str(sid))
        # THE SAME DE-DUPLICATION, for kidney-vs-bladder.
        #
        # The kidney detector's compartment logic has a `bladder` branch, so a
        # stone its ROI reaches inside the bladder is labelled "Bladder" -- and
        # the dedicated bladder detector reports it too. Seen on 8674941 and
        # 8676429: identical size and density, twice.
        #
        # Detected here by MEASUREMENT rather than by centroid, because the two
        # detectors crop differently and their centroids can differ by a voxel
        # or two while describing the same object; identical size AND density is
        # unambiguous. The bladder detector's row is kept: it measures inside the
        # lumen with the wall excluded, and "Bladder" is the correct organ.
        def _key(r):
            return (str(r.get("Size (in mm)")), str(r.get("Density (HU)")))
        bl_keys = {_key(r) for r in bl_rows}
        n_dup_bl = sum(1 for r in kc if _key(r) in bl_keys)
        if n_dup_bl:
            kc = [r for r in kc if _key(r) not in bl_keys]
            print(f"  {sid}: dropped {n_dup_bl} kidney row(s) duplicating a "
                  f"bladder row", flush=True)
        c = kc + uc_rows + bl_rows
        # strip the private keys before anything is written
        for _r in c:
            _r.pop("_row", None)
            _r.pop("_level", None)
        # ALWAYS WRITE, even with no rows.
        #
        # These used to be written only `if k:` / `if c:`, which means a study
        # whose findings all get rejected KEEPS ITS PREVIOUS FILE. Seen on
        # 8674941: five ureteric rows were correctly rejected as
        # tubular_not_stone, the detector recorded zero accepted candidates, and
        # the report on disk still showed all five -- dated an hour earlier.
        #
        # That is worse than a wrong number. A stale file looks current: nothing
        # in it says which run produced it, so re-running a study after a fix
        # leaves the old findings in place, and the fix appears not to have
        # worked -- or worse, appears to have worked while the old output is
        # what gets read. An empty table with a header is an honest answer.
        KCOLS = ["study_id", "Side", "Size (in mm) : Volume (in cc)",
                 "HUN/HN", "Calculus", "PFS", "Stent"]
        CCOLS = ["study_id", "Organ", "Side", "Size (in mm)", "Density (HU)",
                 "Location", "A/P"]
        pd.DataFrame(k, columns=None if k else KCOLS).to_csv(
            os.path.join(OUT, f"{sid}_findings.csv"), index=False)
        all_k += k
        pd.DataFrame(c, columns=None if c else CCOLS).to_csv(
            os.path.join(OUT, f"{sid}_calculi.csv"), index=False)
        all_c += c
        # Near misses go in their OWN file, never into the calculus table. They
        # are a review list, not findings. See near_miss_rows().
        NMCOLS = ["study_id", "where", "Side", "Size (in mm)", "Density (HU)",
                  "Location", "why_not_reported", "measured", "threshold",
                  "closeness"]
        nm = near_miss_rows(str(sid))
        pd.DataFrame(nm, columns=None if nm else NMCOLS).to_csv(
            os.path.join(OUT, f"{sid}_near_miss.csv"), index=False)
        if nm:
            print(f"  {sid}: {len(nm)} near-miss candidate(s) listed for review",
                  flush=True)
    if all_k:
        pd.DataFrame(all_k).to_csv(os.path.join(OUT, "all_findings.csv"),
                                   index=False)
    if all_c:
        pd.DataFrame(all_c).to_csv(os.path.join(OUT, "all_calculi.csv"),
                                   index=False)
    print(f"wrote {len(ids)} studies -> {OUT}")
    print(f"  {len(all_k)} kidney/bladder rows, {len(all_c)} calculus rows")
    print(f"\nnot yet measured (emitted as '{NA}'): hydronephrosis, "
          f"perinephric fat stranding, stent,\nand bladder calculi. Ureteric "
          f"rows are present but their distance rests on a UVJ\nlandmark that "
          f"has not been validated against a radiologist's click.")


if __name__ == "__main__":
    main()
