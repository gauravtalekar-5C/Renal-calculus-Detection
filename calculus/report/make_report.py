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
ZONE = {"upper_pole": "upper calyx",
        "interpolar": "middle calyx",
        "lower_pole": "lower calyx"}


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


def fmt_size(a, b, c):
    return f"{a:.1f} x {b:.1f} x {c:.1f}"


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
                # bladder calculi need the whole-tract ROI, which is off in
                # Part 1 -- so 0 here would be a claim we have not tested
                "Calculus": NA, "PFS": NA, "Stent": NA,
            })
    return rows


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
            "Side": (r.side or "").title(),
            "Size (in mm)": size,
            "Density (HU)": int(r.hu_max),
            "Location": ZONE.get(r.location, r.location or NA),
            "A/P": a,
        })
    return rows


ZONE_UR = {"upper": "upper", "mid": "mid", "lower": "lower", "vuj": "VUJ"}


def ureteric_rows(sid, ucand):
    """Ureteric calculi, as the report writes them: an Organ of
    'Ureter (lower)' and a Location of 'Distance from UVJ: X mm'.

    Only rows the detector marked `report_this` are used -- the top two per side
    by density. Every accepted candidate stays in csv/ureter_candidates.csv with
    its rank, so nothing is hidden, but the 37-study validation showed the full
    accepted set over-counts badly (a median of 3 per study against reports
    describing about one), and a clinical table is the wrong place to publish a
    number we know is inflated.

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
        loc = (f"Distance from UVJ: {float(d):.1f} mm"
               if d is not None and pd.notna(d) else NA)
        rows.append({
            "study_id": sid,
            "Organ": f"Ureter ({zone})" if zone else "Ureter",
            "Side": (r.side or "").title(),
            "Size (in mm)": size,
            "Density (HU)": int(r.hu_max),
            "Location": loc,
            # A/P is a within-kidney reference and has no meaning for a stone
            # in the ureter, so it is left not-assessed rather than invented
            "A/P": NA,
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
        c = calculi_rows(str(sid), st) + ureteric_rows(str(sid), ucand)
        if k:
            pd.DataFrame(k).to_csv(
                os.path.join(OUT, f"{sid}_findings.csv"), index=False)
            all_k += k
        if c:
            pd.DataFrame(c).to_csv(
                os.path.join(OUT, f"{sid}_calculi.csv"), index=False)
            all_c += c
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
