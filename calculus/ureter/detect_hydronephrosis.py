"""Hydronephrosis: dilatation of the collecting system, per kidney.

WHY THIS IS WORTH BUILDING
--------------------------
Two payoffs, which is why it comes before the classifier:

1. The report template has a HUN/HN column and we print "-". 101 reports in the
   cohort mention hydronephrosis, so it is a real finding we do not report at
   all -- and clinically it is what makes an obstructing stone urgent rather
   than incidental.

2. It is the strongest evidence available against our ureteric false positives.
   An obstructing stone dilates the system ABOVE it. A dense object in the
   ureteric corridor with a normal collecting system on that side is far more
   likely a phlebolith or a vessel calcification. Density and geometry have hit
   their ceiling at 54 % precision; this attacks the problem from a direction
   they cannot.

HOW IT IS MEASURED, with no annotation and no new model
------------------------------------------------------
TotalSegmentator's kidney class is PARENCHYMA ONLY -- it excludes the collecting
system and the sinus. That exclusion is the measurement:

    closing the mask by SINUS_FILL_MM fills the hilar concavity and the sinus
    cavity = (closed mask) MINUS (parenchyma mask), restricted to fluid density

Sinus FAT sits near -100 HU and urine near 0-20 HU, so a density window
separates a fat-filled normal sinus from a urine-filled dilated one. A normal
kidney yields a few millilitres; an obstructed one yields tens.

Deliberately NOT a threshold on kidney volume: an obstructed kidney can have a
normal total volume early on (dilatation replaces parenchyma), which is exactly
why study 8507585's thin parenchymal rim reads as a "bad mask" today.

GRADING is calibrated against report text, not invented: --calibrate sweeps the
cavity-volume cut to best separate studies whose report mentions hydronephrosis
from those whose report does not, and prints the sensitivity/specificity at each
cut so the choice is visible rather than buried.

    csv/hydronephrosis.csv     one row per study, both sides

Usage:
    CALCULUS_RUN=stone_analysis ./venv/bin/python utils/detect_hydronephrosis.py
    CALCULUS_RUN=stone_analysis ./venv/bin/python utils/detect_hydronephrosis.py --calibrate
"""
import argparse
import glob
import os
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
# package root -> project root: calculus/<sub>/x.py is two levels down
ROOT = os.path.dirname(os.path.dirname(HERE))
from calculus.common.paths import CSV, NIFTI, SEG                     # noqa: E402

SINUS_FILL_MM = 15.0        # same radius Part 1 uses to fill the sinus
URINE_LO, URINE_HI = -10.0, 30.0    # urine is ~0-20 HU; fat is ~-100
MIN_CAVITY_MM3 = 200.0      # below this it is noise at the mask boundary
# Grades, in mL of fluid inside the sinus. Provisional until --calibrate is run
# against the 101 reports that mention hydronephrosis.
GRADE = [(3.0, "none"), (8.0, "mild"), (20.0, "moderate"), (1e9, "severe")]


def ball(radius_mm, spacing):
    r = [max(1, int(round(radius_mm / s))) for s in spacing]
    zz, yy, xx = np.ogrid[-r[0]:r[0]+1, -r[1]:r[1]+1, -r[2]:r[2]+1]
    return ((zz / r[0])**2 + (yy / r[1])**2 + (xx / r[2])**2) <= 1.0


def cavity_for_side(vol, mask, spacing, se):
    """Fluid-density volume inside the sinus, in mL, plus its largest diameter.

    The subtraction is what makes this work: anything the closing added back is
    by definition NOT parenchyma, so it is sinus -- fat in a normal kidney,
    urine in a dilated one. Only the density window then has to do any work.
    """
    if not mask.any():
        return dict(cavity_ml=0.0, cavity_max_mm=0.0, n_cavities=0)
    closed = ndimage.binary_closing(mask, structure=se)
    sinus = closed & ~mask
    if not sinus.any():
        return dict(cavity_ml=0.0, cavity_max_mm=0.0, n_cavities=0)
    fluid = sinus & (vol >= URINE_LO) & (vol <= URINE_HI)
    if not fluid.any():
        return dict(cavity_ml=0.0, cavity_max_mm=0.0, n_cavities=0)
    vx = float(np.prod(spacing))
    lab, n = ndimage.label(fluid)
    sizes = ndimage.sum(fluid, lab, range(1, n + 1)) * vx
    keep = [i + 1 for i, s in enumerate(sizes) if s >= MIN_CAVITY_MM3]
    if not keep:
        return dict(cavity_ml=0.0, cavity_max_mm=0.0, n_cavities=0)
    total = float(sum(sizes[i - 1] for i in keep)) / 1000.0
    big = max(keep, key=lambda i: sizes[i - 1])
    idx = np.argwhere(lab == big) * np.array(spacing)
    dmax = float(np.linalg.norm(idx.max(axis=0) - idx.min(axis=0)))
    return dict(cavity_ml=round(total, 2), cavity_max_mm=round(dmax, 1),
                n_cavities=len(keep))
# WHAT THIS FUNCTION DOES: measures how much urine-density fluid sits in the
# middle of one kidney, which is what a dilated collecting system looks like.


def grade(ml):
    for cut, name in GRADE:
        if ml < cut:
            return name
    return "severe"


def one(sid):
    nii = os.path.join(NIFTI, f"{sid}.nii.gz")
    d = os.path.join(SEG, sid)
    if not (os.path.exists(nii) and os.path.isdir(d)):
        return None
    img = nib.load(nii)
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    vol = np.asanyarray(img.dataobj).astype(np.float32)
    se = ball(SINUS_FILL_MM, spacing)
    row = {"study_id": sid, "voxel_mm": "x".join(f"{s:.2f}" for s in spacing)}
    for side, name in (("left", "kidney_left"), ("right", "kidney_right")):
        p = os.path.join(d, f"{name}.nii.gz")
        if not os.path.exists(p):
            row.update({f"{side}_cavity_ml": "", f"{side}_grade": ""})
            continue
        m = np.asanyarray(nib.load(p).dataobj) > 0
        c = cavity_for_side(vol, m, spacing, se)
        pml = m.sum() * float(np.prod(spacing)) / 1000.0
        row.update({
            f"{side}_parenchyma_ml": round(pml, 1),
            f"{side}_cavity_ml": c["cavity_ml"],
            f"{side}_cavity_max_mm": c["cavity_max_mm"],
            # ratio matters as much as the absolute: a large cavity in a large
            # kidney is less abnormal than the same cavity in a thin rim
            f"{side}_cavity_ratio": (round(c["cavity_ml"] / pml, 3)
                                     if pml > 0 else ""),
            f"{side}_grade": grade(c["cavity_ml"]),
        })
    return row


def calibrate(df):
    """Sweep the cavity cut against reports that mention hydronephrosis."""
    import re
    import pandas as pd
    x = pd.read_excel(os.path.join(ROOT, "calculus_with_report.xlsx"),
                      sheet_name="jun-jul-2026")
    x["study_id"] = x.study_id.astype(str)
    txt = x.set_index("study_id").report_content.astype(str).to_dict()
    HN = re.compile(r"hydronephro|hydroureter|pelvicaly[cs]eal dilat|"
                    r"pcs dilat|dilated pelvi", re.I)
    NEG = re.compile(r"no (evidence of )?(gross )?hydronephro|"
                     r"not dilated|no pelvicaly[cs]eal dilat", re.I)
    lab = {}
    for sid in df.study_id:
        t = txt.get(sid, "")
        if not t:
            continue
        lab[sid] = bool(HN.search(t)) and not bool(NEG.search(t))
    df = df[df.study_id.isin(lab)].copy()
    df["hn_report"] = df.study_id.map(lab)
    df["max_cavity"] = df[["left_cavity_ml", "right_cavity_ml"]].apply(
        lambda r: max([v for v in r if v != ""] or [0]), axis=1)
    pos = int(df.hn_report.sum())
    print(f"\ncalibration set: {len(df)} studies, {pos} with hydronephrosis "
          f"stated in the report, {len(df)-pos} without")
    print(f"\n{'cut mL':>7} {'sens':>7} {'spec':>7} {'youden':>7}")
    best = None
    for cut in (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30):
        det = df.max_cavity >= cut
        tp = int((det & df.hn_report).sum()); fn = int((~det & df.hn_report).sum())
        tn = int((~det & ~df.hn_report).sum()); fp = int((det & ~df.hn_report).sum())
        se = 100*tp/max(tp+fn, 1); sp = 100*tn/max(tn+fp, 1)
        print(f"{cut:7} {se:6.1f}% {sp:6.1f}% {se+sp-100:6.1f}")
        if best is None or se + sp > best[1]:
            best = (cut, se + sp, se, sp)
    print(f"\nbest cut {best[0]} mL: sensitivity {best[2]:.1f}%, "
          f"specificity {best[3]:.1f}%  (Youden {best[1]-100:.1f})")
    print("NOTE the report label is imperfect: 'mild fullness' and 'prominent "
          "pelvis' are\nnot counted, and a report may omit mild dilatation "
          "entirely.")


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", nargs="*", default=None)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    dest = os.path.join(CSV, "hydronephrosis.csv")
    if os.path.exists(dest) and not a.overwrite and not a.studies:
        df = pd.read_csv(dest)
        df["study_id"] = df.study_id.astype(str)
        print(f"{dest} exists with {len(df)} studies (--overwrite to redo)")
        if a.calibrate:
            calibrate(df)
        return

    ids = a.studies or sorted(
        os.path.basename(f).split(".")[0]
        for f in glob.glob(os.path.join(NIFTI, "*.nii.gz")))
    rows = []
    for i, sid in enumerate(ids, 1):
        try:
            r = one(sid)
        except Exception as e:
            print(f"[{i}/{len(ids)}] {sid} FAILED {type(e).__name__}: {e}",
                  flush=True)
            continue
        if r is None:
            print(f"[{i}/{len(ids)}] {sid} no nifti/seg", flush=True)
            continue
        rows.append(r)
        print(f"[{i}/{len(ids)}] {sid}  L {r.get('left_cavity_ml','-')} mL "
              f"({r.get('left_grade','-')})   R {r.get('right_cavity_ml','-')} mL "
              f"({r.get('right_grade','-')})", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(CSV, exist_ok=True)
    df.to_csv(dest, index=False)
    print(f"\nwrote {dest}  ({len(df)} studies)")
    if a.calibrate:
        calibrate(df)


if __name__ == "__main__":
    main()
