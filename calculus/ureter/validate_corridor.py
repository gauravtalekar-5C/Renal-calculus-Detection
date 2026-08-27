#!/usr/bin/env python
"""Measure whether the corridor got closer to real stones.

THE TEST. A ureteric calculus is, by definition, inside the ureter. So for any
calculus the radiologist confirmed, distance-to-centreline is a direct error
measurement on the centreline itself -- no annotation needed. If the corridor is
right, confirmed stones sit within a few millimetres of it.

The control matters as much as the test: mimics (candidates in studies whose
report names no calculus) must NOT come closer by the same amount. A path that
moves toward everything has just gotten wider, not better.

Old corridor for comparison is loaded from a saved copy, so both are measured on
identical masks in one pass.
"""
import argparse
import glob
import importlib.util
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from calculus.ureter import ureter_corridor as new   # noqa: E402

ROIS = ("kidney_left", "kidney_right", "urinary_bladder",
        "iliac_artery_left", "iliac_artery_right", "aorta",
        "vertebrae_L1", "vertebrae_L2", "vertebrae_L3",
        "vertebrae_L4", "vertebrae_L5", "sacrum")


def load_old(path):
    spec = importlib.util.spec_from_file_location("corridor_old", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def masks_for(seg_dir):
    out = {}
    for r in ROIS:
        p = os.path.join(seg_dir, f"{r}.nii.gz")
        if os.path.exists(p):
            out[r] = np.asanyarray(nib.load(p).dataobj) > 0
    return out


def off_path(path, cen, spacing):
    """Shortest distance in mm from a point to the polyline's sample points."""
    d = (np.asarray(path, float) - np.asarray(cen, float)) * np.asarray(spacing)
    return float(np.linalg.norm(d, axis=1).min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="/tmp/corridor_backup.py")
    ap.add_argument("--studies", nargs="+", required=True,
                    help="sid:nifti:seg_dir:cand_csv:reported(0/1)")
    a = ap.parse_args()
    old = load_old(a.old)

    rows = []
    for spec in a.studies:
        sid, nii, seg, cand, rep = spec.split(":")
        if not (os.path.exists(nii) and os.path.isdir(seg)
                and os.path.exists(cand)):
            print(f"  {sid}: missing inputs, skipped")
            continue
        img = nib.load(nii)
        spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
        shape = img.shape
        M = masks_for(seg)
        if "urinary_bladder" not in M:
            print(f"  {sid}: no bladder mask, skipped")
            continue
        try:
            B_old = old.build(M, shape, spacing)
            B_new = new.build(M, shape, spacing)
        except Exception as e:
            print(f"  {sid}: corridor failed ({type(e).__name__}: {e})")
            continue
        d = pd.read_csv(cand)
        if "is_stone" in d:
            d = d[d.is_stone.astype(bool)]
        for r in d.itertuples():
            s = str(getattr(r, "side", "")).lower()
            if s not in B_old or s not in B_new:
                continue
            cv = str(getattr(r, "centroid_vox", ""))
            try:
                cen = [float(x) for x in cv.split(",")]
            except Exception:
                continue
            rows.append({
                "study_id": sid, "side": s, "reported": int(rep),
                "mm": float(getattr(r, "max_diameter_mm", np.nan)),
                "hu": float(getattr(r, "hu_max", np.nan)),
                "zone": getattr(r, "zone", ""),
                "off_old": off_path(B_old[s]["path"], cen, spacing),
                "off_new": off_path(B_new[s]["path"], cen, spacing),
                "n_lumbar": B_new[s].get("n_lumbar", 0),
            })
        print(f"  {sid}: {len(B_new)} side(s), "
              f"lumbar waypoints {[B_new[k].get('n_lumbar') for k in B_new]}")

    if not rows:
        print("no measurements"); return 1
    D = pd.DataFrame(rows)
    D.to_csv("/tmp/corridor_validation.csv", index=False)
    print(f"\n{len(D)} candidate(s) across {D.study_id.nunique()} studies")
    for tag, sub in (("REPORTED ureteric calculus (ground truth: in the ureter)",
                      D[D.reported == 1]),
                     ("CONTROL - studies reporting no calculus",
                      D[D.reported == 0])):
        if not len(sub):
            continue
        print(f"\n{tag}   n={len(sub)}")
        print(f"  off-path OLD  median {sub.off_old.median():6.1f} mm   "
              f"p75 {sub.off_old.quantile(.75):6.1f}")
        print(f"  off-path NEW  median {sub.off_new.median():6.1f} mm   "
              f"p75 {sub.off_new.quantile(.75):6.1f}")
        imp = (sub.off_old - sub.off_new)
        print(f"  improvement   median {imp.median():+6.1f} mm   "
              f"closer in {int((imp > 0).sum())}/{len(sub)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
