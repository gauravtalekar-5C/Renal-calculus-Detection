"""EXPERIMENT: why does seed_peak_hu disagree with hu_max? Read-only.

TOUCHES NOTHING. Reads candidates.csv and the volumes; writes one CSV of its own.

THE OBSERVATION
---------------
On the 54-study audit cohort, 32 of 137 `no_dense_core` rejections have an
hu_max more than 2.5x their seed_peak_hu. Worst cases are 7-10 mm objects our own
FWHM measurement scored at 800-980 HU, discarded because the seed read ~150.

Only one of those two numbers can be right, and which one decides what we do:

  A) the BRIGHT VOXEL IS INSIDE the candidate's own >=GROW_HU component
     -> seed_peak_hu is being read off the wrong component, or off a component
        that clips the bright core. A BUG. Fixing it recovers real stones at no
        cost to precision.

  B) the BRIGHT VOXEL IS OUTSIDE that component
     -> the FWHM refinement has expanded into a neighbouring dense structure
        (vessel wall, rib edge, bowel content) and hu_max is INFLATED. That is
        worse than a missed stone: every density we report would be suspect.

WHAT THIS SCRIPT DOES
For each disagreeing candidate:
  1. crop a box around the recorded centroid
  2. rebuild the >=GROW_HU connected component containing that centroid, exactly
     as the detector does
  3. find where the hu_max value actually lives inside the box
  4. report whether that voxel is in the component, and how far it is from it

Usage:
    python -m experiments.seed_disagreement --ratio 2.5 --limit 20
"""
import argparse
import ast
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calculus.common.paths import CSV, NIFTI, RUN      # noqa: E402
from calculus.kidney import detect_stones as ds        # noqa: E402

BOX_MM = 20.0


def _centroid(v):
    try:
        return [float(x) for x in ast.literal_eval(str(v))]
    except (ValueError, SyntaxError):
        return None


def examine(vol, spacing, cen, hu_max_target, roi=None):
    """Return (verdict, dist_mm, comp_vox, comp_peak, target_found)."""
    half = [int(np.ceil(BOX_MM / s)) for s in spacing]
    sl = tuple(slice(max(0, int(round(cen[i])) - half[i]),
                     min(vol.shape[i], int(round(cen[i])) + half[i] + 1))
               for i in range(3))
    sub = vol[sl]
    loc = tuple(int(round(cen[i])) - sl[i].start for i in range(3))
    if not all(0 <= loc[i] < sub.shape[i] for i in range(3)):
        return "centroid outside box", np.nan, 0, np.nan, False

    # the candidate's own component, as the detector builds it. The ROI
    # RESTRICTION IS ESSENTIAL: without it the >=130 HU component bridges through
    # partial-volume voxels into rib and vertebra, producing components of tens of
    # thousands of voxels peaking at 1000-2250 HU. First run of this script omitted
    # it and every candidate duly came back "INSIDE component" -- comparing a
    # bone-contaminated component against the detector's confined one.
    m = sub >= ds.GROW_HU
    if roi is not None:
        m &= roi[sl]
    if not m[loc]:
        return "centroid not >= GROW_HU", np.nan, 0, np.nan, False
    lab, _ = ndimage.label(m)
    comp = lab == lab[loc]
    comp_peak = float(sub[comp].max())
    comp_vox = int(comp.sum())

    # where does the recorded hu_max actually live?
    hit = np.isclose(sub, hu_max_target, atol=1.0)
    if not hit.any():
        return "hu_max value not in box", np.nan, comp_vox, comp_peak, False
    inside = bool((hit & comp).any())
    if inside:
        return "INSIDE component", 0.0, comp_vox, comp_peak, True
    # distance from the component to the nearest voxel holding that value
    d = ndimage.distance_transform_edt(~comp, sampling=spacing)
    return "OUTSIDE component", float(d[hit].min()), comp_vox, comp_peak, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=2.5)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    c = pd.read_csv(os.path.join(CSV, "candidates.csv"))
    rr = c.reject_reason.fillna("").astype(str).str.strip()
    d = c[(rr == "no_dense_core") & (c.seed_peak_hu > 0)].copy()
    d["ratio"] = (d.hu_max / d.seed_peak_hu).round(2)
    d = d[d.ratio >= a.ratio].nlargest(a.limit, "hu_max")
    print(f"{len(d)} disagreeing candidates (ratio >= {a.ratio})\n")

    rows, cache = [], {}
    print(f"{'study':11s} {'size':>7s} {'seed':>6s} {'hu_max':>7s} {'ratio':>6s} "
          f"{'compPeak':>9s} {'compVox':>8s}  verdict")
    print("-" * 92)
    for r in d.itertuples():
        sid = str(r.study_id)
        if sid not in cache:
            p = os.path.join(NIFTI, f"{sid}.nii.gz")
            if not os.path.exists(p):
                cache[sid] = None
            else:
                img = nib.load(p)
                v = np.asanyarray(img.dataobj).astype(np.float32)
                spc = tuple(float(x) for x in np.abs(np.diag(img.affine))[:3])
                masks = ds.load_masks(sid)
                roi, _ = ds.build_roi(masks, v.shape, spc)
                cache[sid] = (v, spc, roi)
        if cache[sid] is None:
            continue
        vol, sp, _roi = cache[sid]
        cen = _centroid(r.centroid_vox)
        if cen is None:
            continue
        verdict, dist, cvox, cpeak, found = examine(vol, sp, cen, float(r.hu_max),
                                                    roi=cache[sid][2])
        print(f"{sid:11s} {r.max_diameter_mm:6.1f}m {r.seed_peak_hu:6.0f} "
              f"{r.hu_max:7.0f} {r.ratio:6.1f} {cpeak:9.0f} {cvox:8d}  {verdict}"
              + ("" if not np.isfinite(dist) or dist == 0 else f"  ({dist:.1f} mm away)"))
        rows.append({"study_id": sid, "max_diameter_mm": r.max_diameter_mm,
                     "seed_peak_hu": r.seed_peak_hu, "hu_max": r.hu_max,
                     "ratio": r.ratio, "component_peak_hu": cpeak,
                     "component_voxels": cvox, "verdict": verdict,
                     "distance_mm": dist})
    if not rows:
        raise SystemExit("nothing examined")
    out = pd.DataFrame(rows)
    dest = a.out or os.path.join(RUN, "csv", "seed_disagreement.csv")
    out.to_csv(dest, index=False)

    print("\n" + "=" * 92)
    print(out.verdict.value_counts().to_string())
    print()
    ins = out[out.verdict == "INSIDE component"]
    if len(ins):
        print(f"INSIDE  ({len(ins)}): the bright voxel IS in the candidate's own "
              "component.\n  -> seed_peak_hu is misread. A BUG, free to fix.")
        print(f"  component peak vs recorded seed: "
              f"median {float((ins.component_peak_hu / ins.seed_peak_hu).median()):.1f}x higher")
    outs = out[out.verdict == "OUTSIDE component"]
    if len(outs):
        print(f"\nOUTSIDE ({len(outs)}): the bright voxel is NOT in the component, "
              f"median {outs.distance_mm.median():.1f} mm away.\n"
              "  -> FWHM refinement is reaching into a neighbouring structure and "
              "hu_max is INFLATED.")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
