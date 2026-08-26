"""EXPERIMENT: can a stent be told from a stone by CONTINUITY? Read-only.

TOUCHES NOTHING. Imports the corridor builder and the detector's constants,
writes one CSV of its own. Delete this file and the pipeline is unchanged.

THE PROBLEM
-----------
On validation case 8633709 a double-J ureteric stent produced TWELVE fabricated
"ureteric calculi" at 583-1025 HU, bilateral, all 32-36 mm from the UVJ -- while
both calculi the radiologist actually reported were missed. Nothing in the
pipeline knows what a stent is.

Density cannot help: a stent is a polyurethane tube with barium filler and reads
squarely in stone territory. Shape helps a little -- measured elongation 0.175
to 0.316 for the stent fragments against a median 0.439 for real stones -- but
per-candidate shape flagged real stones at the same rate, because a large
impacted ureteric stone is also long and thin.

THE HYPOTHESIS
--------------
The difference is not what a stent LOOKS like, it is how far it EXTENDS.

    a STONE is FOCAL      one dense spot, 3-20 mm of the tract
    a STENT is CONTINUOUS dense material at every point over 200-250 mm,
                          because it is a single tube from renal pelvis to
                          bladder

So: walk down the ureter and ask "is there anything dense here?" at every step.
A stone answers yes once. A stent answers yes all the way.

HOW IT IS MEASURED
------------------
Binned by ARC POSITION along the tract, not by distance to the centreline. That
matters: the interpolated centreline sits 6-15 mm off the true ureter (the stent
fragments were recorded at 6-15 mm off path), so sampling a narrow tube around
the centreline would miss the stent entirely. Instead every corridor voxel is
assigned to the nearest centreline point, and each 5 mm bin keeps its MAXIMUM
HU. The statistic is the longest unbroken run of dense bins.

NO THRESHOLD IS PROPOSED HERE. The point is to measure the separation between
the known stent study and the other 16 first. Choosing a cut before looking is
how MAX_DIAM_MM = 22 came to delete a real 23.2 mm calculus.

Usage:
    python -m experiments.stent_profile --studies 8633709 8659576
    python -m experiments.stent_profile              # every study on disk
"""
import argparse
import glob
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calculus.common.paths import CSV, NIFTI, RUN            # noqa: E402
from calculus.kidney import detect_stones as ds              # noqa: E402
from calculus.ureter import detect_ureteric as du            # noqa: E402
from calculus.ureter import ureter_corridor as uc            # noqa: E402

DENSE_HU = 300.0        # same floor the ureteric detector uses for "stone-like"
BIN_MM = 5.0            # tract resolution; a 5 mm stone occupies 1-2 bins
GAP_BINS = 1            # allow one empty bin inside a run: a stent's radio-
                        # opaque wall can dip below the floor where the tube
                        # runs obliquely through a voxel


def longest_run(dense, gap=GAP_BINS):
    """Longest run of True, tolerating up to `gap` consecutive False inside it."""
    best = cur = 0
    holes = 0
    for v in dense:
        if v:
            cur += 1
            holes = 0
        else:
            holes += 1
            if holes > gap:
                best = max(best, cur)
                cur = holes = 0
            else:
                cur += 1          # keep the run alive across a single dip
    return max(best, cur)


def profile_side(vol, sd, spacing):
    """Max HU per 5 mm of tract, for one side."""
    corridor = sd["corridor"]
    idx = np.argwhere(corridor)
    if not len(idx):
        return None
    sp = np.asarray(spacing, float)
    # nearest centreline point for every corridor voxel -> its arc position
    tree = cKDTree(sd["path"] * sp)
    _, near = tree.query(idx * sp, workers=-1)
    arc = np.asarray(sd["arclen"], float)[near]
    hu = vol[corridor]

    total = float(sd["arclen"][-1])
    nb = max(1, int(np.ceil(total / BIN_MM)))
    b = np.clip((arc / BIN_MM).astype(int), 0, nb - 1)
    prof = np.full(nb, -1000.0)
    np.maximum.at(prof, b, hu)              # max HU in each bin
    dense = prof >= DENSE_HU
    run = longest_run(dense)
    return {"tract_mm": round(total, 1),
            "n_bins": nb,
            "dense_bins": int(dense.sum()),
            "dense_frac": round(float(dense.mean()), 3),
            "longest_run_bins": int(run),
            "longest_run_mm": round(run * BIN_MM, 1),
            "run_frac_of_tract": round(run * BIN_MM / max(total, 1e-6), 3)}


def run_study(sid):
    p = os.path.join(NIFTI, f"{sid}.nii.gz")
    if not os.path.exists(p):
        return []
    img = nib.load(p)
    vol = np.asanyarray(img.dataobj).astype(np.float32)
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    masks = ds.load_masks(sid)
    if not masks:
        return []
    sides = uc.build(masks, vol.shape, spacing)
    if not sides:
        return []
    # the corridor must have BONE REMOVED exactly as the detector removes it,
    # or every profile is a plateau of vertebral and iliac cortex and the
    # statistic measures the skeleton instead of the ureter
    voxel_mm3 = float(np.prod(spacing))
    dense = vol >= ds.BONE_HU
    import cc3d
    dlab, dn = cc3d.connected_components(dense, connectivity=26, return_N=True)
    big = np.zeros(dn + 1, bool)
    big[1:] = (cc3d.statistics(dlab)["voxel_counts"][1:] * voxel_mm3) >= du.BONE_MIN_VOL_MM3
    bone = big[dlab]
    for b in ("vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4",
              "vertebrae_L5", "sacrum", "hip_left", "hip_right"):
        if b in masks:
            bone |= masks[b]
    kidney = np.zeros(vol.shape, bool)
    for k in ("kidney_left", "kidney_right"):
        if k in masks:
            kidney |= masks[k]
    from scipy import ndimage
    kidney_out = ndimage.binary_dilation(kidney, iterations=2) if kidney.any() else kidney

    out = []
    for name, sd in sides.items():
        sd = dict(sd)
        sd["corridor"] = sd["corridor"] & ~bone & ~kidney_out
        r = profile_side(vol, sd, spacing)
        if r:
            out.append({"study_id": sid, "side": name, **r})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ids = a.studies or sorted(os.path.basename(f)[:-7]
                              for f in glob.glob(os.path.join(NIFTI, "*.nii.gz")))
    rows = []
    for sid in ids:
        r = run_study(sid)
        for x in r:
            print(f"  {x['study_id']:9s} {x['side']:5s} tract {x['tract_mm']:6.1f} mm "
                  f"dense {x['dense_frac']:.3f}  longest run {x['longest_run_mm']:6.1f} mm "
                  f"({x['run_frac_of_tract']:.3f} of tract)", flush=True)
        rows += r
    if not rows:
        raise SystemExit("nothing profiled")
    d = pd.DataFrame(rows)
    dest = a.out or os.path.join(RUN, "csv", "stent_profile.csv")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    d.to_csv(dest, index=False)
    print(f"\nwrote {dest}")
    print("\nranked by longest continuous dense run:")
    print(d.sort_values("longest_run_mm", ascending=False)
           [["study_id", "side", "tract_mm", "dense_frac",
             "longest_run_mm", "run_frac_of_tract"]].to_string(index=False))


if __name__ == "__main__":
    main()
