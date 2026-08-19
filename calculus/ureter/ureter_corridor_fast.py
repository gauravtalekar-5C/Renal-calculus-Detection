"""The same ureteric corridor, computed ~20x faster, with identical output.

WHAT WAS SLOW, MEASURED
-----------------------
Timing one study end to end:

    load volume                              2.3 s
    load 12 masks                            5.0 s
    BUILD CORRIDOR (full-volume EDT x2)    115.1 s   <-- the whole cost
    dense components over full volume        0.3 s
    denoise one round on the crop            0.4 s

Denoising -- the step everyone assumes is expensive, including me for three
attempts -- takes under half a second. Building the corridor takes two minutes.

WHY
---
ureter_corridor.build runs, once per side:

    d = ndimage.distance_transform_edt(~pm, sampling=spacing)
    corridor = d <= radius_mm

That computes the distance to the ureter path for ALL 171 million voxels of the
scan, then keeps the ~400,000 within 20 mm. Over 99 % of the work is discarded,
and a float64 output over a full volume is also what makes each worker hold
~15 GB.

THE FIX, AND WHY IT CANNOT CHANGE THE ANSWER
--------------------------------------------
Every voxel of the corridor is within `radius_mm` of the path. So take the
bounding box of the path, pad it by radius_mm in every direction, and compute
the distance transform only inside that box. Any voxel outside the box is
further than radius_mm from every path voxel BY CONSTRUCTION -- it could never
have satisfied `d <= radius_mm`. The corridor is therefore bit-identical, not
approximately equal.

`dist_to_path` is returned as a full-volume float32 array with +inf outside the
box: outside values are unknown but provably greater than radius_mm, and every
candidate that uses this map lies inside the corridor and therefore inside the
box, where the value is exact. float32 rather than float64 halves the memory at
0.001 mm precision, which is four orders of magnitude finer than a voxel.

Everything else -- the landmarks, the centreline, the smoothing -- is imported
from ureter_corridor, not copied, so the two cannot drift apart.

VERIFY BEFORE TRUSTING:
    ./venv/bin/python utils/ureter_corridor_fast.py --check 8231547
compares both implementations voxel by voxel and prints the timings.
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
# package root -> project root: calculus/<sub>/x.py is two levels down
ROOT = os.path.dirname(os.path.dirname(HERE))
from calculus.ureter import ureter_corridor as uc                       # noqa: E402
from calculus.ureter.ureter_corridor import (CORRIDOR_MM, arclen_mm, centreline,   # noqa: E402
                             classify_zone, landmark_iliac, landmark_puj,
                             landmark_uvj, orientation_ok, path_to_mask,
                             zone_bounds)


def _box(path_mask, shape, spacing, radius_mm):
    """Bounding box of the path, padded by radius_mm in voxels per axis."""
    idx = np.nonzero(path_mask)
    pad = [int(np.ceil(radius_mm / s)) + 1 for s in spacing]
    return tuple(slice(max(0, int(i.min()) - p),
                       min(n, int(i.max()) + p + 1))
                 for i, p, n in zip(idx, pad, shape))


def build(masks, shape, spacing, radius_mm=CORRIDOR_MM):
    """Drop-in replacement for ureter_corridor.build, same keys, same values."""
    out = {}
    bladder = masks.get("urinary_bladder")
    if bladder is None or not bladder.any():
        return out
    mx = uc._midline_x(masks, shape)

    for side in ("left", "right"):
        kid = masks.get(f"kidney_{side}")
        if kid is None or not kid.any():
            continue
        puj = landmark_puj(kid, mx)
        uvj = landmark_uvj(bladder, side, mx)
        if puj is None or uvj is None:
            continue
        ili, real = landmark_iliac(masks, side, puj, uvj)
        path = centreline(puj, ili, uvj)
        pm = path_to_mask(path, shape)
        if not pm.any():
            continue

        # ---- the whole optimisation is these four lines --------------------
        box = _box(pm, shape, spacing, radius_mm)
        d_sub = ndimage.distance_transform_edt(~pm[box], sampling=spacing)
        dist = np.full(shape, np.inf, np.float32)   # +inf = provably > radius
        dist[box] = d_sub.astype(np.float32)
        corridor = np.zeros(shape, bool)
        corridor[box] = d_sub <= radius_mm

        out[side] = {
            "corridor": corridor,
            "dist_to_path": dist,
            "path": path,
            "arclen": arclen_mm(path, spacing),
            "puj": puj, "iliac": ili, "uvj": uvj,
            "iliac_real": real,
        }
    return out
# WHAT THIS FUNCTION DOES: builds the same ureteric corridor as the original, but
# measures distances only inside a box around the path instead of across the
# whole scan -- which is where all the time was going.


def check(sid):
    """Prove the two implementations agree, and time both."""
    import nibabel as nib
    from calculus.common.paths import NIFTI, SEG
    img = nib.load(os.path.join(NIFTI, f"{sid}.nii.gz"))
    shape = img.shape
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    masks = {}
    for n in ("kidney_left", "kidney_right", "urinary_bladder",
              "iliac_artery_left", "iliac_artery_right"):
        p = os.path.join(SEG, sid, f"{n}.nii.gz")
        if os.path.exists(p):
            masks[n] = np.asanyarray(nib.load(p).dataobj) > 0

    t = time.time(); slow = uc.build(masks, shape, spacing); t_slow = time.time() - t
    t = time.time(); fast = build(masks, shape, spacing);    t_fast = time.time() - t

    print(f"study {sid}   volume {shape}  voxel "
          f"{spacing[0]:.2f}x{spacing[1]:.2f}x{spacing[2]:.2f} mm")
    print(f"  original   {t_slow:7.1f}s")
    print(f"  fast       {t_fast:7.1f}s     {t_slow / max(t_fast, 1e-9):.1f}x faster")
    ok = set(slow) == set(fast)
    for side in sorted(slow):
        a, b = slow[side]["corridor"], fast[side]["corridor"]
        same = bool((a == b).all())
        ok &= same
        # distances only have to agree where they are used: inside the corridor
        da = slow[side]["dist_to_path"][a]
        db = fast[side]["dist_to_path"][a]
        dmax = float(np.abs(da - db).max()) if da.size else 0.0
        ok &= dmax < 1e-3
        print(f"  {side:5}  corridor voxels {int(a.sum()):8}  identical: {same}   "
              f"max distance difference {dmax:.2e} mm")
    print(f"\n  {'IDENTICAL - safe to use' if ok else 'DIFFERENT - do not use'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", metavar="STUDY_ID", required=True)
    a = ap.parse_args()
    sys.exit(0 if check(a.check) else 1)
