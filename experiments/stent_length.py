"""EXPERIMENT: find a stent by measuring the OBJECT, not our corridor. Read-only.

TOUCHES NOTHING. Writes one CSV of its own; imports from calculus/ only.

WHY A SECOND ATTEMPT
experiments/stent_profile.py asked "is there dense material at every point along
the ureteric corridor?" It failed, and the failure is informative:

    8633709 (KNOWN STENT)  longest continuous dense run   15 mm / 25 mm
    8659576 (stones only)  longest continuous dense run   20 mm / 30 mm
    8662768 (staghorn)     longest continuous dense run   15 mm / 20 mm
    8664459 (staghorn)     longest continuous dense run   25 mm /  5 mm

No separation at all -- the stent study is not even the highest.

The reason is that it measured OUR APPROXIMATION rather than the stent. The
corridor centreline is a geometric guess running 6-15 mm off the true ureter
(all 12 stent detections were recorded 6-15 mm off path), and all 12 sat in a
single 4 mm band of arc position -- so the corridor and the stent occupy the
same space for one short stretch of a 236 mm tract. Continuity measured along
the corridor cannot report continuity along the ureter.

THE DIRECT MEASUREMENT
A double-J stent is a single connected object 24-30 CM long. That is a property
of the stent itself, independent of any geometry we construct, so measure it:

    every dense (>= BONE_HU) connected component, minus the bone masks
      -> its length, its volume, and length / volume^(1/3)

    a stone      3 - 30 mm long,  compact  -> aspect ~1-3
    a stent    200 - 300 mm long, hollow   -> aspect ~20-30

An order of magnitude, on a quantity that needs no centreline.

A stent also survives the detector's own bone rule by volume: a 30 cm tube of
2 mm bore is about 950 mm3, well under BONE_MIN_VOL_MM3 = 3000. So it is already
sitting in the data, intact, and nothing has ever looked at it.

RISK BEING TESTED, NOT ASSUMED: the stent passes the sacrum and iliac bone, so
its dense component may fuse with bone and vanish into a "bone" component. The
bone MASKS are therefore subtracted before labelling, which should cut the stent
into a few long pieces rather than losing it entirely. Whether that actually
happens is exactly what this script is for -- and if the longest object in the
stent study is not conspicuous, this approach fails too and I would rather find
that out here than tune a threshold into it.

Usage:
    python -m experiments.stent_length --studies 8633709 8659576 8677813
    python -m experiments.stent_length              # every study on disk
"""
import argparse
import glob
import os
import sys

import cc3d
import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calculus.common.paths import NIFTI, RUN               # noqa: E402
from calculus.kidney import detect_stones as ds            # noqa: E402

BONE_KEYS = ("vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4",
             "vertebrae_L5", "vertebrae_T12", "sacrum",
             "hip_left", "hip_right", "femur_left", "femur_right")
MIN_VOL_MM3 = 20.0          # ignore specks
TOP_N = 6


def run_study(sid):
    p = os.path.join(NIFTI, f"{sid}.nii.gz")
    if not os.path.exists(p):
        return []
    img = nib.load(p)
    vol = np.asanyarray(img.dataobj).astype(np.float32)
    sp = np.array([float(z) for z in img.header.get_zooms()[:3]])
    voxel_mm3 = float(np.prod(sp))
    masks = ds.load_masks(sid)

    bone = np.zeros(vol.shape, bool)
    for b in BONE_KEYS:
        if b in masks:
            bone |= masks[b]
    # ribs and costal cartilage too: dense, long, and they would otherwise
    # dominate any "longest thin object" ranking
    for k in masks:
        if k.startswith("rib") or k.startswith("costal"):
            bone |= masks[k]

    lab, n = cc3d.connected_components((vol >= ds.BONE_HU) & ~bone,
                                       connectivity=26, return_N=True)
    if not n:
        return []
    st = cc3d.statistics(lab)
    counts, bb = st["voxel_counts"], st["bounding_boxes"]
    rows = []
    for i in range(1, len(counts)):
        v = float(counts[i]) * voxel_mm3
        if v < MIN_VOL_MM3:
            continue
        b = bb[i]
        ext = np.array([(b[a].stop - b[a].start) * sp[a] for a in range(3)])
        # bounding-box diagonal, not a caliper across the voxels: a stent curves,
        # so its true arc length exceeds any straight line, and the diagonal is
        # a cheap lower bound that is still an order of magnitude above a stone
        diag = float(np.linalg.norm(ext))
        rows.append({"study_id": sid, "label": i, "volume_mm3": round(v, 1),
                     "bbox_len_mm": round(diag, 1),
                     "ext_cc_mm": round(float(ext[2]), 1),
                     "aspect": round(diag / max(v ** (1 / 3.0), 1e-6), 2),
                     "hu_max": round(float(vol[lab == i].max()))})
    rows.sort(key=lambda r: -r["bbox_len_mm"])
    return rows[:TOP_N]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", nargs="*", default=None)
    a = ap.parse_args()
    ids = a.studies or sorted(os.path.basename(f)[:-7]
                              for f in glob.glob(os.path.join(NIFTI, "*.nii.gz")))
    allr = []
    for sid in ids:
        r = run_study(sid)
        print(f"\n{sid}  (top {len(r)} longest dense non-bone objects)", flush=True)
        for x in r:
            print(f"    len {x['bbox_len_mm']:7.1f} mm  cc-ext {x['ext_cc_mm']:6.1f} mm"
                  f"  vol {x['volume_mm3']:9.1f} mm3  aspect {x['aspect']:6.2f}"
                  f"  {x['hu_max']:5d} HU", flush=True)
        allr += r
    if not allr:
        raise SystemExit("nothing measured")
    d = pd.DataFrame(allr)
    out = os.path.join(RUN, "csv", "stent_length.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    d.to_csv(out, index=False)
    print(f"\nwrote {out}")
    print("\nLONGEST object per study, ranked -- a stent should stand out:")
    top = d.sort_values("bbox_len_mm", ascending=False).groupby("study_id").head(1)
    print(top.sort_values("bbox_len_mm", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
