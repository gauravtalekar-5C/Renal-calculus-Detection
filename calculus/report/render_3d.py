#!/usr/bin/env python
"""One self-contained interactive 3D scene per study, for the client-facing view.

WHY 3D AT ALL, AND WHAT IT MUST NOT IMPLY
A stone report answers three questions: where is it, how big is it, and is
anything obstructed. The flat captures answer the middle one well and the first
one badly -- a reader has to assemble a mental model of the tract from a stack of
axial crops. A single rotatable scene answers "where" directly, and stone BURDEN
(five small stones scattered through one kidney versus one large one at the
junction) is a shape the eye reads instantly in 3D and slowly in a table.

What it must NOT do is imply precision we do not have. So:

  * kidneys and bladder are drawn from the ACTUAL segmentation masks. Those are
    real surfaces, TotalSegmentator Dice ~0.96 and ~0.90.
  * bone is drawn faintly, purely so the viewer knows which way is up.
  * NO URETER IS DRAWN. We cannot see the ureter on non-contrast CT and our
    estimated corridor sits up to 20 mm from the stones it is meant to contain.
    Drawing a tube there would be the most convincing lie in the picture.
  * stones are spheres at their measured centroid, scaled to their measured
    diameter. A sphere is honest about being a summary; a lumpy mesh would
    suggest we had segmented the stone's true shape at that fidelity.

NO DEPENDENCIES BEYOND WHAT IS INSTALLED. plotly, pyvista and trimesh are all
absent, so surfaces come from skimage.marching_cubes and the viewer is a small
hand-written WebGL renderer inlined into the file. That also makes the output
safe to publish: one HTML file, no CDN, no external requests.
"""
import argparse
import base64
import json
import os
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.measure import marching_cubes

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from calculus.common.paths import CSV, NIFTI, SEG, ensure   # noqa: E402

# Surfaces are extracted from a downsampled mask, at a resolution chosen per
# structure by how much the viewer is meant to look at it. Measured on 8583083
# at a uniform 2 mm the file came to 3.9 MB, of which BONE was 2.7 MB (116,678
# triangles) -- two thirds of the download spent on scenery nobody inspects.
# Triangle count scales with the square of the resolution, so 5 mm bone is ~6x
# cheaper and looks identical at 10% opacity.
TARGET_MM = 2.0             # kidneys: small, and the thing being examined
TARGET_MM_BLADDER = 3.0     # large and smooth, detail buys nothing
TARGET_MM_BONE = 5.0        # a ghost for orientation
SMOOTH_SIGMA = 0.8          # softens the voxel staircase before meshing

ORGANS = [
    # name, mask, colour, opacity, group
    ("Right kidney", "kidney_right", (0.35, 0.62, 0.95), 0.26, "organ"),
    ("Left kidney",  "kidney_left",  (0.35, 0.62, 0.95), 0.26, "organ"),
    ("Bladder",      "urinary_bladder", (0.98, 0.78, 0.28), 0.22, "organ"),
]
BONES = ["sacrum", "hip_left", "hip_right",
         "vertebrae_L1", "vertebrae_L2", "vertebrae_L3",
         "vertebrae_L4", "vertebrae_L5"]


def _mask(seg_dir, name):
    p = os.path.join(seg_dir, f"{name}.nii.gz")
    if not os.path.exists(p):
        return None
    m = np.asanyarray(nib.load(p).dataobj) > 0
    return m if m.any() else None


def surface(mask, spacing, target_mm=None):
    """(verts_mm, faces) for one binary mask, decimated by downsampling.

    Downsampling the MASK rather than decimating the mesh afterwards: it is one
    zoom instead of a quadric-collapse implementation, and at 2 mm the loss is
    invisible against an organ 50 mm across.
    """
    target_mm = target_mm or TARGET_MM
    zoom = [float(s) / target_mm for s in spacing]
    small = ndimage.zoom(mask.astype(np.float32), zoom, order=1)
    if SMOOTH_SIGMA:
        small = ndimage.gaussian_filter(small, SMOOTH_SIGMA)
    if small.max() < 0.5:
        return None
    try:
        v, f, _, _ = marching_cubes(small, level=0.5,
                                    spacing=(target_mm,) * 3)
    except (ValueError, RuntimeError):
        return None
    return v.astype(np.float32), f.astype(np.uint32)


def b64(a, dtype):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


def _csv(path):
    import pandas as pd
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def stones_for(sid, spacing):
    """Every reported calculus as {centre_mm, diameter_mm, hu, organ, side}."""
    import pandas as pd
    per = os.path.join(CSV, "per_study")
    out = []

    def add(d, organ, filt=None):
        if not len(d) or "is_stone" not in d:
            return
        d = d[d.is_stone.astype(bool)]
        if filt:
            d = filt(d)
        for r in d.itertuples():
            cv = str(getattr(r, "centroid_vox", ""))
            try:
                c = [float(x) for x in cv.split(",")]
            except Exception:
                continue
            dia = float(getattr(r, "max_diameter_mm", float("nan")))
            if not np.isfinite(dia):
                continue
            out.append({
                "c": [c[i] * spacing[i] for i in range(3)],
                "d": round(dia, 2),
                "hu": int(getattr(r, "hu_max", 0) or 0),
                "organ": organ if organ != "auto"
                         else str(getattr(r, "compartment", "kidney")),
                "side": str(getattr(r, "side", "") or ""),
                "zone": str(getattr(r, "zone", "") or ""),
            })

    k = _csv(os.path.join(per, f"{sid}_candidates.csv"))
    if len(k) and "compartment" in k.columns:
        k = k[~k.compartment.astype(str).str.startswith("bladder")]
        # the same fill rule the detector now applies
        if "fill_fraction" in k.columns:
            k = k[k.fill_fraction.fillna(1.0) >= 0.05]
    add(k, "auto")
    u = _csv(os.path.join(per, f"{sid}_ureter_candidates.csv"))
    add(u, "ureter", lambda d: d[d.report_this.fillna(True).astype(bool)]
        if "report_this" in d.columns else d)
    b = _csv(os.path.join(per, f"{sid}_bladder_candidates.csv"))
    if len(b) and "fill_fraction" in b.columns:
        b = b[b.fill_fraction.fillna(1.0) >= 0.04]
    add(b, "bladder")
    return out


def build(sid, run, seg_dir=None, nifti=None):
    """Write one self-contained HTML scene. Returns its path, or None."""
    from calculus.report._scene_html import TEMPLATE
    nii = nifti or os.path.join(NIFTI, f"{sid}.nii.gz")
    seg = seg_dir or os.path.join(SEG, str(sid))
    if not os.path.exists(nii) or not os.path.isdir(seg):
        return None
    spacing = tuple(float(z) for z in nib.load(nii).header.get_zooms()[:3])

    surfaces, mins, maxs = [], [], []
    def push(mask, col, alpha, rim, group, target=None):
        r = surface(mask, spacing, target)
        if r is None:
            return
        v, f = r
        mins.append(v.min(axis=0)); maxs.append(v.max(axis=0))
        surfaces.append({"v": b64(v.ravel(), np.float32),
                         "f": b64(f.ravel(), np.uint32),
                         "col": [round(c, 3) for c in col],
                         "alpha": alpha, "rim": rim, "group": group})

    for _label, name, col, alpha, group in ORGANS:
        m = _mask(seg, name)
        if m is not None:
            push(m, col, alpha, 0.85, "organ",
                 TARGET_MM_BLADDER if name == "urinary_bladder" else TARGET_MM)
    # Bone as ONE merged surface: it is scenery, and eight separate meshes cost
    # eight draw calls to say the same thing.
    bone = None
    for name in BONES:
        m = _mask(seg, name)
        if m is None:
            continue
        bone = m if bone is None else (bone | m)
    if bone is not None:
        push(bone, (0.29, 0.33, 0.40), 0.10, 0.30, "bone", TARGET_MM_BONE)

    if not surfaces:
        return None
    lo = np.min(np.array(mins), axis=0); hi = np.max(np.array(maxs), axis=0)
    centre = ((lo + hi) / 2.0).tolist()
    radius = float(np.linalg.norm(hi - lo) / 2.0)

    stones = stones_for(sid, spacing)
    n_renal = sum(1 for s in stones if "kidney" in s["organ"] or "renal" in s["organ"])
    n_ur = sum(1 for s in stones if s["organ"] == "ureter")
    n_bl = sum(1 for s in stones if s["organ"] == "bladder")
    largest = max((s["d"] for s in stones), default=None)
    densest = max((s["hu"] for s in stones), default=None)

    def chip(v, label):
        return (f'<div class="chip"><b>{v}</b><span>{label}</span></div>')
    stats = "".join([
        chip(len(stones), "calculi"),
        chip(n_renal, "renal"), chip(n_ur, "ureteric"), chip(n_bl, "bladder"),
        chip(f"{largest:.1f} mm" if largest else "&mdash;", "largest"),
        chip(f"{densest} HU" if densest else "&mdash;", "max density"),
    ])
    verdict = "Abnormal &mdash; calculi detected" if stones else "No calculus detected"
    data = {"surfaces": surfaces, "stones": stones,
            "centre": [round(c, 2) for c in centre], "radius": round(radius, 2)}

    html = (TEMPLATE
            .replace("__TITLE__", f"Urinary Tract {sid}")
            .replace("__HEADING__", f"CT KUB &mdash; study {sid}")
            .replace("__SUBTITLE__", verdict +
                     f" &nbsp;·&nbsp; voxel "
                     f"{spacing[0]:.2f} × {spacing[1]:.2f} × {spacing[2]:.2f} mm")
            .replace("__STATS__", stats)
            .replace("__FOOT__",
                     "Kidney and bladder surfaces are the actual segmentation. "
                     "Bone is shown for orientation only. "
                     "<b>No ureter is drawn</b> &mdash; it is not visible on "
                     "non-contrast CT, so its course is estimated and not "
                     "accurate enough to display. Calculi are spheres at the "
                     "measured centroid, scaled to the measured diameter. "
                     "Research output, not a diagnostic device.")
            .replace("__DATA__", json.dumps(data, separators=(",", ":"))))

    out_dir = os.path.join(run, "overlays", "scene3d")
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, f"{sid}.html")
    with open(dest, "w") as fh:
        fh.write(html)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", nargs="+", required=True)
    ap.add_argument("--seg", default=None)
    ap.add_argument("--nifti", default=None)
    a = ap.parse_args()
    run = ensure()
    for sid in a.studies:
        seg = os.path.join(a.seg, str(sid)) if a.seg else None
        nii = os.path.join(a.nifti, f"{sid}.nii.gz") if a.nifti else None
        p = build(str(sid), run, seg, nii)
        if p:
            print(f"  {sid}: {p}  ({os.path.getsize(p)/1024:.0f} KB)")
        else:
            print(f"  {sid}: skipped (missing nifti or seg)")


if __name__ == "__main__":
    main()
