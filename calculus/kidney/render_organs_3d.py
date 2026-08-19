"""3D surface renders of the KIDNEYS and BLADDER together, per study.

WHY THIS EXISTS SEPARATELY FROM render_kidney_3d.py
---------------------------------------------------
`render_kidney_3d.py` is a QC tool for ONE question: is the kidney mask the
right shape. It frames tightly on the kidneys and deliberately shows nothing
else, because anything else in frame makes the kidney smaller and the answer
harder to see.

This script answers a different question -- are the ORGAN segmentations right,
in relation to each other -- both ends of the ureteric corridor in one frame.
That means the kidneys render smaller than in the tight view. Both are useful;
neither replaces the other, so they are kept apart rather than merged behind a
flag.

WHAT IS DRAWN
    kidneys   red        opaque, shaded
    bladder   green      the UVJ reference for Part 2 sits on its wall
    calculi   cyan       real voxels recovered from the CT, not markers

Nothing else. Liver and spleen were tried and removed on request -- they are not
part of the stone pipeline's 14-class mask set, so drawing them meant a separate
TotalSegmentator pass, and a 1664 mL liver in frame shrinks the kidneys to the
point where their shape is hard to judge.

WHY BOTH TOGETHER
The kidneys and the bladder are the two ends of the ureteric corridor that Part 2
searches. Seeing them in one frame at a common scale shows the span the corridor
has to cover, and whether the bladder -- which carries the UVJ landmark every
distance is measured from -- is segmented sensibly.

OUTPUT per study, in 3D_organ/<study_id>/:
    views.png    two viewpoints, anterior and anterior-oblique
    organs.stl   every organ merged, for rotating in a real viewer
    (STL carries geometry only, no colour -- the colour lives in the PNG)

Usage:
    ./venv/bin/python utils/render_organs_3d.py
    ./venv/bin/python utils/render_organs_3d.py --studies 8563509
"""
import argparse
import glob
import os
import struct
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
import nibabel as nib                        # noqa: E402
import numpy as np                           # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from scipy import ndimage                    # noqa: E402
from skimage.measure import marching_cubes   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# package root -> project root: calculus/<sub>/x.py is two levels down
ROOT = os.path.dirname(os.path.dirname(HERE))
from calculus.common.paths import NIFTI, SEG                 # noqa: E402

OUT = os.path.join(ROOT, "3D_organ")

GROW_HU = 130.0            # same as detect_stones: outer extent of a stone
STONE_BOX_MM = 30.0

# Decimation. At step 1 a kidney yields ~200k triangles and matplotlib's software
# renderer takes minutes per figure. Step 2 cuts that ~8x and costs nothing
# visible on a smooth ~10 cm organ. Stones stay at step 1: a 2 mm calculus has
# few voxels to spare.
STEP = {"kidneys": 2, "bladder": 2, "stones": 1}

# name -> (base colour, alpha), in DRAW ORDER. The bladder is kept slightly
# translucent so a stone sitting at the vesico-ureteric junction -- on its wall,
# which is exactly where many ureteric calculi lodge -- is not hidden inside it.
ORGANS = [
    ("kidneys", (0.74, 0.10, 0.10), 0.95),
    ("bladder", (0.16, 0.62, 0.28), 0.55),
    ("stones",  (0.10, 0.85, 0.85), 1.00),
]

VIEWS = [("anterior", 8, -90), ("anterior-oblique", 16, -60)]
LIGHT = np.array([0.35, -0.80, 0.48])


def write_stl(path, verts, faces):
    """Minimal binary STL -- 15 lines of struct beats adding a dependency."""
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 0)
    with open(path, "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(tri)))
        for i in range(len(tri)):
            f.write(struct.pack("<12fH", *n[i], *tri[i, 0], *tri[i, 1],
                                *tri[i, 2], 0))
# WHAT THIS FUNCTION DOES: writes a triangle mesh to a binary STL so the surfaces
# can be rotated in any 3D viewer instead of only seen from the fixed angles in
# the PNG.


def shade(verts, faces, rgb, light=LIGHT):
    """Per-face Lambertian shading -> one colour per triangle.

    Without it every organ is a flat silhouette, which defeats the purpose:
    surface relief is how you judge whether a mask has the right shape.
    """
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 0)
    l = light / np.linalg.norm(light)
    # abs(): marching-cubes winding is not guaranteed outward-consistent, so a
    # signed dot product would render random facets black
    f = 0.42 + 0.58 * np.abs(n @ l)
    return np.clip(np.asarray(rgb)[None, :] * f[:, None], 0, 1)
# WHAT THIS FUNCTION DOES: works out how brightly each triangle is lit so the
# rendered organ shows its real surface shape rather than a flat blob.


def surface(mask, spacing, step=1, smooth=0.8):
    """Marching-cubes surface of a binary mask, in mm coordinates."""
    if mask is None or not mask.any():
        return None, None
    m = mask.astype(np.float32)
    if smooth > 0:
        m = ndimage.gaussian_filter(m, smooth)
    try:
        v, f, _, _ = marching_cubes(m, level=0.5, spacing=spacing, step_size=step)
    except (RuntimeError, ValueError):
        return None, None
    return v, f
# WHAT THIS FUNCTION DOES: turns a filled binary mask into a lightly smoothed
# triangle mesh of its outer surface, so an organ looks like an organ rather than
# a stack of cubes.


def draw_box(ax, lo, hi):
    """The translucent bounding cube: fixes scale and orientation for the eye."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    faces = [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
    ]
    pc = Poly3DCollection(faces, alpha=0.09, linewidths=0.6)
    pc.set_facecolor("#aecfe8")
    pc.set_edgecolor("#5b7fa6")
    ax.add_collection3d(pc)
# WHAT THIS FUNCTION DOES: draws a faint box around the organs so scale and
# orientation are readable at a glance from any viewing angle.


def stone_voxels(vol, centroid, spacing, shape):
    """Recover a stone's real voxel set from its stored centroid.

    Deliberately not a sphere: a staghorn drawn as a ball would misrepresent
    exactly the shape a urologist reads the render for.
    """
    half = [max(2, int(round(STONE_BOX_MM / s))) for s in spacing]
    lo = [max(0, int(c) - h) for c, h in zip(centroid, half)]
    hi = [min(n, int(c) + h + 1) for c, h, n in zip(centroid, half, shape)]
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    sub = vol[sl] >= GROW_HU
    if not sub.any():
        return None, None
    lab, _ = ndimage.label(sub)
    local = tuple(int(c) - a for c, a in zip(centroid, lo))
    want = lab[local]
    if not want:                      # centroid landed just off the object
        idx = np.argwhere(lab > 0)
        if not len(idx):
            return None, None
        near = idx[np.argmin(np.linalg.norm((idx - np.array(local)) * spacing, axis=1))]
        want = lab[tuple(near)]
    return (lab == want), sl
# WHAT THIS FUNCTION DOES: given a recorded stone centroid, goes back to the
# original CT and returns the actual blob of stone voxels, so the shape shown is
# the measured object rather than a stand-in.


def render(sid, stones_df, dpi=104):
    nii = os.path.join(NIFTI, f"{sid}.nii.gz")
    d = os.path.join(SEG, sid)
    if not os.path.exists(nii) or not os.path.isdir(d):
        return f"{sid}: no nifti or no seg"
    img = nib.load(nii)
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    vx = float(np.prod(spacing))

    def load(name):
        p = os.path.join(d, f"{name}.nii.gz")
        return (np.asanyarray(nib.load(p).dataobj) > 0) if os.path.exists(p) else None

    kl, kr = load("kidney_left"), load("kidney_right")
    kid = np.zeros(img.shape, bool)
    for k in (kl, kr):
        if k is not None:
            kid |= k
    masks = {"kidneys": kid if kid.any() else None,
             "bladder": load("urinary_bladder")}

    # stones: real voxels from the CT
    rows = (stones_df[stones_df.study_id.astype(str) == sid]
            if stones_df is not None else None)
    stone = np.zeros(img.shape, bool)
    n_stone = 0
    if rows is not None and len(rows):
        vol = np.asanyarray(img.dataobj).astype(np.float32)
        for _, r in rows.iterrows():
            try:
                cen = [int(float(v)) for v in
                       str(r.centroid_vox).strip("[]() ").split(",")]
            except (ValueError, TypeError):
                continue
            if len(cen) != 3:
                continue
            m, sl = stone_voxels(vol, cen, spacing, img.shape)
            if m is not None:
                stone[sl] |= m
                n_stone += 1
        del vol
    masks["stones"] = stone if stone.any() else None

    meshes = {}
    for name, _, _ in ORGANS:
        v, f = surface(masks.get(name), spacing, STEP.get(name, 2))
        if v is not None:
            meshes[name] = (v, f)
    if "kidneys" not in meshes:
        return f"{sid}: no kidney mask"

    allv = np.vstack([m[0] for m in meshes.values()])
    lo, hi = allv.min(axis=0) - 5.0, allv.max(axis=0) + 5.0
    ctr = (lo + hi) / 2.0
    rad = float((hi - lo).max()) / 2.0 * 1.03

    # Sanity checks that need only the two masks we have. A kidney overlapping
    # the bladder is anatomically impossible and means one of the two has leaked;
    # a bladder above the kidneys means the volume is stored in an orientation
    # this render assumes it is not.
    leaks = []
    k, b = masks.get("kidneys"), masks.get("bladder")
    if k is not None and b is not None:
        ov = float((k & b).sum()) * vx / 1000.0
        if ov > 0.5:
            leaks.append(f"kidney/bladder overlap {ov:.1f} mL - impossible")
        kz = np.where(k.any(axis=(0, 1)))[0]
        bz = np.where(b.any(axis=(0, 1)))[0]
        if kz.size and bz.size and bz.mean() > kz.mean():
            leaks.append("bladder ABOVE kidneys - check orientation")
    if b is None:
        leaks.append("no bladder mask - Part 2 has no UVJ reference")

    fig = plt.figure(figsize=(14, 8))
    fig.subplots_adjust(left=0, right=1, top=0.90, bottom=0.02, wspace=-0.05)
    for i, (title, elev, azim) in enumerate(VIEWS, 1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        draw_box(ax, lo, hi)
        for name, rgb, alpha in ORGANS:       # draw order = ORGANS order
            if name not in meshes:
                continue
            v, f = meshes[name]
            pc = Poly3DCollection(v[f], alpha=alpha, linewidths=0)
            pc.set_facecolor(shade(v, f, rgb))
            ax.add_collection3d(pc)
        ax.set_xlim(ctr[0] - rad, ctr[0] + rad)
        ax.set_ylim(ctr[1] - rad, ctr[1] + rad)
        ax.set_zlim(ctr[2] - rad, ctr[2] + rad)
        ax.view_init(elev=elev, azim=azim)
        try:
            ax.set_box_aspect((1, 1, 1), zoom=1.42)
        except TypeError:                     # matplotlib < 3.4
            ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        ax.set_title(title, fontsize=10, y=0.98)

    os.makedirs(os.path.join(OUT, sid), exist_ok=True)
    # one merged STL: face indices must be offset as meshes are concatenated
    vs, fs, off = [], [], 0
    for name in ("kidneys", "bladder", "stones"):
        if name in meshes:
            v, f = meshes[name]
            vs.append(v)
            fs.append(f + off)
            off += len(v)
    if vs:
        write_stl(os.path.join(OUT, sid, "organs.stl"),
                  np.vstack(vs), np.vstack(fs))

    ml = {n: (masks[n].sum() * vx / 1000.0 if masks.get(n) is not None else None)
          for n in ("kidneys", "bladder")}
    head = f"{sid}    " + "    ".join(
        f"{n} {v:.0f} mL" for n, v in ml.items() if v is not None)
    if n_stone:
        head += f"    {n_stone} calculi"
    sub = "red = kidneys    green = bladder" + \
          ("    cyan = calculi" if n_stone else "")
    fig.suptitle(head + "\n" + sub, fontsize=11.5)
    if leaks:
        fig.text(0.5, 0.015, "CHECK: " + "   |   ".join(leaks), ha="center",
                 fontsize=11.5, color="#c0392b", weight="bold")
    fig.savefig(os.path.join(OUT, sid, "views.png"), dpi=dpi)
    plt.close(fig)
    return (f"{sid}: " + " ".join(f"{n}={v:.0f}" for n, v in ml.items() if v is not None)
            + (f"  <-- {'; '.join(leaks)}" if leaks else ""))
# WHAT THIS FUNCTION DOES: builds surfaces for the kidneys and bladder of one
# study, draws them together from two angles at a common scale, writes a merged
# STL, and flags anything anatomically impossible -- the two masks overlapping, the
# bladder sitting above the kidneys, or no bladder at all.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", nargs="*", default=None)
    ap.add_argument("--stones-csv",
                    default=os.path.join(ROOT, "run_v5", "csv", "baseline_stones.csv"))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    stones = None
    if os.path.exists(a.stones_csv):
        import pandas as pd
        stones = pd.read_csv(a.stones_csv)
        print(f"stones from {a.stones_csv}: {len(stones)} accepted")

    ids = a.studies or sorted(os.path.basename(f).split(".")[0]
                              for f in glob.glob(os.path.join(NIFTI, "*.nii.gz")))
    os.makedirs(OUT, exist_ok=True)
    if not a.overwrite:
        skip = {s for s in ids if os.path.exists(os.path.join(OUT, s, "views.png"))}
        if skip:
            print(f"skipping {len(skip)} already rendered")
            ids = [s for s in ids if s not in skip]

    for i, sid in enumerate(ids, 1):
        try:
            msg = render(sid, stones)
        except Exception as e:
            msg = f"{sid}: FAILED {type(e).__name__}: {e}"
        print(f"[{i}/{len(ids)}] {msg}", flush=True)
    print(f"\nwrote {OUT}/<study_id>/views.png + organs.stl")


if __name__ == "__main__":
    main()
