"""3D surface renders of each kidney with its stones, one folder per study.

WHY
---
The 2D overlays answer "is this detection real". They do not answer "where is
this stone inside the kidney, and what does the collecting system look like
around it" -- questions a urologist asks before deciding on ESWL versus
ureteroscopy versus PCNL. A surface render answers those in one look.

WHAT IS RENDERED
----------------
    kidney parenchyma   translucent red    (TotalSegmentator mask)
    calculi             opaque yellow      (real voxels, see below)
    cysts               translucent blue   (TotalSegmentator, when present)

THE STONES ARE REAL GEOMETRY, NOT MARKERS
-----------------------------------------
`baseline_stones.csv` stores a centroid and a diameter, not a mask. Drawing a
sphere of the measured diameter at the centroid would look identical to a real
render and be a lie -- a staghorn would appear as a ball. So for each accepted
stone this script goes back to the ORIGINAL CT, takes a small box around the
stored centroid, thresholds at GROW_HU and keeps the connected component that
contains the centroid. That is the same voxel set the measurement was taken
from, so the render and the numbers agree by construction.

Stones REJECTED by the detector are not drawn. The render shows the clinical
answer, and `candidates.csv` remains the audit trail.

OUTPUT per study, in 3d_kidneys/<study_id>/:
    views.png          four viewpoints in one figure, labelled with the counts
    kidneys.stl        the parenchyma surface, for a real 3D viewer
    stones.stl         the calculi surface (absent when there are none)

The STLs matter as much as the PNGs: a still image cannot be rotated, and
"is that stone in the lower pole or behind it" is a question that needs
rotation. Any 3D viewer opens an STL.

Usage:
    ./venv/bin/python utils/render_kidney_3d.py                  # all studies
    ./venv/bin/python utils/render_kidney_3d.py --studies 8513308
    ./venv/bin/python utils/render_kidney_3d.py --stones-csv run_v5/csv/baseline_stones.csv
"""
import argparse
import glob
import os
import struct
import sys

import matplotlib
matplotlib.use("Agg")                       # headless: no display on this box
import matplotlib.pyplot as plt             # noqa: E402
import numpy as np                          # noqa: E402
import nibabel as nib                       # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection   # noqa: E402
from scipy import ndimage                   # noqa: E402
from skimage.measure import marching_cubes  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# package root -> project root: calculus/<sub>/x.py is two levels down
ROOT = os.path.dirname(os.path.dirname(HERE))
from calculus.common.paths import NIFTI, SEG                # noqa: E402

OUT = os.path.join(ROOT, "3d_kidneys")

GROW_HU = 130.0          # same as detect_stones: the outer extent of a stone
STONE_BOX_MM = 30.0      # half-width of the box searched around a centroid

# Decimation. marching_cubes on a full 512x512x500 mask yields ~400k triangles,
# which matplotlib's software renderer takes minutes to draw. step_size=2
# samples every other voxel, cutting triangles ~8x. The kidney is a smooth
# ~10 cm organ, so this costs nothing visible; stones are rendered at step 1
# because a 2 mm stone has few voxels to spare.
KIDNEY_STEP = 2
STONE_STEP = 1

# Single anterior-oblique viewpoint, matching the reference figure. Slightly
# off-axis rather than dead anterior, because a pure anterior view flattens the
# kidney into a silhouette and loses the hilar concavity.
VIEW = ("anterior-oblique", 14, -78)

# Solid organ colours. Shaded per-face (see `shade_faces`), so these are the
# BASE colour at full illumination -- a flat fill looks like a paper cut-out.
KIDNEY_RGB = (0.72, 0.09, 0.09)     # deep red, as in the reference
STONE_RGB = (0.10, 0.85, 0.85)      # cyan: the one colour that survives being
                                    # seen through a red surface
CYST_RGB = (0.25, 0.55, 0.90)

# Direction the light comes from, in (x, y, z) voxel-axis order. Chosen so the
# lit side faces the camera at the VIEW angle above.
LIGHT = np.array([0.35, -0.80, 0.48])


def write_stl(path, verts, faces):
    """Minimal binary STL. Avoids adding a dependency for 15 lines of struct."""
    tri = verts[faces]                                  # (n, 3, 3)
    # facet normal, by the right-hand rule over each triangle's two edges
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 0)
    with open(path, "wb") as f:
        f.write(b"\0" * 80)                             # 80-byte header
        f.write(struct.pack("<I", len(tri)))            # triangle count
        for i in range(len(tri)):
            f.write(struct.pack("<12fH", *n[i], *tri[i, 0], *tri[i, 1],
                                *tri[i, 2], 0))
# WHAT THIS FUNCTION DOES: writes a triangle mesh to a binary STL file so the
# surface can be opened and rotated in any 3D viewer, rather than only seen
# from the fixed angles in the PNG.


def surface(mask, spacing, step=1, smooth=0.8):
    """Marching-cubes surface of a binary mask, in mm coordinates.

    The mask is blurred slightly first and the iso-surface taken at 0.5. Run on
    a raw binary mask, marching cubes returns the blocky voxel staircase; a
    small blur turns that into the smooth organ surface the anatomy actually
    has, without moving the boundary (0.5 is the half-way level either way).
    """
    if not mask.any():
        return None, None
    m = mask.astype(np.float32)
    if smooth > 0:
        m = ndimage.gaussian_filter(m, smooth)
    try:
        v, f, _, _ = marching_cubes(m, level=0.5, spacing=spacing,
                                    step_size=step)
    except (RuntimeError, ValueError):
        return None, None                   # too few voxels to form a surface
    return v, f
# WHAT THIS FUNCTION DOES: converts a filled binary mask into a triangle mesh
# of its outer surface, lightly smoothed so a kidney looks like a kidney rather
# than a stack of cubes.


def stone_voxels(vol, centroid, spacing, shape):
    """Recover the real voxel set of one stone from its stored centroid.

    Deliberately NOT a sphere at the centroid. See the module docstring: the
    render has to show the same object the measurement was taken from, or it
    misrepresents shape -- which is precisely what a urologist reads it for.
    """
    half = [max(2, int(round(STONE_BOX_MM / s))) for s in spacing]
    lo = [max(0, int(c) - h) for c, h in zip(centroid, half)]
    hi = [min(n, int(c) + h + 1) for c, h, n in zip(centroid, half, shape)]
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    sub = vol[sl] >= GROW_HU
    if not sub.any():
        return None, None
    lab, n = ndimage.label(sub)
    local = tuple(int(c) - a for c, a in zip(centroid, lo))
    # the component the centroid sits in; if the centroid landed just off the
    # object (rounding), fall back to the nearest labelled voxel
    want = lab[local] if lab[local] else 0
    if not want:
        idx = np.argwhere(lab > 0)
        if not len(idx):
            return None, None
        near = idx[np.argmin(np.linalg.norm((idx - np.array(local)) * spacing,
                                            axis=1))]
        want = lab[tuple(near)]
    return (lab == want), sl
# WHAT THIS FUNCTION DOES: given the centroid recorded for an accepted stone, it
# goes back to the original CT, thresholds a small box around that point, and
# returns the actual connected blob of stone voxels -- so the 3D shape shown is
# the measured object and not a stand-in.


# Plausible parenchyma volume for ONE adult kidney. Outside this range the mask
# is the first thing to suspect, not the patient. QC on run_v4 found masks from
# 0.0 mL (nothing found) to 543 mL (leaked into liver or spleen), which is why
# this figure exists at all.
KIDNEY_ML_LO, KIDNEY_ML_HI = 60.0, 320.0


def shade_faces(verts, faces, rgb, light=LIGHT):
    """Per-face Lambertian shading -> an (n,3) colour array.

    Matplotlib's Poly3DCollection fills every triangle with one flat colour, so
    an unshaded organ renders as a featureless silhouette -- useless for judging
    whether a mask has the right SHAPE, which is the entire point of this view.
    Shading by the angle between each face normal and a fixed light restores the
    surface relief: bulges, notches and the hilar concavity all become visible.
    """
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 0)
    l = light / np.linalg.norm(light)
    # abs(): marching-cubes winding is not guaranteed outward-consistent, so a
    # signed dot product would render random facets black
    inten = np.abs(n @ l)
    f = 0.42 + 0.58 * inten                      # ambient floor + diffuse
    return np.clip(np.asarray(rgb)[None, :] * f[:, None], 0, 1)
# WHAT THIS FUNCTION DOES: works out how brightly each triangle should be lit
# and returns one colour per triangle, so the rendered organ shows its real
# surface shape instead of appearing as a flat coloured blob.


def draw_box(ax, lo, hi, colour="#aecfe8", alpha=0.10, edge="#5b7fa6"):
    """The translucent bounding cube, as in the paper's figure.

    It is not decoration. Without a box the eye has no reference for scale or
    orientation, so a kidney rendered from an oblique angle could be any size
    facing any way. The box fixes both -- its faces are the anatomical planes.
    """
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    # the six faces, each as four corners in order
    faces = [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],   # inferior
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],   # superior
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],   # anterior
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],   # posterior
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],   # right
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],   # left
    ]
    pc = Poly3DCollection(faces, alpha=alpha, linewidths=0.6)
    pc.set_facecolor(colour)
    pc.set_edgecolor(edge)
    ax.add_collection3d(pc)
# WHAT THIS FUNCTION DOES: draws a faint six-sided box around the rendered
# organs so the viewer can tell scale and orientation at a glance, the way the
# light blue cube does in the reference figure.


def render(sid, stones_df, dpi=105):
    """One study -> views.png + STL files."""
    nii = os.path.join(NIFTI, f"{sid}.nii.gz")
    d = os.path.join(SEG, sid)
    if not os.path.exists(nii) or not os.path.isdir(d):
        return f"{sid}: no nifti or no seg"
    img = nib.load(nii)
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])

    def load(name):
        p = os.path.join(d, f"{name}.nii.gz")
        return (np.asanyarray(nib.load(p).dataobj) > 0) if os.path.exists(p) \
            else None

    kl, kr = load("kidney_left"), load("kidney_right")
    kid = np.zeros(img.shape, bool)
    for k in (kl, kr):
        if k is not None:
            kid |= k
    if not kid.any():
        return f"{sid}: no kidney mask"
    cyst = np.zeros(img.shape, bool)
    for c in ("kidney_cyst_left", "kidney_cyst_right"):
        m = load(c)
        if m is not None:
            cyst |= m

    # stones: real voxels, recovered from the original CT
    rows = stones_df[stones_df.study_id.astype(str) == sid] \
        if stones_df is not None else None
    stone = np.zeros(img.shape, bool)
    n_stone = 0
    if rows is not None and len(rows):
        vol = np.asanyarray(img.dataobj).astype(np.float32)
        for _, r in rows.iterrows():
            # stored as "[188, 331, 203]" -- brackets included, so they have to
            # come off before int(). Parsing this with a bare try/except is how
            # the first version silently rendered every study as stone-free.
            try:
                cen = [int(float(v)) for v in
                       str(r.centroid_vox).strip("[]() ").split(",")]
            except (ValueError, TypeError) as e:
                print(f"    {sid}: unparseable centroid {r.centroid_vox!r} ({e})")
                continue
            if len(cen) != 3:
                print(f"    {sid}: centroid has {len(cen)} values, need 3")
                continue
            m, sl = stone_voxels(vol, cen, spacing, img.shape)
            if m is not None:
                stone[sl] |= m
                n_stone += 1
        del vol

    os.makedirs(os.path.join(OUT, sid), exist_ok=True)
    meshes = {}
    for name, mask, step in (("kidneys", kid, KIDNEY_STEP),
                             ("stones", stone, STONE_STEP),
                             ("cysts", cyst, KIDNEY_STEP)):
        v, f = surface(mask, spacing, step)
        if v is not None:
            meshes[name] = (v, f)
            if name in ("kidneys", "stones"):
                write_stl(os.path.join(OUT, sid, f"{name}.stl"), v, f)

    if "kidneys" not in meshes:
        return f"{sid}: kidney surface failed"

    # common axis limits so the four views are the same scale
    kv = meshes["kidneys"][0]
    all_v = np.vstack([m[0] for m in meshes.values()])
    box_lo = all_v.min(axis=0) - 4.0            # 4 mm of air around everything
    box_hi = all_v.max(axis=0) + 4.0
    ctr = (box_lo + box_hi) / 2.0
    rad = float((box_hi - box_lo).max()) / 2.0 * 1.04
    box_mm = box_hi - box_lo

    # ---- QC: per-side volume, and whether it is plausible ----------------
    vx = float(np.prod(spacing))
    ml = {s: (m.sum() * vx / 1000.0 if m is not None else 0.0)
          for s, m in (("L", kl), ("R", kr))}
    flags = []
    for s in ("L", "R"):
        if ml[s] == 0.0:
            flags.append(f"{s} MISSING")
        elif ml[s] < KIDNEY_ML_LO:
            flags.append(f"{s} SMALL {ml[s]:.0f} mL")
        elif ml[s] > KIDNEY_ML_HI:
            flags.append(f"{s} LARGE {ml[s]:.0f} mL - suspect leak")
    # a big left/right mismatch is the classic signature of a leak into liver
    # or spleen on one side only
    if ml["L"] > 0 and ml["R"] > 0:
        ratio = max(ml.values()) / min(ml.values())
        if ratio > 1.8:
            flags.append(f"ASYMMETRIC {ratio:.1f}x")

    title, elev, azim = VIEW
    fig = plt.figure(figsize=(8.6, 8.8))
    ax = fig.add_subplot(111, projection="3d")
    draw_box(ax, box_lo, box_hi)
    # kidney first, stones last: matplotlib paints in insertion order within a
    # depth bucket, so the stone must go on top or it vanishes into the organ
    for name, rgb, alpha in (("kidneys", KIDNEY_RGB, 0.93),
                             ("cysts", CYST_RGB, 0.55),
                             ("stones", STONE_RGB, 1.0)):
        if name not in meshes:
            continue
        v, f = meshes[name]
        pc = Poly3DCollection(v[f], alpha=alpha, linewidths=0)
        pc.set_facecolor(shade_faces(v, f, rgb))
        ax.add_collection3d(pc)
    for lim, c in ((ax.set_xlim, ctr[0]), (ax.set_ylim, ctr[1]),
                   (ax.set_zlim, ctr[2])):
        lim(c - rad, c + rad)
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_box_aspect((1, 1, 1), zoom=1.5)
    except TypeError:                   # matplotlib < 3.4 has no zoom
        ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()

    # dimension label rotated up the left edge, as in the reference figure
    fig.text(0.055, 0.5, f"{box_mm[2]:.0f} x {box_mm[1]:.0f} x {box_mm[0]:.0f} mm",
             rotation=90, va="center", ha="center", fontsize=11, color="#2c3e50")

    head = f"{sid}    L {ml['L']:.0f} mL    R {ml['R']:.0f} mL"
    if n_stone:
        head += f"    {n_stone} calculi, largest {rows.max_diameter_mm.max():.1f} mm"
    sub = ("red = kidney parenchyma" +
           ("    cyan = calculi" if n_stone else "") +
           ("    blue = cyst" if cyst.any() else ""))
    fig.suptitle(head + "\n" + sub, fontsize=12)
    if flags:
        # QC verdict in red across the bottom -- the whole reason for the figure
        fig.text(0.5, 0.035, "CHECK MASK:  " + "   |   ".join(flags),
                 ha="center", fontsize=12, color="#c0392b", weight="bold")
    fig.savefig(os.path.join(OUT, sid, "views.png"), dpi=dpi)
    plt.close(fig)
    return (f"{sid}: L {ml['L']:.0f} R {ml['R']:.0f} mL, {n_stone} calculi"
            + (f"  <-- {'; '.join(flags)}" if flags else ""))
# WHAT THIS FUNCTION DOES: builds the surfaces for one study, draws them from
# four angles into a single labelled figure, and writes the kidney and stone
# meshes as STL files so they can be rotated in a real viewer.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", nargs="*", default=None)
    ap.add_argument("--stones-csv",
                    default=os.path.join(ROOT, "run_v5", "csv",
                                         "baseline_stones.csv"))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    stones = None
    if os.path.exists(a.stones_csv):
        stones = pd_read(a.stones_csv)
        print(f"stones from {a.stones_csv}: {len(stones)} accepted")
    else:
        print(f"WARNING: {a.stones_csv} not found -- kidneys only, no calculi")

    ids = a.studies or sorted(os.path.basename(f).split(".")[0]
                              for f in glob.glob(os.path.join(NIFTI, "*.nii.gz")))
    os.makedirs(OUT, exist_ok=True)
    if not a.overwrite:
        skip = {s for s in ids
                if os.path.exists(os.path.join(OUT, s, "views.png"))}
        if skip:
            print(f"skipping {len(skip)} already rendered")
            ids = [s for s in ids if s not in skip]

    for i, sid in enumerate(ids, 1):
        try:
            msg = render(sid, stones)
        except Exception as e:
            msg = f"{sid}: FAILED {type(e).__name__}: {e}"
        print(f"[{i}/{len(ids)}] {msg}", flush=True)
    print(f"\nwrote {OUT}/<study_id>/views.png + kidneys.stl + stones.stl")


def pd_read(p):
    import pandas as pd
    return pd.read_csv(p)


if __name__ == "__main__":
    main()
