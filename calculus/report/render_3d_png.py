#!/usr/bin/env python
"""Static PNG views of the 3D scene, for anywhere WebGL cannot go.

The interactive scene is the better artefact, but it cannot be pasted into a PDF
report, attached to an email, or embedded in the JSON response. These are the
same surfaces and the same stones, rendered from fixed viewpoints.

Deliberately coarse meshes: matplotlib's 3D backend is a painter's-algorithm
renderer, not a depth-buffered one, so triangle count costs time quadratically
and buys nothing at these image sizes. 5 mm organs and 9 mm bone look identical
at 1100 px wide.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection           # noqa: E402
import nibabel as nib                                            # noqa: E402
import numpy as np                                               # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from calculus.common.paths import NIFTI, SEG, ensure              # noqa: E402
from calculus.report.render_3d import (_mask, surface, stones_for,  # noqa: E402
                                       BONES)

BG = "#0b0f16"
INK = "#e8eef7"
DIM = "#93a3b8"
# anterior first: it is the projection a KUB reader already has in their head
VIEWS = [("anterior", 90, -90), ("oblique", 70, -55),
         ("lateral", 0, -90), ("posterior", -90, -90)]


def hu_colour(hu):
    t = max(0.0, min(1.0, (hu - 200) / 1300.0))
    stops = np.array([[1, .83, .47], [1, .62, .30], [1, .37, .32], [.78, .18, .43]])
    x = t * (len(stops) - 1)
    i = min(len(stops) - 2, int(x))
    return tuple(stops[i] + (stops[i + 1] - stops[i]) * (x - i))


def add(ax, mask, spacing, colour, alpha, target, lw=0.0):
    r = surface(mask, spacing, target)
    if r is None:
        return None
    v, f = r
    tri = Poly3DCollection(v[f], alpha=alpha, linewidths=lw)
    tri.set_facecolor(colour)
    tri.set_edgecolor("none")
    ax.add_collection3d(tri)
    return v


def ball(ax, c, d, colour, seg=9):
    u = np.linspace(0, 2 * np.pi, seg * 2)
    w = np.linspace(0, np.pi, seg)
    r = max(d / 2.0, 1.4)
    x = c[0] + r * np.outer(np.cos(u), np.sin(w))
    y = c[1] + r * np.outer(np.sin(u), np.sin(w))
    z = c[2] + r * np.outer(np.ones_like(u), np.cos(w))
    ax.plot_surface(x, y, z, color=colour, shade=True, linewidth=0,
                    antialiased=True)


def build(sid, out_dir, seg_dir=None, nifti=None):
    nii = nifti or os.path.join(NIFTI, f"{sid}.nii.gz")
    seg = seg_dir or os.path.join(SEG, str(sid))
    if not (os.path.exists(nii) and os.path.isdir(seg)):
        return []
    spacing = tuple(float(z) for z in nib.load(nii).header.get_zooms()[:3])
    stones = stones_for(sid, spacing)
    os.makedirs(out_dir, exist_ok=True)
    written = []

    for name, elev, azim in VIEWS:
        fig = plt.figure(figsize=(7.4, 8.6), facecolor=BG)
        ax = fig.add_subplot(111, projection="3d", facecolor=BG)
        pts = []
        for organ, colour, alpha, tgt in (
                ("kidney_left",  "#4f92e8", 0.20, 4.0),
                ("kidney_right", "#4f92e8", 0.20, 4.0),
                ("urinary_bladder", "#e8b93c", 0.16, 5.0)):
            m = _mask(seg, organ)
            if m is not None:
                v = add(ax, m, spacing, colour, alpha, tgt)
                if v is not None:
                    pts.append(v)
        bone = None
        for b in BONES:
            m = _mask(seg, b)
            if m is not None:
                bone = m if bone is None else (bone | m)
        if bone is not None:
            v = add(ax, bone, spacing, "#39404d", 0.055, 9.0)
            if v is not None:
                pts.append(v)
        for st in stones:
            ball(ax, st["c"], st["d"], hu_colour(st["hu"]))

        if not pts:
            plt.close(fig)
            continue
        P = np.vstack(pts)
        lo, hi = P.min(axis=0), P.max(axis=0)
        mid, half = (lo + hi) / 2, (hi - lo).max() / 2 * 1.05
        ax.set_xlim(mid[0] - half, mid[0] + half)
        ax.set_ylim(mid[1] - half, mid[1] + half)
        ax.set_zlim(mid[2] - half, mid[2] + half)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        n_r = sum(1 for s in stones if "kidney" in s["organ"] or "renal" in s["organ"])
        n_u = sum(1 for s in stones if s["organ"] == "ureter")
        n_b = sum(1 for s in stones if s["organ"] == "bladder")
        big = max((s["d"] for s in stones), default=0)
        fig.text(.06, .965, f"CT KUB  ·  study {sid}", color=INK,
                 fontsize=12, weight="semibold")
        fig.text(.06, .942,
                 f"{len(stones)} calculi   renal {n_r}   ureteric {n_u}   "
                 f"bladder {n_b}   largest {big:.1f} mm",
                 color=DIM, fontsize=9)
        fig.text(.94, .965, name.upper(), color=DIM, fontsize=9,
                 ha="right", weight="semibold")
        fig.text(.06, .022,
                 "Kidney and bladder from the segmentation. Bone for orientation. "
                 "No ureter drawn — not visible on non-contrast CT.",
                 color=DIM, fontsize=7.5)
        dest = os.path.join(out_dir, f"{sid}_{name}.png")
        fig.savefig(dest, dpi=150, facecolor=BG, bbox_inches="tight",
                    pad_inches=0.12)
        plt.close(fig)
        written.append(dest)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seg", default=None)
    ap.add_argument("--nifti", default=None)
    a = ap.parse_args()
    for sid in a.studies:
        seg = os.path.join(a.seg, str(sid)) if a.seg else None
        nii = os.path.join(a.nifti, f"{sid}.nii.gz") if a.nifti else None
        d = os.path.join(a.out, str(sid))
        got = build(str(sid), d, seg, nii)
        for p in got:
            print(f"  {os.path.basename(p):<44}{os.path.getsize(p)/1024:>7.0f} KB")
        if not got:
            print(f"  {sid}: skipped (missing nifti or seg)")


if __name__ == "__main__":
    main()
