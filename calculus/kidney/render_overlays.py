"""Render visual overlays so detections can actually be reviewed by eye.

Produces, per study, into overlays/<study_id>/:

    _coronal_mip.png   coronal maximum-intensity projection of the whole scan
                       with kidney/bladder outlines and every detection marked.
                       This is the one image that shows the whole stone burden
                       at a glance, and it is how a radiologist will sanity
                       check the run.
    _contact_sheet.png every detection as a small axial crop, numbered
    stone_NN.png       one axial view per stone: zoomed crop + wider context,
                       with the measured values printed on it

Orientation follows the array layout written by extract_series.py:
    axis0 -> patient LEFT, axis1 -> POSTERIOR, axis2 -> SUPERIOR
so axial views are displayed radiological (patient right on image left,
anterior at top) and coronal views are head-up.

Bone-ish window (level 400, width 1800) is used throughout -- calculi are far
easier to judge on a wide window than on soft tissue settings.

Usage:
    ./venv/bin/python render_overlays.py
    ./venv/bin/python render_overlays.py --study 8193874
    ./venv/bin/python render_overlays.py --max-stones 20
"""
import argparse
import ast
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import nibabel as nib                     # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402

# this file lives in utils/, so the project root is one level up. All data
# directories hang off ROOT, never off the script's own folder.
HERE = os.path.dirname(os.path.abspath(__file__))
# package root -> project root: calculus/<sub>/x.py is two levels down
ROOT = os.path.dirname(os.path.dirname(HERE))
from calculus.common.paths import CSV, NIFTI, SEG     # noqa: E402  results dir is per-run
from calculus.common.paths import OVERLAYS as OUT     # noqa: E402

WL, WW = 400, 1800          # window level / width -- wide, for calculi
CROP_MM = 40                # half-size of the zoomed axial crop
CONTEXT_MM = 110


def win(a):
    return np.clip((a - (WL - WW / 2)) / WW, 0, 1)


def load_mask(sid, name):
    p = os.path.join(SEG, sid, f"{name}.nii.gz")
    if not os.path.exists(p):
        return None
    m = np.asanyarray(nib.load(p).dataobj) > 0
    return m if m.any() else None


def outline(ax, mask2d, colour, lw=0.8):
    if mask2d is not None and mask2d.any():
        ax.contour(mask2d.T, levels=[0.5], colors=[colour], linewidths=lw)


def coronal_mip(sid, vol, spacing, stones, masks, path):
    """Whole-scan coronal MIP with every detection marked."""
    mip = vol.max(axis=1)                       # (x, z)
    fig, ax = plt.subplots(figsize=(7, 11))
    ax.imshow(win(mip).T, cmap="gray", origin="lower", aspect=spacing[2] / spacing[0])
    for name, colour in (("kidney_left", "#4da3ff"), ("kidney_right", "#4da3ff"),
                         ("urinary_bladder", "#ffd24d")):
        m = masks.get(name)
        if m is not None:
            proj = m.any(axis=1)
            ax.contour(proj.T, levels=[0.5], colors=[colour], linewidths=1.0,
                       alpha=.9)
    for r in stones.itertuples():
        x, y, z = r.centroid
        ax.plot(x, z, "o", mfc="none", mec="#ff3b3b", mew=1.6, ms=13)
        ax.text(x + 7, z, f"{r.stone_id}", color="#ff3b3b", fontsize=8,
                va="center")
    ax.set_title(f"{sid} - coronal MIP - {len(stones)} detections\n"
                 "blue = kidneys, yellow = bladder, red = detection",
                 fontsize=10)
    ax.set_xlabel("patient left  →")
    ax.set_ylabel("↑ head")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def stone_figure(sid, vol, spacing, r, masks, path):
    """Zoomed axial crop + wider context for one detection."""
    x, y, z = r.centroid
    z = int(np.clip(z, 0, vol.shape[2] - 1))
    sl = vol[:, :, z]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))

    for ax, half_mm, tag in ((axes[0], CROP_MM, "zoom"),
                             (axes[1], CONTEXT_MM, "context")):
        hx = int(half_mm / spacing[0])
        hy = int(half_mm / spacing[1])
        x0, x1 = max(0, x - hx), min(vol.shape[0], x + hx)
        y0, y1 = max(0, y - hy), min(vol.shape[1], y + hy)
        sub = sl[x0:x1, y0:y1]
        ax.imshow(win(sub).T, cmap="gray", origin="upper")
        if tag == "context":
            for name, colour in (("kidney_left", "#4da3ff"),
                                 ("kidney_right", "#4da3ff"),
                                 ("urinary_bladder", "#ffd24d"),
                                 ("aorta", "#7CFC00"),
                                 ("iliac_artery_left", "#7CFC00"),
                                 ("iliac_artery_right", "#7CFC00")):
                m = masks.get(name)
                if m is not None:
                    outline(ax, m[x0:x1, y0:y1, z], colour)
        ax.plot(x - x0, y - y0, "o", mfc="none", mec="#ff3b3b", mew=1.8, ms=22)
        ax.set_title(f"{tag}  (±{half_mm} mm)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    loc = f"{r.side} {r.compartment}"
    if isinstance(r.location, str) and r.location:
        loc += f" / {r.location}"
    fig.suptitle(
        f"{sid}  stone {r.stone_id}   {loc}\n"
        f"{r.max_diameter_mm:.1f} mm   {r.volume_mm3:.0f} mm³   "
        f"HU max {r.hu_max:.0f}, mean {r.hu_mean:.0f}   (slice z={z})",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .90))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def kidney_axials(sid, vol, spacing, stones, masks, path, n=8):
    """Axial slices through the kidneys with the segmentation drawn on.

    This is the QC image for the two steps the whole pipeline rests on:
    did TotalSegmentator find the kidneys, and did detection stay inside them?
    """
    kid = None
    for k in ("kidney_left", "kidney_right"):
        m = masks.get(k)
        if m is not None:
            kid = m if kid is None else (kid | m)
    if kid is None:
        return False
    zs = np.where(kid.any(axis=(0, 1)))[0]
    if len(zs) == 0:
        return False
    picks = np.linspace(zs[0], zs[-1], n).astype(int)

    # crop tight around the kidneys so they are actually visible
    xs = np.where(kid.any(axis=(1, 2)))[0]
    ys = np.where(kid.any(axis=(0, 2)))[0]
    mx, my = int(40 / spacing[0]), int(40 / spacing[1])
    x0, x1 = max(0, xs[0] - mx), min(vol.shape[0], xs[-1] + mx)
    y0, y1 = max(0, ys[0] - my), min(vol.shape[1], ys[-1] + my)

    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.4 * nrow),
                             squeeze=False)
    for i, z in enumerate(picks):
        ax = axes[i // ncol][i % ncol]
        ax.imshow(win(vol[x0:x1, y0:y1, z]).T, cmap="gray", origin="upper")
        for k, colour in (("kidney_left", "#4da3ff"), ("kidney_right", "#00e5a0")):
            m = masks.get(k)
            if m is not None:
                outline(ax, m[x0:x1, y0:y1, z], colour, lw=1.1)
        hit = 0
        for r in stones.itertuples():
            sx, sy, sz = r.centroid
            if abs(sz - z) <= max(1, int(2.0 / spacing[2])):
                ax.plot(sx - x0, sy - y0, "o", mfc="none", mec="#ff3b3b",
                        mew=1.6, ms=18)
                hit += 1
        ax.set_title(f"axial slice {z}" + (f"  ({hit} stone)" if hit else ""),
                     fontsize=8)
        ax.axis("off")
    for j in range(len(picks), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"{sid} - kidney segmentation on axial slices\n"
                 "blue = left kidney, green = right kidney, red = detection",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def contact_sheet(sid, vol, spacing, stones, path, ncol=5):
    n = len(stones)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 2.7 * nrow),
                             squeeze=False)
    half = int(30 / spacing[0])
    for i, r in enumerate(stones.itertuples()):
        ax = axes[i // ncol][i % ncol]
        x, y, z = r.centroid
        z = int(np.clip(z, 0, vol.shape[2] - 1))
        x0, x1 = max(0, x - half), min(vol.shape[0], x + half)
        y0, y1 = max(0, y - half), min(vol.shape[1], y + half)
        ax.imshow(win(vol[x0:x1, y0:y1, z]).T, cmap="gray", origin="upper")
        ax.plot(x - x0, y - y0, "o", mfc="none", mec="#ff3b3b", mew=1.4, ms=16)
        ax.set_title(f"#{r.stone_id} {r.max_diameter_mm:.1f}mm "
                     f"{r.hu_max:.0f}HU", fontsize=7)
        ax.axis("off")
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"{sid} - all {n} detections", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default=None)
    ap.add_argument("--max-stones", type=int, default=15,
                    help="per-stone figures to render per study")
    args = ap.parse_args()

    stones = pd.read_csv(os.path.join(CSV, "baseline_stones.csv"))
    stones["study_id"] = stones["study_id"].astype(str)
    stones["centroid"] = stones["centroid_vox"].apply(
        lambda s: [int(v) for v in ast.literal_eval(s)])

    # every study with a volume gets rendered, including ones with no
    # detections -- an empty kidney MIP is itself a result worth seeing
    all_ids = sorted(os.path.splitext(os.path.basename(p))[0].replace(".nii", "")
                     for p in glob.glob(os.path.join(NIFTI, "*.nii.gz")))
    ids = [args.study] if args.study else all_ids
    os.makedirs(OUT, exist_ok=True)
    print(f"rendering overlays for {len(ids)} studies -> {OUT}\n")

    for i, sid in enumerate(ids, 1):
        s = stones[stones.study_id == sid].reset_index(drop=True)
        vpath = os.path.join(NIFTI, f"{sid}.nii.gz")
        if not os.path.exists(vpath):
            print(f"[{i}/{len(ids)}] {sid} skipped (no volume)")
            continue
        nii = nib.load(vpath)
        vol = np.asanyarray(nii.dataobj).astype(np.float32)
        spacing = tuple(float(v) for v in nii.header.get_zooms()[:3])
        masks = {n: load_mask(sid, n) for n in
                 ("kidney_left", "kidney_right", "urinary_bladder", "aorta",
                  "iliac_artery_left", "iliac_artery_right")}

        d = os.path.join(OUT, sid)
        os.makedirs(d, exist_ok=True)
        # Clear stale per-stone images. A previous run that found more stones
        # would otherwise leave orphan stone_NN.png files behind, and those
        # look exactly like current results to anyone browsing the folder.
        for old in glob.glob(os.path.join(d, "stone_*.png")):
            os.remove(old)
        coronal_mip(sid, vol, spacing, s, masks,
                    os.path.join(d, "_coronal_mip.png"))
        kidney_axials(sid, vol, spacing, s, masks,
                      os.path.join(d, "_kidney_axials.png"))
        if not s.empty:
            contact_sheet(sid, vol, spacing, s,
                          os.path.join(d, "_contact_sheet.png"))
        for r in s.head(args.max_stones).itertuples():
            stone_figure(sid, vol, spacing, r, masks,
                         os.path.join(d, f"stone_{int(r.stone_id):02d}.png"
                             if pd.notna(r.stone_id) else
                             f"stone_{i:02d}.png"))
        print(f"[{i}/{len(ids)}] {sid}: {len(s)} detections, "
              f"{min(len(s), args.max_stones) + 2} images", flush=True)

    print(f"\nopen {OUT}/<study_id>/_coronal_mip.png first")


if __name__ == "__main__":
    main()
