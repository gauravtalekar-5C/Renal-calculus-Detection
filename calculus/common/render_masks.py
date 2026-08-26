"""Organ masks drawn on the CT, one sheet per study — mask QC by eye.

The ureteric detector never sees the ureter. It infers where the ureter must be
from the organs around it, so if those masks are wrong the corridor is wrong and
every detection inside it is meaningless. This renders what the masks actually
look like, so that assumption can be checked rather than trusted.

WHAT IS DRAWN
    coronal MIP, whole scan   every mask projected, so a missing or leaked organ
                              is obvious at a glance
    three axial slices        at the kidneys, at the iliac crossing, and at the
                              bladder -- the three levels the corridor is built
                              from. A mask can look fine in projection and be
                              wrong on the slice that matters.

    kidney   red        bladder  yellow     iliac artery  green
    aorta    orange     IVC      cyan       bone          grey      cyst  blue

The title carries each organ's volume in mL, because the eye is poor at judging
whether a kidney is 90 mL or 40 mL and the number is not.

Usage:
    CALCULUS_RUN=ureteric_whole_stone_data \
    CALCULUS_NIFTI=ureteric_whole_stone_data/nifti \
    CALCULUS_SEG=ureteric_whole_stone_data/seg \
    python -m calculus.common.render_masks
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import nibabel as nib                    # noqa: E402
import numpy as np                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
from calculus.common.paths import NIFTI, RUN, SEG      # noqa: E402

OUT = os.path.join(RUN, "mask_overlays")
WL, WW = 60, 400            # soft-tissue window: organs, not stones

STRUCTURES = [
    ("kidney_left", "#e03030"), ("kidney_right", "#e03030"),
    ("kidney_cyst_left", "#4da3ff"), ("kidney_cyst_right", "#4da3ff"),
    ("urinary_bladder", "#ffd24d"),
    ("iliac_artery_left", "#39d353"), ("iliac_artery_right", "#39d353"),
    ("aorta", "#ff8c1a"), ("inferior_vena_cava", "#00cfcf"),
    ("sacrum", "#9aa5b1"), ("hip_left", "#9aa5b1"), ("hip_right", "#9aa5b1"),
    ("vertebrae_L1", "#c8b4e0"), ("vertebrae_L5", "#c8b4e0"),
]


def win(a):
    return np.clip((a - (WL - WW / 2)) / WW, 0, 1)


def outline(ax, m2d, colour, lw=0.9):
    if m2d is not None and m2d.any():
        ax.contour(m2d.T, levels=[0.5], colors=[colour], linewidths=lw)


def sheet(sid, dpi=105):
    nii = os.path.join(NIFTI, f"{sid}.nii.gz")
    seg = os.path.join(SEG, sid)
    if not (os.path.exists(nii) and os.path.isdir(seg)):
        return f"{sid}: no volume or no masks"
    img = nib.load(nii)
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    vol = np.asanyarray(img.dataobj).astype(np.float32)
    vx = float(np.prod(spacing))

    masks, ml = {}, {}
    for name, _ in STRUCTURES:
        p = os.path.join(seg, f"{name}.nii.gz")
        if os.path.exists(p):
            m = np.asanyarray(nib.load(p).dataobj) > 0
            if m.any():
                masks[name] = m
                ml[name] = m.sum() * vx / 1000.0
    if not masks:
        return f"{sid}: every mask empty"

    # the three levels the corridor is built from -- a mask can project fine and
    # still be wrong on the slice that actually matters
    def zmid(*names):
        zs = [np.nonzero(masks[n])[2].mean() for n in names if n in masks]
        return int(np.mean(zs)) if zs else None
    z_kid = zmid("kidney_left", "kidney_right")
    z_ili = zmid("iliac_artery_left", "iliac_artery_right")
    z_bla = zmid("urinary_bladder")
    levels = [(z, lab) for z, lab in ((z_kid, "kidneys"), (z_ili, "iliac crossing"),
                                      (z_bla, "bladder")) if z is not None]

    fig = plt.figure(figsize=(4.2 + 3.4 * len(levels), 8.4))
    gs = fig.add_gridspec(1, 1 + len(levels), width_ratios=[1.25] + [1] * len(levels),
                          wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(win(vol.max(axis=1)).T, cmap="gray", origin="lower",
              aspect=spacing[2] / spacing[0])
    for name, colour in STRUCTURES:
        if name in masks:
            ax.contour(masks[name].any(axis=1).T, levels=[0.5], colors=[colour],
                       linewidths=1.0, alpha=0.9)
    for z, lab in levels:
        ax.axhline(z, color="w", lw=0.6, ls=":", alpha=0.6)
    ax.set_title("coronal, all masks projected", fontsize=9)
    ax.set_xlabel("patient left  →", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

    for i, (z, lab) in enumerate(levels, 1):
        a = fig.add_subplot(gs[0, i])
        a.imshow(win(vol[:, :, z]).T, cmap="gray", origin="upper")
        for name, colour in STRUCTURES:
            if name in masks:
                outline(a, masks[name][:, :, z], colour)
        a.set_title(f"{lab}  (slice {z})", fontsize=9)
        a.set_xticks([]); a.set_yticks([])

    head = f"{sid}    {len(masks)} masks"
    for k, lab in (("kidney_left", "L kidney"), ("kidney_right", "R kidney"),
                   ("urinary_bladder", "bladder")):
        if k in ml:
            head += f"    {lab} {ml[k]:.0f} mL"
    sub = ("red kidney · blue cyst · yellow bladder · green iliac · "
           "orange aorta · cyan IVC · grey bone · lilac vertebrae")
    fig.suptitle(head + "\n" + sub, fontsize=11)

    # flag the mask faults that would break the corridor, in red
    bad = []
    for k, lab in (("kidney_left", "L kidney"), ("kidney_right", "R kidney")):
        if k not in ml:
            bad.append(f"{lab} MISSING")
        elif ml[k] < 60:
            bad.append(f"{lab} only {ml[k]:.0f} mL")
    if "urinary_bladder" not in ml:
        bad.append("BLADDER MISSING - no UVJ, so no corridor on either side")
    if bad:
        fig.text(0.5, 0.028, "CHECK:  " + "   |   ".join(bad), ha="center",
                 fontsize=11, color="#c0392b", weight="bold")

    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, f"{sid}.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return (f"{sid}: {len(masks)} masks"
            + (f"   <-- {'; '.join(bad)}" if bad else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", nargs="*", default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    ids = a.studies or sorted(os.path.basename(p.rstrip("/"))
                              for p in glob.glob(os.path.join(SEG, "*/")))
    if not a.overwrite:
        ids = [s for s in ids if not os.path.exists(os.path.join(OUT, f"{s}.png"))]
    print(f"{len(ids)} sheet(s) -> {OUT}\n")
    for i, sid in enumerate(ids, 1):
        try:
            msg = sheet(sid)
        except Exception as e:
            msg = f"{sid}: FAILED {type(e).__name__}: {e}"
        print(f"[{i}/{len(ids)}] {msg}", flush=True)


if __name__ == "__main__":
    main()
