#!/usr/bin/env python
"""One composite PNG per study: the whole stone picture on a single sheet.

WHY THIS AND NOT THE 3D SCENE
The 3D scene draws the segmentation. This draws the SCAN. For a reader deciding
what to do about a stone, those are not equally useful: the segmentation is our
interpretation, the coronal MIP is the evidence. A sheet also prints, pastes into
a PDF, attaches to an email and embeds in the JSON response, none of which a
WebGL canvas does.

The first attempt at static output rendered the 3D surfaces through matplotlib's
3D axes. It looked bad for a structural reason worth recording: that backend is a
painter's-algorithm renderer with no depth buffer, so translucent organs sort
incorrectly against the stones inside them -- the exact relationship the picture
exists to show -- and it reserves a cube of axes space, leaving the anatomy a
small island in a large empty frame.

LAYOUT
    left   a large coronal maximum-intensity projection of the real volume,
           kidneys and bladder outlined, every calculus ringed and numbered
    right  one zoomed axial crop per calculus, in the same numbering, each
           captioned with what was measured
Numbering runs top to bottom of the body, so the sheet reads the way a reader
scans: kidney, then down the tract, then bladder.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.patches import Circle                             # noqa: E402
import nibabel as nib                                             # noqa: E402
import numpy as np                                                # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from calculus.common.paths import NIFTI, SEG                       # noqa: E402
from calculus.report.render_3d import _mask, stones_for            # noqa: E402
# The SAME zone mapping the report and the JSON use. Restating it here is how
# make_report_full ended up saying "Ureter (VUJ)" for hours after the report
# had moved to three zones.
from calculus.report.make_report import ZONE_UR                    # noqa: E402

BG = "#0b0f16"
INK = "#e8eef7"
DIM = "#8d9cb0"
KID = "#5a9ef2"
BLA = "#f0be3d"
WIN = (-150.0, 1100.0)      # wide window: a calculus is easier to judge wide
CROP_MM = 26.0              # half-width of a per-stone axial crop


def hu_colour(hu):
    """Amber to magenta over 200-1500 HU. Same ramp as the interactive scene."""
    t = max(0.0, min(1.0, (hu - 200) / 1300.0))
    stops = np.array([[1, .83, .47], [1, .62, .30], [1, .37, .32], [.78, .18, .43]])
    x = t * (len(stops) - 1)
    i = min(len(stops) - 2, int(x))
    return tuple(stops[i] + (stops[i + 1] - stops[i]) * (x - i))


def _label(st):
    o = st["organ"]
    if "kidney" in o or "renal" in o:
        base = "Kidney"
    elif o == "ureter":
        base = "Ureter"
    elif o == "bladder":
        base = "Bladder"
    else:
        base = o.replace("_", " ").title()
    side = st["side"][:1].upper() if st["side"] in ("left", "right") else ""
    z = str(st.get("zone") or "")
    zone = f" {ZONE_UR.get(z, z)}" if z and base == "Ureter" else ""
    return f"{side}{' ' if side else ''}{base}{zone}".strip()


def build(sid, out_dir, seg_dir=None, nifti=None):
    nii = nifti or os.path.join(NIFTI, f"{sid}.nii.gz")
    seg = seg_dir or os.path.join(SEG, str(sid))
    if not os.path.exists(nii):
        return None
    img = nib.load(nii)
    vol = np.asanyarray(img.dataobj).astype(np.float32)
    sp = tuple(float(z) for z in img.header.get_zooms()[:3])

    stones = stones_for(sid, sp)
    # superior first, so the sheet reads the way a reader scans the body
    stones.sort(key=lambda s: -s["c"][2])
    for i, s in enumerate(stones, 1):
        s["n"] = i

    n = len(stones)
    cols = 2 if n > 4 else 1
    rows = max(1, int(np.ceil(n / cols))) if n else 1
    fig_w = 8.6 + (3.1 * cols if n else 0)
    fig_h = max(9.4, 2.35 * rows + 1.6)
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)
    # Whether the stat block fits beside the title depends only on the figure
    # width and the number of cells, both known here -- so decide it now and let
    # the grid start below it. Deciding it after the grid was created is how the
    # "CALCULI" label ended up printed on top of the "Coronal MIP" caption.
    CELL_IN, N_CELLS = 1.12, 6
    _start_in = fig_w - 0.35 - CELL_IN * N_CELLS
    second_row = _start_in < 4.7
    gs = fig.add_gridspec(
        rows, 1 + cols, left=.035, right=.972,
        top=.800 if second_row else .905, bottom=.095,
        wspace=.10, hspace=.30,
        width_ratios=[3.05] + [1.0] * cols)

    # ---- the coronal MIP ---------------------------------------------------
    ax = fig.add_subplot(gs[:, 0], facecolor=BG)
    mip = np.clip(vol, WIN[0], WIN[1]).max(axis=1).T
    ax.imshow(mip, cmap="gray", origin="lower", aspect=sp[2] / sp[0],
              vmin=WIN[0], vmax=WIN[1], interpolation="bilinear")
    for name, col in (("kidney_left", KID), ("kidney_right", KID),
                      ("urinary_bladder", BLA)):
        m = _mask(seg, name) if os.path.isdir(seg) else None
        if m is not None:
            ax.contour(m.any(axis=1).T, levels=[.5], colors=[col],
                       linewidths=1.15, alpha=.85)
    for s in stones:
        x, z = s["c"][0] / sp[0], s["c"][2] / sp[2]
        c = hu_colour(s["hu"])
        # ring scaled to the stone but floored so a 2 mm calculus is findable
        r = max(s["d"] / 2.0 / sp[0], 7.0)
        ax.add_patch(Circle((x, z), r * 1.9, fill=False, ec=c, lw=1.7, alpha=.95))
        # A filled badge, not bare text. On this study seven detections cluster
        # over the bladder and bare numerals landed on top of each other and on
        # the rings; a badge with a dark ground stays legible in a pile.
        ax.annotate(str(s["n"]), (x, z),
                    xytext=(r * 1.9 + 9, r * 1.9 + 4),
                    textcoords="offset points", color="#0b0f16", fontsize=8.5,
                    fontweight="bold", ha="center", va="center", zorder=6,
                    bbox=dict(boxstyle="circle,pad=0.28", fc=c, ec="none"))
    ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    ax.set_title("Coronal MIP", color=DIM, fontsize=9, loc="left", pad=6)
    ax.text(.99, -.012, "patient left →", transform=ax.transAxes,
            color=DIM, fontsize=7.5, ha="right", va="top")

    # ---- one crop per calculus --------------------------------------------
    for k, s in enumerate(stones):
        r_, c_ = k % rows, 1 + k // rows
        if c_ > cols:
            break
        a = fig.add_subplot(gs[r_, c_], facecolor=BG)
        ci = [int(round(s["c"][i] / sp[i])) for i in range(3)]
        hp = [int(np.ceil(CROP_MM / z)) for z in sp]
        sl = tuple(slice(max(0, ci[i] - hp[i]), min(vol.shape[i], ci[i] + hp[i] + 1))
                   for i in range(3))
        zi = ci[2] if sl[2].start <= ci[2] < sl[2].stop else sl[2].start
        sub = vol[sl[0], sl[1], zi]
        a.imshow(sub.T, cmap="gray", origin="lower", vmin=WIN[0], vmax=WIN[1],
                 interpolation="bilinear")
        col = hu_colour(s["hu"])
        a.add_patch(Circle((ci[0] - sl[0].start, ci[1] - sl[1].start),
                           max(s["d"] / 2.0 / sp[0] * 1.7, 9.0),
                           fill=False, ec=col, lw=1.6))
        a.set_xticks([]); a.set_yticks([])
        for sp_ in a.spines.values():
            sp_.set_color("#20293a")
        a.set_title(f"{s['n']}   {_label(s)}", color=col, fontsize=9,
                    loc="left", pad=4, fontweight="bold")
        a.text(0, -.055, f"{s['d']:.1f} mm   ·   {s['hu']} HU",
               transform=a.transAxes, color=DIM, fontsize=8.5, va="top")

    # ---- header and footer ------------------------------------------------
    n_r = sum(1 for s in stones if "kidney" in s["organ"] or "renal" in s["organ"])
    n_u = sum(1 for s in stones if s["organ"] == "ureter")
    n_b = sum(1 for s in stones if s["organ"] == "bladder")
    big = max((s["d"] for s in stones), default=0)
    dense = max((s["hu"] for s in stones), default=0)
    verdict = "ABNORMAL — calculi detected" if stones else "No calculus detected"

    fig.text(.035, .962, "CT KUB · urinary calculi", color=INK,
             fontsize=15, fontweight="bold")
    fig.text(.035, .935, f"study {sid}", color=DIM, family="monospace",
             fontsize=8.5 if len(str(sid)) < 40 else 7.3)
    fig.text(.035, .909, verdict, color="#ff8a5c" if stones else "#6fd58f",
             fontsize=10.5, fontweight="bold")
    # LAID OUT IN INCHES, not figure fractions. The figure width changes with
    # the number of calculi (each column of crops widens the sheet), so x
    # positions that clear the study id on a 14.8 in sheet ran straight through
    # it on the 8.6 in sheet a normal study produces.
    cells = [(str(n), "calculi"), (str(n_r), "renal"), (str(n_u), "ureteric"),
             (str(n_b), "bladder"),
             (f"{big:.1f} mm" if big else "\u2014", "largest"),
             (f"{dense} HU" if dense else "\u2014", "densest")]
    # second_row and _start_in were settled before the grid; see there
    start_in = 0.30 if second_row else _start_in
    # An explicit vertical stack. The header has four rows and the stat block
    # is the only one that moves, so its second-row position is chosen to clear
    # the verdict above it rather than being nudged until it looked right:
    #   .962 title  .935 study id  .909 verdict  .865/.842 stats  grid from .80
    y_v, y_l = ((.865, .842) if second_row else (.953, .930))
    for i_, (v, lab) in enumerate(cells):
        x = (start_in + i_ * CELL_IN) / fig_w
        fig.text(x, y_v, v, color=INK, fontsize=13, fontweight="bold",
                 family="monospace")
        fig.text(x, y_l, lab.upper(), color=DIM, fontsize=7, fontweight="bold")

    fig.text(.035, .017,
             "Kidney and bladder outlines are the segmentation; the image is the "
             "unmodified scan.  No ureter is outlined — it is not visible on "
             "non-contrast CT, so its course is estimated and not drawn.  "
             "Ring colour encodes density.  Research output, not a diagnostic device.",
             color=DIM, fontsize=7.2, wrap=True)

    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, f"{sid}_sheet.png")
    fig.savefig(dest, dpi=150, facecolor=BG)
    plt.close(fig)
    return dest


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
        p = build(str(sid), a.out, seg, nii)
        print(f"  {sid}: {p}  ({os.path.getsize(p)/1024:.0f} KB)" if p
              else f"  {sid}: skipped (no nifti)")


if __name__ == "__main__":
    main()
