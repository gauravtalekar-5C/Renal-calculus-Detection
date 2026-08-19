"""Overlays for the ureteric detections -- one reviewable sheet per study.

WHY THIS EXISTS
---------------
detect_ureteric.py writes 147 accepted stones to a CSV and not one image. Every
claim about Part 2 so far rests on numbers nobody has seen on a slice. That is
exactly the situation Part 1's overlays were built to end.

The 37-study validation says the count is wrong -- median 3 accepted stones per
study against reports that describe about one, and both ureters firing in 24 of
35 studies. Deciding WHY needs a picture: a false positive on a phlebolith, on a
vessel wall, on bowel content and on the sacral cortex look completely different
on a slice and completely alike in a CSV.

WHAT EACH SHEET SHOWS
---------------------
left, full height   coronal MIP of the whole scan
                      blue   kidneys          yellow  bladder
                      green  the interpolated PUJ -> iliac -> UVJ centreline
                      red    every accepted detection, numbered
                    -- answers "is this thing anywhere near the ureter's course"

right, one row per detection
    axial context (+-45 mm)  the slice it was found on, with the kidney,
                             bladder and arterial outlines, plus a green cross
                             at the centreline's nearest point
    axial zoom (+-14 mm)     the stone's own voxels contoured in cyan, recovered
                             from the CT at GROW_HU -- the same voxel set the
                             measurement came from, not a marker drawn at the
                             stored diameter

Each row is captioned with side, zone, size, peak HU, distance to the UVJ, how
far off the centreline it sits, and its per-side HU rank.

REJECTED CANDIDATES
-------------------
--rejected N adds the N densest REJECTED candidates per study, captioned in red
with the reason that killed them. That is how a miss gets diagnosed: study
8264423 has a 5 x 3 mm, 758 HU stone in the report and zero accepted
detections, so whatever happened to it happened in the rejection chain.

Usage:
    CALCULUS_RUN=run_ureter ./venv/bin/python utils/render_ureteric_overlays.py
    CALCULUS_RUN=run_ureter ./venv/bin/python utils/render_ureteric_overlays.py \
        --studies 8264423 --rejected 6
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")                    # headless: no display on this box
import matplotlib.pyplot as plt          # noqa: E402
import nibabel as nib                    # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# package root -> project root: calculus/<sub>/x.py is two levels down
ROOT = os.path.dirname(os.path.dirname(HERE))
from calculus.common.paths import CSV, NIFTI, SEG        # noqa: E402
from calculus.common.paths import OVERLAYS as OUT        # noqa: E402
from calculus.ureter import ureter_corridor as uc             # noqa: E402
# the stone-voxel recovery is Part 1's, imported rather than copied so the
# contour and the measured object can never drift apart
from calculus.kidney.render_kidney_3d import stone_voxels   # noqa: E402

WL, WW = 400, 1800          # window level / width -- wide, for calculi
CONTEXT_MM = 45             # half-size of the axial context panel
ZOOM_MM = 14                # half-size of the tight panel
MAX_ROWS = 8                # accepted detections drawn per sheet

# Masks drawn as outlines on the context panel. Vessels are included because
# arterial wall calcification is the false positive this detector was built to
# reject -- seeing the artery next to the detection is half the diagnosis.
CONTEXT_MASKS = [("kidney_left", "#4da3ff"), ("kidney_right", "#4da3ff"),
                 ("urinary_bladder", "#ffd24d"), ("aorta", "#7CFC00"),
                 ("iliac_artery_left", "#7CFC00"),
                 ("iliac_artery_right", "#7CFC00"),
                 ("inferior_vena_cava", "#00cfcf")]

MASK_NAMES = [m for m, _ in CONTEXT_MASKS] + ["sacrum", "hip_left", "hip_right"]


def win(a):
    return np.clip((a - (WL - WW / 2)) / WW, 0, 1)


def load_masks(sid):
    out = {}
    for name in MASK_NAMES:
        p = os.path.join(SEG, sid, f"{name}.nii.gz")
        if os.path.exists(p):
            m = np.asanyarray(nib.load(p).dataobj) > 0
            if m.any():
                out[name] = m
    return out


def outline(ax, mask2d, colour, lw=0.8):
    if mask2d is not None and mask2d.any():
        ax.contour(mask2d.T, levels=[0.5], colors=[colour], linewidths=lw)


def centrelines(masks, shape):
    """{side: (N,3) path} using the detector's own landmark functions.

    The corridor mask itself is NOT rebuilt: it needs a full-volume distance
    transform and nothing here draws it. Only the landmarks and the interpolated
    path are needed, and those are cheap.
    """
    out = {}
    bladder = masks.get("urinary_bladder")
    if bladder is None:
        return out
    mx = uc._midline_x(masks, shape)
    for side in ("left", "right"):
        kid = masks.get(f"kidney_{side}")
        if kid is None:
            continue
        puj = uc.landmark_puj(kid, mx)
        uvj = uc.landmark_uvj(bladder, side, mx)
        if puj is None or uvj is None:
            continue
        ili, _ = uc.landmark_iliac(masks, side, puj, uvj)
        out[side] = uc.centreline(puj, ili, uvj)
    return out
# WHAT THIS FUNCTION DOES: rebuilds the curved line from the kidney outlet, past
# the iliac crossing, to the bladder, for each side -- the course a stone has to
# lie near to be called ureteric -- so it can be drawn next to the detections.


def parse_centroid(v):
    """'203,268,154' or '[203, 268, 154]' -> [203, 268, 154].

    Both spellings exist in our CSVs. Parsing this with a bare try/except is how
    an earlier renderer silently drew every study as detection-free.
    """
    try:
        c = [int(float(t)) for t in str(v).strip("[]() ").split(",")]
    except (ValueError, TypeError):
        return None
    return c if len(c) == 3 else None


def caption(r, rejected=False):
    bits = [f"#{int(r.candidate_id)}", str(r.side), str(r.zone),
            f"{r.max_diameter_mm:.1f} mm", f"{int(r.hu_max)} HU"]
    if r.dist_to_uvj_along_mm == r.dist_to_uvj_along_mm:
        bits.append(f"{r.dist_to_uvj_along_mm:.0f} mm to UVJ")
    if r.off_path_mm == r.off_path_mm:
        bits.append(f"{r.off_path_mm:.0f} mm off path")
    if rejected:
        bits.append(f"REJECTED: {r.reject_reason}")
    else:
        rank = "" if r.hu_rank_side != r.hu_rank_side else f"rank {int(r.hu_rank_side)}"
        bits.append(rank + ("  REPORTED" if bool(r.report_this) else ""))
    return "   ".join(b for b in bits if b)


def sheet(sid, cand, max_rows=MAX_ROWS, n_rejected=0, dpi=105):
    nii = os.path.join(NIFTI, f"{sid}.nii.gz")
    if not os.path.exists(nii):
        return f"{sid}: no nifti"
    img = nib.load(nii)
    vol = np.asanyarray(img.dataobj).astype(np.float32)
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    masks = load_masks(sid)
    paths = centrelines(masks, vol.shape)

    acc = cand[cand.is_stone.astype(bool)].sort_values(
        ["report_this", "hu_max"], ascending=[False, False])
    rej = cand[~cand.is_stone.astype(bool)].sort_values("hu_max",
                                                        ascending=False)
    shown = acc.head(max_rows)
    shown_rej = rej.head(n_rejected) if n_rejected else rej.iloc[:0]
    rows = [(r, False) for r in shown.itertuples()] + \
           [(r, True) for r in shown_rej.itertuples()]
    if not rows:
        rows = []

    nrow = max(1, len(rows))
    fig = plt.figure(figsize=(15.5, 2.75 * nrow + 1.4))
    # the MIP keeps its anatomical aspect, so it renders tall and narrow however
    # much width it is given; 1.7 stops it shrinking to a sliver on long sheets
    gs = fig.add_gridspec(nrow, 3, width_ratios=[1.7, 1, 1],
                          hspace=0.22, wspace=0.06)

    # ---- left column: coronal MIP, spanning every row ---------------------
    axm = fig.add_subplot(gs[:, 0])
    axm.imshow(win(vol.max(axis=1)).T, cmap="gray", origin="lower",
               aspect=spacing[2] / spacing[0])
    for name, colour in CONTEXT_MASKS[:3]:
        m = masks.get(name)
        if m is not None:
            axm.contour(m.any(axis=1).T, levels=[0.5], colors=[colour],
                        linewidths=1.0, alpha=0.9)
    for side, p in paths.items():
        axm.plot(p[:, 0], p[:, 2], "-", color="#39d353", lw=1.4, alpha=0.9)
    for r in acc.itertuples():
        c = parse_centroid(r.centroid_vox)
        if c:
            axm.plot(c[0], c[2], "o", mfc="none", mec="#ff3b3b", mew=1.6, ms=13)
            axm.text(c[0] + 8, c[2], str(int(r.candidate_id)), color="#ff3b3b",
                     fontsize=8, va="center")
    axm.set_title("coronal MIP\nblue kidneys, yellow bladder, "
                  "green ureteric course, red detections", fontsize=9)
    axm.set_xlabel("patient left  →", fontsize=8)
    axm.set_ylabel("↑ head", fontsize=8)
    axm.set_xticks([]); axm.set_yticks([])

    # ---- right columns: one row per detection -----------------------------
    for i, (r, is_rej) in enumerate(rows):
        c = parse_centroid(r.centroid_vox)
        axes = [fig.add_subplot(gs[i, 1]), fig.add_subplot(gs[i, 2])]
        if c is None:
            for ax in axes:
                ax.text(.5, .5, f"unparseable centroid {r.centroid_vox!r}",
                        ha="center", va="center", fontsize=8)
                ax.set_axis_off()
            continue
        x, y, z = [int(np.clip(v, 0, n - 1)) for v, n in zip(c, vol.shape)]
        sl = vol[:, :, z]

        # the centreline point nearest this slice, so "off path" is visible and
        # not just a number in the caption
        near = None
        p = paths.get(str(r.side))
        if p is not None and len(p):
            near = p[int(np.argmin(np.abs(p[:, 2] - z)))]

        smask, ssl = stone_voxels(vol, [x, y, z], spacing, vol.shape)

        for ax, half_mm, tag in ((axes[0], CONTEXT_MM, "context"),
                                 (axes[1], ZOOM_MM, "zoom")):
            hx = max(4, int(half_mm / spacing[0]))
            hy = max(4, int(half_mm / spacing[1]))
            x0, x1 = max(0, x - hx), min(vol.shape[0], x + hx)
            y0, y1 = max(0, y - hy), min(vol.shape[1], y + hy)
            ax.imshow(win(sl[x0:x1, y0:y1]).T, cmap="gray", origin="upper")
            if tag == "context":
                for name, colour in CONTEXT_MASKS:
                    m = masks.get(name)
                    if m is not None:
                        outline(ax, m[x0:x1, y0:y1, z], colour)
                if near is not None:
                    ax.plot(near[0] - x0, near[1] - y0, "+", color="#39d353",
                            mew=1.6, ms=13)
                ax.plot(x - x0, y - y0, "o", mfc="none",
                        mec="#ff3b3b" if not is_rej else "#ff8c1a",
                        mew=1.6, ms=20)
            else:
                # the detection's own voxels, contoured. Nothing else is drawn
                # here: this panel exists to answer "is that a stone".
                if smask is not None:
                    full = np.zeros(vol.shape[:2], bool)
                    full[ssl[0], ssl[1]] = smask[:, :, z - ssl[2].start] \
                        if ssl[2].start <= z < ssl[2].stop else False
                    outline(ax, full[x0:x1, y0:y1], "#00e5e5", lw=1.2)
            ax.set_xticks([]); ax.set_yticks([])
            # The caption is long and belongs to the ROW, not to one panel. Set
            # as a title on the zoom panel too it collided with the caption and
            # both became unreadable, so the scale goes inside the frame.
            if tag == "context":
                ax.set_title(caption(r, is_rej), fontsize=8.5,
                             color="#c0392b" if is_rej else "#123", loc="left")
            else:
                ax.text(0.02, 0.97, f"±{half_mm} mm", transform=ax.transAxes,
                        fontsize=7.5, color="w", va="top", ha="left")

    n_acc, n_rej = len(acc), len(rej)
    head = (f"{sid}    {n_acc} accepted detection(s), {n_rej} rejected "
            f"candidate(s)")
    if n_acc > len(shown):
        # no silent caps: say what was left out, or the sheet reads as complete
        head += f"    (showing the {len(shown)} densest of {n_acc})"
    if not len(rows):
        head += "    -- nothing to draw"
    fig.suptitle(head, fontsize=12, y=0.995)
    os.makedirs(OUT, exist_ok=True)
    dest = os.path.join(OUT, f"{sid}_ureteric.png")
    fig.savefig(dest, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return f"{sid}: {n_acc} accepted ({len(shown)} drawn), {n_rej} rejected"
# WHAT THIS FUNCTION DOES: draws one review sheet for a study -- the whole-scan
# coronal view with the ureteric course and every detection on it, plus a
# close-up pair for each detection -- and saves it as a PNG.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", nargs="*", default=None)
    ap.add_argument("--candidates",
                    default=os.path.join(CSV, "ureter_candidates.csv"))
    ap.add_argument("--max-rows", type=int, default=MAX_ROWS)
    ap.add_argument("--rejected", type=int, default=0,
                    help="also draw the N densest rejected candidates, "
                         "captioned with the reason they were killed")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(a.candidates):
        raise SystemExit(f"no {a.candidates} -- run detect_ureteric.py first "
                         f"(or set CALCULUS_RUN to the run that has it)")
    cand = pd.read_csv(a.candidates)
    cand["study_id"] = cand["study_id"].astype(str)
    ids = [str(s) for s in a.studies] if a.studies else \
        sorted(cand.study_id.unique())
    if not a.overwrite:
        skip = {s for s in ids
                if os.path.exists(os.path.join(OUT, f"{s}_ureteric.png"))}
        if skip:
            print(f"skipping {len(skip)} already drawn")
            ids = [s for s in ids if s not in skip]

    print(f"{len(ids)} study sheet(s) -> {OUT}\n")
    for i, sid in enumerate(ids, 1):
        sub = cand[cand.study_id == sid]
        try:
            msg = sheet(sid, sub, max_rows=a.max_rows, n_rejected=a.rejected)
        except Exception as e:
            msg = f"{sid}: FAILED {type(e).__name__}: {e}"
        print(f"[{i}/{len(ids)}] {msg}", flush=True)


if __name__ == "__main__":
    main()
