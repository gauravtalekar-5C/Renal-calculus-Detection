"""Draw the candidates the detector THREW AWAY, so a human can judge the call.

WHY THIS EXISTS
---------------
render_overlays draws accepted stones only -- it reads baseline_stones.csv. So
every rejection was invisible: the detector saw an object, discarded it, and
nothing in the output showed what it had looked at. That is the wrong default for
a system whose misses matter. On the audit cohort, 4 of the 10 renal misses were
objects we DETECTED and rejected as `no_dense_core`, with hu_max of 177-245
against a 200 HU seed threshold. Those are exactly the calls a radiologist should
be able to overrule, and they were unreviewable.

Each panel shows, around one rejected candidate:
    axial + coronal at the centroid, +/- ZOOM_MM
    a cyan cross at the centroid
    size, hu_max, hu_mean, volume, and the reject_reason that killed it

Usage:
    python -m calculus.kidney.render_rejected --studies 8622144 8364723
    python -m calculus.kidney.render_rejected --studies 8622144 --out /some/dir
"""
import argparse
import ast
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

from calculus.common.paths import CSV, NIFTI, RUN

ZOOM_MM = 25.0          # half-width of the context view
WIN = (-150.0, 1200.0)  # display window: soft tissue up to dense stone


def _parse_centroid(v):
    """centroid_vox round-trips through CSV as a string like '[12, 34, 56]'."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return [float(x) for x in v]
    try:
        return [float(x) for x in ast.literal_eval(str(v))]
    except (ValueError, SyntaxError):
        return None


def _panel(ax, img, title):
    ax.imshow(img.T, cmap="gray", vmin=WIN[0], vmax=WIN[1], origin="lower")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def render_study(sid, cand, out_dir):
    nii = os.path.join(NIFTI, f"{sid}.nii.gz")
    if not os.path.exists(nii):
        print(f"  {sid}: no volume at {nii}")
        return 0
    img = nib.load(nii)
    vol = np.asanyarray(img.dataobj).astype(np.float32)
    sp = np.abs(np.diag(img.affine))[:3]

    rows = cand[cand.study_id.astype(str) == str(sid)].copy()
    rr = rows.reject_reason.fillna("").astype(str).str.strip()
    rows = rows[rr != ""]           # rejected only; kept ones are in overlays/
    if not len(rows):
        print(f"  {sid}: no rejected candidates")
        return 0

    n = 0
    for k, r in enumerate(rows.itertuples(), start=1):
        cen = _parse_centroid(r.centroid_vox)
        if cen is None:
            print(f"  {sid}: candidate {k} has no usable centroid")
            continue
        ci = [int(round(c)) for c in cen]
        half = [int(np.ceil(ZOOM_MM / s)) for s in sp]
        sl = tuple(slice(max(0, ci[i] - half[i]),
                         min(vol.shape[i], ci[i] + half[i] + 1)) for i in range(3))
        sub = vol[sl]
        # centroid position inside the crop, for the marker
        loc = [ci[i] - sl[i].start for i in range(3)]

        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9))
        _panel(axes[0], sub[:, :, loc[2]], f"axial  z={ci[2]}")
        axes[0].plot(loc[0], loc[1], "+", color="cyan", ms=13, mew=1.6)
        _panel(axes[1], sub[:, loc[1], :], f"coronal  y={ci[1]}")
        axes[1].plot(loc[0], loc[2], "+", color="cyan", ms=13, mew=1.6)

        reason = str(r.reject_reason)
        fig.suptitle(
            f"{sid}   REJECTED: {reason}\n"
            f"{r.max_diameter_mm:.1f} mm   hu_max {r.hu_max:.0f}   "
            f"hu_mean {r.hu_mean:.0f}   {r.volume_mm3:.1f} mm3   "
            f"{r.compartment}  {r.side or '-'}",
            fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.86])
        dest = os.path.join(out_dir, f"{sid}_rejected_{k:02d}_{reason}.png")
        fig.savefig(dest, dpi=110)
        plt.close(fig)
        n += 1
    print(f"  {sid}: {n} rejected candidate panel(s)")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = a.out or os.path.join(RUN, "rejected")
    os.makedirs(out, exist_ok=True)
    cand = pd.read_csv(os.path.join(CSV, "candidates.csv"))
    total = sum(render_study(s, cand, out) for s in a.studies)
    print(f"\n{total} panel(s) -> {out}")


if __name__ == "__main__":
    main()
