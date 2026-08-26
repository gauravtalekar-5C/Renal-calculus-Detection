"""Secondary-capture images for the API: one per renal stone, one coronal for
the ureteric and bladder stones together.

WHY A SEPARATE RENDERER
-----------------------
The existing overlays are review sheets: render_overlays writes a coronal MIP, a
contact sheet, per-stone axials and a kidney montage; render_ureteric_overlays
writes a multi-panel figure with a row per detection AND the densest rejections.
Those are the right thing for a radiologist auditing a run, and the wrong thing
to hand an integration -- a caller wants ONE image per finding, at a predictable
path, with nothing else in it.

So this produces exactly what the API references, and leaves the review sheets
untouched.

    secondary/kidney_01.png    one per RENAL calculus, in report order
    secondary/kidney_02.png
    secondary/coronal.png      ONE coronal, with every ureteric AND bladder
                               stone marked on it

WHY ONE COMBINED CORONAL AND NOT ONE IMAGE PER STONE
A ureteric stone's clinical meaning is its position along the tract, and that is
only legible against the whole course -- kidney, ureter, bladder, in one frame.
Three separate crops of three stones lose the thing a urologist is looking for,
which is where they sit relative to each other and to the UVJ. Bladder stones go
on the same frame for the same reason: a stone at the VUJ and a stone in the
bladder lumen 20 mm away are a different problem from two stones in the ureter,
and only one picture shows that.

Renal stones are the opposite case: a 2 mm calyceal stone is invisible on a
whole-abdomen coronal, so each gets its own zoomed axial.
"""
import argparse
import ast
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import nibabel as nib                     # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402

from calculus.common.paths import CSV, NIFTI, SEG, ensure   # noqa: E402
from calculus.report.make_report import fmt_size            # noqa: E402

# Bone-ish window: a calculus is far easier to judge wide than on soft tissue.
WIN = (-150.0, 1200.0)
ZOOM_MM = 22.0          # half-width of a per-stone crop
CONTEXT_MM = 60.0       # half-width of its context panel
SUBDIR = "secondary"


def _parse_centroid(v):
    """centroid_vox round-trips as '12,34,56' or '[12, 34, 56]'."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    txt = str(v).strip()
    if not txt or txt.lower() == "nan":
        return None
    try:
        if txt.startswith("["):
            return [float(x) for x in ast.literal_eval(txt)]
        return [float(x) for x in txt.split(",")]
    except (ValueError, SyntaxError):
        return None


def _load(sid):
    p = os.path.join(NIFTI, f"{sid}.nii.gz")
    if not os.path.exists(p):
        return None, None
    img = nib.load(p)
    return (np.asanyarray(img.dataobj).astype(np.float32),
            tuple(float(z) for z in img.header.get_zooms()[:3]))


def _mask(sid, name):
    p = os.path.join(SEG, sid, f"{name}.nii.gz")
    if not os.path.exists(p):
        return None
    return np.asanyarray(nib.load(p).dataobj) > 0


def _csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def kidney_capture(sid, vol, spacing, row, n, out_dir):
    """One renal stone: zoomed axial plus a wider context view."""
    cen = _parse_centroid(getattr(row, "centroid_vox", None))
    if cen is None:
        return None
    ci = [int(round(c)) for c in cen]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.7))
    for ax, half, tag in ((axes[0], ZOOM_MM, "zoom"),
                          (axes[1], CONTEXT_MM, "context")):
        hp = [int(np.ceil(half / s)) for s in spacing]
        sl = tuple(slice(max(0, ci[i] - hp[i]),
                         min(vol.shape[i], ci[i] + hp[i] + 1)) for i in range(3))
        sub = vol[sl][:, :, ci[2] - sl[2].start] if (
            sl[2].start <= ci[2] < sl[2].stop) else vol[sl][:, :, 0]
        ax.imshow(sub.T, cmap="gray", vmin=WIN[0], vmax=WIN[1], origin="lower")
        # a ring, not a filled marker: the stone must stay visible inside it
        ax.plot(ci[0] - sl[0].start, ci[1] - sl[1].start, "o", mfc="none",
                mec="#ff3b3b", mew=1.6, ms=16 if tag == "zoom" else 10)
        ax.set_title(f"{tag}  (+/-{half:.0f} mm)", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    # fmt_size, not a local f-string: the index below is matched against the
    # report's size cell, and two independent formatters drifting apart is
    # exactly how the kidney captures silently vanished from the response once
    # the axis label was added to one of them.
    size = fmt_size(getattr(row, "dim_tr_mm", float("nan")),
                    getattr(row, "dim_ap_mm", float("nan")),
                    getattr(row, "dim_cc_mm", float("nan")))
    # The raw column holds internal tokens -- "lower_pole", "interpolar" --
    # which read as code in a caption. Use the report's own wording, and say
    # "calyx" only when the stone is in the collecting system, exactly as
    # make_report does.
    ZONE = {"upper_pole": "upper", "interpolar": "mid", "lower_pole": "lower"}
    raw = str(getattr(row, "location", "") or "")
    third = ZONE.get(raw, raw.replace("_", " "))
    in_cs = bool(getattr(row, "in_collecting_system", False))
    loc = (f"{third} pole calyx" if third and in_cs
           else f"{third} third" if third else "")
    fig.suptitle(f"renal calculus {n}   {size} mm   "
                 f"{getattr(row, 'hu_max', float('nan')):.0f} HU   "
                 f"{str(getattr(row, 'side', '') or '').title()} {loc}\n{sid}",
                 fontsize=9, linespacing=1.6)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    dest = os.path.join(out_dir, f"kidney_{n:02d}.png")
    fig.savefig(dest, dpi=110, pil_kwargs={"optimize": True})
    plt.close(fig)
    # Identity, not position. The report groups calculi right-side-then-left,
    # while these are written in CSV row order, so the two sequences disagree
    # -- kidney_01 was a LEFT stone being handed to the report's first RIGHT
    # entry. A capture attached to the wrong stone is worse than no capture,
    # because a reader has no way to tell.
    return {"file": os.path.basename(dest),
            "side": str(getattr(row, "side", "") or "").lower(),
            "density_hu": int(round(float(getattr(row, "hu_max", 0) or 0))),
            "size_mm": size}


def coronal_capture(sid, vol, spacing, ureteric, bladder, out_dir):
    """ONE coronal MIP with every ureteric and bladder stone marked."""
    if not len(ureteric) and not len(bladder):
        return None
    fig, ax = plt.subplots(figsize=(5.4, 7.4))
    ax.imshow(np.clip(vol.max(axis=1), WIN[0], WIN[1]).T, cmap="gray",
              origin="lower", aspect=spacing[2] / spacing[0])
    for name, colour in (("kidney_left", "#4da6ff"), ("kidney_right", "#4da6ff"),
                         ("urinary_bladder", "#ffd24d")):
        m = _mask(sid, name)
        if m is not None and m.any():
            ax.contour(m.any(axis=1).T, levels=[0.5], colors=[colour],
                       linewidths=1.0, alpha=0.9)
    n = 0
    # Ureteric and bladder stones are numbered in ONE sequence across both, so
    # the caption matches what a reader counts on the image rather than
    # restarting per compartment.
    #
    # BLADDER IS DRAWN LAST, larger and dashed. On 8676429 there are eight
    # ureteric rings clustered above the bladder and a single bladder stone
    # among them; drawn first and the same size, it was indistinguishable. The
    # one finding that changes the organ should not be the hardest to see.
    for label, df, colour, ms, ls, lw in (
            ("U", ureteric, "#ff3b3b", 14, "solid", 1.8),
            ("B", bladder, "#ff8c1a", 22, "dashed", 2.4)):
        for r in df.itertuples():
            c = _parse_centroid(getattr(r, "centroid_vox", None))
            if c is None:
                continue
            n += 1
            ax.plot(c[0], c[2], "o", mfc="none", mec=colour, mew=lw, ms=ms,
                    ls=ls)
            # Bladder labels go BELOW the ring. To the right is where the
            # ureteric labels sit, and on 8676429 the ureteric detections
            # cluster right on top of the bladder, so a right-hand bladder
            # label lands in the middle of them.
            dx, dy = ((0, -ms * 0.9) if label == "B" else (ms * 0.7, 0))
            ax.text(c[0] + dx, c[2] + dy, f"{label}{n}", color=colour,
                    fontsize=10 if label == "B" else 9,
                    ha="center" if label == "B" else "left", va="center",
                    fontweight="bold" if label == "B" else "normal")
    ax.set_title("coronal MIP -- blue kidneys, yellow bladder,\n"
                 f"red ureteric, orange bladder calculi\n{sid}", fontsize=8)
    ax.set_xlabel("patient left  ->", fontsize=8)
    ax.set_ylabel("^ head", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    dest = os.path.join(out_dir, "coronal.png")
    fig.savefig(dest, dpi=110, pil_kwargs={"optimize": True})
    plt.close(fig)
    return dest


def render(sid, run):
    """Write the secondary captures for one study; return the paths."""
    vol, spacing = _load(sid)
    if vol is None:
        return {"kidney": [], "coronal": None, "error": "no nifti"}
    out_dir = os.path.join(run, "overlays", SUBDIR, str(sid))
    os.makedirs(out_dir, exist_ok=True)

    per = os.path.join(CSV, "per_study")
    kid = _csv(os.path.join(per, f"{sid}_candidates.csv"))
    if len(kid) and "is_stone" in kid.columns:
        kid = kid[kid.is_stone.astype(bool)]
        # renal only: a kidney-detector row whose compartment is bladder is the
        # same object the bladder detector reports, and it belongs on the
        # coronal, not in its own crop
        if "compartment" in kid.columns:
            kid = kid[~kid.compartment.astype(str).str.startswith("bladder")]
    ure = _csv(os.path.join(per, f"{sid}_ureter_candidates.csv"))
    if len(ure) and "is_stone" in ure.columns:
        ure = ure[ure.is_stone.astype(bool)]
        if "report_this" in ure.columns:
            keep = ure.report_this.fillna(True).astype(bool)
            ure = ure[keep]
    bla = _csv(os.path.join(per, f"{sid}_bladder_candidates.csv"))
    if len(bla) and "is_stone" in bla.columns:
        bla = bla[bla.is_stone.astype(bool)]

    kpaths = []
    for n, r in enumerate(kid.itertuples() if len(kid) else [], start=1):
        p = kidney_capture(sid, vol, spacing, r, n, out_dir)
        if p:
            kpaths.append(p)
    with open(os.path.join(out_dir, "index.json"), "w") as fh:
        json.dump({"kidney": kpaths}, fh, indent=2)
    cpath = coronal_capture(sid, vol, spacing,
                            ure if len(ure) else pd.DataFrame(),
                            bla if len(bla) else pd.DataFrame(), out_dir)
    return {"kidney": kpaths, "coronal": cpath}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", nargs="+", required=True)
    a = ap.parse_args()
    run = ensure()
    for sid in a.studies:
        r = render(str(sid), run)
        print(f"  {sid}: {len(r['kidney'])} renal capture(s), "
              f"coronal={'yes' if r['coronal'] else 'no'}")


if __name__ == "__main__":
    main()
