"""EXPERIMENT: MSER stability vs the fixed 200 HU seed. Read-only.

TOUCHES NOTHING. This file imports from calculus/ but writes only to its own
output CSV. No detector, threshold, or report is modified. Delete this file and
the pipeline is exactly as it was.

WHY
---
Our existence test is a fixed threshold: a blob is real only if it contains a
voxel >= SEED_HU (200) on the raw volume. Measured on the 54-study audit cohort
that single rule rejected 137 of 301 candidates -- including 7-10 mm objects our
own FWHM measurement scored at 800-980 HU, killed on a seed value near 150.

Liu et al. (Med Phys 2015, PMC4277558) replaced the threshold with MSER and
measured, on identical data: 35% sensitivity for thresholding, 69% for MSER
(p < 0.001). MSER asks a different question -- not "is any voxel bright enough"
but "is there a region whose AREA stays stable while the threshold sweeps".

  a real stone     looks like the same compact blob at 150, 250, 350, 450 HU
                   -> area barely changes -> STABLE
  noise speckle    appears at one level and dissolves at the next
                   -> area collapses -> UNSTABLE

That is why it survives partial volume. A genuine 800 HU microlith averaged down
to a 180 HU peak still forms a stable region against 30 HU parenchyma: it fails
the 200 HU test and passes the stability test.

HOW MSER IS COMPUTED HERE
-------------------------
Matas et al.'s definition, discretised as an explicit threshold sweep rather
than via a component tree, because the sweep is directly inspectable -- you can
print the area at every level and see the stability for yourself.

For each candidate, inside a small box around its centroid:
    for T in SWEEP:
        label (vol >= T); take the component containing the centroid
        record its area A(T)
    variation q(T) = (A(T - DELTA) - A(T + DELTA)) / A(T)
    the candidate is ACCEPTED if min_T q(T) <= MAX_VARIATION

Low variation = the region is insensitive to where you put the threshold, which
is the signature of a genuine dense object with a sharp edge.

Usage:
    python -m experiments.mser_vs_seed --studies 8622144 fk8676714
"""
import argparse
import ast
import os
import sys

import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# read-only imports: constants and the ROI builder, exactly as the detector uses
from calculus.common.paths import CSV, NIFTI, RUN          # noqa: E402
from calculus.kidney import detect_stones as ds            # noqa: E402

import nibabel as nib                                      # noqa: E402

# --- MSER parameters -------------------------------------------------------
# The sweep starts at the paper's own floor (130 HU, from calcium-scoring
# standards -- the same value as our GROW_HU) and runs past dense stone.
# The sweep must extend DELTA BELOW the lowest threshold we score, or the
# variation at the bottom edge is computed against a clamped neighbour and every
# region looks spuriously stable there. First run had every optimum sitting at
# exactly 130 HU for precisely that reason.
SWEEP = np.arange(70.0, 900.0, 20.0)
SCORE_LO = 130.0        # the paper's floor, and our GROW_HU
SCORE_HI = 800.0
DELTA = 60.0            # +/- HU used to measure how fast the area is changing
MAX_VARIATION = 0.35    # accept if the area changes by less than this fraction
MIN_VOX = 3             # the paper's 3-pixel minimum, same as ours
BOX_MM = 18.0           # half-width of the local box around a candidate


def _centroid(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return [float(x) for x in v]
    try:
        return [float(x) for x in ast.literal_eval(str(v))]
    except (ValueError, SyntaxError):
        return None


def mser_at(vol, cen_vox, spacing, verbose=False):
    """Sweep thresholds around one point; return (accepted, best_variation,
    area_at_best, threshold_at_best, trace)."""
    half = [int(np.ceil(BOX_MM / s)) for s in spacing]
    sl = tuple(slice(max(0, int(round(cen_vox[i])) - half[i]),
                     min(vol.shape[i], int(round(cen_vox[i])) + half[i] + 1))
               for i in range(3))
    sub = vol[sl]
    loc = tuple(int(round(cen_vox[i])) - sl[i].start for i in range(3))
    if not all(0 <= loc[i] < sub.shape[i] for i in range(3)):
        return False, float("nan"), 0, float("nan"), []

    # area of the component containing the centroid, at every threshold
    areas = {}
    for T in SWEEP:
        m = sub >= T
        if not m[loc]:
            areas[T] = 0
            continue
        lab, _ = ndimage.label(m)
        areas[T] = int((lab == lab[loc]).sum())

    trace, best = [], (float("inf"), 0, float("nan"))
    for T in SWEEP:
        # only score thresholds with a full DELTA of sweep on BOTH sides
        if T < max(SCORE_LO, SWEEP[0] + DELTA) or T > min(SCORE_HI, SWEEP[-1] - DELTA):
            continue
        a = areas.get(T, 0)
        if a < MIN_VOX:
            continue
        lo = areas[min(SWEEP, key=lambda x: abs(x - (T - DELTA)))]
        hi = areas[min(SWEEP, key=lambda x: abs(x - (T + DELTA)))]
        q = (lo - hi) / float(a)          # Matas' variation, > 0 as area shrinks
        trace.append((T, a, q))
        if q < best[0]:
            best = (q, a, T)
    if verbose:
        for T, a, q in trace:
            print(f"        T={T:5.0f}  area={a:6d}  variation={q:6.3f}")
    if best[0] == float("inf"):
        return False, float("nan"), 0, float("nan"), trace
    return best[0] <= MAX_VARIATION, best[0], best[1], best[2], trace


def run_study(sid, cand, verbose=False):
    p = os.path.join(NIFTI, f"{sid}.nii.gz")
    if not os.path.exists(p):
        print(f"{sid}: no volume"); return []
    img = nib.load(p)
    vol = np.asanyarray(img.dataobj).astype(np.float32)
    sp = tuple(float(x) for x in np.abs(np.diag(img.affine))[:3])

    rows = cand[cand.study_id.astype(str) == str(sid)]
    print(f"\n{'='*78}\n{sid}   {vol.shape}  {np.round(sp,2)} mm   "
          f"{len(rows)} recorded candidate(s)\n{'='*78}")
    out = []
    for r in rows.itertuples():
        cen = _centroid(r.centroid_vox)
        if cen is None:
            continue
        cur = str(r.reject_reason) if pd.notna(r.reject_reason) and str(r.reject_reason).strip() else "KEPT"
        ok, q, area, T, _ = mser_at(vol, cen, sp, verbose=verbose)
        agree = ("both keep" if (cur == "KEPT" and ok) else
                 "both drop" if (cur != "KEPT" and not ok) else
                 "MSER RECOVERS" if (cur != "KEPT" and ok) else
                 "MSER drops ours")
        print(f"  {r.max_diameter_mm:6.1f} mm  seed {r.seed_peak_hu:6.0f}  "
              f"hu_max {r.hu_max:6.0f}  ours={cur:20s} "
              f"mser={'accept' if ok else 'reject':6s} "
              f"var={q:6.3f} @ {T:5.0f} HU  -> {agree}")
        out.append({"study_id": sid, "max_diameter_mm": r.max_diameter_mm,
                    "seed_peak_hu": r.seed_peak_hu, "hu_max": r.hu_max,
                    "hu_mean": r.hu_mean, "volume_mm3": r.volume_mm3,
                    "current": cur, "mser_accept": bool(ok),
                    "mser_variation": None if q != q else round(float(q), 3),
                    "mser_area_vox": int(area),
                    "mser_threshold_hu": None if T != T else float(T),
                    "outcome": agree})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", nargs="+", required=True)
    ap.add_argument("--verbose", action="store_true",
                    help="print the full area-vs-threshold sweep per candidate")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cand = pd.read_csv(os.path.join(CSV, "candidates.csv"))
    rows = []
    for sid in a.studies:
        rows += run_study(sid, cand, a.verbose)
    if not rows:
        raise SystemExit("nothing scored")
    d = pd.DataFrame(rows)

    out = a.out or os.path.join(RUN, "csv", "mser_experiment.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    d.to_csv(out, index=False)

    print(f"\n{'='*78}\nSUMMARY  ({d.study_id.nunique()} studies, {len(d)} candidates)\n{'='*78}")
    print(d.outcome.value_counts().to_string())
    rec = d[d.outcome == "MSER RECOVERS"]
    lost = d[d.outcome == "MSER drops ours"]
    print(f"\nrecovered by MSER: {len(rec)}")
    if len(rec):
        print(rec[["study_id", "max_diameter_mm", "seed_peak_hu", "hu_max",
                   "mser_variation"]].to_string(index=False))
    print(f"\nOURS kept but MSER rejects: {len(lost)}   <- regression risk")
    if len(lost):
        print(lost[["study_id", "max_diameter_mm", "seed_peak_hu", "hu_max",
                    "mser_variation"]].to_string(index=False))
    # The paper's metric convention: extra detections per patient is the cost
    # side of any sensitivity gain, so report it alongside, never alone.
    n = d.study_id.nunique()
    print(f"\nEXTRA candidates admitted per patient: {len(rec)/n:.1f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
