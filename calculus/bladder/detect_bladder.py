"""Find calculi in the BLADDER. A compartment nothing has ever searched.

WHY THIS DID NOT EXIST
----------------------
The kidney detector searches the sinus-closed kidney plus a 3 mm capsular cuff.
The ureteric detector searches a corridor that STOPS at the ureterovesical
junction. The bladder lumen is outside both, so a vesical calculus was never a
candidate -- not rejected, never considered.

Measured cost on the validation cohort:

    8676429   report: "a large vesical calculus measuring approximately
                       28 x 51 mm (AP x TR) with HU: 300 is present in the
                       right lumen of the bladder"
              ours:   nothing in the bladder. The distal end of the ureteric
                      corridor grazed its edge and reported a 22 mm "ureteric
                      calculus" at 336 HU -- a real stone, in the wrong organ,
                      at less than half its size.
    8676857   report: "a tiny subtle hyperdense foci is noted in the urinary
                       bladder, possible vesical calculus (microlith)"
              ours:   nothing.

The first of those is the dangerous shape: not a silent miss but a confident
mislabel, which reads as plausible.

WHY THE BLADDER IS EASIER THAN THE URETER
The ureter is a 3-5 mm tube we have to infer the course of, threading past bone,
bowel and calcified vessels. The bladder is a large, well-segmented, fluid-filled
bag. On a non-contrast scan its contents are urine at 0-20 HU, so ANY compact
object above the calcium-scoring floor is a stone until shown otherwise. There is
no corridor to guess and no vessel to confuse.

WHAT IS DIFFERENT HERE, AND WHY THE KIDNEY RULES DO NOT TRANSFER
  * NO UPPER SIZE BOUND. Vesical calculi grow unobstructed and routinely reach
    30-50 mm; this cohort has one at 51 mm. MAX_DIAM_MM = 22 deleting a real
    23 mm ureteric stone this morning is the standing reminder of what a cap
    fitted to a small sample does.
  * THE WALL MUST BE EXCLUDED. The bladder mask includes the detrusor, and the
    wall against perivesical fat is a partial-volume edge that thresholds like a
    thin stone. So the ROI is the mask ERODED by WALL_ERODE_MM.
  * DEPENDENT POSITION IS A CUE, NOT A RULE. Stones sit in the dependent part of
    the lumen, but "dependent" depends on how the patient is lying, and a stone
    in a bladder diverticulum is not dependent at all. Recorded, never used to
    reject.
  * A FOLEY BALLOON IS THE MIMIC. A catheter balloon sits in the lumen and its
    fill can be dense. It is round, thin-walled and hollow, so it is flagged by
    the same lucent-centre reasoning the phlebolith test uses -- flagged, not
    rejected, because there is no case in this cohort to validate a rejection.

MEASUREMENT reuses Part 1's engine unchanged -- crop, FWHM re-threshold, marching
cubes, partial-volume integration -- so a bladder stone's size and density are
produced by the same code, and the same phantom validation, as a renal one.

Usage:
    python -m calculus.bladder.detect_bladder --studies 8676429
    python -m calculus.bladder.detect_bladder            # every study on disk
"""
import argparse
import glob
import os
import traceback

import cc3d
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

from calculus.common.paths import CSV, NIFTI, SEG, ensure
from calculus.kidney import detect_stones as ds

# Thresholds are IMPORTED, not re-tuned. GROW_HU (130, the calcium-scoring
# floor) and SEED_HU (200) were fitted on 120 studies; a bladder stone is the
# same material as a renal one, and two definitions of "stone" in one report
# would be indefensible.
GROW_HU = ds.GROW_HU
SEED_HU = ds.SEED_HU

WALL_ERODE_MM = 3.0      # keep the lumen, drop the detrusor and its outer edge
MIN_DIAM_MM = 1.5        # same as the kidney: reports go down to ~1.1 mm
MIN_VOL_MM3 = 0.5
# NO MAX. See the module docstring.

# A Foley balloon is round, thin-walled and hollow. `lucency` measures how much
# darker the centre is than the shell; a solid stone is not darker inside.
BALLOON_LUCENCY_HU = -80.0
BALLOON_MIN_DIAM_MM = 12.0    # balloons are 10-30 mm; small stones are not

COLS = ["study_id", "stone_id", "max_diameter_mm", "volume_mm3",
        "dim_tr_mm", "dim_ap_mm", "dim_cc_mm", "hu_max", "hu_mean",
        "elongation", "fill_fraction", "dependent_frac", "lucency_hu",
        "wall_dist_mm", "measurement_flag", "review_flag",
        "reject_reason", "is_stone", "centroid_vox"]


def lucency(sub_vol, refined, spacing):
    """Centre minus shell, in HU. Negative means a hollow object.

    Same reasoning as the ureteric phlebolith cue: erode to the core, compare it
    with the rind. A calculus is solid, so its core is at least as bright as its
    edge; a balloon is not.
    """
    if refined.sum() < 8:
        return float("nan")
    d = ds.dist_mm(~refined, spacing)          # distance INTO the object
    core = d >= max(1.0, 0.4 * float(d.max()))
    shell = refined & ~core
    if not core.any() or not shell.any():
        return float("nan")
    return float(sub_vol[core].mean() - sub_vol[shell].mean())


def analyse(study_id, verbose=False, denoise=True):
    sid = str(study_id)
    nii = os.path.join(NIFTI, f"{sid}.nii.gz")
    if not os.path.exists(nii):
        return [], {"study_id": sid, "error": "no nifti"}
    img = nib.load(nii)
    vol = np.asanyarray(img.dataobj).astype(np.float32)
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    voxel_mm3 = float(np.prod(spacing))

    bp = os.path.join(SEG, sid, "urinary_bladder.nii.gz")
    if not os.path.exists(bp):
        return [], {"study_id": sid, "error": "no bladder mask"}
    bladder = np.asanyarray(nib.load(bp).dataobj) > 0
    if not bladder.any():
        return [], {"study_id": sid, "error": "empty bladder mask"}

    summ = {"study_id": sid,
            "spacing_mm": "x".join(f"{s:.3f}" for s in spacing),
            "bladder_ml": round(float(bladder.sum()) * voxel_mm3 / 1000.0, 1)}

    # ---- the lumen: erode the wall away -------------------------------------
    # Cropped, as everywhere else: a full-volume distance transform on a 512^3
    # volume costs tens of seconds to answer a question about one organ.
    box = ds._pad_box(bladder, WALL_ERODE_MM + 2.0, spacing, vol.shape)
    wall_d = ds.dist_mm(~bladder[box], spacing)      # distance inward from wall
    lumen_sub = wall_d > WALL_ERODE_MM
    if not lumen_sub.any():
        return [], {**summ, "error": "bladder too small to erode a lumen"}
    summ["lumen_ml"] = round(float(lumen_sub.sum()) * voxel_mm3 / 1000.0, 1)

    sub_vol = vol[box]
    det = np.clip(sub_vol, ds.CLIP_LOW, ds.CLIP_HIGH).astype(np.float32)
    rounds = 0
    for rounds in range(1, (ds.DENOISE_MAX_ROUNDS if denoise else 0) + 1):
        det = ds.denoise_ct(det, ds.DENOISE_ITERS)
        _, nc = cc3d.connected_components((det >= GROW_HU) & lumen_sub,
                                          connectivity=26, return_N=True)
        if nc < ds.DENOISE_TARGET_CC:
            break
    summ["denoise_rounds"] = rounds

    # urine reference: the lumen's own median. On a plain scan this should be
    # 0-20 HU; well above that means contrast, and a stone cannot be told from
    # opacified urine, so we abstain exactly as the kidney detector does.
    urine = float(np.median(sub_vol[lumen_sub]))
    summ["urine_median_hu"] = round(urine, 1)
    if urine > ds.KIDNEY_PLAIN_MAX_HU:
        return [], {**summ, "error": ("opacified urine "
                                      f"(lumen median {urine:.0f} HU) - "
                                      "not analysable for calculi")}

    lab, n = cc3d.connected_components((det >= GROW_HU) & lumen_sub,
                                       connectivity=26, return_N=True)
    if not n:
        return [], {**summ, "n_candidates": 0, "n_stones": 0}

    # hysteresis on the RAW volume, as Part 1 does: denoising drags peaks down
    # by tens of HU, so seeding on the filtered copy loses borderline stones
    stats = cc3d.statistics(lab)
    counts, bboxes = stats["voxel_counts"], stats["bounding_boxes"]
    keep = []
    for i in range(1, len(counts)):
        if not counts[i]:
            continue
        bb = bboxes[i]
        m = lab[bb] == i
        if float(sub_vol[bb][m].max()) >= SEED_HU:
            keep.append(i)
    summ["n_candidates_raw"] = int(n)
    if not keep:
        return [], {**summ, "n_candidates": 0, "n_stones": 0}

    # dependent direction: within the bladder, "down" is +axis1 (posterior) and
    # -axis2 (caudal) in the RAS volumes we build. Reported as a cue only.
    lum_idx = np.argwhere(lumen_sub)
    z_lo, z_hi = lum_idx[:, 2].min(), lum_idx[:, 2].max()

    rows, sid_n = [], 0
    for i in keep:
        bb = bboxes[i]
        comp_sub = lab[bb] == i
        nvox = int(comp_sub.sum())
        vmm3 = nvox * voxel_mm3
        if vmm3 < MIN_VOL_MM3:
            continue
        # measure with Part 1's engine, on a padded crop of the ORIGINAL volume
        margin = tuple(max(1, int(round(8.0 / s))) for s in spacing)
        full = np.zeros(vol.shape, bool)
        off = [box[a].start for a in range(3)]
        full[tuple(slice(off[a] + bb[a].start, off[a] + bb[a].stop)
                   for a in range(3))] = comp_sub
        sl = ds.crop_box(full, margin, vol.shape)
        refined, thr, peak, bg, pv = ds.fwhm_measure(vol, full, spacing, sl)
        shape = ds.shape_metrics(vol[sl], refined, spacing, thr) or {}
        dmax = shape.get("max_diameter_mm") or ds.max_diameter_mm(refined, spacing)
        hu_max = (float(vol[sl][refined].max()) if refined.any()
                  else float(vol[sl][comp_sub].max()))
        hu_mean = float(vol[sl][refined].mean()) if refined.any() else np.nan
        luc = lucency(vol[sl], refined, spacing)

        cen = np.array(ndimage.center_of_mass(comp_sub)) + \
            [off[a] + bb[a].start for a in range(3)]
        # COORDINATE SPACES. z_lo/z_hi come from argwhere(lumen_sub), which is in
        # CROPPED box coordinates; `cen` is in FULL volume coordinates. Mixing
        # them produced dependent_frac = -2.53 on 8676429, on a quantity defined
        # to lie in 0-1. Harmless here because this field is a cue and never a
        # verdict -- but a number outside its own range is exactly the kind of
        # thing that gets read as meaningful later.
        cen_box_z = cen[2] - off[2]
        dep = float((z_hi - cen_box_z) / max(z_hi - z_lo, 1))   # 1 = at the floor
        dep = min(max(dep, 0.0), 1.0)
        wd = float(wall_d[tuple(np.rint(
            np.array(ndimage.center_of_mass(comp_sub))
            + [bb[a].start for a in range(3)]).astype(int))])

        reason = ""
        if dmax < MIN_DIAM_MM:
            reason = "too_small"
        # BEAM-HARDENING STREAKS. Measured on the first full bladder run: 3 of 5
        # detections read 2601, 2816 and 3071 HU with volumes of 286-473 mm3
        # against calipers of 21-44 mm -- fill 0.011-0.058. Streaks from dense
        # bone or metal crossing the lumen: thin, and far denser than any stone.
        #
        # This is a rejection rather than a flag, and the distinction from
        # tubular_not_stone in the ureter matters. There, rejecting on shape
        # removed a real stone whose caliper was leak-inflated. Here the test is
        # DENSITY, and it is a physical impossibility rather than a fitted
        # bound: calcium oxalate monohydrate, the densest common calculus, peaks
        # near 1500-1700 HU. Nothing at 2600 HU is a stone. The two genuine
        # calculi in this cohort read 331 and 724 HU, nowhere near it.
        elif hu_max > ds.HU_IMPLAUSIBLE:
            reason = "not_calculus_density"

        review = []
        # a Foley balloon: large, and darker inside than at its edge
        if (dmax >= BALLOON_MIN_DIAM_MM and np.isfinite(luc)
                and luc < BALLOON_LUCENCY_HU):
            review.append("balloon_like")
        mf = ds.measurement_flags(pv, dmax, hu_max)

        if not reason:
            sid_n += 1
        rows.append({
            "study_id": sid, "stone_id": sid_n if not reason else None,
            "max_diameter_mm": round(float(dmax), 2),
            "volume_mm3": round(float(pv), 2),
            "dim_tr_mm": round(shape.get("dim_tr_mm", float("nan")), 2),
            "dim_ap_mm": round(shape.get("dim_ap_mm", float("nan")), 2),
            "dim_cc_mm": round(shape.get("dim_cc_mm", float("nan")), 2),
            "hu_max": round(hu_max), "hu_mean": round(hu_mean),
            "elongation": (round(float(shape["elongation"]), 3)
                           if "elongation" in shape else None),
            "fill_fraction": round(ds.fill_fraction(pv, dmax), 4),
            "dependent_frac": round(dep, 2),
            "lucency_hu": round(luc, 1) if np.isfinite(luc) else None,
            "wall_dist_mm": round(wd, 1),
            "measurement_flag": mf,
            "review_flag": ";".join(review),
            "reject_reason": reason, "is_stone": reason == "",
            "centroid_vox": ",".join(str(int(round(c))) for c in cen),
        })

    stones = [r for r in rows if r["is_stone"]]
    summ.update({"n_candidates": len(rows), "n_stones": len(stones),
                 "largest_stone_mm": round(max((r["max_diameter_mm"]
                                                for r in stones), default=0.0), 2)})
    if verbose:
        print(f"  {sid}: lumen {summ.get('lumen_ml')} mL, urine "
              f"{summ.get('urine_median_hu')} HU -> {len(rows)} candidates, "
              f"{len(stones)} stones", flush=True)
    return rows, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", nargs="*", default=None)
    ap.add_argument("--no-denoise", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    run = ensure()
    per = os.path.join(CSV, "per_study")
    os.makedirs(per, exist_ok=True)
    ids = a.studies or sorted(os.path.basename(f)[:-7]
                              for f in glob.glob(os.path.join(NIFTI, "*.nii.gz")))
    ids = [i for i in ids if os.path.isdir(os.path.join(SEG, i))]
    print(f"bladder: {len(ids)} studies\n")

    allrows, summaries = [], []
    for k, sid in enumerate(ids, 1):
        try:
            rows, summ = analyse(sid, verbose=a.verbose, denoise=not a.no_denoise)
        except Exception as e:                    # one study must not kill a run
            rows, summ = [], {"study_id": sid,
                              "error": f"{type(e).__name__}: {e}",
                              "traceback": traceback.format_exc()[-400:]}
        print(f"[{k}/{len(ids)}] {sid}: "
              + (summ.get("error") or f"{summ.get('n_stones', 0)} stone(s)"),
              flush=True)
        d = pd.DataFrame(rows, columns=COLS if not rows else None)
        d.to_csv(os.path.join(per, f"{sid}_bladder_candidates.csv"), index=False)
        pd.DataFrame([summ]).to_csv(
            os.path.join(per, f"{sid}_bladder_summary.csv"), index=False)
        allrows += rows
        summaries.append(summ)

    # aggregate, written even when empty so downstream readers keep working
    pd.DataFrame(allrows, columns=COLS if not allrows else None).to_csv(
        os.path.join(CSV, "bladder_candidates.csv"), index=False)
    pd.DataFrame(summaries).to_csv(
        os.path.join(CSV, "bladder_summary.csv"), index=False)
    print(f"\nwrote {os.path.join(CSV, 'bladder_candidates.csv')} "
          f"({len(allrows)} candidates)")


if __name__ == "__main__":
    main()
