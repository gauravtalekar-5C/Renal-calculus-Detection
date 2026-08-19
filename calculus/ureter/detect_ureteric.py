"""Find calculi in the URETER, locate them, and measure their distance to the UVJ.

This is Part 2. Part 1 (`detect_stones.py`) searches the kidney only, on
purpose: the whole-tract mode there used a straight kidney->bladder line as a
stand-in for the ureter and produced ~70 false positives on a single study.
This module replaces that line with an anatomical corridor
(`ureter_corridor.py`) and adds the rejection logic that the ureter needs and
the kidney does not.

WHAT MAKES THE URETER HARDER THAN THE KIDNEY
--------------------------------------------
Inside the kidney, "dense blob" is nearly sufficient -- there is no bone and
almost nothing else that is calcium-dense. Outside it, three things look
exactly like a ureteric stone on non-contrast CT:

  BONE. The sacrum, iliac wing, ischial spine and lumbar transverse processes
  all sit within centimetres of the ureter. Handled the way Part 1 handles
  ribs: a dense component larger than BONE_MIN_VOL_MM3 is bone, plus the
  anatomical masks, plus `split_bone_bridges` so a stone fused to bone by
  partial volume is separated rather than discarded with it.

  VASCULAR CALCIFICATION. The ureter crosses the common iliac artery at the
  pelvic brim -- calcified plaque there is millimetres from where a stone
  would be. Handled with the artery masks plus a margin.

  PHLEBOLITHS. Calcified pelvic vein thrombi. These are the hard ones: same
  density, same size, same neighbourhood, and no mask exists for them. They
  are THE classic mimic on CT KUB. Handled with four measured features
  (below), reported rather than silently applied.

THE PHLEBOLITH FEATURES, AND WHAT THEY ARE WORTH
------------------------------------------------
  rim_hu        Mean HU in a thin shell around the object. A ureteric stone is
                inside the ureter, so it is wrapped in ureteric wall -- soft
                tissue, roughly 30-60 HU. A phlebolith sits in pelvic fat,
                roughly -80 HU. This is the radiological "soft-tissue rim
                sign", and of the four it is the one with a real mechanism.
  off_path_mm   Distance from the expected ureter centreline. Phleboliths sit
                lateral and inferior to the ureter. Weakened by the fact that
                the centreline is itself an estimate.
  lucency       Centre HU minus rim-of-object HU. Phleboliths are lamellated
                and often have a lucent centre; stones are usually uniform.
  elongation    minor/major axis. Phleboliths are round. So are small stones,
                so this is the weakest of the four.

NONE OF THESE THRESHOLDS ARE VALIDATED. We have no phlebolith ground truth --
no report in this dataset says "this is a phlebolith". So the score is
COMPUTED AND REPORTED, and only the most clear-cut cases are auto-rejected.
Everything else goes out with its features attached for a human to judge.
Read `PHLEBOLITH_*` below before trusting any of it.

WHAT IS VALIDATED, AND HOW
--------------------------
Detection is scored against the 37 studies whose report describes a ureteric
calculus with a side and a location. The question asked is containment and
agreement, not Dice: does a surviving candidate exist on the reported side, in
the reported third of the ureter, at roughly the reported size?

Output goes to $CALCULUS_RUN (default run_ureter/), never overwriting Part 1.

Usage:
    CALCULUS_RUN=run_ureter ./venv/bin/python utils/detect_ureteric.py --workers 2
    CALCULUS_RUN=run_ureter ./venv/bin/python utils/detect_ureteric.py --studies 8563509
"""
import argparse
import glob
import os
import sys

import cc3d
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
# package root -> project root: calculus/<sub>/x.py is two levels down
ROOT = os.path.dirname(os.path.dirname(HERE))

from calculus.kidney import detect_stones as ds                          # noqa: E402
from calculus.ureter import ureter_corridor as uc                        # noqa: E402
from calculus.common.paths import CSV, NIFTI, OVERLAYS, SEG         # noqa: E402

# ---------------------------------------------------------------- thresholds
# Detection reuses Part 1's hysteresis, which was swept against report labels
# on 120 studies. Changing it here would mean two different definitions of
# "stone" in one report, so it is imported rather than re-tuned.
GROW_HU = ds.GROW_HU                # 130: outer extent of a candidate
SEED_HU = ds.SEED_HU                # 200: every real stone must reach this

MIN_VOL_MM3 = 0.5                   # below this is noise, not a stone
MIN_DIAM_MM = 1.0                   # the reports go down to 1.1 mm, so must we
# An upper bound is a real anatomical constraint in the URETER that does not
# apply in the kidney. The ureter is 3-5 mm and dilates to maybe 15 mm; a
# 30 mm staghorn fits in a renal pelvis but cannot sit in a ureter. The largest
# ureteric stone in our 37 reports is 16 mm. Anything bigger is bone, bowel
# content or a mis-merged component, so it is rejected rather than reported.
MAX_DIAM_MM = 22.0

# DENSITY FLOOR, and it is the single most powerful filter in this module.
#
# Measured on 5 studies with a report-confirmed ureteric stone (222 survivors
# before this rule, of which 5 were the real stones):
#
#     hu_max      TRUE stones   median 1434   p10 709
#                 FALSE         median  153   p10 138   p90 208
#
# The false positives cluster in a narrow 130-210 HU band -- barely over GROW_HU
# and just clearing SEED_HU. They are partial-volume fragments at bone and vessel
# edges, which is confirmed by their surroundings: rim_hu median 104 (dense)
# against 28 (soft tissue) for the real stones. And by shape: volume median
# 3.5 mm3 against 62.8, while their apparent DIAMETER is larger -- thin wisps,
# not compact objects.
#
#     hu_max >= 300 keeps 5/5 true stones and cuts 217 false -> 5.
#     44 false positives per study becomes 1.
#
# WHAT IT COSTS. Of 20 ureteric stones whose report states a density, four are
# under 300 HU: 106, 129, 150, 271. But 150 and 271 are RENAL, not ureteric, and
# 106 and 129 are already below GROW_HU = 130 -- they never become candidates at
# all, whatever happens here. So on this cohort the rule costs nothing
# measurable for ureteric stones, and the real sensitivity ceiling is GROW_HU.
#
# STILL FITTED ON FIVE POSITIVES. That is the same footing on which SEED_HU was
# once moved the wrong way. Validate on all 37 report-confirmed studies before
# treating this number as settled.
HU_FLOOR = 300.0

# Candidates are also RANKED, because each study has only one or two ureteric
# stones. Ordering by hu_max alone put the true stone first in 4 of 5 studies
# and second in the fifth. A composite of log(HU), log(volume) and off-path
# distance was tried and did WORSE (ranks 1,2,1,2,2), so the extra terms were
# noise. Rank is reported, never used to discard: the audit CSV keeps everything.
TOP_K_REPORTED = 2

BONE_MIN_VOL_MM3 = ds.BONE_MIN_VOL_MM3
BONE_MARGIN_MM = ds.BONE_MARGIN_MM
BONE_FRAC_REJECT = 0.5              # majority bone-adjacent -> bone

# Vascular calcification lies IN the arterial wall, so a candidate whose
# centre is within a few mm of the lumen mask is plaque. Kept tight: the
# ureter genuinely touches the iliac artery at the brim, and a stone there is
# real. 3 mm is Part 1's value.
VESSEL_MARGIN_MM = ds.VESSEL_MARGIN_MM

# --- phlebolith cues. UNVALIDATED. See the module docstring. ---------------
# Only the clearest cases are auto-rejected; the rest are flagged.
PHLEBOLITH_RIM_HU = -30.0           # shell this fatty -> not inside a ureter
PHLEBOLITH_OFF_PATH_MM = 16.0       # this far off course -> probably not ureter
PHLEBOLITH_AUTO_REJECT = 2          # cues needed before we drop it outright

# Shell used for the rim sign, in mm from the object surface. Thin enough to
# sample the ureteric wall rather than whatever is beyond it.
RIM_INNER_MM, RIM_OUTER_MM = 0.8, 2.5


def _sphere(mm, spacing):
    """Voxel radius tuple for a mm-radius ball at this spacing."""
    return tuple(max(1, int(round(mm / s))) for s in spacing)
# WHAT THIS FUNCTION DOES: converts a physical radius in millimetres into the
# per-axis voxel counts needed to cover it, so that a "2 mm shell" really is
# 2 mm on every axis even when the slices are thicker than the pixels.


def rim_stats(vol, comp, spacing, exclude):
    """Mean HU in a thin shell around a candidate -- the soft-tissue rim sign.

    A stone inside the ureter is surrounded by ureteric wall (soft tissue,
    ~30-60 HU). A phlebolith sits in pelvic fat (~-80 HU). Bone and vessel
    voxels are excluded from the shell so a candidate lying beside the sacrum
    does not read as "soft tissue rim" purely because cortex is dense.
    """
    d = ndimage.distance_transform_edt(~comp, sampling=spacing)
    shell = (d >= RIM_INNER_MM) & (d <= RIM_OUTER_MM) & ~exclude
    if not shell.any():
        return float("nan"), 0
    return float(np.median(vol[shell])), int(shell.sum())
# WHAT THIS FUNCTION DOES: measures what the candidate is sitting in, by taking
# the median density of a thin layer just outside it. Soft tissue means it is
# wrapped in a ureter wall; fat means it is loose in the pelvis and more likely
# a phlebolith.


def lucency(vol, comp, spacing):
    """Centre HU minus outer-object HU, in HU.

    Negative means a lucent centre, which is characteristic of a lamellated
    phlebolith. Returns nan for objects too small to have an inside.
    """
    d = ndimage.distance_transform_edt(comp, sampling=spacing)
    if d.max() < 1.2:                       # no interior to speak of
        return float("nan")
    core = d >= d.max() * 0.6               # innermost part
    outer = comp & (d < d.max() * 0.4)      # the object's own outer layer
    if not core.any() or not outer.any():
        return float("nan")
    return float(np.median(vol[core]) - np.median(vol[outer]))
# WHAT THIS FUNCTION DOES: compares the density at the centre of the object
# with the density of its outer layer. A calculus is usually solid all through;
# a phlebolith often has a darker middle, so a clearly negative value is a hint
# that this is a vein calcification rather than a stone.


def analyse_ureter(study_id, verbose=False, denoise=True):
    """One study: find, measure and locate every candidate in the ureter."""
    sid = str(study_id)
    nii = os.path.join(NIFTI, f"{sid}.nii.gz")
    if not os.path.exists(nii):
        return [], {"study_id": sid, "error": "no nifti"}

    img = nib.load(nii)
    vol = np.asanyarray(img.dataobj).astype(np.float32)
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    voxel_mm3 = float(np.prod(spacing))
    masks = ds.load_masks(sid)

    summ = {"study_id": sid, "spacing_mm": "x".join(f"{s:.3f}" for s in spacing),
            "voxel_mm3": round(voxel_mm3, 4), "n_masks": len(masks)}

    # ---- 0. orientation. Every 'medial'/'posterior' below assumes it. -----
    ori = uc.orientation_ok(masks)
    summ["orientation_ok"] = ori
    if ori is False:
        return [], {**summ, "error": "unexpected image orientation"}

    # ---- 1. the corridor -------------------------------------------------
    sides = uc.build(masks, vol.shape, spacing)
    if not sides:
        return [], {**summ, "error": "no corridor (kidney or bladder missing)"}
    bounds = uc.zone_bounds(masks)
    summ["sides_with_corridor"] = ",".join(sorted(sides))
    summ["sacrum_landmarks"] = bool(bounds)
    summ["iliac_real"] = ",".join(f"{s}={sides[s]['iliac_real']}" for s in sides)

    # kidney parenchyma is Part 1's territory; exclude it plus a small cuff so
    # a renal-pelvic stone is not reported twice, once by each pipeline
    kidney = np.zeros(vol.shape, bool)
    for k in ("kidney_left", "kidney_right"):
        if k in masks:
            kidney |= masks[k]
    kidney_out = ndimage.binary_dilation(kidney, iterations=2) if kidney.any() \
        else kidney

    corridor = np.zeros(vol.shape, bool)
    for s in sides.values():
        corridor |= s["corridor"]
    corridor &= ~kidney_out

    # ---- 2. bone and vessel maps ----------------------------------------
    # Same rule as Part 1: a dense component bigger than BONE_MIN_VOL_MM3 is
    # bone. Part 1 then subtracts the kidney, because nothing inside a kidney
    # is bone. We CANNOT do that here -- the sacrum genuinely lies inside the
    # corridor -- so instead we lean on split_bone_bridges, which separates a
    # stone fused to bone rather than discarding the pair.
    dense = vol >= ds.BONE_HU
    dlab, dn = cc3d.connected_components(dense, connectivity=26, return_N=True)
    big = np.zeros(dn + 1, bool)
    big[1:] = (cc3d.statistics(dlab)["voxel_counts"][1:] * voxel_mm3) >= BONE_MIN_VOL_MM3
    bone_seed = big[dlab]
    for b in ("vertebrae_L1", "vertebrae_L5", "sacrum", "hip_left", "hip_right"):
        if b in masks:
            bone_seed |= masks[b]
    bone_dist = ds.dist_mm(bone_seed, spacing)

    vessel = np.zeros(vol.shape, bool)
    for v in ("aorta", "inferior_vena_cava", "iliac_artery_left",
              "iliac_artery_right"):
        if v in masks:
            vessel |= masks[v]
    vessel_dist = ds.dist_mm(vessel, spacing) if vessel.any() else None

    # CARVE BONE OUT OF THE CORRIDOR. Without this the tube clips the lumbar
    # transverse processes, the sacral ala and the iliac wing, and TRABECULAR
    # bone sits at 130-350 HU -- above GROW_HU. The first run produced 27 and
    # 43 "stones" per study, of which the largest were 55 mm objects at 150-350
    # HU: not stones at all, just the slice of vertebra the corridor happened
    # to intersect.
    #
    # Subtracted WITHOUT a margin, deliberately. A stone lying against the
    # sacrum is real and must survive; the partial-volume shell left at the
    # bone surface is thin, and `bone_frac` rejects it downstream. Adding a
    # cuff here would erase genuine peri-osseous stones instead.
    corridor &= ~bone_seed
    summ["corridor_mL"] = round(float(corridor.sum()) * voxel_mm3 / 1000.0, 1)
    if not corridor.any():
        return [], {**summ, "error": "empty corridor"}

    # ---- 3. candidates ---------------------------------------------------
    # Crop to the corridor before denoising. Curvature anisotropic diffusion on
    # a 512x512x524 volume is minutes of CPU; on the corridor's bounding box it
    # is seconds, and every voxel we care about is inside it.
    bb = np.nonzero(corridor)
    crop = tuple(slice(max(0, int(a.min()) - int(np.ceil(6.0 / sp))),
                       min(nn, int(a.max()) + int(np.ceil(6.0 / sp)) + 1))
                 for a, nn, sp in zip(bb, vol.shape, spacing))
    sub_roi = corridor[crop]
    raw_sub = vol[crop]
    det_sub = np.clip(raw_sub, ds.CLIP_LOW, ds.CLIP_HIGH).astype(np.float32)
    # Same adaptive stopping rule as Part 1 (Elton et al.): filter, recount
    # blobs, filter again, until the count drops below 200. A fixed iteration
    # count under-denoises a noisy scan and over-denoises a clean one.
    denoise_rounds = 0
    for denoise_rounds in range(1, (ds.DENOISE_MAX_ROUNDS if denoise else 0) + 1):
        det_sub = ds.denoise_ct(det_sub, ds.DENOISE_ITERS)
        _, nc = cc3d.connected_components((det_sub >= GROW_HU) & sub_roi,
                                          connectivity=26, return_N=True)
        if nc < ds.DENOISE_TARGET_CC:
            break
    summ["denoise_rounds"] = denoise_rounds

    sub_lab, n = cc3d.connected_components((det_sub >= GROW_HU) & sub_roi,
                                           connectivity=26, return_N=True)
    if not n:
        return [], {**summ, "n_candidates": 0, "n_stones": 0}
    # Hysteresis the way Part 1 does it: extent comes from the DENOISED image
    # at GROW_HU, but the seed test uses the ORIGINAL. Denoising pulls peaks
    # down by tens of HU, so testing the seed on the filtered copy would quietly
    # drop borderline stones near 200 HU.
    labels = np.zeros(vol.shape, np.int32)
    labels[crop] = sub_lab
    peak_of = {}
    keep = np.zeros(n + 1, bool)
    for i in range(1, n + 1):
        m = sub_lab == i
        if not m.any():
            continue
        pk = float(raw_sub[m].max())
        if pk >= SEED_HU:
            keep[i] = True
            peak_of[i] = pk
    labels = np.where(keep[labels], labels, 0)
    if not labels.any():
        return [], {**summ, "n_candidates": 0, "n_stones": 0}
    labels, peak_of, n, n_bridges = ds.split_bone_bridges(
        labels, int(labels.max()), peak_of, bone_dist, vol, voxel_mm3)
    summ["n_bone_bridges_split"] = n_bridges

    rows = []
    for i in np.unique(labels):
        if not i:
            continue
        comp = labels == i
        nvox = int(comp.sum())
        vol_mm3 = nvox * voxel_mm3
        if vol_mm3 < MIN_VOL_MM3:
            continue
        idx = np.argwhere(comp)
        cen = idx.mean(axis=0)

        # which side's corridor is this in, and how far off its centreline?
        best_side, off = None, np.inf
        for sname, sd in sides.items():
            v = float(sd["dist_to_path"][tuple(np.rint(cen).astype(int))])
            if v < off:
                best_side, off = sname, v
        sd = sides[best_side]

        # position along the tract, via the nearest point on the centreline
        pv = (sd["path"] - cen) * np.asarray(spacing)
        k = int(np.argmin(np.linalg.norm(pv, axis=1)))
        along_from_puj = float(sd["arclen"][k])
        along_to_uvj = float(sd["arclen"][-1] - sd["arclen"][k])
        straight_to_uvj = float(np.linalg.norm((cen - sd["uvj"]) * np.asarray(spacing)))
        straight_to_puj = float(np.linalg.norm((cen - sd["puj"]) * np.asarray(spacing)))

        # measurement, on the ORIGINAL volume, using Part 1's engine.
        # crop_box wants a PER-AXIS padding in voxels, so 8 mm has to be
        # converted separately for each axis -- slices are usually thicker
        # than pixels, and a scalar would pad anisotropically.
        margin = tuple(max(1, int(round(8.0 / s))) for s in spacing)
        sl = ds.crop_box(comp, margin, vol.shape)
        refined, thr, peak, bg, pv_mm3 = ds.fwhm_measure(vol, comp, spacing, sl)
        full = np.zeros(vol.shape, bool)
        full[sl] = refined
        shape = ds.shape_metrics(vol[sl], refined, spacing, thr) or {}
        dmax = shape.get("max_diameter_mm") or ds.max_diameter_mm(full, spacing)

        # rejection features
        bone_frac = float((bone_dist[comp] < BONE_MARGIN_MM).mean())
        vessel_mm = (float(vessel_dist[tuple(np.rint(cen).astype(int))])
                     if vessel_dist is not None else float("nan"))
        rim, rim_n = rim_stats(vol[sl], refined, spacing,
                               (bone_seed | vessel)[sl])
        luc = lucency(vol[sl], refined, spacing)
        elong = shape.get("elongation", float("nan"))

        # phlebolith cues, counted not thresholded into a single verdict
        cues = []
        if np.isfinite(rim) and rim < PHLEBOLITH_RIM_HU:
            cues.append("fatty_rim")
        if off > PHLEBOLITH_OFF_PATH_MM:
            cues.append("off_path")
        if np.isfinite(luc) and luc < -60:
            cues.append("lucent_centre")
        if np.isfinite(elong) and elong > 0.85:
            cues.append("round")

        reason = ""
        if bone_frac > BONE_FRAC_REJECT:
            reason = "bone"
        elif np.isfinite(vessel_mm) and vessel_mm < VESSEL_MARGIN_MM:
            reason = "vascular_calcification"
        elif dmax < MIN_DIAM_MM:
            reason = "too_small"
        elif dmax > MAX_DIAM_MM:
            reason = "too_large_for_ureter"
        elif float(vol[comp].max()) < HU_FLOOR:
            reason = "below_hu_floor"
        elif len(cues) >= PHLEBOLITH_AUTO_REJECT:
            reason = "phlebolith_likely"

        zone = uc.classify_zone(int(round(cen[2])), bounds, along_to_uvj)
        rows.append({
            "study_id": sid, "candidate_id": int(i), "side": best_side,
            "zone": zone,
            "dist_to_uvj_along_mm": round(along_to_uvj, 1),
            "dist_to_uvj_straight_mm": round(straight_to_uvj, 1),
            "dist_from_puj_along_mm": round(along_from_puj, 1),
            "dist_from_puj_straight_mm": round(straight_to_puj, 1),
            "max_diameter_mm": round(float(dmax), 2),
            "volume_mm3": round(float(pv_mm3), 2),
            "dim_tr_mm": round(shape.get("dim_tr_mm", float("nan")), 2),
            "dim_ap_mm": round(shape.get("dim_ap_mm", float("nan")), 2),
            "dim_cc_mm": round(shape.get("dim_cc_mm", float("nan")), 2),
            "hu_max": round(float(vol[comp].max())),
            "hu_mean": round(float(vol[full].mean()) if full.any() else np.nan),
            "off_path_mm": round(off, 1),
            "rim_hu": round(rim, 1) if np.isfinite(rim) else None,
            "rim_voxels": rim_n,
            "lucency_hu": round(luc, 1) if np.isfinite(luc) else None,
            "elongation": round(float(elong), 3) if np.isfinite(elong) else None,
            "bone_frac": round(bone_frac, 2),
            "vessel_dist_mm": round(vessel_mm, 1) if np.isfinite(vessel_mm) else None,
            "phlebolith_cues": ";".join(cues),
            "n_phlebolith_cues": len(cues),
            "reject_reason": reason,
            "is_stone": reason == "",
            "centroid_vox": ",".join(str(int(round(c))) for c in cen),
        })

    # rank the survivors by density, highest first, per side. Reported not
    # applied -- rank 1-2 is what goes in the report, the rest stay auditable.
    stones = [r for r in rows if r["is_stone"]]
    for side in ("left", "right"):
        ss = sorted([r for r in stones if r["side"] == side],
                    key=lambda r: -r["hu_max"])
        for k, r in enumerate(ss, 1):
            r["hu_rank_side"] = k
            r["report_this"] = k <= TOP_K_REPORTED
    summ.update({
        "n_candidates": len(rows),
        "n_stones": len(stones),
        "n_rejected_bone": sum(r["reject_reason"] == "bone" for r in rows),
        "n_rejected_vessel": sum(r["reject_reason"] == "vascular_calcification"
                                 for r in rows),
        "n_rejected_phlebolith": sum(r["reject_reason"] == "phlebolith_likely"
                                     for r in rows),
        "n_rejected_small": sum(r["reject_reason"] == "too_small" for r in rows),
        "n_rejected_large": sum(r["reject_reason"] == "too_large_for_ureter"
                                for r in rows),
        "largest_stone_mm": round(max((r["max_diameter_mm"] for r in stones),
                                      default=0.0), 2),
        "sides_with_stone": ",".join(sorted({r["side"] for r in stones})),
    })
    if verbose:
        print(f"  {sid}: {len(rows)} candidates -> {len(stones)} stones "
              f"(bone {summ['n_rejected_bone']}, vessel "
              f"{summ['n_rejected_vessel']}, phlebolith "
              f"{summ['n_rejected_phlebolith']})", flush=True)
    return rows, summ
# WHAT THIS FUNCTION DOES: the whole Part 2 pipeline for one study. It builds
# the ureter corridor from the organ masks, finds every dense blob inside it,
# measures each one with the same engine Part 1 uses, works out where along the
# tract it sits and how far it is from the UVJ, then decides whether it is a
# stone or one of the three mimics -- bone, vascular calcification, phlebolith.


def _one(sid, kw):
    try:
        return sid, analyse_ureter(sid, **kw)
    except Exception as e:                  # one bad study must not kill a run
        import traceback
        return sid, ([], {"study_id": sid, "error": f"{type(e).__name__}: {e}",
                          "traceback": traceback.format_exc()[-500:]})
# WHAT THIS FUNCTION DOES: a picklable wrapper so studies can be analysed in
# parallel worker processes, turning any crash into an error row instead of
# taking the whole run down with it.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", nargs="*", default=None,
                    help="study ids; default = every nifti on disk")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-denoise", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    ids = a.studies or sorted(
        os.path.basename(f).split(".")[0]
        for f in glob.glob(os.path.join(NIFTI, "*.nii.gz")))
    per = os.path.join(CSV, "per_study")
    os.makedirs(per, exist_ok=True)
    os.makedirs(OVERLAYS, exist_ok=True)
    if not a.overwrite:
        done = {s for s in ids
                if os.path.exists(os.path.join(per, f"{s}_ureter_summary.csv"))}
        if done:
            print(f"resuming: {len(done)} already done, {len(ids)-len(done)} left")
            ids = [s for s in ids if s not in done]

    kw = {"verbose": a.verbose, "denoise": not a.no_denoise}

    def write(sid, rows, summ):
        # candidates first, summary LAST -- the summary is what resume checks,
        # so a crash between the two re-runs the study instead of skipping it
        if rows:
            pd.DataFrame(rows).to_csv(
                os.path.join(per, f"{sid}_ureter_candidates.csv"), index=False)
        pd.DataFrame([summ]).to_csv(
            os.path.join(per, f"{sid}_ureter_summary.csv"), index=False)

    if a.workers > 1 and len(ids) > 1:
        import concurrent.futures as cf
        import multiprocessing as mp
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
            os.environ[v] = "1"
        # ProcessPoolExecutor, not multiprocessing.Pool: a broken worker pipe
        # raises here instead of blocking forever in .get(). Part 1 hung at
        # study 27 that way and lost 45 minutes of compute.
        with cf.ProcessPoolExecutor(max_workers=a.workers,
                                    mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(_one, s, kw): s for s in ids}
            for i, f in enumerate(cf.as_completed(futs), 1):
                sid, (rows, summ) = f.result()
                write(sid, rows, summ)
                print(f"[{i}/{len(ids)}] {sid}: {summ.get('n_stones', '-')} stones"
                      f"{'  ERROR: ' + summ['error'] if summ.get('error') else ''}",
                      flush=True)
    else:
        for i, sid in enumerate(ids, 1):
            sid, (rows, summ) = _one(sid, kw)
            write(sid, rows, summ)
            print(f"[{i}/{len(ids)}] {sid}: {summ.get('n_stones', '-')} stones"
                  f"{'  ERROR: ' + summ['error'] if summ.get('error') else ''}",
                  flush=True)

    # gather everything on disk, including partial earlier runs
    cand, summs = [], []
    for f in sorted(glob.glob(os.path.join(per, "*_ureter_candidates.csv"))):
        cand += pd.read_csv(f).to_dict("records")
    for f in sorted(glob.glob(os.path.join(per, "*_ureter_summary.csv"))):
        summs += pd.read_csv(f).to_dict("records")
    if cand:
        pd.DataFrame(cand).to_csv(os.path.join(CSV, "ureter_candidates.csv"),
                                  index=False)
    if summs:
        pd.DataFrame(summs).to_csv(os.path.join(CSV, "ureter_summary.csv"),
                                   index=False)
    print(f"\nwrote {CSV}/ureter_candidates.csv  ({len(cand)} candidates)")
    print(f"wrote {CSV}/ureter_summary.csv    ({len(summs)} studies)")


if __name__ == "__main__":
    main()
