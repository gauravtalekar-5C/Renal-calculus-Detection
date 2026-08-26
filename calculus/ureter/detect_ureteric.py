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
from calculus.ureter import ureter_corridor as uc
from calculus.ureter import vertebral_level as vl                        # noqa: E402
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
# apply in the kidney: the ureter is 3-5 mm and dilates to maybe 15 mm, so a
# very large object there is more likely bone, bowel content or a mis-merged
# component than a stone.
#
# BUT THE OLD VALUE DELETED A REAL STONE, and it is worth being precise about
# how. It was 22.0, chosen because "the largest ureteric stone in our 37 reports
# is 16 mm" -- a bound fitted to the range of a small sample. On validation case
# 8659576 the report describes an obstructing 21 mm calculus at 1466 HU in the
# right mid ureter causing SEVERE hydronephrosis. We detected it cleanly:
# 23.24 mm, 1561 HU, bone_frac 0.00, 7.5 mm off the centreline, correct side,
# correct zone. It was then discarded for exceeding the cap by 1.2 mm.
#
# That is the most clinically urgent stone in the whole validation set, and the
# report said "0 calculi on the right". A cap fitted to a small sample fails on
# the first case outside it, and it fails SILENTLY -- nothing in the output says
# "found and discarded".
#
# 8664459's report also describes a 26 x 12 x 8 mm distal ureteric calculus, so
# 26 mm is not even the ceiling in this cohort of 17. Set well above anything
# anatomically plausible; bone and bowel are already caught by bone_frac and the
# vessel test, which is where that work belongs.
MAX_DIAM_MM = 40.0

# ...and keep the SIGNAL without the deletion. Anything above this is unusual
# for a ureter and worth a human look, so it is marked for review and still
# reported, rather than removed from the record.
LARGE_FOR_URETER_MM = 20.0

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
#
# THE CAP IS OFF, and the rank column is now purely informational.
#
# It used to be 2, and that was defensible when it was set: ureteric precision
# was 53.8%, so roughly half of what we reported was not a stone, and printing
# only the two densest per side was the cheapest way to stop a radiologist
# reading a page of noise. It suppressed false positives by suppressing output.
#
# The bone/L2-L4 mask fix, the refined-mask HU fix and the touching-stone
# splitter removed the CAUSES of those false positives -- on 8677121, 214
# candidates now resolve to 3 accepted with zero false positives. At that point
# the cap stopped protecting the report and started censoring it: the third
# left-sided stone, 428 HU against the radiologist's 425 HU, was detected,
# measured, written to the audit CSV -- and then withheld, because it ranked
# third. Three obstructing calculi in one ureter is a different clinical
# situation from two, so that omission is a clinical error, not a tidy-up.
#
# If precision regresses, the fix is the cascade that admitted the false
# positive, not a cap that hides however many happen to rank low. Set this to an
# int to re-cap; None reports every accepted stone.
TOP_K_REPORTED = None

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

# ONE CUE IS ENOUGH WHEN THE OBJECT IS ALSO A TUBE.
#
# Measured across the 17 validation studies: 10 of 42 accepted ureteric "stones"
# are BOTH elongated (< 0.45) AND sitting in fat (rim < -20 HU). Eight of them
# are in 8633709, bilateral and symmetric 32-36 mm above the UVJ, 11-22 mm long,
# 583-1025 HU, and 17-47 mm from any segmented vessel; two more are in 8675824,
# on the LEFT, in a study whose report describes only a RIGHT-sided stone. In a
# male pelvis that pattern is calcified vas deferens or pelvic phleboliths.
#
# The existing rule nearly caught them: `fatty_rim` fired on 6 of the 8, but
# rejection needs two cues and `off_path` did not fire -- they sat 11-15 mm off
# the centreline against a 16 mm threshold. One cue short, every time.
#
# The discriminating fact is not just the fat, it is the fat AROUND A TUBE. A
# calculus inside a ureter is wrapped in ureteric wall, soft tissue at +30 to
# +60 HU, and it is compact. These are wrapped in fat and shaped like a duct.
# Requiring BOTH conditions is what keeps this off real stones: 8674625's
# genuine 3.3 mm distal ureteric calculus is elongated (0.297) but has a rim of
# +38 HU, and every confirmed VUJ stone in the cohort has a soft-tissue rim.
MIMIC_ELONGATION = 0.45     # tube-shaped rather than lump-shaped
MIMIC_RIM_HU = -20.0        # lying in fat, not in a ureteric wall

# Shell used for the rim sign, in mm from the object surface. Thin enough to
# sample the ureteric wall rather than whatever is beyond it.
RIM_INNER_MM, RIM_OUTER_MM = 0.8, 2.5


# --- URETERIC STENT ---------------------------------------------------------
# NO VALIDATION DATA EXISTS FOR THIS RULE. Read that before trusting it.
#
# The validation cohort was built to include a "stent in situ" case and does
# not contain one. 8633709 was selected by text-matching "stent" in the report
# and the match was "CBD stent in situ with distal end in the second part of the
# duodenum" -- a BILIARY stent, in the duodenum, nothing to do with the urinary
# tract. The one genuine case, 8399313 ("Right-sided DJ ureteric stent is seen
# in situ"), is 42 days old and past the API's ~33-day retention window, so it
# could not be downloaded.
#
# So this rule is written from the anatomy and physics of the device, and it has
# NEVER been tested against a positive example. It therefore only ever FLAGS.
# It must not reject anything until a stent case has been seen, because a rule
# fitted to zero examples deleting findings is exactly how MAX_DIAM_MM = 22 came
# to remove a real 23.2 mm obstructing calculus.
#
# WHAT A DOUBLE-J STENT IS
#   a polyurethane tube with barium filler, 24-30 CM long, 1.5-2.5 mm bore
#   proximal end coils in the RENAL PELVIS, distal end coils in the BLADDER,
#   the middle lies inside the URETER LUMEN
#   it reads 300-1000+ HU, squarely in stone territory -- density cannot help
#
# WHAT DISTINGUISHES IT FROM A STONE
#   a stone is FOCAL and COMPACT      3-30 mm, elongation typically > 0.4
#   a stent is LONG and TUBULAR       200-300 mm, elongation ~0.1-0.3, and its
#                                     length is 20-30x the cube root of its
#                                     volume because it is a thin hollow tube
#
# AN APPROACH THAT DID NOT WORK, recorded so it is not retried: asking whether
# dense material is present at every point along the ureteric CORRIDOR. Measured
# on 8633709 the longest continuous dense run was 15-25 mm, against 15-30 mm for
# plain stone studies -- no separation at all. The corridor centreline is a
# geometric guess running 6-15 mm off the true ureter, so continuity measured
# along it does not measure continuity along the ureter. Any future stent work
# should measure the OBJECT, which is what this does.
STENT_MIN_LEN_MM = 60.0        # far below a real stent's 240-300 mm, because a
                               # stent fused to bone or clipped by the corridor
                               # arrives in pieces; a stone never reaches this
STENT_MAX_ELONGATION = 0.35    # minor/major axis: a tube, not a lump
STENT_MIN_ASPECT = 8.0         # length / volume^(1/3); a stone sits at 1-3


def stent_like(dmax_mm, volume_mm3, elongation):
    """Does this object look like a segment of a ureteric stent? FLAG ONLY.

    All three conditions must hold, because each alone has a real counterexample
    in the cohort: an impacted 23 mm ureteric calculus is long, a calcified vas
    deferens is elongated, and a thin partial-volume rind at a bone edge has a
    high aspect ratio. Requiring all three is what keeps this from touching the
    stones we currently get right.

    Returns a reason string, or "" -- never a rejection. See the block comment
    above for why this cannot reject until a stent case has been seen.
    """
    if not (np.isfinite(dmax_mm) and np.isfinite(volume_mm3) and volume_mm3 > 0):
        return ""
    if dmax_mm < STENT_MIN_LEN_MM:
        return ""
    if not np.isfinite(elongation) or elongation > STENT_MAX_ELONGATION:
        return ""
    if dmax_mm / max(volume_mm3 ** (1.0 / 3.0), 1e-6) < STENT_MIN_ASPECT:
        return ""
    return "stent_like"
# WHAT THIS FUNCTION DOES: marks an object that is far too long, far too thin
# and far too hollow to be a calculus, which is what a length of ureteric stent
# looks like. It only ever adds a review flag.


# --- THE VERDICT, AS PURE FUNCTIONS -----------------------------------------
# WHY THESE ARE FUNCTIONS AND NOT INLINE CODE
#
# The rejection precedence used to be an if/elif chain written inline in the
# candidate loop. Adding a flag-only test in the middle of it silently
# reattached `elif hu_max < HU_FLOOR` to the new `if`, so the 300 HU density
# floor only ran on candidates that were already rejected -- i.e. never. On
# 8664459 that admitted five "calculi" at 156-293 HU, and nothing failed: no
# exception, no test, just wrong output that looked plausible.
#
# Inline branching in a 200-line loop cannot be unit tested, so every edit to it
# is a gamble. These functions are the single source of truth for the verdict,
# they are covered by tests that assert the PRECEDENCE and not merely the
# outcome, and the loop below only supplies measurements.
#
# The three tiers exist for speed: tier 0 is two array lookups, tier 1 needs the
# FWHM measurement, tier 2 needs the expensive descriptors. On 8677121, 197 of
# 214 candidates are settled by tier 0 alone. Splitting the verdict across three
# functions preserves that while keeping each one testable.


def verdict_cheap(bone_frac, vessel_mm):
    """Tier 0: what two precomputed distance maps alone can decide.

    Returns a reject reason, or "" to continue to tier 1.
    """
    if bone_frac > BONE_FRAC_REJECT:
        return "bone"
    if np.isfinite(vessel_mm) and vessel_mm < VESSEL_MARGIN_MM:
        return "vascular_calcification"
    return ""


def verdict_measured(dmax, hu_max, volume_mm3=None):
    """Tier 1: size and density, once the object has actually been measured.

    Returns (reason, review_flag). `reason` rejects; `review_flag` never does.

    ORDER MATTERS AND IS TESTED. The density floor comes BEFORE the
    large-for-ureter flag, so a large object that is not dense is still
    rejected -- putting the flag first is precisely the bug described above.

    THE COMPACTNESS TEST, and why it applies only to LARGE objects.

    Raising MAX_DIAM_MM from 22 to 40 recovered a real 21 mm obstructing stone
    on 8659576 that the old cap had deleted. It also admitted large tubular
    structures -- almost certainly pelvic vessels outside TotalSegmentator's
    mask coverage -- in four studies. So the cap was doing two jobs at once, and
    removing it kept the good one and lost the other.

    Compactness restores the lost one without restoring the harm. Measured on
    every accepted ureteric object over 20 mm across the 18 validation studies:

        six false positives      fill 0.004 - 0.023
        the one real 21 mm stone fill 0.145

    A 6x gap. FILL_SUSPECT = 0.05 sits in it.

    It is applied ONLY above LARGE_FOR_URETER_MM, deliberately. Small stones can
    legitimately have a poor fill -- a 3 mm stone spans four voxels and its
    caliper is quantised -- and rejecting on that would delete microliths. A
    30 mm strand holding 200 mm3 is a different claim entirely.
    """
    if dmax < MIN_DIAM_MM:
        return "too_small", ""
    if dmax > MAX_DIAM_MM:
        return "too_large_for_ureter", ""
    if hu_max < HU_FLOOR:
        return "below_hu_floor", ""
    if dmax > LARGE_FOR_URETER_MM:
        f = ds.fill_fraction(volume_mm3, dmax) if volume_mm3 is not None \
            else float("nan")
        if np.isfinite(f) and f < ds.FILL_SUSPECT:
            return "tubular_not_stone", ""
        return "", "large_for_ureter"
    return "", ""


def verdict_mimic(cues, tube_in_fat):
    """Tier 2: the mimics. Returns a reject reason, or "".

    A tube lying in fat is rejected on that alone -- see MIMIC_ELONGATION. Any
    other two cues together also reject.
    """
    if tube_in_fat:
        return "extraureteric_calcification"
    if len(cues) >= PHLEBOLITH_AUTO_REJECT:
        return "phlebolith_likely"
    return ""
# WHAT THESE FUNCTIONS DO: hold the entire decision about whether a dense object
# in the ureteric corridor is a calculus, in one place, in the order the
# evidence should be weighed, so the order can be tested rather than trusted.


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
    # Vertebral spans, for the localisation the reports actually use ("at L4
    # level"). One pass over the vertebral masks; see vertebral_level for why
    # this replaces relying on the guessed UVJ landmark.
    vspans = vl.spans(masks)
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
    for b in ("vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4",
              "vertebrae_L5", "sacrum", "hip_left", "hip_right"):
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
    # ...then separate stones fused to EACH OTHER. Order matters: bone first,
    # because a stone welded to the sacrum must be freed from it before we ask
    # whether it is one stone or two. Both run BEFORE any rejection, so every
    # piece gets judged on its own evidence.
    #
    # On 8677121 the report describes three left ureteric calculi 6.3, 7.2 and
    # 9.7 mm above the UVJ; the 130 HU bridge between them made two blobs, so
    # one stone was invisible and the other two were reported oversized
    # (11.7 mm where the report says 9.1 x 15.4, 9.7 mm where it says 4.5 x 6.1).
    labels, peak_of, n, n_touch = ds.split_touching_stones(
        labels, int(labels.max()), peak_of, vol, spacing, voxel_mm3)
    summ["n_touching_split"] = n_touch

    # A candidate is judged in THREE TIERS, cheapest evidence first.
    #
    # WHY THE ORDER MATTERS. Measured on 8677121: 214 candidates, of which 197
    # die to `bone` and 14 to `below_hu_floor`. Only 3 survive. Before this
    # restructure every one of those 211 first paid for an FWHM re-threshold, a
    # marching-cubes surface, a convex hull, an O(N^2) caliper search, a rim
    # shell and a lucency profile -- and the test that actually killed them was
    # a single lookup in a precomputed distance map. The most expensive
    # candidates were the worst offenders: the sprawling 21-40 mm partial-volume
    # shells along the iliac wing and sacrum, all with bone_frac == 1.00.
    #
    # THE VERDICT CANNOT CHANGE. The tiers follow the rejection precedence
    # exactly -- bone, then vascular_calcification (tier 0), then too_small,
    # too_large_for_ureter, below_hu_floor (tier 1), then phlebolith_likely
    # (tier 2). Because the chain is an if/elif, an earlier reason always won
    # anyway; stopping early only skips work whose result was already discarded.
    #
    # WHAT DOES CHANGE, and it is a real cost: the DESCRIPTIVE columns of rows
    # rejected at tier 0. Without an FWHM mask there is no refined object to
    # measure, so those rows carry the coarse component's diameter, volume and
    # HU rather than the refined ones, and `measure_stage` records which. The
    # verdict, the side, the zone, the distances and bone_frac are unaffected.
    # Anything comparing hu_max against hu_max_component must filter on
    # measure_stage, because at tier 0 they are equal by construction.
    #
    # Hoisted out of the loop: `bone_seed | vessel` was a full-volume OR
    # rebuilt once per candidate.
    bone_or_vessel = bone_seed | vessel
    margin = tuple(max(1, int(round(8.0 / s))) for s in spacing)

    # ONE pass over the label image for every candidate's count, centroid and
    # bounding box. The loop used to open with
    #
    #     for i in np.unique(labels):        # sorts 102 M int32
    #         comp = labels == i             # reads 408 MB, writes 102 MB
    #         nvox = int(comp.sum())         # scans 102 M voxels
    #         idx = np.argwhere(comp)        # scans 102 M voxels
    #
    # i.e. three full sweeps of a 512x512x389 volume per candidate, 214 times,
    # to locate blobs of 10-500 voxels. Profiling the tiered detector put 215 of
    # its 299 seconds here -- while the geometry everyone assumes is expensive
    # (marching cubes, convex hull, FWHM) measured 0.8-2.1 MILLISECONDS a call.
    #
    # cc3d.statistics returns all three quantities for all labels at once, and
    # its bounding boxes come back as slice tuples ready to index with.
    #
    # AXIS ORDER IS VERIFIED, not assumed: the docstring says "x,y,z" and
    # "xmin,xmax,...", which would be a silent catastrophe if it meant anything
    # other than numpy axis 0,1,2 -- the centroid indexes dist_to_path and the
    # zone classifier, so a transposed centroid reads the wrong voxel and
    # mislabels the side. Checked on a deliberately asymmetric volume:
    # cc3d centroid == np.argwhere(mask).mean(axis=0) exactly, and the bounding
    # box slices == the mask's own extent.
    stats = cc3d.statistics(labels)
    counts, bboxes, centroids = (stats["voxel_counts"], stats["bounding_boxes"],
                                 stats["centroids"])

    rows = []
    # ascending label order, exactly as np.unique gave, so candidate_id order
    # and therefore CSV row order are unchanged
    for i in range(1, len(counts)):
        nvox = int(counts[i])
        if not nvox:
            continue                # a label erased by the seed test or a split
        vol_mm3 = nvox * voxel_mm3
        if vol_mm3 < MIN_VOL_MM3:
            continue
        cen = centroids[i]

        # bb.start == idx.min() and bb.stop == idx.max() + 1, so this is the
        # same box the argwhere version built, without the scan.
        bb = bboxes[i]
        sl = tuple(slice(max(0, int(bb[ax].start) - margin[ax]),
                         min(vol.shape[ax], int(bb[ax].stop) + margin[ax]))
                   for ax in range(3))
        sub_vol = vol[sl]                       # a view, not a copy
        comp_sub = labels[sl] == i              # small: the box, not the volume
        cen_i = tuple(np.rint(cen).astype(int))

        # which side's corridor is this in, and how far off its centreline?
        best_side, off = None, np.inf
        for sname, sd in sides.items():
            v = float(sd["dist_to_path"][cen_i])
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
        zone = uc.classify_zone(int(round(cen[2])), bounds, along_to_uvj)

        # ---- tier 0: two lookups in precomputed maps, no geometry ----------
        bone_frac = float((bone_dist[sl][comp_sub] < BONE_MARGIN_MM).mean())
        vessel_mm = (float(vessel_dist[cen_i])
                     if vessel_dist is not None else float("nan"))

        reason = verdict_cheap(bone_frac, vessel_mm)

        # defaults for everything the later tiers would have filled in
        stage = "coarse"
        review = ""            # non-fatal "a human should look at this" marker
        shape = {}
        refined = None
        dmax = ds.max_diameter_mm(comp_sub, spacing)
        pv_mm3 = vol_mm3
        hu_max = float(sub_vol[comp_sub].max())
        hu_mean = float(sub_vol[comp_sub].mean())
        rim, rim_n, luc, elong = float("nan"), 0, float("nan"), float("nan")
        cues = []

        if not reason:
            # ---- tier 1: measurement. Needed for the size and HU tests. ----
            stage = "refined"
            # fwhm_measure takes a full-volume mask and crops it itself. Its
            # signature is deliberately NOT changed -- doing that once before
            # broke this detector while appearing to make it 7x faster, because
            # the crash was mistaken for a speedup. Scattering the small box
            # into a zero volume costs ~30 ms and happens only for the handful
            # of candidates that get this far (17 of 214 on 8677121), where
            # `labels == i` would have read the whole label image again.
            comp = np.zeros(vol.shape, bool)
            comp[sl] = comp_sub
            refined, thr, peak, bg, pv_mm3 = ds.fwhm_measure(
                vol, comp, spacing, sl)
            shape = ds.shape_metrics(sub_vol, refined, spacing, thr) or {}
            # max_diameter_mm on the CROPPED mask: a caliper is translation
            # invariant, so cropping cannot change it, and the full-volume
            # version rescanned 137 M voxels to find the same points.
            dmax = shape.get("max_diameter_mm") or ds.max_diameter_mm(refined,
                                                                     spacing)
            # BOTH on the refined mask. They used to be measured on DIFFERENT
            # objects -- hu_max over the coarse >=GROW_HU component and hu_mean
            # over the FWHM-refined stone. On 8677121 that made hu_mean EXCEED
            # hu_max in 125 of 144 rows, which is arithmetically impossible for
            # one object and proves they described different ones. The coarse
            # component is threshold-shaped and can clip a stone's bright core;
            # the refined mask is the object we actually report, so it is the
            # one to measure. hu_max_component is kept for traceability.
            if refined.any():
                hu_max = float(sub_vol[refined].max())
                hu_mean = float(sub_vol[refined].mean())

            reason, review = verdict_measured(dmax, hu_max, pv_mm3)

            # FLAG-ONLY, strictly after the verdict so it cannot alter one.
            # stent_like has no validation data; see stent_like().
            if not reason:
                sl_flag = stent_like(dmax, pv_mm3,
                                     shape.get("elongation", float("nan")))
                if sl_flag:
                    review = (review + ";" + sl_flag) if review else sl_flag
                # physical-plausibility flags: see ds.measurement_flags
                mf = ds.measurement_flags(pv_mm3, dmax, hu_max)
                if mf:
                    review = (review + ";" + mf) if review else mf

        if not reason:
            # ---- tier 2: the phlebolith cues, the expensive descriptors -----
            stage = "full"
            rim, rim_n = rim_stats(sub_vol, refined, spacing, bone_or_vessel[sl])
            luc = lucency(sub_vol, refined, spacing)
            elong = shape.get("elongation", float("nan"))

            # counted, not thresholded into a single verdict
            if np.isfinite(rim) and rim < PHLEBOLITH_RIM_HU:
                cues.append("fatty_rim")
            if off > PHLEBOLITH_OFF_PATH_MM:
                cues.append("off_path")
            if np.isfinite(luc) and luc < -60:
                cues.append("lucent_centre")
            if np.isfinite(elong) and elong > 0.85:
                cues.append("round")
            # A tube lying in fat is not a calculus, whatever else is true of
            # it. See MIMIC_ELONGATION for the measurement behind this.
            tube_in_fat = bool(np.isfinite(elong) and elong < MIMIC_ELONGATION
                               and np.isfinite(rim) and rim < MIMIC_RIM_HU)
            if tube_in_fat:
                cues.append("tube_in_fat")
            reason = verdict_mimic(cues, tube_in_fat)

        rows.append({
            "study_id": sid, "candidate_id": int(i), "side": best_side,
            "zone": zone,
            # Objective localisation against bone -- independent of the UVJ
            # landmark, which was 49 mm out on 8676809's distended bladder.
            "vertebral_level": vl.level_at(cen[2], vspans),
            "dist_to_uvj_along_mm": round(along_to_uvj, 1),
            "dist_to_uvj_straight_mm": round(straight_to_uvj, 1),
            "dist_from_puj_along_mm": round(along_from_puj, 1),
            "dist_from_puj_straight_mm": round(straight_to_puj, 1),
            "max_diameter_mm": round(float(dmax), 2),
            "volume_mm3": round(float(pv_mm3), 2),
            "dim_tr_mm": round(shape.get("dim_tr_mm", float("nan")), 2),
            "dim_ap_mm": round(shape.get("dim_ap_mm", float("nan")), 2),
            "dim_cc_mm": round(shape.get("dim_cc_mm", float("nan")), 2),
            "hu_max": round(hu_max),
            "hu_mean": round(hu_mean),
            "hu_max_component": round(float(sub_vol[comp_sub].max())),
            # How far this candidate got before a verdict was reached, so the
            # columns above can be read for what they are. "coarse" = rejected
            # on the >=GROW_HU component alone, no FWHM mask exists;
            # "refined" = measured, then rejected on size or density;
            # "full" = every descriptor computed.
            "measure_stage": stage,
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
            # Reported AND flagged. Distinct from reject_reason: a review flag
            # never removes a finding, it only asks for a second look.
            "review_flag": review,
            "is_stone": reason == "",
            "centroid_vox": ",".join(str(int(round(c))) for c in cen),
        })

    # rank the survivors by density, highest first, per side. The rank is now
    # informational only -- with TOP_K_REPORTED off, every accepted stone is
    # reported and every candidate stays in the audit CSV either way.
    stones = [r for r in rows if r["is_stone"]]
    for side in ("left", "right"):
        ss = sorted([r for r in stones if r["side"] == side],
                    key=lambda r: -r["hu_max"])
        for k, r in enumerate(ss, 1):
            r["hu_rank_side"] = k
            r["report_this"] = (TOP_K_REPORTED is None
                                or k <= TOP_K_REPORTED)
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
        # The floor was the second-biggest bucket on 8677121 (14 of 214) and had
        # no counter, so it was invisible in the summary sheet.
        "n_rejected_floor": sum(r["reject_reason"] == "below_hu_floor"
                                for r in rows),
        # How much geometry the study actually paid for. n_measured is the
        # candidates that reached tier 1; the rest were settled by two lookups.
        "n_measured": sum(r["measure_stage"] != "coarse" for r in rows),
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
