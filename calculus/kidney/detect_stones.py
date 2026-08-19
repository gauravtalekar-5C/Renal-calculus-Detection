"""Baseline stone detector -- no machine learning.

For each study with a NIfTI volume and TotalSegmentator masks:

  1. build a urinary-tract ROI  = kidneys + bladder, generously dilated
  2. threshold at CAND_HU inside that ROI
  3. group bright voxels into 3D connected components
  4. drop components that are obviously not stones (bone, too small)
  5. measure each survivor: volume, diameters, HU statistics
  6. locate it: which side, kidney vs ureter vs bladder, and for intrarenal
     stones which third of the kidney along its long axis

Everything here is deterministic geometry. It is the baseline every learned
model has to beat, and its false positives define what the model must learn to
reject.

Deliberate design choices worth knowing:
  * The candidate threshold is LOW (130 HU) so recall stays near 100%. Uric acid
    stones can sit at 200-450 HU and small stones lose density to partial
    volume, so a "safe" 200 HU threshold silently drops real stones.
  * Size is measured on a per-stone FULL-WIDTH-HALF-MAXIMUM boundary rather than
    the detection threshold. Fixed thresholds inflate small stones badly
    (blooming), and FWHM is far more stable across reconstruction kernels.
  * Calyceal location is reported as polar thirds. Individual calyces are not
    visible on non-contrast CT unless dilated, so anything finer would be
    invented. Confidence is emitted so borderline stones can be flagged.

Usage:
    ./venv/bin/python detect_stones.py
    ./venv/bin/python detect_stones.py --debug 8584188

"""

#Search inside the kidney regions for dense 3-D calculus candidates, remove common false positives, and measure accepted stones.


import argparse
import glob
import os
import sys

import cc3d
import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError
from skimage.measure import marching_cubes

# this file lives in utils/, so the project root is one level up. All data
# directories hang off ROOT, never off the script's own folder.
HERE = os.path.dirname(os.path.abspath(__file__))
# package root -> project root: calculus/<sub>/x.py is two levels down
ROOT = os.path.dirname(os.path.dirname(HERE))
from calculus.common.paths import CSV, NIFTI, SEG     # noqa: E402  results dir is per-run

GROW_HU = 130 
SEED_HU = 200          # every real stone must contain at least one voxel here

#130 HU determines how far the object extends and 200 HU determines whether the object is sufficiantly stone-like

CAND_HU = GROW_HU      # kept for backwards compatibility in summaries
MIN_VOXELS = 3         # Elton et al. use a 3-pixel minimum
MIN_DIAM_MM = 1.5      # smallest individually reportable stone and mazimum stone dia is 30 mm, we dont rejct them but we say it s marked for review  
KIDNEY_DILATE_MM = 12  # (legacy, whole-tract mode only)
SINUS_FILL_MM = 15     # closing radius: fills renal sinus/pelvis concavity for kidney 
CAPSULE_CUFF_MM = 3    # small outward cuff so capsular stones are not clipped, not to miss capsular stone


# An unenhanced kidney is ~30 HU. Above this the parenchyma is enhanced, and in
# a delayed/excretory phase the collecting system fills with contrast that is
# indistinguishable from stone -- while the AORTA has already washed out, so
# aorta-based phase detection alone passes it as "plain". Measured: an
# excretory-phase study read 96 HU with 19,647 mm3 of >=200 HU material inside
# the kidneys, against 29-40 HU and 3-74 mm3 for genuine plain scans.
KIDNEY_PLAIN_MAX_HU = 60
EXCRETION_VOL_MM3 = 1500
BLADDER_DILATE_MM = 5
URETER_CORRIDOR_MM = 18  # crude tube around the kidney->bladder line
BONE_MARGIN_MM = 2.0   # candidates this close to bone are partial volume
BONE_HU = 300
VESSEL_MARGIN_MM = 3.0   # candidates inside a vessel mask are calcification
BONE_MIN_VOL_MM3 = 3000  # a dense component this large is bone, never a stone
MAX_STONE_DIAM_MM = 30   # bigger than this is flagged, not silently reported
# Edge-preserving denoising applied BEFORE thresholding, following Elton et al.
# 2022 (Med Phys). Detection runs on the denoised volume; all measurements are
# taken from the original.
# Swept against noise-only phantoms (sd 50 HU) and stone phantoms:
#   kappa 120 / 5 iters -> noise blobs >=200 HU fall 373 -> 4 (99% removed)
#   while a 2 mm/300 HU stone still peaks at 296 HU, well clear of SEED_HU.
#   20 iterations flattens that stone to 52 HU, i.e. destroys it. kappa must sit
#   ABOVE the noise gradient scale (~50-90 HU) or the filter mistakes noise for
#   edges and preserves it.
# Elton et al. iterate "until connected components drop below 200" rather than
# using a fixed count -- the stopping point then adapts to each scan's noise
# instead of under-denoising a noisy scan and over-denoising a clean one.



DENOISE_ITERS = 1          # filter iterations per round (theirs: 1)
DENOISE_MAX_ROUNDS = 10    # theirs: while iter < 10
DENOISE_TARGET_CC = 200    # theirs: MAX_COMPONENTS = 200
CLIP_LOW, CLIP_HIGH = -200.0, 1000.0   # theirs, before denoising
# Their minimum is a VOLUME (0.25 mm3 in detector.py), not a voxel count, so it
# behaves the same at any spacing. 0.25 mm3 is about a 0.8 mm sphere -- far below
# anything reportable, deliberately: their CNN does the rejecting, so stage 1
# keeps everything. We keep them too, tagged with a reject_reason.
MIN_CANDIDATE_MM3 = 0.25
# Peak estimator switch-over, in voxels. `peak` feeds both the FWHM threshold
# and the partial-volume normaliser, so getting it wrong scales every volume.
#
# The 95th percentile was chosen to shrug off single-voxel noise spikes, but it
# only does that once there are enough voxels to have a 95th percentile: on a
# 3-voxel candidate p95 IS very nearly the max (measured: 3 vox -> p95 470 on a
# 200 HU plateau with one 500 HU spike; 100 vox -> p95 200). So below this count
# the robustness was imaginary, while the cost was real -- partial volume
# depresses a small stone's observed peak below its true attenuation, and p95
# depresses it further, inflating the volume.
#
# Swept on 72 sphere phantoms (8 diameters x 3 densities x 3 noise seeds) at
# 0.65x0.65x1.25 mm, scoring the resulting VOLUME error:
#
#     peak rule                 all: mean / mean abs     <=2.5 mm: mean / abs
#     p95 always (was)             +1.4% /  8.1%            +11.6% / 13.4%
#     max always                   -4.7% /  9.1%             +4.1% /  7.7%
#     max if <20 vox else p95      -1.2% /  6.2%             +4.7% /  8.2%
#     max if <30 vox else p95      -1.3% /  6.1%   <- here    +4.5% /  8.0%
#     max if <50 vox else p95      -1.9% /  6.4%             +4.1% /  7.7%
#
# 30 is the flat part of the curve, not a knife-edge: 20 and 50 are within
# 0.3 points of it. Small stones improve most (13.4% -> 8.0%) because that is
# where the two errors were compounding.
PEAK_MIN_VOX = 30




#cleaning kidney masks, removes small disconnected segmentation fragments
def largest_components(mask, k=1):
    """Keep the k largest connected components, dropping speckle.

    Elton et al. do this on their kidney segmentation to remove spurious
    objects. One component per kidney label, since left and right are separate
    masks here.
    """
    # label every separate blob; 26 = voxels touching at faces, edges OR corners

    #using this we label every seperate kidney 3d component 
    lab, n = cc3d.connected_components(mask, connectivity=26, return_N=True)
    if n <= k:                       # already k blobs or fewer -> nothing to drop
        return mask
    
    # voxel_counts[0] is the background, so [1:] leaves one entry per real blob
    sizes = cc3d.statistics(lab)["voxel_counts"][1:] 

    # argsort ascending -> [::-1] descending -> take the k biggest.
    # +1 converts a 0-based position in `sizes` back to its cc3d label id.
    keep = set(int(i) + 1 for i in np.argsort(sizes)[::-1][:k])
    return np.isin(lab, list(keep))  # boolean mask of just those labels
# WHAT THIS FUNCTION DOES: takes a binary mask that may contain several
# disconnected blobs and returns only the k biggest ones. Used on each kidney
# mask so that a stray speck of mislabelled tissue somewhere else in the
# abdomen does not get treated as part of the kidney.



def load_masks(study_id):
    """Return dict of TotalSegmentator masks that exist for this study."""
    d = os.path.join(SEG, study_id)          # seg/<study_id>/ holds one file per organ
    out = {}
    # Only the structures this pipeline actually uses. Kidneys define the search
    # region; bones and vessels are needed to REJECT false positives (rib cortex
    # and vascular calcification both look exactly like stone on CT).
    for name in ["kidney_left", "kidney_right", "urinary_bladder", "aorta",
                 "inferior_vena_cava", "iliac_artery_left", "iliac_artery_right",
                 "vertebrae_L1", "vertebrae_L5", "sacrum", "hip_left",
                 "hip_right", "kidney_cyst_left", "kidney_cyst_right"]:
        p = os.path.join(d, f"{name}.nii.gz")
        if os.path.exists(p):                # absent = TotalSegmentator found none
            # dataobj + asanyarray reads lazily; `> 0` turns the label map boolean
            m = np.asanyarray(nib.load(p).dataobj) > 0
            if m.any():                      # a file can exist but be all zeros
                # de-speckle the kidneys only. Cysts are legitimately multiple,
                # and the bone/vessel masks are only used as distance maps, so
                # extra fragments there are harmless.
                if name.startswith("kidney_") and not name.startswith("kidney_cyst"):
                    m = largest_components(m, k=1)
                out[name] = m
    return out                               # missing keys mean "not available"
# WHAT THIS FUNCTION DOES: loads the TotalSegmentator organ masks for one study
# off disk into a dictionary of boolean arrays. Anything the segmenter did not
# produce is simply absent from the dict, so every caller has to check before
# using a mask rather than assuming it exists.


def denoise_ct(vol, iterations=1):
    """One pass of curvature anisotropic diffusion, via SimpleITK.

    This is the filter Elton et al. use (`CurvatureAnisotropicDiffusionImageFilter`,
    NumberOfIterations=1, default time step, no explicit spacing -- so it works in
    voxel units). Replaces a hand-written Perona-Malik implementation of mine that
    had a sign error in the divergence and amplified noise instead of removing it.
    On pure-noise phantoms this filter takes voxels >=200 HU from 39 to 0 in a
    single iteration.
    """
    # Edge-PRESERVING smoothing: it diffuses (blurs) inside flat regions but
    # stops at strong gradients, so noise dies while stone borders survive.
    f = sitk.CurvatureAnisotropicDiffusionImageFilter()
    f.SetNumberOfIterations(iterations)      # 1 per round; the caller loops
    img = sitk.GetImageFromArray(vol.astype(np.float32))   # numpy -> ITK image
    try:
        out = f.Execute(img)
    except RuntimeError:
        # the CFL limit for 3D is stricter than SimpleITK's default time step.
        # Too large a step makes the diffusion equation blow up numerically, and
        # ITK raises rather than returning garbage. Halve it and retry.
        f.SetTimeStep(0.03125)
        out = f.Execute(img)
    return sitk.GetArrayFromImage(out)       # ITK image -> numpy
# WHAT THIS FUNCTION DOES: removes CT noise without softening stone edges, so
# that thresholding at 130 HU finds real objects instead of noise speckles.
# Detection runs on this filtered copy; every MEASUREMENT is taken from the
# untouched original, because the filter does move HU values slightly.


def dist_mm(mask, spacing):
    """Euclidean distance in mm from every voxel to the nearest True voxel.

    One distance transform beats repeated binary_dilation with a large
    spherical element: dilation by any radius becomes a threshold on this,
    and the same map answers "how far is this candidate from bone/kidney".
    """
    # `~mask` is the key. distance_transform_edt measures the distance to the
    # nearest ZERO, so inverting makes the object the zeros: the result is 0
    # INSIDE the object and grows outward. `sampling=spacing` makes those
    # distances true millimetres -- without it one step in z would count the
    # same as one step in x, even though a z step is usually twice as far.
    return ndimage.distance_transform_edt(~mask, sampling=spacing)
# WHAT THIS FUNCTION DOES: builds a "how far am I from this structure" map in
# real millimetres. One map answers many questions -- dilation by any radius r
# is just `d <= r`, and the same array tells a candidate how close it sits to
# bone or to the kidney. Computing it once is far cheaper than repeated
# binary_dilation with a large spherical element.


def dilate_mm(mask, mm, spacing, dist=None):
    if mm <= 0:                              # nothing to do; return unchanged
        return mask
    # reuse a distance map if the caller already built one, else make it now
    d = dist if dist is not None else dist_mm(mask, spacing)
    return d <= mm                           # dilation IS a threshold on distance
# WHAT THIS FUNCTION DOES: grows a mask outward by a distance in millimetres
# (not in voxels), which matters because voxels are not cubes. Expressing it as
# a threshold on the distance map means the radius can be any real number and
# costs nothing extra when the map is already available.


def crop_box(mask, margin_vox, shape):
    """Bounding slice of mask, padded, for doing per-stone work on a small
    subvolume instead of the full 512x512x800 array."""
    idx = np.nonzero(mask)                   # coordinate arrays, one per axis
    sl = []
    # a = coords on this axis, n = array size on this axis, m = padding in voxels
    for a, n, m in zip(idx, shape, margin_vox):
        # clamp to [0, n] so padding never runs off the edge of the volume
        sl.append(slice(max(0, int(a.min()) - m), min(n, int(a.max()) + m + 1)))
    return tuple(sl)                         # usable directly as vol[sl]
# WHAT THIS FUNCTION DOES: returns the small padded box that contains a mask, so
# per-stone work runs on a ~40x40x20 subvolume instead of the whole scan. This
# is what keeps cost per candidate flat no matter how large the study is.


def build_roi(masks, shape, spacing, kidney_only=True):
    """Search region for candidates.

    kidney_only=True (default) restricts the search to the kidneys plus a
    KIDNEY_DILATE_MM margin, which covers the collecting system and renal
    pelvis without leaving the kidney.

    The wider mode adds the bladder and a straight-line "ureteric corridor".
    That corridor is a poor model of the ureter -- the real ureter curves back
    along the psoas, so a straight tube cuts through bowel, mesentery, sacrum
    and iliac vessels. On the first test study it produced 70 spurious ureteric
    detections. Ureteric work needs the learned corridor described in the
    design, not this stand-in, so it is off by default.
    """
    kidneys = np.zeros(shape, bool)          # start empty, OR the two sides in
    for k in ("kidney_left", "kidney_right"):
        if k in masks:                       # one side may be missing
            kidneys |= masks[k]
    # one distance map serves the closing, the cuff, and later the
    # perirenal-vs-ureteric decision for each stone
    kidney_dist = dist_mm(kidneys, spacing) if kidneys.any() else None

    if kidney_only:
        if kidney_dist is None:              # no kidney found -> empty ROI,
            return np.zeros(shape, bool), None   # caller reports "no segmentation"
        # CLOSING, not dilation. Dilating 12 mm outward to capture the renal
        # pelvis also swallows 12 mm of perinephric fat, rib and bowel, and
        # detections duly appeared outside the kidney. Closing fills the hilar
        # concavity -- which is exactly where the pelvis and calyces sit -- while
        # leaving the outer surface almost untouched. A small 3 mm cuff is kept
        # so stones abutting the capsule are not clipped.
        # CLOSING = dilate then erode by the SAME radius.
        dil = kidney_dist <= SINUS_FILL_MM           # step 1: grow 15 mm outward
        # step 2: erode 15 mm back. Eroding a mask == dilating its INVERSE and
        # inverting the result, which is what this line does -- reusing dist_mm
        # instead of needing a separate erosion with a 15 mm ball.
        closed = ~(dist_mm(~dil, spacing) <= SINUS_FILL_MM)
        # Grow-then-shrink returns the outer surface to almost where it started
        # but leaves concavities FILLED -- and the hilum is the concavity we
        # want, because that is where the pelvis and calyces sit. Add a small
        # outward cuff so a stone touching the capsule is not clipped.
        roi = closed | (kidney_dist <= CAPSULE_CUFF_MM)
        return roi, kidney_dist

    # ---- wider whole-tract mode, OFF by default (see the docstring) ---------
    bladder = masks.get("urinary_bladder", np.zeros(shape, bool))
    seeds = kidneys | bladder
    if bladder.any() and kidneys.any():
        bc = np.array(ndimage.center_of_mass(bladder))   # bladder centre
        for k in ("kidney_left", "kidney_right"):
            if k not in masks:
                continue
            kc = np.array(ndimage.center_of_mass(masks[k]))   # kidney centre
            # walk 120 evenly spaced points along the straight kidney->bladder
            # line and mark each as a seed, i.e. draw the crude "ureter"
            for t in np.linspace(0, 1, 120):
                p = kc + (bc - kc) * t                # linear interpolation
                idx = tuple(int(round(v)) for v in p)  # nearest voxel
                if all(0 <= idx[i] < shape[i] for i in range(3)):   # in bounds?
                    seeds[idx] = True
    # one dilation covers kidneys, bladder and the corridor line at the widest
    # of the three radii
    roi = dist_mm(seeds, spacing) <= max(KIDNEY_DILATE_MM, BLADDER_DILATE_MM,
                                         URETER_CORRIDOR_MM)
    return roi, kidney_dist

    
# WHAT THIS FUNCTION DOES: decides WHERE in the scan we are allowed to look for
# stones. Getting this wrong dominates everything downstream -- too tight and
# stones in the renal pelvis are never seen, too loose and rib cortex and bowel
# contents become "stones". The closing trick is the compromise: it reaches into
# the hilum without reaching out into fat. Measured across 37 studies, closing
# adds +37% to +58% volume while plain dilation adds +195% to +379%.


def kidney_frame(mask, spacing):
    """Long axis of a kidney, oriented to point SUPERIOR.

    Array layout from extract_series.py is (left, posterior, superior), so the
    superior component is index 2. Orienting on index 0 (left-right) would flip
    the axis unpredictably from kidney to kidney and scramble pole assignment.
    """
    idx = np.array(np.nonzero(mask), float).T          # (N,3) in voxels
    world = idx * np.array(spacing)     # voxels -> millimetres
    c = world.mean(0)                   # centroid of the kidney
    # SVD of the centred point cloud. vt[0] is the direction of greatest spread,
    # i.e. the kidney's long axis -- found from the data, not assumed vertical.
    u, s, vt = np.linalg.svd(world - c, full_matrices=False)
    axis = vt[0]
    # SVD returns a direction, not an orientation: vt[0] and -vt[0] are equally
    # valid answers and which one you get is arbitrary. Force it to point head-
    # ward so "upper pole" means the same thing in every study.
    if axis[2] < 0:                     # index 2 = superior
        axis = -axis
    return c, axis, world
# WHAT THIS FUNCTION DOES: finds each kidney's own long axis and centre, so a
# stone's position can be described anatomically (upper/inter/lower pole)
# instead of by raw voxel coordinates. Doing it per kidney matters because
# kidneys are tilted, and differently tilted from each other.


def polar_zone(point_world, c, axis, world):
    """Which third of the kidney along its long axis, plus a confidence."""
    # project every kidney voxel onto the long axis -> position along it, in mm
    proj = (world - c) @ axis
    # 0.5/99.5 percentiles rather than min/max: a single stray voxel at either
    # end would otherwise stretch the scale and shift every zone boundary
    lo, hi = np.percentile(proj, 0.5), np.percentile(proj, 99.5)
    # where the stone sits on that axis, rescaled to 0..1 (0 = caudal tip)
    t = ((point_world - c) @ axis - lo) / max(hi - lo, 1e-6)   # max() avoids /0
    t = float(np.clip(t, 0, 1))         # a stone just past the tip clamps to the end
    # the axis points superior, so t=0 is the caudal (lower) end of the kidney
    if t <= 1 / 3:
        zone = "lower_pole"
    elif t <= 2 / 3:
        zone = "interpolar"
    else:
        zone = "upper_pole"
    # Confidence falls off near the 1/3 and 2/3 boundaries. A stone at t=0.334
    # is called interpolar but is a hair from lower_pole, and the reader needs
    # to know that. Distance to the nearer boundary, scaled so the middle of a
    # third scores 1.0 and sitting exactly on a boundary scores 0.0.
    dist = min(abs(t - 1 / 3), abs(t - 2 / 3))
    conf = float(np.clip(dist / (1 / 6), 0, 1))
    return zone, round(t, 3), round(conf, 2)
# WHAT THIS FUNCTION DOES: converts a stone's 3D position into the words a
# radiologist uses -- upper pole, interpolar, lower pole -- plus a number saying
# how borderline that call is. Thirds are as fine as we can honestly go:
# individual calyces are not visible on non-contrast CT unless dilated, so
# naming a specific calyx would be invented detail.


def split_bone_bridges(labels, n, peak_of, bone_dist, vol, voxel_mm3):
    """Rescue calculi that partial volume has fused to bone.

    At GROW_HU (130) a stone lying against a rib or vertebra is joined to it by
    a bridge of partial-volume voxels, so the two become ONE connected
    component. That component is then majority-bone, gets rejected as
    bone_partial_volume, and the stone is discarded along with the rib. It is a
    silent false negative -- nothing in the output says a stone was there.

    Seen in 8379961: a component with bone_frac 0.84, i.e. 16% of it was NOT
    bone, thrown away whole. The report says that kidney contains calculi.

    The fix splits instead of rejecting. For a majority-bone component:

        bone part      voxels within BONE_MARGIN_MM of bone   -> kept as its own
                       candidate, so it still appears in candidates.csv and is
                       still rejected. The audit trail and the Part 2 training
                       negatives are preserved.
        non-bone part  everything else, re-labelled into connected pieces ->
                       each becomes its own candidate and is judged on its own
                       bone_frac, which is now low.

    No voxel is lost either way, so a component that really was all bone simply
    produces one bone candidate and nothing else -- identical to the old
    behaviour. Only genuinely mixed components change outcome.

    Peak HU for a new piece is the raw maximum inside it. That differs from the
    seed-region peak used for unsplit components, and is the right choice here:
    the question for a rescued fragment is whether IT contains a dense core, not
    whether the original merged blob did.
    """
    stats = cc3d.statistics(labels)      # bounding box + voxel count per label
    out = np.zeros_like(labels)          # the rebuilt label volume we return
    new_peak = {}                        # new label id -> its peak HU
    next_id = 0                          # labels are renumbered as we emit them
    n_split = 0                          # how many components we actually split

    for lab in range(1, n + 1):          # label 0 is background, so start at 1
        bb = stats["bounding_boxes"][lab]
        if bb is None:                   # label absent (can happen after remap)
            continue
        sub = labels[bb] == lab          # this component, inside its own box
        if not sub.any():
            continue
        # which voxels of the box are within 2 mm of bone
        near_bone = bone_dist[bb] < BONE_MARGIN_MM
        # `near_bone[sub]` selects only THIS component's voxels, so .mean() is
        # the fraction of the component that is bone-adjacent
        if float(near_bone[sub].mean()) <= 0.5:
            next_id += 1                       # untouched: majority not bone
            out[bb][sub] = next_id             # copy through under a new id
            new_peak[next_id] = peak_of.get(lab, 0.0)   # keep its original peak
            continue

        # majority bone -> split into the bone part and the rest
        rescued = 0
        free = sub & ~near_bone          # the part of the blob AWAY from bone
        if free.any():
            # the free part may itself be several separate pieces (a stone on
            # each side of a rib), so label it again rather than assuming one
            pieces, npc = cc3d.connected_components(free, connectivity=26,
                                                    return_N=True)
            for p in range(1, npc + 1):
                pm = pieces == p
                if pm.sum() * voxel_mm3 < MIN_CANDIDATE_MM3:
                    continue                   # too small to be anything
                next_id += 1
                out[bb][pm] = next_id
                # peak from the RAW volume inside this piece: the question for a
                # rescued fragment is whether IT has a dense core, not whether
                # the merged blob did
                new_peak[next_id] = float(vol[bb][pm].max())
                rescued += 1
        bone_part = sub & near_bone      # the bone-adjacent remainder
        if bone_part.any():
            # emitted as its own candidate, NOT deleted. It will be rejected
            # downstream as bone_partial_volume, which keeps the audit trail and
            # gives Part 2 a labelled negative to train against.
            next_id += 1
            out[bb][bone_part] = next_id
            new_peak[next_id] = float(vol[bb][bone_part].max())
        if rescued:                      # only count a split if something survived
            n_split += 1

    return out, new_peak, next_id, n_split
# WHAT THIS FUNCTION DOES: fixes a silent false negative. A stone touching a rib
# is joined to it by partial-volume voxels, so the two label as ONE object,
# which is then majority-bone and gets thrown away -- taking the stone with it,
# with nothing in the output to say it happened. This splits such an object into
# its bone part and its non-bone part and lets each be judged separately. No
# voxel is discarded, so a component that really was all bone behaves exactly as
# before; only genuinely mixed ones change outcome.


#code review for measurement
def fwhm_measure(vol, comp_mask, spacing, sl):
    """Re-threshold a single stone at full-width-half-maximum, within a crop.

    A fixed threshold overestimates small stones (blooming). FWHM uses each
    stone's own peak and local background, so it is far more stable across
    kernels. All work happens inside `sl`, a small bounding box, so cost does
    not scale with the size of the scan.
    """
    sub_vol, sub_comp = vol[sl], comp_mask[sl] #crop to the padded bounding box, here sub_vol is the rawHU and sub_comp is the blob
    d = dist_mm(sub_comp, spacing) #millimeter distance from blob surface

    # "How bright is the stone?" -- see PEAK_MIN_VOX for why the estimator is
    # size-aware. Below 30 voxels the percentile has nothing to average over,
    # so it just subtracts signal from an already partial-volumed peak.
    core_vals = sub_vol[sub_comp]
    peak = float(core_vals.max() if core_vals.size < PEAK_MIN_VOX
                 else np.percentile(core_vals, 95))
     
    # a hollow 2 mm rind hugging the stone: d>0 excludes the stone itself
    shell = (d > 0) & (d <= 2.0)
    # MEDIAN, not mean: if a rib sits in the rind, a mean would be dragged up by
    # hundreds of HU, while the median ignores it unless bone is over half the rind
    bg = float(np.median(sub_vol[shell])) if shell.any() else 0.0 #check the briggtness of the surroundings, neighboring stone may land in this too
    thr = (peak + bg) / 2.0                  # the half-maximum level itself
    # Two conditions ANDed. `sub_vol >= thr` is the FWHM rule; `d <= 1.5` is a
    # leash, so the boundary can only move +-1.5 mm from where detection put it.
    # Without the leash a stone touching a rib would grow into the whole rib.
    refined = (d <= 1.5) & (sub_vol >= thr) #Halfway, leashed to +-1.5 mm so it refines rather than re-detects
    
    # Keep only the piece belonging to the original component.
    #
    # 26-connectivity via cc3d, NOT ndimage.label. ndimage.label defaults to
    # 6-connectivity in 3D, so two voxels touching at a corner count as two
    # objects -- while every other component call in this file uses cc3d at 26.
    # That mismatch could split a diagonally-joined stone here and then measure
    # only half of it.
    lab, n = cc3d.connected_components(refined, connectivity=26, return_N=True)
    if n > 1:
        # Count how many voxels of the ORIGINAL blob fell into each new piece,
        # and keep the piece with the most. Most-overlap, not largest: if a rib
        # corner cleared the threshold and happened to be bigger, "largest"
        # would discard the stone and keep the rib.
        overlap = np.bincount(lab[sub_comp], minlength=n + 1)[1:]
        if overlap.max() == 0:
            # No new piece touches the original blob at all. argmax would
            # return 0 here and silently keep label 1 -- whichever fragment
            # happens to come first in scan order. Fall back to the candidate
            # we started from instead of measuring an arbitrary object.
            refined = sub_comp
        else:
            refined = lab == int(overlap.argmax()) + 1
    if refined.sum() < 1:                # threshold killed everything -> safety
        refined = sub_comp               # net: never return an empty mask
    # Integrate partial occupancy over the stone plus a 1 mm boundary layer.
    # Radius and reference percentile were swept against phantoms: a 1.0 mm
    # shell gives 6.1% mean absolute volume error and -1.3% bias with the
    # size-aware peak above. A 2 mm shell drags in too much blurred halo (30%
    # error on small stones).
    #
    # This integral beat both alternatives on the same 72 phantoms -- volume
    # from the marching-cubes iso-surface came in at 27.2% mean abs error
    # (the half-maximum surface under-encloses a blurred small object) and
    # plain voxel counting at 16.8%. So geometry comes from the mesh in
    # shape_metrics(), but VOLUME stays with this integral.
    pv = partial_volume_mm3(sub_vol, d <= 1.0, spacing, peak, bg)
    # thr/peak/bg are returned as well as the mask so every row in the CSV
    # records the exact threshold that produced it -- measurements stay auditable
    return refined, thr, peak, bg, pv
# WHAT THIS FUNCTION DOES: redraws one stone's boundary at HALF ITS OWN
# brightness instead of at a fixed 130 HU, and integrates its volume.
#
# Why that matters: CT blurs edges, so where you say the stone "ends" depends on
# the cutoff you pick. A fixed cutoff sits far out in the fade for a bright
# stone (making it read too big) and close in for a dim one. Half-maximum is the
# point where the blur's outward and inward errors cancel, so each stone gets a
# threshold scaled to itself and the measurement stops depending on density.
# Measured on phantoms: a fixed threshold's error grows 8x from soft to dense
# stones (+0.12 -> +0.99 mm); the FWHM error stays flat.


def max_diameter_mm(mask, spacing):
    """Longest caliper distance across the object, in mm. VOXEL-GRID fallback.

    Kept only for the cases shape_metrics() cannot mesh (a candidate so small
    that the half-maximum surface does not close). Measured between voxel
    CENTRES, then extended by half a voxel because the surface lies beyond the
    outermost centre at each end.

    Prefer shape_metrics(): this function is quantised to the voxel grid, which
    on small stones is the dominant error. Phantoms at 0.65x0.65x1.25 mm: a
    2.0 mm and a 2.5 mm sphere BOTH measure 2.16 mm here, because they occupy
    the same voxels. The mesh separates them (2.04 and 2.48).
    """
    # every True voxel as an (N,3) point list, converted from index to mm
    pts = np.array(np.nonzero(mask), float).T * np.array(spacing)
    # Per-axis half-voxel, from the object's own orientation. The old code used
    # the mean of the two SMALLEST spacings, i.e. always in-plane, which
    # under-extended a craniocaudally elongated stone by 0.3 mm at 1.25 mm
    # slices. Weight each axis by how much of the extent lies along it.
    if len(pts) == 0:
        return 0.0                   # empty mask -> no diameter
    span = pts.max(axis=0) - pts.min(axis=0)      # bounding-box size per axis
    w = span / span.sum() if span.sum() > 0 else np.full(3, 1 / 3)   # weights
    edge = 0.5 * float(np.dot(w, spacing))        # weighted half-voxel, in mm
    if len(pts) == 1:
        return edge                  # a single voxel: its own half-width
    # Convex hull, not a random subsample. The extreme points of any object are
    # always hull vertices, so this is EXACT where the old 400-point subsample
    # was merely likely -- and it is faster, because the hull collapses a few
    # thousand points to a few dozen before the pairwise distance matrix.
    # Measured on a branching staghorn-like phantom: subsample 11.26 mm vs
    # hull 31.85 mm (-20.6 mm error), and the hull ran 19x quicker.
    if len(pts) > 4:             # a hull needs at least 4 non-coplanar points
        try:
            pts = pts[ConvexHull(pts).vertices]     # keep only corner points
        except QhullError:
            pass                 # degenerate/coplanar: fall through to all pts
    # pts[:,None] - pts[None] broadcasts to an (N,N,3) array of differences;
    # norm along the last axis gives every pairwise distance at once
    d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    return float(d.max()) + edge         # longest pair, plus the surface offset
# WHAT THIS FUNCTION DOES: measures the longest straight line across a stone
# using voxel centres. This is now only a FALLBACK -- shape_metrics() does the
# same job on a sub-voxel surface and is three times more accurate on small
# stones. It survives because a candidate of a few voxels cannot form a closed
# surface, and something still has to return a number for it.

# sub_vol takes small cropped region from org CT containing raw HU values, refined- mask identifying refined stone component,spacing is physical voxel data, threshold
#refined mask → tells the function where the stone is raw HU image → determines the final sub-voxel surface

#using threshold we can check stone boundary
def shape_metrics(sub_vol, refined, spacing, thr):
    """Sub-voxel diameter and three dimensions, from the half-maximum surface.

    Everything above works on voxels, which are blocks. Blocks quantise: a
    2.0 mm and a 2.5 mm stone can occupy exactly the same voxels and so return
    exactly the same diameter. Marching cubes instead fits a continuous surface
    where the intensity crosses `thr` -- the same FWHM level fwhm_measure
    chose -- and interpolates WITHIN each voxel, so the answer is no longer
    tied to the grid.

    Validated on sphere phantoms at 0.65x0.65x1.25 mm (mean abs diameter error
    over sizes 1.5-3 mm): voxel method 0.33 mm, mesh 0.11 mm.

    Returns a dict, or None when the surface cannot be built (too few voxels).
    Volume is deliberately NOT taken from this mesh -- see fwhm_measure.

    Keys:
        max_diameter_mm  longest caliper across the surface (hull-exact)
        axis_major/intermediate/minor_mm
                         extents along the object's OWN principal axes, found
                         by SVD, so they are rotation-invariant. A stone lying
                         diagonally still reports its true long axis.
        dim_tr/ap/cc_mm  extents along the scanner axes (transverse,
                         antero-posterior, craniocaudal) -- what a radiologist
                         reads off axial and coronal images.
        elongation       minor/major, 1.0 = round, ->0 = needle
        flatness         minor/intermediate
        ellipsoid_volume_mm3
                         volume of the ellipsoid with these three axes. A
                         cross-check on the integrated volume, and the number
                         comparable to reports that quote A x B x C.
    """
    #create a small neighborhood around the stone,generate_binary_structure uses full 26 neighbour connectivity - Face-touching voxels, Edge-touching voxels, Corner-touching voxels.
    near = ndimage.binary_dilation(
        refined, ndimage.generate_binary_structure(3, 3), iterations=2)
    # Everything outside `near` is driven far below the level so no surface can
    # form there. Pad by 1 so a stone touching the crop edge still closes.
    #remocve everything outside the neighbourhood
    field = np.where(near, sub_vol, thr - 1000.0).astype(np.float32)

    # add a low intensity border 0 adds one layer around the field
    field = np.pad(field, 1, constant_values=thr - 1000.0)
    try:
        # `level=thr` uses the SAME half-maximum value fwhm_measure chose, so
        # the geometry and the threshold can never disagree. `spacing=spacing`
        # makes the returned vertices millimetres rather than voxel indices.

        #Marching cubes is an algorithm that turns a 3-D intensity image into a surface made of small triangles.Marching cubes performs this operation over groups of eight neighbouring voxels and connects all the threshold crossings using triangles:
        verts, faces, _, _ = marching_cubes(field, level=thr, spacing=spacing)
    except (RuntimeError, ValueError):
        return None                  # surface will not close -> caller falls back
    if len(verts) < 4:
        return None                  # too few vertices to be a solid

    # Longest caliper. Hull first: extremes are always hull vertices.
    try:

        hull_pts = verts[ConvexHull(verts).vertices]
    except QhullError:
        hull_pts = verts             # flat/degenerate surface: use all vertices
    dmax = float(np.linalg.norm(hull_pts[:, None] - hull_pts[None],
                                axis=-1).max())      # biggest pairwise distance

    # Principal axes. Centre the vertices, then SVD: the rows of Vt are the
    # object's own axes, ordered by how much the surface spreads along each.
    # Projecting the vertices onto them and taking (max - min) gives the extent
    # along each axis. This is what makes the numbers rotation-invariant --
    # bounding-box extents would change if the patient were tilted.
    centred = verts - verts.mean(axis=0)     # move the centroid to the origin
    _, _, vt = np.linalg.svd(centred, full_matrices=False)   # vt rows = axes
    proj = centred @ vt.T                    # vertices in the object's own frame
    ax = np.sort(proj.max(axis=0) - proj.min(axis=0))[::-1]   # major -> minor

    # Scanner-axis extents. Array axis 0/1/2 are transverse / antero-posterior
    # / craniocaudal in the RAS volumes we build, which is what a radiologist
    # measures on axial (TR, AP) and coronal (CC) views.
    aabb = verts.max(axis=0) - verts.min(axis=0)

    major, inter, minor = (float(v) for v in ax)
    return {
        "max_diameter_mm": dmax,             # longest line anywhere in the stone
        "axis_major_mm": major,              # the stone's own long axis
        "axis_intermediate_mm": inter,       # its own middle axis
        "axis_minor_mm": minor,              # its own short axis
        "dim_tr_mm": float(aabb[0]),         # left-right, as read on axial
        "dim_ap_mm": float(aabb[1]),         # front-back, as read on axial
        "dim_cc_mm": float(aabb[2]),         # head-foot, as read on coronal
        # 1.0 = round, ->0 = needle. Guards avoid a divide-by-zero on a
        # degenerate surface, returning NaN rather than crashing the study.
        "elongation": minor / major if major > 0 else np.nan,
        "flatness": minor / inter if inter > 0 else np.nan,
        # volume of the ellipsoid with these three axes: a cross-check on the
        # integrated volume, and the figure comparable to reports quoting AxBxC
        "ellipsoid_volume_mm3": float(np.pi / 6.0 * major * inter * minor),
    }
# WHAT THIS FUNCTION DOES: measures a stone's SHAPE on a smooth surface instead
# of on blocky voxels, giving three dimensions rather than one number.
#
# Why it exists: voxels quantise. At 0.65x0.65x1.25 mm a 2.0 mm and a 2.5 mm
# stone fill the same voxels, so the voxel method returned 2.16 mm for both.
# Marching cubes interpolates inside each voxel and separates them (2.04, 2.48).
#
# Two families of dimension are returned on purpose. axis_* are the stone's OWN
# axes and do not change if the patient is tilted -- the right thing for
# tracking a stone between scans. dim_* are the scanner axes, which is what a
# radiologist actually measures. Study 8619669 showed why both are needed: our
# longest caliper was 34.27 mm and the report said 30.2 mm, and the two agreed
# once we compared against dim_cc_mm = 30.42 mm. Not an error -- a different
# measurement convention.
#
# VOLUME is deliberately NOT taken from this mesh. On the same 72 phantoms the
# mesh gave 27.2% mean absolute volume error against 8.2% for the integral,
# because the half-maximum surface under-encloses a blurred small object.



def partial_volume_mm3(sub_vol, core, spacing, peak, bg):
    """Volume by integrating partial occupancy, not by counting voxels.

    Counting voxels above a threshold forces every voxel to be fully in or
    fully out, which loses the whole boundary layer -- the dominant error for
    small stones, where the boundary IS most of the object. Instead each voxel
    contributes the fraction (HU - background) / (stone HU - background),
    clipped to [0,1], which is what its attenuation actually implies.
    """
    # max(...,1.0) guards a candidate barely above its surroundings, where
    # peak-bg could be ~0 and the division would explode
    denom = max(peak - bg, 1.0)
    # occupancy per voxel: reading `peak` -> 1.0 (all stone), reading `bg` -> 0.0
    # (all background), halfway -> 0.5. clip stops a noise spike above peak from
    # contributing more than one voxel's worth, or fat below bg going negative.
    frac = np.clip((sub_vol - bg) / denom, 0.0, 1.0)
    frac = frac * core                        # only near the stone
    return float(frac.sum() * np.prod(spacing))   # fractions -> mm3
# WHAT THIS FUNCTION DOES: computes volume by adding up how FULL each voxel is,
# rather than counting voxels that pass a threshold.
#
# Why: counting forces every voxel to be entirely in or entirely out, which
# throws away the whole blurred boundary layer. On a 2 mm stone at 1.25 mm
# slices that boundary IS most of the object. Integrating occupancy recovers it,
# and it is mathematically exact for a linear imaging system as long as `peak`
# equals the stone's true attenuation -- which is why PEAK_MIN_VOX exists.


def analyse(study_id, verbose=False, kidney_only=True, denoise=True):
    nii = nib.load(os.path.join(NIFTI, f"{study_id}.nii.gz"))   # lazy header read
    # float32 because HU arithmetic below (thresholds, fractions) needs floats,
    # and the stored type is usually int16
    vol = np.asanyarray(nii.dataobj).astype(np.float32)
    spacing = tuple(float(v) for v in nii.header.get_zooms()[:3]) #physical size of one voxel along the 3 axes
    voxel_mm3 = float(np.prod(spacing))    #spacing_x × spacing_y × spacing_z
    masks = load_masks(study_id)
    if not masks:                          # segmentation missing or all empty
        return [], {"study_id": study_id, "error": "no segmentation"}

    denoise_rounds, n_cand = 0, 0          # reported in the summary even if we
                                           # bail out early, so they must exist

    #loads kidney and anatomical masks, creates a kidney search region
    roi, kidney_dist = build_roi(masks, vol.shape, spacing, kidney_only)


    kidney_mask = np.zeros(vol.shape, bool)
    for k in ("kidney_left", "kidney_right"):
        if k in masks:
            kidney_mask |= masks[k]
    #rejects enhanced/excretory scans using kidney HU 
    # MEDIAN parenchymal HU is the phase test: unenhanced kidney ~30 HU,
    # enhanced ~96 HU. Median, so a big stone inside the mask cannot skew it.
    kid_med = float(np.median(vol[kidney_mask])) if kidney_mask.any() else np.nan
    kid_vol_ml = float(kidney_mask.sum() * voxel_mm3 / 1000.0)   # mm3 -> mL
    # recorded for QC only; deliberately NOT used to decide the phase (see below)
    dense_in_kidney = float((vol[kidney_mask] >= SEED_HU).sum() * voxel_mm3) \
        if kidney_mask.any() else 0.0

    enhanced = kid_med > KIDNEY_PLAIN_MAX_HU     # 60 HU sits between the two
    if enhanced:
        return [], {"study_id": study_id, "n_stones": 0, "n_candidates_raw": 0,
                    "kidney_median_hu": round(kid_med, 0),
                    "kidney_volume_ml": round(kid_vol_ml, 1),
                    "dense_in_kidney_mm3": round(dense_in_kidney, 0),
                    "roi": "kidney_only" if kidney_only else "whole_tract",
        "denoised": bool(denoise),
        "denoise_rounds": denoise_rounds,
                    "masks_found": len(masks),
                    "error": "enhanced or excretory phase - not analysable "
                             "for stones"}

    dense = vol >= BONE_HU               # everything bone-bright, anywhere
    dlab, dn = cc3d.connected_components(dense, connectivity=26, return_N=True)
    dstats = cc3d.statistics(dlab)
    big = np.zeros(dn + 1, bool)         # lookup table: is label i big enough?
    # SIZE is the discriminator. Bone is a large connected dense structure; a
    # calculus is not. Even a big staghorn is a few hundred mm3, well under
    # 3000, so this catches ribs and femurs that were never segmented at all.
    big[1:] = (dstats["voxel_counts"][1:] * voxel_mm3) >= BONE_MIN_VOL_MM3
    bone_seed = big[dlab]                # fancy-index the LUT by label -> mask
    # add the bones TotalSegmentator did label, for the cases where a bone is
    # split into pieces each below the size limit
    for b in ("vertebrae_L1", "vertebrae_L5", "sacrum", "hip_left", "hip_right"):
        if b in masks:
            bone_seed |= masks[b]

    # NOTHING INSIDE THE KIDNEY IS BONE. There is no bone in the renal
    # parenchyma or collecting system, so this cannot erase real bone -- ribs
    # and vertebrae lie outside the mask by anatomy.
    #
    # Why it is needed. The size rule above says "a dense component bigger than
    # BONE_MIN_VOL_MM3 is bone". A calculus is above BONE_HU too, so a stone
    # touching a rib merges with it into ONE component, that component clears
    # 3000 mm3, and the whole thing -- stone included -- is declared bone. The
    # stone is then 100% bone-adjacent and rejected, with nothing in the output
    # to say a stone was lost.
    #
    # Measured on 8513308: the merged component was 5043 mm3, of which 478 mm3
    # lay inside the kidney mask. The report describes a 20.2 x 12.0 mm renal
    # pelvic calculus. Every one of the 163 bone rejections in run_v4 had
    # bone_frac EXACTLY 1.00 -- the code was not finding candidates near bone,
    # it had classified them AS bone.
    #
    # This also protects a large staghorn, which can exceed 3000 mm3 on its own
    # and would otherwise be declared bone purely for being big.
    #
    # Deliberately the kidney mask only, with no outward cuff: a rib can pass
    # within a few millimetres of the kidney surface, so cuffing this would
    # start erasing real bone and turn rib cortex into reported stones.
    # Consequence: a stone sitting in the pelvis OUTSIDE the parenchyma mask is
    # still exposed to this failure. Revisit once the masks are better.
    bone_seed &= ~kidney_mask

    bone_dist = dist_mm(bone_seed, spacing)      # "how far from bone am I?"
    
    # Vascular calcification is the single biggest false positive for stones,
    # especially along the iliac arteries where they cross the distal ureter.
    vessel = np.zeros(vol.shape, bool)
    for v in ("aorta", "inferior_vena_cava", "iliac_artery_left",
              "iliac_artery_right"):
        if v in masks:
            vessel |= masks[v]
    vessel_dist = dist_mm(vessel, spacing) if vessel.any() else None

    bb = np.nonzero(roi) #here we get the coordinates of all voxels inside the kidney ROI

    #now we create bounding box containing the ROI, with 6mm padding on every side 
    #reason for cropping is denosiing kidney sized sub-vol is much cheaper than denoising the entire CT
    crop = tuple(slice(max(0, int(a.min()) - int(np.ceil(6.0 / sp))),
                       min(nn, int(a.max()) + int(np.ceil(6.0 / sp)) + 1))
                 for a, nn, sp in zip(bb, vol.shape, spacing)) \
        if len(bb[0]) else tuple(slice(0, nn) for nn in vol.shape)
    #this is kidney ROI inside the crop
    sub_roi = roi[crop]
    #clip between -200 to 1000HU - we are clipping only for detection/denoising
    det_sub = np.clip(vol[crop], CLIP_LOW, CLIP_HIGH).astype(np.float32)
    # CUMULATIVE denoising with an adaptive stopping rule (Elton et al.): filter,
    # recount blobs, filter again, until the count drops below 200. A fixed
    # iteration count would under-denoise a noisy scan and over-denoise a clean
    # one; the blob count is a direct measure of how much noise is left.
    for denoise_rounds in range(1, (DENOISE_MAX_ROUNDS if denoise else 0) + 1):
        det_sub = denoise_ct(det_sub, DENOISE_ITERS)   # feeds its own output back
        _, n_cand = cc3d.connected_components(
            (det_sub >= GROW_HU) & sub_roi, connectivity=26, return_N=True)
        if n_cand < DENOISE_TARGET_CC:     # quiet enough -> stop early
            break
   
    # 3-4. threshold the DENOISED image at 130 HU inside the kidney, label it
    cand_lab, n_cand = cc3d.connected_components(
        (det_sub >= GROW_HU) & sub_roi, connectivity=26, return_N=True)
    #this produces final denoised candidate label image : 0 for BG, 1 for C1 and 2 for C2
    #non denoised CT is obtained here 
    raw_sub = vol[crop] #original CT cropped around kidney ROI and vol is the entire CT
    
    raw_lab, n_raw = cc3d.connected_components(
        (raw_sub >= GROW_HU) & sub_roi, connectivity=26, return_N=True) #here the thresholded raw CT has many connected bright objects

    #cand_lab - candidates found on denoised CT
    #raw_lab  - candidates found on org CT  >=130 HU
    #n_raw = number of raw components
    #n_cand = number of denoised components
    #Example : grown = {7: 310.0,19: 190.0}
     
    #here we are connecting the denoised candidates to the raw components
    grown = {}                     # raw component id -> peak HU of its seed
    if n_cand:
        for ic in range(1, n_cand + 1):
            m = cand_lab == ic # for every denoised candidate , m is the boolean mask
            if not m.any():
                continue
            peak = float(raw_sub[m].max()) #find the brightest raw CT value at the denoised can location
            # WHERE that brightest voxel is. np.where(m, raw_sub, -3000) blanks
            # everything outside the candidate to an impossibly low HU so argmax
            # can only land inside it; unravel_index turns the flat index into
            # a 3D coordinate.
            seed = np.unravel_index(int(np.argmax(np.where(m, raw_sub, -3000))),
                                    raw_sub.shape)
            rid = int(raw_lab[seed])       # which RAW component contains that seed
            if rid == 0:
                continue                   # seed landed on background; discard
            # if two denoised candidates seed the same raw component they merge
            # here, and the higher of the two peaks wins
            grown[rid] = max(grown.get(rid, -3000.0), peak)


            #Several denoised candidates may point into the same raw component. If so, they are merged under the same rid, and the largest seed peak is retained.
            #This is the region-growing step: the seed comes from the denoised detection, while the full extent comes from its connected raw-CT component.

    # From here the labelled volume is the RAW-grown one, restricted to the
    # components a denoised candidate actually seeded. The kept raw ids are
    # sparse (e.g. {3, 17, 25}), so they are RELABELLED to a contiguous 1..K --
    # cc3d.statistics returns arrays sized by label count, and indexing them
    # with an original sparse id runs off the end.

    #Denoised component
       # │
       # │ #choose brightest RAW voxel inside it
       # ▼
    #Seed coordinate
        #│
        #│ look up raw_lab at that coordinate
        #▼
    #Raw ≥130 HU component
        #│
        #│ retain its full connected extent
        #▼
    #Candidate for filtering and measurement
    

    #Map each denoised candidate to the raw >=130 HU connected component
    kept = sorted(grown.keys())          # the sparse raw ids we are keeping
    # lookup table old id -> new id. Entries left at 0 mean "drop this label",
    # which is how unseeded raw components disappear for free.
    remap = np.zeros(int(raw_lab.max()) + 1, np.int32)
    for new_id, old_id in enumerate(kept, start=1):
        remap[old_id] = new_id
    labels = np.zeros(vol.shape, np.int32)   # back to full-volume coordinates
    labels[crop] = remap[raw_lab]        # one fancy-index renumbers everything
    peak_of = {new_id: grown[old_id]     # carry the peaks across the renumbering
               for new_id, old_id in enumerate(kept, start=1)}
    det = np.zeros(vol.shape, np.float32)    # denoised volume, full size again
    det[crop] = det_sub
    n_bright_raw = int(((vol >= GROW_HU) & roi).sum())   # QC: how much was bright
    n = len(kept)
    # rescue stones fused to bone BEFORE any rejection happens, so a mixed
    # component gets the chance to be judged as two separate objects
    labels, peak_of, n, n_bridges = split_bone_bridges(
        labels, n, peak_of, bone_dist, vol, voxel_mm3)
    seed_labels = set(peak_of.keys())    # fast membership test in the loop
    grown = peak_of                      # `grown` now means "post-split peaks"
    stats = cc3d.statistics(labels)      # recomputed: labels changed above

    kidneys = {}                         # per-side long axis, for pole location
    for k in ("kidney_left", "kidney_right"):
        if k in masks:
            kidneys[k] = kidney_frame(masks[k], spacing)

    bladder = masks.get("urinary_bladder")   # may be None; checked before use
    rows, rejected = [], {}              # one row per candidate; tally of reasons

    def drop(reason):                    # count a rejection without discarding
        rejected[reason] = rejected.get(reason, 0) + 1   # the row itself
    #loop through denoised candidates 
    for lab in range(1, n + 1):
        # Stage 1 (Elton et al.): 130 HU + connected components + 3-voxel floor.
        # Everything past that point is OUR interim heuristic standing in for
        # their CNN, so those candidates are measured and kept in the output with
        # a reason attached rather than thrown away -- they are exactly the
        # labelled pool the Part 2 classifier needs.
        nvox = int(stats["voxel_counts"][lab])
        if lab not in seed_labels or nvox == 0:
            continue                       # not seeded by a denoised candidate
        if nvox * voxel_mm3 < MIN_CANDIDATE_MM3:
            drop("below_min_volume")       # below 0.25 mm3, i.e. sub-1 mm sphere
            continue
        # THE hysteresis test: extent came from 130 HU, but to BE a stone the
        # object must contain a voxel at SEED_HU. An empty string means "kept";
        # any other value is the reason it is not counted as a stone.
        reason = "" if grown.get(lab, 0.0) >= SEED_HU else "no_dense_core"
        bb = stats["bounding_boxes"][lab]
        margin = [int(np.ceil(6.0 / s)) for s in spacing]   # 6 mm, in voxels
        # pad the box and clamp to the volume: FWHM needs to see OUTSIDE the
        # blob to find its local background
        sl = tuple(slice(max(0, bb[i].start - margin[i]),
                         min(vol.shape[i], bb[i].stop + margin[i]))
                   for i in range(3))
        comp = np.zeros(vol.shape, bool)   # full-size mask holding just this blob
        comp[sl] = labels[sl] == lab       # written only inside the small box

        # bone partial volume: most of the object sits inside the bone margin.
        # A majority rule, not "any voxel", so a genuine stone merely lying
        # close to the spine or sacrum is not thrown away.
        bone_frac = float((bone_dist[comp] < BONE_MARGIN_MM).mean())
        vess_frac = (float((vessel_dist[comp] <= VESSEL_MARGIN_MM).mean())
                     if vessel_dist is not None else 0.0)
        if not reason and bone_frac > 0.5:
            reason = "bone_partial_volume"
        if not reason and vess_frac > 0.5:
            reason = "vascular_calcification"

        # measure on the PRISTINE volume, never the denoised one
        refined_sub, thr, peak, bg, pv_mm3 = fwhm_measure(vol, comp, spacing, sl)
        vals = vol[sl][refined_sub]  # HU of the refined extent, for the statistics
        volume_mm3 = pv_mm3          # partial-volume integrated; see fwhm_measure
        # kept alongside so you can see per stone how much integration changed it
        volume_voxelcount_mm3 = float(refined_sub.sum() * voxel_mm3)

        # Geometry from the sub-voxel half-maximum surface. Falls back to the
        # voxel-grid caliper only when the surface will not close, which in
        # practice means a candidate of a few voxels.
        shape = shape_metrics(vol[sl], refined_sub, spacing, thr)
        if shape is None:
            shape = {"max_diameter_mm": max_diameter_mm(refined_sub, spacing)}
        dmax = shape["max_diameter_mm"]
        shape_source = "mesh" if len(shape) > 1 else "voxel"
        # `not reason and` throughout: the FIRST failure wins, so the recorded
        # reason is the earliest one in the chain rather than the last tested
        if not reason and dmax < MIN_DIAM_MM:
            reason = "below_min_diameter"          # too small to report on
        # flagged for human review, NOT rejected -- a 30 mm+ object is either a
        # genuine staghorn or a segmentation failure, and only a person can say
        oversized = dmax > MAX_STONE_DIAM_MM
        if reason:
            drop(reason)                           # tally it; the row still goes out

        # centroid inside the crop -> whole-volume voxel index -> millimetres
        cen_local = np.array(ndimage.center_of_mass(refined_sub))
        cen_vox = cen_local + np.array([s.start for s in sl])   # undo the crop
        cen_world = cen_vox * np.array(spacing)    # for distance arithmetic
        cen_idx = tuple(int(round(v)) for v in cen_vox)   # for mask lookups

        # compartment + side
        comp_name, side, zone, tpos, conf = "unknown", "", "", np.nan, np.nan
        for k, (c, axis, world) in kidneys.items():
            if masks[k][cen_idx]:                  # centroid lands inside a kidney
                comp_name = "kidney"
                side = "left" if k.endswith("left") else "right"
                zone, tpos, conf = polar_zone(cen_world, c, axis, world)
                break
        if comp_name == "unknown" and bladder is not None and bladder[cen_idx]:
            comp_name = "bladder"
        if comp_name == "unknown":                 # outside every labelled organ
            # nearest kidney centroid decides the side; the precomputed kidney
            # distance map decides peri-renal vs ureteric
            best, bestd = "", 1e9                  # nearest kidney centroid wins
            for k, (c, axis, world) in kidneys.items():
                d = np.linalg.norm(cen_world - c)
                if d < bestd:
                    best, bestd = k, d
            if best:
                side = "left" if best.endswith("left") else "right"
            near = (kidney_dist is not None
                    and kidney_dist[cen_idx] <= KIDNEY_DILATE_MM)
            # close to the kidney = still peri-renal; far = down the ureter
            comp_name = "renal_pelvis_or_perirenal" if near else "ureter"

        rows.append({
            "study_id": study_id,
            # numbered over ACCEPTED stones only, so stone_id matches what a
            # clinician would count; rejected candidates carry the running value
            "stone_id": sum(1 for r in rows if r["reject_reason"] == "") + 1,
            "compartment": comp_name,
            "reject_reason": reason,
            "is_stone": reason == "",
            # the value the no_dense_core test actually compared against
            # SEED_HU. Without it a rejection reading "no_dense_core" next to
            # "hu_max 575" looks like a contradiction -- it is not, because
            # hu_max spans the wider FWHM-refined extent while this is the
            # candidate's own peak.
            "seed_peak_hu": round(float(grown.get(lab, 0.0)), 0),
            "bone_frac": round(bone_frac, 2),
            "vessel_frac": round(vess_frac, 2),
            "oversized_review": bool(oversized),
            "side": side,
            "location": zone,
            "axis_pos_0to1": tpos,
            "location_conf": conf,
            "volume_mm3": round(volume_mm3, 1),
            "volume_voxelcount_mm3": round(volume_voxelcount_mm3, 1),
            "max_diameter_mm": round(dmax, 2),
            # Three dimensions, from shape_metrics(). axis_* are the stone's
            # own principal axes (rotation-invariant); dim_* are the scanner
            # axes a radiologist reads off axial and coronal images.
            **{k: (round(v, 2) if isinstance(v, float) and not np.isnan(v)
                   else v)
               for k, v in shape.items() if k != "max_diameter_mm"},
            "shape_source": shape_source,
            # A solid object of maximum caliper d cannot exceed the volume of
            # the sphere of diameter d. When it does, the pair of numbers is
            # physically impossible and at least one of them is wrong -- do not
            # let that leave silently. Before the mesh fix this fired on 17 of
            # 63 kept stones, and on 83% of those under 2 mm.
            "geom_consistent": bool(
                volume_mm3 <= (np.pi / 6.0) * dmax ** 3 * 1.001),
            "n_voxels": int(refined_sub.sum()),
            "hu_max": round(float(vals.max()), 0),
            "hu_mean": round(float(vals.mean()), 0),
            "hu_sd": round(float(vals.std()), 0),
            "hu_p90": round(float(np.percentile(vals, 90)), 0),
            "fwhm_threshold": round(thr, 0),
            "local_background_hu": round(bg, 0),
            "centroid_vox": [int(round(v)) for v in cen_vox],
        })

    kept = [r for r in rows if r["reject_reason"] == ""]
    summary = {
        "study_id": study_id,
        "n_candidates_raw": n,
        "n_rejected": sum(rejected.values()),
        **{f"rej_{k}": v for k, v in rejected.items()},
        "n_candidates_stage1": len(rows),
        "n_bone_bridges_split": n_bridges,
        "n_stones": len(kept),
        "bright_voxels_in_roi": n_bright_raw,
        # ACCEPTED stones only. Using `rows` here counted rejected candidates
        # (bone, no-dense-core) toward the size and volume, which put a 20 mm
        # rib fragment in the "largest stone" column of a study with one real
        # 2.7 mm calculus.
        "total_volume_mm3": round(sum(r["volume_mm3"] for r in kept), 1),
        "largest_mm": round(max([r["max_diameter_mm"] for r in kept], default=0), 2),
        "spacing_mm": f"{spacing[0]:.2f}x{spacing[1]:.2f}x{spacing[2]:.2f}",
        "roi": "kidney_only" if kidney_only else "whole_tract",
        "denoised": bool(denoise),
        "denoise_rounds": denoise_rounds,
        "kidney_median_hu": round(kid_med, 0),
        "kidney_volume_ml": round(kid_vol_ml, 1),
        "dense_in_kidney_mm3": round(dense_in_kidney, 0),
        "masks_found": len(masks),
        "error": "",
    }
    if verbose:
        rj = " ".join(f"{k}={v}" for k, v in sorted(rejected.items()))
        print(f"  candidates={n} kept={len(rows)} | rejected: {rj}")
    # rows = one per CANDIDATE (kept and rejected alike); summary = one per study
    return rows, summary
# WHAT THIS FUNCTION DOES: the whole per-study pipeline, in order --
#
#   1. load the CT and its organ masks
#   2. build the kidney search region (build_roi)
#   3. PHASE GATE: bail out if the kidney parenchyma is enhanced, because
#      excreted contrast in the collecting system is indistinguishable from stone
#   4. work out where bone and vessels are, so their calcium can be rejected
#   5. denoise a CROP of the volume until the blob count settles
#   6. threshold the denoised copy at 130 HU to find candidate seeds
#   7. region-grow each seed on the ORIGINAL CT to get its true extent
#   8. split any candidate fused to bone (split_bone_bridges)
#   9. for each candidate: reject or keep, then measure it (fwhm_measure,
#      shape_metrics) and locate it (kidney_frame, polar_zone)
#
# The key design decision is that REJECTED candidates are measured and returned
# too, each tagged with `reject_reason`. Nothing is silently dropped: the CSV is
# both the result and the audit trail, and those rejected rows are exactly the
# labelled negatives a Part 2 classifier will need.


def _analyse_one(sid, kw):
    """Top-level wrapper so multiprocessing can pickle it (a closure cannot).

    verbose=False in workers: their stdout interleaves unreadably. The parent
    prints one ordered line per study as results come back.
    """
    return analyse(sid, verbose=False, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", default=None, help="single study id")
    ap.add_argument("--studies", nargs="*", default=None,
                    help="only these study ids. Unlike --debug this takes a "
                         "list, so a handful of studies can be re-run after a "
                         "triage or mask change without redoing all 137 (~9 h).")
    ap.add_argument("--no-denoise", action="store_true",
                    help="skip anisotropic diffusion before thresholding")
    ap.add_argument("--whole-tract", action="store_true",
                    help="also search bladder + straight-line ureteric corridor "
                         "(noisy; kidneys only by default)")
    ap.add_argument("--out", default=os.path.join(CSV, "baseline_stones.csv"))
    # Studies are independent, so they parallelise perfectly. Serial by
    # default so behaviour does not change unless asked: detection is
    # deterministic per study, so results are IDENTICAL either way -- only the
    # wall clock differs (measured: ~4 min/study serial, 137 studies = ~9 h).
    # Each worker holds one volume (~1 GB) plus denoise working copies, so
    # budget ~4 GB per worker and check `free` before going wide.
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel studies (default 1 = serial)")
    ap.add_argument("--overwrite", action="store_true",
                    help="redo studies that already have per-study results "
                         "(default is to resume and skip them)")
    args = ap.parse_args()

    # one study if --debug, else every NIfTI on disk. The double strip handles
    # the ".nii.gz" double extension that splitext only removes one layer of.
    ids = ([args.debug] if args.debug else
           [str(s) for s in args.studies] if args.studies else
           sorted(os.path.splitext(os.path.basename(p))[0].replace(".nii", "")
                  for p in glob.glob(os.path.join(NIFTI, "*.nii.gz"))))
    # skip anything not yet segmented, so this can run while TotalSegmentator
    # is still working through the rest
    ids = [i for i in ids if os.path.isdir(os.path.join(SEG, i))]
    print(f"analysing {len(ids)} studies\n")

    # PER-STUDY OUTPUT, WRITTEN AS EACH STUDY FINISHES.
    #
    # This used to accumulate everything in memory and write three CSVs at the
    # very end. Two things went wrong with that:
    #   * a run that dies at study 130 of 137 loses ALL of it. A worker pool
    #     broke its result pipe at study 27 and 45 minutes of compute vanished.
    #   * nothing is inspectable while it runs -- the results directory sits
    #     empty for hours, so there is no way to sanity-check early studies.
    # Writing one small CSV per study fixes both, and makes the run RESUMABLE:
    # a study whose file already exists is skipped, so a re-run continues
    # instead of starting over.
    per = os.path.join(CSV, "per_study")
    os.makedirs(per, exist_ok=True)
    kw = dict(kidney_only=not args.whole_tract, denoise=not args.no_denoise)

    if not args.overwrite:
        done = {s for s in ids
                if os.path.exists(os.path.join(per, f"{s}_summary.csv"))}
        if done:
            print(f"resuming: {len(done)} studies already have results, "
                  f"{len(ids) - len(done)} to do")
            ids = [s for s in ids if s not in done]
    print(f"analysing {len(ids)} studies -> {per}\n", flush=True)

    def write_one(sid, rows, summ):
        """Persist a single study immediately. Nothing is held in memory."""
        # candidates first, summary LAST: the summary file is what `resume`
        # checks, so it must only appear once the study is fully written.
        if rows:
            pd.DataFrame(rows).to_csv(
                os.path.join(per, f"{sid}_candidates.csv"), index=False)
        pd.DataFrame([summ]).to_csv(
            os.path.join(per, f"{sid}_summary.csv"), index=False)

    def collect(i, sid, res):
        rows, summ = res
        write_one(sid, rows, summ)
        print(f"[{i}/{len(ids)}] {sid}: {summ.get('n_stones', 0)} stones "
              f"from {summ.get('n_candidates_raw', 0)} candidates", flush=True)

    
    def failed(i, sid, e):
        # one bad study must not lose the others. Record the failure as a
        # summary row so it is visible in the CSV rather than just in
        # scrollback, and carry on.
        print(f"[{i}/{len(ids)}] {sid} ERROR {type(e).__name__}: {e}", flush=True)
        write_one(sid, [], {"study_id": sid,
                            "error": f"{type(e).__name__}: {e}"})
    if args.workers > 1 and len(ids) > 1:
        # Each worker is a separate process, so SimpleITK and numpy must be told
        # not to spawn their own thread pools as well -- otherwise N workers x
        # M internal threads oversubscribes the CPU and runs SLOWER than serial.
        import concurrent.futures as cf
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
            os.environ[var] = "1"
        import multiprocessing as mp
        # ProcessPoolExecutor, not multiprocessing.Pool: when a worker dies or
        # its result pipe breaks, the executor raises BrokenProcessPool on the
        # next future instead of blocking forever in .get(). The Pool version
        # of this loop hung silently at study 27.
        # 'spawn' because the parent has already imported SimpleITK, and
        # forking a process that holds a thread pool can deadlock the child.
        print(f"running {args.workers} studies in parallel\n", flush=True)
        try:
            with cf.ProcessPoolExecutor(
                    max_workers=args.workers,
                    mp_context=mp.get_context("spawn")) as ex:
                futs = {ex.submit(_analyse_one, sid, kw): sid for sid in ids}
                for i, f in enumerate(cf.as_completed(futs), 1):
                    sid = futs[f]
                    try:
                        collect(i, sid, f.result())
                    except Exception as e:
                        failed(i, sid, e)
        except Exception as e:
            # the pool itself died. Everything finished so far is already on
            # disk, so re-running resumes rather than restarting.
            print(f"\nPOOL FAILED ({type(e).__name__}: {e}) -- results so far "
                  f"are in {per}; re-run to resume", flush=True)
    else:
        for i, sid in enumerate(ids, 1):
            try:
                collect(i, sid, analyse(sid, verbose=False, **kw))
            except Exception as e:
                failed(i, sid, e)

    # gather every per-study file, including ones from earlier partial runs
    # EXCLUDE Part 2's files. detect_ureteric writes <id>_ureter_candidates.csv
    # and <id>_ureter_summary.csv into this SAME folder, and "*_candidates.csv"
    # matches them -- so 220 ureteric rows were merged into baseline_stones.csv
    # (NaN compartment, NaN stone_id) and the ureteric per-study summaries were
    # appended to baseline_summary.csv, giving 226 rows for 141 studies. That
    # corrupts the kidney stone table and double-counts studies in scoring.
    def part1_only(pattern):
        return [f for f in sorted(glob.glob(os.path.join(per, pattern)))
                if "_ureter_" not in os.path.basename(f)]

    all_rows, summaries = [], []
    for f in part1_only("*_candidates.csv"):
        all_rows += pd.read_csv(f).to_dict("records")
    for f in part1_only("*_summary.csv"):
        summaries += pd.read_csv(f).to_dict("records")

    os.makedirs(CSV, exist_ok=True)   # a fresh CALCULUS_RUN has no csv/ yet
    cand = pd.DataFrame(all_rows)
    if len(cand):
        # every stage-1 candidate, measured, with the reason it was filtered.
        # This is the labelling pool for the Part 2 CNN.
        cand.to_csv(os.path.join(CSV, "candidates.csv"), index=False)
    # baseline_stones.csv is the ACCEPTED subset -- the clinical answer.
    # candidates.csv above is everything, including what we threw out and why.
    # reject_reason is "" in memory, but the per-study CSVs round-trip it back
    # as NaN -- so `== ""` matched NOTHING and baseline_stones.csv was never
    # written. That silently crashed render_overlays and produced zero overlays,
    # twice. Normalise before comparing, and write even when empty so the header
    # alone keeps the downstream scripts alive.
    rr = (cand.reject_reason.fillna("").astype(str).str.strip()
          if len(cand) else None)
    st = cand[rr == ""] if len(cand) else cand
    if len(cand):
        st.to_csv(args.out, index=False)
    su = pd.DataFrame(summaries)          # one row per study, errors included
    su.to_csv(os.path.join(CSV, "baseline_summary.csv"), index=False)
    print(f"\n{len(st)} stones kept from {len(cand)} stage-1 candidates "
          f"across {len(ids)} studies")
    print(f"wrote {args.out}\n      {os.path.join(CSV,'candidates.csv')}"
          f"\n      {os.path.join(CSV,'baseline_summary.csv')}")
    if len(cand):
        print("\nstage-1 candidates by outcome:")
        print(cand.reject_reason.replace("", "KEPT as stone")
              .value_counts().to_string())
    if len(st):
        print("\nby compartment:")
        print(st.compartment.value_counts().to_string())
        print("\nby location:")
        # location is blank for anything outside a kidney, so filter it out
        print(st[st.location != ""].location.value_counts().to_string())


# WHAT THIS FUNCTION DOES: the command-line entry point. Finds every study that
# has both a NIfTI and a segmentation, runs analyse() on each, and writes three
# CSVs -- candidates.csv (everything, with reasons), baseline_stones.csv (the
# accepted stones), baseline_summary.csv (one row per study). A study that
# raises is logged and skipped rather than killing the batch, which matters when
# a run takes hours.


if __name__ == "__main__":       # only runs when invoked directly, so the
    main()                       # module stays importable by the test harness
