"""Step 2 QC: are the TotalSegmentator kidney masks actually any good?

Every number the pipeline produces sits on top of these masks. If a kidney
outline is wrong, the stone count, the pole assignment and the phase gate are all
wrong with it -- and silently so. This script checks the masks alone, before any
stone logic is involved.

Per kidney it reports volume, craniocaudal length, and the median parenchymal HU,
then flags anything outside normal adult ranges:

    normal single kidney   ~120-180 mL, 9-13 cm long
    unenhanced parenchyma  ~25-45 HU

A flag is not automatically an error -- atrophic, obstructed, transplanted and
post-surgical kidneys are genuinely abnormal. It means "look at this one".

Writes csv/kidney_qc.csv and prints a table.

Usage:
    ./venv/bin/python kidney_qc.py
"""



import glob
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd

# this file lives in utils/, so the project root is one level up. All data
# directories hang off ROOT, never off the script's own folder.
HERE = os.path.dirname(os.path.abspath(__file__))
# package root -> project root: calculus/<sub>/x.py is two levels down
ROOT = os.path.dirname(os.path.dirname(HERE))
from calculus.common.paths import CSV, NIFTI, SEG     # noqa: E402  results dir is per-run

# ---------------------------------------------------------------- thresholds
# WHY THESE CHANGED (the old values were 90-220 mL, 80-140 mm, one flat verdict)
#
# The old floor of 90 mL was taken from a textbook WHOLE-kidney reference range.
# TotalSegmentator's kidney class is PARENCHYMA ONLY -- no collecting system, no
# sinus fat -- so it legitimately measures less. Measured on our own 137 studies
# (272 kidney sides, zeros dropped):
#
#     volume mL     p5= 40.1  p25= 90.9  p50=112.8  p75=138.3  p95=180.1
#     cc length mm  p5= 65.5  p25= 86.0  p50= 95.0  p75=102.2  p95=118.4
#     median HU     p5= 20.0  p25= 26.0  p50= 29.0  p75= 33.0  p95= 39.0
#
# So the 90 mL floor sat at our OWN 25th percentile: a quarter of all kidney
# sides were flagged by construction. 59 of 137 studies came back "review", of
# which only 13 had anything wrong. The 46 false alarms buried the 13 real ones.
#
# NOT set to a percentile of our own distribution. That would be circular --
# our p5 is 40 mL BECAUSE bad masks are in the sample, so a p5 floor would
# quietly accept exactly the failures we are hunting. The FAIL thresholds below
# are anatomical statements (an adult kidney shorter than 65 mm is atrophic or
# truncated, full stop); the distribution was used only to check they do not
# fire on the healthy bulk.
#
# Two tiers, because "bad mask" and "unusual kidney" are different questions:




FAIL_ML, FAIL_LEN_MM = 40.0, 65.0     # almost certainly truncated -> unusable
REVIEW_ML, REVIEW_LEN_MM = 70.0, 80.0  # small but plausible -> human glance
ASYM_FAIL, ASYM_REVIEW = 2.5, 1.8      # larger/smaller volume ratio

#checking if CT is contrast enhanced - may not be always a case where we get bad masks , that is if kidney has HU value >70 its constrast 
HU_CONTRAST = 70.0
HU_MIN, HU_MAX = 15.0, 55.0            # plausible unenhanced parenchyma


#scan-boudary threshold
EDGE_SLICES = 1
# kept for the per-side flag strings, so existing readers of `flags` still work
VOL_MIN_ML, VOL_MAX_ML = REVIEW_ML, 240.0
LEN_MIN_MM, LEN_MAX_MM = REVIEW_LEN_MM, 130.0
ASYMMETRY = ASYM_REVIEW




#converts the left abd right kideny measurements into one result 
def qc_verdict(left, right, pelvis_present=False):
    """'ok' | 'review' | 'fail' | 'contrast' | 'cannot_assess' for one study.

    ORDER MATTERS, and it encodes a distinction that cost us a day of looking in
    the wrong place: is the INPUT unusable, or is the MASK wrong?

      cannot_assess  the scan itself cannot answer the question -- the kidneys
                     are outside the field of view, or sliced off at the edge.
                     No segmenter can fix this. Must not be reported as
                     "no calculi": that reads as a normal negative when the
                     truth is that nobody looked.
      contrast       wrong PHASE. The mask is usually GOOD here, because
                     TotalSegmentator was trained largely on contrast CT.
                     Five studies were miscounted as bad masks before this
                     verdict existed.
      fail           the scan is fine and the mask is genuinely truncated.
      review / ok    usable.

    Coverage is tested before contrast because median HU is measured INSIDE the
    kidney mask -- with no kidney in the image there is no HU to test.
    """
   
    #if we have hip/sacrum masks but neither kidney exists, that is the CT covers only pelvis and begins below kidneys
    if left is None and right is None:
        return "cannot_assess" if pelvis_present else "fail"

    #this can happen in edge case scenarios where the kidney may extend below or above the CT , thats why we have EDGE_SLICES=1 which says 0<=EDGE_SLICES 
    if clipped(left) or clipped(right):
        return "cannot_assess" #if part of my kidney masks lie outside CT then its wrong if kindey stone appears somewhere out of it 

    hus = [s["median_hu"] for s in (left, right) if s] #collect median HU from the available kidney masks
    #if eithr side has median HU > 70 , its contrast 

    if any(h > HU_CONTRAST for h in hus):
        return "contrast"

    #presence or absense of kidney masks
    if left is None or right is None:
        return "fail"                       # a side is missing entirely

    #vol and size failures 
    vols = [s["volume_ml"] for s in (left, right)]
    lens = [s["cc_length_mm"] for s in (left, right)]
    if min(vols) <= 0 or min(lens) <= 0:
        return "fail"
    if min(vols) < FAIL_ML or min(lens) < FAIL_LEN_MM:
        return "fail"

    #if one kidney is 2.5 times the volume of the other 
    asym = max(vols) / min(vols)

    if asym > ASYM_FAIL:
        return "fail"
    if min(vols) < REVIEW_ML or min(lens) < REVIEW_LEN_MM or asym > ASYM_REVIEW:
        return "review"
    
    return "ok"

#recieves one binary kidney mask, original CT vol and voxel spacing 
def measure_kidney(mask, vol, spacing):
    #count mask voxels, false/0 is outside kidney, true/1 is inside kidney, summing gives the mask for number of kidney voxels
    n = int(mask.sum())
    if n == 0:
        return None

    #For every axial slice, this checks whether any kidney voxel exists., so zs has all slice indices occupied by kidney
    zs = np.where(mask.any(axis=(0, 1)))[0]

    nz = mask.shape[2] #gets total number of CT slices


    return {
        "volume_ml": float(n * np.prod(spacing) / 1000.0),
        "cc_length_mm": float((zs[-1] - zs[0] + 1) * spacing[2]),
        "median_hu": float(np.median(vol[mask])),
        "n_voxels": n,
        # slices of scan left BEYOND each end of the kidney. Zero means the
        # kidney runs off the edge of the image -- see clipped() below.
        "slices_below": int(zs[0]),
        "slices_above": int(nz - 1 - zs[-1]),
    }
# WHAT THIS FUNCTION DOES: measures one kidney mask -- volume, head-to-toe
# length, median density -- and also records how much scan is left beyond each
# end of it, which is what tells us whether the kidney was cut off by the edge
# of the scan rather than by the segmenter.


def clipped(side_stats, edge_slices=EDGE_SLICES):
    """Is this kidney cut off by the edge of the SCAN?

    A mask that runs right up to the first or last slice means the kidney
    continues past the edge of the image. The segmenter did not truncate it --
    the acquisition did, and no model can recover what was never scanned.

    Measured on 8271213: kidneys at z 339-380 in a 381-slice scan, i.e. zero
    slices of headroom, reported as 37 and 32 mL. That is not a small kidney,
    it is the bottom 40 mm of a normal one. Controls carry 83-580 slices of
    headroom at both ends, so this is not a marginal distinction.
    """
    if side_stats is None:
        return False
    return (side_stats["slices_below"] <= edge_slices
            or side_stats["slices_above"] <= edge_slices)
# WHAT THIS FUNCTION DOES: answers whether a kidney touches the top or bottom
# slice of the scan, meaning part of it lies outside the image entirely.


def main():
    rows = []
    ids = sorted(os.path.splitext(os.path.basename(p))[0].replace(".nii", "")
                 for p in glob.glob(os.path.join(NIFTI, "*.nii.gz")))
    for sid in ids:
        nii = nib.load(os.path.join(NIFTI, f"{sid}.nii.gz"))
        vol = np.asanyarray(nii.dataobj).astype(np.float32)
        spacing = tuple(float(v) for v in nii.header.get_zooms()[:3])
        row = {"study_id": sid,
               "voxel_mm": f"{spacing[0]:.2f}x{spacing[1]:.2f}x{spacing[2]:.2f}"}
        got = {}
        for side in ("left", "right"):
            p = os.path.join(SEG, sid, f"kidney_{side}.nii.gz")
            m = (np.asanyarray(nib.load(p).dataobj) > 0) if os.path.exists(p) \
                else np.zeros(vol.shape, bool)
            r = measure_kidney(m, vol, spacing)
            got[side] = r
            row[f"{side}_ml"] = round(r["volume_ml"], 1) if r else 0.0
            row[f"{side}_len_mm"] = round(r["cc_length_mm"], 0) if r else 0.0
            row[f"{side}_hu"] = round(r["median_hu"], 0) if r else np.nan
            # how much scan is left beyond each end -- 0 means clipped
            row[f"{side}_slices_below"] = r["slices_below"] if r else -1
            row[f"{side}_slices_above"] = r["slices_above"] if r else -1
            row[f"{side}_clipped"] = clipped(r)

        # Pelvic bone present but no kidneys => the scan starts below the
        # kidneys, rather than the segmenter having missed them. Read from the
        # masks we already have, so this costs nothing extra.
        pelvis_present = False
        for b in ("hip_left", "hip_right", "sacrum"):
            p = os.path.join(SEG, sid, f"{b}.nii.gz")
            if os.path.exists(p) and (np.asanyarray(nib.load(p).dataobj) > 0).any():
                pelvis_present = True
                break
        row["pelvis_in_scan"] = pelvis_present
        row["n_slices"] = int(vol.shape[2])

        flags = []
        for side, r in got.items():
            if r is None:
                flags.append(f"{side}_absent")
                continue
            if not VOL_MIN_ML <= r["volume_ml"] <= VOL_MAX_ML:
                flags.append(f"{side}_vol")
            if not LEN_MIN_MM <= r["cc_length_mm"] <= LEN_MAX_MM:
                flags.append(f"{side}_len")
            if not HU_MIN <= r["median_hu"] <= HU_MAX:
                flags.append(f"{side}_hu")
        if got["left"] and got["right"]:
            a, b = got["left"]["volume_ml"], got["right"]["volume_ml"]
            if max(a, b) / max(min(a, b), 1e-6) > ASYMMETRY:
                flags.append("asymmetric")

        # coverage flags must go in BEFORE `flags` is serialised below -- the
        # first version appended them afterwards, so they never reached the CSV
        for side in ("left", "right"):
            if row[f"{side}_clipped"]:
                flags.append(f"{side}_clipped")
        if got["left"] is None and got["right"] is None and pelvis_present:
            flags.append("fov_no_kidneys")

        row["total_ml"] = round(row["left_ml"] + row["right_ml"], 1)
        row["flags"] = ";".join(flags)
        # the verdict is decided by qc_verdict, NOT by "any flag fired". A flag
        # firing used to mean 'review', which is how 59 of 137 studies ended up
        # in the review queue when only 13 had a real problem.
        row["qc"] = qc_verdict(got["left"], got["right"], pelvis_present)
        rows.append(row)

    df = pd.DataFrame(rows)
    out = os.path.join(CSV, "kidney_qc.csv")
    df.to_csv(out, index=False)

    cols = ["study_id", "voxel_mm", "left_ml", "right_ml", "total_ml",
            "left_len_mm", "right_len_mm", "left_hu", "right_hu", "qc", "flags"]
    print(df[cols].to_string(index=False))
    n = len(df)
    print(f"\n{'verdict':>10}  {'n':>4}   meaning")
    for v, why in (("ok", "usable"),
                   ("review", "small but plausible - a human should look"),
                   ("fail", "mask truncated or missing - do not trust"),
                   ("contrast", "wrong PHASE, not a mask problem"),
                   ("cannot_assess", "the SCAN cannot answer it - kidneys "
                                     "outside or cut off by the FOV")):
        c = int((df.qc == v).sum())
        print(f"{v:>14}  {c:>4}   {why}")
    for v in ("cannot_assess", "fail", "contrast"):
        s = df[df.qc == v]
        if len(s):
            print(f"\n{v.upper()} studies: " + ", ".join(map(str, s.study_id)))
    print(f"\nwrote {out}")
    print("visual check: 3d_kidneys/<study_id>/views.png  (shape - is it a "
          "kidney?)\n              overlays/<study_id>/_kidney_axials.png")


if __name__ == "__main__":
    main()
