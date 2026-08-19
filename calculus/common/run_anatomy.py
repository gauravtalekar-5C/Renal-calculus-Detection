"""Run TotalSegmentator on the extracted NIfTI volumes.

Only the structures the stone pipeline needs are requested (--roi_subset), which
is far faster than the full 117-class run:

    kidney_left / kidney_right   search region + polar location frame
    kidney_cyst_*                cyst walls calcify and mimic stones
    urinary_bladder              distal end of the tract, UVJ reference
    aorta / IVC / iliac arteries vascular calcification, the main false positive
    vertebrae_L1 / L5 / sacrum   craniocaudal landmarks for ureteric segments
    hip_left / hip_right         bone partial volume near the distal ureter

Model weights live in the shared ~/.totalsegmentator cache (already present);
all outputs are written inside this project folder.

Usage:
    ./venv/bin/python run_anatomy.py
    ./venv/bin/python run_anatomy.py --force --fast
"""



import argparse
import glob
import os
import sys
import subprocess
import time

# this file lives in utils/, so the project root is one level up. All data
# directories hang off ROOT, never off the script's own folder.
HERE = os.path.dirname(os.path.abspath(__file__))
# package root -> project root: calculus/<sub>/x.py is two levels down
ROOT = os.path.dirname(os.path.dirname(HERE))

# Taken from paths.py, NOT hardcoded here: a second cohort points NIFTI and SEG
# inside its own folder via CALCULUS_NIFTI / CALCULUS_SEG. Defining them locally
# meant this script silently segmented the MAIN cohort (finding all 142 already
# done) while the new cohort's 181 volumes went untouched -- and the ureteric
# detector then failed on all 181 with "no corridor (kidney or bladder missing)".
from calculus.common.paths import NIFTI, SEG                      # noqa: E402
TS = os.path.join(ROOT, "venv", "bin", "TotalSegmentator")

#requested anatomical structures 
ROIS = ["kidney_left", "kidney_right", "kidney_cyst_left", "kidney_cyst_right",
        "urinary_bladder", "aorta", "inferior_vena_cava",
        "iliac_artery_left", "iliac_artery_right",
        "vertebrae_L1", "vertebrae_L5", "sacrum", "hip_left", "hip_right"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true") 
    ap.add_argument("--fast", action="store_true", help="3 mm model, ~4x faster")
    ap.add_argument("--device", default="gpu")
    args = ap.parse_args()

    os.makedirs(SEG, exist_ok=True)
    vols = sorted(glob.glob(os.path.join(NIFTI, "*.nii.gz"))) #finds all the extracted NIFTI CT volumes
    # paediatric studies are dropped here, before any GPU time is spent: the
    # adult TotalSegmentator model does not work on them and there is no
    # paediatric task to switch to
    from calculus.common.patient_gate import excluded_ids
    gated = excluded_ids()
    if gated:
        keep = [v for v in vols
                if os.path.basename(v).replace(".nii.gz", "") not in gated]
        print(f"patient gate: skipping {len(vols) - len(keep)} study(ies) "
              f"- see csv/patient_gate.csv")
        vols = keep


    print(f"{len(vols)} volumes to segment\n")
    #process each study 
    for i, v in enumerate(vols, 1):
        sid = os.path.basename(v).replace(".nii.gz", "")
        outdir = os.path.join(SEG, sid)
        done = os.path.join(outdir, "kidney_left.nii.gz")
        if os.path.exists(done) and not args.force:
            print(f"[{i}/{len(vols)}] {sid} exists, skip")
            continue
        cmd = [TS, "-i", v, "-o", outdir, "--device", args.device,
               "--roi_subset", *ROIS]
        if args.fast:
            cmd.append("--fast")
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[{i}/{len(vols)}] {sid} FAILED\n{r.stderr[-800:]}")
            continue
        n = len(glob.glob(os.path.join(outdir, "*.nii.gz")))
        print(f"[{i}/{len(vols)}] {sid} ok, {n} masks, "
              f"{time.time()-t0:.0f}s", flush=True)

    print(f"\nmasks in {SEG}")


if __name__ == "__main__":
    main()
