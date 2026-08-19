"""Exclude studies where the adult kidney model does not apply.

TotalSegmentator is trained on an overwhelmingly adult cohort. Measured on our
37 extracted studies:

    body cross-section at kidney level     kidney volume plausible?
    199 cm2  (7Y male)                     no   96.7 mL
    217 cm2  (7Y male)                     no   35.1 mL
    221 cm2  (18Y female, small stature)   no   30.9 mL
    298 - 841 cm2  (34 adults)             yes  all of them

3 of 3 below 230 cm2 failed; 0 of 34 above 298 cm2 failed. There is no
paediatric task in TotalSegmentator (checked all 53), so these studies cannot
be rescued by switching models -- they have to be excluded, and excluded
LOUDLY rather than reported as 30 mL kidneys with stones in them.

ONE RULE: keep age > 18Y, read from the DICOM PatientAge header. It needs
nothing but the zip, so it runs BEFORE TotalSegmentator and a paediatric study
never costs GPU time.

Age alone is enough on this cohort -- it catches all three known segmentation
failures (7Y, 7Y, and an 18Y, which fails "> 18"). Body cross-section is still
measured and recorded because it is three lines and explains WHY the model
fails, but it does not decide anything. A second criterion would only add ways
to be wrong.

Two edge cases, both handled by keeping the study and saying so:

    no age in header   2 of our 44 (one contains the placeholder "000Y").
                       Kept and flagged. 8259702 is one of them and is a
                       perfectly good 40-ish adult -- dropping it would cost
                       real data to guard against nothing.

Known cost: 8416497 is 17Y with a 519 cm2 adult abdomen and a normal 232 mL
kidney volume. It segments correctly and we drop it anyway. One study, in
exchange for a cohort that is unambiguously adult.

The point is not to diagnose paediatric patients, it is to refuse to report
numbers the segmentation cannot support.

Usage:
    ./venv/bin/python utils/patient_gate.py            # write the gate table
    ./venv/bin/python utils/patient_gate.py --apply    # + scrub existing results
"""


import argparse
import glob
import io
import os
import sys
import zipfile

import nibabel as nib
import numpy as np
import pandas as pd
import pydicom

HERE = os.path.dirname(os.path.abspath(__file__))
# package root -> project root: calculus/<sub>/x.py is two levels down
ROOT = os.path.dirname(os.path.dirname(HERE))
from calculus.common.paths import CSV, NIFTI, SEG, ZIPS, ensure   # noqa: E402

MIN_AGE_Y = 18.0          # keep age > this. adults only
MIN_BODY_CM2 = 250.0      # backstop. measured cliff sits between 221 and 298
BODY_HU = -300            # air/table below this


def age_years(zip_path):
    """PatientAge from any slice, in years. None if absent or implausible."""
    try:
        with zipfile.ZipFile(zip_path) as z:
            names = [n for n in z.namelist() if not n.endswith("/")]
            if not names:
                return None
            d = pydicom.dcmread(io.BytesIO(z.read(names[len(names) // 2])),
                                stop_before_pixels=True, force=True)
    except Exception:
        return None
    
    raw = str(getattr(d, "PatientAge", "") or "").strip()
    if len(raw) < 2:
        return None
    num, unit = raw[:-1], raw[-1].upper()
    try:
        n = float(num)
    except ValueError:
        return None
    yrs = {"Y": n, "M": n / 12.0, "W": n / 52.0, "D": n / 365.25}.get(unit)
    # "000Y" means the field was never filled in, not a newborn
    return yrs if yrs and yrs > 0 else None



#optionally measure body cross sectional area and kidney-mask volume 

def body_cm2(sid):
    """Body cross-section area, and kidney volume if a mask exists yet.

    Measured on the slice with the most kidney when segmentation has already
    run, otherwise at the middle of the volume. The two agree closely -- 221 vs
    220 cm2 on 8591756, 494 vs 491 on 8259702 -- because abdominal girth barely
    changes over the few centimetres between mid-volume and kidney level. That
    is what lets this gate run BEFORE TotalSegmentator.
    """
    #constructs a path
    vol_path = os.path.join(NIFTI, f"{sid}.nii.gz")
    if not os.path.exists(vol_path):
        return np.nan, np.nan          # not extracted yet: age gate only

    #load kidney masks optionally
    kid, vol_ml = None, np.nan
    #skip missing masks
    for side in ("left", "right"):
        p = os.path.join(SEG, sid, f"kidney_{side}.nii.gz")
        if not os.path.exists(p):
            continue

        #Mask value >0 → kidney
        m = np.asanyarray(nib.load(p).dataobj) > 0
        kid = m if kid is None else (kid | m)
    #opens the nifti [x-spacing, y-spacing, z-spacing]
    n = nib.load(vol_path)
    #reads voxel spacing
    sp = [float(v) for v in n.header.get_zooms()[:3]]

    #check for non-empty kidney mask exists, select the axial slice containing largest kidney mask cross section
    if kid is not None and kid.any():
        z = int(np.argmax(kid.sum(axis=(0, 1))))
        vol_ml = float(kid.sum() * np.prod(sp) / 1000.0)
    else:
        z = n.shape[2] // 2
    #computes body area 
    sl = np.asarray(n.dataobj[:, :, z]).astype(np.float32)   # lazy: one slice
    area = float((sl > BODY_HU).sum() * sp[0] * sp[1] / 100.0)
    return area, vol_ml


def build():
    ensure()
    rows = []
    # every DOWNLOADED study, not just the extracted ones -- the whole point is
    # to gate before the expensive steps
    ids = sorted(os.path.basename(p)[:-4]
                 for p in glob.glob(os.path.join(ZIPS, "*.zip")))
    for i, sid in enumerate(ids, 1):
        yrs = age_years(os.path.join(ZIPS, f"{sid}.zip"))
        area, vol = body_cm2(sid)
        # ONE criterion: age from the header. Body size is recorded for
        # context but does not decide -- on this cohort the age rule already
        # catches all three known segmentation failures (7Y, 7Y, 18Y), so a
        # second criterion would only add ways to be wrong.
        why = []
        if yrs is not None and yrs <= MIN_AGE_Y:
            why.append(f"age {yrs:.0f}Y not > {MIN_AGE_Y:.0f}Y")
        rows.append({
            "study_id": sid,
            "age_years": None if yrs is None else round(yrs, 1),
            "age_source": "header" if yrs is not None else "missing",
            "body_cm2": None if np.isnan(area) else round(area),
            "kidney_ml": None if np.isnan(vol) else round(vol, 1),
            "excluded": bool(why),
            "reason": ("; ".join(why) + " - adult kidney model not valid"
                       if why else ""),
            "age_unknown": yrs is None,
        })
        note = ("EXCLUDED: " + "; ".join(why) if why else
                "ok (no age in header, passed on body size)" if yrs is None
                else "ok")
        print(f"[{i}/{len(ids)}] {sid}: age="
              f"{'?' if yrs is None else f'{yrs:.0f}Y':>4} "
              f"body={'?' if np.isnan(area) else f'{area:.0f}':>4} cm2  {note}",
              flush=True)
    d = pd.DataFrame(rows)
    out = os.path.join(CSV, "patient_gate.csv")
    d.to_csv(out, index=False)
    print(f"\nwrote {out}")
    print(f"excluded {int(d.excluded.sum())} of {len(d)} studies")
    if d.excluded.any():
        print(d[d.excluded][["study_id", "age_years", "body_cm2",
                             "kidney_ml", "reason"]].to_string(index=False))
    if d.age_unknown.any():
        print(f"\nno age in header, kept on body size alone - worth a look at "
              f"the mask overlay:")
        print(d[d.age_unknown][["study_id", "body_cm2", "kidney_ml",
                                "excluded"]].to_string(index=False))
    return d


def excluded_ids():
    """Study ids the gate rejects. Empty set if the gate has not been built."""
    p = os.path.join(CSV, "patient_gate.csv")
    if not os.path.exists(p):
        return set()
    d = pd.read_csv(p)
    return set(d[d.excluded.astype(bool)].study_id.astype(str))


def apply_to_results(d):
    """Scrub stones already detected in excluded studies.

    detect_stones may have run before the gate existed. Rather than re-running
    2.5 hours of detection, drop those studies' stone rows and mark the summary
    so nothing downstream treats the numbers as real.
    """
    bad = set(d[d.excluded].study_id.astype(str))
    if not bad:
        print("nothing to scrub")
        return
    for name, is_summary in (("baseline_stones.csv", False),
                             ("candidates.csv", False),
                             ("baseline_summary.csv", True)):
        p = os.path.join(CSV, name)
        if not os.path.exists(p):
            continue
        t = pd.read_csv(p)
        t["study_id"] = t.study_id.astype(str)
        hit = t.study_id.isin(bad)
        if is_summary:
            # keep the row, void the numbers, say why
            t.loc[hit, "error"] = t.loc[hit, "study_id"].map(
                d.set_index(d.study_id.astype(str)).reason.to_dict()
            ).radd("paediatric/small-body, adult model not valid - ")
            for c in ("n_stones", "largest_mm", "total_volume_mm3"):
                if c in t.columns:
                    t.loc[hit, c] = np.nan
            print(f"  {name}: voided {int(hit.sum())} study rows")
        else:
            print(f"  {name}: dropped {int(hit.sum())} rows")
            t = t[~hit]
        t.to_csv(p, index=False)



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="also scrub excluded studies out of existing results")
    a = ap.parse_args()
    d = build()
    if a.apply:
        print("\nscrubbing existing results:")
        apply_to_results(d)
