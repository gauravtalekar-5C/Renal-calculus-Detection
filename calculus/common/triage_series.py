"""Step 1: can this data support stone measurement at all?

Reads DICOM headers (and a handful of sampled slices) straight out of
dicoms/zips/*.zip and reports, per series: slice spacing, in-plane spacing,
z-coverage, kernel, kVp, contrast phase, axial-or-not.

Each STUDY is then classified by the best NON-CONTRAST AXIAL series it holds:

    measurable   spacing <= 1.5 mm and coverage >= 250 mm
                 -> full product: detection + volume + HU + location
    detect_only  spacing <= 3.0 mm
                 -> stones still found, but sub-4 mm sensitivity drops sharply
                    and volume error exceeds 50%; report size bands, not mm
    unusable     no non-contrast axial series, or thicker than 3 mm

Why non-contrast: excreted iodine in the collecting system runs 300-3000 HU and
is indistinguishable from a calculus, so contrast series cannot carry detection
or HU claims (they stay useful as hard negatives and for anatomy pretraining).

PHASE DETECTION -- three approaches were tried on studies with known labels:
  1. ContrastBolusAgent header. FAILS: populated on every series of a contrast
     study including the pre-contrast phase.
  2. Absolute pixel threshold on soft-tissue percentiles. FAILS: a global p99 is
     saturated by cortical bone edges (known pre-contrast 357 HU vs post 367),
     and excluding "bone" as >300 HU deletes the opacified aorta itself.
  3. Aorta HU ranked WITHIN each study. WORKS: the aorta is located relative to
     the vertebral body, and the lowest-attenuation series in a multiphase study
     is its pre-contrast phase. Verified on 3 studies with known phase labels.
Relative ranking is used because absolute HU varies with kVp, habitus and
timing. The production pipeline should replace this with a mean over
TotalSegmentator's aorta mask, which is exact; this is the cheap triage stand-in.

Usage:
    ./venv/bin/python triage_series.py
    ./venv/bin/python triage_series.py --zips dicoms/zips --out-prefix triage
"""


import argparse
import os
import sys
import re
import zipfile
from collections import defaultdict
import numpy as np
import pandas as pd
import pydicom
from scipy import ndimage

# this file lives in utils/, so the project root is one level up. All data
# directories hang off ROOT, never off the script's own folder.
HERE = os.path.dirname(os.path.abspath(__file__))  #_file_ is the path of traige_series.py, so HERE becomes /root/Gaurav/kindey_calculus_measurement/utils 
ROOT = os.path.dirname(os.path.dirname(HERE))       # moves one directory upward /root/Gaurav/kindey_calculus_measurement
if HERE not in sys.path:            # so sibling imports work from anywhere
    sys.path.insert(0, HERE)

# product tiers -- tune here
THIN_MM = 1.5
THICK_MM = 3.0 #not considered good for measurement , between 1.5mm and 3mm we can use for measurement
MIN_COVERAGE_MM = 250 #for measureable category at least 250mm of z-axis coverage is needed
MIN_SLICES = 40 #skip slices fewer than 40 slices

PHASE_SAMPLE_SLICES = 7 #upto 7 slices are sampled when estimating aortic HU 


CONTRAST_DELTA_HU = 45 #in multiphase study a series is considired contrast enhanced when its measured aortic HU is more than 45HU above the least-enhanced series 
CONTRAST_ABS_HU = 200 #if only one phase can be measured, an aoratic value above 200HU is treated as contrast 

#this detects uppercase and lower case treated identical , re.I does that
CONTRAST_RE = re.compile(
    r"contrast|post[\s_-]?c|venous|arterial|portal|nephro|excret|delay|"
   
    r"\bcect\b|\bven\b|\bart\b|\bpv\b|\bc\+|\bhrct[\s_-]?c\b|"
    
    r"after[\s_-]?\d+[\s_-]?-?\s?min", re.I)

#plain word detection 
PLAIN_RE = re.compile(
    r"plain|non[\s_-]?contrast|pre[\s_-]?contrast|nc\b|kub|"
    r"\bnect\b|\bnon[\s_-]?enhanced\b|\bunenhanced\b", re.I)

#code searches plain-re before contrast-re because the pre-contrast also contains the word contrast 

#identifying the junk descriptions, it recognises non-diagnostic or rendered series
# "dose" needs the lookbehind: Philips names its iterative reconstruction
# "iDose", so a bare `dose` threw away every "PLAIN THIN, iDose (4)" -- the
# thin non-contrast series we most want -- as if it were a dose-report
# screenshot. Seven studies across the cohorts lost their ONLY usable series
# that way and fell through to verdict=skip. "Dose Report" still matches.
JUNK_RE = re.compile(
    r"scout|topogram|localiz|localis|(?<![a-z])dose(?![a-z])|report|"
    r"screen\s?save|summary|"
    r"\bvr\b|\bmip\b|bone\s?3d|smart\s?prep|monitor", re.I)

#this function accepts arg as ImageOrientationPatient (iop)
def is_axial(iop):
    """True if the slice normal is within ~25 deg of the patient z axis."""
    if iop is None or len(iop) != 6:  #A valid ImageOrientationPatient should contain six values. Missing or malformed orientation is considered non-axial
        return False
    row, col = np.array(iop[:3], float), np.array(iop[3:], float) #splits the first 3 : row and last 3 as column
    return abs(np.cross(row, col)[2]) > 0.9 #computes the dirn perpendicular to the image plane: slice normal , close to 1 or -1 we name as it as axial, else close to 0 we keep it as coronal or sagittal, abs() returns either head to foot and foot to head slice order,  both are accepted 

#helper that converts DICOM value into a python floating point number, if missing we return NAN
def num(v, default=np.nan):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default

#checks if the series is contrast-enhanced or not. the input is z: opened ZIP archieve , names : selected DICOM filenames inside the ZIP 
def aorta_p90(z, names):
    """Median over sampled slices of the 90th-percentile HU in an ROI placed
    anterior to the vertebral body, where the aorta lies."""
    vals = []
    for name in names:
        try:
            ds = pydicom.dcmread(z.open(name), force=True) #open and reads the DICOM
            hu = (ds.pixel_array.astype(np.float32) #implements (HU = PixelValue times RescaleSlope + RescaleIntercept)
                  * float(getattr(ds, "RescaleSlope", 1) or 1)
                  + float(getattr(ds, "RescaleIntercept", 0) or 0))
            ps = float(ds.PixelSpacing[0]) #reads in-plane pixel spacing in millimeters
        except Exception:
            continue

        #locating the vertebral body
        #the aorta is the bodys largest artery, it carried blood from the heart and passes through the abdomen, just in from of the spine
        #we use it to identify whether contrast dye was given or not - without contrast dye the blood in the aorta has low HU , after aorta becomes very bright and its HU increases significantly
        #We need non-contrast CT because contrast dye can appear as bright as a kidney stone and cause false detections

        #Contrast dye is usually an iodine-based liquid injected into a vein before or during a CT scan.
        #Doctors give it because it makes blood vessels and organs appear brighter, helping them see:
    
        H, W = hu.shape #get image width and height
        dense = hu > 250 #create a boolean mask containing dense structures, mainly bone and contrast filled vessels
        dense[:H // 2, :] = False          # Removes the first half of the image and retains the half where the code expects the posterior anatomy and vertebral body.
        lab, k = ndimage.label(dense) #label seperated connected dense regions
        if k == 0: #if no dense component - skip that slice
            continue
        vb = int(np.argmax(ndimage.sum(dense, lab, range(1, k + 1)))) + 1 #computes the size of every dense component, returns the pos of largest one
        cy, cx = ndimage.center_of_mass(lab == vb)
        d1, d2, half = int(15 / ps), int(50 / ps), int(22 / ps)
        box = hu[max(0, int(cy) - d2):max(0, int(cy) - d1),
                 max(0, int(cx) - half):min(W, int(cx) + half)]
        v = box[(box > -20) & (box < 600)]
        if v.size >= 50:
            vals.append(float(np.percentile(v, 90)))
    return float(np.median(vals)) if vals else np.nan

#path to one study ZIP 
def scan_zip(path, detect_phase=True):
    """Return one row dict per series in this study zip."""

    sid = os.path.splitext(os.path.basename(path))[0] #identify the study like dicoms/zips/8261985.zip
    series = defaultdict(lambda: {"n": 0, "hdr": None, "types": set(), "zn": []}) #for every new SeriesInstanceUID, create n : number of DICOM files, hdr : representative DICOM header, types : combined ImageType values, zn:pairs of z-position and ZIP filename 
    
    #opening the ZIP 
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if name.endswith("/"):
                continue
            try:
                ds = pydicom.dcmread(z.open(name), stop_before_pixels=True,
                                     force=True)
            except Exception:
                continue

            #series identifier
            uid = getattr(ds, "SeriesInstanceUID", None)
            if uid is None:
                continue
            
            #create a group for this UID 
            s = series[uid]
            s["n"] += 1 #adds one to the file-count

            #reads the 3-D physical location of the slice
            ipp = getattr(ds, "ImagePositionPatient", None)

            #stores as (Z_position, ZIP filename) , The third value, ipp[2], is used because the series has not yet been fully classified but is expected to be approximately axia
            if ipp is not None and len(ipp) == 3:
                s["zn"].append((num(ipp[2]), name))

            #collecting image types,  reads values like [ ORIGINAL,PRIMARY,AXIAL,DERIVED,SECONDARY ]
            it = getattr(ds, "ImageType", None)
            #converts every value to uppercase and adds to the series level set, using set prevents duplicates
            if it:
                s["types"].update(str(x).upper() for x in it)
            #Stores the first DICOM header encountered for the series.The code assumes properties such as orientation, kernel and pixel spacing are consistent across the series.
            if s["hdr"] is None:
                s["hdr"] = ds

        #creates a list for final per-series dictionaries, processes every grouped series 
        rows = []
        for uid, s in series.items():
            ds = s["hdr"] #gets the representative header 
            if ds is None: #skip if no valid header 
                continue
            desc = str(getattr(ds, "SeriesDescription", "") or "") #reads the series descitpion and guarantees a string 
            zs = sorted(v for v, _ in s["zn"] if v == v) #extracts and sorts valid z pos, v=v is for nan removal
            # spacing from actual positions: the SliceThickness tag lies on
            # overlapping and gapped reconstructions

            #req more than 3 valid positions
            if len(zs) > 3: 
                d = np.diff(zs)
                d = d[d > 0.01] #removes duplicates/nearly duplicate positions
                spacing = float(np.median(d)) if len(d) else np.nan #uses median pos difference as actual slice spacing
                coverage = zs[-1] - zs[0]
            else:
                spacing, coverage = np.nan, np.nan

            #reads in plane pixel spacing
            px = getattr(ds, "PixelSpacing", None)
            #if seres contain this , the intersection becomes non empty and derived becomes true 
            derived = bool({"DERIVED", "SECONDARY"} & s["types"])

            #series is junk if is in junk registry or i is derived
            junk = bool(JUNK_RE.search(desc)) or derived
            #run the orintation function on the rep header 
            axial = is_axial(getattr(ds, "ImageOrientationPatient", None))

            #choosing slices for phase detection
            aorta = np.nan
       
            #pixel sampling occurs when phase det is enabled, seris is axial , not junk and has >40 slices
            if detect_phase and axial and not junk and s["n"] >= MIN_SLICES:
                #sorts the stored (z_pos, filename) pairs
                zn = sorted(s["zn"])

                #takes the mid 50% of the series , For 400 slices, this approximately selects slices 100–299.If that slice range is empty, it falls back to the complete list
                mid = zn[len(zn) // 4:3 * len(zn) // 4] or zn
                step = max(1, len(mid) // PHASE_SAMPLE_SLICES)
                #pass those to aortic HU func
                aorta = aorta_p90(z, [n for _, n in mid[::step]][:PHASE_SAMPLE_SLICES])
            
            rows.append({
                "study_id": sid,
                "series_uid": uid,
                "series_desc": desc,
                "n_slices": s["n"],
                "slice_spacing_mm": round(spacing, 3) if spacing == spacing else np.nan,
                "slice_thickness_tag": num(getattr(ds, "SliceThickness", None)),
                "inplane_mm": round(num(px[0]), 3) if px else np.nan,
                "coverage_mm": round(coverage, 1) if coverage == coverage else np.nan,
                "rows_cols": f"{getattr(ds,'Rows','?')}x{getattr(ds,'Columns','?')}",
                "kernel": str(getattr(ds, "ConvolutionKernel", "") or ""),
                "kvp": num(getattr(ds, "KVP", None)),
                "manufacturer": str(getattr(ds, "Manufacturer", "") or "")[:24],
                "model": str(getattr(ds, "ManufacturerModelName", "") or "")[:24],
                "body_part": str(getattr(ds, "BodyPartExamined", "") or ""),
                "contrast_agent": str(getattr(ds, "ContrastBolusAgent", "") or "")[:20],
                "aorta_p90_hu": round(aorta, 0) if aorta == aorta else np.nan,
                "is_axial": axial,
                "is_derived": derived,
                "is_junk": junk,
            })
    return rows


def assign_phase(rows):
    """Mark each series contrast/plain by ranking aorta HU within the study."""
    meas = [r for r in rows if r["aorta_p90_hu"] == r["aorta_p90_hu"]]
    base = min((r["aorta_p90_hu"] for r in meas), default=np.nan)
    for r in rows:
        a = r["aorta_p90_hu"]
        desc = r["series_desc"]
        # An explicit description outranks the pixel estimate: a series named
        # "POST CON THIN" measured only 18 HU above the pre-contrast phase in
        # testing (poorly timed bolus), which the relative rule alone would have
        # passed as plain -- a false pass that would corrupt every HU value.
        # Check plain first, since "pre contrast" also contains "contrast".
        if PLAIN_RE.search(desc):
            r["is_contrast"], r["phase_src"] = False, "name_plain"
        elif CONTRAST_RE.search(desc):
            r["is_contrast"], r["phase_src"] = True, "name_contrast"
        elif a != a:
            r["is_contrast"] = bool(r["contrast_agent"])
            r["phase_src"] = "header_agent"
        elif len(meas) > 1:
            r["is_contrast"] = (a - base) > CONTRAST_DELTA_HU
            r["phase_src"] = "aorta_rel"
        else:
            r["is_contrast"] = a > CONTRAST_ABS_HU
            r["phase_src"] = "aorta_abs"
    return rows


#A series description containing one of these terms is considered unsuitable for complete renal analysis.This checks SeriesDescription, not BodyPartExamined
NON_ABDOMINAL_RE = re.compile(
    r"thorax|chest|\bhrct\b|\blung\b|brain|\bhead\b|\bneck\b|"
    # pelvis-only acquisitions: 8370673 is seven Pelvis series and no kidneys
    r"pelvis", re.I)



# recognises bone, osseous, sharp
BONE_KERNEL_RE = re.compile(r"\bbone\b|\bosseous\b|\bsharp\b|\bB[7-9]\d\b|"
                            r"\bYD\b|\bYE\b", re.I)


def non_abdominal(r):
    """Is this series named as a region that does not cover the kidneys?"""
    return bool(NON_ABDOMINAL_RE.search(str(r.get("series_desc") or "")))


# WHAT THIS FUNCTION DOES: says whether a series is named as a chest, head, neck
# or pelvis-only acquisition, so a thin chest scan cannot be preferred over a
# thicker abdominal one. Name only -- the DICOM body-part tag proved too
# unreliable to demote on.

#checks series_desc, and kernel
def bone_kernel(r):
    """Is this a sharp bone-kernel reconstruction?"""
    return bool(BONE_KERNEL_RE.search(str(r.get("series_desc") or ""))
                or BONE_KERNEL_RE.search(str(r.get("kernel") or "")))
# WHAT THIS FUNCTION DOES: identifies reconstructions made with a sharp bone
# kernel, which measure calculi badly, so that a soft-kernel series in the same
# study is preferred even when it has thicker slices.

#filter for series 
def series_verdict(r):
    #skip if junk , not axial and slice count<40
    if r["is_junk"] or not r["is_axial"] or r["n_slices"] < MIN_SLICES:
        return "skip"
    #contrast is rejected even before evaluating image thickness
    if r["is_contrast"]:
        return "contrast"
    #creates short var for spacing 
    sp = r["slice_spacing_mm"]
    if sp != sp:
        return "skip"
    # Region is checked BEFORE thickness. A 1 mm chest scan is worse than a
    # 3 mm abdominal one for this task, however good it looks: no amount of
    # resolution recovers a kidney that is outside the field of view.
    if non_abdominal(r):
        return "non_abdominal"
    # A bone kernel is usable but never preferable, so it is demoted to a single
    # bucket rather than being graded on thickness -- otherwise a thin bone
    # recon climbs back above a slightly thicker soft one, which is the bug.

    if bone_kernel(r):
        return "bone_kernel"
    if sp <= THIN_MM and r["coverage_mm"] >= MIN_COVERAGE_MM:
        return "measurable"
    if sp <= THIN_MM:
        return "thin_short_coverage"
    if sp <= THICK_MM:
        return "detect_only"
    return "too_thick"

#ranking categories 
RANK = {"measurable": 0, "thin_short_coverage": 1, "detect_only": 2,
        "bone_kernel": 3, "too_thick": 4, "non_abdominal": 5,
        "contrast": 6, "skip": 7}


def main():
    ap = argparse.ArgumentParser()
    from calculus.common.paths import CSV, ZIPS, ensure
    ensure()
    ap.add_argument("--zips", default=ZIPS)
    ap.add_argument("--out-prefix", default=os.path.join(CSV, "triage"))
    ap.add_argument("--worklist", default=os.path.join(CSV, "worklist_all.csv"))
    ap.add_argument("--no-phase", action="store_true",
                    help="headers only, skip pixel reads (fast but unreliable)")
    args = ap.parse_args()

    zips = sorted(f for f in os.listdir(args.zips) if f.endswith(".zip"))
    print(f"scanning {len(zips)} study zips from {args.zips}\n")

    all_rows = []
    for i, f in enumerate(zips, 1):
        rows = assign_phase(scan_zip(os.path.join(args.zips, f),
                                     detect_phase=not args.no_phase))
        all_rows.extend(rows)
        good = sum(1 for r in rows if not r["is_contrast"] and r["is_axial"])
        print(f"[{i}/{len(zips)}] {f}: {len(rows)} series, "
              f"{good} non-contrast axial", flush=True)

    ser = pd.DataFrame(all_rows)
    ser["verdict"] = ser.apply(series_verdict, axis=1)
    ser["rank"] = ser["verdict"].map(RANK)
    ser = ser.sort_values(["study_id", "rank", "slice_spacing_mm"])
    ser.drop(columns="rank").to_csv(f"{args.out_prefix}_series.csv", index=False)

    # Tie-break order, applied with a STABLE sort:
    #   1. rank            -- measurable beats thin_short_coverage beats ...
    #   2. slice_spacing   -- thinnest first. MUST outrank slice count: a 5 mm
    #                         series with 900 slices is useless to us, a 0.6 mm
    #                         series with 300 is fine.
    #   3. n_slices desc   -- with spacing tied, more coverage is strictly
    #                         better. This alone would have picked the right
    #                         series for 8622562 (881 slices vs 809 at the same
    #                         0.625 mm) even before the CONTRAST_RE fix.
    # kind="stable" matters: the default quicksort is NOT stable, so re-sorting
    # by rank alone was free to scramble the spacing order established above --
    # the tie-break was not reliably applied at all.
    best = (ser.sort_values(["rank", "slice_spacing_mm", "n_slices"],
                            ascending=[True, True, False], kind="stable")
               .groupby("study_id", as_index=False).first())
    # series_uid is recorded so extract_series.py can use the series this
    # ranking actually chose, instead of re-deriving it from the verdict and the
    # thinnest slice. The re-derivation ignores the n_slices tie-break applied
    # above, so the two could silently disagree about which series to extract.
    study = best[["study_id", "verdict", "series_uid", "series_desc", "n_slices",
                  "slice_spacing_mm", "inplane_mm", "coverage_mm", "kernel",
                  "kvp", "aorta_p90_hu", "manufacturer", "model"]].rename(
        columns={"verdict": "study_verdict", "series_desc": "best_series",
                 "series_uid": "best_series_uid"})
    study["n_series_total"] = study.study_id.map(ser.groupby("study_id").size())
    study["has_contrast_phase"] = study.study_id.map(
        ser[ser.is_contrast].groupby("study_id").size()).fillna(0).astype(int) > 0

    if os.path.exists(args.worklist):
        wl = pd.read_csv(args.worklist)
        wl["study_id"] = wl["study_id"].astype(str)
        study["study_id"] = study["study_id"].astype(str)
        study = study.merge(
            wl[["study_id", "tier", "variant", "family", "calculus_flag",
                "calculus_type"]], on="study_id", how="left")
    study.to_csv(f"{args.out_prefix}_study.csv", index=False)

    print(f"\n{'='*62}\nSERIES verdicts ({len(ser)} series)")
    print(ser.verdict.value_counts().to_string())
    print(f"\nSTUDY verdicts ({len(study)} studies)")
    print(study.study_verdict.value_counts().to_string())

    ok = study.study_verdict.eq("measurable").sum()
    print(f"\n>>> MEASURABLE: {ok}/{len(study)} ({ok/len(study)*100:.0f}%)")
    usable = study.study_verdict.isin(
        ["measurable", "detect_only", "thin_short_coverage"]).sum()
    print(f">>> any stone work possible: {usable}/{len(study)}")

    thin = ser[ser.verdict.isin(["measurable", "thin_short_coverage"])]
    if len(thin):
        print(f"\nthin non-contrast series: median spacing "
              f"{thin.slice_spacing_mm.median():.2f} mm, in-plane "
              f"{thin.inplane_mm.median():.3f} mm, coverage "
              f"{thin.coverage_mm.median():.0f} mm")
    if "variant" in study:
        print("\nverdict x variant:")
        print(pd.crosstab(study.study_verdict, study.variant).to_string())
    print(f"\nwrote {args.out_prefix}_series.csv and {args.out_prefix}_study.csv")


if __name__ == "__main__":
    main()
