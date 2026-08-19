"""Extract the chosen non-contrast series from each study zip into NIfTI.

Reads csv/triage_study.csv (study -> best series) and csv/triage_series.csv
(series -> uid), pulls just that series out of dicoms/zips/<study>.zip, stacks
it into a volume in Hounsfield units, and writes nifti/<study_id>.nii.gz.

Built directly on pydicom + nibabel rather than dcm2niix so there is no external
binary to install and we control exactly which series is used and how HU are
computed (slope/intercept applied per slice).

Only studies whose triage verdict is 'measurable' are extracted by default.

Usage:
    ./venv/bin/python extract_series.py
    ./venv/bin/python extract_series.py --include-thick   # also detect_only
"""



import argparse
import os
import sys
import zipfile
from collections import defaultdict
import nibabel as nib
import numpy as np
import pandas as pd
import pydicom

# this file lives in utils/, so the project root is one level up. All data
HERE = os.path.dirname(os.path.abspath(__file__))
# package root -> project root: calculus/<sub>/x.py is two levels down
ROOT = os.path.dirname(os.path.dirname(HERE))
from calculus.common.paths import CSV, ZIPS, NIFTI    # noqa: E402  results dir is per-run

#this function outputs 3-D HU array, spatial coordinate transformaton and technical information

#Convert the best CT series selected by triage_series.py from separate DICOM slices into one correctly oriented 3-D NIfTI volume containing true Hounsfield-unit values.
#basic work is convert DICOM to 3D HU volume, takes one study zip and series_uid 
def load_series(zip_path, series_uid):
    """Stack one series into (volume_HU, affine, meta)."""
    #creates a list for all the selected series  
    slices = []
    with zipfile.ZipFile(zip_path) as z:
        #checks every item in the ZIP 
        for name in z.namelist():
            if name.endswith("/"):
                continue
            try:
                #reads only the dicom header during the first pass 
                ds = pydicom.dcmread(z.open(name), stop_before_pixels=True,
                                     force=True)
            except Exception:
                continue #unreadable files are skipped

            #only slices whose SeriesInstanceUID exactly matches the series chosen by triage_series.py are retained. 
            if getattr(ds, "SeriesInstanceUID", None) != series_uid: #skip unreadable files 
                continue

            #reads the physical slice position
            ipp = getattr(ds, "ImagePositionPatient", None)
            #slice without valid 3-d position are ignored 
            if ipp is None or len(ipp) != 3:
                continue
            slices.append((float(ipp[2]), name)) #stores (z-position, filename)

        if len(slices) < 10:
            return None, None, None

        #sort the slices by z-position, from smallest to largest patient z-coordinat, inferior to superior 
        slices.sort()

        #read pixels and convert to HU 
        #vol is the list of converted slices, red is representative DICOM used for orinatation and spacing
        vol, ref = [], None

        #reads the selected slices in sorted Z-order 
        for _, name in slices:
            #this time we read the complete dicom including the pixel data
            ds = pydicom.dcmread(z.open(name), force=True)
            #decodes the stored pixels and converts them to floating point values
            arr = ds.pixel_array.astype(np.float32)

            #here we do \(HU = PixelValue \times RescaleSlope + RescaleIntercept\), we do this conversion so that slope and intercept can vary between slices
            hu = (arr * float(getattr(ds, "RescaleSlope", 1) or 1)
                  + float(getattr(ds, "RescaleIntercept", 0) or 0))

            #we do it for every slice 
            vol.append(hu)
            if ref is None:
                ref = ds


    #constructs the inital volume
    vol = np.stack(vol)                       # (z, rows, cols)
    zs = [z for z, _ in slices]
    dz = float(np.median(np.diff(zs))) #computes the median differnce between consecutive slices

    #becomes through-plane spacing
    dy, dx = float(ref.PixelSpacing[0]), float(ref.PixelSpacing[1])
    
    # DICOM ImageOrientationPatient = [row_dir(3), col_dir(3)] where row_dir is
    # the direction of INCREASING COLUMN INDEX and col_dir the direction of
    # increasing row index. PixelSpacing = [between-rows, between-columns].
    #
    # vol is (slice, row, col); transposing to (col, row, slice) makes
    #   axis0 = column index -> direction row_dir, spacing PixelSpacing[1]
    #   axis1 = row index    -> direction col_dir, spacing PixelSpacing[0]
    # Getting these two the wrong way round transposes the image and silently
    # swaps the patient's left and right, which mislabels every stone.


    #dicoms init has(slice, row, coln), nifti is easier to rep as spatial x/y/x like axes so we rearrange like (coln, row, slice )
    data = np.transpose(vol, (2, 1, 0))
    iop = [float(v) for v in ref.ImageOrientationPatient]
    row_dir, col_dir = np.array(iop[:3]), np.array(iop[3:])
    normal = np.cross(row_dir, col_dir)

    origin = np.array([float(v) for v in ref.ImagePositionPatient])
    slice_dir = normal if dz > 0 else -normal
    if slice_dir[2] < 0:                       # keep axis2 increasing head-ward
        data = data[:, :, ::-1]
        origin = origin + slice_dir * abs(dz) * (len(zs) - 1)
        slice_dir = -slice_dir


    #An affine is a 4 × 4 matrix that maps array indices into real patient coordinates
    affine = np.eye(4)
    affine[:3, 0] = row_dir * dx               # dx = PixelSpacing[1]
    affine[:3, 1] = col_dir * dy               # dy = PixelSpacing[0]
    affine[:3, 2] = slice_dir * abs(dz)
    affine[:3, 3] = origin
    # DICOM is LPS, NIfTI is RAS: flip the first two world axes
    affine = np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine

    meta = {"n_slices": len(zs), "dz": abs(dz), "dy": dy, "dx": dx,
            "kernel": str(getattr(ref, "ConvolutionKernel", "") or ""),
            "kvp": float(getattr(ref, "KVP", 0) or 0),
            "manufacturer": str(getattr(ref, "Manufacturer", "") or ""),
            "shape": data.shape}
    return data, affine, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-thick", action="store_true",
                    help="also extract detect_only (<=3 mm) studies")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--studies", nargs="*", default=None,
                    help="only these study ids. Use with --force to re-extract "
                         "a handful without rewriting all 137 NIfTIs, which is "
                         "many GB of pointless disk churn on a shared machine.")
    args = ap.parse_args()

    os.makedirs(NIFTI, exist_ok=True)
    study = pd.read_csv(os.path.join(CSV, "triage_study.csv"))
    ser = pd.read_csv(os.path.join(CSV, "triage_series.csv"))
    study["study_id"] = study["study_id"].astype(str)
    ser["study_id"] = ser["study_id"].astype(str)

    keep = ["measurable"] + (["detect_only", "thin_short_coverage"]
                             if args.include_thick else [])
    todo = study[study.study_verdict.isin(keep)]
    if args.studies:
        want = {str(s) for s in args.studies}
        missing = want - set(todo.study_id)
        if missing:
            print(f"NOT extractable (verdict not in {keep}): {sorted(missing)}")
        todo = todo[todo.study_id.isin(want)]
    # paediatric studies never make it to a NIfTI: the adult kidney model
    # cannot segment them, so there is nothing downstream to do with them
    from calculus.common.patient_gate import excluded_ids
    gated = excluded_ids() & set(todo.study_id)
    if gated:
        print(f"patient gate: skipping {len(gated)} study(ies) "
              f"{sorted(gated)} - see csv/patient_gate.csv")
        todo = todo[~todo.study_id.isin(gated)]
    print(f"{len(todo)} studies to extract (verdicts: {keep})\n")

    rows = []
    for i, r in enumerate(todo.itertuples(), 1):
        sid = r.study_id
        dest = os.path.join(NIFTI, f"{sid}.nii.gz")
        if os.path.exists(dest) and not args.force:
            print(f"[{i}/{len(todo)}] {sid} exists, skip")
            continue
        # Prefer the uid triage actually chose. The fallback below re-derives it
        # from verdict + thinnest slice, which ignores triage's n_slices
        # tie-break and so can pick a DIFFERENT series than the one reported in
        # triage_study.csv -- meaning the CSV would describe one series while the
        # NIfTI came from another.
        uid = getattr(r, "best_series_uid", None)
        if not isinstance(uid, str) or not uid:
            cand = ser[(ser.study_id == sid) & (ser.verdict == r.study_verdict)]
            if cand.empty:
                print(f"[{i}/{len(todo)}] {sid} NO MATCHING SERIES")
                continue
            uid = cand.sort_values("slice_spacing_mm").iloc[0].series_uid

        data, affine, meta = load_series(os.path.join(ZIPS, f"{sid}.zip"), uid)
        if data is None:
            print(f"[{i}/{len(todo)}] {sid} FAILED to stack")
            continue
        nib.save(nib.Nifti1Image(data.astype(np.int16), affine), dest)
        print(f"[{i}/{len(todo)}] {sid} {meta['shape']} "
              f"{meta['dx']:.2f}x{meta['dy']:.2f}x{meta['dz']:.2f} mm "
              f"[{meta['kernel']}]", flush=True)
        rows.append({"study_id": sid, "series_uid": uid, "nifti": dest, **meta})

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(CSV, "extracted.csv"), index=False)
    print(f"\nNIfTI in {NIFTI}")


if __name__ == "__main__":
    main()
