"""PREPARE one study: DICOM -> NIfTI -> organ masks. No detection.

WHY THIS IS A SEPARATE ENTRY POINT
----------------------------------
Detection is CPU-bound and parallelises across studies; segmentation is GPU-bound
and must NOT. This box shares its GPU with the CT-abdomen API in production --
22 of 41 GB were already committed to it when this was written -- so running
several TotalSegmentator instances to speed our own batch up risks an OOM in
someone else's service.

Splitting prepare from detect lets the batch runner segment strictly one study at
a time while running the detectors N-wide. infer_study is unchanged and still
does the whole job per study: it simply finds the NIfTI and masks already on disk
and skips straight to DETECT, exactly as it does on any re-run.

This is not a return to stage-by-stage cohort processing. Each study is still
prepared individually, and the full per-study pipeline still runs per study. Only
the mask cache is warmed in a controlled order.

Usage:
    python -m calculus.pipeline.prepare_study path/to/8677121.zip --id 8677121
"""
import argparse
import os
import sys
import time

from calculus.common.paths import NIFTI, SEG            # noqa: F401
from calculus.pipeline.dicom_to_3d import (_allow_directories, pick_series,
                                           resolve, segment, to_nifti)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="zip, directory, or study id")
    ap.add_argument("--id", default=None)
    ap.add_argument("--device", default="gpu")
    a = ap.parse_args()

    _allow_directories()
    sid, path = resolve(a.target)
    if a.id:
        sid = a.id
    t0 = time.time()

    best, _rows = pick_series(path)
    if best is None:
        raise SystemExit("no DICOM series found in this archive")
    print(f"  {sid}  series {best['series_desc']!r} [{best['verdict']}] "
          f"{best['n_slices']} sl, {best['slice_spacing_mm']} mm", flush=True)
    if best["verdict"] not in ("measurable", "thin_short_coverage"):
        print(f"  {sid}  WARNING verdict '{best['verdict']}': shape is worth "
              "looking at, sizes less reliable", flush=True)
    nii = to_nifti(sid, path, best["series_uid"], force=False)
    segment(sid, nii, device=a.device, force=False)
    print(f"  {sid}  prepared in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
