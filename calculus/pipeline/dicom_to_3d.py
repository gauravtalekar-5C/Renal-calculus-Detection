"""One command: a DICOM study in, one 3D kidney image out.

WHY THIS EXISTS
---------------
Getting from a DICOM zip to 3d_kidneys/<sid>/views.png took four separate
scripts, three of which sweep the whole cohort when you only wanted one study.
This drives the same four steps for a single study and stops.

It adds NO new logic. Series selection, HU stacking, segmentation and rendering
are all the existing functions, imported -- so the picture this produces is the
same picture run_part1.sh would produce, not a second implementation that can
drift away from it.

    1. pick series   triage_series.scan_zip + series_verdict + RANK
    2. -> NIfTI      extract_series.load_series      -> nifti/<sid>.nii.gz
    3. -> masks      TotalSegmentator (run_anatomy's ROI list) -> seg/<sid>/
    4. -> picture    render_kidney_3d.render         -> 3d_kidneys/<sid>/

INPUT may be any of:
    a study id          8193874                (looks in dicoms/zips/)
    a zip path          /path/to/study.zip
    a directory         /path/to/loose/dicoms/ (walked recursively)

WHAT IT WILL NOT DO
-------------------
Nothing already on disk is recomputed or overwritten without --force. A study
that is already in the cohort keeps its existing NIfTI, its existing masks and
its existing render; the script just tells you where they are. That is
deliberate -- these directories are shared with every other run.

It also does not run stone detection. Calculi appear in the render only if the
study already has rows in the stones CSV, because detect_stones writes into
$CALCULUS_RUN/csv and running it here would rewrite a finished run's numbers.
For a brand-new study you get kidneys and cysts, which is what this view is
for: checking the segmentation.

Usage:
    ./venv/bin/python utils/dicom_to_3d.py 8193874
    ./venv/bin/python utils/dicom_to_3d.py /data/new_case.zip --id NEW01
    ./venv/bin/python utils/dicom_to_3d.py /data/dcm_folder --out /tmp/case.png
    ./venv/bin/python utils/dicom_to_3d.py 8193874 --device cpu   # GPU is shared
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
import zipfile

import nibabel as nib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# package root -> project root: calculus/<sub>/x.py is two levels down
ROOT = os.path.dirname(os.path.dirname(HERE))

from calculus.common.paths import CSV, NIFTI, SEG, ZIPS        # noqa: E402
from calculus.common import triage_series as tri                    # noqa: E402
from calculus.common import extract_series as ex                    # noqa: E402
from calculus.common import run_anatomy as anat                     # noqa: E402
from calculus.kidney import render_kidney_3d as r3                  # noqa: E402


class _DirArchive:
    """A zipfile.ZipFile stand-in over a directory of loose DICOM files.

    Both scan_zip and load_series talk to their archive through exactly three
    calls -- namelist(), open(name) and the context manager. Supplying those
    over a plain directory lets a loose folder of DICOMs go through the same
    code path as a zip, with no copy of the data and no second reader to keep
    in step with the first.
    """

    def __init__(self, root, *_, **__):
        self.root = root

    def namelist(self):
        out = []
        for dirpath, _, names in os.walk(self.root):
            for n in names:
                out.append(os.path.relpath(os.path.join(dirpath, n), self.root))
        return sorted(out)

    def open(self, name, *_, **__):
        return open(os.path.join(self.root, name), "rb")

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False
# WHAT THIS CLASS DOES: makes a folder of DICOM files behave like an opened zip,
# so a study that was never zipped can be read by the same functions without
# duplicating them or copying gigabytes into a temporary archive.


def _allow_directories():
    """Route zipfile.ZipFile through _DirArchive when handed a directory."""
    real = zipfile.ZipFile

    def factory(path, *a, **k):
        if isinstance(path, str) and os.path.isdir(path):
            return _DirArchive(path)
        return real(path, *a, **k)

    tri.zipfile.ZipFile = factory
    ex.zipfile.ZipFile = factory
# WHAT THIS FUNCTION DOES: patches the two modules that read study archives so
# that passing a directory works exactly like passing a zip.


def resolve(target):
    """(study_id, archive_path) from an id, a zip path, or a directory."""
    if os.path.isdir(target) or os.path.isfile(target):
        p = os.path.normpath(os.path.abspath(target))
        sid = os.path.basename(p)
        if sid.endswith(".zip"):
            sid = sid[:-4]
        return sid, p
    p = os.path.join(ZIPS, f"{target}.zip")
    if not os.path.exists(p):
        raise SystemExit(f"no such study: {target}\n"
                         f"  not a path, and {p} does not exist")
    return str(target), p


def pick_series(path, detect_phase=True):
    """The one series this study should be measured on, with its verdict.

    Same ranking as triage_series: verdict tier first, then thinnest slice, then
    most slices. Reimplemented here only as a three-line sort over the imported
    functions -- the verdict rules themselves are not copied.
    """
    rows = tri.assign_phase(tri.scan_zip(path, detect_phase=detect_phase))
    if not rows:
        return None, rows
    for r in rows:
        r["verdict"] = tri.series_verdict(r)

    def key(r):
        sp = r["slice_spacing_mm"]
        # NaN spacing sorts last rather than unpredictably: a comparison against
        # NaN is false either way, which silently corrupts the ordering.
        return (tri.RANK[r["verdict"]], sp if sp == sp else 1e9, -r["n_slices"])

    return sorted(rows, key=key)[0], rows
# WHAT THIS FUNCTION DOES: looks at every series in the study and returns the
# plain thin axial one that covers the kidneys, using the project's existing
# triage rules rather than a fresh set.


def to_nifti(sid, path, uid, force=False):
    dest = os.path.join(NIFTI, f"{sid}.nii.gz")
    if os.path.exists(dest) and not force:
        print(f"  nifti exists, keeping {dest}")
        return dest
    os.makedirs(NIFTI, exist_ok=True)
    data, affine, meta = ex.load_series(path, uid)
    if data is None:
        raise SystemExit(f"could not stack series {uid} (fewer than 10 slices "
                         f"with a position tag)")
    nib.save(nib.Nifti1Image(data.astype(np.int16), affine), dest)
    print(f"  nifti {meta['shape']} "
          f"{meta['dx']:.2f}x{meta['dy']:.2f}x{meta['dz']:.2f} mm "
          f"[{meta['kernel'] or 'no kernel tag'}] -> {dest}")
    return dest


def segment(sid, nii, device="gpu", fast=False, force=False):
    """TotalSegmentator into seg/<sid>/, or reuse what is already there.

    The full ROI list from run_anatomy is requested, not just the two kidneys
    the render needs. A seg folder holding only kidney masks still looks
    finished to run_anatomy (it tests for kidney_left.nii.gz), so a short cut
    here would leave that study permanently missing the bladder and vessel
    masks Part 2 depends on -- and nothing would ever report it.
    """
    outdir = os.path.join(SEG, sid)
    done = os.path.join(outdir, "kidney_left.nii.gz")
    if os.path.exists(done) and not force:
        n = len(glob.glob(os.path.join(outdir, "*.nii.gz")))
        print(f"  masks exist ({n} in {outdir}), skipping TotalSegmentator")
        return outdir
    cmd = [anat.TS, "-i", nii, "-o", outdir, "--device", device,
           "--roi_subset", *anat.ROIS] + (["--fast"] if fast else [])
    print(f"  TotalSegmentator on {device}, {len(anat.ROIS)} ROIs ...",
          flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("TotalSegmentator failed:\n" + r.stderr[-1500:])
    n = len(glob.glob(os.path.join(outdir, "*.nii.gz")))
    print(f"  {n} masks in {time.time()-t0:.0f}s -> {outdir}")
    return outdir


def main():
    ap = argparse.ArgumentParser(
        description="DICOM study -> one 3D kidney image")
    ap.add_argument("target", help="study id, zip path, or DICOM directory")
    ap.add_argument("--id", default=None,
                    help="study id to file the outputs under (default: the "
                         "zip or folder name)")
    ap.add_argument("--out", default=None,
                    help="also copy the finished PNG here")
    ap.add_argument("--device", default="gpu",
                    help="gpu (default) or cpu. The GPU on this box is shared "
                         "with other services; cpu takes ~10x longer but "
                         "competes with nothing.")
    ap.add_argument("--fast", action="store_true",
                    help="3 mm TotalSegmentator model, ~4x faster, coarser")
    ap.add_argument("--stones-csv",
                    default=os.path.join(ROOT, "run_v5", "csv",
                                         "baseline_stones.csv"),
                    help="accepted stones to draw, if this study is in it")
    ap.add_argument("--no-phase", action="store_true",
                    help="skip the aorta HU read when picking a series: much "
                         "faster, but a contrast study can then be chosen")
    ap.add_argument("--force", action="store_true",
                    help="redo every step, overwriting the NIfTI, the masks "
                         "and the render for this study")
    args = ap.parse_args()

    _allow_directories()
    sid, path = resolve(args.target)
    if args.id:
        sid = args.id
    kind = "directory" if os.path.isdir(path) else "zip"
    print(f"study {sid}   ({kind}: {path})\n")

    # The age gate is a warning here, not a veto: you asked for this study by
    # name. It still matters -- the adult kidney model returns fragmented 30 mL
    # masks on a child, which look like a segmentation bug rather than a
    # model/patient mismatch.
    try:
        from calculus.common.patient_gate import excluded_ids
        if sid in excluded_ids():
            print(f"  WARNING: {sid} is excluded by the patient gate "
                  f"(see {os.path.join(CSV, 'patient_gate.csv')}). The adult "
                  f"model segments paediatric kidneys badly; the render will "
                  f"look wrong for that reason, not because of a bug.\n")
    except Exception:
        pass

    print("1/4  picking the series")
    best, rows = pick_series(path, detect_phase=not args.no_phase)
    if best is None:
        raise SystemExit("no DICOM series found in this archive")
    print(f"  {len(rows)} series; chose {best['series_desc']!r}  "
          f"[{best['verdict']}]  {best['n_slices']} slices  "
          f"{best['slice_spacing_mm']} mm  {best['coverage_mm']} mm coverage")
    if best["verdict"] not in ("measurable", "thin_short_coverage"):
        # Said plainly rather than swallowed: on a bone-kernel or 5 mm series
        # the kidney surface is still usable for a shape check, but any number
        # measured off it is not, and the render carries numbers.
        print(f"  NOTE verdict is '{best['verdict']}', not 'measurable'. The "
              f"shape is worth looking at; the volumes in the title are less "
              f"reliable than on a thin plain series.")

    print("\n2/4  stacking it into a NIfTI volume")
    nii = to_nifti(sid, path, best["series_uid"], force=args.force)

    print("\n3/4  segmenting the kidneys")
    segment(sid, nii, device=args.device, fast=args.fast, force=args.force)

    print("\n4/4  rendering")
    png = os.path.join(r3.OUT, sid, "views.png")
    if os.path.exists(png) and not args.force:
        print(f"  render exists, keeping {png}  (--force to redo)")
    else:
        stones = None
        if os.path.exists(args.stones_csv):
            import pandas as pd
            stones = pd.read_csv(args.stones_csv)
            n = (stones.study_id.astype(str) == sid).sum()
            print(f"  {n} accepted stone(s) for this study in "
                  f"{os.path.basename(args.stones_csv)}")
        else:
            print(f"  no {args.stones_csv} -- kidneys and cysts only")
        print("  " + str(r3.render(sid, stones)))

    if not os.path.exists(png):
        raise SystemExit("the render did not produce a PNG -- see the line "
                         "above for the reason")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                    exist_ok=True)
        shutil.copyfile(png, args.out)
        print(f"  copied to {args.out}")

    print(f"\nIMAGE  {png}")
    stl = os.path.join(r3.OUT, sid, "kidneys.stl")
    if os.path.exists(stl):
        print(f"MESH   {stl}"
              + ("  + stones.stl" if os.path.exists(
                  os.path.join(r3.OUT, sid, "stones.stl")) else ""))


if __name__ == "__main__":
    main()
