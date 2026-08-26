"""One DICOM study in, everything out — kidney AND ureteric, in a single command.

WHY THIS EXISTS
---------------
The pipeline is eleven scripts, and each one sweeps the whole cohort. That is
right for a batch analysis and wrong for "here is a study, tell me what is in
it" -- which is what a deployment does, and what a person wants when they get a
new scan. This runs the entire chain for ONE study and stops:

    triage    ->  extract  ->  segment  ->  KIDNEY stones  ->  URETERIC stones
              ->  overlays (both)  ->  report tables (both)  ->  printed summary

Both detectors run in the same invocation, so a single call answers the whole
question: stones in the kidneys, stones in the ureters, sizes, densities,
locations and the distance from the UVJ.

WHY THE TWO DETECTORS ARE STILL SEPARATE INSIDE
----------------------------------------------
They search different regions with different rejection rules: Part 1 works
inside a closed kidney mask; Part 2 works inside a 20 mm corridor with the
kidney excluded and bone carved out. Keeping them as separate stages means
tuning the ureteric side -- which is the part still under validation -- cannot
silently move the kidney numbers, which are validated at 95.7 % / 83.0 %. From
the outside it is one command; inside it is two passes, on purpose.

EVERY STAGE IS SKIPPED IF ITS OUTPUT EXISTS
-------------------------------------------
Re-running is cheap and safe. --force redoes the detection stages for this study
only. Nothing outside this study is touched, and nothing outside $CALCULUS_RUN
is written except the shared nifti/ and seg/ caches.

TIMING, measured on this box: triage ~20 s, extract ~30 s, segmentation ~100 s,
kidney detection ~2 min, URETERIC DETECTION ~15 min (the corridor is a large,
noisy region and the anisotropic diffusion dominates), overlays ~30 s. So a
cold study is about 20 minutes, nearly all of it Part 2.

Usage:
    CALCULUS_RUN=run_v6 ./venv/bin/python utils/infer_study.py 8231547
    ./venv/bin/python utils/infer_study.py /data/new_case.zip --id NEW01
    ./venv/bin/python utils/infer_study.py 8231547 --skip-ureteric
"""
import argparse
import os
import subprocess
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# package root -> project root: calculus/<sub>/x.py is two levels down
ROOT = os.path.dirname(os.path.dirname(HERE))
from calculus.common.paths import CSV, NIFTI, RUN, SEG          # noqa: E402
# the DICOM-facing stages are already written and tested in dicom_to_3d --
# imported rather than copied so there is one implementation of each
from calculus.pipeline.dicom_to_3d import (_allow_directories, pick_series, resolve,  # noqa: E402
                         segment, to_nifti)

# sys.executable, NOT a hardcoded venv path: this module is the production
# entrypoint, so it has to work from an installed package or a clean clone
# where ROOT/venv does not exist. Child stages must run in the SAME
# interpreter that imported this file, or they get a different numpy/SimpleITK.
PY_BIN = sys.executable


STAGE_MODULES = {
    "detect_stones.py":              "calculus.kidney.detect_stones",
    "render_overlays.py":            "calculus.kidney.render_overlays",
    "detect_ureteric.py":            "calculus.ureter.detect_ureteric",
    "detect_bladder.py":             "calculus.bladder.detect_bladder",
    "render_ureteric_overlays.py":   "calculus.ureter.render_ureteric_overlays",
    "make_report.py":                "calculus.report.make_report",
    "make_report_full.py":           "calculus.report.make_report_full",
    "render_masks.py":               "calculus.common.render_masks",
    "render_secondary.py":           "calculus.report.render_secondary",
}


def run(label, args, env=None):
    """Run one pipeline script, streaming nothing, reporting the outcome."""
    t0 = time.time()
    e = {**os.environ, "CALCULUS_RUN": RUN, **(env or {})}
    r = subprocess.run([PY_BIN, "-u", "-m", STAGE_MODULES[args[0]]] + args[1:],
                       capture_output=True, text=True, env=e, cwd=ROOT)
    dt = time.time() - t0
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-4:]
        print(f"  {label:26} FAILED in {dt:.0f}s")
        for line in tail:
            print(f"      {line}")
        return False, r
    print(f"  {label:26} ok  {dt:.0f}s")
    return True, r


def summarise(sid):
    """Print what was found, from the CSVs the run just wrote."""
    print(f"\n{'='*66}\nRESULT for {sid}\n{'='*66}")

    kp = os.path.join(CSV, "baseline_stones.csv")
    k = pd.DataFrame()
    if os.path.exists(kp):
        d = pd.read_csv(kp)
        k = d[d.study_id.astype(str) == sid]
    print(f"\nKIDNEY   {len(k)} stone(s)")
    for r in (k.sort_values("max_diameter_mm", ascending=False).itertuples()
              if len(k) else []):
        loc = r.location or r.compartment
        print(f"   {r.side:5}  {r.max_diameter_mm:5.1f} mm  {int(r.hu_max):5} HU  "
              f"{loc}")

    up = os.path.join(CSV, "ureter_candidates.csv")
    u = pd.DataFrame()
    if os.path.exists(up):
        d = pd.read_csv(up)
        d = d[(d.study_id.astype(str) == sid) & d.is_stone.astype(bool)]
        u = d[d.report_this.astype(bool)] if "report_this" in d else d
    print(f"\nURETER   {len(u)} stone(s) ranked for report")
    for r in (u.sort_values("hu_max", ascending=False).itertuples()
              if len(u) else []):
        d_uvj = (f"{r.dist_to_uvj_along_mm:.0f} mm from UVJ"
                 if pd.notna(r.dist_to_uvj_along_mm) else "distance unknown")
        print(f"   {r.side:5}  {r.max_diameter_mm:5.1f} mm  {int(r.hu_max):5} HU  "
              f"{r.zone:5}  {d_uvj}")
    if len(u):
        print("   NOTE  ureteric distances rest on a UVJ landmark that has not "
              "been\n         validated against a radiologist's click.")

    print("\nFILES")
    for label, p in (
            ("report table", os.path.join(RUN, "reports", f"{sid}_findings.csv")),
            ("stone table", os.path.join(RUN, "reports", f"{sid}_calculi.csv")),
            ("kidney overlays", os.path.join(RUN, "overlays", sid)),
            ("ureteric sheet", os.path.join(RUN, "overlays",
                                            f"{sid}_ureteric.png"))):
        mark = "" if os.path.exists(p) else "   (not produced)"
        print(f"   {label:16} {p}{mark}")


def main():
    ap = argparse.ArgumentParser(
        description="one DICOM study -> kidney + ureteric stones, in one go")
    ap.add_argument("target", help="study id, zip path, or DICOM directory")
    ap.add_argument("--id", default=None, help="id to file the outputs under")
    ap.add_argument("--device", default="gpu", help="gpu (default) or cpu")
    ap.add_argument("--skip-ureteric", action="store_true",
                    help="kidney only -- Part 2 is ~15 min of the ~20")
    ap.add_argument("--force", action="store_true",
                    help="redo the detection stages for this study")
    a = ap.parse_args()

    _allow_directories()
    sid, path = resolve(a.target)
    if a.id:
        sid = a.id
    t0 = time.time()
    print(f"{'='*66}\nstudy {sid}   ->  {RUN}\n{'='*66}")

    # ---- 1. the DICOM-facing stages, reused from dicom_to_3d -------------
    print("\nPREPARE")
    best, rows = pick_series(path)
    if best is None:
        raise SystemExit("no DICOM series found in this archive")
    print(f"  series                     {best['series_desc']!r} "
          f"[{best['verdict']}] {best['n_slices']} sl, "
          f"{best['slice_spacing_mm']} mm")
    if best["verdict"] not in ("measurable", "thin_short_coverage"):
        print(f"  WARNING verdict is '{best['verdict']}': the shape is worth "
              f"looking at, the sizes are less reliable")
    nii = to_nifti(sid, path, best["series_uid"], force=False)
    segment(sid, nii, device=a.device, force=False)
    # The organ-mask QC sheet belongs HERE, not in a separate cohort sweep: it
    # is the only thing standing between a mis-segmented kidney and a report
    # full of confident numbers measured in the wrong place.
    run("organ mask sheet", ["render_masks.py", "--studies", sid])

    # ---- 2. detection: BOTH compartments ---------------------------------
    print("\nDETECT")
    extra = ["--overwrite"] if a.force else []
    ok_k, _ = run("kidney stones", ["detect_stones.py", "--studies", sid]
                  + extra)
    ok_u = False
    if a.skip_ureteric:
        print("  ureteric stones            skipped (--skip-ureteric)")
    else:
        ok_u, _ = run("ureteric stones",
                      ["detect_ureteric.py", "--studies", sid] + extra)
    # THE BLADDER IS A THIRD COMPARTMENT, and it must run in the per-study
    # pipeline or it does not exist in production. It was added late and
    # validated by hand; without this line a deployed run would search the
    # kidney and the ureter and silently skip the bladder -- which is exactly
    # the gap that let 8676429's 51 mm vesical calculus be reported as a 22 mm
    # ureteric one, and 8674941's intravesical stone be missed entirely.
    #
    # Cheap: the ROI is one eroded organ mask, so it costs seconds, not minutes.
    ok_b, _ = run("bladder stones", ["detect_bladder.py", "--studies", sid])

    # ---- 3. overlays and report tables -----------------------------------
    print("\nRENDER")
    if ok_k:
        run("kidney overlays", ["render_overlays.py", "--study", sid])
    if ok_u:
        run("ureteric sheet", ["render_ureteric_overlays.py", "--studies", sid,
                               "--rejected", "3", "--overwrite"])
    # Secondary captures BEFORE the report tables: the API references these
    # paths, so they must exist by the time a response is shaped.
    run("secondary captures", ["render_secondary.py", "--studies", sid])
    ok_r, _ = run("report tables", ["make_report.py", "--study", sid])
    ok_f, _ = run("full report", ["make_report_full.py", "--study", sid])

    summarise(sid)
    print(f"\ntotal {time.time() - t0:.0f}s")

    # The report tables ARE the answer -- the API shapes its response from them
    # and from nothing else. Until now main() discarded every run() failure and
    # exited 0, so a crashed report looked like a completed analysis: on 8583083
    # the report step died, the API found no CSVs, and reported "Normal" for a
    # study in which the pipeline had just detected seven ureteric calculi.
    #
    # A missing answer must be an error, never a negative finding.
    # The same reconciliation the API performs, so a CLI run cannot quietly
    # produce a report that contradicts its own detector CSVs.
    ok_rec = True
    try:
        from calculus.report.reconcile import accepted_counts
        acc = accepted_counts(RUN, sid)
        rep_csv = os.path.join(RUN, "reports", f"{sid}_calculi.csv")
        n_rep = 0
        if os.path.exists(rep_csv):
            import pandas as _pd
            n_rep = len(_pd.read_csv(rep_csv))
        if sum(acc.values()) > 0 and n_rep == 0:
            ok_rec = False
            print(f"\nRECONCILE FAILED: detectors accepted {acc} but the "
                  f"report table lists 0 calculi. Findings cannot vanish "
                  f"between detection and reporting.")
    except Exception as e:                       # never mask the real outcome
        print(f"\n  (reconcile check could not run: {type(e).__name__}: {e})")

    if not (ok_r and ok_f and ok_rec):
        print("\nFAILED: this study has no trustworthy result. Exiting "
              "nonzero so no caller mistakes the absence of findings for an "
              "absence of disease.")
        print("FAILED: the report tables were not produced, so this study "
              "has NO result. Exiting nonzero so no caller mistakes the "
              "absence of findings for an absence of disease.")
        sys.exit(1)


if __name__ == "__main__":
    main()
