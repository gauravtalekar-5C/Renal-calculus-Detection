"""Download every study whose report describes a URETERIC calculus.

The cohort on disk was built around renal stones: 37 of its 142 studies have a
ureteric calculus, which is why Part 2's precision has a confidence interval of
42-65 %. There are 1,855 such cases in the report sheet. This fetches them into
their own folder so nothing mixes with the existing cohort.

    ureteric_stone_dataset/
        worklist.csv      the 1,855 studies, with the report sentence
        zips/<id>.zip     one zip per study
        manifest.csv      one row per attempt: ok / fail / timeout_504

WHY A WRAPPER RATHER THAN A NEW DOWNLOADER
------------------------------------------
download_dicoms.py already handles the parts that are easy to get wrong: it
writes to .part and renames only after the zip verifies, resumes without
re-fetching, respects a disk floor, and -- importantly -- names files by
study_id and IGNORES the server's Content-Disposition header, which carries the
patient name. Reimplementing that to change one output path would be a way to
lose those properties. So this module rebinds its three path globals and calls
its main().

THE RETENTION CLIFF IS THE REAL LIMIT
-------------------------------------
The API keeps studies about 33 days. These reports span June-July, so a large
share are already gone and will come back 404 or 504. That is expected, not a
bug: the manifest records the outcome per study, so the yield is measurable
afterwards. `--order oldest` is used deliberately -- it attempts the studies
nearest the cliff first, so anything still rescuable is rescued before it
expires, and the long-gone ones fail fast and cost little.

Usage:
    ./venv/bin/python utils/download_ureteric.py --build-worklist
    nohup ./venv/bin/python utils/download_ureteric.py --workers 3 > dl.log 2>&1 &
    ./venv/bin/python utils/download_ureteric.py --dry-run
"""
import argparse
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# package root -> project root: calculus/<sub>/x.py is two levels down
ROOT = os.path.dirname(os.path.dirname(HERE))

OUT = os.path.join(ROOT, "ureteric_stone_dataset")
ZIPS = os.path.join(OUT, "zips")
WORKLIST = os.path.join(OUT, "worklist.csv")
XLSX = os.path.join(ROOT, "calculus_with_report.xlsx")
SHEET = "jun-jul-2026"

VUJ_RE = re.compile(r"\bvuj\b|\buvj\b|vesico-?urete|vesical junction", re.I)


def build_worklist():
    """Every study whose report attaches a calculus to the ureter."""
    from calculus.evaluate.compare_measurements import (BLADDER_RE, CALC_RE, NOT_CALC_RE,
                                      URETER_RE)
    from calculus.evaluate.compare_reports import clauses, negated

    x = pd.read_excel(XLSX, sheet_name=SHEET)
    x["study_id"] = x.study_id.astype(str)
    rows = []
    for r in x.itertuples():
        line = str(r.calculus_line or "")
        hit = vuj = False
        sent = ""
        for p in clauses(line):
            if not CALC_RE.search(p) or negated(p) or NOT_CALC_RE.search(p):
                continue
            # a bladder-only clause is not ureteric; a clause naming both is
            if BLADDER_RE.search(p) and not URETER_RE.search(p):
                continue
            if URETER_RE.search(p) or VUJ_RE.search(p):
                hit = True
                vuj |= bool(VUJ_RE.search(p))
                sent = sent or p.strip()
        if not hit:
            continue
        rows.append({
            "study_id": r.study_id,
            "study_iuid": r.study_iuid,
            # download_dicoms sorts on these two, so they must exist
            "report_created_at": r.report_created_at,
            "priority": 0 if vuj else 1,     # VUJ cases first: the distal
                                             # measurement is what we cannot
                                             # currently validate at all
            "tier": "ureteric",
            "family": r.family,
            "variant": r.variant,
            "mentions_vuj": vuj,
            "sentence": sent[:300],
        })
    d = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    d.to_csv(WORKLIST, index=False)
    print(f"wrote {WORKLIST}  ({len(d)} studies)")
    print(f"  mentioning the VUJ/UVJ : {int(d.mentions_vuj.sum())}")
    print(f"  variant mix            : {d.variant.value_counts().to_dict()}")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-worklist", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-free-gb", type=float, default=300.0)
    ap.add_argument("--max-gb", type=float, default=None)
    ap.add_argument("--order", default="newest",
                    choices=["newest", "oldest", "priority"])
    ap.add_argument("--max-age-days", type=float, default=40.0,
                    help="skip studies older than this. The API keeps studies "
                         "~33 days, and a study past retention costs a worker a "
                         "full 300 s HTTP 504 before failing -- so attempting "
                         "the 919 June studies would burn ~37 h to return "
                         "nothing. 40 gives a margin either side of the cliff.")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.build_worklist or not os.path.exists(WORKLIST):
        build_worklist()
        if a.build_worklist:
            return

    # Drop the studies that are certainly past retention BEFORE handing the
    # worklist over. Ordering alone is not enough: oldest-first spent 8 minutes
    # on three June studies and returned zero bytes, all three HTTP 504.
    if a.max_age_days:
        import pandas as _pd
        wl = _pd.read_csv(WORKLIST)
        t = _pd.to_datetime(wl.report_created_at, format="mixed", utc=True)
        age = (_pd.Timestamp.utcnow() - t).dt.days
        keep = wl[age <= a.max_age_days]
        live = os.path.join(OUT, "worklist_live.csv")
        keep.to_csv(live, index=False)
        print(f"age filter <= {a.max_age_days:.0f} days: {len(keep)} of "
              f"{len(wl)} studies kept ({len(wl)-len(keep)} past retention)")
        globals()["WORKLIST_USED"] = live
    else:
        globals()["WORKLIST_USED"] = WORKLIST

    import download_dicoms as dl
    # redirect the downloader's output into our own folder. Rebinding the module
    # globals works because its functions read them at call time.
    dl.DICOM_DIR = OUT
    dl.ZIP_DIR = ZIPS
    dl.MANIFEST = os.path.join(OUT, "manifest.csv")
    os.makedirs(ZIPS, exist_ok=True)

    argv = ["download_dicoms.py", "--worklist", WORKLIST_USED,
            "--workers", str(a.workers), "--order", a.order,
            "--min-free-gb", str(a.min_free_gb)]
    if a.limit:
        argv += ["--limit", str(a.limit)]
    if a.max_gb:
        argv += ["--max-gb", str(a.max_gb)]
    if a.retry_failed:
        argv += ["--retry-failed"]
    if a.dry_run:
        argv += ["--dry-run"]
    sys.argv = argv
    print(f"downloading into {ZIPS}\n")
    dl.main()


if __name__ == "__main__":
    main()
