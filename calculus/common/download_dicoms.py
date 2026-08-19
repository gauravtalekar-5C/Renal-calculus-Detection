"""Download study DICOMs from the 5C API into ./dicoms/.

Endpoint returns a zip of the whole study:
    https://api.5cnetwork.com/dicom/download/<study_iuid>

Studies are large (0.2-1.5 GB each, ~6 MB/s), so this script is built to be
interrupted and resumed:
  * downloads to <study_id>.zip.part, renames only after the zip verifies
  * skips studies already present and valid
  * appends one row per study to manifest.csv
  * stops before filling the disk (--min-free-gb)

NOTE the server sets Content-Disposition to the patient name; we ignore it and
name files by study_id so nothing PHI-ish lands on disk.

Usage:
    python download_dicoms.py --worklist worklist_pilot.csv
    python download_dicoms.py --worklist worklist_phase1.csv --workers 4
    python download_dicoms.py --worklist worklist_pilot.csv --tier ureteric
    python download_dicoms.py --worklist worklist_pilot.csv --limit 5 --dry-run
    python download_dicoms.py --iuid 1.2.840.113704.9.1000.16.0.20260731160940285
"""
import argparse
import csv
import os
import shutil
import signal
import sys
import threading
import time
import zipfile
import concurrent.futures as cf

import pandas as pd
import requests

# this file lives in utils/, so the project root is one level up. All data
# directories hang off ROOT, never off the script's own folder.
HERE = os.path.dirname(os.path.abspath(__file__))
# package root -> project root: calculus/<sub>/x.py is two levels down
ROOT = os.path.dirname(os.path.dirname(HERE))
DICOM_DIR = os.path.join(ROOT, "dicoms")
ZIP_DIR = os.path.join(DICOM_DIR, "zips")
MANIFEST = os.path.join(DICOM_DIR, "manifest.csv")

API = "https://api.5cnetwork.com/dicom/download/{iuid}"
CHUNK = 1 << 20            # 1 MiB
CONNECT_TIMEOUT = 30
# The API zips the study server-side before streaming, and the gateway kills
# the request with HTTP 504 at ~300 s. Keep the read timeout just above that so
# we observe the real 504 instead of a client-side timeout we can't interpret.
READ_TIMEOUT = 330
RETRIES = 3
BACKOFF = 30               # seconds, multiplied by attempt number

MANIFEST_COLS = ["study_id", "study_iuid", "tier", "family", "variant",
                 "calculus_flag", "status", "bytes", "n_files", "seconds",
                 "attempts", "error"]

_stop = threading.Event()
_lock = threading.Lock()


def _sigint(signum, frame):
    if _stop.is_set():
        print("\nsecond interrupt - exiting hard", flush=True)
        sys.exit(130)
    print("\ninterrupt received - finishing in-flight studies, then stopping "
          "(ctrl-c again to force)", flush=True)
    _stop.set()


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def load_manifest():
    """study_id -> row, for resume."""
    if not os.path.exists(MANIFEST):
        return {}
    done = {}
    with open(MANIFEST, newline="") as f:
        for row in csv.DictReader(f):
            done[str(row["study_id"])] = row
    return done


def append_manifest(row):
    new = not os.path.exists(MANIFEST)
    with _lock:
        with open(MANIFEST, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
            if new:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in MANIFEST_COLS})


def verify_zip(path):
    """Return n_files if the zip is readable and non-empty, else None."""
    try:
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if not n.endswith("/")]
            if not names:
                return None
            if z.testzip() is not None:     # first corrupt member, if any
                return None
            return len(names)
    except (zipfile.BadZipFile, OSError):
        return None


def download_one(rec, session, args):
    """Fetch a single study. Returns the manifest row."""
    sid = str(rec["study_id"])
    iuid = str(rec["study_iuid"]).strip()
    dest = os.path.join(ZIP_DIR, f"{sid}.zip")
    part = dest + ".part"

    row = {"study_id": sid, "study_iuid": iuid, "tier": rec.get("tier", ""),
           "family": rec.get("family", ""), "variant": rec.get("variant", ""),
           "calculus_flag": rec.get("calculus_flag", ""), "attempts": 0}

    if os.path.exists(dest):
        n = verify_zip(dest)
        if n:
            row.update(status="skip", bytes=os.path.getsize(dest),
                       n_files=n, seconds=0)
            return row
        os.remove(dest)   # corrupt leftover, re-fetch

    last_err = ""
    for attempt in range(1, RETRIES + 1):
        if _stop.is_set():
            row.update(status="aborted", error="user interrupt")
            return row
        row["attempts"] = attempt
        t0 = time.time()
        got = 0
        try:
            with session.get(API.format(iuid=iuid), stream=True,
                             timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)) as r:
                if r.status_code != 200:
                    last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                    if r.status_code in (502, 503, 504):
                        # Server could not prepare the archive within the
                        # gateway's ~300 s limit - the study is in cold storage
                        # and that will not change in the next minute. Retrying
                        # costs 300 s per attempt and has never succeeded in
                        # testing, so bail out now and record it for a later
                        # bulk sweep.
                        row["status_hint"] = "gateway_timeout"
                        break
                    time.sleep(BACKOFF * attempt)
                    continue
                with open(part, "wb") as f:
                    for chunk in r.iter_content(CHUNK):
                        if _stop.is_set():
                            raise KeyboardInterrupt
                        f.write(chunk)
                        got += len(chunk)

            n = verify_zip(part)
            if n is None:
                # server sometimes returns a JSON error body with HTTP 200
                with open(part, "rb") as f:
                    head = f.read(200)
                last_err = f"not a valid zip ({got} B); head={head[:120]!r}"
                os.remove(part)
                time.sleep(BACKOFF * attempt)
                continue

            os.replace(part, dest)
            row.update(status="ok", bytes=got, n_files=n,
                       seconds=round(time.time() - t0, 1))
            return row

        except KeyboardInterrupt:
            if os.path.exists(part):
                os.remove(part)
            row.update(status="aborted", error="user interrupt")
            return row
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if os.path.exists(part):
                os.remove(part)
            if attempt < RETRIES:
                time.sleep(BACKOFF * attempt)

    status = "timeout_504" if row.get("status_hint") == "gateway_timeout" else "fail"
    row.update(status=status, bytes=0, n_files=0, seconds=0, error=last_err)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worklist",
                    default=os.path.join(ROOT, "csv", "worklist_pilot.csv"))
    ap.add_argument("--iuid", default=None,
                    help="download ONE study by StudyInstanceUID, no worklist "
                         "needed. study_id is looked up in calculus.xlsx if "
                         "present, else the iuid becomes the filename.")
    ap.add_argument("--study-id", default=None,
                    help="filename to save as, when using --iuid")
    ap.add_argument("--workers", type=int, default=3,
                    help="parallel studies; keep low, each is 0.2-1.5 GB")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tier", default=None, help="only this tier")
    ap.add_argument("--min-free-gb", type=float, default=150.0,
                    help="stop starting new studies below this free space")
    ap.add_argument("--max-gb", type=float, default=None,
                    help="stop after downloading this much this run")
    ap.add_argument("--order", default="newest",
                    choices=["newest", "oldest", "priority"],
                    help="newest: fastest to retrieve, best for pilots. "
                         "oldest: rescues studies nearest the ~6-week retention "
                         "cliff first, at the cost of more 504s. "
                         "priority: strict tier order.")
    ap.add_argument("--retry-failed", action="store_true",
                    help="re-attempt studies previously marked fail")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _sigint)
    os.makedirs(ZIP_DIR, exist_ok=True)

    if args.iuid:
        iuid = args.iuid.strip()
        sid = args.study_id
        if sid is None:
            # prefer the real study_id so the file matches the rest of the
            # dataset; fall back to the iuid, which is not PHI
            xlsx = os.path.join(ROOT, "calculus.xlsx")
            if os.path.exists(xlsx):
                cohort = pd.read_excel(xlsx)
                hit = cohort[cohort.study_iuid.astype(str).str.strip() == iuid]
                if len(hit):
                    sid = str(hit.iloc[0].study_id)
            sid = sid or iuid
        wl = pd.DataFrame([{"study_id": sid, "study_iuid": iuid, "tier": "adhoc",
                            "priority": 0, "family": "", "variant": "",
                            "calculus_flag": "",
                            "report_created_at": "1970-01-01T00:00:00+00:00"}])
        print(f"single study: iuid={iuid} -> dicoms/zips/{sid}.zip")
    else:
        wl = pd.read_csv(args.worklist)
    if args.tier:
        wl = wl[wl["tier"] == args.tier]
    wl["_reported"] = pd.to_datetime(wl["report_created_at"], format="mixed",
                                     utc=True)
    if args.order == "newest":
        wl = wl.sort_values(["priority", "_reported"], ascending=[True, False])
    elif args.order == "oldest":
        wl = wl.sort_values("_reported", ascending=True)
    else:
        wl = wl.sort_values(["priority", "study_id"])

    done = load_manifest()
    todo = []
    for rec in wl.to_dict("records"):
        sid = str(rec["study_id"])
        prev = done.get(sid)
        if prev and prev["status"] in ("ok", "skip"):
            continue
        if prev and prev["status"] in ("fail", "timeout_504") and not args.retry_failed:
            continue
        todo.append(rec)
    if args.limit:
        todo = todo[:args.limit]

    print(f"worklist : {args.worklist} ({len(wl)} studies)")
    print(f"already  : {sum(1 for v in done.values() if v['status'] in ('ok','skip'))}")
    print(f"to fetch : {len(todo)}")
    print(f"free disk: {free_gb(DICOM_DIR):.0f} GB (floor {args.min_free_gb:.0f} GB)")
    if args.dry_run:
        for r in todo[:20]:
            print(f"  {r['study_id']}  {r['tier']:<10} {r.get('variant','')}")
        print("dry run - nothing downloaded")
        return

    session = requests.Session()
    session.headers["User-Agent"] = "5c-calculus-downloader/1.0"

    t_start = time.time()
    counts = {"ok": 0, "skip": 0, "fail": 0, "timeout_504": 0, "aborted": 0}
    total_bytes = 0
    n_done = 0

    def submit_ready():
        return (not _stop.is_set()
                and free_gb(DICOM_DIR) > args.min_free_gb
                and (args.max_gb is None or total_bytes / 1e9 < args.max_gb))

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        it = iter(todo)
        futures = {}
        # prime the pool
        for _ in range(args.workers):
            rec = next(it, None)
            if rec is None or not submit_ready():
                break
            futures[ex.submit(download_one, rec, session, args)] = rec

        while futures:
            done_set, _ = cf.wait(futures, return_when=cf.FIRST_COMPLETED)
            for fut in done_set:
                rec = futures.pop(fut)
                row = fut.result()
                append_manifest(row)
                counts[row["status"]] = counts.get(row["status"], 0) + 1
                total_bytes += int(row.get("bytes") or 0)
                n_done += 1
                mb = (row.get("bytes") or 0) / 1e6
                secs = row.get("seconds") or 0
                rate = f"{mb/secs:.1f} MB/s" if secs else "-"
                msg = (f"[{n_done}/{len(todo)}] {row['study_id']} "
                       f"{row['tier']:<9} {row['status']:<7} "
                       f"{mb:7.0f} MB  {row.get('n_files','')} files  {rate}")
                if row.get("error"):
                    msg += f"  ERR {row['error'][:120]}"
                print(msg, flush=True)

                if submit_ready():
                    nxt = next(it, None)
                    if nxt is not None:
                        futures[ex.submit(download_one, nxt, session, args)] = nxt
                elif not _stop.is_set():
                    reason = ("disk floor reached" if free_gb(DICOM_DIR) <= args.min_free_gb
                              else "--max-gb reached")
                    print(f"** {reason}; draining in-flight downloads **", flush=True)
                    _stop.set()

    dt = time.time() - t_start
    print(f"\nDONE in {dt/60:.1f} min | ok={counts['ok']} skip={counts['skip']} "
          f"fail={counts['fail']} timeout_504={counts['timeout_504']} "
          f"aborted={counts['aborted']}")
    print(f"downloaded {total_bytes/1e9:.1f} GB -> {ZIP_DIR}")
    if counts["ok"]:
        print(f"mean {total_bytes/1e9/counts['ok']*1000:.0f} MB/study | "
              f"free disk now {free_gb(DICOM_DIR):.0f} GB")
    if counts["fail"]:
        print("re-run with --retry-failed to retry failures")


if __name__ == "__main__":
    main()
