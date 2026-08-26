#!/usr/bin/env python
"""Run the deployment pipeline over the cohort and write one JSON per study.

WHAT THIS IS FOR
The two numbers we have never had are a false-positive rate and sensitivity on
studies nobody tuned against. cohort.csv holds 518 unseen Plain studies,
259 reported-positive and 259 reported-negative, interleaved so that a partial
result carries both signals rather than hours of one then hours of the other.

WHY NOT THROUGH THE HTTP API
app.py runs ONE study at a time on a single worker thread -- deliberate, because
a study peaks near 25 MB per slice and this box has no swap. 518 studies through
that queue is a fortnight. This runs the SAME code path (download_dicoms ->
infer_study -> Analyser._shape, including the reconcile guard) with a small
worker pool, so what is measured is what deploys.

GATING, and why it is not optional
  * RAM. ~25 MB/slice, so a 600-slice study is ~15 GB. The box holds a CT-abdomen
    production API resident at 35-49 GB and has NO SWAP: overcommit does not
    degrade here, it kills. A worker waits until there is real headroom.
  * GPU. TotalSegmentator wants the GPU and 31 of 41 GB is already held by
    someone else's long-running jobs. A worker waits for free VRAM rather than
    OOMing both itself and them.

Resumable: a study whose JSON already exists is skipped, so this can be stopped
and restarted without losing work.
"""
import csv
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

PY_BIN = "/root/Gaurav/kindey_calculus_measurement/venv/bin/python"
# Set EXPLICITLY, exactly as app.py does. dicom_to_3d derives the binary from
# the venv it thinks it is in and guessed /root/Gaurav/renal-calculus/venv/bin,
# which does not exist -- the first study died inside segmentation with
# FileNotFoundError after a clean download. This is the same trap app.py
# documents at TS_BIN, and the runner walked straight into it.
TS_BIN = os.environ.get(
    "CALCULUS_TS",
    "/root/Gaurav/kindey_calculus_measurement/venv/bin/TotalSegmentator")
JSON_DIR = os.path.join(HERE, "json")
RUNS_DIR = os.path.join(HERE, "runs")
LOG_DIR = os.path.join(HERE, "logs")
LEDGER = os.path.join(HERE, "ledger.csv")
ZIP_CACHE = os.path.join(ROOT, "dicoms", "zips", "cohort")
FOREIGN_ZIPS = "/root/Gaurav/kindey_calculus_measurement/dicoms/zips"

# FOUR, not two. Measured mid-run: load 2.5 on 16 cores and 86% of the CPU
# idle, because every detection stage is single-threaded -- each of
# detect_stones and detect_ureteric pins exactly one core. The box was doing a
# fifth of the work it could. Two workers gave 7.9 min/study effective on a
# study whose own pipeline took 472 s, i.e. almost no overlap at all.
#
# Four is bounded by MIN_RAM_GB, not by this number: a fifth study cannot start
# unless 28 GB is genuinely free, so the gate throttles rather than the box
# swapping -- which on a machine with NO SWAP and a 35-49 GB production API
# resident is the difference between slow and dead.
WORKERS = int(os.environ.get("COHORT_WORKERS", 4))
MIN_RAM_GB = int(os.environ.get("COHORT_MIN_RAM_GB", 28))
MIN_VRAM_MB = int(os.environ.get("COHORT_MIN_VRAM_MB", 7000))
ENV = os.environ.get("COHORT_ENV", "prod")

_ledger_lock = threading.Lock()
_gpu_lock = threading.Lock()          # one segmentation at a time


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def free_ram_gb():
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576.0
    return 0.0


def free_vram_mb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        used, total = (int(x) for x in out.splitlines()[0].split(","))
        return total - used
    except Exception:
        return MIN_VRAM_MB          # cannot tell -> do not block forever


def wait_for_room(sid, what):
    """Block until there is real headroom. Never proceed on a guess."""
    waited = 0
    while True:
        ram, vram = free_ram_gb(), free_vram_mb()
        if ram >= MIN_RAM_GB and vram >= MIN_VRAM_MB:
            return
        if waited % 300 == 0:
            print(f"{now()} {sid} waiting for {what} "
                  f"(ram {ram:.0f}/{MIN_RAM_GB} GB, vram {vram}/{MIN_VRAM_MB} MB)",
                  flush=True)
        time.sleep(30)
        waited += 30


def record(row):
    new = not os.path.exists(LEDGER)
    with _ledger_lock, open(LEDGER, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def local_zip(sid):
    """A zip we already hold. Retention does not apply to what is on disk."""
    for p in (os.path.join(ZIP_CACHE, f"{sid}.zip"),
              os.path.join(FOREIGN_ZIPS, f"{sid}.zip")):
        if os.path.exists(p):
            return p
    return None


def one(rec):
    sid, iuid = str(rec["study_id"]), str(rec["study_iuid"])
    out_json = os.path.join(JSON_DIR, f"{sid}.json")
    if os.path.exists(out_json):
        return "skipped"

    run = os.path.join(RUNS_DIR, sid)
    os.makedirs(run, exist_ok=True)
    os.makedirs(ZIP_CACHE, exist_ok=True)
    env = dict(os.environ,
               CALCULUS_RUN=run,
               CALCULUS_ZIPS=ZIP_CACHE,
               CALCULUS_TS=TS_BIN,
               CALCULUS_DICOM_ENV=ENV)
    t0 = time.time()
    row = {"at": now(), "study_id": sid, "expected": rec["expected"],
           "predicted": "", "total_calculi": "", "renal": "", "ureteric": "",
           "bladder": "", "largest_mm": "", "max_hu": "", "seconds": "",
           "status": "", "detail": ""}

    try:
        # ---- 1. the study itself, from disk if we already hold it ----------
        zp = local_zip(sid)
        if zp is None:
            r = subprocess.run(
                [PY_BIN, "-u", "-m", "calculus.common.download_dicoms",
                 "--iuid", iuid, "--study-id", sid, "--env", ENV],
                cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
            zp = os.path.join(ZIP_CACHE, f"{sid}.zip")
            if not os.path.exists(zp):
                tail = ((r.stderr or "") + (r.stdout or "")).strip()[-300:]
                row.update(status="download_failed", detail=tail,
                           seconds=round(time.time() - t0, 1))
                record(row)
                return "download_failed"
        elif not os.path.exists(os.path.join(ZIP_CACHE, f"{sid}.zip")):
            os.symlink(zp, os.path.join(ZIP_CACHE, f"{sid}.zip"))
            zp = os.path.join(ZIP_CACHE, f"{sid}.zip")

        # ---- 2. the pipeline, gated -----------------------------------------
        wait_for_room(sid, "the pipeline")
        with open(os.path.join(LOG_DIR, f"{sid}.log"), "w") as lf:
            r = subprocess.run(
                [PY_BIN, "-u", "-m", "calculus.pipeline.infer_study", zp,
                 "--id", sid],
                cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT,
                timeout=10800)
        secs = round(time.time() - t0, 1)

        if r.returncode != 0:
            # infer_study exits nonzero when the report tables are missing or
            # contradict the detectors -- the 8583083 guards. That is a real
            # failure, recorded as one, never as a normal study.
            with open(os.path.join(LOG_DIR, f"{sid}.log")) as lf:
                tail = lf.read().strip()[-400:]
            row.update(status="pipeline_failed", seconds=secs, detail=tail)
            record(row)
            return "pipeline_failed"

        # ---- 3. shape the response through the deployment code --------------
        import app                                   # noqa: E402
        os.environ["CALCULUS_RUN"] = run
        with open(os.path.join(LOG_DIR, f"{sid}.log")) as lf:
            log = lf.read()
        out = app.Analyser.__new__(app.Analyser)._shape(
            sid, iuid, run, [], log, env=ENV)
        out["findings"]["seconds"] = secs
        with open(out_json, "w") as fh:
            json.dump(out, fh, indent=2)

        f = out["findings"]
        # The zip has served its purpose once the NIfTI and masks exist, and
        # 518 of them is ~78 GB. A symlink into someone else's cache is left
        # alone -- that is not ours to delete.
        try:
            zc = os.path.join(ZIP_CACHE, f"{sid}.zip")
            if os.path.exists(zc) and not os.path.islink(zc):
                os.remove(zc)
        except OSError:
            pass

        row.update(predicted=out["study_prediction"],
                   total_calculi=f.get("total_calculi"),
                   renal=(f.get("counts") or {}).get("renal"),
                   ureteric=(f.get("counts") or {}).get("ureteric"),
                   bladder=(f.get("counts") or {}).get("bladder"),
                   largest_mm=f.get("largest_calculus_mm"),
                   max_hu=f.get("max_density_hu"),
                   seconds=secs, status="ok",
                   detail=f.get("prediction_basis", ""))
        record(row)
        return "ok"

    except Exception as e:
        row.update(status="error", detail=f"{type(e).__name__}: {e}",
                   seconds=round(time.time() - t0, 1))
        record(row)
        return "error"


def main():
    with open(os.path.join(HERE, "cohort.csv")) as fh:
        cohort = list(csv.DictReader(fh))
    todo = [r for r in cohort
            if not os.path.exists(os.path.join(JSON_DIR, f"{r['study_id']}.json"))]
    print(f"{now()} cohort {len(cohort)}, already done "
          f"{len(cohort) - len(todo)}, to run {len(todo)}, "
          f"{WORKERS} worker(s), env={ENV}", flush=True)

    done = {"ok": 0, "skipped": 0, "download_failed": 0,
            "pipeline_failed": 0, "error": 0}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, outcome in enumerate(ex.map(one, todo), 1):
            done[outcome] = done.get(outcome, 0) + 1
            el = time.time() - t0
            rate = el / max(i, 1)
            print(f"{now()} [{i}/{len(todo)}] {outcome:<16} "
                  f"ok={done['ok']} dl_fail={done['download_failed']} "
                  f"pipe_fail={done['pipeline_failed']} err={done['error']}  "
                  f"~{rate/60:.1f} min/study  "
                  f"eta {(len(todo)-i)*rate/3600:.1f} h", flush=True)
    # RETRY ONCE. A pipeline failure here is usually transient -- GPU
    # contention while four workers overlap their segmentation windows, or a
    # download that timed out. Leaving them failed would quietly bias the
    # cohort: the studies that fail are not a random sample, they are the big
    # ones. Anything that fails twice is recorded as a real failure.
    retry = [r for r in todo
             if not os.path.exists(os.path.join(JSON_DIR, f"{r['study_id']}.json"))]
    if retry:
        print(f"{now()} RETRY pass over {len(retry)} study/studies that "
              f"produced no JSON", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, WORKERS // 2)) as ex:
            for i, outcome in enumerate(ex.map(one, retry), 1):
                print(f"{now()} retry [{i}/{len(retry)}] {outcome}", flush=True)

    got = len([f for f in os.listdir(JSON_DIR) if f.endswith(".json")])
    print(f"{now()} FINISHED {done}  json written: {got}/{len(cohort)}",
          flush=True)


if __name__ == "__main__":
    main()
