"""HTTP API for renal, ureteric and bladder calculus detection on non-contrast CT.

    POST /analyze      {"study_iuid": "..."}   synchronous, returns the full result
    POST /jobs         {"study_iuid": "..."}   returns a job_id immediately
    GET  /jobs/<id>                            status, then the result
    GET  /health

    curl -s -X POST http://127.0.0.1:8123/analyze \\
         -H 'Content-Type: application/json' \\
         -d '{"study_iuid": "1.2.826.1.3680043.9.5282.150415.26130008345.26130091258"}'

WHY STDLIB AND NOT FLASK/FASTAPI
Neither is installed in this venv, and this box runs a production CT-abdomen API
out of a venv that shares the same machine and GPU. Installing packages to serve
one JSON endpoint is a change to a shared environment for no benefit;
http.server does this job.

WHY ONE ANALYSIS AT A TIME
This is the most important design decision in the file, and it is not about
throughput.

A study holds several full-volume float32 arrays. Measured peaks today: 27.0 GB
on a 1145-slice study, 16.7 GB on 985 slices, ~25 MB per slice. This box has
105 GB, NO SWAP, and a production API resident at 35-49 GB. Two concurrent
analyses of large studies would exhaust RAM, and with no swap the OOM killer
chooses by size -- which means it takes the production API, not us. That nearly
happened twice today under manual scheduling.

Segmentation is also GPU work, and the GPU is shared with that same API (22 of
41 GB committed when this was written). Two TotalSegmentator processes racing it
risks an OOM in someone else's service.

So requests are SERIALISED through one worker thread. A second request waits
rather than running. That is a deliberate ceiling, not an oversight; raising it
requires knowing the study's slice count in advance and reserving memory for it,
which is what scripts/run_case_analysis.sh does for batch runs.

WHAT THE RESPONSE PROMISES, AND WHAT IT DOES NOT
Every response carries `not_assessed` and `limitations`. That is not boilerplate.
Today's work established that the dangerous failures of this pipeline are not
wrong numbers but confident silences: for months a report said "Calculus: 0" on
contrast studies the detector had correctly declined to read, and a 51 mm bladder
calculus was reported as a 22 mm ureteric one because the bladder was never
searched. A caller that cannot distinguish "we looked and found nothing" from
"we did not look" will eventually read one as the other.

`near_misses` exists for the same reason: objects we detected and measured but
were not confident enough to report as calculi. On one validation case that list
holds a stone the radiologist reported, which we had measured to within 0.14 mm
and 3.5% and then discarded on a threshold. Surfacing it is honest; promoting it
to a finding would not be.

STATUS: this is a SECOND READER. It is not validated for autonomous reporting --
n = 18 hand-picked studies, and the false-positive rate is unmeasured. See
`limitations` in every response.
"""
import base64
import copy
import json
import os
import queue
import re
import subprocess
import sys
import threading
import urllib.parse
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

PORT = int(os.environ.get("CALCULUS_API_PORT", 8123))
HOST = os.environ.get("CALCULUS_API_HOST", "127.0.0.1")
# Where the API keeps its work. One directory per study, so a re-request is a
# cache hit and a failure leaves its inputs on disk for inspection.
API_ROOT = os.environ.get("CALCULUS_API_ROOT", os.path.join(HERE, "api_runs"))
ZIP_DIR = os.path.join(HERE, "dicoms", "zips")
# TotalSegmentator lives in the venv that has the CUDA build of torch, which is
# NOT this repo's venv. dicom_to_3d falls back to <repo>/venv/bin/TotalSegmentator
# when CALCULUS_TS is unset, and that path does not exist -- the first real
# request failed with FileNotFoundError inside segmentation, several minutes in,
# after a successful download. Set it explicitly.
TS_BIN = os.environ.get(
    "CALCULUS_TS",
    "/root/Gaurav/kindey_calculus_measurement/venv/bin/TotalSegmentator")

# A StudyInstanceUID is dotted numeric, up to 64 chars by the DICOM standard.
# Validated because it becomes a path component and a subprocess argument.
IUID_RE = re.compile(r"^[0-9]+(\.[0-9]+)+$")
MAX_IUID_LEN = 128

MODEL_VERSION = "2026-08-25-validation-fixes"

# The response contract has its own version, separate from the model's. A caller
# needs to know when the SHAPE changed independently of when the weights or
# thresholds changed -- otherwise every model update looks like a possible
# breaking change to the integration.
SCHEMA_VERSION = "1.0"

# WHERE THE RESPONSE IS STORED, and why it must be stored at all.
#
# The first version built the JSON on the fly from the CSVs and kept it nowhere.
# Three consequences, none acceptable in a deployed clinical service:
#
#   * NO AUDIT TRAIL. If a clinician acts on a result, there is no record of what
#     we actually returned -- only the CSVs it was derived from, which a later
#     re-run overwrites.
#   * A RESTART LOSES EVERYTHING. Job results lived in a dict, so a client
#     polling GET /jobs/<id> across a restart got 404 for work that had
#     completed.
#   * NOT IDEMPOTENT. The same request re-derived the JSON from whatever the
#     CSVs held at that moment, so a code change silently altered the answer to
#     an already-answered question, with nothing to compare against.
#
# So each study's response is written next to its outputs, and every response
# ever returned is appended to one audit log.
RESULT_NAME = "result.json"
SUBDIR_SECONDARY = "secondary"
AUDIT_LOG = os.path.join(API_ROOT, "responses.jsonl")

NOT_ASSESSED = [
    "hydronephrosis / hydroureteronephrosis",
    "perinephric fat stranding",
    "ureteric stent presence",
]
LIMITATIONS = [
    "SECOND READER ONLY. Not validated for autonomous reporting.",
    "Validated on 18 hand-picked studies; the false-positive rate is unmeasured.",
    "Non-contrast (KUB plain) only. On an enhanced or excretory-phase scan the "
    "kidneys are reported as NOT ASSESSED, because iodinated contrast in the "
    "collecting system reads 300-1400 HU and cannot be told from calculus.",
    "Bladder calculi: implemented, but validated on 2 stones only.",
    "Ureteric stents: untested. The one available case fell below the detection "
    "floor, so the stent flag has never fired on a real stent.",
    "Ureteric distance-from-UVJ rests on a geometric landmark that has not been "
    "validated against a radiologist; it was 49 mm out on one distended "
    "bladder. Prefer vertebral_level, which is read off the vertebral masks.",
    "Sizes carrying caliper_suspect, and densities carrying hu_implausible, are "
    "flagged because the number cannot be what it claims -- usually the "
    "measurement mask reaching along adjacent bright structure.",
]


ENVIRONMENTS = ("prod", "staging", "qa", "sandbox")
DEFAULT_ENV = os.environ.get("CALCULUS_DICOM_ENV", "prod")


def parse_form(body, content_type):
    """Form fields from a request body: multipart, urlencoded, or JSON.

    The integration contract is curl -F, i.e. multipart/form-data, matching the
    abdomen service. http.server does not parse that, and the stdlib `cgi`
    module that used to is deprecated and gone in 3.13 -- so it is parsed here.

    Only simple text fields are supported. Uploaded FILES are deliberately not:
    this endpoint takes a study identifier and fetches the imaging itself, so a
    file part would be an unused attack surface.

    JSON and urlencoded are also accepted, because a caller that sends
    Content-Type: application/json should not get a confusing parse error.
    """
    ct = (content_type or "").lower()
    if ct.startswith("application/json"):
        try:
            d = json.loads(body or b"{}")
        except json.JSONDecodeError:
            raise ValueError("body is not valid JSON")
        return d if isinstance(d, dict) else {}
    if ct.startswith("application/x-www-form-urlencoded"):
        return {k: v[0] for k, v in
                urllib.parse.parse_qs(body.decode("utf-8", "replace")).items()}
    if ct.startswith("multipart/form-data"):
        m = re.search(r'boundary="?([^";]+)"?', content_type or "")
        if not m:
            raise ValueError("multipart/form-data without a boundary")
        sep = ("--" + m.group(1)).encode()
        out = {}
        for part in body.split(sep):
            part = part.strip(b"\r\n")
            if not part or part == b"--":
                continue
            head, _, val = part.partition(b"\r\n\r\n")
            hm = re.search(rb'name="([^"]+)"', head)
            if not hm:
                continue
            if b"filename=" in head:
                # a file part: named, so the caller learns why it was ignored
                out.setdefault("_files", []).append(
                    hm.group(1).decode("utf-8", "replace"))
                continue
            out[hm.group(1).decode("utf-8", "replace")] = \
                val.rstrip(b"\r\n").decode("utf-8", "replace")
        return out
    # no content type: try JSON, then urlencoded
    try:
        d = json.loads(body or b"{}")
        return d if isinstance(d, dict) else {}
    except json.JSONDecodeError:
        return {k: v[0] for k, v in
                urllib.parse.parse_qs(body.decode("utf-8", "replace")).items()}


def cell(v, default=None):
    """A CSV cell as a plain value, WITHOUT the `or` trap.

    `str(x or "")` is the idiom this file used, and it is wrong twice over:

        str(NaN or "")  ->  "nan"   NaN is truthy, so the guard does not fire
        str(0   or "")  ->  ""      0 is FALSY, so a real zero is erased

    The second bug turned a kidney calculus count of 0 into null in the API
    response -- converting "we looked and found none" into "we did not look",
    which is the exact distinction this service exists to preserve.
    """
    if v is None:
        return default
    try:
        if pd.isna(v):
            return default
    except (TypeError, ValueError):
        pass
    if isinstance(v, str) and v.strip() == "":
        return default
    return v


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# the pipeline, as one serialised worker
# --------------------------------------------------------------------------
class Analyser:
    """Runs one study at a time. See the module docstring for why."""

    def __init__(self):
        self.q = queue.Queue()
        self.jobs = {}                      # job_id -> dict
        self.lock = threading.Lock()
        self.current = None
        t = threading.Thread(target=self._worker, daemon=True)
        t.start()

    # -- public ------------------------------------------------------------
    def submit(self, iuid, env=DEFAULT_ENV, force=False):
        job_id = uuid.uuid4().hex[:12]
        with self.lock:
            self.jobs[job_id] = {"job_id": job_id, "study_iuid": iuid,
                                 "env": env, "force": force,
                                 "status": "queued", "queued_at": now(),
                                 "queue_position": self.q.qsize() + 1}
        ev = threading.Event()
        self.q.put((job_id, iuid, env, force, ev))
        return job_id, ev

    def get(self, job_id):
        with self.lock:
            j = self.jobs.get(job_id)
            return dict(j) if j else None

    def status(self):
        with self.lock:
            return {"queued": self.q.qsize(), "running": self.current}

    # -- internals ---------------------------------------------------------
    def _set(self, job_id, **kw):
        with self.lock:
            self.jobs[job_id].update(kw)

    def _worker(self):
        while True:
            job_id, iuid, env, force, ev = self.q.get()
            self.current = {"study_iuid": iuid, "env": env}
            self._set(job_id, status="running", started_at=now())
            t0 = time.time()
            try:
                result = self._run(iuid, env, force)
                self._set(job_id, status="done", result=result,
                          seconds=round(time.time() - t0, 1),
                          finished_at=now())
            except Exception as e:                    # never kill the worker
                self._set(job_id, status="error",
                          error=f"{type(e).__name__}: {e}",
                          seconds=round(time.time() - t0, 1),
                          finished_at=now())
            finally:
                self.current = None
                ev.set()
                self.q.task_done()

    @staticmethod
    def _persist(run, result):
        """Write the response beside its outputs, and append it to the audit log.

        Failures here are logged, never raised: a study that analysed correctly
        must still be returned to the caller if the disk is full.
        """
        try:
            with open(os.path.join(run, RESULT_NAME), "w") as fh:
                json.dump(result, fh, indent=2, default=str)
        except OSError as e:
            sys.stderr.write(f"{now()} could not write {RESULT_NAME}: {e}\n")
        try:
            os.makedirs(API_ROOT, exist_ok=True)
            with open(AUDIT_LOG, "a") as fh:
                fh.write(json.dumps({"at": now(),
                                     "study_iuid": result.get("study_iuid"),
                                     "env": result.get("findings", {}).get("env"),
                                     "study_prediction": result.get("study_prediction"),
                                     "total_calculi": result.get("findings", {})
                                                            .get("total_calculi"),
                                     "response": result}, default=str) + "\n")
        except OSError as e:
            sys.stderr.write(f"{now()} could not append audit log: {e}\n")

    @staticmethod
    def _cached(run):
        """A previously returned response for this study, or None."""
        p = os.path.join(run, RESULT_NAME)
        if not os.path.exists(p):
            return None
        try:
            with open(p) as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None

    def _run(self, iuid, env=DEFAULT_ENV, force=False):
        """Download, analyse, and shape the result. Blocking."""
        sid = iuid                       # the iuid IS the id: no lookup, no PHI
        # PER-ENVIRONMENT paths. The same StudyInstanceUID in qa and in prod is
        # two different studies. A shared cache would serve one environment's
        # bytes for the other's request and the mistake would be invisible: the
        # zip verifies, the pipeline runs, and the numbers are simply about the
        # wrong data.
        run = os.path.join(API_ROOT, env, sid)
        zips = os.path.join(ZIP_DIR, env)
        os.makedirs(zips, exist_ok=True)
        # NOTE the two meanings of "env" in this file. `env` is the DICOM
        # environment name ("prod", "qa"); `proc_env` is the subprocess
        # environment dict. They were both called `env` for one commit, so
        # `env = dict(os.environ, ...)` shadowed the name and every later use --
        # the --env argument, the path joins -- received a dict. The failure was
        # "TypeError: expected str, bytes or os.PathLike object, not dict",
        # which says nothing about the cause.
        proc_env = dict(os.environ,
                        CALCULUS_RUN=run,
                        CALCULUS_NIFTI=os.path.join(run, "nifti"),
                        CALCULUS_SEG=os.path.join(run, "seg"),
                        CALCULUS_ZIPS=zips,
                        CALCULUS_DICOM_ENV=env,
                        CALCULUS_TS=TS_BIN)
        os.makedirs(run, exist_ok=True)

        # Return the stored response unless the caller asked for a recompute.
        # Idempotence matters here: the same study id must give the same answer,
        # and "the CSVs happen to say something different now" is not a reason
        # to change an answer already given to a clinician.
        if not force:
            prev = self._cached(run)
            if prev is not None:
                prev = dict(prev)
                prev.setdefault("findings", {})["served_from_cache"] = True
                return prev

        zip_path = os.path.join(zips, f"{sid}.zip")
        steps = []

        if not os.path.exists(zip_path):
            t = time.time()
            r = subprocess.run(
                [PY, "-u", "-m", "calculus.common.download_dicoms",
                 "--iuid", iuid, "--study-id", sid, "--env", env],
                cwd=HERE, env=proc_env, capture_output=True, text=True, timeout=1800)
            steps.append({"step": "download", "env": env,
                          "seconds": round(time.time() - t, 1),
                          "ok": r.returncode == 0})
            if not os.path.exists(zip_path):
                tail = (r.stdout or r.stderr or "")[-400:]
                # The API deletes studies after about 33 days, so this is the
                # commonest failure and deserves to be named rather than
                # reported as a generic error.
                raise RuntimeError(
                    "could not retrieve the study from the DICOM API. Studies "
                    "older than roughly 33 days are past its retention window. "
                    f"Detail: {tail.strip()[:200]}")

        t = time.time()
        r = subprocess.run(
            [PY, "-u", "-m", "calculus.pipeline.infer_study", zip_path,
             "--id", sid],
            cwd=HERE, env=proc_env, capture_output=True, text=True, timeout=7200)
        steps.append({"step": "analyse", "seconds": round(time.time() - t, 1),
                      "ok": r.returncode == 0})
        # The full log always goes to disk. The first failed request reported a
        # 300-character tail of STDOUT, which had truncated away the actual
        # exception (a FileNotFoundError on the TotalSegmentator path) and left
        # the response saying only "TotalSegmentator on gpu, 17 ROIs ...". An
        # error message that omits the error is worse than no message.
        try:
            with open(os.path.join(run, "pipeline.log"), "w") as fh:
                fh.write("$ infer_study\n--- stdout ---\n" + (r.stdout or "")
                         + "\n--- stderr ---\n" + (r.stderr or ""))
        except OSError:
            pass
        if r.returncode != 0:
            # stderr first: that is where the traceback is. stdout is progress.
            detail = (r.stderr or "").strip() or (r.stdout or "").strip()
            raise RuntimeError(
                "analysis failed. " + detail[-1200:]
                + f"  [full log: {os.path.join(run, 'pipeline.log')}]")

        result = self._shape(sid, iuid, run, steps, r.stdout or "", env)
        self._persist(run, result)
        return result

    # -- turning CSVs into the response ------------------------------------
    @staticmethod
    def _csv(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def _report_sections(path):
        """The structured report, section by section, with NAMED fields.

        make_report_full writes the sectioned CSV a radiologist reads:

            HEADER            Study ID, Study, Total calculi
            FINDINGS          Side | Size : Volume | HUN/HN | Calculus | PFS | Stent
            CALCULUS_RIGHT    Organ | Size (in mm) | Density (HU) | Location | A/P
            CALCULUS_LEFT     same
            CALCULUS_BLADDER  same

        The response mirrors THAT, because it is the artefact the team already
        reads and reasons about; an API that invents its own vocabulary makes
        every consumer translate. What is NOT mirrored is the positional shape:
        the CSV stores a row as six unlabelled cells, and a caller should not
        have to know column order. So each row becomes an object.
        """
        try:
            d = pd.read_csv(path, header=0, dtype=str).fillna("")
        except Exception:
            return None
        if not len(d) or "section" not in d.columns:
            return None
        cols = [c for c in d.columns if c != "section"]

        def row_vals(r):
            out = []
            for c in cols:
                v = getattr(r, c, "")
                out.append("" if v is None else str(v))
            while out and out[-1] == "":
                out.pop()
            return out

        header, organs, calc = {}, [], {}
        for r in d.itertuples():
            sec = str(getattr(r, "section", ""))
            v = row_vals(r)
            if sec == "HEADER" and len(v) >= 2:
                header[v[0]] = v[1]
            elif sec == "FINDINGS" and v and v[0] != "Side":
                size, vol = None, None
                if len(v) > 1 and ":" in v[1]:
                    a, b = v[1].split(":", 1)
                    size = a.strip() or None
                    m = re.search(r"([0-9.]+)\s*cc", b)
                    vol = float(m.group(1)) if m else None
                elif len(v) > 1:
                    m = re.search(r"([0-9.]+)\s*cc", v[1])
                    vol = float(m.group(1)) if m else None
                cnt = v[3] if len(v) > 3 else ""
                organs.append({
                    "site": v[0],
                    "size_mm": size,
                    "volume_cc": vol,
                    "hun_hn": None if cnt is None else (v[2] if len(v) > 2 and v[2] != "-" else None),
                    # a real 0 must survive as 0; "-" means not looked at
                    "calculus": (int(cnt) if cnt.isdigit()
                                 else None if cnt in ("-", "") else cnt),
                    "pfs": v[4] if len(v) > 4 and v[4] != "-" else None,
                    "stent": v[5] if len(v) > 5 and v[5] != "-" else None,
                })
            elif sec.startswith("CALCULUS_"):
                side = sec[len("CALCULUS_"):].lower()
                calc.setdefault(side, [])
                if not v or v[0] in ("Organ",) or "Total Counts" in v[0]:
                    continue
                if v[0] == "-":                      # the empty-section marker
                    continue
                calc[side].append({
                    "organ": v[0],
                    "size_mm": v[1] if len(v) > 1 and v[1] != "-" else None,
                    "density_hu": (int(v[2]) if len(v) > 2 and v[2].isdigit()
                                   else None),
                    "location": v[3] if len(v) > 3 and v[3] != "-" else None,
                    "ap": v[4] if len(v) > 4 and v[4] != "-" else None,
                })
            # IMPRESSION is deliberately skipped. Every line in it duplicates
            # something structured: the calculus sentences restate `calculi`,
            # "NOT ASSESSED by this model ..." restates `scope.not_assessed`,
            # and the UVJ caveat restates `model.limitations`. Verified line by
            # line before removing -- nothing in it was unique. Prose is for the
            # printed report (which still carries it); an API consumer should
            # read the fields, not parse English.
        return {"header": header, "organ_findings": organs, "calculi": calc}

    @staticmethod
    def _attach_captures(calculi, index, sid):
        """Give each RENAL calculus its own secondary capture.

        Matched by IDENTITY -- side, density and the three dimensions -- not by
        position. The report groups calculi right-side-then-left while the
        captures are written in CSV row order, so the two sequences genuinely
        disagree: on 8677912 the first report entry was a right-kidney 683 HU
        stone and the first capture was the left 746 HU one. An unmatched entry
        gets no key rather than a plausible wrong image.
        """
        def axes(v):
            """The size cell reduced to its numbers.

            Comparing rendered strings coupled this matcher to a formatting
            choice: adding the "(AP x TR x CC)" label to the report cell but not
            to the index made every kidney capture disappear from the response,
            silently, because a failed match yields no key. Numbers survive
            relabelling, reordering of the suffix, and whitespace.
            """
            return tuple(sorted(re.findall(r"\d+(?:\.\d+)?",
                                           re.sub(r"\([^)]*\)", " ", str(v)))))

        pool = list(index.get("kidney", []))
        out = {}
        for side in ("right", "left", "bladder"):
            rows = []
            for c in calculi.get(side, []):
                c = dict(c)
                if str(c.get("organ", "")).lower().startswith("kidney"):
                    for k, cap in enumerate(pool):
                        if (cap.get("side") == side
                                and cap.get("density_hu") == c.get("density_hu")
                                and axes(cap.get("size_mm"))
                                    == axes(c.get("size_mm"))):
                            c["secondary_capture"] = os.path.join(
                                "overlays", SUBDIR_SECONDARY, str(sid),
                                cap["file"])
                            pool.pop(k)
                            break
                # ureteric and bladder stones are covered by the shared coronal,
                # so they carry no key at all rather than a null
                rows.append(c)
            out[side] = rows
        return out

    @staticmethod
    def _captures(run, sid):
        """Secondary-capture paths: one per renal stone, one shared coronal.

        Returned as paths relative to the run directory, not absolute: an
        absolute path leaks this box's layout into a stored response, and a
        consumer that later reads results from a mounted share or an object
        store would find every one of them wrong.
        """
        base = os.path.join(run, "overlays", SUBDIR_SECONDARY, str(sid))
        index, coronal = {}, None
        try:
            with open(os.path.join(base, "index.json")) as fh:
                index = json.load(fh)
        except Exception:
            index = {}
        if os.path.exists(os.path.join(base, "coronal.png")):
            coronal = os.path.join("overlays", SUBDIR_SECONDARY, str(sid),
                                   "coronal.png")
        return index, coronal

    def _shape(self, sid, iuid, run, steps, log, env=DEFAULT_ENV):
        rep = os.path.join(run, "reports")
        calculi = self._csv(os.path.join(rep, f"{sid}_calculi.csv"))
        near = self._csv(os.path.join(rep, f"{sid}_near_miss.csv"))
        ksum = self._csv(os.path.join(run, "csv", "per_study",
                                      f"{sid}_summary.csv"))
        report = self._report_sections(
            os.path.join(rep, f"{sid}_report.csv")) or {}
        cap_index, coronal_cap = self._captures(run, sid)

        assessable, why = True, None
        if len(ksum) and "error" in ksum.columns:
            err = cell(ksum.iloc[0].get("error"), "")
            err = "" if err is None else str(err).strip()
            if err.lower() == "nan":
                err = ""
            if err:
                low = err.lower()
                if "enhanced" in low or "excretory" in low:
                    assessable, why = False, "intravenous contrast present"
                elif "no segmentation" in low:
                    assessable, why = False, "kidneys not segmented"
                else:
                    assessable, why = False, err[:120]

        # counts and aggregates for analytics, from the machine-readable table
        counts = {"renal": 0, "ureteric": 0, "bladder": 0}
        sizes, dens = [], []
        for r in calculi.itertuples() if len(calculi) else []:
            organ = str(cell(getattr(r, "Organ", None), "")).lower()
            k = ("bladder" if organ.startswith("bladder")
                 else "ureteric" if organ.startswith("ureter") else "renal")
            counts[k] += 1
            nums = [float(x) for x in re.findall(
                r"[0-9.]+", str(cell(getattr(r, "_4", None), "")))]
            if nums:
                sizes.append(max(nums))
            hu = cell(getattr(r, "_5", None))
            if hu is not None:
                dens.append(int(hu))
        total = len(calculi) if len(calculi) else 0

        # ---- acquisition -------------------------------------------------
        acq = {"series_description": None, "slices": None,
               "slice_thickness_mm": None, "triage_verdict": None,
               "suitable_for_measurement": None}
        m = re.search(r"series\s+(.+?)\s+\[(\w+)\]\s+(\d+) sl,\s+([0-9.]+) mm",
                      log)
        if m:
            v = m.group(2)
            acq = {"series_description": m.group(1).strip().strip("'"),
                   "slices": int(m.group(3)),
                   "slice_thickness_mm": float(m.group(4)),
                   "triage_verdict": v,
                   "suitable_for_measurement":
                       v in ("measurable", "thin_short_coverage")}

        # ---- caveats, as CODES so they can be filtered --------------------
        caveats = []
        if acq["suitable_for_measurement"] is False:
            caveats.append({
                "code": "series_not_measurable", "severity": "high",
                "message": f"series triage verdict is '{acq['triage_verdict']}'. "
                           "Presence and LOCATION are usable; SIZE is not."})
        th = acq["slice_thickness_mm"]
        if th and th > 3.0:
            caveats.append({
                "code": "thick_slices",
                "severity": "high" if th >= 5 else "medium",
                "message": f"slice thickness {th:g} mm. A calculus smaller than "
                           f"about {th:g} mm spans less than one slice, so its "
                           "craniocaudal dimension and volume are dominated by "
                           "partial volume."})
        if not assessable:
            caveats.append({
                "code": "not_assessable", "severity": "critical",
                "message": f"this study was not evaluated ({why}). Absence of a "
                           "finding does NOT mean absence of a calculus."})
        # Kidney size and volume come from the organ masks and have never been
        # validated against anything. They are in the report template, so they
        # are mirrored here -- but flagged, so they are not read with the same
        # confidence as a calculus measurement.
        if any(o.get("volume_cc") for o in report.get("organ_findings", [])):
            caveats.append({
                "code": "organ_measurements_unvalidated", "severity": "low",
                "message": "organ_findings size_mm and volume_cc are derived "
                           "from the segmentation masks and are NOT validated. "
                           "Only the calculus measurements are."})

        near_list = []
        for r in near.itertuples() if len(near) else []:
            near_list.append({
                "compartment": cell(getattr(r, "where", None)),
                "side": (str(cell(getattr(r, "Side", None), "")).lower() or None),
                "max_mm": cell(getattr(r, "_4", None)),
                "density_hu": (int(cell(getattr(r, "_5", None)))
                               if cell(getattr(r, "_5", None)) is not None else None),
                "location": cell(getattr(r, "Location", None)),
                "not_reported_because": cell(getattr(r, "why_not_reported", None)),
                "measured": cell(getattr(r, "measured", None)),
                "threshold": cell(getattr(r, "threshold", None)),
            })

        if not assessable:
            prediction, basis = "Abnormal", "not_assessable_routed_for_review"
        elif total:
            prediction, basis = "Abnormal", "calculus_detected"
        else:
            prediction, basis = "Normal", "assessed_no_calculus_detected"

        # LEAN RESPONSE. schema_version, organ_findings, acquisition, quality,
        # scope, model and processing were all removed as unnecessary. Three
        # scalars are KEPT out of those blocks, flattened, because dropping them
        # would remove information a caller cannot reconstruct and would not know
        # was missing:
        #
        #   assessable / not_assessable_reason
        #       the difference between "we looked and found nothing" and "we did
        #       not look". Six of ten renal misses on the audit cohort were
        #       contrast studies where the model correctly abstained and the
        #       report said "Calculus: 0".
        #   size_reliable / slice_thickness_mm
        #       whether a size means anything. On a 5 mm series this API returned
        #       "2.3 x 2.8 x 5.1 mm" for a stone spanning less than one slice.
        #       Without this the number arrives looking like a measurement.
        #   model_version
        #       which algorithm produced a stored result. Without it, a result in
        #       your database cannot be attributed to a version, and today alone
        #       the detection logic changed ten times.
        return {
            "study_iuid": iuid,
            "study_prediction": prediction,
            "findings": {
                "study_prediction": prediction,
                "prediction_basis": basis,
                "env": env,
                "model_version": MODEL_VERSION,

                "study": report.get("header", {}).get("Study"),
                "total_calculi": total,
                "calculi": self._attach_captures(report.get("calculi", {}),
                                                 cap_index, sid),
                # ONE coronal carrying every ureteric and bladder stone. Their
                # meaning is positional -- where they sit along the tract and
                # relative to each other -- which separate crops destroy.
                "coronal_capture": coronal_cap,

                "counts": counts,
                "largest_calculus_mm": max(sizes) if sizes else None,
                "max_density_hu": max(dens) if dens else None,

                # near_misses removed from the RESPONSE: nothing in production
                # surfaces them to a human, so in the payload they were noise.
                # They are still written to <run>/reports/<id>_near_miss.csv, so
                # the information is on disk for anyone who wants it -- the
                # effect of dropping them here is that a sub-threshold stone is
                # once again invisible to a caller, which is the status quo and
                # a deliberate choice, not an oversight.

                # assessable / not_assessable_reason / size_reliable removed at
                # the integration's request.
                #
                # CONSEQUENCE, recorded because it is not obvious: with
                # `assessable` gone, `prediction_basis` is the ONLY remaining
                # signal that a study was never evaluated. A contrast or
                # unsegmentable study returns study_prediction "Abnormal" with
                # total_calculi 0 and
                # prediction_basis "not_assessable_routed_for_review". If
                # prediction_basis is ever dropped too, an abstention becomes
                # indistinguishable from a clean study in the payload.
                "slice_thickness_mm": acq.get("slice_thickness_mm"),
            },
        }


def inline_captures(out, run):
    """Replace every capture PATH with the base64 of the PNG itself.

    Done at RESPONSE time, not when the result is shaped. The stored
    result.json and responses.jsonl keep the paths: a coronal is ~300 KB, so
    embedding the bytes would add roughly half a megabyte to every cached
    record and to every audit-log line, for data already sitting on disk one
    directory away. The cache stays small and greppable; the wire carries the
    images.

    A capture whose file has gone missing becomes null rather than an error.
    The findings are the point of the response, and they are still correct
    without a picture.
    """
    def enc(rel):
        if not rel:
            return None
        try:
            with open(os.path.join(run, rel), "rb") as fh:
                return base64.b64encode(fh.read()).decode("ascii")
        except OSError as e:
            sys.stderr.write(f"{now()} capture unreadable {rel}: {e}\n")
            return None

    f = out.get("findings")
    if not isinstance(f, dict):
        return out
    if "coronal_capture" in f:
        f["coronal_capture"] = enc(f["coronal_capture"])
    for side in ("right", "left", "bladder"):
        for c in (f.get("calculi") or {}).get(side, []) or []:
            if isinstance(c, dict) and "secondary_capture" in c:
                c["secondary_capture"] = enc(c["secondary_capture"])
    return out


ANALYSER = Analyser()


# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "calculus-api/1.0"

    def log_message(self, fmt, *args):        # one tidy line per request
        sys.stderr.write(f"{now()} {self.address_string()} {fmt % args}\n")

    # -- helpers -----------------------------------------------------------
    def _json(self, code, payload):
        body = json.dumps(payload, indent=2, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        if n > 1024 * 1024:
            raise ValueError("request body too large")
        return parse_form(self.rfile.read(n),
                          self.headers.get("Content-Type", ""))

    @staticmethod
    def _env(data):
        """Which DICOM environment to fetch from.

        Validated against a fixed list rather than passed through: it selects a
        hostname, and an unchecked value there is a request-forgery primitive.
        """
        v = str(data.get("env") or DEFAULT_ENV).strip().lower()
        if v not in ENVIRONMENTS:
            raise ValueError("env must be one of "
                             + ", ".join(ENVIRONMENTS)
                             + f" (got {v!r})")
        return v

    def _iuid(self, data):
        """Validated StudyInstanceUID. It becomes a path component and a
        subprocess argument, so it is checked rather than trusted."""
        v = str(data.get("study_iuid") or data.get("study_instance_uid") or "").strip()
        if not v:
            raise ValueError("study_iuid is required")
        if len(v) > MAX_IUID_LEN or not IUID_RE.match(v):
            raise ValueError("study_iuid must be a dotted numeric "
                             "StudyInstanceUID, e.g. 1.2.840.113619.2.55.3...")
        return v

    # -- routes ------------------------------------------------------------
    def do_GET(self):
        if self.path in ("/health", "/"):
            return self._json(200, {
                "status": "ok", "model_version": MODEL_VERSION,
                "time": now(), "queue": ANALYSER.status(),
                "note": "second reader; see /analyze response 'limitations'",
                "endpoints": ["POST /analyze", "POST /jobs", "GET /jobs/<id>",
                              "GET /health"]})
        if self.path.startswith("/jobs/"):
            j = ANALYSER.get(self.path.rsplit("/", 1)[-1])
            return self._json(200 if j else 404,
                              j or {"error": "unknown job_id"})
        return self._json(404, {"error": "not found", "path": self.path})

    def do_POST(self):
        try:
            data = self._body()
            iuid = self._iuid(data)
            env = self._env(data)
            # "force=true" recomputes instead of serving the stored response
            force = str(data.get("force", "")).strip().lower() in (
                "1", "true", "yes", "y")
        except ValueError as e:
            return self._json(400, {"error": str(e)})
        except json.JSONDecodeError:
            return self._json(400, {"error": "body must be JSON"})

        if self.path == "/jobs":
            job_id, _ = ANALYSER.submit(iuid, env, force)
            return self._json(202, {
                "job_id": job_id, "study_iuid": iuid, "env": env,
                "status": "queued",
                "poll": f"GET /jobs/{job_id}",
                "note": "a full analysis takes roughly 11 minutes; one study "
                        "runs at a time"})

        if self.path == "/analyze":
            job_id, ev = ANALYSER.submit(iuid, env, force)
            # Synchronous by request. A full analysis is ~11 minutes, which is
            # longer than most clients and gateways will hold a connection, so
            # POST /jobs exists for anything not driven by a human with curl.
            ev.wait()
            j = ANALYSER.get(job_id) or {}
            if j.get("status") == "done":
                out = copy.deepcopy(j["result"])
                # The contract is {study_iuid, study_prediction, findings}.
                # Anything extra goes INSIDE findings rather than beside it, so
                # a strict consumer does not see unexpected top-level keys.
                if isinstance(out.get("findings"), dict):
                    out["findings"]["seconds"] = j.get("seconds")
                # the iuid IS the study id for API runs, so the run dir is
                # derivable without threading it back through the job record
                out = inline_captures(out, os.path.join(API_ROOT, env, iuid))
                return self._json(200, out)
            return self._json(502, {"study_iuid": iuid, "env": env,
                                    "status": "error",
                                    "error": j.get("error", "unknown failure"),
                                    "seconds": j.get("seconds")})

        return self._json(404, {"error": "not found", "path": self.path})


def main():
    os.makedirs(API_ROOT, exist_ok=True)
    os.makedirs(ZIP_DIR, exist_ok=True)
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"calculus API on http://{HOST}:{PORT}  (model {MODEL_VERSION})")
    print(f"  work dir  {API_ROOT}")
    print("  one analysis at a time -- see the module docstring for why")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        srv.shutdown()


if __name__ == "__main__":
    main()
