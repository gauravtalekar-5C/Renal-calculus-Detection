#!/usr/bin/env bash
# VALIDATION CASE RUN -- 18 studies, 8 calculus categories, for the validation team.
#
#   phase 0  download                  3 at a time, resumable
#   phase 1  prepare  (NIfTI + masks)  STRICTLY ONE AT A TIME  -- GPU
#   phase 2  detect + render           up to MAX_WORKERS at once -- CPU
#   phase 3  cohort roll-up
#
# WHY THE PHASES SPLIT THIS WAY
# Detection is CPU-bound and parallelises across studies. Segmentation is
# GPU-bound and must not: this box shares its GPU with the CT-abdomen API, which
# already held 22 of 41 GB when this was written. Several TotalSegmentator
# instances racing that would risk an OOM in a production service, which is a
# far worse outcome than a slow batch.
#
# Phase 2 still runs the FULL per-study pipeline per study -- infer_study finds
# the masks already on disk and skips to DETECT. This is not stage-by-stage
# cohort processing; each study still goes end to end on its own.
#
# THE RAM GATE, AND WHY A FREE-MEMORY CHECK IS NOT ENOUGH.
#
# The first version only asked "is MIN_FREE_GB available right now?" before
# starting a worker. That is unsound, because a study's footprint GROWS after it
# starts: three workers each passed a 22 GB check and then grew to 93 GB used
# between them and the CT-abdomen API, leaving 11 GB on a box with NO SWAP. On
# no-swap hardware, exhausting RAM invokes the OOM killer, and it chooses by
# size -- which means it takes the API, not us. I had to kill a worker by hand
# twice to prevent that.
#
# So the gate now RESERVES by predicted footprint. Measured peaks:
#
#     1145 slices -> 27.0 GB      ~24 MB/slice
#      985 slices -> 16.7 GB      ~17 MB/slice
#      761 slices -> 16.1 GB      ~21 MB/slice
#      310 slices ->  7.8 GB      ~25 MB/slice
#
# MB_PER_SLICE is set at the top of that range, because underestimating costs an
# OOM in someone else's service and overestimating only costs wall clock.
MB_PER_SLICE="${MB_PER_SLICE:-25}"
#
# PHI: the downloader names files by study_id and ignores Content-Disposition,
# which carries the patient name. Zips stay under dicoms/ and are NOT part of
# what the validation team receives; nifti, PNGs and CSVs are.
#
# Usage:  scripts/run_case_analysis.sh
#         MAX_WORKERS=2 scripts/run_case_analysis.sh
set -uo pipefail
cd "$(dirname "$0")/.."

OUT="${OUT:-case_analysis}"
PY=/root/Gaurav/kindey_calculus_measurement/venv/bin/python
WL=$OUT/worklist_validation.csv
MAX_WORKERS="${MAX_WORKERS:-3}"
MIN_FREE_GB="${MIN_FREE_GB:-22}"

# Respect these if the caller already set them. A re-run against ALREADY
# PREPARED studies must be able to point at an existing nifti/seg cache:
# without this, a second run into a new output folder finds empty caches and
# re-runs TotalSegmentator 17 times on a GPU shared with the CT-abdomen API.
export CALCULUS_ZIPS="${CALCULUS_ZIPS:-$PWD/$OUT/zips}"
export CALCULUS_NIFTI="${CALCULUS_NIFTI:-$PWD/$OUT/nifti}"
export CALCULUS_SEG="${CALCULUS_SEG:-$PWD/$OUT/seg}"
export CALCULUS_RUN="$PWD/$OUT"
export CALCULUS_TS=/root/Gaurav/kindey_calculus_measurement/venv/bin/TotalSegmentator

mkdir -p "$OUT" "$CALCULUS_ZIPS" "$CALCULUS_NIFTI" "$CALCULUS_SEG" "$OUT/logs"
LEDGER=$OUT/case_ledger.csv
: > "$OUT/.running_slices"     # per-worker memory reservations, in slices
[ -f "$LEDGER" ] || echo "study_id,category,pick,status,seconds,finished_at" > "$LEDGER"

mapfile -t ROWS < <($PY - "$WL" <<'PY'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1]).sort_values(["priority", "study_id"])
for r in d.itertuples():
    print(f"{r.study_id}|{r.category}|{r.pick}")
PY
)
N=${#ROWS[@]}
sid_of() { echo "${1%%|*}"; }

echo "=============================================================="
echo " VALIDATION CASE ANALYSIS   $N studies"
echo " workers      $MAX_WORKERS   (ram floor ${MIN_FREE_GB} GB)"
echo " results ->   $PWD/$OUT"
echo " started      $(date)"
echo "=============================================================="

# ---- phase 0: downloads --------------------------------------------------
echo; echo "### PHASE 0  downloads  $(date +%H:%M:%S)"
nohup nice -n 5 $PY -u -m calculus.common.download_dicoms \
      --worklist "$WL" --workers 3 --order priority --retry-failed \
      > "$OUT/logs/download.log" 2>&1 &
DL_PID=$!
echo "downloader pid $DL_PID -> $OUT/logs/download.log"

# ---- phase 1: prepare, one at a time ------------------------------------
echo; echo "### PHASE 1  prepare (NIfTI + masks), serial, GPU  $(date +%H:%M:%S)"
READY=()
WAIT_MAX=3600
for row in "${ROWS[@]}"; do
  sid=$(sid_of "$row")
  src="dicoms/zips/$sid.zip"
  w=0
  while [ ! -f "$src" ]; do
    kill -0 $DL_PID 2>/dev/null || break
    [ $w -ge $WAIT_MAX ] && break
    sleep 20; w=$((w + 20))
  done
  if [ ! -f "$src" ]; then
    echo "  $sid  NO ZIP -- skipping (download failed or past retention)"
    echo "$sid,$(echo "$row"|cut -d'|' -f2),$(echo "$row"|cut -d'|' -f3),no_zip,0,$(date +%H:%M:%S)" >> "$LEDGER"
    continue
  fi
  [ -e "$CALCULUS_ZIPS/$sid.zip" ] || ln -s "$PWD/$src" "$CALCULUS_ZIPS/$sid.zip"
  if [ -d "$CALCULUS_SEG/$sid" ] && [ -n "$(ls -A "$CALCULUS_SEG/$sid" 2>/dev/null)" ]; then
    echo "  $sid  masks already present, skipping prepare"
    READY+=("$row"); continue
  fi
  if nice -n 5 $PY -u -m calculus.pipeline.prepare_study \
        "$CALCULUS_ZIPS/$sid.zip" --id "$sid" 2>&1 | grep -v Warning; then
    READY+=("$row")
  else
    echo "  $sid  PREPARE FAILED"
    echo "$sid,$(echo "$row"|cut -d'|' -f2),$(echo "$row"|cut -d'|' -f3),prepare_fail,0,$(date +%H:%M:%S)" >> "$LEDGER"
  fi
done
echo "### PHASE 1 done: ${#READY[@]} of $N prepared  $(date +%H:%M:%S)"

# ---- phase 2: detect + render, N-wide ------------------------------------
echo; echo "### PHASE 2  detect + render, ${MAX_WORKERS}-wide, CPU  $(date +%H:%M:%S)"
sid_slice_count() {
  # slice count from the NIfTI header. Cheap: nibabel reads the header only.
  $PY - "$CALCULUS_NIFTI/$1.nii.gz" <<'PYSL' 2>/dev/null || echo 400
import sys, nibabel as nib
try:
    print(int(nib.load(sys.argv[1]).shape[2]))
except Exception:
    print(400)          # a safe middling default if the header will not read
PYSL
}


one_study() {
  local row="$1" sid cat pick t0 st
  sid=${row%%|*}; local rest=${row#*|}; cat=${rest%%|*}; pick=${rest##*|}
  if grep -q "^$sid,.*,ok," "$LEDGER" 2>/dev/null; then
    echo "  $sid already done"; return 0
  fi
  t0=$SECONDS
  echo "  START $sid  $cat ($pick)  $(date +%H:%M:%S)"
  if nice -n 10 $PY -u -m calculus.pipeline.infer_study \
        "$CALCULUS_ZIPS/$sid.zip" --id "$sid" > "$OUT/logs/$sid.log" 2>&1; then
    st=ok
  else
    st=fail
  fi
  echo "$sid,$cat,$pick,$st,$((SECONDS - t0)),$(date +%H:%M:%S)" >> "$LEDGER"
  # release this study's reservation
  if [ -f "$OUT/.running_slices" ]; then
    local mine
    mine=$(sid_slice_count "$sid")
    grep -v -m1 -x "$mine" "$OUT/.running_slices" > "$OUT/.running_slices.tmp" 2>/dev/null || true
    mv -f "$OUT/.running_slices.tmp" "$OUT/.running_slices" 2>/dev/null || true
  fi
  echo "  DONE  $sid  $st in $((SECONDS - t0))s"
}

for row in "${READY[@]}"; do
  # cap concurrency
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_WORKERS" ]; do sleep 10; done
  # RAM gate. Two conditions, both required:
  #   1. MemAvailable is at least MIN_FREE_GB right now, AND
  #   2. this study's PREDICTED footprint fits in what is left after the
  #      studies already running finish growing.
  # Condition 2 is the one that matters -- see the MB_PER_SLICE comment above.
  sid_slices=$(sid_slice_count "$sid")
  need_gb=$(( (sid_slices * MB_PER_SLICE) / 1024 + 1 ))
  while true; do
    free_gb=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
    # headroom still owed to workers already running, from their own slice counts
    owed_gb=0
    for rs in $(cat "$OUT/.running_slices" 2>/dev/null); do
      owed_gb=$(( owed_gb + (rs * MB_PER_SLICE) / 1024 ))
    done
    used_gb=$(ps -eo rss,args --no-headers \
              | grep -E '[-]m calculus\.(kidney|ureter)\.detect' \
              | awk '{s+=$1} END {printf "%d", s/1048576}')
    owed_gb=$(( owed_gb > used_gb ? owed_gb - used_gb : 0 ))
    if [ "$free_gb" -ge "$MIN_FREE_GB" ] && \
       [ $(( free_gb - owed_gb )) -ge "$need_gb" ]; then break; fi
    echo "  ... holding $sid (${sid_slices} sl, needs ${need_gb} GB): "\
         "${free_gb} GB free, ${owed_gb} GB still owed to running studies"
    sleep 30
  done
  echo "$sid_slices" >> "$OUT/.running_slices"
  one_study "$row" &
  sleep 5
done
wait

# ---- phase 3 -------------------------------------------------------------
echo; echo "### PHASE 3  cohort roll-up  $(date +%H:%M:%S)"
nice -n 10 $PY -u -m calculus.report.combine_stone_analysis 2>&1 | grep -v Warning || true

ok=$(grep -c ",ok," "$LEDGER" 2>/dev/null || echo 0)
echo
echo "=== DONE $(date)   ok=$ok of $N ==="
echo "   reports  -> $OUT/reports/"
echo "   overlays -> $OUT/overlays/   (kidney per-stone + <sid>_ureteric.png)"
echo "   ledger   -> $LEDGER"
