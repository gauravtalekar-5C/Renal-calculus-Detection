#!/bin/bash
# Unattended overnight supervisor. Keeps the cohort run alive, reclaims disk,
# chains phase 2, and writes the verdict. Everything here is idempotent: the
# runner skips any study whose JSON already exists, so a restart never loses
# work and never double-counts.
cd "$(dirname "$0")"
PY=/root/Gaurav/kindey_calculus_measurement/venv/bin/python
log() { echo "$(date -Is) [supervisor] $*" >> supervisor.log; }

janitor() {
  # A study is finished once its JSON exists. The NIfTI (~120 MB) and the 17
  # masks (~50 MB) are then dead weight, and 1514 studies of them is ~260 GB.
  # The CSVs, overlays and JSON -- the things anyone would want to re-read --
  # are kept.
  local freed=0
  for j in json/*.json; do
    [ -f "$j" ] || continue
    local sid; sid=$(basename "$j" .json)
    for p in "../nifti/${sid}.nii.gz" "runs/${sid}/nifti" "runs/${sid}/seg"; do
      if [ -e "$p" ]; then rm -rf "$p" && freed=$((freed+1)); fi
    done
  done
  [ "$freed" -gt 0 ] && log "janitor reclaimed $freed path(s); disk now $(df -h /root | awk 'NR==2{print $4}') free"
}

run_phase() {
  local file="$1" label="$2" target="$3"
  for attempt in 1 2 3 4 5 6; do
    local have; have=$(ls json/*.json 2>/dev/null | wc -l)
    if [ "$have" -ge "$target" ]; then
      log "$label complete ($have/$target JSON)"; return 0
    fi
    log "$label attempt $attempt (have $have/$target)"
    COHORT_FILE="$file" $PY -u runner.py >> progress.log 2>&1
    log "$label attempt $attempt exited rc=$?"
    janitor
    $PY score.py > /dev/null 2>&1
    sleep 60
  done
  log "$label gave up after 6 attempts"
}

log "supervisor starting; waiting for the phase-1 runner already in flight"
while pgrep -f "[r]unner.py" > /dev/null; do
  sleep 120
  janitor
  $PY score.py > /dev/null 2>&1        # RESULT.txt stays current all night
done
log "phase-1 runner exited"

run_phase cohort.csv "phase 1" 518
log "starting phase 2: the remaining 996 reported-negative studies"
run_phase cohort_phase2.csv "phase 2" 1514

janitor
$PY score.py > /dev/null 2>&1
log "ALL DONE -- see RESULT.txt"
