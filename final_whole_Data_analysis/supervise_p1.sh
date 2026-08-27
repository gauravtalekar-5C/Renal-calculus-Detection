#!/bin/bash
# Phase 1 ONLY. Phase 2 was cancelled: measuring a 58% false-positive rate more
# precisely on 996 further negatives answers nothing we do not already know from
# 164 of them.
cd "$(dirname "$0")"
PY=/root/Gaurav/kindey_calculus_measurement/venv/bin/python
log() { echo "$(date -Is) [supervisor-p1] $*" >> supervisor.log; }
janitor() {
  for j in json/*.json; do
    [ -f "$j" ] || continue
    sid=$(basename "$j" .json)
    for p in "runs/${sid}/nifti"; do
      [ -e "$p" ] && rm -rf "$p"
    done
  done
}
log "phase-2 supervisor replaced; phase 1 only, target 518"
while pgrep -f "[r]unner.py" > /dev/null; do
  sleep 120; janitor; $PY score.py > /dev/null 2>&1
done
for attempt in 1 2 3 4; do
  have=$(ls json/*.json 2>/dev/null | wc -l)
  [ "$have" -ge 518 ] && { log "phase 1 complete ($have/518)"; break; }
  log "phase 1 restart $attempt (have $have/518)"
  COHORT_FILE=cohort.csv $PY -u runner.py >> progress.log 2>&1
  janitor; $PY score.py > /dev/null 2>&1; sleep 60
done
janitor; $PY score.py > /dev/null 2>&1
log "PHASE 1 DONE -- phase 2 deliberately not started"
