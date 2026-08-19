#!/usr/bin/env bash
# Ureteric detection over a cohort, then metrics.
#
# Stages run one at a time. Overlapping detection with TotalSegmentator looked
# free (one GPU-bound, one CPU-bound) and was not: with memory down to ~20 GB
# nnU-Net's dataloader deadlocked -- 65 minutes on a 100-second study, GPU at 0 %.
#
# A lock file allows only one instance: two copies once starved each other until
# every worker was OOM-killed and nothing completed for 40 minutes.
#
#   SEGPID=<pid>   wait for that organ-mask job to exit first
#   WORKERS=n      default 3
#
# Usage: SEGPID=12345 scripts/run_ureteric.sh ureteric_whole_stone_data
set -uo pipefail
cd "$(dirname "$0")/.."
NEW="${1:-ureteric_whole_stone_data}"
W="${WORKERS:-3}"
PY=./venv/bin/python
NICE="nice -n 10"
G="grep --line-buffered -v Warning"
LOCK=.ureteric_chain.lock
if [ -e "$LOCK" ] && kill -0 "$(cat $LOCK 2>/dev/null)" 2>/dev/null; then
  echo "already running as pid $(cat $LOCK)"; exit 1
fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT

export CALCULUS_ZIPS="$NEW/zips" CALCULUS_NIFTI="$NEW/nifti" CALCULUS_SEG="$NEW/seg"

if [ "${SEGPID:-0}" != "0" ]; then
  echo "waiting for organ masks (pid $SEGPID)"
  while kill -0 "$SEGPID" 2>/dev/null; do sleep 120; done
  echo "organ masks done at $(date): $(ls -d $NEW/seg/*/ 2>/dev/null | wc -l) studies"
fi

batches () {   # $1 = list file, $2 = run dir, $3 = label
  for pass in $(seq 1 60); do
    local todo="" n dn
    while read -r s; do
      [ -f "$NEW/seg/$s/kidney_left.nii.gz" ] || continue
      [ -f "$2/csv/per_study/${s}_ureter_summary.csv" ] && continue
      todo="$todo $s"
    done < "$1"
    n=$(echo $todo | wc -w)
    dn=$(ls $2/csv/per_study/*_ureter_summary.csv 2>/dev/null | wc -l)
    echo "--- $3 pass $pass $(date +%H:%M) done=$dn remaining=$n ---"
    [ "$n" -eq 0 ] && break
    local fg=$(free -g | awk 'NR==2{print $7}')
    if [ "$fg" -lt 25 ]; then echo "only ${fg} GB free, waiting"; sleep 600; continue; fi
    CALCULUS_RUN="$2" $NICE $PY -u -m calculus.ureter.detect_ureteric \
        --workers "$W" --studies $(echo $todo | tr ' ' '\n' | head -12 | tr '\n' ' ') \
        2>&1 | $G
  done
}

ls $NEW/nifti/*.nii.gz | xargs -n1 basename | sed 's/\.nii\.gz//' | sort > $NEW/all_ids.txt
[ -f "$NEW/subset_100.txt" ] && { echo "=== the 100-study subset first ==="; batches "$NEW/subset_100.txt" "$NEW" subset; }
CALCULUS_RUN="$NEW" $NICE $PY -u -m calculus.ureter.eval_ureteric 2>&1 | $G
CALCULUS_RUN="$NEW" $NICE $PY -u -m calculus.ureter.render_ureteric_overlays --rejected 3 2>&1 | $G

echo "=== the rest of the cohort ==="
batches "$NEW/all_ids.txt" "$NEW" full
CALCULUS_RUN="$NEW" $NICE $PY -u -m calculus.ureter.eval_ureteric 2>&1 | $G

echo "=== main cohort: the report-negative studies, for specificity ==="
CALCULUS_RUN=stone_analysis CALCULUS_NIFTI=nifti CALCULUS_SEG=seg \
    CALCULUS_ZIPS=dicoms/zips $NICE $PY -u -m calculus.ureter.detect_ureteric \
    --workers "$W" 2>&1 | $G
CALCULUS_RUN=stone_analysis $NICE $PY -u -m calculus.ureter.ureter_sweep 2>&1 | $G
echo "=== DONE $(date) ==="
