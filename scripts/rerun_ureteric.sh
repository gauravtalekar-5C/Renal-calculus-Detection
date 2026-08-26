#!/usr/bin/env bash
# Re-run URETERIC detection + reports across the cohort, after a change that
# affects only the ureteric verdict.
#
# WHY NOT THE WHOLE PIPELINE. The compactness test (tubular_not_stone) changes
# ureteric verdicts. The kidney-side change of the same batch adds columns
# (measurement_flag, fill_fraction) without altering any verdict, and the PUJ
# de-duplication is report-time. So kidney detection output is unchanged and
# re-running it would cost an hour to reproduce identical numbers.
#
# EVERY study is re-run, including ones whose result cannot change, so the whole
# cohort is produced by one version of the code. Mixing versions across a cohort
# is how a regression hides: half the studies carry the old rule and the
# comparison means nothing.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT="${OUT:-final_check_deployment}"
PY=/root/Gaurav/kindey_calculus_measurement/venv/bin/python
W="${W:-3}"
export CALCULUS_RUN="$PWD/$OUT"
export CALCULUS_NIFTI="${CALCULUS_NIFTI:-$PWD/case_analysis/nifti}"
export CALCULUS_SEG="${CALCULUS_SEG:-$PWD/case_analysis/seg}"

# wait for anything still running from the previous phase
while pgrep -f '[-]m calculus.pipeline.infer_study' >/dev/null; do sleep 20; done

mapfile -t IDS < <(ls "$CALCULUS_NIFTI"/*.nii.gz | xargs -n1 basename | sed 's/\.nii\.gz$//' | sort)
echo "=== URETERIC RE-RUN  ${#IDS[@]} studies, ${W}-wide  $(date)"

i=0
for sid in "${IDS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$W" ]; do sleep 10; done
  # same RAM discipline as the main runner
  while [ "$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)" -lt 25 ]; do
    echo "  ... waiting for RAM"; sleep 30
  done
  i=$((i+1))
  ( echo "  [$i/${#IDS[@]}] $sid start $(date +%H:%M:%S)"
    nice -n 10 $PY -u -m calculus.ureter.detect_ureteric --studies "$sid" \
      --overwrite > "$OUT/logs/${sid}_ureteric_v2.log" 2>&1 \
      && echo "  [$i/${#IDS[@]}] $sid ok $(date +%H:%M:%S)" \
      || echo "  [$i/${#IDS[@]}] $sid FAIL $(date +%H:%M:%S)" ) &
  sleep 5
done
wait
echo "=== detection done $(date +%H:%M:%S); regenerating reports and overlays"

for sid in "${IDS[@]}"; do
  nice -n 10 $PY -u -m calculus.report.make_report --study "$sid" 2>&1 | grep -E "dropped|calculus rows" || true
  nice -n 10 $PY -u -m calculus.report.make_report_full --study "$sid" >/dev/null 2>&1 || true
  rm -f "$OUT/overlays/${sid}_ureteric.png"
  nice -n 10 $PY -u -m calculus.ureter.render_ureteric_overlays --studies "$sid" \
      --rejected 3 --overwrite >/dev/null 2>&1 || true
done

echo "=== rebuilding the validation pack"
nice -n 10 $PY -u -m calculus.report.make_validation_pack 2>&1 | tail -25
echo "=== ALL DONE $(date)"
