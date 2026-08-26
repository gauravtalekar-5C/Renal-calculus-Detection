#!/usr/bin/env bash
# Sweep SEED_HU and score each value. The last sensitivity gap, made measurable.
#
# WHY ONLY KIDNEY + BLADDER ARE RE-RUN
# The ureteric detector applies HU_FLOOR = 300 to every measured candidate, so a
# candidate admitted by a lower seed is rejected downstream regardless. Lowering
# SEED_HU therefore cannot change ureteric output, and re-running it would spend
# four hours reproducing identical numbers. Verified by reading the verdict
# chain, not assumed: verdict_measured tests hu_max < HU_FLOOR before anything
# else can accept.
#
# WHAT THE ANSWER LOOKS LIKE
# Lowering the seed can only ADD candidates, so `matched` can only rise or hold
# and `unmatched` can only rise or hold. The question is the RATIO. Three
# microliths are currently invisible by construction; if recovering them costs a
# handful of extra candidates that is a good trade, and if it costs sixty it is
# not. Either way it is now a number rather than a judgement call.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/root/Gaurav/kindey_calculus_measurement/venv/bin/python
SRC="${SRC:-final_check_deployment}"
VALUES="${VALUES:-200 170 150}"
W="${W:-3}"

export CALCULUS_NIFTI="$PWD/case_analysis/nifti"
export CALCULUS_SEG="$PWD/case_analysis/seg"
mapfile -t IDS < <(ls "$CALCULUS_NIFTI"/*.nii.gz | xargs -n1 basename | sed 's/\.nii\.gz$//' | sort)

for V in $VALUES; do
  OUT="sweep_seed/seed_$V"
  echo "=============================================================="
  echo " SEED_HU = $V   ->  $OUT    $(date +%H:%M:%S)"
  echo "=============================================================="
  mkdir -p "$OUT/csv/per_study" "$OUT/logs"
  # the ureteric results are seed-independent: copy them rather than recompute
  cp "$SRC"/csv/per_study/*_ureter_*.csv "$OUT/csv/per_study/" 2>/dev/null || true
  cp "$SRC"/csv/ureter_candidates.csv "$OUT/csv/" 2>/dev/null || true

  export CALCULUS_RUN="$PWD/$OUT" CALCULUS_SEED_HU="$V"
  for sid in "${IDS[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "$W" ]; do sleep 8; done
    while [ "$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)" -lt 28 ]; do sleep 25; done
    ( nice -n 10 $PY -u -m calculus.kidney.detect_stones --studies "$sid" \
        --overwrite > "$OUT/logs/${sid}_k.log" 2>&1
      nice -n 10 $PY -u -m calculus.bladder.detect_bladder --studies "$sid" \
        > "$OUT/logs/${sid}_b.log" 2>&1 ) &
    sleep 3
  done
  wait
  for sid in "${IDS[@]}"; do
    nice -n 10 $PY -m calculus.report.make_report --study "$sid" >/dev/null 2>&1
  done
  echo "--- scoring SEED_HU = $V"
  nice -n 10 $PY -m calculus.evaluate.score_cohort --run "$OUT" 2>&1 \
    | grep -E "matched|MISSED|unmatched|SIZE|DENSITY" | sed 's/^/    /'
done
echo "=== SWEEP DONE $(date +%H:%M:%S)"
