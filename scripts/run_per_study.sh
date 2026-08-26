#!/usr/bin/env bash
# PER-STUDY inference. One study goes all the way through before the next one
# starts -- the same path a deployment takes when a single DICOM arrives.
#
#   for each study:  triage -> extract -> segment -> mask sheet
#                           -> KIDNEY stones -> overlays
#                           -> URETERIC stones -> review sheet
#                           -> report tables -> full report
#
# WHY THIS SHAPE AND NOT STAGE-BY-STAGE
# Production never has a cohort. It has one study and a question. Driving the
# same code per study means what we measure here is what runs there: the same
# order, the same caching, the same failure modes. A batch sweep can hide a bug
# that only bites when one study is asked for on its own -- infer_study was in
# exactly that state, pointing at pre-restructure file paths that no cohort run
# ever touched.
#
# A failed study does NOT stop the loop. It is recorded and the next one starts,
# because one unreadable archive should not cost a night's run.
#
# Segmentation and detection never overlap: inside one study they are strictly
# sequential, and studies run one at a time. That matters -- when a detector's
# worker pool was alive alongside nnU-Net's dataloader, the two deadlocked
# (65 min on a 100 s study, GPU at 0%, threads in futex_wait).
#
# COHORT AGGREGATION happens once at the end. It is not inference: joining the
# per-study CSVs and scoring against the audit are reporting steps that need
# every study present by definition.
#
# Usage:  scripts/run_per_study.sh                  every zip in $CALCULUS_ZIPS
#         scripts/run_per_study.sh 8349056 8393951  named studies only
#         SKIP_URETERIC=1 scripts/run_per_study.sh  kidney only, much faster
set -uo pipefail
cd "$(dirname "$0")/.."
D=Missed_cases
OUT="${OUT:-Results_missed}"
PY="${PY:-/root/Gaurav/kindey_calculus_measurement/venv/bin/python}"

export CALCULUS_ZIPS="${CALCULUS_ZIPS:-$D/zips}"
export CALCULUS_NIFTI="${CALCULUS_NIFTI:-$D/nifti}"
export CALCULUS_SEG="${CALCULUS_SEG:-$D/seg}"
export CALCULUS_RUN="$OUT"
export CALCULUS_TS="${CALCULUS_TS:-/root/Gaurav/kindey_calculus_measurement/venv/bin/TotalSegmentator}"

EXTRA=""
[ "${SKIP_URETERIC:-0}" = "1" ] && EXTRA="--skip-ureteric"

if [ "$#" -gt 0 ]; then
  STUDIES=("$@")
else
  mapfile -t STUDIES < <(for z in "$CALCULUS_ZIPS"/*.zip; do
                           basename "$z" .zip; done | sort)
fi
N=${#STUDIES[@]}
mkdir -p "$OUT"
LEDGER="$OUT/per_study_ledger.csv"
[ -f "$LEDGER" ] || echo "study_id,status,seconds,finished_at" > "$LEDGER"

echo "=============================================================="
echo " PER-STUDY INFERENCE      $N studies, one at a time"
echo " results ->               $OUT"
echo " ledger  ->               $LEDGER"
echo " started                  $(date)"
echo "=============================================================="

i=0; ok=0; bad=0
for sid in "${STUDIES[@]}"; do
  i=$((i + 1))
  # Already finished? The ledger is the record, so a restart costs nothing.
  if grep -q "^$sid,ok," "$LEDGER" 2>/dev/null; then
    echo "[$i/$N] $sid  already done, skipping"
    ok=$((ok + 1)); continue
  fi
  echo
  echo "########## [$i/$N] $sid  $(date +%H:%M:%S) ##########"
  t0=$SECONDS
  if nice -n 10 "$PY" -u -m calculus.pipeline.infer_study \
        "$CALCULUS_ZIPS/$sid.zip" --id "$sid" $EXTRA 2>&1 \
        | grep --line-buffered -v Warning; then
    st=ok;   ok=$((ok + 1))
  else
    st=fail; bad=$((bad + 1))
  fi
  echo "$sid,$st,$((SECONDS - t0)),$(date +%H:%M:%S)" >> "$LEDGER"
  echo "########## [$i/$N] $sid $st in $((SECONDS - t0))s  (ok=$ok fail=$bad)"
done

echo
echo "=============================================================="
echo " COHORT AGGREGATION  (reporting, not inference)"
echo "=============================================================="
nice -n 10 "$PY" -u -m calculus.kidney.kidney_qc 2>&1 | grep -v Warning
nice -n 10 "$PY" -u -m calculus.report.combine_stone_analysis 2>&1 | grep -v Warning
# Scores our detections against the audit's "what the radiologist missed" text.
# Recall only -- see the module docstring for why precision is not computable
# on this cohort.
nice -n 10 "$PY" -u -m calculus.report.score_missed 2>&1 | grep -v Warning || true

echo
echo "=== DONE $(date)   ok=$ok fail=$bad of $N ==="
echo "   reports  -> $OUT/reports/"
echo "   overlays -> $OUT/overlays/"
echo "   masks    -> $OUT/mask_overlays/"
echo "   ledger   -> $LEDGER"
