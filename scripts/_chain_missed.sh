#!/usr/bin/env bash
# Wait for the download retry to finish, then run inference end to end.
# Sequential by design: segmentation must finish before detection starts, or the
# nnU-Net dataloader deadlocks against our worker pool (seen before: 65 min on a
# 100 s study, GPU idle, threads parked in futex_wait).
set -uo pipefail
cd /root/Gaurav/renal-calculus
RETRY_PID="${1:-0}"

if [ "$RETRY_PID" != "0" ]; then
  echo "waiting on download retry pid $RETRY_PID ..."
  while kill -0 "$RETRY_PID" 2>/dev/null; do sleep 30; done
  echo "retry finished $(date +%H:%M)"
fi

N=$(ls Missed_cases/zips/*.zip 2>/dev/null | wc -l)
echo "launching inference over $N studies"
WORKERS=2 ./scripts/run_missed_cases.sh
