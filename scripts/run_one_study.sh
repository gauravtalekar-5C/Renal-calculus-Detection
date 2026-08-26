#!/usr/bin/env bash
# One study, end to end, into its own results folder. Same per-study path a
# deployment takes: zip in, overlays + 3 report CSVs out.
#
# Usage:  scripts/run_one_study.sh 8679874 [/path/to/study.zip]
set -uo pipefail
cd "$(dirname "$0")/.."
SID="$1"
ZIP="${2:-/root/Gaurav/kindey_calculus_measurement/dicoms/zips/$SID.zip}"
OUT="${OUT:-study_wise_analysis}"
PY="${PY:-/root/Gaurav/kindey_calculus_measurement/venv/bin/python}"

export CALCULUS_NIFTI="$OUT/nifti"
export CALCULUS_SEG="$OUT/seg"
export CALCULUS_RUN="$OUT/$SID"
export CALCULUS_TS="${CALCULUS_TS:-/root/Gaurav/kindey_calculus_measurement/venv/bin/TotalSegmentator}"
mkdir -p "$CALCULUS_NIFTI" "$CALCULUS_SEG" "$CALCULUS_RUN"

echo "study $SID"
echo "zip   $ZIP  ($(du -h "$ZIP" | cut -f1))"
echo "out   $CALCULUS_RUN"
echo
nice -n 8 "$PY" -u -m calculus.pipeline.infer_study "$ZIP" --id "$SID" 2>&1 \
  | grep --line-buffered -viE "^warning|userwarning"
