#!/usr/bin/env bash
# A cohort from DICOM zips to reports and metrics.
#
#   scripts/run_cohort.sh                       the main cohort
#   CALCULUS_RUN=my_run scripts/run_cohort.sh   results elsewhere
#
# Resumable at every stage: anything already on disk is skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=./venv/bin/python
W="${WORKERS:-2}"
NICE="nice -n 10"
G="grep --line-buffered -v Warning"
export CALCULUS_RUN="${CALCULUS_RUN:-stone_analysis}"
$PY -c 'from calculus.common.paths import ensure; print("results ->", ensure())'

step () { echo; echo "--- $1 ---"; shift; $NICE $PY -u -m "$@" 2>&1 | $G; }
step "triage: pick the measurable series" calculus.common.triage_series
step "patient gate: adults only"          calculus.common.patient_gate
step "extract the chosen series"          calculus.common.extract_series
step "organ masks (TotalSegmentator)"     calculus.common.run_anatomy
step "kidney mask QC"                     calculus.kidney.kidney_qc
step "PART 1: stones in the kidneys"      calculus.kidney.detect_stones --workers "$W"
step "kidney stone overlays"              calculus.kidney.render_overlays
step "PART 2: stones in the ureter"       calculus.ureter.detect_ureteric --workers "$W"
step "ureteric review sheets"             calculus.ureter.render_ureteric_overlays --rejected 3
step "report tables"                      calculus.report.make_report
step "full report per study"              calculus.report.make_report_full
step "join both compartments"             calculus.report.combine_stone_analysis
step "score against the reports"          calculus.evaluate.compare_reports
step "sizes vs reported sizes"            calculus.evaluate.compare_measurements
step "one comparison table"               calculus.evaluate.compare_all
step "kidney metrics"                     calculus.evaluate.kidney_metrics
step "studies needing a human"            calculus.evaluate.list_issues
echo; echo "=== DONE $(date) ==="
