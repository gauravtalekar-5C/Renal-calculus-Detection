#!/usr/bin/env bash
# Infer on the audit's missed cases -- studies where the radiologist missed a calculus.
#
# WHY THIS COHORT MATTERS
# Every other cohort is labelled by what the report SAYS. Here the audit says the
# report was WRONG and names the calculus that was missed. So a detection we make
# that the report does not contain is evidence of a TRUE positive here, where in
# every other cohort it counts against us. These are the studies that can tell us
# how many of our 24 kidney and 30 ureteric "false positives" are actually
# radiologist misses.
#
# Everything lives under Missed_cases/, nothing touches the other cohorts:
#   Missed_cases/zips     the downloads
#   Missed_cases/nifti    extracted volumes
#   Missed_cases/seg      organ masks
#   Results_missed        csv/, overlays/, reports/   <- the deliverables
#
# Needs the analysis environment; point PY at it, or install this package into it:
#   pip install -e /root/Gaurav/renal-calculus
#
# Usage: scripts/run_missed_cases.sh            everything
#        STAGE=prep scripts/run_missed_cases.sh triage + extract + masks only
set -uo pipefail
cd "$(dirname "$0")/.."
D=Missed_cases
OUT="${OUT:-Results_missed}"    # overlays + report CSVs land here
PY="${PY:-/root/Gaurav/kindey_calculus_measurement/venv/bin/python}"
W="${WORKERS:-2}"
NICE="nice -n 10"
G="grep --line-buffered -v Warning"

export CALCULUS_ZIPS="$D/zips"
export CALCULUS_NIFTI="$D/nifti"
export CALCULUS_SEG="$D/seg"
export CALCULUS_RUN="$OUT"
export CALCULUS_TS="${CALCULUS_TS:-/root/Gaurav/kindey_calculus_measurement/venv/bin/TotalSegmentator}"

echo "=============================================================="
echo " MISSED CASES     $(ls $D/zips/*.zip 2>/dev/null | wc -l) studies downloaded"
echo " results ->       $OUT"
echo " started          $(date)"
echo "=============================================================="
step () { echo; echo "--- $1 ---"; shift; $NICE $PY -u -m "$@" 2>&1 | $G; }

step "triage: pick the measurable series"  calculus.common.triage_series
step "extract the chosen series"           calculus.common.extract_series --include-thick
step "organ masks"                         calculus.common.run_anatomy
step "mask overlays (check them by eye)"   calculus.common.render_masks
[ "${STAGE:-all}" = "prep" ] && { echo; echo "prep only, stopping"; exit 0; }

step "kidney mask QC"                      calculus.kidney.kidney_qc
step "PART 1: stones in the kidneys"       calculus.kidney.detect_stones --workers "$W"
step "kidney stone overlays"               calculus.kidney.render_overlays
step "PART 2: stones in the ureter"        calculus.ureter.detect_ureteric --workers "$W"
step "ureteric review sheets"              calculus.ureter.render_ureteric_overlays --rejected 3
step "report tables"                       calculus.report.make_report
step "full report per study"               calculus.report.make_report_full
step "join both compartments"              calculus.report.combine_stone_analysis
echo; echo "=== DONE $(date) ==="
echo "   reports  -> $OUT/reports/"
echo "   overlays -> $OUT/overlays/"
echo "   masks    -> $OUT/mask_overlays/"
