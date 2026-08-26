#!/bin/bash
# Progress and the two numbers this cohort exists to measure.
cd "$(dirname "$0")"
PY=/root/Gaurav/kindey_calculus_measurement/venv/bin/python
echo "=== progress"
tail -2 progress.log 2>/dev/null | sed 's/^/  /'
echo "  json written: $(ls json/*.json 2>/dev/null | wc -l) / 518"
echo "  in flight   : $(ps -eo args --no-headers | grep -c '[i]nfer_study')"
echo "  running     : $(pgrep -f '[r]unner.py' >/dev/null && echo yes || echo NO - finished or stopped)"
echo
[ -f ledger.csv ] && $PY - <<'PY'
import pandas as pd
d = pd.read_csv("ledger.csv")
ok = d[d.status == "ok"]
print(f"=== outcomes ({len(d)} attempted)")
for k, v in d.status.value_counts().items():
    print(f"  {k:<18} {v}")
if len(ok):
    print(f"\n=== the two numbers (on {len(ok)} completed)")
    pos = ok[ok.expected == "abnormal"]
    neg = ok[ok.expected == "normal"]
    if len(pos):
        tp = (pos.predicted == "Abnormal").sum()
        print(f"  sensitivity        {tp}/{len(pos)} = {100*tp/len(pos):.0f}%"
              "   (reported calculus, we called Abnormal)")
    if len(neg):
        fp = (neg.predicted == "Abnormal").sum()
        print(f"  false positive     {fp}/{len(neg)} = {100*fp/len(neg):.0f}%"
              "   (no calculus reported, we called Abnormal)")
        print(f"  specificity        {len(neg)-fp}/{len(neg)} = "
              f"{100*(len(neg)-fp)/len(neg):.0f}%")
    print(f"\n  median runtime     {ok.seconds.median():.0f} s")
PY
