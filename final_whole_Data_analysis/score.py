#!/usr/bin/env python
"""The pre-deployment verdict, computed from the cohort JSON.

Answers exactly three questions and refuses to imply more than the data carries:
  1. sensitivity   -- of studies whose report names a calculus, how many did we
                      call Abnormal
  2. false positive rate -- of studies whose report names none, how many did we
                      call Abnormal anyway
  3. agreement     -- where both we and the report give a size or a density, how
                      close are they

A "false positive" here means "we reported a calculus the radiologist did not".
That is NOT the same as "we were wrong": the report is one reader on one day,
and this cohort contains bladder-focused and pelvis-focused reads that never
mention the ureters. The number is an upper bound on our error rate, and it is
labelled as one everywhere it appears.
"""
import glob
import json
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:x\s*\d+(?:\.\d+)?\s*)*(mm|cm)\b", re.I)
HU_RE = re.compile(r"(\d{2,4})\s*(?:HU|hu)\b|(?:HU|attenuation)[:\s]*(\d{2,4})")


def report_max_mm(text):
    """Largest size the REPORT quotes, in mm. Requires an explicit unit."""
    best = None
    for m in re.finditer(r"((?:\d+(?:\.\d+)?\s*x\s*)*\d+(?:\.\d+)?)\s*(mm|cm)\b",
                         str(text), re.I):
        nums = [float(v) for v in re.findall(r"\d+(?:\.\d+)?", m.group(1))]
        if not nums:
            continue
        v = max(nums) * (10.0 if m.group(2).lower() == "cm" else 1.0)
        best = v if best is None else max(best, v)
    return best


def report_max_hu(text):
    vals = [int(g) for m in HU_RE.finditer(str(text)) for g in m.groups() if g]
    return max(vals) if vals else None


def main():
    cohorts = []
    for f in ("cohort.csv", "cohort_phase2.csv"):
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            c = pd.read_csv(p)
            c["phase"] = 1 if f == "cohort.csv" else 2
            cohorts.append(c)
    if not cohorts:
        print("no cohort file"); return 1
    c = pd.concat(cohorts, ignore_index=True)
    c["study_id"] = c.study_id.astype(str)

    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, "json", "*.json"))):
        sid = os.path.basename(f)[:-5]
        try:
            with open(f) as fh:
                d = json.load(fh)
        except Exception:
            continue
        fi = d.get("findings") or {}
        cn = fi.get("counts") or {}
        rows.append({"study_id": sid,
                     "predicted": d.get("study_prediction"),
                     "basis": fi.get("prediction_basis"),
                     "total": fi.get("total_calculi"),
                     "renal": cn.get("renal"), "ureteric": cn.get("ureteric"),
                     "bladder": cn.get("bladder"),
                     "our_mm": fi.get("largest_calculus_mm"),
                     "our_hu": fi.get("max_density_hu"),
                     "seconds": fi.get("seconds")})
    if not rows:
        print("no JSON results yet"); return 1
    r = pd.DataFrame(rows)
    m = c.merge(r, on="study_id", how="inner")
    m["rep_mm"] = m.calculus_line.apply(report_max_mm)
    m["rep_hu"] = m.calculus_line.apply(report_max_hu)

    out = []
    A = out.append
    A("=" * 74)
    A("PRE-DEPLOYMENT COHORT RESULT")
    A("=" * 74)
    A(f"cohort defined      {len(c)} studies (phase 1 balanced 518, "
      f"phase 2 negatives {len(c[c.phase == 2]) if (c.phase == 2).any() else 0})")
    A(f"answered so far     {len(m)}")
    A("")

    pos, neg = m[m.expected == "abnormal"], m[m.expected == "normal"]
    if len(pos):
        tp = int((pos.predicted == "Abnormal").sum())
        A(f"SENSITIVITY         {tp}/{len(pos)} = {100*tp/len(pos):.1f}%")
        A("                    (report names a calculus; we said Abnormal)")
        miss = pos[pos.predicted != "Abnormal"]
        if len(miss):
            A(f"  MISSED            {len(miss)} study/studies -- listed below")
    if len(neg):
        fp = int((neg.predicted == "Abnormal").sum())
        A("")
        A(f"FALSE POSITIVE RATE {fp}/{len(neg)} = {100*fp/len(neg):.1f}%   "
          "<- UPPER BOUND")
        A(f"SPECIFICITY         {len(neg)-fp}/{len(neg)} = "
          f"{100*(len(neg)-fp)/len(neg):.1f}%")
        A("                    A 'false positive' is a calculus we reported and")
        A("                    the radiologist did not. That is not the same as")
        A("                    being wrong: this cohort holds bladder- and")
        A("                    pelvis-focused reads that never mention the")
        A("                    ureters. Treat it as a ceiling on our error.")

    A("")
    A("BY COMPARTMENT, on studies we called Abnormal")
    ab = m[m.predicted == "Abnormal"]
    if len(ab):
        for k in ("renal", "ureteric", "bladder"):
            v = pd.to_numeric(ab[k], errors="coerce").fillna(0)
            A(f"  {k:<9} studies with >=1: {int((v > 0).sum()):>4}   "
              f"total calculi: {int(v.sum()):>5}")

    both = m[(m.expected == "abnormal") & m.rep_mm.notna() & m.our_mm.notna()]
    if len(both):
        d = both.our_mm - both.rep_mm
        A("")
        A(f"SIZE, largest per study, where both give one   n={len(both)}")
        A(f"  mean abs error    {d.abs().mean():.2f} mm")
        A(f"  median error      {d.median():+.2f} mm  "
          f"({'we read larger' if d.median() > 0 else 'we read smaller'})")
        A(f"  within 2 mm       {int((d.abs() <= 2).sum())}/{len(both)}")
    bh = m[(m.expected == "abnormal") & m.rep_hu.notna() & m.our_hu.notna()]
    if len(bh):
        ratio = bh.our_hu / bh.rep_hu
        A("")
        A(f"DENSITY, max per study, where both give one    n={len(bh)}")
        A(f"  median ratio      {ratio.median():.2f}x")
        A(f"  within 20%        {int(((ratio-1).abs() <= .2).sum())}/{len(bh)}")

    led = []
    for f in ("ledger.csv", "cohort_phase2_ledger.csv"):
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            led.append(pd.read_csv(p))
    if led:
        L = pd.concat(led, ignore_index=True)
        A("")
        A("OUTCOMES (from the ledger, including studies that produced no JSON)")
        for k, v in L.status.value_counts().items():
            A(f"  {k:<20} {v}")
        bad = L[L.status != "ok"]
        if len(bad):
            A("")
            A("  studies with no result -- these are NOT negatives, they are")
            A("  unmeasured, and excluding them silently would flatter the rates:")
            for t in bad.head(25).itertuples():
                A(f"    {t.study_id}  {t.status}  {str(t.detail)[:70]}")
            if len(bad) > 25:
                A(f"    ... and {len(bad)-25} more")

    if len(pos):
        miss = pos[pos.predicted != "Abnormal"]
        if len(miss):
            A("")
            A("MISSED POSITIVES -- the list that decides whether this deploys")
            for t in miss.itertuples():
                A(f"  {t.study_id}  we said {t.predicted}/{t.basis}")
                A(f"     report: {str(t.calculus_line)[:150]}")

    if len(neg):
        worst = neg[neg.predicted == "Abnormal"].copy()
        if len(worst):
            worst["tot"] = pd.to_numeric(worst.total, errors="coerce").fillna(0)
            worst = worst.sort_values("tot", ascending=False)
            A("")
            A("LARGEST FALSE POSITIVES -- where to look for a systematic cause")
            for t in worst.head(15).itertuples():
                A(f"  {t.study_id}  {int(t.tot)} calculi  "
                  f"largest {t.our_mm} mm  {t.our_hu} HU  "
                  f"(r{t.renal}/u{t.ureteric}/b{t.bladder})")

    txt = "\n".join(out)
    print(txt)
    with open(os.path.join(HERE, "RESULT.txt"), "w") as fh:
        fh.write(txt + "\n")
    m.to_csv(os.path.join(HERE, "per_study_result.csv"), index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
