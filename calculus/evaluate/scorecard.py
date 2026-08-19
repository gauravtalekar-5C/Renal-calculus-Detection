"""One file with every headline number, so results are readable off disk.

score_run.py prints sensitivity and specificity to the terminal and saves
nothing; the side / pole / compartment agreement was not computed by any script
at all. That means the results existed only in scrollback -- fine while someone
is watching, useless the next day.

Writes:
    SCORECARD.md                    everything, in one readable file
    csv/agreement_by_field.csv      side / pole / compartment, per study

Presence metrics are taken by RUNNING score_run.py and capturing its output
verbatim, not by recomputing them here. Two implementations of "sensitivity"
in one pipeline is how the QC thresholds ended up with two definitions of a bad
mask, one of them stale.

Usage:
    CALCULUS_RUN=run_v6 ./venv/bin/python utils/scorecard.py
"""
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# package root -> project root: calculus/<sub>/x.py is two levels down
ROOT = os.path.dirname(os.path.dirname(HERE))
from calculus.common.paths import CSV, RUN            # noqa: E402

PY = os.path.join(ROOT, "venv", "bin", "python")


def agree_set(a, b):
    """side/pole/compartment are '+'-joined sets, so compare them as sets.

    'exact'   the same set
    'partial' they share at least one member -- e.g. the report says
              'renal+ureteric' and we say 'renal'. Counting that as a failure
              overstates disagreement, and counting it as a pass hides a real
              gap, so it gets its own column.
    'none'    no member in common
    """
    A = {x for x in str(a).split("+") if x not in ("", "nan")}
    B = {x for x in str(b).split("+") if x not in ("", "nan")}
    if not A or not B:
        return None
    return "exact" if A == B else ("partial" if A & B else "none")


def size_stats(df, col):
    v = pd.to_numeric(df[col], errors="coerce").dropna()
    if not len(v):
        return None
    a = v.abs()
    return {"n": len(v), "bias": v.mean(), "median_bias": v.median(),
            "mean_abs": a.mean(), "median_abs": a.median(),
            "w1": 100 * (a <= 1).mean(), "w2": 100 * (a <= 2).mean(),
            "w3": 100 * (a <= 3).mean(), "w5": 100 * (a <= 5).mean()}


def main():
    out = [f"# Scorecard — {os.path.basename(RUN)}",
           "",
           f"Generated {pd.Timestamp.now():%d %b %Y %H:%M}. "
           f"All comparisons are against radiologist report text.", ""]

    # ---- 1. presence: score_run's own output, verbatim --------------------
    out += ["## 1 · Presence — sensitivity and specificity", ""]
    r = subprocess.run([PY, os.path.join(HERE, "score_run.py")],
                       capture_output=True, text=True,
                       env={**os.environ, "CALCULUS_RUN": RUN})
    txt = (r.stdout or "") + (r.stderr or "")
    out += ["```", txt.strip() or "score_run.py produced no output", "```", ""]

    # ---- 2. side / pole / compartment ------------------------------------
    p = os.path.join(CSV, "baseline_vs_report.csv")
    out += ["## 2 · Side, pole and compartment", ""]
    if os.path.exists(p):
        d = pd.read_csv(p)
        rows = []
        out += ["| Field | n | Exact | Partial | Disagree |",
                "|---|---|---|---|---|"]
        for field, rc, mc in (("side", "report_side", "model_side"),
                              ("pole", "report_pole", "model_pole"),
                              ("compartment", "report_compartment",
                               "model_compartment")):
            v = [agree_set(a, b) for a, b in zip(d[rc], d[mc])]
            keep = [x for x in v if x]
            d[f"{field}_agreement"] = v
            n = len(keep)
            if not n:
                continue
            e, q, z = keep.count("exact"), keep.count("partial"), keep.count("none")
            out.append(f"| {field} | {n} | {100*e/n:.0f}% | {100*q/n:.0f}% | "
                       f"{100*z/n:.0f}% |")
            rows.append({"field": field, "n": n, "exact_pct": round(100*e/n),
                         "partial_pct": round(100*q/n),
                         "disagree_pct": round(100*z/n)})
        cols = ["study_id", "report_side", "model_side", "side_agreement",
                "report_pole", "model_pole", "pole_agreement",
                "report_compartment", "model_compartment",
                "compartment_agreement", "calculus_line"]
        d[[c for c in cols if c in d.columns]].to_csv(
            os.path.join(CSV, "agreement_by_field.csv"), index=False)
        out += ["", "Per-study detail in `csv/agreement_by_field.csv`.", "",
                "*partial* means the sets overlap without matching — a report "
                "of 'renal+ureteric' against our 'renal'. It is neither a hit "
                "nor a clean miss, so it is reported separately rather than "
                "folded into either.", ""]
    else:
        out += [f"`{p}` not found — run `compare_reports.py`.", ""]

    # ---- 3. sizes ---------------------------------------------------------
    out += ["## 3 · Stone measurement vs reported size", ""]
    sp = os.path.join(CSV, "size_vs_report_study.csv")
    mp = os.path.join(CSV, "size_vs_report_matched.csv")
    if os.path.exists(sp):
        s, m = pd.read_csv(sp), pd.read_csv(mp)
        out += ["| Comparison | n | Bias | Median abs | ≤1 mm | ≤2 mm | ≤3 mm |",
                "|---|---|---|---|---|---|---|"]
        for label, df, col in (
                ("Report max vs our 3D caliper", s, "diff_mm_vs_caliper"),
                ("Report max vs our in-plane max", s,
                 "diff_mm_report_minus_model"),
                ("Per-stone, nearest-size match", m, "diff_mm")):
            st = size_stats(df, col)
            if st:
                out.append(f"| {label} | {st['n']} | {st['bias']:+.2f} mm | "
                           f"{st['median_abs']:.2f} mm | {st['w1']:.0f}% | "
                           f"{st['w2']:.0f}% | {st['w3']:.0f}% |")
        flag = [c for c in s.columns if c.startswith("diff_ge_")]
        if flag:
            n = int((s[flag[0]] == "yes").sum())
            out += ["", f"{n} study(ies) differ by ≥5 mm — listed in "
                        f"`csv/size_vs_report_study.csv`, worth opening the "
                        f"overlay for.", ""]
    else:
        out += [f"`{sp}` not found — run `compare_measurements.py`.", ""]

    # ---- 4. what is NOT compared ------------------------------------------
    out += ["## 4 · Not compared against reports", "",
            "- **Ureteric distance from the UVJ.** The landmark has never been "
            "checked against a radiologist's click, so a disagreement would not "
            "say which side is wrong.",
            "- **Hydronephrosis, perinephric fat stranding, stent.** Not "
            "implemented; emitted as `-`.",
            "- **Bladder calculi.** Out of scope; reported sizes are kept in a "
            "separate column and excluded from the error statistics.", "",
            "A report is an imperfect reference: it records what was clinically "
            "worth saying, not every stone present. Small calyceal stones are "
            "routinely omitted, which flatters sensitivity and penalises "
            "specificity.", ""]

    dest = os.path.join(RUN, "SCORECARD.md")
    open(dest, "w").write("\n".join(out))
    print(f"wrote {dest}")
    print(f"wrote {os.path.join(CSV, 'agreement_by_field.csv')}")


if __name__ == "__main__":
    main()
