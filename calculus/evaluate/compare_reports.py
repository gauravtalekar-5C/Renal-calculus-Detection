"""Compare the baseline detector against what the radiologist actually wrote.

The cohort spreadsheet already carries `calculus_line` -- the sentences the
reporting radiologist wrote about calculi. That is free, zero-annotation ground
truth for the things a report states plainly:

    was there a stone at all?      calculus_flag
    which side?                    "right kidney", "left renal"
    which compartment?             renal / ureteric / VUJ / PUJ / pelvis
    roughly how big?               "4 mm", "1-2mm", "1.2 x 0.8 cm"
    which pole?                    "lower pole", "upper calyx", "interpolar"

This is NOT a substitute for voxel annotation -- reports omit plenty, sizes are
rounded, and "few calculi" is not a count. It is a cheap first sanity check that
answers: is the baseline in the right ballpark, and where does it fail?

Outputs csv/baseline_vs_report.csv, one row per study, model result beside
report text.

Usage:
    ./venv/bin/python compare_reports.py
"""
import os
import sys
import re

import numpy as np
import pandas as pd

# this file lives in utils/, so the project root is one level up. All data
# directories hang off ROOT, never off the script's own folder.
HERE = os.path.dirname(os.path.abspath(__file__))
# package root -> project root: calculus/<sub>/x.py is two levels down
ROOT = os.path.dirname(os.path.dirname(HERE))
from calculus.common.paths import CSV                 # noqa: E402  results dir is per-run

SIDE_RE = {
    "right": re.compile(r"\bright\b|\brt\b|\bb/l\b|bilateral", re.I),
    "left": re.compile(r"\bleft\b|\blt\b|\bb/l\b|bilateral", re.I),
}
POLE_RE = {
    "upper_pole": re.compile(r"upper (pole|calyx|calyce)", re.I),
    "interpolar": re.compile(r"(mid|middle|inter ?polar) ?(pole|calyx|calyce)?", re.I),
    "lower_pole": re.compile(r"lower (pole|calyx|calyce)", re.I),
}
COMPARTMENT_RE = {
    "kidney": re.compile(r"renal|kidney|calyx|calyce|pole", re.I),
    "ureter": re.compile(r"ureter", re.I),
    "vuj": re.compile(r"\bvuj\b|vesico ?ureteric", re.I),
    "puj": re.compile(r"\bpuj\b|pelvi ?ureteric", re.I),
    "pelvis": re.compile(r"renal pelvis|pelvic calculus", re.I),
    "bladder": re.compile(r"bladder|vesical calcul", re.I),
}
# "4 mm", "1-2 mm", "1.2 cm", "12x8 mm"
SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:[-x×]\s*(\d+(?:\.\d+)?))?\s*(mm|cm)", re.I)
# Sizes that are NOT stone sizes: "3 to 4 cm above the vesico-ureteric junction"
# is a distance, and my first pass read it as a 40 mm stone.
SIZE_EXCLUDE_RE = re.compile(r"(above|below|from|proximal to|distal to)\s*the", re.I)

# Part 1 only looks inside the kidneys, so scoring against every calculus in the
# report is wrong: a report describing only a distal ureteric stone SHOULD give
# zero kidney stones. These decide whether a report mentions an INTRARENAL
# calculus at all.
RENAL_RE = re.compile(
    r"(renal|kidney|calyx|calyce|calyceal|pole|nephrolith|staghorn|"
    r"renal pelvis|pelvicalyceal)", re.I)
NON_RENAL_ONLY_RE = re.compile(
    r"(ureter|vuj|vesico|bladder|gall\s?bladder|cholelith)", re.I)


# A calculus the report says is GONE must not be scored as present. 8610030
# reads "The previously described 5 mm calculus in the interpolar calyx of the
# right kidney is not visualized on the present study" -- the stone has passed.
# We found nothing, which is CORRECT, and were marked as a false negative for
# it. The spreadsheet's calculus_type has the same blind spot (it says
# "ureteric, renal"), which is consistent with both being derived from the same
# prose.
NEGATION_RE = re.compile(
    r"(no longer (?:seen|visualized|identified|present)|"
    r"not (?:visualized|seen|identified|appreciated|demonstrated)|"
    r"has (?:passed|resolved)|now resolved|"
    r"resolution of the|interval passage|"
    r"previously (?:described|seen|noted).{0,60}?\bnot\b|"
    r"no (?:residual|evidence of a?)\s*(?:renal\s*)?calcul)", re.I)


# Splitting a report into clauses on "." also splits DECIMAL NUMBERS:
# "calculus measuring 30.2 mm" becomes "...measuring 30" + "2 mm in the...",
# so the size parser read 30.2 mm as 2 mm. It made the renal-only size error
# look WORSE (mean abs 11.8 mm) than the contaminated version it replaced, and
# produced impossible values like "report 0.0 mm".
# Split on "|" or on a period that is NOT between two digits.
CLAUSE_SPLIT_RE = re.compile(r"\||(?<!\d)\.|\.(?!\d)")


def clauses(text):
    """Report text split into clauses, without breaking decimal numbers."""
    return [p for p in CLAUSE_SPLIT_RE.split(text or "") if p and p.strip()]


def negated(part):
    """True if this clause says the calculus is absent or has passed."""
    return bool(NEGATION_RE.search(part or ""))


def mentions_renal_calculus(text):
    """True if the report describes a calculus inside the kidney itself."""
    if not text:
        return False
    hit = False
    for part in clauses(text):
        if not RENAL_RE.search(part):
            continue
        # "left ureteric calculus at the renal pelvis level" style phrasing:
        # require the renal word to not be swamped by a ureteric/VUJ context
        if NON_RENAL_ONLY_RE.search(part) and not re.search(
                r"(calyx|calyce|calyceal|pole|nephrolith|staghorn|renal pelvis)",
                part, re.I):
            continue
        if negated(part):
            continue                    # the stone is reported as gone
        hit = True
    return hit


def renal_sizes_mm(text):
    """Sizes the report attaches to an INTRARENAL calculus, only.

    Scoring our kidney measurement against "the largest size anywhere in the
    report" was meaningless: 8506983's biggest number is a 13 mm GALLBLADDER
    stone, and several studies' largest is ureteric. That inflated mean absolute
    error to 6.3 mm while the median was -1.5 mm -- the spread was entirely
    contamination, not measurement error.
    """
    out = []
    for part in clauses(text):
        if not RENAL_RE.search(part) or negated(part):
            continue
        if NON_RENAL_ONLY_RE.search(part) and not re.search(
                r"(calyx|calyce|calyceal|pole|nephrolith|staghorn|renal pelvis)",
                part, re.I):
            continue
        out.extend(report_sizes_mm(part))
    return out


def report_sizes_mm(text):
    out = []
    for m in SIZE_RE.finditer(text or ""):
        tail = (text or "")[m.end():m.end() + 24]
        if SIZE_EXCLUDE_RE.match(tail.strip()):
            continue
        for g in (m.group(1), m.group(2)):
            if g:
                v = float(g)
                out.append(v * 10 if m.group(3).lower() == "cm" else v)
    return out


def flags(text, table):
    return {k: bool(rx.search(text or "")) for k, rx in table.items()}


def main():
    stones_p = os.path.join(CSV, "baseline_stones.csv")
    summ_p = os.path.join(CSV, "baseline_summary.csv")
    master_p = os.path.join(CSV, "study_master.csv")
    stones = pd.read_csv(stones_p) if os.path.exists(stones_p) else pd.DataFrame()
    summ = pd.read_csv(summ_p)
    if not os.path.exists(master_p):
        sys.exit(f"missing {master_p}\n"
                 f"it holds the report text we score against - build it with:\n"
                 f"  ./venv/bin/python utils/summarize.py")
    master = pd.read_csv(master_p)
    for d in (stones, summ, master):
        if len(d):
            d["study_id"] = d["study_id"].astype(str)

    def as_int(v):
        """A study that errored has NaN counts; int(nan) raises."""
        return 0 if v is None or (isinstance(v, float) and np.isnan(v)) else int(v)

    rows = []
    for r in summ.itertuples():
        sid = str(r.study_id)
        m = master[master.study_id == sid]
        line = str(m.calculus_line.iloc[0]) if len(m) and pd.notna(
            m.calculus_line.iloc[0]) else ""
        flag = bool(m.calculus_flag.iloc[0]) if len(m) else np.nan
        # calculus_type is the spreadsheet's own location label ("renal",
        # "ureteric, renal", "VUJ, ureteric"). It agrees with our regex on
        # 35/37 studies, and where they differ the spreadsheet is usually
        # right -- so it is the primary truth and the regex is the check.
        ctype = str(m.calculus_type.iloc[0]) if len(m) and pd.notna(
            m.calculus_type.iloc[0]) else ""
        # ...but it inherits the same negation blind spot, because it was also
        # derived from this prose. 8610030 is labelled "ureteric, renal" while
        # the report says the calculus "is not visualized on the present
        # study". So a negated report overrides the label.
        xls_renal = ("renal" in ctype.lower() or "pelvis" in ctype.lower())
        if xls_renal and line and all(
                negated(p) or not RENAL_RE.search(p)
                for p in clauses(line) if RENAL_RE.search(p)):
            xls_renal = False
        s = stones[stones.study_id == sid] if len(stones) else pd.DataFrame()

        rep_side = flags(line, SIDE_RE)
        rep_comp = flags(line, COMPARTMENT_RE)
        rep_pole = flags(line, POLE_RE)
        rsizes = report_sizes_mm(line)
        rsizes_renal = renal_sizes_mm(line)

        model_sides = set(s.side.dropna()) if len(s) else set()
        model_comps = set(s.compartment.dropna()) if len(s) else set()
        model_poles = set(s.location.dropna()) - {""} if len(s) else set()

        rows.append({
            "study_id": sid,
            "report_says_stone": flag,
            # primary truth: spreadsheet label, negation-corrected
            "report_renal_calculus": xls_renal,
            "calculus_type": ctype,
            # the text regex, kept as an independent second opinion so a
            # disagreement is visible instead of averaged away
            "regex_renal_calculus": mentions_renal_calculus(line),
            "model_n_stones": as_int(getattr(r, "n_stones", 0)),
            "model_largest_mm": getattr(r, "largest_mm", np.nan),
            "report_sizes_mm": ";".join(f"{v:g}" for v in sorted(rsizes)) or "",
            "report_max_mm": max(rsizes) if rsizes else np.nan,
            # renal-only: the ONLY sizes our kidney measurement is comparable
            # against. report_max_mm is kept for context but must not be scored.
            "report_renal_sizes_mm": ";".join(f"{v:g}"
                                              for v in sorted(rsizes_renal)) or "",
            "report_renal_max_mm": max(rsizes_renal) if rsizes_renal else np.nan,
            "detected_any": as_int(getattr(r, "n_stones", 0)) > 0,
            "agree_presence": (as_int(getattr(r, "n_stones", 0)) > 0)
                              == xls_renal,
            # NOTE str(nan or "") is "nan", not "" -- that silently marked
            # every study unanalysable. Test for NaN explicitly.
            "not_analysable": ("" if pd.isna(getattr(r, "error", None))
                               else str(getattr(r, "error", ""))),
            "report_side": "+".join(k for k, v in rep_side.items() if v),
            "model_side": "+".join(sorted(model_sides)),
            "report_compartment": "+".join(k for k, v in rep_comp.items() if v),
            "model_compartment": "+".join(sorted(model_comps)),
            "report_pole": "+".join(k for k, v in rep_pole.items() if v),
            "model_pole": "+".join(sorted(model_poles)),
            "n_candidates_raw": getattr(r, "n_candidates_raw", np.nan),
            "calculus_line": line,
        })

    df = pd.DataFrame(rows)
    out = os.path.join(CSV, "baseline_vs_report.csv")
    df.to_csv(out, index=False)

    n = len(df)
    print(f"{n} studies\n")
    an = df[df.not_analysable.fillna("").eq("")]
    print(f"analysable: {len(an)}/{n} "
          f"({n - len(an)} rejected as enhanced/excretory)\n")
    print("PRESENCE of an INTRARENAL calculus (Part 1 only looks in kidneys,\n"
          "so a report describing only a ureteric/VUJ stone should give zero)")
    print(pd.crosstab(an.report_renal_calculus, an.detected_any,
                      rownames=["report renal"],
                      colnames=["model found"]).to_string())
    tp = int(((an.report_renal_calculus) & (an.detected_any)).sum())
    fn = int(((an.report_renal_calculus) & (~an.detected_any)).sum())
    tn = int(((~an.report_renal_calculus) & (~an.detected_any)).sum())
    fp = int(((~an.report_renal_calculus) & (an.detected_any)).sum())
    print(f"\n  sensitivity {tp}/{tp+fn}" +
          (f" ({tp/(tp+fn)*100:.0f}%)" if tp + fn else "") +
          f"   specificity {tn}/{tn+fp}" +
          (f" ({tn/(tn+fp)*100:.0f}%)" if tn + fp else ""))

    pos = an[an.report_renal_calculus == True]  # noqa: E712
    if len(pos):
        print(f"\nSIDE (on {len(pos)} report-positive studies)")
        both = pos[(pos.report_side != "") & (pos.model_side != "")]
        hit = sum(any(x in r.report_side for x in r.model_side.split("+"))
                  for r in both.itertuples())
        print(f"  side overlaps report: {hit}/{len(both)}")
        # renal-only sizes. Scoring against report_max_mm compared our kidney
        # measurement to whatever was biggest anywhere in the report, including
        # a gallbladder stone in 8506983.
        sized = pos.dropna(subset=["report_renal_max_mm"])
        sized = sized[sized.model_largest_mm > 0]
        if len(sized):
            err = (sized.model_largest_mm - sized.report_renal_max_mm)
            print(f"\nSIZE (largest RENAL stone, {len(sized)} studies where the "
                  f"report gives a renal size)\n"
                  f"  median error {err.median():+.1f} mm, "
                  f"mean abs {err.abs().mean():.1f} mm, "
                  f"within 2 mm: {(err.abs() <= 2).sum()}/{len(sized)}")
            worst = sized.reindex(err.abs().sort_values(ascending=False).index)
            print("  largest disagreements:")
            for r in worst.head(3).itertuples():
                print(f"    {r.study_id}: model {r.model_largest_mm:.1f} mm vs "
                      f"report {r.report_renal_max_mm:.1f} mm")
    print(f"\nwrote {out}")
    print("\nper study:")
    cols = ["study_id", "report_renal_calculus", "model_n_stones", "model_largest_mm",
            "report_max_mm", "model_side", "report_side", "n_candidates_raw"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
