"""ONE csv: what the radiologist wrote, beside what we found, per study.

The comparison already existed but spread across four files -- presence in
score_run's stdout, sizes in size_vs_report_study.csv, side and pole in
agreement_by_field.csv, HU nowhere. Answering "does this study agree" meant
opening four things and joining them by eye.

    csv/report_vs_model.csv     one row per study, every field side by side
    reports/stone_check.csv     the same comparison cut down to 11 columns for
                                checking by eye, worst rows first, ending in the
                                path of the overlay that settles the question

COLUMN PAIRS, always report_* then model_*, so a spreadsheet reads left to right:

    presence     did the report state a calculus / did we detect one
    count        how many we found (kidney and ureteric separately). The report
                 side is usually absent: "few calculi" and "multiple calculi"
                 are not counts, so report_n_stated is blank far more often
                 than not. That is a property of reports, not a gap in parsing.
    side         left / right / both
    compartment  renal / ureteric / bladder as stated, against what we searched
    size         the largest stone, ours as the 3D caliper (the like-for-like
                 comparison -- see compare_measurements for why)
    HU           report HU against our hu_max AND hu_mean. hu_max reads high
                 because a radiologist places an ROI and reads a mean, so
                 hu_mean is the comparable column and both are given.
    verdict      agree / MISS / FALSE POSITIVE / size disagreement

WHAT A DISAGREEMENT DOES NOT MEAN
---------------------------------
A report records what was clinically worth saying, not everything present.
Small calyceal stones are routinely omitted, so some of our "false positives"
are real stones nobody mentioned. Rows are labelled, not judged.

Usage:
    CALCULUS_RUN=stone_analysis ./venv/bin/python utils/compare_all.py
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# package root -> project root: calculus/<sub>/x.py is two levels down
ROOT = os.path.dirname(os.path.dirname(HERE))
from calculus.common.paths import CSV, RUN                                   # noqa: E402
from calculus.evaluate.compare_measurements import (BLADDER_RE, CALC_RE, NOT_CALC_RE,  # noqa: E402
                                 URETER_RE, calculus_sizes)
from calculus.evaluate.compare_reports import SIDE_RE, clauses, negated         # noqa: E402

XLSX = os.path.join(ROOT, "calculus_with_report.xlsx")
SHEET = "jun-jul-2026"
SIZE_FLAG_MM = 5.0

HU_A = re.compile(r"(\d{2,4}(?:\.\d+)?)\s*(?:HU|hu)\b")
HU_B = re.compile(r"(?:HU|hu)\s*[:~=]?\s*(\d{2,4}(?:\.\d+)?)")
POLE = {"upper_pole": "upper", "interpolar": "mid", "lower_pole": "lower"}
POLE_RE = {"upper": re.compile(r"upper (pole|calyx|calyce)", re.I),
           "mid": re.compile(r"(mid|middle|inter ?polar)", re.I),
           "lower": re.compile(r"lower (pole|calyx|calyce)", re.I)}


def report_hu(text):
    """HU values stated for a calculus, in stone-plausible range."""
    out = set()
    for part in clauses(text):
        if not CALC_RE.search(part) or negated(part) or NOT_CALC_RE.search(part):
            continue
        for rx in (HU_A, HU_B):
            for m in rx.finditer(part):
                v = float(m.group(1))
                if 20 <= v <= 3000:
                    out.add(v)
    return sorted(out)


def sides_in(text):
    return "+".join(sorted(k for k, rx in SIDE_RE.items() if rx.search(text)))


def compartments_in(text):
    out = set()
    for part in clauses(text):
        if not CALC_RE.search(part) or negated(part) or NOT_CALC_RE.search(part):
            continue
        out.add("bladder" if BLADDER_RE.search(part) else
                "ureteric" if URETER_RE.search(part) else "renal")
    return "+".join(sorted(out))


def poles_in(text):
    out = set()
    for part in clauses(text):
        if not CALC_RE.search(part) or negated(part):
            continue
        for k, rx in POLE_RE.items():
            if rx.search(part):
                out.add(k)
    return "+".join(sorted(out))


def main():
    ks = pd.read_csv(os.path.join(CSV, "baseline_stones.csv"))
    ks["study_id"] = ks.study_id.astype(str)
    summ = pd.read_csv(os.path.join(CSV, "baseline_summary.csv"))
    summ["study_id"] = summ.study_id.astype(str)

    up = os.path.join(CSV, "ureter_candidates.csv")
    us = None
    if os.path.exists(up):
        d = pd.read_csv(up)
        d["study_id"] = d.study_id.astype(str)
        d = d[d.is_stone.astype(bool)]
        us = d[d.report_this.astype(bool)] if "report_this" in d else d
    ur_done = set(os.path.basename(f).split("_ureter")[0] for f in
                  glob.glob(os.path.join(CSV, "per_study",
                                         "*_ureter_summary.csv")))

    rep = pd.read_excel(XLSX, sheet_name=SHEET)
    rep["study_id"] = rep.study_id.astype(str)
    rep = rep.set_index("study_id")

    rows = []
    for sid in sorted(summ.study_id):
        line = str(rep.calculus_line.get(sid, "") or "")
        flag = bool(rep.calculus_flag.get(sid, False))
        rs = calculus_sizes(line)
        r_sizes = sorted(rs["renal"] + rs["ureteric"], reverse=True)
        r_hu = report_hu(line)

        k = ks[ks.study_id == sid]
        u = us[us.study_id == sid] if us is not None else pd.DataFrame()
        m_sizes = sorted(list(k.max_diameter_mm.dropna())
                         + (list(u.max_diameter_mm.dropna()) if len(u) else []),
                         reverse=True)
        m_hu_max = max(list(k.hu_max) + (list(u.hu_max) if len(u) else []),
                       default=np.nan)
        m_hu_mean = (pd.concat([k.hu_mean, u.hu_mean if len(u) else
                                pd.Series(dtype=float)]).max()
                     if (len(k) or len(u)) else np.nan)
        m_sides = "+".join(sorted(set(list(k.side.dropna())
                                      + (list(u.side.dropna()) if len(u) else []))))
        m_comp = "+".join(sorted(
            ({"renal"} if len(k) else set())
            | ({"ureteric"} if len(u) else set())))
        m_poles = "+".join(sorted({POLE.get(x, x) for x in k.location.dropna()
                                   if x} )) if len(k) else ""

        det = bool(len(k) or len(u))
        r_max = max(r_sizes) if r_sizes else np.nan
        m_max = max(m_sizes) if m_sizes else np.nan
        sdiff = (r_max - m_max) if (r_sizes and m_sizes) else np.nan

        verdict = ("agree" if flag == det else
                   "MISS" if flag and not det else "FALSE POSITIVE")
        if verdict == "agree" and pd.notna(sdiff) and abs(sdiff) >= SIZE_FLAG_MM:
            verdict = "agree, size disagrees"

        rows.append({
            "study_id": sid,
            "report_has_stone": flag,
            "model_has_stone": det,
            "verdict": verdict,
            "report_n_stated": "",           # reports say "few"/"multiple"
            "model_n_kidney": len(k),
            "model_n_ureteric": len(u),
            "ureteric_search_done": sid in ur_done,
            "report_sides": sides_in(line),
            "model_sides": m_sides,
            "report_compartments": compartments_in(line),
            "model_compartments": m_comp,
            "report_poles": poles_in(line),
            "model_poles": m_poles,
            "report_sizes_mm": ";".join(f"{v:g}" for v in r_sizes),
            "report_max_mm": round(r_max, 1) if r_sizes else "",
            "model_sizes_mm": ";".join(f"{v:.1f}" for v in m_sizes),
            "model_max_mm": round(m_max, 1) if m_sizes else "",
            "size_diff_mm": round(sdiff, 2) if pd.notna(sdiff) else "",
            "size_flag_5mm": ("yes" if pd.notna(sdiff)
                              and abs(sdiff) >= SIZE_FLAG_MM else "no"),
            "report_hu": ";".join(f"{v:g}" for v in r_hu),
            "report_hu_max": max(r_hu) if r_hu else "",
            "model_hu_max": int(m_hu_max) if pd.notna(m_hu_max) else "",
            "model_hu_mean": int(m_hu_mean) if pd.notna(m_hu_mean) else "",
            "report_bladder_mm": ";".join(f"{v:g}" for v in rs["bladder"]),
            "report_line": line[:400],
        })

    d = pd.DataFrame(rows)
    dest = os.path.join(CSV, "report_vs_model.csv")
    d.to_csv(dest, index=False)

    # ---- the short version, for checking a study by eye -------------------
    # 26 columns is right for analysis and wrong for "is the stone there".
    # This one is 11 columns and ends with the path to the overlay, so a row
    # can be verified against the image without hunting for the file.
    check = pd.DataFrame({
        "study_id": d.study_id,
        "report_says_stone": np.where(d.report_has_stone, "YES", "no"),
        "we_found_stone": np.where(d.model_has_stone, "YES", "no"),
        "match": d.verdict,
        "report_side": d.report_sides,
        "our_side": d.model_sides,
        "report_largest_mm": d.report_max_mm,
        "our_largest_mm": d.model_max_mm,
        "n_kidney": d.model_n_kidney,
        "n_ureteric": np.where(d.ureteric_search_done, d.model_n_ureteric,
                               "not searched yet"),
        "report_says": d.report_line.str.slice(0, 200),
        "check_this_image": [
            os.path.join(os.path.basename(RUN), "overlays", s,
                         "_coronal_mip.png") for s in d.study_id],
    })
    # worst first: misses, then false positives, then size disagreements, so the
    # rows that need a human are at the top of the file rather than in study-id
    # order somewhere in the middle
    order = {"MISS": 0, "FALSE POSITIVE": 1, "agree, size disagrees": 2,
             "agree": 3}
    check = check.assign(_o=check.match.map(order)).sort_values(
        ["_o", "study_id"]).drop(columns="_o")
    cdest = os.path.join(RUN, "reports", "stone_check.csv")
    os.makedirs(os.path.dirname(cdest), exist_ok=True)
    check.to_csv(cdest, index=False)
    print(f"wrote {cdest}   (the short check list, worst rows first)")
    print(f"wrote {dest}   ({len(d)} studies, {len(d.columns)} columns)\n")
    print(d.verdict.value_counts().to_string())
    n = int((d.size_flag_5mm == "yes").sum())
    print(f"\nsize disagreement >= {SIZE_FLAG_MM:g} mm: {n} studies")
    print(f"ureteric search not yet run for {int((~d.ureteric_search_done).sum())} "
          f"studies -- their ureteric columns are blank, not negative")


if __name__ == "__main__":
    main()
