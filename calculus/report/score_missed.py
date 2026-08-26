"""Score our detections against the audit's "what the radiologist missed" text.

WHY THIS SCORER IS DIFFERENT FROM score_run
-------------------------------------------
Everywhere else the report is ground truth, so a detection the report does not
mention counts AGAINST us. In this cohort the audit states the report was WRONG
and names the calculus that was missed. So here:

    detection matching an audit finding  ->  we caught what a radiologist missed
    detection matching nothing           ->  UNVERIFIED, not a false positive

That asymmetry is the whole point, and it bounds what this script can report:

  * RECALL against the audit list is a clean point estimate. The audit names the
    compartment and usually the side, so each finding is a hard target.

  * PRECISION is NOT computable here. The audit says what was missed, not what
    was absent. A detection outside the audit list may be a false positive or a
    second miss the auditor also walked past. We print those counts as
    "unverified" and refuse to fold them into a precision number.

SIDE INVERSION is reported separately, because it is a known open defect: the
landmark_uvj rule produced 4 side inversions on VUJ stones in the last cohort.
A target counted as "wrong_side" means we found a stone in the right compartment
but only on the opposite side -- the detector saw it and mislabelled it, which is
a different bug from not seeing it at all.

Usage:
    python -m calculus.report.score_missed
    python -m calculus.report.score_missed --worklist Missed_cases/worklist_live.csv
"""
import argparse
import os
import re

import pandas as pd

from calculus.common import paths

# --- parsing the audit's free text -----------------------------------------
# The audit column is a radiologist's phrase, e.g.
#   "right ureteric obstructing calculus"
#   "two non-obstructing left lower calyceal calculi"
#   "left distal ureteric calculus ; right mid-ureteric non-obstructive calculus"
# Clauses are separated by ";". Each clause is one finding.

RENAL_RE = re.compile(
    r"renal|kidney|calyc|caly[xc]|staghorn|nephrolith|urolith|concretion|"
    r"microlith|pyelo|interpolar|mid-?pole|upper pole|lower pole", re.I)
URETER_RE = re.compile(r"ureter|vuj|uvj|vesico|vesicoureteric", re.I)
BLADDER_RE = re.compile(r"bladder|vesical calc", re.I)
# PUJ sits at the renal end of the ureter. Reports use it for both, so a clause
# naming PUJ counts as a hit for either compartment rather than being forced
# into one -- scoring it strictly would penalise a correct detection.
PUJ_RE = re.compile(r"\bpuj\b|pelvi-?ureteric", re.I)

LEFT_RE = re.compile(r"\bleft\b|\blt\b", re.I)
RIGHT_RE = re.compile(r"\bright\b|\brt\b", re.I)
BILAT_RE = re.compile(r"bilateral|both kidney|b/l", re.I)
OBSTRUCT_RE = re.compile(r"obstruct", re.I)
NONOBSTRUCT_RE = re.compile(r"non-?obstruct", re.I)


def compartments(clause):
    """Which compartments this clause could refer to. May be more than one."""
    out = []
    if RENAL_RE.search(clause):
        out.append("renal")
    if URETER_RE.search(clause):
        out.append("ureteric")
    if BLADDER_RE.search(clause):
        out.append("bladder")
    if PUJ_RE.search(clause) and not out:
        out = ["renal", "ureteric"]     # ambiguous by convention, accept either
    return out or ["unclear"]


def sides(clause):
    if BILAT_RE.search(clause):
        return ["left", "right"]
    has_l, has_r = bool(LEFT_RE.search(clause)), bool(RIGHT_RE.search(clause))
    if has_l and has_r:
        return ["left", "right"]
    if has_l:
        return ["left"]
    if has_r:
        return ["right"]
    return ["any"]      # audit did not state a side; do not invent one


def targets_for(text):
    """Explode one audit cell into individual (compartment, side) targets."""
    out = []
    for clause in re.split(r";|\band\b(?=.*calc)", str(text)):
        clause = clause.strip()
        if not clause or clause.lower() == "nan":
            continue
        for comp in compartments(clause):
            for side in sides(clause):
                out.append({
                    "clause": clause,
                    "compartment": comp,
                    "side": side,
                    "obstructing": bool(OBSTRUCT_RE.search(clause)
                                        and not NONOBSTRUCT_RE.search(clause)),
                })
    return out


# --- matching ---------------------------------------------------------------
SRC_TO_COMP = {"kidney": "renal", "ureter": "ureteric", "ureteric": "ureteric",
               "bladder": "bladder"}


def score(worklist, stones, summary):
    rows, per_study = [], []
    det_by_study = dict(tuple(stones.groupby("study_id"))) if len(stones) else {}
    done = set(summary.study_id.astype(str)) if len(summary) else set()

    for rec in worklist.itertuples():
        sid = str(rec.study_id)
        if sid not in done:
            continue                    # not inferred yet, do not score as a miss
        det = det_by_study.get(sid, pd.DataFrame(columns=stones.columns))
        det = det.copy()
        det["comp"] = det.source.map(lambda s: SRC_TO_COMP.get(str(s), str(s)))

        tgts = targets_for(rec.calculus_missed)
        hit_n = 0
        for t in tgts:
            same = det[det.comp == t["compartment"]]
            if t["compartment"] == "unclear":
                same = det                      # any stone anywhere counts
            side_ok = same if t["side"] == "any" else \
                same[same.side.astype(str).str.lower() == t["side"]]

            if len(side_ok):
                verdict = "hit"
                best = side_ok.loc[side_ok.max_diameter_mm.idxmax()]
            elif len(same):
                verdict = "wrong_side"          # seen, mislabelled -- see docstring
                best = same.loc[same.max_diameter_mm.idxmax()]
            else:
                verdict = "miss"
                best = None
            hit_n += verdict == "hit"

            rows.append({
                "study_id": sid,
                "grade": getattr(rec, "grade", ""),
                "audit_clause": t["clause"],
                "compartment": t["compartment"],
                "side": t["side"],
                "obstructing": t["obstructing"],
                "verdict": verdict,
                "matched_mm": None if best is None else round(float(best.max_diameter_mm), 1),
                "matched_hu_mean": None if best is None else round(float(best.hu_mean), 0),
                "matched_side": None if best is None else best.side,
                "matched_location": None if best is None else best.location,
                "n_det_kidney": int((det.comp == "renal").sum()),
                "n_det_ureteric": int((det.comp == "ureteric").sum()),
            })

        per_study.append({
            "study_id": sid,
            "grade": getattr(rec, "grade", ""),
            "audit_text": rec.calculus_missed,
            "n_targets": len(tgts),
            "n_hit": hit_n,
            "all_found": hit_n == len(tgts) and len(tgts) > 0,
            "any_found": hit_n > 0,
            "n_detections": len(det),
        })

    return pd.DataFrame(rows), pd.DataFrame(per_study)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worklist", default=None,
                    help="audit worklist csv (default: Missed_cases/worklist_live.csv)")
    ap.add_argument("--out", default=None, help="output dir (default: <run>/csv)")
    a = ap.parse_args()

    run = paths.ensure()
    csvdir = a.out or os.path.join(run, "csv")
    os.makedirs(csvdir, exist_ok=True)

    wl = a.worklist or os.path.join(paths.ROOT, "Missed_cases", "worklist_live.csv")
    worklist = pd.read_csv(wl)

    sp = os.path.join(run, "csv", "overall_stones.csv")
    mp = os.path.join(run, "csv", "overall_summary.csv")
    if not os.path.exists(sp):
        raise SystemExit(f"no detections yet: {sp} missing "
                         "(run combine_stone_analysis first)")
    stones = pd.read_csv(sp)
    stones["study_id"] = stones.study_id.astype(str)
    summary = pd.read_csv(mp) if os.path.exists(mp) else pd.DataFrame(columns=["study_id"])
    summary["study_id"] = summary.study_id.astype(str)

    tg, st = score(worklist, stones, summary)
    if not len(tg):
        raise SystemExit("no scored studies yet -- inference has not produced "
                         "output for any worklist study")

    tg.to_csv(os.path.join(csvdir, "missed_targets.csv"), index=False)
    st.to_csv(os.path.join(csvdir, "missed_per_study.csv"), index=False)

    def pct(n, d):
        return f"{100.0 * n / d:5.1f}%  ({n}/{d})" if d else "    -   (0/0)"

    print(f"\nscored {len(st)} studies, {len(tg)} audit findings\n")
    print("RECALL AGAINST THE AUDIT  (stones a radiologist missed)")
    print(f"  per finding        {pct((tg.verdict == 'hit').sum(), len(tg))}")
    print(f"  per study, all     {pct(st.all_found.sum(), len(st))}")
    print(f"  per study, any     {pct(st.any_found.sum(), len(st))}")

    print("\n  by compartment")
    for comp, g in tg.groupby("compartment"):
        print(f"    {comp:10s}       {pct((g.verdict == 'hit').sum(), len(g))}")

    print("\n  by audit grade")
    for grade, g in tg.groupby("grade"):
        print(f"    {grade:16s} {pct((g.verdict == 'hit').sum(), len(g))}")

    obs = tg[tg.obstructing]
    if len(obs):
        print(f"\n  obstructing only   {pct((obs.verdict == 'hit').sum(), len(obs))}"
              "   <- highest clinical consequence")

    ws = int((tg.verdict == "wrong_side").sum())
    if ws:
        print(f"\nSIDE INVERSION: {ws} finding(s) detected in the right compartment "
              "but only on the\n  opposite side -- seen and mislabelled, not missed. "
              "Check the landmark_uvj rule.")
        for r in tg[tg.verdict == "wrong_side"].itertuples():
            print(f"    {r.study_id:12s} audit={r.side:5s} ours={r.matched_side:5s} "
                  f"{r.audit_clause[:48]}")

    print("\nUNVERIFIED DETECTIONS  (not in the audit list)")
    print("  These are NOT scored. The audit records what was missed, not what was")
    print("  absent, so a detection outside its list may be a true finding the")
    print("  auditor also passed over. No precision number is computable here.")
    print(f"  kidney   {int(tg.n_det_kidney.groupby(tg.study_id).first().sum())} detections "
          f"over {len(st)} studies")
    print(f"  ureteric {int(tg.n_det_ureteric.groupby(tg.study_id).first().sum())} detections")

    miss = st[~st.any_found]
    if len(miss):
        print(f"\nSTUDIES WHERE WE FOUND NOTHING THE AUDIT NAMED  ({len(miss)})")
        for r in miss.itertuples():
            print(f"  {r.study_id:12s} {str(r.grade):16s} {str(r.audit_text)[:60]}")

    print(f"\nwrote {csvdir}/missed_targets.csv  and  missed_per_study.csv")


if __name__ == "__main__":
    main()
