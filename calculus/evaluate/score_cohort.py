"""Score a run against the radiologists' reports, automatically. Read-only.

WHY THIS EXISTS
---------------
Two serious bugs shipped and were caught only because a human read a CSV column
by eye:

  * a comment inserted into the ureteric rejection chain silently reattached
    `elif hu_max < HU_FLOOR` to the wrong `if`, disabling the 300 HU density
    floor. Five fabricated calculi at 156-293 HU reached a report.
  * per-study reports were written only when a study HAD findings, so a study
    whose findings were all rejected kept its previous file. One report showed
    five stale calculi dated an hour earlier.

Neither raised an exception. Neither failed a unit test. Both produced output
that looked entirely plausible. Eyeballing 18 cases does not scale, and in
production it is how a bad release ships. So: one command, one agreement report,
run after every change.

WHAT MAKES MATCHING HARD, AND WHAT WENT WRONG BEFORE
The first attempt at this paired each report finding with the LARGEST detection
on that side. That made the size and location columns describe whichever object
happened to be biggest rather than the one the radiologist meant, and the output
was unusable -- worse than nothing, because it looked authoritative.

This version instead:
  1. parses each report clause into a structured target (side, compartment,
     zone, size, HU) -- and records what it could NOT parse, so unparsed text is
     visible rather than silently dropped;
  2. builds a cost matrix over (target, detection) pairs, with side and
     compartment as HARD constraints and size/density as the cost;
  3. solves a one-to-one assignment (Hungarian), so no detection can be claimed
     by two findings and no finding can quietly absorb the biggest object.

VALIDATED AGAINST HAND PAIRING. 21 report findings were paired to detections by
hand while reading the reports one at a time. `--check-hand` replays those and
reports how many the automatic matcher reproduces. A harness whose matching
disagrees with careful human reading is measuring itself, not the model.

WHAT IT DELIBERATELY DOES NOT DO
It does not produce a single "accuracy" number. A report clause can describe
several calculi ("a few calculi in the lower pole"), the summary clause often
repeats an earlier finding, and an unmatched detection may be a real stone the
radiologist did not mention. So it reports matched / missed / unmatched
separately, with the errors on the matched set, and leaves the interpretation
where it belongs.

Usage:
    python -m calculus.evaluate.score_cohort --run final_check_deployment
    python -m calculus.evaluate.score_cohort --run final_check_deployment \\
        --compare case_analysis
    python -m calculus.evaluate.score_cohort --run final_check_deployment \\
        --check-hand case_analysis/analysis/paired_findings.csv
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:                                     # pragma: no cover
    linear_sum_assignment = None


# --------------------------------------------------------------------------
# parsing the radiologist's free text
# --------------------------------------------------------------------------
# Sizes appear as "20 x 14 mm", "2.2 x 3.1 x 2.9 cm", "~4.5 mm",
# "4.9 x 3.7 mm", "12.1 x 10.8 mm". Take the LARGEST stated axis: our own
# max_diameter_mm is a caliper, and comparing a caliper against a mid-axis is a
# category error that would show as a systematic underestimate.
SIZE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:x|×)\s*(\d+(?:\.\d+)?)"
    r"(?:\s*(?:x|×)\s*(\d+(?:\.\d+)?))?\s*(mm|cm)", re.I)
SIZE1_RE = re.compile(r"(?:approximately|about|~|of)\s*(\d+(?:\.\d+)?)\s*(mm|cm)", re.I)
# Density: "1396 HU", "HU ~1174", "HU: 606", "attenuation of 607 HU",
# "with 22 HU", "(1078 HU)", "density 1497 HU"
HU_RE = re.compile(r"(?:hu[\s:~]*|attenuation of\s*|density\s*)(\d{2,4})"
                   r"|(\d{2,4})\s*hu", re.I)

LEFT_RE = re.compile(r"\bleft\b|\blt\b", re.I)
RIGHT_RE = re.compile(r"\bright\b|\brt\b", re.I)
BILAT_RE = re.compile(r"bilateral|both kidney|b/l", re.I)

# Compartment, most specific first: a clause naming the VUJ is about the VUJ
# even though it also contains the word "ureteric".
# ORDER MATTERS AND IS SUBTLE. "left vesicoureteric Junction intravesically"
# names BOTH the junction and the bladder. It must resolve to vuj: the junction
# is the specific anatomical statement and "intravesically" only says the stone
# has begun to pass into the bladder. Testing bladder first made 8676809's
# target incompatible with our (correct) ureteric detection, and the case
# scored as a miss when the stone had in fact been found.
COMPARTMENTS = [
    ("vuj", re.compile(r"\bvuj\b|vesico-?ureteric|vesicoureteral|"
                       r"vesicoureteric", re.I)),
    ("bladder", re.compile(r"bladder|vesical(?!\s*[-]?ureteric)|intravesical", re.I)),
    ("puj", re.compile(r"\bpuj\b|pelvi-?ureteric|pelviureteric", re.I)),
    ("ureter", re.compile(r"ureter", re.I)),
    ("pelvis", re.compile(r"renal pelvis|pyelo", re.I)),
    ("renal", re.compile(r"kidney|renal|calyc|caly[xc]|pole|staghorn|"
                         r"nephrolith|microlith|concretion", re.I)),
]
ZONES = [
    ("upper", re.compile(r"upper", re.I)),
    ("mid", re.compile(r"\bmid\b|mid-?pole|interpolar", re.I)),
    ("lower", re.compile(r"lower|low\b|inferior", re.I)),
]

# "a few", "multiple", "several" -- a single clause covering several calculi.
# Recorded so a validator is not shown a phantom "missed" for stones two and
# three of a clause that named only its largest.
PLURAL_RE = re.compile(r"\bfew\b|\bmultiple\b|\bseveral\b|\bnumerous\b|calculi",
                       re.I)


def _mm(val, unit):
    v = float(val)
    return v * 10.0 if unit.lower() == "cm" else v


def parse_size_mm(text):
    """Largest stated axis, in mm, or None."""
    best = None
    for m in SIZE_RE.finditer(text):
        axes = [m.group(1), m.group(2), m.group(3)]
        unit = m.group(4)
        for a in axes:
            if a:
                v = _mm(a, unit)
                best = v if best is None else max(best, v)
    if best is None:
        m = SIZE1_RE.search(text)
        if m:
            best = _mm(m.group(1), m.group(2))
    return best


# OUR OWN size column is "11.7 x 7.6 x 5.1" -- three axes, no unit, because the
# unit is in the header ("Size (in mm)"). The report parser above REQUIRES an
# explicit mm/cm, correctly: "20 x 14" in free text without a unit is not
# necessarily millimetres. So our sizes need their own parser.
#
# This was caught by --check-hand on the first run: all 21 hand pairings
# disagreed with "auto nan", because every one of our sizes parsed as None and
# the cost function fell back to its unknown-size penalty for all of them. The
# summary numbers still printed, and looked reasonable. Exactly the class of
# silent wrongness this harness exists to catch -- found in the harness itself.
OURS_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)")


def parse_our_size_mm(text):
    """Largest axis of our own 'A x B x C (AP x TR x CC)' size string, in mm.

    The axis label is stripped explicitly. It happens to hold no digits, so
    findall would survive it today, but relying on that would make any future
    label containing a number silently corrupt every size comparison.
    """
    txt = re.sub(r"\([^)]*\)", " ", str(text))
    vals = [float(v) for v in OURS_SIZE_RE.findall(txt)]
    return max(vals) if vals else None


def parse_hu(text):
    for m in HU_RE.finditer(text):
        v = m.group(1) or m.group(2)
        if v:
            iv = int(v)
            # a plain "22 HU" is in the data and is implausible for a stone, but
            # it is what the report says -- keep it and let the comparison show
            if 10 <= iv <= 4000:
                return float(iv)
    return None


def parse_side(text):
    if BILAT_RE.search(text):
        return "both"
    l, r = bool(LEFT_RE.search(text)), bool(RIGHT_RE.search(text))
    if l and r:
        return "both"
    return "left" if l else ("right" if r else None)


def parse_compartment(text):
    for name, rx in COMPARTMENTS:
        if rx.search(text):
            return name
    return None


def parse_zone(text):
    """The zone, or None when the clause names more than one.

    "few small calculi in the UPPER and MID pole calyx" states two zones. Taking
    the first match invents a precision the report does not have, and then
    penalises a detection in the other named zone.
    """
    hits = [name for name, rx in ZONES if rx.search(text)]
    return hits[0] if len(hits) == 1 else None


def is_summary_clause(clause, earlier):
    """Report impressions repeat the findings. A clause that names nothing new
    is a restatement, not an additional finding, and counting it would inflate
    the miss count.
    """
    if parse_size_mm(clause) is not None or parse_hu(clause) is not None:
        return False                       # states a number: a real finding
    cs, cc = parse_side(clause), parse_compartment(clause)
    for e in earlier:
        if parse_compartment(e) != cc:
            continue
        es = parse_side(e)
        # A BILATERAL summary restates one-sided findings, so its side is "both"
        # and never equals "left" or "right". Comparing them directly made
        # 8677561's impression ("Bilateral non-obstructive renal calculi -- a few
        # in the ... right kidney and a single calculus in the ... left kidney")
        # count as a fifth, unmatched finding, and the study scored a miss for a
        # sentence that only repeated two findings already matched.
        if cs == es or cs == "both" or es == "both" or cs is None or es is None:
            return True
    return False


def targets_from_report(text):
    """One structured target per report finding."""
    clauses = [c.strip() for c in str(text).split("|")
               if c.strip() and c.strip().lower() != "nan"]
    out, seen = [], []
    for c in clauses:
        if is_summary_clause(c, seen):
            seen.append(c)
            continue
        seen.append(c)
        comp = parse_compartment(c)
        if comp is None:
            out.append({"clause": c, "compartment": None, "side": None,
                        "zone": None, "mm": None, "hu": None,
                        "plural": False, "unparsed": True})
            continue
        side = parse_side(c)
        sides = ["left", "right"] if side == "both" else [side]
        for sd in sides:
            out.append({"clause": c, "compartment": comp, "side": sd,
                        "zone": parse_zone(c), "mm": parse_size_mm(c),
                        "hu": parse_hu(c),
                        "plural": bool(PLURAL_RE.search(c)),
                        "unparsed": False})
    return out


# --------------------------------------------------------------------------
# our detections
# --------------------------------------------------------------------------
# Our Organ strings, mapped onto the same compartment vocabulary. "Ureter (vuj)"
# is a VUJ stone; "Renal Pelvis Or Perirenal" is pelvis.
def detection_compartment(organ, location):
    """Our Organ/Location strings -> the report's compartment vocabulary.

    THE LOCATION STRING IS NOT A COMPARTMENT. An earlier version tested
    `"vuj" in location`, which made "Near VUJ - ~14 mm from UVJ" classify as a
    VUJ stone. On 8676429 that let a 12.8 mm lower-ureteric detection be matched
    to a reported 51 mm BLADDER calculus (bladder is compatible with vuj), while
    the actual 53.1 mm bladder detection was left unmatched -- an mm_err of
    -38.2 recorded as a successful match.
    "Near VUJ" means near, not at. Only the ORGAN field decides the compartment.
    """
    o = str(organ).lower()
    loc = str(location).lower()
    if "bladder" in o:
        return "bladder"
    if "vuj" in o:                      # "Ureter (vuj)" -- the organ, not the location
        return "vuj"
    if "ureter" in o:
        if "upper" in o or "near puj" in loc:
            return "puj"
        return "ureter"
    if "pelvis" in o or "perirenal" in o:
        return "pelvis"
    return "renal"


# Which report compartments a detection may satisfy. The junctions are genuinely
# ambiguous in reports -- a stone at the PUJ is described as renal pelvic by one
# radiologist and proximal ureteric by another -- so those are allowed to match
# either, and the ambiguity is stated rather than resolved by fiat.
COMPATIBLE = {
    "renal":   {"renal", "pelvis"},
    "pelvis":  {"pelvis", "renal", "puj"},
    "puj":     {"puj", "pelvis", "ureter", "renal"},
    "ureter":  {"ureter", "puj", "vuj"},
    "vuj":     {"vuj", "ureter", "bladder"},
    "bladder": {"bladder", "vuj"},
}


def detections_for(run, sid):
    p = os.path.join(run, "reports", f"{sid}_calculi.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    d = pd.read_csv(p)
    if not len(d):
        return d
    d = d.copy()
    d["comp"] = [detection_compartment(o, l)
                 for o, l in zip(d["Organ"], d["Location"])]
    d["side_l"] = d["Side"].astype(str).str.lower()
    d["mm"] = [parse_our_size_mm(s) for s in d["Size (in mm)"]]
    d["hu"] = pd.to_numeric(d["Density (HU)"], errors="coerce")
    d["zone_l"] = [parse_zone(str(l)) for l in d["Location"]]
    return d


# --------------------------------------------------------------------------
# matching
# --------------------------------------------------------------------------
BIG = 1e6


def pair_cost(t, r):
    """Cost of claiming detection `r` for target `t`. BIG means incompatible."""
    if t["compartment"] is None:
        return BIG
    if r["comp"] not in COMPATIBLE.get(t["compartment"], {t["compartment"]}):
        return BIG
    # Side is only a constraint when BOTH sides are actually known. Our bladder
    # rows carry Side = "-" because a bladder calculus has no side, and "-" is a
    # truthy string: comparing "right" against "-" declared them incompatible
    # and scored 8674941's genuinely-found stone as a miss.
    r_side = r["side_l"] if r["side_l"] not in ("-", "", "nan", "none") else None
    if t["side"] and r_side and t["side"] != r_side:
        return BIG
    cost = 0.0
    # size: mm of disagreement, the dominant term because it is the most
    # reliably stated quantity in a report
    if t["mm"] is not None and r["mm"] is not None and np.isfinite(r["mm"]):
        cost += abs(t["mm"] - r["mm"])
    else:
        cost += 5.0                      # unknown on either side: mild penalty
    # density: scaled so 200 HU of disagreement costs about as much as 2 mm
    if t["hu"] is not None and r["hu"] is not None and np.isfinite(r["hu"]):
        cost += abs(t["hu"] - r["hu"]) / 100.0
    else:
        cost += 2.0
    # Zone, when both state one. Weighted LOW on purpose. At 3.0 it outweighed
    # 2.5 mm of size disagreement, so on 8675246 a 4.5 mm target matched a 2.0 mm
    # stone in the nominally-correct third rather than the 4.5 mm stone one third
    # away -- and our thirds boundary is itself approximate. Size is the more
    # reliable signal; zone breaks ties.
    if t["zone"] and r["zone_l"] and t["zone"] != r["zone_l"]:
        cost += 1.0
    return cost


def match(targets, dets):
    """One-to-one assignment. Returns (pairs, unmatched_target_idx,
    unmatched_det_idx)."""
    if not targets or not len(dets):
        return [], list(range(len(targets))), list(range(len(dets)))
    C = np.full((len(targets), len(dets)), BIG)
    for i, t in enumerate(targets):
        for j, r in enumerate(dets.to_dict("records")):
            C[i, j] = pair_cost(t, r)
    if linear_sum_assignment is None:      # pragma: no cover
        raise SystemExit("scipy is required for the assignment step")
    ri, ci = linear_sum_assignment(C)
    pairs, ut, ud = [], set(range(len(targets))), set(range(len(dets)))
    for i, j in zip(ri, ci):
        if C[i, j] >= BIG:
            continue                        # incompatible: not a match
        pairs.append((i, j, C[i, j]))
        ut.discard(i)
        ud.discard(j)
    return pairs, sorted(ut), sorted(ud)


# --------------------------------------------------------------------------
def score_run(run, cases):
    rows = []
    for case in cases.itertuples():
        sid = str(case.study_id)
        dets = detections_for(run, sid)
        targets = targets_from_report(case.finding_reported)
        pairs, ut, ud = match(targets, dets)
        drec = dets.to_dict("records") if len(dets) else []
        for i, j, c in pairs:
            t, r = targets[i], drec[j]
            rows.append({
                "study_id": sid, "category": case.category, "outcome": "matched",
                "report_clause": t["clause"][:100],
                "report_comp": t["compartment"], "report_side": t["side"],
                "report_mm": t["mm"], "report_hu": t["hu"],
                "our_organ": r["Organ"], "our_side": r["Side"],
                "our_mm": r["mm"], "our_hu": r["hu"],
                "our_location": r["Location"],
                "mm_err": (None if t["mm"] is None or r["mm"] is None
                           else round(r["mm"] - t["mm"], 2)),
                "hu_ratio": (None if not t["hu"] or not r["hu"]
                             else round(r["hu"] / t["hu"], 3)),
                "plural_clause": t["plural"], "cost": round(c, 2)})
        for i in ut:
            t = targets[i]
            rows.append({
                "study_id": sid, "category": case.category,
                "outcome": "unparsed" if t["unparsed"] else "MISSED",
                "report_clause": t["clause"][:100],
                "report_comp": t["compartment"], "report_side": t["side"],
                "report_mm": t["mm"], "report_hu": t["hu"],
                "plural_clause": t["plural"]})
        for j in ud:
            r = drec[j]
            rows.append({
                "study_id": sid, "category": case.category,
                "outcome": "UNMATCHED_DETECTION",
                "our_organ": r["Organ"], "our_side": r["Side"],
                "our_mm": r["mm"], "our_hu": r["hu"],
                "our_location": r["Location"]})
    return pd.DataFrame(rows)


def summarise(d, label=""):
    n_t = int((d.outcome.isin(["matched", "MISSED"])).sum())
    n_m = int((d.outcome == "matched").sum())
    n_x = int((d.outcome == "MISSED").sum())
    n_u = int((d.outcome == "UNMATCHED_DETECTION").sum())
    n_p = int((d.outcome == "unparsed").sum())
    print(f"\n{'='*74}\n{label or 'SCORE'}\n{'='*74}")
    print(f"  report findings parsed      {n_t}")
    print(f"  matched                     {n_m}"
          + (f"   ({100.0*n_m/n_t:.0f}%)" if n_t else ""))
    print(f"  MISSED                      {n_x}")
    print(f"  unmatched detections        {n_u}"
          "   <- NOT necessarily false: may be real and unmentioned")
    if n_p:
        print(f"  clauses we could not parse  {n_p}   <- fix the parser, not the model")
    mm = d[d.outcome == "matched"].mm_err.dropna()
    if len(mm):
        print(f"\n  SIZE   n={len(mm)}  mean abs {mm.abs().mean():.2f} mm"
              f"   median {mm.median():+.2f}"
              f"   within 2 mm {int((mm.abs()<=2).sum())}/{len(mm)}")
    hu = d[d.outcome == "matched"].hu_ratio.dropna()
    if len(hu):
        print(f"  DENSITY n={len(hu)}  median ratio {hu.median():.2f}x"
              f"   within 20% {int(((hu>=0.8)&(hu<=1.2)).sum())}/{len(hu)}")
    print("\n  by category:")
    for cat, g in d.groupby("category"):
        t = int((g.outcome.isin(["matched", "MISSED"])).sum())
        m = int((g.outcome == "matched").sum())
        u = int((g.outcome == "UNMATCHED_DETECTION").sum())
        print(f"    {cat:24s} matched {m}/{t}   unmatched detections {u}")
    return {"targets": n_t, "matched": n_m, "missed": n_x, "unmatched": n_u}


def check_hand(d, hand_csv):
    """Does the automatic matcher agree with careful hand pairing?"""
    if not os.path.exists(hand_csv):
        print(f"\nno hand-paired file at {hand_csv}")
        return
    h = pd.read_csv(hand_csv)
    h["study"] = h.study.astype(str)
    m = d[d.outcome == "matched"]
    ok = miss = 0
    print(f"\n{'='*74}\nAGREEMENT WITH HAND PAIRING  ({len(h)} pairs)\n{'='*74}")
    print("  a harness whose matching disagrees with careful human reading is")
    print("  measuring itself, not the model\n")
    for r in h.itertuples():
        cand = m[(m.study_id == r.study)
                 & (m.report_mm.notna())
                 & (np.isclose(m.report_mm.astype(float), float(r.rep_mm),
                               atol=0.35))]
        if len(cand) and np.isclose(float(cand.iloc[0].our_mm), float(r.our_mm),
                                    atol=0.35):
            ok += 1
        else:
            miss += 1
            got = (f"{cand.iloc[0].our_mm:.1f}" if len(cand) else "no match")
            print(f"  DISAGREE  {r.study}  {r.finding[:34]:34s} "
                  f"report {r.rep_mm:5.1f}  hand {r.our_mm:5.1f}  auto {got}")
    print(f"\n  reproduced {ok}/{len(h)} hand pairings")
    if miss:
        print("  investigate every disagreement before trusting the numbers above")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--compare", default=None, help="a second run, for A/B")
    ap.add_argument("--cases", default="validation_cases.csv")
    ap.add_argument("--check-hand", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cases = pd.read_csv(a.cases)
    d = score_run(a.run, cases)
    s1 = summarise(d, f"RUN: {a.run}")
    out = a.out or os.path.join(a.run, "score_cohort.csv")
    d.to_csv(out, index=False)
    print(f"\nwrote {out}")

    if a.check_hand:
        check_hand(d, a.check_hand)

    if a.compare:
        d2 = score_run(a.compare, cases)
        s2 = summarise(d2, f"COMPARE: {a.compare}")
        print(f"\n{'='*74}\nA/B\n{'='*74}")
        print(f"{'':24s} {a.compare:>22s} {a.run:>22s}")
        for k in ("targets", "matched", "missed", "unmatched"):
            print(f"  {k:22s} {s2[k]:>22d} {s1[k]:>22d}")
        print("\n  matched UP and unmatched DOWN is an improvement. Either one")
        print("  alone is not: a run can match more by reporting everything.")


if __name__ == "__main__":
    main()
