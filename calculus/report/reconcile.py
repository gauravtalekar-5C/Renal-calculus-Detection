"""Cross-check what the detectors accepted against what the report says.

WHY THIS EXISTS
On 8583083 the three detectors accepted seven ureteric calculi and the API
answered "Normal, 0 calculi". Nothing in the pipeline noticed. The detector CSVs
and the report were both on disk, one contradicting the other, and no code ever
compared them -- so a crash in the reporting path was indistinguishable from a
patient with no stones.

The guards added alongside this module make the specific 8583083 crash loud
(infer_study exits nonzero, the API refuses to shape a response without a report
table). But those guard the failures we have already seen. This module guards the
CONSEQUENCE instead, which is the part that must never happen again regardless of
which new bug produces it: findings that exist in the detector output must not
disappear on the way to the answer.

THE INVARIANT IS DELIBERATELY WEAK, AND THAT IS THE POINT
The obvious check -- reported count == accepted count -- would be wrong. Real,
intended transformations sit between the two:

  * drop_puj_duplicates removes a ureteric row that names the same stone as a
    kidney row at the pelvi-ureteric junction.
  * the ureteric detector's report_this column withholds ranked-out rows.
  * near-miss rows are listed separately and never counted as calculi.

A check that fires on those is a check somebody switches off within a week. So
the HARD error is only the unambiguous case:

    any detector accepted >= 1 stone   AND   the report counts 0

There is no legitimate path from "we found something" to "we found nothing" --
deduplication merges findings, it never empties them. Everything else is
recorded as a delta on the response so a future divergence is visible instead of
silent, which is the actual failure being prevented.
"""
import os

import pandas as pd


def _csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def accepted_counts(run, sid):
    """How many stones each detector actually accepted, from its own CSV."""
    per = os.path.join(run, "csv", "per_study")
    out = {}

    kid = _csv(os.path.join(per, f"{sid}_candidates.csv"))
    if len(kid) and "is_stone" in kid.columns:
        k = kid[kid.is_stone.astype(bool)]
        # a kidney-detector row whose compartment is the bladder is the same
        # object the bladder detector reports; it must not be counted twice
        if "compartment" in k.columns:
            k = k[~k.compartment.astype(str).str.startswith("bladder")]
        out["renal"] = int(len(k))
    else:
        out["renal"] = 0

    ure = _csv(os.path.join(per, f"{sid}_ureter_candidates.csv"))
    if len(ure) and "is_stone" in ure.columns:
        u = ure[ure.is_stone.astype(bool)]
        if "report_this" in u.columns:
            u = u[u.report_this.fillna(True).astype(bool)]
        out["ureteric"] = int(len(u))
    else:
        out["ureteric"] = 0

    bla = _csv(os.path.join(per, f"{sid}_bladder_candidates.csv"))
    out["bladder"] = (int(bla.is_stone.astype(bool).sum())
                      if len(bla) and "is_stone" in bla.columns else 0)
    return out


def reconcile(run, sid, reported_total, reported_counts=None):
    """Compare detector output with the reported answer.

    Returns (ok, detail, delta). ok is False ONLY for the unambiguous case:
    something was detected and nothing was reported. `delta` always describes
    the difference so a caller can surface it.
    """
    acc = accepted_counts(run, sid)
    acc_total = sum(acc.values())
    delta = {"accepted": acc, "reported_total": int(reported_total)}
    if reported_counts:
        delta["reported"] = {k: int(v) for k, v in reported_counts.items()}

    if acc_total > 0 and int(reported_total) == 0:
        parts = ", ".join(f"{k}={v}" for k, v in acc.items() if v)
        return False, (
            "the detectors accepted "
            f"{acc_total} calculus/calculi ({parts}) but the report counts 0. "
            "Findings cannot vanish between detection and reporting: this is a "
            "bug in the reporting path, not a normal study. Refusing to answer."
        ), delta

    return True, "", delta
