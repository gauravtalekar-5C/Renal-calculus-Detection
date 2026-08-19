"""What does SEED_HU cost us? Replayed offline, no re-detection.

SEED_HU is the "must contain at least one voxel this bright" test. It exists
only because we have no CNN yet: at a bare 130 HU threshold (what Elton et al.
use, because their CNN cleans up afterwards) we measured 81 false stones per
scan. Raising the seed to 200 cut that to 9 -- but it also discards low-density
calculi, and uric acid stones live around 150-250 HU.

8379961 is the concrete case: the report says that kidney contains calculi, and
we rejected candidates peaking at 176 and 183 HU as "no_dense_core".

This sweep needs NO re-run of detection. SEED_HU is used in exactly one
accept/reject comparison, and candidates.csv records every candidate that was
ever generated -- including the rejected ones -- with the peak the test compared
(seed_peak_hu), plus bone_frac, vessel_frac and max_diameter_mm. So the whole
decision chain can be replayed at any threshold from the CSV.

The rejection order is reproduced exactly as detect_stones applies it:

    no_dense_core  ->  bone_partial_volume  ->  vascular_calcification
                   ->  below_min_diameter

which matters: lowering the seed does not automatically promote a candidate to
"stone", it just lets the next test have a say.

Usage:
    CALCULUS_RUN=run_full44 ./venv/bin/python utils/seed_sweep.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# package root -> project root: calculus/<sub>/x.py is two levels down
ROOT = os.path.dirname(os.path.dirname(HERE))
from calculus.common.paths import CSV                                   # noqa: E402
from calculus.kidney import detect_stones as ds                              # noqa: E402
from calculus.evaluate.compare_reports import negated, RENAL_RE           # noqa: E402
import re                                               # noqa: E402

SEEDS = [130, 150, 175, 200, 250, 300]


def verdict(row, seed):
    """Replay detect_stones' rejection chain at a given SEED_HU."""
    if row.seed_peak_hu < seed:
        return "no_dense_core"
    if row.bone_frac > 0.5:
        return "bone_partial_volume"
    if row.vessel_frac > 0.5:
        return "vascular_calcification"
    if row.max_diameter_mm < ds.MIN_DIAM_MM:
        return "below_min_diameter"
    return ""


def truth_table():
    """study_id -> does the report describe an intrarenal calculus."""
    m = pd.read_csv(os.path.join(CSV, "study_master.csv"))
    m["study_id"] = m.study_id.astype(str)
    out = {}
    for r in m.itertuples():
        ctype = str(getattr(r, "calculus_type", "") or "")
        line = str(getattr(r, "calculus_line", "") or "")
        renal = "renal" in ctype.lower() or "pelvis" in ctype.lower()
        if renal and line:
            parts = [p for p in re.split(r"[|.]", line) if RENAL_RE.search(p)]
            if parts and all(negated(p) for p in parts):
                renal = False
        out[r.study_id] = renal
    return out


def main():
    cp = os.path.join(CSV, "candidates.csv")
    if not os.path.exists(cp):
        sys.exit(f"missing {cp} - run detect_stones.py first")
    c = pd.read_csv(cp)
    if "seed_peak_hu" not in c.columns:
        sys.exit("candidates.csv predates the seed_peak_hu column - re-run "
                 "detect_stones.py so the sweep has the value the test uses")
    c["study_id"] = c.study_id.astype(str)

    summ = pd.read_csv(os.path.join(CSV, "baseline_summary.csv"))
    summ["study_id"] = summ.study_id.astype(str)
    # studies the pipeline itself refused (contrast phase, paediatric) are not
    # evidence either way
    bad = set(summ[summ.error.notna() & (summ.error.astype(str) != "")].study_id)
    gp = os.path.join(CSV, "patient_gate.csv")
    if os.path.exists(gp):
        g = pd.read_csv(gp)
        bad |= set(g[g.excluded.astype(bool)].study_id.astype(str))

    truth = truth_table()
    studies = [s for s in summ.study_id if s not in bad and s in truth]
    print(f"{len(studies)} analysable studies "
          f"({sum(truth[s] for s in studies)} report a renal calculus)\n")

    rows = []
    for seed in SEEDS:
        v = c.apply(lambda r: verdict(r, seed), axis=1)
        kept = c[v == ""]
        n_by_study = kept.groupby("study_id").size().to_dict()
        tp = fp = fn = tn = 0
        for s in studies:
            found = n_by_study.get(s, 0) > 0
            if truth[s] and found:
                tp += 1
            elif truth[s]:
                fn += 1
            elif found:
                fp += 1
            else:
                tn += 1
        # false candidates per scan: kept candidates in studies the report says
        # have no renal calculus -- the closest thing to FP/scan we can measure
        # without annotation
        neg = [s for s in studies if not truth[s]]
        fp_per_scan = (sum(n_by_study.get(s, 0) for s in neg) / len(neg)
                       if neg else np.nan)
        rows.append({
            "SEED_HU": seed,
            "stones_kept": int((v == "").sum()),
            "sens": f"{tp}/{tp+fn}" + (f" ({100*tp/(tp+fn):.0f}%)" if tp+fn else ""),
            "spec": f"{tn}/{tn+fp}" + (f" ({100*tn/(tn+fp):.0f}%)" if tn+fp else ""),
            "sens_pct": 100*tp/(tp+fn) if tp+fn else np.nan,
            "spec_pct": 100*tn/(tn+fp) if tn+fp else np.nan,
            "false_cand_per_neg_scan": round(fp_per_scan, 2),
            "rej_no_dense_core": int((v == "no_dense_core").sum()),
            "rej_bone": int((v == "bone_partial_volume").sum()),
        })
    d = pd.DataFrame(rows)
    print(d[["SEED_HU", "stones_kept", "sens", "spec",
             "false_cand_per_neg_scan", "rej_no_dense_core"]].to_string(index=False))

    out = os.path.join(CSV, "seed_sweep.csv")
    d.to_csv(out, index=False)
    print(f"\nwrote {out}")

    cur = d[d.SEED_HU == ds.SEED_HU]
    if len(cur):
        print(f"\ncurrent setting is SEED_HU={ds.SEED_HU}: "
              f"sens {cur.sens.iloc[0]}, spec {cur.spec.iloc[0]}")
    best = d.dropna(subset=["sens_pct"])
    if len(best):
        b = best.loc[(best.sens_pct + best.spec_pct).idxmax()]
        print(f"best sens+spec in this sweep: SEED_HU={int(b.SEED_HU)} "
              f"(sens {b.sens}, spec {b.spec})")
        print("\nNote this optimises presence-per-study on a small sample, not "
              "per-stone accuracy. It says where the knee is, not what to ship.")


if __name__ == "__main__":
    main()
