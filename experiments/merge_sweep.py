"""EXPERIMENT: calibrate merge_fragments' two constants. Read-only.

TOUCHES NOTHING. It monkeypatches ds.merge_fragments with a wrapper that DUMPS
the pre-merge state to disk and returns its input unchanged, runs the detector
once per study to collect that state, then sweeps the real merge_fragments over
the dumped states offline. Nothing in calculus/ is modified and no result CSV is
overwritten.

WHY A SWEEP AND NOT A JUDGEMENT CALL
MAX_DIAM_MM = 22 was a judgement call -- "the largest ureteric stone in our 37
reports is 16 mm, so 22 is safe" -- and it deleted a real 23.2 mm obstructing
calculus on the first validation case that exceeded it. Both merge constants are
picked from measurement here instead, against studies whose ground truth is
known in BOTH directions:

  MUST MERGE                                        MUST NOT MERGE
  8662768  16 pieces -> report says 1 staghorn      8677561  3 stones, bilateral,
  8664459  13 pieces -> report says 3                        genuinely separate
  8674625   4 pieces -> report says 1 (4.3x7 mm)    8675742  multiple stones in
  8677813   2 pieces -> report says 1 (18 mm)                different calyces

A rule that collapses the staghorns but also collapses 8677561's three separate
stones is not a fix, it is a different error. Both directions are reported.

Usage:
    python -m experiments.merge_sweep --dump      # run detection, save state
    python -m experiments.merge_sweep --sweep     # sweep offline (fast)
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calculus.common.paths import NIFTI, RUN               # noqa: E402
from calculus.kidney import detect_stones as ds            # noqa: E402

STATE = os.path.join(RUN, "merge_state")

TRUTH = {           # study -> (pieces we produced, calculi the report describes)
    "8662768": (16, 1), "8664459": (13, 3),
    "8674625": (4, 1),  "8677813": (2, 1),
    "8677561": (3, 3),  "8675742": (16, None),
}
MUST_MERGE = ["8662768", "8664459", "8674625", "8677813"]
MUST_NOT   = ["8677561", "8675742"]


def dump(studies):
    """Run detection, capturing the labels as they are just before the merge."""
    os.makedirs(STATE, exist_ok=True)
    real = ds.merge_fragments

    def capture(labels, n, peak_of, vol, spacing, voxel_mm3, **kw):
        sid = capture.sid
        with open(os.path.join(STATE, f"{sid}.pkl"), "wb") as f:
            # store only the label box and the HU inside it -- a whole volume
            # per study would be gigabytes
            box = ds._pad_box(labels > 0, 8.0, spacing, labels.shape)
            pickle.dump({"labels": labels[box].astype(np.int32),
                         "vol": vol[box].astype(np.float32),
                         "n": n, "peak_of": peak_of, "spacing": spacing,
                         "voxel_mm3": voxel_mm3,
                         "forbid": kw.get("forbid") or {}}, f)
        print(f"  {sid}: captured {n} labels", flush=True)
        return labels, peak_of, n, 0          # unchanged: the detector runs on

    ds.merge_fragments = capture
    for sid in studies:
        capture.sid = sid
        try:
            ds.analyse(sid, verbose=False)
        except Exception as e:                 # one bad study must not stop it
            print(f"  {sid}: {type(e).__name__}: {e}", flush=True)
    ds.merge_fragments = real


def sweep():
    files = sorted(glob.glob(os.path.join(STATE, "*.pkl")))
    if not files:
        raise SystemExit(f"no captured state in {STATE} -- run --dump first")
    rows = []
    for f in files:
        sid = os.path.basename(f)[:-4]
        with open(f, "rb") as fh:
            st = pickle.load(fh)
        for gap in (2.0, 3.0, 4.0, 5.0, 6.0):
            for hu in (50.0, 65.0, 80.0, 100.0, 120.0):
                _, _, m, nm = ds.merge_fragments(
                    st["labels"], st["n"], st["peak_of"], st["vol"],
                    st["spacing"], st["voxel_mm3"],
                    gap_mm=gap, bridge_hu=hu, forbid=st["forbid"])
                rows.append({"study_id": sid, "gap_mm": gap, "bridge_hu": hu,
                             "labels_before": st["n"], "labels_after": m,
                             "n_merges": nm})
    d = pd.DataFrame(rows)
    out = os.path.join(RUN, "csv", "merge_sweep.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    d.to_csv(out, index=False)

    piv = d.pivot_table(index=["gap_mm", "bridge_hu"], columns="study_id",
                        values="labels_after")
    print("\nlabels remaining after merge (rows = parameters, cols = study)")
    print(piv.to_string())
    print("\ntargets:  " + "  ".join(
        f"{s}->{TRUTH[s][1]}" for s in TRUTH if TRUTH[s][1]))

    have = [c for c in piv.columns if c in MUST_MERGE]
    nots = [c for c in piv.columns if c in MUST_NOT]
    print("\nscored: total excess pieces on MUST-MERGE studies (0 = perfect),")
    print("        and any loss on MUST-NOT studies (0 = nothing wrongly fused)")
    sc = []
    for (gap, hu), r in piv.iterrows():
        excess = sum(max(0, int(r[c]) - TRUTH[c][1]) for c in have
                     if pd.notna(r[c]) and TRUTH[c][1])
        harm = sum(max(0, TRUTH[c][1] - int(r[c])) for c in nots
                   if pd.notna(r[c]) and TRUTH[c][1])
        sc.append({"gap_mm": gap, "bridge_hu": hu,
                   "excess_on_must_merge": excess, "harm_on_must_not": harm})
    sd = pd.DataFrame(sc).sort_values(["harm_on_must_not", "excess_on_must_merge"])
    print(sd.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--studies", nargs="*", default=None)
    a = ap.parse_args()
    if a.dump:
        dump(a.studies or list(TRUTH))
    if a.sweep or not a.dump:
        sweep()


if __name__ == "__main__":
    main()
