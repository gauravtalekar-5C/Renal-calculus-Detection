"""Join Part 1 (kidney) and Part 2 (ureteric) into one stone analysis.

Until now the two halves lived in separate CSVs with different schemas, so
"what does this patient have" could not be answered from one file. This writes
the joined view:

    overall_stones.csv    one row per stone, kidney and ureteric together
    overall_summary.csv   one row per study, both compartments side by side

WHAT IS AND IS NOT COUNTED
--------------------------
Kidney rows come from baseline_stones.csv, which holds only accepted stones.
Ureteric rows come from ureter_candidates.csv, which holds every candidate;
only `is_stone` rows are taken; `report_this` marks what the detector releases to
the report (every accepted stone -- see detect_ureteric.TOP_K_REPORTED).

The summary counts BOTH: `n_ureteric` is every accepted ureteric stone and
`n_ureteric_reported` only the top-ranked ones. They differ a lot -- the
37-study validation found a median of 3 accepted per study against reports
describing about one -- and collapsing them into a single number would hide
exactly the disagreement that still needs settling.

CONFIDENCE IS NOT UNIFORM ACROSS THESE COLUMNS, and the summary says so per
study in `caveats`:
  * kidney counts and sizes are validated (95.6% sens / 82.7% spec, sub-mm size)
  * ureteric side is validated (27/30 on report-confirmed studies)
  * ureteric COUNT is not -- expect false positives
  * every distance to the UVJ or PUJ rests on an unvalidated landmark

Usage:
    CALCULUS_RUN=overall_stone_analysis ./venv/bin/python \
        utils/combine_stone_analysis.py
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# package root -> project root: calculus/<sub>/x.py is two levels down
ROOT = os.path.dirname(os.path.dirname(HERE))
from calculus.common.paths import CSV                      # noqa: E402

COLS = ["study_id", "source", "stone_id", "side", "location",
        "compartment", "in_collecting_system", "vertebral_level",
        "max_diameter_mm", "volume_mm3", "hu_max", "hu_mean",
        "dist_to_uvj_mm", "dist_from_puj_mm", "off_path_mm",
        "review_flag", "report_this", "centroid_vox"]


def read(name):
    p = os.path.join(CSV, name)
    if not os.path.exists(p):
        print(f"  (no {name})")
        return None
    d = pd.read_csv(p)
    d["study_id"] = d["study_id"].astype(str)
    return d


def kidney_rows(d):
    if d is None or not len(d):
        return pd.DataFrame(columns=COLS)
    out = pd.DataFrame({
        "study_id": d.study_id,
        "source": "kidney",
        "stone_id": d.get("stone_id"),
        "side": d.get("side"),
        "location": d.get("location"),
        "max_diameter_mm": d.get("max_diameter_mm"),
        "volume_mm3": d.get("volume_mm3"),
        "hu_max": d.get("hu_max"),
        "hu_mean": d.get("hu_mean"),
        "dist_to_uvj_mm": pd.NA,
        "dist_from_puj_mm": pd.NA,
        "off_path_mm": pd.NA,
        # every row in baseline_stones.csv is an accepted stone, so all of them
        # are reportable; the audit trail for rejects is candidates.csv
        "compartment": d.get("compartment"),
        # carried through so make_report can name a calyx; without it the flag
        # dies here and every calyceal stone prints as a bare third again
        "in_collecting_system": d.get("in_collecting_system", False),
        "vertebral_level": "",
        "review_flag": "",
        "report_this": True,
        "centroid_vox": d.get("centroid_vox"),
    })
    return out[COLS]


def ureteric_rows(d):
    if d is None or not len(d):
        return pd.DataFrame(columns=COLS)
    d = d[d.is_stone.astype(bool)]
    out = pd.DataFrame({
        "study_id": d.study_id,
        "source": "ureteric",
        "stone_id": d.get("candidate_id"),
        "side": d.get("side"),
        "location": d.get("zone"),
        "max_diameter_mm": d.get("max_diameter_mm"),
        "volume_mm3": d.get("volume_mm3"),
        "hu_max": d.get("hu_max"),
        "hu_mean": d.get("hu_mean"),
        "dist_to_uvj_mm": d.get("dist_to_uvj_along_mm"),
        "dist_from_puj_mm": d.get("dist_from_puj_along_mm"),
        "off_path_mm": d.get("off_path_mm"),
        "compartment": "ureter",
        "in_collecting_system": False,
        "vertebral_level": d.get("vertebral_level", ""),
        "review_flag": d.get("review_flag", ""),
        "report_this": d.get("report_this"),
        "centroid_vox": d.get("centroid_vox"),
    })
    return out[COLS]


def caveats(row):
    """Per-study warnings, so a reader of one row knows what not to trust."""
    out = []
    if row["qc_verdict"] in ("fail", "cannot_assess", "contrast"):
        out.append(f"kidney mask {row['qc_verdict']}")
    if row["n_ureteric"] > row["n_ureteric_reported"]:
        out.append(f"{row['n_ureteric']} ureteric candidates accepted, "
                   f"{row['n_ureteric_reported']} ranked for report")
    if row["n_ureteric"] > 0:
        out.append("ureteric distances rest on an unvalidated UVJ landmark")
    if row["ureteric_sides"] == "left,right":
        out.append("bilateral ureteric detection - uncommon, check the overlay")
    return "; ".join(out)


def main():
    print(f"reading {CSV}")
    kid = read("baseline_stones.csv")
    ure = read("ureter_candidates.csv")
    qc = read("kidney_qc.csv")
    summ = read("baseline_summary.csv")

    stones = pd.concat([kidney_rows(kid), ureteric_rows(ure)],
                       ignore_index=True)
    stones = stones.sort_values(["study_id", "source", "hu_max"],
                                ascending=[True, True, False])
    dest = os.path.join(CSV, "overall_stones.csv")
    stones.to_csv(dest, index=False)
    print(f"\noverall_stones.csv: {len(stones)} rows "
          f"({(stones.source == 'kidney').sum()} kidney, "
          f"{(stones.source == 'ureteric').sum()} ureteric)")

    # every study that got as far as a volume, not just those with a stone --
    # "no stones found" is a result and has to appear in the summary
    ids = set(stones.study_id)
    for d in (qc, summ, ure):
        if d is not None:
            ids |= set(d.study_id)

    rows = []
    for sid in sorted(ids):
        k = stones[(stones.study_id == sid) & (stones.source == "kidney")]
        u = stones[(stones.study_id == sid) & (stones.source == "ureteric")]
        ur = u[u.report_this.astype(bool)] if len(u) else u
        qrow = qc[qc.study_id == sid] if qc is not None else None
        rows.append({
            "study_id": sid,
            "qc_verdict": (qrow.iloc[0].get("verdict", "")
                           if qrow is not None and len(qrow) else ""),
            "n_kidney": len(k),
            "kidney_sides": ",".join(sorted(set(k.side.dropna()))),
            "kidney_locations": ",".join(sorted(set(k.location.dropna()))),
            "largest_kidney_mm": (round(float(k.max_diameter_mm.max()), 1)
                                  if len(k) else 0.0),
            "n_ureteric": len(u),
            "n_ureteric_reported": len(ur),
            "ureteric_sides": ",".join(sorted(set(ur.side.dropna()))),
            "ureteric_zones": ",".join(sorted(set(ur.location.dropna()))),
            "largest_ureteric_mm": (round(float(ur.max_diameter_mm.max()), 1)
                                    if len(ur) else 0.0),
            "min_dist_to_uvj_mm": (round(float(ur.dist_to_uvj_mm.min()), 1)
                                   if len(ur) and ur.dist_to_uvj_mm.notna().any()
                                   else ""),
            "total_stones": len(k) + len(ur),
        })
    s = pd.DataFrame(rows)
    s["caveats"] = s.apply(caveats, axis=1)
    s.to_csv(os.path.join(CSV, "overall_summary.csv"), index=False)

    print(f"overall_summary.csv: {len(s)} studies")
    print(f"  with a kidney stone   {(s.n_kidney > 0).sum()}")
    print(f"  with a ureteric stone {(s.n_ureteric > 0).sum()}"
          f"  ({(s.n_ureteric_reported > 0).sum()} after ranking)")
    print(f"  with both             {((s.n_kidney > 0) & (s.n_ureteric > 0)).sum()}")
    print(f"  with neither          {(s.total_stones == 0).sum()}")
    print(f"  bilateral ureteric    {(s.ureteric_sides == 'left,right').sum()}"
          f"   <- uncommon in reports; the known false-positive signature")


if __name__ == "__main__":
    main()
