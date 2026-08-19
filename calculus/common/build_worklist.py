"""Build a stratified download worklist from calculus.xlsx.

The sheet has 24,774 report-mined studies (Jun-Jul 2026). Downloading all of
them is ~10-15 TB, so this script selects a prioritised subset.

Tiers (highest value first for the calculus pipeline):
    1 ureteric  - positives mentioning ureteric / VUJ / PUJ / pelvis.
                  These drive ureteric localisation + distance-from-UVJ, the
                  hardest part of the problem and the rarest data.
    2 urography - CT Urography studies. Non-contrast + excretory phases in one
                  session => free ground-truth ureter paths via registration.
    3 renal     - renal-only positives. Intrarenal detection, counting,
                  calyceal location, HU.
    4 negative  - reported-negative studies. Specificity testing + hard
                  negatives (vascular calcification, phleboliths).
    5 other     - positives whose type mined as 'other' (needs manual review;
                  may be gallstone / non-urinary calculi).

Usage:
    python build_worklist.py                     # pilot plan (60 studies)
    python build_worklist.py --plan phase1       # 800 studies
    python build_worklist.py --plan full         # everything (warns)
    python build_worklist.py --plan pilot --out worklist_pilot.csv
"""
import argparse
import os
import sys

import pandas as pd

# this file lives in utils/, so the project root is one level up. All data
# directories hang off ROOT, never off the script's own folder.
HERE = os.path.dirname(os.path.abspath(__file__))
# package root -> project root: calculus/<sub>/x.py is two levels down
ROOT = os.path.dirname(os.path.dirname(HERE))
XLSX = os.path.join(ROOT, "calculus.xlsx")

# tier -> number of studies per plan
PLANS = {
    "pilot":  {"ureteric": 20, "urography": 10, "renal": 15, "negative": 15, "other": 0},
    "phase1": {"ureteric": 300, "urography": 150, "renal": 200, "negative": 150, "other": 0},
    "full":   {"ureteric": None, "urography": None, "renal": None,
               "negative": None, "other": None},
}

EXTRA_SITES = "ureteric|VUJ|PUJ|pelvis"


def assign_tiers(df):
    """Label every row with exactly one tier, in priority order."""
    ctype = df["calculus_type"].fillna("").str.lower()
    pos = df["calculus_flag"] == True  # noqa: E712 - column holds real bools

    # positives whose type failed to mine land here; kept visible, not dropped
    tier = pd.Series("positive_untyped", index=df.index)
    # order matters: first match wins
    tier[(~pos)] = "negative"
    tier[pos & ctype.str.strip().eq("other")] = "other"
    tier[pos & ctype.str.contains("renal|pelvis")] = "renal"
    tier[df["family"].eq("CT Urography")] = "urography"
    tier[pos & ctype.str.contains(EXTRA_SITES)] = "ureteric"
    return tier


def stratify(sub, n, seed):
    """Take n rows from sub, spread across variant so we get a mix of
    Plain / Plain+Contrast / Contrast rather than whatever sorts first."""
    if n is None or n >= len(sub):
        return sub
    frac = n / len(sub)
    parts = []
    for _, g in sub.groupby("variant"):
        parts.append(g.sample(min(len(g), max(1, round(len(g) * frac))),
                              random_state=seed))
    out = pd.concat(parts)
    if len(out) > n:
        out = out.sample(n, random_state=seed)
    elif len(out) < n:  # top up from whatever is left
        rest = sub.drop(out.index)
        out = pd.concat([out, rest.sample(min(n - len(out), len(rest)),
                                          random_state=seed)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="pilot", choices=list(PLANS))
    ap.add_argument("--xlsx", default=XLSX)
    ap.add_argument("--out", default=None)  # -> csv/worklist_<plan>.csv
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--after", default="2026-06-29",
                    help="drop studies reported before this date - the API 504s "
                         "on them. Measured 2026-08-01 over 38 studies: every "
                         "study <= Jun 28 failed (19/19), every study >= Jun 29 "
                         "succeeded (15/15). That is a ~33-day retention window, "
                         "so this cutoff MOVES FORWARD about a day per day and "
                         "must be re-probed before each bulk download. Set to "
                         "1900-01-01 to disable.")
    args = ap.parse_args()

    out = args.out or os.path.join(ROOT, "csv", f"worklist_{args.plan}.csv")
    plan = PLANS[args.plan]

    df = pd.read_excel(args.xlsx)
    before = len(df)
    df = df.drop_duplicates(subset="study_iuid", keep="first")
    df = df[df["study_iuid"].notna() & df["study_iuid"].astype(str).str.strip().ne("")]
    print(f"loaded {before} rows -> {len(df)} unique study_iuid")

    reported = pd.to_datetime(df["report_created_at"], format="mixed", utc=True)
    keep = reported >= pd.Timestamp(args.after, tz="UTC")
    print(f"reported on/after {args.after}: {keep.sum()} "
          f"({keep.mean()*100:.0f}%) - dropping {(~keep).sum()} likely-archived")
    df = df[keep]

    df["tier"] = assign_tiers(df)
    print("\navailable per tier:")
    print(df["tier"].value_counts().to_string())

    picks = []
    for tier, n in plan.items():
        sub = df[df["tier"] == tier]
        if sub.empty:
            continue
        # within a tier, prefer positives and thinner-looking protocols first
        got = stratify(sub, n, args.seed)
        picks.append(got)
        print(f"  {tier:<10} selected {len(got):>5} / {len(sub)}")

    wl = pd.concat(picks).sample(frac=1, random_state=args.seed)  # shuffle
    wl["priority"] = wl["tier"].map(
        {"ureteric": 1, "urography": 2, "renal": 3, "negative": 4, "other": 5})
    wl = wl.sort_values(["priority", "study_id"])

    cols = ["study_id", "study_iuid", "tier", "priority", "family", "variant",
            "calculus_flag", "calculus_type", "calculus_line", "report_created_at"]
    wl[cols].to_csv(out, index=False)

    print(f"\nworklist: {len(wl)} studies -> {out}")
    print(wl.groupby(["tier", "variant"]).size().to_string())
    print("\nEstimated download at ~0.5 GB/study: "
          f"~{len(wl) * 0.5:.0f} GB")


if __name__ == "__main__":
    main()
