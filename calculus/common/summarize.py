"""Build distribution CSVs into csv/ so the cohort can be inspected in a viewer.

Joins the four data sources produced so far:
    calculus.xlsx        report-mined cohort (24,742 unique studies)
    csv/worklist_*.csv   what was selected for download
    dicoms/manifest.csv  what actually downloaded (status, bytes, files)
    csv/triage_*.csv     series geometry + contrast phase

Writes:
    csv/study_master.csv          one row per downloaded study, everything joined
    csv/dist_cohort.csv           full cohort by tier x month x fetchable
    csv/dist_slice_spacing.csv    spacing histogram over triaged series
    csv/dist_acquisition.csv      kernel / kVp / manufacturer / in-plane counts
    csv/dist_download.csv         download status and size stats by tier

Usage:
    ./venv/bin/python summarize.py
"""
import os
import sys

import numpy as np
import pandas as pd

from calculus.common import build_worklist as bw

# this file lives in utils/, so the project root is one level up. All data
# directories hang off ROOT, never off the script's own folder.
HERE = os.path.dirname(os.path.abspath(__file__))
# package root -> project root: calculus/<sub>/x.py is two levels down
ROOT = os.path.dirname(os.path.dirname(HERE))
from calculus.common.paths import CSV                 # noqa: E402  results dir is per-run
FETCHABLE_AFTER = "2026-06-29"   # measured retention cliff; see README


def out(df, name):
    p = os.path.join(CSV, name)
    df.to_csv(p, index=False)
    print(f"  {name:<28} {len(df):>6} rows")
    return p


def main():
    os.makedirs(CSV, exist_ok=True)
    print("writing distribution CSVs to csv/\n")

    # ---- full report-mined cohort -------------------------------------
    df = pd.read_excel(os.path.join(ROOT, "calculus.xlsx"))
    df = df.drop_duplicates(subset="study_iuid")
    df["tier"] = bw.assign_tiers(df)
    rep = pd.to_datetime(df["report_created_at"], format="mixed", utc=True)
    df["month"] = rep.dt.strftime("%Y-%m")
    df["date"] = rep.dt.strftime("%Y-%m-%d")
    df["fetchable"] = rep >= pd.Timestamp(FETCHABLE_AFTER, tz="UTC")

    cohort = (df.groupby(["tier", "month", "variant", "family", "fetchable"])
                .size().reset_index(name="n_studies"))
    out(cohort, "dist_cohort.csv")

    daily = (df.groupby(["date", "tier"]).size().reset_index(name="n_studies")
               .pivot(index="date", columns="tier", values="n_studies")
               .fillna(0).astype(int).reset_index())
    out(daily, "dist_cohort_daily.csv")

    # ---- downloads -----------------------------------------------------
    man_path = os.path.join(ROOT, "dicoms", "manifest.csv")
    man = pd.read_csv(man_path) if os.path.exists(man_path) else pd.DataFrame()
    if len(man):
        man["study_id"] = man["study_id"].astype(str)
        man["mb"] = man["bytes"].fillna(0) / 1e6
        dl = (man.groupby(["tier", "status"])
                 .agg(n=("study_id", "size"), total_mb=("mb", "sum"),
                      mean_mb=("mb", "mean"), median_mb=("mb", "median"),
                      mean_files=("n_files", "mean"),
                      mean_seconds=("seconds", "mean"))
                 .round(1).reset_index())
        out(dl, "dist_download.csv")

    # ---- triage --------------------------------------------------------
    sp = os.path.join(CSV, "triage_series.csv")
    st = os.path.join(CSV, "triage_study.csv")
    if os.path.exists(sp):
        ser = pd.read_csv(sp)
        ser["study_id"] = ser["study_id"].astype(str)

        bins = [0, .7, 1.01, 1.26, 1.51, 2.01, 3.01, 5.01, 99]
        labels = ["<=0.7", "0.7-1.0", "1.0-1.25", "1.25-1.5", "1.5-2.0",
                  "2.0-3.0", "3.0-5.0", ">5.0"]
        ser["spacing_bin"] = pd.cut(ser["slice_spacing_mm"], bins=bins,
                                    labels=labels, right=True)
        spacing = (ser.groupby(["spacing_bin", "is_contrast", "is_axial"],
                               observed=True)
                      .size().reset_index(name="n_series"))
        out(spacing, "dist_slice_spacing.csv")

        acq = []
        for col in ["kernel", "kvp", "manufacturer", "model", "rows_cols",
                    "phase_src", "verdict"]:
            if col in ser:
                c = ser[col].fillna("(blank)").astype(str).value_counts()
                acq.append(pd.DataFrame({"field": col, "value": c.index,
                                         "n_series": c.values}))
        out(pd.concat(acq, ignore_index=True), "dist_acquisition.csv")

    # ---- master study table -------------------------------------------
    if os.path.exists(st):
        study = pd.read_csv(st)
        study["study_id"] = study["study_id"].astype(str)
        base = df[["study_id", "study_iuid", "tier", "variant", "family",
                   "calculus_flag", "calculus_type", "calculus_line",
                   "date", "month", "fetchable"]].copy()
        base["study_id"] = base["study_id"].astype(str)
        m = study.merge(base, on="study_id", how="left",
                        suffixes=("", "_cohort"))
        if len(man):
            m = m.merge(man[["study_id", "status", "bytes", "n_files",
                             "seconds"]], on="study_id", how="left")
            m["size_mb"] = (m["bytes"] / 1e6).round(1)
            m = m.drop(columns="bytes")
        front = ["study_id", "tier", "variant", "family", "study_verdict",
                 "slice_spacing_mm", "inplane_mm", "coverage_mm", "n_slices",
                 "kernel", "kvp", "manufacturer", "has_contrast_phase",
                 "calculus_flag", "calculus_type", "size_mb", "n_files", "date"]
        cols = [c for c in front if c in m] + [c for c in m if c not in front]
        out(m[cols].sort_values(["tier", "study_id"]), "study_master.csv")

    print(f"\nall CSVs in {CSV}")
    print("\nquick view:")
    if os.path.exists(st):
        s = pd.read_csv(os.path.join(CSV, "study_master.csv"))
        print(pd.crosstab(s.study_verdict, s.variant).to_string())


if __name__ == "__main__":
    main()
