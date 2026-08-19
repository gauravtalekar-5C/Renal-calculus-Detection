"""Every study that did not get a clean answer, and exactly why.

"Sensitivity is 95.7 %" is only half the story: it is measured on the studies
that reached the detector. This lists the ones that did not, and the ones that
did but disagreed with the report, in a single file:

    reports/ISSUES.csv     one row per study with a problem, worst first

CATEGORIES, in the order they appear
-----------------------------------
    not_extracted     a DICOM zip exists but no volume came out of it: series
                      triage found nothing measurable in the study
    not_segmented     a volume exists but TotalSegmentator never ran, or the
                      paediatric age gate stopped it
    mask_fail         the mask is genuinely truncated -- the detector was not
                      given a fair search region
    cannot_assess     the kidney is cut off by the field of view. NOT a model
                      failure, and critically NOT the same as "no stones":
                      nobody looked
    contrast          enhanced study; contrast in the collecting system is
                      indistinguishable from calculus
    missed_stone      report states a calculus, we found none  (false negative)
    extra_stone       we found a calculus, report states none  (false positive)
    size_disagrees    both agree a stone is present, largest differs by >= 5 mm
    ureteric_pending  Part 2 has not run this study yet, so its ureteric columns
                      are blank rather than negative

Each row carries what the report said and the path of the image that settles it.

Usage:
    CALCULUS_RUN=stone_analysis ./venv/bin/python utils/list_issues.py
"""
import glob
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# package root -> project root: calculus/<sub>/x.py is two levels down
ROOT = os.path.dirname(os.path.dirname(HERE))
from calculus.common.paths import CSV, NIFTI, RUN, SEG, ZIPS          # noqa: E402

SEV = ["missed_stone", "not_extracted", "not_segmented", "mask_fail",
       "cannot_assess", "contrast", "extra_stone", "size_disagrees",
       "ureteric_pending"]


def main():
    zips = {os.path.basename(f)[:-4] for f in glob.glob(os.path.join(ZIPS, "*.zip"))}
    nii = {os.path.basename(f).split(".")[0]
           for f in glob.glob(os.path.join(NIFTI, "*.nii.gz"))}
    seg = {d for d in os.listdir(SEG)
           if os.path.exists(os.path.join(SEG, d, "kidney_left.nii.gz"))} \
        if os.path.isdir(SEG) else set()
    urdone = {os.path.basename(f).split("_ureter")[0] for f in
              glob.glob(os.path.join(CSV, "per_study", "*_ureter_summary.csv"))}

    qc = pd.read_csv(os.path.join(CSV, "kidney_qc.csv"))
    qc["study_id"] = qc.study_id.astype(str)
    qcol = "verdict" if "verdict" in qc.columns else "qc"
    qcv = dict(zip(qc.study_id, qc[qcol]))

    cmp_p = os.path.join(CSV, "report_vs_model.csv")
    cmp_ = pd.read_csv(cmp_p) if os.path.exists(cmp_p) else pd.DataFrame()
    if len(cmp_):
        cmp_["study_id"] = cmp_.study_id.astype(str)

    rows = []

    def add(sid, cat, detail, line=""):
        img = os.path.join(os.path.basename(RUN), "overlays", sid,
                           "_coronal_mip.png")
        rows.append({"study_id": sid, "issue": cat, "detail": detail,
                     "qc_verdict": qcv.get(sid, ""),
                     "report_says": str(line)[:200],
                     "check_this_image": img if os.path.isdir(
                         os.path.join(RUN, "overlays", sid)) else ""})

    for sid in sorted(zips - nii):
        add(sid, "not_extracted",
            "zip present, no volume: triage found no measurable series")
    for sid in sorted(nii - seg):
        add(sid, "not_segmented",
            "volume present, no kidney mask: age gate or segmentation failure")
    for sid, v in sorted(qcv.items()):
        if v == "fail":
            add(sid, "mask_fail", "kidney mask truncated - search region wrong")
        elif v == "cannot_assess":
            add(sid, "cannot_assess",
                "kidney cut off by the field of view - NOT the same as no stones")
        elif v == "contrast":
            add(sid, "contrast", "enhanced study - contrast mimics calculus")
    if len(cmp_):
        for r in cmp_.itertuples():
            if r.verdict == "MISS":
                add(r.study_id, "missed_stone",
                    "report states a calculus, we found none", r.report_line)
            elif r.verdict == "FALSE POSITIVE":
                add(r.study_id, "extra_stone",
                    f"we found {r.model_n_kidney} kidney + "
                    f"{r.model_n_ureteric} ureteric, report states none",
                    r.report_line)
            elif r.verdict == "agree, size disagrees":
                add(r.study_id, "size_disagrees",
                    f"report {r.report_max_mm} mm vs ours {r.model_max_mm} mm "
                    f"(diff {r.size_diff_mm})", r.report_line)
    for sid in sorted(nii - urdone):
        add(sid, "ureteric_pending", "Part 2 has not run this study yet")

    d = pd.DataFrame(rows)
    d["_o"] = d.issue.map({c: i for i, c in enumerate(SEV)})
    d = d.sort_values(["_o", "study_id"]).drop(columns="_o")
    dest = os.path.join(RUN, "reports", "ISSUES.csv")
    d.to_csv(dest, index=False)

    print(f"wrote {dest}\n")
    print(f"{'issue':18} {'n':>4}   studies")
    for cat in SEV:
        s = d[d.issue == cat]
        if not len(s):
            continue
        ids = ", ".join(s.study_id.astype(str)[:12])
        more = f" (+{len(s)-12} more)" if len(s) > 12 else ""
        print(f"{cat:18} {len(s):4}   {ids}{more}")
    print(f"\ncohort: {len(zips)} zips -> {len(nii)} volumes -> {len(seg)} "
          f"segmented；ureteric done {len(urdone)}")


if __name__ == "__main__":
    main()
