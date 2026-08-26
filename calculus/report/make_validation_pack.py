"""Build a CASE-WISE pack for the validation team, nested by category.

            validation_pack/
                stent_in_situ/
                    8399313/        <- everything inferred for this study
                    8633709/
                renal_calculus/
                    8677561/
                    ...

A validator works category by category, so the category is the outer folder and
the study id is the inner one. Nothing is duplicated: each study appears once,
under the category it was selected to test.

WHY THIS EXISTS
---------------
Our outputs are organised by TYPE -- all reports in reports/, all overlays in
overlays/, all audit CSVs in csv/per_study/. That is right for engineering and
wrong for a validator, who works one case at a time and needs the radiologist's
words and our answer side by side, with the pictures that justify it.

So this reorganises the same files by CASE, and adds the one artefact that does
not exist anywhere else: a direct comparison of what the report says against
what we found.

WHAT EACH CASE FOLDER CONTAINS

    COMPARISON.txt      the radiologist's findings, then ours, then a verdict
                        line per finding. This is the file to read first.
    comparison.csv      the same, machine-readable, for tallying
    our_calculi.csv     our stone table (organ, side, size, HU, location)
    our_report.csv      our full structured report including the impression
    overlays/           coronal MIP, contact sheet, one PNG per stone, and the
                        ureteric sheet with accepted + rejected candidates
    organ_masks.png     segmentation QC -- if the kidney outline is wrong,
                        nothing downstream can be right
    audit/              every candidate we considered and why it was rejected

PHI: filenames use study_id only. Nothing here carries a patient name. The
DICOM zips are NOT copied in -- their headers do carry names.

MATCHING IS DELIBERATELY NOT AUTOMATIC. An earlier attempt paired each report
finding with the LARGEST detection on that side, which made the size and
location columns describe whichever object happened to be biggest rather than
the one the radiologist meant. So this prints both lists in full, side by side,
plus objective counts, and leaves the pairing to the human. A wrong automatic
pairing is worse than none: it looks authoritative.

Usage:
    python -m calculus.report.make_validation_pack
    python -m calculus.report.make_validation_pack --out /some/where
"""
import argparse
import os
import re
import shutil
import textwrap

import pandas as pd

from calculus.common import paths

CASES_CSV = "validation_cases.csv"


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def our_findings(run, sid):
    """Our stone table for one study, or an empty frame."""
    p = os.path.join(run, "reports", f"{sid}_calculi.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    d = pd.read_csv(p)
    return d[d.study_id.astype(str) == str(sid)] if "study_id" in d else d


def audit_counts(run, sid):
    """How many candidates were considered and why they were dropped."""
    out = {}
    for tag, name in (("kidney", f"{sid}_candidates.csv"),
                      ("ureteric", f"{sid}_ureter_candidates.csv")):
        p = os.path.join(run, "csv", "per_study", name)
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        if not len(d):
            continue
        rr = d.reject_reason.fillna("ACCEPTED").astype(str).str.strip()
        rr = rr.replace("", "ACCEPTED")
        out[tag] = {"n_candidates": len(d),
                    "by_reason": rr.value_counts().to_dict()}
    return out


def summary_fields(run, sid):
    """Selected per-study counters worth showing a validator."""
    keep = ("n_branched_kept_whole", "n_fragments_merged", "n_touching_split",
            "n_bone_bridges_split", "denoise_rounds", "kidney_median_hu")
    out = {}
    for name in (f"{sid}_summary.csv", f"{sid}_ureter_summary.csv"):
        p = os.path.join(run, "csv", "per_study", name)
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        if not len(d):
            continue
        pre = "kidney." if "ureter" not in name else "ureteric."
        for k in keep:
            if k in d.columns and pd.notna(d.iloc[0][k]):
                out[pre + k] = d.iloc[0][k]
    return out


def flagged_rows(run, sid):
    """Detections carrying a review or measurement flag, with the reason.

    WHY A VALIDATOR NEEDS THIS SEPARATELY. The flags live in the audit CSVs, not
    in the stone table, so a validator reading only the report cannot tell a
    measurement we trust from one we have labelled as impossible. A size of
    21.8 mm on a stone the radiologist called 4.3 x 7 mm looks like a plain
    error; the same row marked `caliper_suspect` says we know the number is
    wrong and why.
    """
    out = []
    for tag, name in (("kidney", f"{sid}_candidates.csv"),
                      ("ureteric", f"{sid}_ureter_candidates.csv")):
        p = os.path.join(run, "csv", "per_study", name)
        if not os.path.exists(p):
            continue
        try:
            d = pd.read_csv(p)
        except Exception:
            continue
        if not len(d) or "is_stone" not in d.columns:
            continue
        d = d[d.is_stone.astype(bool)]
        # The kidney detector gained `measurement_flag` after this cohort's
        # kidney pass had already run, so the column may be absent. It is a PURE
        # FUNCTION of volume_mm3, max_diameter_mm and hu_max -- all of which are
        # in the CSV -- so compute it here rather than spending an hour of
        # detection to reproduce numbers that would come out identical.
        if ("measurement_flag" not in d.columns
                and {"volume_mm3", "max_diameter_mm", "hu_max"} <= set(d.columns)):
            from calculus.kidney import detect_stones as _ds
            d = d.copy()
            d["measurement_flag"] = [
                _ds.measurement_flags(v, m, h)
                for v, m, h in zip(d.volume_mm3, d.max_diameter_mm, d.hu_max)]
        for col in ("review_flag", "measurement_flag"):
            if col not in d.columns:
                continue
            f = d[d[col].fillna("").astype(str).str.strip() != ""]
            for r in f.itertuples():
                out.append({
                    "where": tag,
                    "side": getattr(r, "side", ""),
                    "mm": getattr(r, "max_diameter_mm", ""),
                    "hu": getattr(r, "hu_max", ""),
                    "flag": str(getattr(r, col)),
                })
    return out


FLAG_MEANING = {
    "caliper_suspect":
        "the object holds too little volume for its measured length -- the SIZE "
        "is unreliable (the mask has probably reached along an adjacent bright "
        "structure). Density and position are unaffected.",
    "hu_implausible":
        "over 2000 HU, which no calculus reaches -- this is metal, contrast, or "
        "a measurement that has touched cortical bone.",
    "large_for_ureter":
        "over 20 mm, unusual for a ureter but reported rather than discarded. A "
        "cap here previously deleted a real 21 mm obstructing calculus.",
    "stent_like":
        "long, thin and hollow -- consistent with a length of ureteric stent. "
        "UNVALIDATED: it only flags, never removes.",
}


def write_comparison(dest, case, ours, audit, summ, flags=None):
    """The file a validator reads first."""
    sid = str(case.study_id)
    clauses = [c.strip() for c in str(case.finding_reported).split("|")
               if c.strip() and c.strip().lower() != "nan"]

    L = []
    L.append("=" * 78)
    L.append(f"STUDY {sid}    {case.category}  ({case.pick})")
    L.append("=" * 78)
    L.append("")
    L.append(f"scan          {case.variant} {case.family}")
    L.append(f"model scope   {case.model_capability}")
    L.append("")
    L.append("-" * 78)
    L.append("1. WHAT THE RADIOLOGIST REPORTED")
    L.append("-" * 78)
    for i, c in enumerate(clauses, 1):
        L.append(f"  [R{i}] " + "\n       ".join(textwrap.wrap(c, 70)))
    L.append("")
    L.append("-" * 78)
    L.append("2. WHAT THE MODEL FOUND")
    L.append("-" * 78)
    if not len(ours):
        L.append("  (no calculi reported by the model)")
    else:
        L.append(f"  {'#':>3}  {'organ':<26} {'side':<6} {'size (mm)':<22} "
                 f"{'HU':>6}  location")
        for i, r in enumerate(ours.itertuples(), 1):
            L.append(f"  {('M'+str(i)):>3}  {str(r.Organ):<26} "
                     f"{str(r.Side):<6} {str(getattr(r, '_4', '')):<22} "
                     f"{str(getattr(r, '_5', '')):>6}  {str(r.Location)}")
    L.append("")
    L.append("-" * 78)
    L.append("3. SIDE-BY-SIDE COUNT")
    L.append("-" * 78)
    L.append(f"  findings described by the radiologist   {len(clauses)}")
    L.append(f"  calculi reported by the model           {len(ours)}")
    L.append("")
    L.append("  NOTE: these counts are not directly comparable. A single report")
    L.append("  clause can describe several calculi (\"a few calculi in the lower")
    L.append("  pole\"), and the summary clause often repeats an earlier finding.")
    L.append("  Pair them by reading, not by subtracting.")
    L.append("")
    if flags:
        L.append("-" * 78)
        L.append("3b. DETECTIONS WE HAVE FLAGGED -- read these before scoring")
        L.append("-" * 78)
        seen = set()
        for f in flags:
            L.append(f"  {f['where']:8s} {str(f['side']):5s} "
                     f"{f['mm']:>7} mm  {f['hu']:>6} HU   {f['flag']}")
            for part in str(f["flag"]).split(";"):
                if part and part not in seen:
                    seen.add(part)
        for part in sorted(seen):
            if part in FLAG_MEANING:
                L.append("")
                L.append(f"  {part}:")
                for line in textwrap.wrap(FLAG_MEANING[part], 70):
                    L.append(f"      {line}")
        L.append("")
    L.append("-" * 78)
    L.append("4. WHAT THE MODEL CONSIDERED AND REJECTED")
    L.append("-" * 78)
    if not audit:
        L.append("  (no audit CSV for this study)")
    for tag, a in audit.items():
        L.append(f"  {tag}: {a['n_candidates']} candidates examined")
        for k, v in sorted(a["by_reason"].items(), key=lambda x: -x[1]):
            L.append(f"      {v:5d}  {k}")
    L.append("")
    if summ:
        L.append("-" * 78)
        L.append("5. PROCESSING NOTES")
        L.append("-" * 78)
        for k, v in summ.items():
            L.append(f"  {k:38s} {v}")
        L.append("")
    L.append("-" * 78)
    L.append("6. KNOWN LIMITS -- please do not score these as model errors")
    L.append("-" * 78)
    L.append("  * BLADDER calculi are not detected. The search covers the kidney")
    L.append("    plus a corridor along the ureter that stops at the bladder")
    L.append("    entrance; the bladder lumen is outside it.")
    L.append("  * URETERIC STENTS: an object that is long, thin and hollow is")
    L.append("    marked 'stent_like' in review_flag and is still REPORTED --")
    L.append("    never removed, because a stone can sit above an obstructing")
    L.append("    stent. The cohort contains one genuine DJ ureteric stent")
    L.append("    (8399313, see stent_in_situ/). Note that 8633709 was ALSO")
    L.append("    filed under stent_in_situ but its report describes a CBD")
    L.append("    (bile duct) stent in the duodenum, not a ureteric one -- it")
    L.append("    is a renal-calculus case in practice.")
    L.append("  * HYDRONEPHROSIS, perinephric fat stranding and stent presence are")
    L.append("    not measured; they appear as '-'.")
    L.append("  * DISTANCE FROM THE UVJ rests on a geometric landmark that has not")
    L.append("    been validated against a radiologist's click, and was measured")
    L.append("    49 mm out on one distended bladder. Prefer the VERTEBRAL LEVEL")
    L.append("    column, which is read off the vertebral masks and matched the")
    L.append("    report exactly on 2 of the 3 cases that state a level.")
    L.append("")
    with open(os.path.join(dest, "COMPARISON.txt"), "w") as f:
        f.write("\n".join(L) + "\n")

    rows = []
    for i, c in enumerate(clauses, 1):
        rows.append({"study_id": sid, "category": case.category,
                     "source": "radiologist", "ref": f"R{i}", "text": c})
    for i, r in enumerate(ours.itertuples(), 1):
        rows.append({"study_id": sid, "category": case.category,
                     "source": "model", "ref": f"M{i}",
                     "text": (f"{r.Organ} | {r.Side} | "
                              f"{getattr(r,'_4','')} mm | "
                              f"{getattr(r,'_5','')} HU | {r.Location}")})
    pd.DataFrame(rows).to_csv(os.path.join(dest, "comparison.csv"), index=False)
    return len(clauses), len(ours)


def copy_if(src, dst):
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="pack directory")
    ap.add_argument("--cases", default=None, help="validation_cases.csv")
    a = ap.parse_args()

    run = paths.ensure()
    cases = pd.read_csv(a.cases or os.path.join(paths.ROOT, CASES_CSV))
    pack = a.out or os.path.join(run, "validation_pack")
    os.makedirs(pack, exist_ok=True)

    ledger = os.path.join(run, "case_ledger.csv")
    status = {}
    if os.path.exists(ledger):
        L = pd.read_csv(ledger)
        status = dict(zip(L.study_id.astype(str), L.status))

    # CATEGORY is the outer folder, STUDY ID the inner one, because a validator
    # works one category at a time: "show me the staghorn cases" should be one
    # directory, not a filter over a flat list. primary/backup is recorded in
    # COMPARISON.txt and INDEX.csv rather than in the folder name, so the study
    # id is the only thing to match against a worklist.
    cases = cases.sort_values(["category", "pick"], ascending=[True, True])
    index = []
    for case in cases.itertuples():
        sid = str(case.study_id)
        cat = slug(case.category)
        name = os.path.join(cat, sid)
        dest = os.path.join(pack, cat, sid)
        os.makedirs(dest, exist_ok=True)

        st = status.get(sid, "not_run")
        if st != "ok":
            with open(os.path.join(dest, "NOT_ANALYSED.txt"), "w") as f:
                f.write(f"study {sid}  ({case.category}, {case.pick})\n"
                        f"status: {st}\n\n"
                        "This study was not analysed. 'no_zip' means the DICOM "
                        "could not be retrieved from the API -- studies older "
                        "than about 33 days are past its retention window.\n\n"
                        f"the report describes:\n{case.finding_reported}\n")
            index.append({"category": case.category, "study_id": sid,
                          "pick": case.pick, "folder": name, "status": st,
                          "n_report_findings": None, "n_model_calculi": None})
            continue

        # A study that was unfinished on an earlier build of this pack left a
        # NOT_ANALYSED.txt behind. Remove it now that real results exist, or the
        # folder contradicts itself.
        stale = os.path.join(dest, "NOT_ANALYSED.txt")
        if os.path.exists(stale):
            os.remove(stale)

        ours = our_findings(run, sid)
        nrep, nours = write_comparison(dest, case, ours,
                                       audit_counts(run, sid),
                                       summary_fields(run, sid),
                                       flagged_rows(run, sid))

        copy_if(os.path.join(run, "reports", f"{sid}_calculi.csv"),
                os.path.join(dest, "our_calculi.csv"))
        copy_if(os.path.join(run, "reports", f"{sid}_report.csv"),
                os.path.join(dest, "our_report.csv"))
        copy_if(os.path.join(run, "reports", f"{sid}_findings.csv"),
                os.path.join(dest, "our_organ_findings.csv"))
        copy_if(os.path.join(run, "mask_overlays", f"{sid}.png"),
                os.path.join(dest, "organ_masks.png"))
        copy_if(os.path.join(run, "overlays", f"{sid}_ureteric.png"),
                os.path.join(dest, "overlays", f"{sid}_ureteric.png"))
        src_ov = os.path.join(run, "overlays", sid)
        if os.path.isdir(src_ov):
            for f in sorted(os.listdir(src_ov)):
                copy_if(os.path.join(src_ov, f),
                        os.path.join(dest, "overlays", f))
        for f in (f"{sid}_candidates.csv", f"{sid}_ureter_candidates.csv",
                  f"{sid}_summary.csv", f"{sid}_ureter_summary.csv"):
            copy_if(os.path.join(run, "csv", "per_study", f),
                    os.path.join(dest, "audit", f))

        index.append({"category": case.category, "study_id": sid,
                      "pick": case.pick, "folder": name, "status": st,
                      "n_report_findings": nrep, "n_model_calculi": nours})
        print(f"  {name:44s} report {nrep} findings, model {nours} calculi")

    idx = pd.DataFrame(index)
    idx.to_csv(os.path.join(pack, "INDEX.csv"), index=False)

    # one flat sheet of every finding from both sides, for tallying
    allrows = []
    for row in index:
        p = os.path.join(pack, row["folder"], "comparison.csv")
        if os.path.exists(p):
            allrows.append(pd.read_csv(p))
    if allrows:
        pd.concat(allrows, ignore_index=True).to_csv(
            os.path.join(pack, "ALL_FINDINGS_report_vs_model.csv"), index=False)

    with open(os.path.join(pack, "README.txt"), "w") as f:
        f.write(textwrap.dedent(f"""\
            VALIDATION PACK -- renal and ureteric calculus detection
            ========================================================

            Nested by category, then by study:

                <category>/<study_id>/

            e.g.  stent_in_situ/8399313/
                  staghorn_calculus/8664459/

            In each study folder, read COMPARISON.txt first. It shows the
            radiologist's findings, then the model's, then what the model
            considered and rejected.

            Also in each folder:
                our_calculi.csv        stone table: organ, side, size, HU, location
                our_report.csv         full structured report incl. impression
                our_organ_findings.csv per-organ table (kidney sizes, bladder volume)
                organ_masks.png        SEGMENTATION QC -- check this first if a
                                       finding looks wrong. If the kidney outline
                                       is wrong, nothing downstream can be right.
                overlays/              coronal MIP, contact sheet, one image per
                                       stone, and the ureteric sheet showing both
                                       accepted and rejected candidates
                audit/                 every candidate and why it was rejected

            At the top level:
                INDEX.csv                          all cases with counts
                ALL_FINDINGS_report_vs_model.csv   every finding from both sides

            KNOWN LIMITS -- please do not score these as errors:
              * bladder calculi are NOT detected (outside the search region)
              * ureteric stents are UNTESTED (no stent case could be retrieved)
              * hydronephrosis, fat stranding and stent presence are not measured
              * "distance from UVJ" rests on an unvalidated landmark; prefer the
                vertebral level

            {len(idx)} cases, {int((idx.status == 'ok').sum())} analysed.
            """))
    print(f"\n{len(idx)} case folders -> {pack}")
    print(f"  INDEX.csv, ALL_FINDINGS_report_vs_model.csv, README.txt")


if __name__ == "__main__":
    main()
