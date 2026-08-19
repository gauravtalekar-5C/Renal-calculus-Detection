"""Read the FULL radiology report and decide, per kidney, whether it describes
a calculus.

WHY THIS REPLACES THE OLD APPROACH
----------------------------------
Scoring used to run against `calculus_line` -- a snippet someone had already
extracted from the report. The snippet regularly lost the information that
mattered:

    "or renal calculi identified."   <- "No ... " had been cut off the front
    "Renal calculi ?"                <- lifted out of the HISTORY section
    "H/O renal calculi 2 years ago"  <- past history read as a current finding

All three were scored "stone present", so the detector was marked wrong for
correctly finding nothing. Reading the full report fixes this at the source:
8493113, 8539655 and 8613169 all say "No calculi" under BOTH kidneys and
"No renal calculi identified" in the impression.

WHAT THE REPORT LOOKS LIKE
--------------------------
`report_content` is flattened HTML -- the tag names survive as bare words
separated by double spaces:

    sectionTitle  HISTORY  p  table  tr  td  p  Abdominal pain  td  p  Renal calculi ?
    sectionTitle  OBSERVATIONS  p  Liver:- Normal in size ...
    p  Right kidney:- Measures 8.3 x 3.6 cm. No calculi / hydronephrosis seen.
    p  Right ureter is normal in course and calibre. No calculus is seen.
    sectionTitle  IMPRESSION  p  ul  li  lic  No renal calculi ... identified bilaterally.

That structure is worth exploiting rather than regexing the whole blob:

  * HISTORY and PROTOCOL are the clinician's question and the scan technique.
    Nothing in them is a finding. They are the source of most of the old
    false positives, and they are dropped outright.
  * Each organ has its own paragraph, so "Right kidney:- ..." gives a
    PER-SIDE label -- and the ureter has a separate paragraph, which is how a
    ureteric stone stops being counted as a renal one.
  * IMPRESSION is the radiologist's summary and is used to confirm.

Per-side labels roughly double the sample size for scoring and make the
left/right agreement check meaningful.

Output: csv/report_labels.csv, one row per study.

Usage:
    ./venv/bin/python utils/parse_reports.py
"""
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# package root -> project root: calculus/<sub>/x.py is two levels down
ROOT = os.path.dirname(os.path.dirname(HERE))
from calculus.common.paths import CSV                             # noqa: E402

XLSX = os.path.join(ROOT, "calculus_with_report.xlsx")

# Sections that never contain a finding. HISTORY is the big one -- it is where
# "Renal calculi ?" and "H/O renal calculi" live.
DROP_SECTIONS = {"HISTORY", "PROTOCOL", "TECHNIQUE", "ADVICE", "KEY IMAGES",
                 "DIFFERENTIAL DIAGNOSIS", "CLINICAL HISTORY", "INDICATION"}

# Flattened-HTML tag names appearing as bare words. Stripped so they cannot be
# mistaken for prose (a stray "p" inside a sentence changes nothing, but "li
# lic" between two clauses would otherwise glue them together).
TAGS = re.compile(r"\s+(?:p|ul|li|lic|ol|table|tr|td|th|br|div|span|section|"
                  r"reportHeader|reportTitle|reportSubTitle|sectionTitle)\s+",
                  re.I)

SECTION_RE = re.compile(r"sectionTitle\s+([A-Z][A-Z &/\-]{2,30})")

# Sentence boundary that does NOT split a decimal: "10.2 mm" must stay whole.
SENT_SPLIT = re.compile(r"(?<!\d)\.(?!\d)|;|\|")

CALC = re.compile(r"calcul|stone|lithiasis|nephrolith|staghorn", re.I)

# Words that make a calculus mention NEGATIVE. Checked within the sentence, so
# "No calculi / hydronephrosis" is negative while "A calculus measuring 4 mm"
# is not.
NEG = re.compile(
    r"\bno\b|\bnot\b|\bnil\b|\bwithout\b|\bfree of\b|\bnegative for\b|"
    r"absent|unremarkable|"
    r"no longer|has passed|resolved|interval passage|"
    r"previously.{0,60}\bnot\b", re.I)

# History/past-tense markers -- a stone two years ago is not a finding today.
PAST = re.compile(r"\bh/?o\b|history of|previously|status post|post[\s-]?op|"
                  r"operated|years? (ago|back)|earlier scan|prior (study|scan)",
                  re.I)

# Structures that are NOT the kidney. Used to keep ureteric/VUJ/bladder and
# gallbladder stones out of the renal label.
NON_RENAL = re.compile(r"ureter|vuj|vesico|bladder|gall\s?bladder|cholelith|"
                       r"urethra|prostate", re.I)
RENAL_WORD = re.compile(r"renal|kidney|calyx|calyce|calyceal|pole|"
                        r"nephrolith|staghorn|pelvicalyceal", re.I)
# Unambiguously INSIDE the kidney. Used to rescue a sentence that names both a
# renal and a non-renal structure, e.g. "calculus in the renal pelvis extending
# into the upper ureter" -- that one IS a renal stone.
# Bare "pelvis" is included because inside a KIDNEY paragraph that is what it
# means. 8513308 reads "a hyperdense calculus measuring approximately
# 20.2 x 12.0 mm in size in the region of the pelvis is extending into the
# upper ureter" -- a renal pelvic stone, which the ureter-exclusion above threw
# away until "pelvis" counted as intrarenal. In whole_findings mode a renal
# word is required as well, so the bony pelvis cannot sneak in this way.
INTRARENAL = re.compile(r"calyx|calyce|calyceal|pole|nephrolith|staghorn|"
                        r"renal pelvis|pelvicalyceal|intrarenal|\bpelvis\b",
                        re.I)

SIZE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:x|×)?\s*(?:\d+(?:\.\d+)?)?\s*"
                  r"(?:x|×)?\s*(?:\d+(?:\.\d+)?)?\s*mm", re.I)


def clean(text):
    """Flattened HTML -> plain prose."""
    t = " " + str(text or "") + " "
    for _ in range(3):                  # tags can sit adjacent: "p  ul  li"
        t = TAGS.sub(" ", t)
    return re.sub(r"\s{2,}", " ", t).strip()


def split_sections(text):
    """{SECTION NAME: prose}. Text before the first title goes to ''."""
    t = str(text or "")
    marks = list(SECTION_RE.finditer(t))
    if not marks:
        return {"": clean(t)}
    out = {"": clean(t[:marks[0].start()])}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(t)
        name = m.group(1).strip()
        # a section title can repeat; keep both halves
        out[name] = (out.get(name, "") + " " + clean(t[m.end():end])).strip()
    return out


def findings_only(text):
    """Everything the radiologist OBSERVED, with history and technique removed."""
    sec = split_sections(text)
    keep = [v for k, v in sec.items() if k.upper() not in DROP_SECTIONS]
    return " ".join(p for p in keep if p)


def sentences(text):
    return [s.strip() for s in SENT_SPLIT.split(text or "") if s and s.strip()]


def kidney_paragraphs(text):
    """{'right': prose, 'left': prose} from the per-organ paragraphs.

    A paragraph runs from "Right kidney:-" to the next organ heading. The
    ureter has its own heading, which is exactly why a ureteric stone does not
    contaminate the renal label.
    """
    t = findings_only(text)
    out = {}
    # organ headings look like "Word(s):-" or "Word(s):"
    heads = list(re.finditer(r"([A-Z][A-Za-z/&' ]{2,40}?)\s*:-?\s", t))
    for i, m in enumerate(heads):
        name = m.group(1).strip().lower()
        end = heads[i + 1].start() if i + 1 < len(heads) else len(t)
        body = t[m.end():end]
        for side in ("right", "left"):
            if re.fullmatch(rf"{side}\s+kidney", name):
                out[side] = (out.get(side, "") + " " + body).strip()
        if re.fullmatch(r"(both\s+)?kidneys", name):
            for side in ("right", "left"):
                out[side] = (out.get(side, "") + " " + body).strip()
    return out


def positive_calculus(text, require_renal=False, exclude_non_renal=True):
    """Does this prose assert a calculus that is present NOW?

    Sentence by sentence, because one paragraph routinely contains both
    "A calculus measuring 4 mm" and "No hydronephrosis".

    exclude_non_renal matters even inside a kidney paragraph. Radiologists
    routinely describe the URETERIC stone under the kidney heading:

        "Right kidney:- Normal in size (measures 8.0 x 4.8 cm) ... An
         obstructive calculus is seen at the right vesicoureteric junction
         measuring 2.8 x 2.5 mm ..."          (8563509)

    Part 1 only searches the kidney, so counting that as a renal stone would
    mark a correct "nothing in the kidney" as a miss. A sentence naming the
    ureter, VUJ or bladder is therefore dropped -- UNLESS it also names a
    definitely-intrarenal structure, which covers phrasings like "calculus in
    the renal pelvis extending into the upper ureter".
    """
    for s in sentences(text):
        if not CALC.search(s):
            continue
        if NEG.search(s) or PAST.search(s):
            continue                     # absent, or a past event
        if exclude_non_renal and NON_RENAL.search(s) and not INTRARENAL.search(s):
            continue                     # ureteric / VUJ / bladder / gallbladder
        if require_renal and not RENAL_WORD.search(s):
            continue
        return True
    return False


def sizes_mm(text):
    out = []
    for s in sentences(text):
        if not CALC.search(s) or NEG.search(s) or PAST.search(s):
            continue
        if NON_RENAL.search(s) and not INTRARENAL.search(s):
            continue                     # that size belongs to a ureteric stone
        out += [float(v) for v in SIZE.findall(s)]
    return out


def label_study(text):
    """One study -> the fields we score against."""
    paras = kidney_paragraphs(text)
    per_side = {side: positive_calculus(body) for side, body in paras.items()}

    if per_side:
        renal = any(per_side.values())
        source = "kidney_paragraph"
    else:
        # no per-organ layout: fall back to the whole findings text, demanding
        # a renal word in the same sentence
        renal = positive_calculus(findings_only(text), require_renal=True)
        source = "whole_findings"

    sec = split_sections(text)
    imp = " ".join(v for k, v in sec.items()
                   if k.upper() in ("IMPRESSION", "CONCLUSION", "DIAGNOSIS"))
    return {
        "report_renal_calculus": bool(renal),
        "renal_right": per_side.get("right"),
        "renal_left": per_side.get("left"),
        "label_source": source,
        "impression_says_renal": positive_calculus(imp, require_renal=True)
                                 if imp else None,
        "renal_sizes_mm": ";".join(
            f"{v:g}" for v in sorted(
                {s for side in paras.values() for s in sizes_mm(side)})),
        "kidney_text": " || ".join(f"{k}: {v[:260]}" for k, v in paras.items()),
    }


def main():
    if not os.path.exists(XLSX):
        sys.exit(f"{XLSX} not found")
    x = pd.read_excel(XLSX).drop_duplicates("study_id", keep="last")
    rows = []
    for r in x.itertuples():
        if not isinstance(r.report_content, str):
            continue
        rows.append({"study_id": r.study_id, **label_study(r.report_content)})
    out = pd.DataFrame(rows)
    os.makedirs(CSV, exist_ok=True)
    path = os.path.join(CSV, "report_labels.csv")
    out.to_csv(path, index=False)
    print(f"wrote {path}  ({len(out)} studies)")
    print(f"\nrenal calculus present : {int(out.report_renal_calculus.sum())}")
    print(f"absent                 : {int((~out.report_renal_calculus).sum())}")
    print(f"\nlabel source:\n{out.label_source.value_counts().to_string()}")
    both = out[out.impression_says_renal.notna()]
    agree = (both.report_renal_calculus == both.impression_says_renal).mean()
    print(f"\nfindings vs impression agree: {100*agree:.0f}%  "
          f"(disagreement = worth a look, not necessarily an error)")


if __name__ == "__main__":
    main()
