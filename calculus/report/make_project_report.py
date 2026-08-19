"""Build the end-to-end project report as a single PDF.

Everything we have done on this project, in one document that can be read away
from a terminal: the problem, the pipeline stage by stage, the results, the
experiments that failed, the figures, and what is still open.

WHERE THE NUMBERS COME FROM
---------------------------
Anything that can be recomputed IS recomputed, here, from the CSVs on disk --
QC verdicts, ureteric detection counts, rejection reasons, cohort sizes. Only
history that no longer exists as a file (the run_v4 comparison, the phantom
tests, the falsified hypotheses) is written as prose. So re-running this after
a new analysis produces a report that matches the analysis, instead of one that
quietly still describes last month's.

Figures are the pipeline's own overlays and renders, not redrawn for the
document.

Usage:
    ./venv/bin/python utils/make_project_report.py
    ./venv/bin/python utils/make_project_report.py --out /tmp/report.pdf
"""
import argparse
import os
import sys

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
                                NextPageTemplate, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
# package root -> project root: calculus/<sub>/x.py is two levels down
ROOT = os.path.dirname(os.path.dirname(HERE))

INK = colors.HexColor("#12181F")
MUTED = colors.HexColor("#5A6875")
ACCENT = colors.HexColor("#0E6F79")
RULE = colors.HexColor("#C8D2DA")
BAND = colors.HexColor("#EEF3F6")
OK = colors.HexColor("#2C7A55")
RISK = colors.HexColor("#A93A31")
WARN = colors.HexColor("#8A5B08")


# ----------------------------------------------------------------- styles ---
def styles():
    s = getSampleStyleSheet()
    base = dict(fontName="Times-Roman", fontSize=10.2, leading=14.6,
                textColor=INK, alignment=TA_JUSTIFY, spaceAfter=7)
    out = {
        "body": ParagraphStyle("body", **base),
        "lead": ParagraphStyle("lead", **{**base, "fontSize": 11.4,
                                          "leading": 16.2, "textColor": MUTED,
                                          "spaceAfter": 11}),
        "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=17,
                             leading=21, textColor=INK, spaceBefore=2,
                             spaceAfter=3),
        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=12.4,
                             leading=16, textColor=INK, spaceBefore=15,
                             spaceAfter=4),
        "h3": ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=10.4,
                             leading=13.5, textColor=ACCENT, spaceBefore=11,
                             spaceAfter=3),
        "kicker": ParagraphStyle("kicker", fontName="Helvetica", fontSize=8,
                                 leading=11, textColor=ACCENT,
                                 spaceAfter=5, tracking=1),
        "cap": ParagraphStyle("cap", fontName="Helvetica-Oblique", fontSize=8.4,
                              leading=11.5, textColor=MUTED, spaceBefore=3,
                              spaceAfter=12),
        "mono": ParagraphStyle("mono", fontName="Courier", fontSize=8.2,
                               leading=11.4, textColor=INK, spaceAfter=8),
        "cell": ParagraphStyle("cell", fontName="Times-Roman", fontSize=8.8,
                               leading=11.6, textColor=INK),
        "cellb": ParagraphStyle("cellb", fontName="Times-Bold", fontSize=8.8,
                                leading=11.6, textColor=INK),
        "cellh": ParagraphStyle("cellh", fontName="Helvetica-Bold",
                                fontSize=7.8, leading=10.4, textColor=MUTED),
        "title": ParagraphStyle("title", fontName="Times-Bold", fontSize=27,
                                leading=31, textColor=INK, spaceAfter=12),
        "sub": ParagraphStyle("sub", fontName="Times-Italic", fontSize=12.6,
                              leading=17, textColor=MUTED, spaceAfter=20),
        "tocline": ParagraphStyle("tocline", fontName="Times-Roman",
                                  fontSize=10, leading=16, textColor=INK),
    }
    return out


ST = styles()


def P(txt, style="body"):
    return Paragraph(txt, ST[style])


def H(txt, level=2):
    return Paragraph(txt, ST[f"h{level}"])


def bullets(items, style="body"):
    """Bulleted list as one paragraph per item -- keeps reportlab simple and
    lets each bullet carry inline markup."""
    return [Paragraph(f"<bullet>&bull;</bullet>{t}", ST[style])
            for t in items]


def table(rows, widths, header=True, align_right=(), font_size=8.8):
    """rows: list of lists of strings (markup allowed)."""
    data = []
    for i, row in enumerate(rows):
        st = "cellh" if (header and i == 0) else "cell"
        data.append([Paragraph(str(c), ST[st]) for c in row])
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), BAND),
                  ("LINEBELOW", (0, 0), (-1, 0), 0.9, MUTED)]
    for c in align_right:
        style.append(("ALIGN", (c, 0), (c, -1), "RIGHT"))
    t.setStyle(TableStyle(style))
    return t


def callout(title, body, tone=ACCENT):
    """A boxed note. Used sparingly, for things that cost us time to learn."""
    inner = [Paragraph(f'<font color="{tone.hexval()}" size="7.6">'
                       f'<b>{title.upper()}</b></font>', ST["cell"]),
             Spacer(1, 3),
             Paragraph(body, ST["cell"])]
    t = Table([[inner]], colWidths=[165 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BAND),
        ("LINEBEFORE", (0, 0), (0, -1), 2.2, tone),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
    ]))
    return [Spacer(1, 4), t, Spacer(1, 10)]


def figure(path, caption, max_w=165 * mm, max_h=195 * mm):
    """Embed a PNG, scaled to fit, with its caption. Missing files are skipped
    with a visible note rather than silently -- a report that quietly drops a
    figure looks complete when it is not."""
    if not os.path.exists(path):
        return [P(f'<font color="{RISK.hexval()}">[figure missing: '
                  f'{os.path.basename(path)}]</font>', "cap")]
    w, h = PILImage.open(path).size
    scale = min(max_w / w, max_h / h)
    img = Image(path, width=w * scale, height=h * scale)
    img.hAlign = "CENTER"
    return [img, P(caption, "cap")]


# ------------------------------------------------------------------ data ----
def gather():
    """Every number the report can compute from files on disk."""
    d = {}

    def rd(p):
        return pd.read_csv(p) if os.path.exists(p) else None

    d["qc"] = rd(os.path.join(ROOT, "run_v5", "csv", "kidney_qc.csv"))
    d["stones"] = rd(os.path.join(ROOT, "run_v5", "csv", "baseline_stones.csv"))
    d["cand"] = rd(os.path.join(ROOT, "run_v5", "csv", "candidates.csv"))
    d["ucand"] = rd(os.path.join(ROOT, "run_ureter", "csv",
                                 "ureter_candidates.csv"))
    d["usumm"] = rd(os.path.join(ROOT, "run_ureter", "csv",
                                 "ureter_summary.csv"))
    d["gt"] = rd(os.path.join(ROOT, "csv", "ureteric_stone_studies.csv"))
    d["triage"] = rd(os.path.join(ROOT, "csv", "triage_study.csv"))
    d["n_nifti"] = len([f for f in os.listdir(os.path.join(ROOT, "nifti"))
                        if f.endswith(".nii.gz")]) \
        if os.path.isdir(os.path.join(ROOT, "nifti")) else 0
    d["n_seg"] = len(os.listdir(os.path.join(ROOT, "seg"))) \
        if os.path.isdir(os.path.join(ROOT, "seg")) else 0
    return d


def ureteric_stats(d):
    """Side agreement and detection counts for the 37-study validation."""
    out = {}
    c, s, g = d["ucand"], d["usumm"], d["gt"]
    if c is None or s is None:
        return out
    c = c.copy()
    c["study_id"] = c.study_id.astype(str)
    s = s.copy()
    s["study_id"] = s.study_id.astype(str)
    acc = c[c.is_stone.astype(bool)]
    out["n_studies"] = len(s)
    out["n_accepted"] = len(acc)
    out["median_per_study"] = float(s.n_stones.median())
    out["max_per_study"] = int(s.n_stones.max())
    out["zero"] = s[s.n_stones == 0].study_id.tolist()
    sides = acc.groupby("study_id").side.nunique()
    out["bilateral"] = int((sides == 2).sum())
    out["unilateral"] = int((sides == 1).sum())
    out["zones"] = acc.zone.value_counts().to_dict()
    out["reasons"] = c[~c.is_stone.astype(bool)].reject_reason \
        .value_counts().head(6).to_dict()
    out["hu_median"] = float(acc.hu_max.median())
    out["hu_p10"] = float(acc.hu_max.quantile(.10))
    out["offpath_median"] = float(acc.off_path_mm.median())

    if g is not None:
        g = g.copy()
        g["study_id"] = g.study_id.astype(str)
        gt = g[g.study_id.isin(s.study_id)].set_index("study_id").sentence \
            .to_dict()

        def side_of(t):
            t = str(t).lower()
            l, r = "left" in t, "right" in t
            return "left" if l and not r else "right" if r and not l else "?"

        known = {k: side_of(v) for k, v in gt.items()}
        known = {k: v for k, v in known.items() if v != "?"}
        rep = acc[acc.report_this.astype(bool)]
        hit = sum(1 for k, v in known.items()
                  if v in set(rep[rep.study_id == k].side))
        out["side_hit"], out["side_n"] = hit, len(known)
        out["gt"] = gt
    return out


# ----------------------------------------------------------------- pages ----
def on_page(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(22 * mm, 16 * mm, 188 * mm, 16 * mm)
    canvas.setFont("Helvetica", 7.4)
    canvas.setFillColor(MUTED)
    canvas.drawString(22 * mm, 11 * mm,
                      "Renal and ureteric calculus detection  ·  5C Network")
    canvas.drawRightString(188 * mm, 11 * mm, f"{doc.page}")
    canvas.restoreState()


def on_title(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(ACCENT)
    canvas.rect(0, 272 * mm, 210 * mm, 25 * mm, fill=1, stroke=0)
    canvas.restoreState()


def build(out_path, d, u):
    doc = BaseDocTemplate(out_path, pagesize=A4,
                          leftMargin=22 * mm, rightMargin=22 * mm,
                          topMargin=22 * mm, bottomMargin=22 * mm,
                          title="Renal and Ureteric Calculus Detection",
                          author="5C Network")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                  id="f")
    doc.addPageTemplates([
        PageTemplate(id="title", frames=[frame], onPage=on_title),
        PageTemplate(id="body", frames=[frame], onPage=on_page),
    ])
    S = []

    # ---------------------------------------------------------- title page --
    S += [Spacer(1, 46 * mm),
          P("Renal and ureteric calculus detection "
            "and measurement on non-contrast CT", "title"),
          P("What the system does, how each decision is made, what has been "
            "validated, and what has not.", "sub")]
    meta = [
        ["Cohort", f"{d['n_nifti']} volumes extracted, {d['n_seg']} segmented"],
        ["Part 1 &mdash; kidney", "137 studies, 120 scored against reports"],
        ["Part 2 &mdash; ureteric", f"{u.get('n_studies', 0)} report-confirmed "
                                    f"studies validated"],
        ["Working directory", "<font face='Courier' size='8'>"
                              "/root/Gaurav/kindey_calculus_measurement</font>"],
        ["Report generated", pd.Timestamp.now().strftime("%d %B %Y")],
    ]
    S += [table([[a, b] for a, b in meta], [42 * mm, 123 * mm], header=False),
          Spacer(1, 14 * mm)]
    S += callout(
        "In one paragraph",
        "A plain CT shows a calculus as the brightest thing in a region that "
        "should be dark, so detection is a thresholding problem rather than a "
        "recognition problem. Almost all of the difficulty is in establishing "
        "<i>where you are</i>: the same 400&nbsp;HU dot is a stone in the renal "
        "pelvis, a phlebolith in the pelvic sidewall, and cortical bone on the "
        "sacrum. Nine of the eleven pipeline stages exist to establish location; "
        "only two are about density. Stones inside the kidney are detected and "
        "measured reliably. Stones in the ureter are found on the right side but "
        "over-counted, and every distance along the ureter still rests on a "
        "landmark nobody has checked.")
    S += [NextPageTemplate("body"), PageBreak()]

    # ---------------------------------------------------------- contents ----
    S += [H("Contents", 1), Spacer(1, 4)]
    toc = [
        ("1", "The problem, and why it is built this way"),
        ("2", "The data: cohort and filters"),
        ("3", "The pipeline, stage by stage"),
        ("4", "Part 1 &mdash; stones inside the kidney"),
        ("5", "Measurement accuracy"),
        ("6", "Kidney segmentation QC"),
        ("7", "Part 2 &mdash; stones in the ureter"),
        ("8", "The 37-study ureteric validation"),
        ("9", "Experiments, including the ones that failed"),
        ("10", "Figures: what the outputs look like"),
        ("11", "Where the project stands, and what is next"),
    ]
    for n, t in toc:
        S.append(Paragraph(
            f'<font face="Helvetica-Bold" color="{ACCENT.hexval()}">{n}'
            f'</font>&nbsp;&nbsp;&nbsp;{t}', ST["tocline"]))
    S += [Spacer(1, 8)]
    S += callout(
        "How to read the claims in this report",
        "<b>Validated</b> means measured against ground truth with a number "
        "attached. <b>Fitted</b> means tuned on a small sample &mdash; plausible, "
        "and the exact way a threshold was once moved the wrong way in this "
        "project. <b>Unvalidated</b> means no ground truth exists yet, so the "
        "error is unbounded rather than merely unknown. Each major claim below "
        "is labelled.", MUTED)

    # ------------------------------------------------------------ 1 problem --
    S += [PageBreak(), P("SECTION 1", "kicker"),
          H("The problem, and why it is built this way", 1),
          P("A radiologist reading a non-contrast CT for calculi answers a "
            "fixed set of questions: is there a stone, how many, how big, how "
            "dense, where exactly, and is anything obstructed. The brief is to "
            "answer those automatically &mdash; count, volume, maximum "
            "diameter, HU, calyceal location, and for ureteric stones the "
            "distance from the vesico-ureteric junction, plus perinephric fat "
            "stranding.", "lead")]
    S += [H("Two facts shape every design decision", 3)]
    S += bullets([
        "<b>Nothing is enhanced.</b> On a plain CT a stone is simply the "
        "brightest thing in a region that should be dark. That makes detection "
        "a thresholding problem, not a recognition problem &mdash; which is why "
        "this pipeline is mostly classical image processing and uses a neural "
        "network only to find the organs.",
        "<b>Everything hinges on knowing where you are.</b> A 400&nbsp;HU dot "
        "is a calculus in the renal pelvis, a phlebolith in the pelvic "
        "sidewall, and cortical bone on the sacrum. The pipeline spends most of "
        "its effort establishing location, then applies a comparatively simple "
        "density test.",
    ])
    S += [H("Why no off-the-shelf model was used", 3),
          P("Every published renal-calculus model we could source is licensed "
            "for non-commercial use only, which rules it out for 5C regardless "
            "of quality:")]
    S += [table([
        ["Source", "Blocker"],
        ["Elton et al. <font face='Courier' size='8'>rsummers11/Renal-Calculi"
         "</font>", "Non-commercial licence, and the weights were never "
                    "released at all"],
        ["AbdomenAtlas / AtlasNet", "CC BY-NC 4.0"],
        ["Zenodo 6042410 (Diagnostics 2022)", "Restricted access, no licence "
                                              "stated, 2.6&nbsp;TB, requires a "
                                              "stated academic use case"],
        ["<font face='Courier' size='8'>mmhoan/ureter_segmentation</font>",
         "LICENSE says MIT, README says academic use only; repository contains "
         "no code"],
    ], [58 * mm, 107 * mm]), Spacer(1, 6)]
    S += [P("A published ureter segmentation network was obtained and run "
            "anyway, to test the idea rather than to ship it. It failed for a "
            "structural reason described in section 9. The pipeline below is "
            "what was built instead.")]

    # -------------------------------------------------------------- 2 data --
    S += [PageBreak(), P("SECTION 2", "kicker"),
          H("The data: cohort and filters", 1),
          P("Studies are selected from report text, not from image content. "
            "Reports mentioning a calculus give positives; reports explicitly "
            "stating none give true negatives. Both are needed &mdash; a cohort "
            "of positives can measure sensitivity and nothing else.", "lead")]
    S += [table([
        ["Stage", "Count", "What removes studies here"],
        ["Report spreadsheet", "24,774", "&mdash;"],
        ["Requested for download", "200", "Sampling"],
        ["DICOM zips retrieved", "169", "31 past the API's 33-day retention "
                                        "cliff"],
        ["NIfTI extracted", f"{d['n_nifti']}", "Series triage found nothing "
                                               "measurable"],
        ["Segmented", f"{d['n_seg']}", "Paediatric age gate"],
        ["Scored against reports", "120", "Mask QC and contrast rejection"],
    ], [52 * mm, 22 * mm, 91 * mm], align_right=(1,)), Spacer(1, 8)]
    S += callout(
        "A negative in this cohort is not a healthy control",
        "About <b>31% of the 52 negatives have a stone elsewhere</b> in the "
        "tract &mdash; ureter, VUJ or bladder. That makes the negative set "
        "adversarially hard rather than easy, and specificity measured against "
        "it is a harsher number than it would be against healthy scans.")
    S += [P("PHI handling is fixed by two rules. Downloads are named by "
            "<font face='Courier' size='8'>study_id</font> and never by the "
            "server's <font face='Courier' size='8'>Content-Disposition</font> "
            "header, which carries the patient name. DICOM zips are never sent "
            "to annotators; the derived NIfTI files are safe to send, verified "
            "field by field &mdash; "
            "<font face='Courier' size='8'>descrip</font>, "
            "<font face='Courier' size='8'>aux_file</font>, "
            "<font face='Courier' size='8'>db_name</font> and "
            "<font face='Courier' size='8'>intent_name</font> are all empty.")]

    # ---------------------------------------------------------- 3 pipeline --
    S += [PageBreak(), P("SECTION 3", "kicker"),
          H("The pipeline, stage by stage", 1),
          P("Eleven stages, each a script in "
            "<font face='Courier' size='8'>utils/</font>. Every one is "
            "resumable and writes its own CSV, so a failure never costs more "
            "than the study it happened on.", "lead")]
    stages = [
        ("01", "build_worklist.py", "Choose studies from report text",
         "Positives and explicit negatives, so both sensitivity and "
         "specificity are measurable."),
        ("02", "triage_series.py", "Pick the one series worth measuring",
         "A study zip holds 3&ndash;40 series. Exactly one should be measured, "
         "and choosing wrong corrupts everything downstream silently."),
        ("03", "patient_gate.py", "Exclude paediatric studies",
         "Before any GPU time. The adult model returns fragmented ~30&nbsp;mL "
         "kidneys on a child and there is no paediatric task to switch to."),
        ("04", "extract_series.py", "DICOM &rarr; Hounsfield-unit volume",
         "Per-slice rescale slope and intercept. Orientation verified: "
         "axis0&rarr;LEFT, axis1&rarr;POSTERIOR, axis2&rarr;SUPERIOR."),
        ("05", "run_anatomy.py", "TotalSegmentator, 14 ROIs",
         "The only learned component, and it segments organs, not stones."),
        ("06", "kidney_qc.py", "Decide whether the mask can be trusted",
         "A stone count from a broken mask is worse than no count, because it "
         "looks like a result."),
        ("07", "detect_stones.py", "PART 1 &mdash; stones inside the kidney",
         "Hysteresis thresholding in a closed kidney ROI, then sub-voxel "
         "measurement."),
        ("08", "detect_ureteric.py", "PART 2 &mdash; stones in the ureter",
         "An anatomical corridor plus an eight-test rejection chain."),
        ("09", "render_overlays.py, render_ureteric_overlays.py",
         "Draw every detection back onto its slice",
         "A CSV cannot distinguish a phlebolith from a calculus. A slice can."),
        ("10", "render_kidney_3d.py, dicom_to_3d.py", "3D surfaces and STLs",
         "Answers whether a mask has the right shape, which no single slice "
         "shows."),
        ("11", "compare_reports.py, score_run.py", "Score against the report",
         "Sensitivity, specificity and size agreement."),
    ]
    rows = [["#", "Script", "What it does", "Why it exists"]]
    for n, script, what, why in stages:
        rows.append([f"<b>{n}</b>",
                     f"<font face='Courier' size='7.4'>{script}</font>",
                     f"<b>{what}</b>", why])
    S += [table(rows, [8 * mm, 38 * mm, 48 * mm, 71 * mm]), Spacer(1, 8)]

    S += [H("Stage 2 in detail: series triage", 2),
          P("This stage produced the two largest single corrections in the "
            "project, so it is worth understanding. Every series is scored and "
            "ranked worst-preferred-last:")]
    S += [table([
        ["Rank", "Verdict", "Meaning"],
        ["0", "<font face='Courier' size='8'>measurable</font>",
         "&le;1.5&nbsp;mm slices, adequate coverage &mdash; full measurement"],
        ["1", "<font face='Courier' size='8'>thin_short_coverage</font>",
         "Thin, but the kidneys may be clipped"],
        ["2", "<font face='Courier' size='8'>detect_only</font>",
         "&le;3&nbsp;mm &mdash; count stones, do not trust sizes"],
        ["3", "<font face='Courier' size='8'>bone_kernel</font>",
         "Usable, never preferable &mdash; sharp kernels inflate calculi"],
        ["4", "<font face='Courier' size='8'>too_thick</font>", "&gt;3&nbsp;mm"],
        ["5", "<font face='Courier' size='8'>non_abdominal</font>",
         "Chest, head or neck by name"],
        ["6&ndash;7", "<font face='Courier' size='8'>contrast &middot; skip"
                      "</font>", "Enhanced, derived, or too few slices"],
    ], [16 * mm, 46 * mm, 103 * mm], align_right=(0,))]
    S += [P("Region is tested <i>before</i> thickness, deliberately: a "
            "1&nbsp;mm chest scan is worse than a 3&nbsp;mm abdominal one, "
            "because no amount of resolution recovers a kidney outside the "
            "field of view. Ties break on thinnest slice first, then on slice "
            "count.")]
    S += callout(
        "What this stage caught",
        "Before the region and kernel rules existed, study <b>8283706</b> was "
        "measured on a <font face='Courier' size='8'>Thorax HRCT</font> because "
        "it was 1.0&nbsp;mm and therefore ranked above the 1.5&nbsp;mm "
        "abdominal series. The kidneys were clipped at slice 0 and volume read "
        "89&nbsp;mL; on the correct series it reads <b>191&nbsp;mL</b>. Study "
        "<b>8589526</b> was measured on a "
        "<font face='Courier' size='8'>BONE THIN</font> reconstruction and "
        "reported a <b>103&nbsp;mm</b> calculus; on the soft-kernel series the "
        "same stones measure <b>4.30 and 2.44&nbsp;mm</b> against a report of "
        "4.7 and 2.2.", WARN)

    # ------------------------------------------------------------ 4 part 1 --
    S += [PageBreak(), P("SECTION 4", "kicker"),
          H("Part 1 &mdash; stones inside the kidney", 1),
          P("Validated on 120 studies with report-derived labels: 68 positive, "
            "52 negative.", "lead")]
    S += [table([
        ["Metric", "run_v4", "run_v5", "95% CI"],
        ["Sensitivity", "94.1%", f"<b>95.6%</b> (65/68)", "88&ndash;98"],
        ["Specificity", "76.9%", f"<b>82.7%</b> (43/52)", "70&ndash;91"],
        ["Youden index", "71.0", "<b>78.3</b>", "&mdash;"],
        ["False positives", "12", "<b>9</b>", "&mdash;"],
        ["False negatives", "4", "<b>3</b>", "&mdash;"],
    ], [45 * mm, 30 * mm, 50 * mm, 40 * mm], align_right=(1, 2, 3)),
        Spacer(1, 6)]
    S += [P("For orientation only: Elton et al. report 88% / 91% externally, a "
            "Youden of 79. Ours is 78.3, but on a different cohort with a "
            "deliberately hard negative set, so the numbers are not "
            "comparable.")]
    S += [H("How a candidate becomes a stone", 2),
          Paragraph(
              "ROI&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;kidney mask CLOSED by 15 mm + "
              "3 mm capsule cuff<br/>"
              "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(closing fills "
              "the renal sinus, where the collecting system and<br/>"
              "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;therefore "
              "most stones sit; plain dilation swallows fat instead)<br/><br/>"
              "detect&nbsp;&nbsp;clip [-200, 1000] -&gt; anisotropic diffusion "
              "until components &lt; 200<br/>"
              "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;grow at 130 HU "
              "on the DENOISED volume&nbsp;&nbsp;(extent)<br/>"
              "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;keep only "
              "components whose RAW peak reaches 200 HU&nbsp;&nbsp;(existence)"
              "<br/><br/>"
              "reject&nbsp;&nbsp;bone&nbsp;&nbsp;&nbsp;&nbsp;dense component "
              "&gt; 3000 mm3, or an anatomical bone mask<br/>"
              "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;vessel&nbsp;&nbsp;"
              "centre within 3 mm of an artery lumen<br/>"
              "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;size&nbsp;&nbsp;&nbsp;&nbsp;"
              "&lt; 1.5 mm, or &gt; 30 mm flagged for review<br/><br/>"
              "measure&nbsp;on the ORIGINAL volume: FWHM boundary, marching-"
              "cubes surface,<br/>"
              "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;convex-hull max "
              "caliper, SVD principal axes, partial-volume integral",
              ST["mono"])]
    S += [P("The hysteresis split matters. Denoising pulls peaks down by tens "
            "of HU, so seeding on the filtered copy would silently drop "
            "borderline stones; growing on the raw copy would let noise "
            "wander. Extent comes from the smoothed volume, existence from the "
            "original.")]
    if d["stones"] is not None:
        st = d["stones"]
        S += [P(f"Current Part 1 output: <b>{len(st)} accepted stones</b> "
                f"across <b>{st.study_id.nunique()} studies</b>, largest "
                f"{st.max_diameter_mm.max():.1f}&nbsp;mm, median "
                f"{st.max_diameter_mm.median():.1f}&nbsp;mm.")]

    # ------------------------------------------------------- 5 measurement --
    S += [PageBreak(), P("SECTION 5", "kicker"),
          H("Measurement accuracy", 1),
          P("Sizes do not come from counting voxels above a threshold &mdash; "
            "that quantises a 3&nbsp;mm stone into a number that jumps by half "
            "a millimetre per voxel. They come from a full-width-half-maximum "
            "boundary against local background, a marching-cubes sub-voxel "
            "surface, a convex-hull maximum caliper, and SVD principal axes.",
            "lead")]
    S += [H("Against phantoms, where ground truth is exact", 3),
          table([
        ["Voxel size", "Diameter error", "Volume error"],
        ["0.7&nbsp;mm isotropic", "<b>0.11&nbsp;mm</b>", "6%"],
        ["0.8 &times; 0.8 &times; 1.25&nbsp;mm", "<b>0.17&nbsp;mm</b>", "6%"],
        ["3&nbsp;mm slices", "0.55&nbsp;mm", "<b>20%</b>"],
        ["Stones &le;3&nbsp;mm (any spacing)", "&mdash;", "<b>16%</b>"],
    ], [62 * mm, 52 * mm, 51 * mm], align_right=(1, 2)), Spacer(1, 4)]
    S += [P("Three-axis output survives rotation: major 0.49, intermediate "
            "0.20, minor 0.07&nbsp;mm. The degradation at 3&nbsp;mm is not an "
            "algorithmic weakness &mdash; no method recovers information the "
            "acquisition did not record. It is why triage demotes thick series "
            "to <font face='Courier' size='8'>detect_only</font> rather than "
            "pretending to measure them.")]
    S += [H("Against radiologist reports, which is harder", 3),
          P("Median absolute error is <b>1.2&ndash;1.5&nbsp;mm</b> across "
            "39&ndash;47 studies. Three things explain the gap from the phantom "
            "figures, and only one of them is ours:")]
    S += bullets([
        "<b>Axis convention (ours).</b> Reports quote <i>in-plane</i> axes; we "
        "quoted the 3D maximum caliper. Switching to like-for-like moves the "
        "bias from <b>+1.81&nbsp;mm to &minus;0.02&nbsp;mm</b>, and costs "
        "nothing because those dimensions are already computed.",
        "<b>The radiologist's own caliper placement and rounding</b>, which is "
        "not an error in either direction so much as a different measurement.",
        "<b>Per-stone ambiguity.</b> Management-category agreement currently "
        "reads 51%, but that compares <i>largest measured</i> against "
        "<i>largest reported</i> per study &mdash; in a multi-stone kidney "
        "those are often different stones. Treat 51% as unmeasured, not as 51%.",
    ])
    S += callout(
        "The honest summary of measurement error",
        "For a study acquired at 1&nbsp;mm or thinner, diameter error is "
        "sub-millimetre and the number can be reported. At 3&nbsp;mm, diameter "
        "is still usable and volume is not. Below 3&nbsp;mm stone size, partial "
        "volume dominates and cannot be engineered away &mdash; which matters "
        "clinically only because the 5&nbsp;mm threshold for spontaneous "
        "passage sits just above that range.")

    # ---------------------------------------------------------------- 6 QC --
    S += [PageBreak(), P("SECTION 6", "kicker"),
          H("Kidney segmentation QC", 1),
          P("Every study gets one of six verdicts, from per-side volume, "
            "craniocaudal length, left/right asymmetry, median parenchymal HU, "
            "and whether the kidney touches the first or last slice.", "lead")]
    S += [table([
        ["Test", "Review", "Fail", "Reasoning"],
        ["Single-kidney volume", "&lt;70&nbsp;mL", "&lt;40&nbsp;mL",
         "Anatomical floor, not a percentile of our own data"],
        ["Craniocaudal length", "&lt;80&nbsp;mm", "&lt;65&nbsp;mm",
         "Catches a mask that found only one pole"],
        ["L/R asymmetry", "&gt;1.8&times;", "&gt;2.5&times;",
         "The signature of a one-sided leak into liver or spleen"],
        ["Median parenchymal HU", "&mdash;", "&gt;70",
         "An enhanced study that slipped past triage"],
        ["Slices beyond the kidney", "&mdash;", "&le;1",
         "<font face='Courier' size='8'>cannot_assess</font> &mdash; the organ "
         "is cut off"],
    ], [40 * mm, 22 * mm, 22 * mm, 81 * mm], align_right=(1, 2)), Spacer(1, 6)]
    if d["qc"] is not None and "verdict" in d["qc"].columns:
        vc = d["qc"].verdict.value_counts()
        rows = [["Verdict", "n", "Meaning"]]
        meaning = {
            "ok": "Usable",
            "review": "Small but plausible &mdash; a human should look",
            "fail": "The scan is fine and the mask is genuinely truncated",
            "contrast": "Wrong phase &mdash; the mask is usually good",
            "cannot_assess": "The scan cannot answer the question",
        }
        for k in ("ok", "review", "fail", "contrast", "cannot_assess"):
            if k in vc:
                rows.append([f"<font face='Courier' size='8'>{k}</font>",
                             f"<b>{vc[k]}</b>", meaning.get(k, "")])
        S += [table(rows, [38 * mm, 16 * mm, 111 * mm], align_right=(1,)),
              Spacer(1, 6)]
    S += callout(
        "A calibration mistake worth remembering",
        "The original floor was 90&nbsp;mL, taken from a textbook "
        "<i>whole-kidney</i> range. But TotalSegmentator's class is "
        "<b>parenchyma only</b> &mdash; no collecting system, no sinus fat. "
        "Against our cohort median of 113&nbsp;mL that floor sat at our own "
        "25th percentile, flagging a quarter of all studies by construction: 59 "
        "of 137 landed in a review queue where only 13 had a real problem. "
        "Equally, the new thresholds are deliberately <i>not</i> percentiles of "
        "our own distribution &mdash; our 5th percentile is 40&nbsp;mL "
        "<i>because</i> the bad masks are in the sample, so fitting to it would "
        "define the failures as normal.", WARN)
    S += [H("The 'bad masks' premise was mostly wrong", 3),
          P("Thirteen studies were failing QC and the working assumption was "
            "that TotalSegmentator was at fault. On inspection:")]
    S += [table([
        ["n", "Cause", "Implication"],
        ["3", "The scan does not contain the kidney &mdash; outside the field "
              "of view, or clipped at the first or last slice",
         "No model fixes this. Now reported as "
         "<font face='Courier' size='8'>cannot_assess</font>."],
        ["2", "Paediatric, ages 7 and 18 &mdash; age-gate escapes",
         "Masks look plausible; the adult volume threshold is what fails."],
        ["&ge;2", "Genuinely abnormal kidneys &mdash; obstructive atrophy, "
                  "hydronephrosis", "The mask is right. The kidney is small."],
        ["~5", "Unexplained", "Needs a human eye on the 3D renders."],
    ], [12 * mm, 76 * mm, 77 * mm], align_right=(0,)), Spacer(1, 6)]
    S += [P("The distinction between <font face='Courier' size='8'>fail</font> "
            "and <font face='Courier' size='8'>cannot_assess</font> matters "
            "more than it looks. Only <b>fail</b> is a reason to improve "
            "segmentation. And in production, "
            "<font face='Courier' size='8'>cannot_assess</font> must never be "
            "reported as 'no calculi' &mdash; a pelvis-only scan currently "
            "yields zero stones, which reads as a normal negative when the "
            "truth is that nobody looked.")]

    # ------------------------------------------------------------ 7 part 2 --
    S += [PageBreak(), P("SECTION 7", "kicker"),
          H("Part 2 &mdash; stones in the ureter", 1),
          P("The ureter is invisible on plain CT: a 3&ndash;5&nbsp;mm "
            "soft-tissue tube running through soft tissue. It cannot be "
            "segmented, so instead the pipeline constructs the corridor it must "
            "lie in, from organs that can be segmented.", "lead")]
    S += [H("The corridor", 3)]
    S += bullets([
        "<b>PUJ</b> &mdash; the medial-inferior point of the kidney mask, where "
        "the ureter leaves.",
        "<b>Iliac crossing</b> &mdash; from the iliac artery mask at the pelvic "
        "brim, nudged forward.",
        "<b>UVJ</b> &mdash; the postero-lateral corner of the bladder base on "
        "that side, where the trigone sits.",
    ])
    S += [P("These three are interpolated into a smooth curve (400 points, "
            "Gaussian smoothing) and inflated into a 20&nbsp;mm tube. The "
            "iliac waypoint is what makes it a curve: a straight "
            "kidney-to-bladder line, the obvious first attempt, cuts through "
            "bowel and iliac vessels and produced roughly <b>70 spurious "
            "detections per scan</b>.")]
    S += [H("The eight tests a candidate must survive", 2),
          P("The first test a candidate fails becomes its recorded rejection "
            "reason, so every rejection is auditable.")]
    S += [table([
        ["#", "Test", "Constant", "Why"],
        ["1", "Inside the corridor, outside the kidney", "20&nbsp;mm radius",
         "Kidney+2 voxels is Part 1's territory"],
        ["2", "Outside bone", "no margin",
         "Trabecular bone is 130&ndash;350&nbsp;HU. Subtracted without a margin "
         "on purpose, so a stone against the sacrum survives"],
        ["3", "Reaches 130&nbsp;HU denoised", "GROW_HU", "Defines extent"],
        ["4", "Reaches 200&nbsp;HU raw", "SEED_HU", "Seed test"],
        ["5", "Reaches 300&nbsp;HU raw", "HU_FLOOR", "The density cut"],
        ["6", "Not mostly bone; &ge;3&nbsp;mm from an artery",
         "bone_frac &le;0.5", "Vascular calcification is the main false "
                              "positive"],
        ["7", "Diameter 1.0&ndash;22&nbsp;mm", "MAX_DIAM",
         "A ureter is 3&ndash;5&nbsp;mm wide; 30&nbsp;mm fits a renal pelvis"],
        ["8", "Fewer than 2 phlebolith cues", "2 of 4",
         "Fatty rim, off-path, lucent centre, roundness"],
    ], [8 * mm, 52 * mm, 27 * mm, 78 * mm], align_right=(0,)), Spacer(1, 6)]
    S += [P("Survivors are ranked by peak HU per side and the top two marked "
            "for report. A composite score using log-volume and off-path "
            "distance was tried and ranked <i>worse</i> than plain peak HU, so "
            "the extra terms were dropped as noise.")]
    S += callout(
        "Two hard ceilings, neither fixable by tuning",
        "<b>130&nbsp;HU.</b> The growth threshold is a floor on what can ever "
        "become a candidate. Two reported ureteric stones in this cohort are at "
        "106 and 129&nbsp;HU, so they are unreachable whatever any later test "
        "does &mdash; roughly <b>11% of ureteric stones</b>. This is confirmed, "
        "not estimated: the 129&nbsp;HU case is one of only two complete misses "
        "in the validation run.<br/><br/>"
        "<b>Slice thickness.</b> At 3&nbsp;mm, measurement error is 0.55&nbsp;mm "
        "and 20% by volume against 0.11&nbsp;mm and 6% at 0.7&nbsp;mm.", RISK)

    # ------------------------------------------------------- 8 validation ----
    S += [PageBreak(), P("SECTION 8", "kicker"),
          H("The 37-study ureteric validation", 1)]
    if u:
        S += [P(f"The pilot used five report-confirmed stones and looked "
                f"excellent: 5 of 5 found, correct side every time, mean "
                f"absolute size error 0.73&nbsp;mm, and the 300&nbsp;HU floor "
                f"cutting false positives from 44 per study to 1. "
                f"<b>{u['n_studies']} report-confirmed studies were then run to "
                f"check it.</b> The result is split.", "lead")]
        S += [H("What held", 3)]
        S += bullets([
            f"<b>Side is right.</b> In {u['side_hit']} of {u['side_n']} "
            f"studies whose report names a side, a detection is on that side "
            f"&mdash; {100*u['side_hit']/max(1,u['side_n']):.0f}%.",
            f"<b>Almost nothing is missed entirely.</b> Only "
            f"{len(u['zero'])} of {u['n_studies']} studies produced no "
            f"detection at all.",
            "<b>The rejection chain works.</b> Bone rejected "
            f"{u['reasons'].get('bone', 0):,} candidates and the HU floor "
            f"{u['reasons'].get('below_hu_floor', 0):,} more.",
        ])
        S += [H("What did not hold", 3)]
        S += bullets([
            f"<b>The count is wrong.</b> {u['n_accepted']} stones were "
            f"accepted across {u['n_studies']} studies whose reports describe "
            f"roughly one each &mdash; a median of {u['median_per_study']:.0f} "
            f"per study, maximum {u['max_per_study']}.",
            f"<b>{u['bilateral']} of "
            f"{u['bilateral'] + u['unilateral']} studies fire on both "
            f"ureters</b>, where the report names one. Bilateral ureteric "
            "calculi are uncommon, so most of those second-side detections "
            "are wrong.",
            f"<b>The survivors cluster near the floor.</b> Median peak density "
            f"is {u['hu_median']:.0f}&nbsp;HU but the 10th percentile is "
            f"{u['hu_p10']:.0f}&nbsp;HU &mdash; just above the 300 cut, which "
            "is exactly where the pilot's false positives lived.",
        ])
        rows = [["Rejection reason", "n", "What it is"]]
        expl = {
            "bone": "Cortical or trabecular bone inside the corridor",
            "below_hu_floor": "Below 300&nbsp;HU &mdash; the density cut",
            "too_large_for_ureter": "Wider than 22&nbsp;mm",
            "vascular_calcification": "Within 3&nbsp;mm of an artery",
            "phlebolith_likely": "Two or more phlebolith cues",
            "too_small": "Under 1&nbsp;mm",
        }
        for k, v in u["reasons"].items():
            rows.append([f"<font face='Courier' size='8'>{k}</font>",
                         f"{v:,}", expl.get(k, "")])
        S += [Spacer(1, 4), table(rows, [50 * mm, 18 * mm, 97 * mm],
                                  align_right=(1,)), Spacer(1, 6)]
        if u.get("zones"):
            z = u["zones"]
            S += [P("Accepted detections by zone: " + ", ".join(
                f"<b>{k}</b> {v}" for k, v in
                sorted(z.items(), key=lambda x: -x[1])) +
                ". The zone label depends on the same unvalidated UVJ landmark "
                "as the distances, so it should be read as provisional.")]
        S += [H("The two complete misses, which are the most informative rows",
                3)]
        misses = [["Study", "What the report says", "Why it was missed"]]
        for sid in u["zero"]:
            sent = str(u.get("gt", {}).get(sid, ""))[:120]
            if "129" in sent:
                why = ("<b>Expected.</b> 129&nbsp;HU is below the 130&nbsp;HU "
                       "growth threshold, so it was never a candidate. This is "
                       "the known ceiling, confirmed.")
            else:
                why = ("<b>Unexplained.</b> A dense stone of reportable size "
                       "that the corridor or the rejection chain lost. The "
                       "single most informative failure in the set.")
            misses.append([f"<b>{sid}</b>", sent, why])
        S += [table(misses, [20 * mm, 80 * mm, 65 * mm]), Spacer(1, 6)]
        S += callout(
            "What this means for the claim we can make",
            "'There is a calculus in the left ureter' is defensible today. "
            "'There are two calculi in the left ureter, 4.2&nbsp;mm and "
            "3.1&nbsp;mm, 18&nbsp;mm from the UVJ' is not &mdash; the count is "
            "inflated and the distance rests on a landmark that has never been "
            "compared to a radiologist's click. Part 1 already carries all of "
            "that detail; Part 2 does not yet.", RISK)
    else:
        S += [P("No ureteric validation CSVs found on disk.")]

    # ---------------------------------------------------- 9 experiments -----
    S += [PageBreak(), P("SECTION 9", "kicker"),
          H("Experiments, including the ones that failed", 1),
          P("Recorded so they are not repeated. Each was a plausible idea; "
            "most were wrong, and the wrong ones were more useful than the "
            "right ones.", "lead")]
    exps = [
        ("Ureter segmentation with a published nnU-Net", "Failed",
         "A network trained on 119 dual-energy virtual-unenhanced volumes, run "
         "as a 5-fold ensemble with a softmax sweep down to 0.05. It found the "
         "dilated <b>renal pelvis</b> &mdash; 2,744 voxels in one component "
         "entirely inside the kidney's own z-range, with the bladder 150 slices "
         "away. Lowering the threshold made the component count explode from 1 "
         "to 19 while the largest barely grew: the signature of noise, not of "
         "an under-confident model. <b>Root cause:</b> its labels came from "
         "contrast-filled ureters, so the segment below an obstructing stone "
         "was trained as background &mdash; the exact thing we need."),
        ("The 3&nbsp;mm crop model would fix the bad kidney masks", "Falsified",
         "Prediction: TotalSegmentator's 6&nbsp;mm <font face='Courier' "
         "size='8'>--roi_subset</font> crop pass was truncating kidneys, so "
         "<font face='Courier' size='8'>-rsr</font> (3&nbsp;mm) would recover "
         "them. Result: <b>12 of 13 unchanged or worse</b>. One study went "
         "26.6&nbsp;mL to 3.7; another went 11.1/19.7 to 0/0."),
        ("An HU padding floor explains the mask failures", "Falsified",
         "11 of 13 failing studies have a &minus;8192 or &minus;3024 padding "
         "floor &mdash; but so do 10 of 20 studies that segment perfectly. Not "
         "causal."),
        ("A composite ranking score beats plain peak HU", "Falsified",
         "Log-HU plus log-volume plus off-path distance ranked the true stone "
         "1,2,1,2,2 against plain peak HU's 1,1,1,2,1. The extra terms were "
         "noise."),
        ("An absolute aorta-HU threshold can catch contrast studies",
         "Falsified",
         "76 of 169 studies have a chosen series reading above 200&nbsp;HU "
         "there, including many named <font face='Courier' size='8'>Thin Plain"
         "</font>, because the measure is a cheap proxy whose 90th percentile "
         "catches calcified wall and bone. The reliable test is median density "
         "inside the kidney mask (&gt;70&nbsp;HU), which needs segmentation to "
         "have run first."),
        ("A straight kidney-to-bladder corridor", "Failed, then fixed",
         "Produced roughly 70 spurious detections per scan by cutting through "
         "bowel and iliac vessels. Replaced by the three-landmark curve, which "
         "is the corridor in use today."),
        ("Skipping the bone carve inside the corridor", "Failed, then fixed",
         "First ureteric runs produced 27 and 43 'stones' per study, because "
         "trabecular bone sits at 130&ndash;350&nbsp;HU, above the growth "
         "threshold. Bone is now subtracted from the corridor &mdash; without "
         "a margin, so a stone lying against the sacrum still survives."),
        ("The 300&nbsp;HU floor costs us the 106 and 129&nbsp;HU stones",
         "Self-correction",
         "It does not. Those are below the 130&nbsp;HU growth threshold and "
         "were never candidates at any density cut. A claim made in error and "
         "withdrawn."),
        ("QC thresholds from a textbook whole-kidney range", "Recalibrated",
         "The 90&nbsp;mL floor sat at our own 25th percentile because "
         "TotalSegmentator segments parenchyma only. Recalibrated to anatomical "
         "values, cutting the review queue from 59 studies to 16."),
    ]
    rows = [["Experiment", "Outcome", "What happened"]]
    for name, outcome, detail in exps:
        tone = (OK if outcome.startswith("Recalib") or "fixed" in outcome
                else RISK)
        rows.append([f"<b>{name}</b>",
                     f'<font color="{tone.hexval()}"><b>{outcome}</b></font>',
                     detail])
    S += [table(rows, [42 * mm, 24 * mm, 99 * mm])]

    # ---------------------------------------------------------- 10 figures --
    S += [PageBreak(), P("SECTION 10", "kicker"),
          H("Figures: what the outputs look like", 1),
          P("Every number this pipeline produces can be checked on an image. "
            "That is not decoration &mdash; a CSV cannot distinguish a "
            "phlebolith from a calculus from a sacral cortical edge, and a "
            "slice can.", "lead")]
    S += figure(os.path.join(ROOT, "run_v5", "overlays", "8231547",
                             "_coronal_mip.png"),
                "Figure 1 &mdash; Part 1 review view for study 8231547. A "
                "whole-scan coronal MIP with the kidney and bladder masks "
                "outlined and every accepted calculus circled. This is the "
                "first thing to look at when a study's numbers seem wrong.",
                max_h=150 * mm)
    S += [PageBreak()]
    S += figure(os.path.join(ROOT, "3d_kidneys", "8287088", "views.png"),
                "Figure 2 &mdash; 3D parenchymal surface for study 8287088, "
                "with per-side volumes and the bounding-box dimensions. This "
                "view answers whether a mask has the right <i>shape</i>, which "
                "no single slice shows: study 8193874 read as plausible on "
                "slices and rendered as a normal right kidney beside a tiny "
                "isolated speck on the left.",
                max_h=165 * mm)
    S += [PageBreak()]
    S += figure(os.path.join(ROOT, "run_ureter", "overlays",
                             "8193874_ureteric.png"),
                "Figure 3 &mdash; Part 2 review sheet for study 8193874: 13 "
                "accepted detections, 157 rejected. Left, the ureteric course "
                "interpolated from PUJ to UVJ in green with every detection "
                "marked. Right, one row per detection &mdash; anatomical "
                "context with vessel outlines, then the stone's own voxels "
                "contoured. The top-ranked detection matches the report's "
                "'9 &times; 6 mm calculus in left upper ureter'; the two "
                "orange rows at the bottom are rejected vascular "
                "calcifications, visibly on the vessel wall.",
                max_h=225 * mm)

    # ----------------------------------------------------------- 11 status --
    S += [PageBreak(), P("SECTION 11", "kicker"),
          H("Where the project stands, and what is next", 1)]
    S += [table([
        ["Claim", "Evidence", "Status"],
        ["Kidney stone present or absent",
         "95.6% sensitivity (65/68), 82.7% specificity (43/52), n=137",
         f'<font color="{OK.hexval()}"><b>Validated</b></font>'],
        ["Kidney stone size",
         "Phantoms: 0.11&nbsp;mm at 0.7&nbsp;mm, 0.55&nbsp;mm at 3&nbsp;mm",
         f'<font color="{OK.hexval()}"><b>Validated</b></font>'],
        ["Kidney stone volume", "6% typical, 20% at 3&nbsp;mm, 16% below "
                                "3&nbsp;mm stone size",
         f'<font color="{OK.hexval()}"><b>Validated</b></font>'],
        ["Calyceal location", "Geometric projection, confidence reported",
         f'<font color="{WARN.hexval()}"><b>Reasoned</b></font>'],
        ["Ureteric stone side",
         f"{u.get('side_hit','?')} of {u.get('side_n','?')} report-confirmed "
         f"studies",
         f'<font color="{OK.hexval()}"><b>Validated</b></font>'],
        ["Ureteric stone count",
         f"{u.get('n_accepted','?')} detected against ~1 per report; "
         f"{u.get('bilateral','?')} bilateral",
         f'<font color="{RISK.hexval()}"><b>Contradicted</b></font>'],
        ["HU_FLOOR = 300", "Fitted on 5 positives, not confirmed at n=37",
         f'<font color="{WARN.hexval()}"><b>Fitted</b></font>'],
        ["Distance to UVJ or PUJ",
         "The landmark has never been checked against a radiologist's click",
         f'<font color="{RISK.hexval()}"><b>Unvalidated</b></font>'],
        ["Ureteric zone", "Rests on the same landmark, plus the sacrum span",
         f'<font color="{RISK.hexval()}"><b>Unvalidated</b></font>'],
        ["Phlebolith rejection", "No report in the cohort says 'phlebolith'",
         f'<font color="{RISK.hexval()}"><b>No ground truth</b></font>'],
        ["Hydronephrosis, fat stranding",
         "Not implemented; 101 and 109 reports mention them",
         f'<font color="{RISK.hexval()}"><b>Not started</b></font>'],
    ], [42 * mm, 82 * mm, 41 * mm]), Spacer(1, 8)]
    S += [H("Next, in priority order", 2)]
    S += bullets([
        "<b>Cut the ureteric false-positive rate.</b> Sweep the density floor, "
        "the corridor radius and the top-K cut against the 37 validated "
        "studies. The new overlays show accepted detections clustering "
        f"{u.get('offpath_median', 0):.0f}&nbsp;mm off the centreline inside a "
        "20&nbsp;mm corridor, so tightening the radius is now a testable lever "
        "rather than a guess. Reporting the top 1 per side instead of the top 2 "
        "would roughly halve the count for free.",
        "<b>Diagnose the one unexplained miss.</b> A dense, reportable stone "
        "that the pipeline lost teaches more than twenty confirmations.",
        "<b>Validate the UVJ landmark.</b> About 40 studies, two clicks each, "
        "roughly an hour of radiologist time. Not to train anything &mdash; to "
        "bound the error on a rule every distance depends on. If it lands "
        "within 5&nbsp;mm, no annotation is ever needed again; if it is off by "
        "15&nbsp;mm, that is far better learned now than after shipping.",
        "<b>Build hydronephrosis and fat stranding, or descope them.</b> Both "
        "are in the brief and currently emit a placeholder. Hydronephrosis is "
        "worth building for a second reason: it currently masquerades as a "
        "segmentation failure, and its signature is measurable &mdash; normal "
        "length, low parenchymal volume, fluid-density cavity inside.",
        "<b>Per-stone matching against report text</b>, which is what turns "
        "today's study-level agreement into a real per-lesion figure.",
    ])
    S += [Spacer(1, 8)]
    S += callout(
        "The failure mode to avoid",
        "It is not being wrong. It is being wrong <i>confidently</i>, in a "
        "column that looks exactly like a validated one. Every number this "
        "system emits should carry which of the four buckets above it came "
        "from, all the way through to the report template.", ACCENT)

    doc.build(S)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT,
                                                  "CALCULUS_PROJECT_REPORT.pdf"))
    a = ap.parse_args()
    d = gather()
    u = ureteric_stats(d)
    path = build(a.out, d, u)
    print(f"wrote {path}  ({os.path.getsize(path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
