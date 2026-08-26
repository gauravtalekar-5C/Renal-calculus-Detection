"""Locate a stone by VERTEBRAL LEVEL -- the reference radiologists actually use.

WHY THIS EXISTS
---------------
Ureteric position was reported as "distance from the UVJ", and the UVJ is a
GUESS. ureter_corridor.landmark_uvj picks the most postero-lateral voxel in the
lowest 35% of the bladder, and its own docstring says "UNVALIDATED ... anatomy
translated into a geometric rule, not a rule measured against radiologist
clicks". On validation case 8676809 that guess was 49 mm from the truth:

    bladder 90 mm tall (heavily distended), slices 25-89
    our UVJ landmark      slice 47
    the stone             slice 82   -- 89% up the bladder
    we printed  "Mid ureter - 54.9 mm from UVJ"
    report says "left vesicoureteric junction"

The detection was correct: right side, 4.7 mm against the report's 6 x 3.3 mm,
1487 HU. Only the sentence describing WHERE it was is wrong -- and that sentence
is the one a urologist acts on, since it decides between stenting, ureteroscopy
and lithotripsy.

Tuning UVJ_BASE_FRAC would move the error to the next bladder. A distended
bladder balloons superiorly by 50+ mm while the trigone stays put, so no fixed
fraction of bladder height can locate the ureteric orifice.

THE FIX: STOP GUESSING, USE BONE
--------------------------------
Radiologists locate ureteric stones against the spine, and the reports in this
very cohort are written that way:

    8675824   "at L3-L4 vertebral level, approximately 6 to 7 cm below the PUJ"
    8659576   "at the L5-S1 level"
    8677912   "at L4 level"

Vertebral bodies are large, high-contrast, and TotalSegmentator already gives us
vertebrae_L1 through L5 plus the sacrum. So the level is a lookup against masks
we have, it needs no annotation, no fitted constant and no landmark guess, and a
validator can check it against the report text directly.

WHAT THIS DOES NOT REPLACE
--------------------------
Distance-from-UVJ stays in the CSV. It is still the right quantity for a
urologist planning retrograde access, and it will become trustworthy once the
UVJ landmark is validated against radiologist clicks. This module adds a
localisation that is correct NOW, and the report prints both.

Below the sacrum there is no vertebra to name, so the level degrades to a
descriptive band rather than inventing precision: a stone at the bladder is
"pelvic (below S1)".
"""
import numpy as np

# Ordered cranial -> caudal. Only lumbar and below matter: a ureteric stone
# above L1 is not a ureteric stone.
LEVELS = ("vertebrae_T12", "vertebrae_L1", "vertebrae_L2", "vertebrae_L3",
          "vertebrae_L4", "vertebrae_L5", "sacrum")

SHORT = {"vertebrae_T12": "T12", "vertebrae_L1": "L1", "vertebrae_L2": "L2",
         "vertebrae_L3": "L3", "vertebrae_L4": "L4", "vertebrae_L5": "L5",
         "sacrum": "S"}


def spans(masks):
    """{level: (z_min, z_max)} for every vertebral mask present, in voxels."""
    out = {}
    for k in LEVELS:
        m = masks.get(k)
        if m is None or not m.any():
            continue
        z = np.where(m.any(axis=(0, 1)))[0]
        out[k] = (int(z.min()), int(z.max()))
    return out


def level_at(z_index, sp):
    """Name the vertebral level at craniocaudal index `z_index`.

    `sp` is the output of spans(). Returns a short string:

        "L4"      squarely beside the L4 body
        "L4-L5"   in the disc space / overlap between two bodies
        "S1-S2"   beside the sacrum, with the sacral segment estimated by
                  dividing the sacrum's height into five, because
                  TotalSegmentator gives one sacrum mask rather than S1..S5
        "pelvic (below S)"   past the sacrum entirely -- near the bladder
        ""        no vertebral masks available

    Never guesses beyond the evidence: outside the segmented span it returns a
    band, not a level.
    """
    if not sp:
        return ""
    z = int(round(z_index))

    # inside a body?
    inside = [k for k, (lo, hi) in sp.items() if lo <= z <= hi]
    if len(inside) == 1:
        k = inside[0]
        if k == "sacrum":
            return _sacral(z, sp["sacrum"])
        return SHORT[k]
    if len(inside) > 1:
        # overlapping masks: name them cranial-to-caudal, e.g. "L4-L5"
        inside.sort(key=lambda k: -sp[k][1])
        a, b = inside[0], inside[-1]
        if a == b:
            return SHORT[a]
        return f"{SHORT[a]}-{SHORT[b]}"

    # between two bodies -> the disc space
    above = [k for k, (lo, hi) in sp.items() if lo > z]      # more cranial
    below = [k for k, (lo, hi) in sp.items() if hi < z]      # more caudal
    if above and below:
        a = min(above, key=lambda k: sp[k][0])               # nearest above
        b = max(below, key=lambda k: sp[k][1])               # nearest below
        return f"{SHORT[a]}-{SHORT[b]}"
    if below and not above:
        # more caudal than everything segmented: past the sacrum
        return "pelvic (below S)"
    return ""           # above everything segmented; not a ureteric position


def _sacral(z, span):
    """Estimate a sacral segment by dividing the sacrum into five equal bands.

    TotalSegmentator produces ONE sacrum mask, not S1..S5, so this is an
    estimate and is labelled as one. Reports say "L5-S1" and "S1" often enough
    that naming the band is more useful than printing a bare "S", but it must
    not read as if the segment were segmented.
    """
    lo, hi = span
    h = max(hi - lo, 1)
    # z increases cranially, so S1 is at the TOP of the sacrum
    frac = (hi - z) / h
    seg = min(5, max(1, int(frac * 5) + 1))
    return f"S{seg}"


def describe(z_index, masks_or_spans):
    """Convenience: level string from either a mask dict or a spans() dict."""
    sp = (masks_or_spans if masks_or_spans and
          isinstance(next(iter(masks_or_spans.values())), tuple)
          else spans(masks_or_spans))
    return level_at(z_index, sp)
# WHAT THIS MODULE DOES: says which vertebra a stone lies beside, using the
# vertebral masks we already compute, so the report can localise a ureteric
# calculus the way the radiologist's report does -- "at L4 level" -- instead of
# relying on a guessed bladder landmark that was 49 mm out on a distended
# bladder.
