"""The ureteric verdict: tests that assert PRECEDENCE, not just outcomes.

WHY THIS FILE EXISTS
--------------------
The rejection logic was inline in a 200-line loop. Inserting a flag-only test
into the middle of the if/elif chain silently reattached
`elif hu_max < HU_FLOOR` to the new `if`, so the 300 HU density floor only ran
on candidates that had ALREADY been rejected -- never, in practice. On validation
case 8664459 that admitted five "ureteric calculi" at 156, 189, 229, 249 and
293 HU, all of which the floor exists to remove.

Nothing failed. No exception, no assertion, no crash -- just wrong output that
looked entirely plausible in a CSV. It was caught only because a human read the
density column of one report.

So these tests do not merely check that a bad candidate is rejected. They check
WHICH rule rejects it, because the bug was an ordering bug, and an ordering bug
is invisible to a test that only asks "was it rejected".
"""
import numpy as np
import pytest

from calculus.ureter import detect_ureteric as du


# --- tier 0: two lookups ---------------------------------------------------

def test_bone_beats_everything():
    assert du.verdict_cheap(1.0, 50.0) == "bone"
    assert du.verdict_cheap(0.9, 0.0) == "bone", "bone must win over vessel"


def test_vessel_when_not_bone():
    assert du.verdict_cheap(0.0, 0.5) == "vascular_calcification"
    assert du.verdict_cheap(0.0, 50.0) == ""


def test_nan_vessel_distance_is_not_a_rejection():
    """A study with no arterial masks must not have every candidate called
    vascular calcification."""
    assert du.verdict_cheap(0.0, float("nan")) == ""


# --- tier 1: size and density ---------------------------------------------

def test_density_floor_fires():
    reason, review = du.verdict_measured(6.0, 150.0)
    assert reason == "below_hu_floor"
    assert review == ""


def test_LARGE_LOW_DENSITY_IS_STILL_REJECTED():
    """THE REGRESSION TEST.

    A 25 mm object at 200 HU is over LARGE_FOR_URETER_MM and under HU_FLOOR.
    The broken ordering gave it the review flag and accepted it. The density
    floor must win, because the flag does not reject and the floor does.
    """
    reason, review = du.verdict_measured(25.0, 200.0)
    assert reason == "below_hu_floor", (
        "a large object below the density floor must be REJECTED, not flagged")
    assert review == ""


@pytest.mark.parametrize("hu", [156.0, 189.0, 229.0, 249.0, 293.0])
def test_the_five_hu_values_that_leaked(hu):
    """The exact densities that reached 8664459's report while the chain was
    broken. Every one is below the 300 HU floor and must be rejected."""
    reason, _ = du.verdict_measured(10.0, hu)
    assert reason == "below_hu_floor", f"{hu} HU must not be accepted"


def test_large_and_dense_is_flagged_not_rejected():
    """8659576's real stone: 23.24 mm at 1561 HU. The report describes a 21 mm
    obstructing calculus causing severe hydronephrosis, and MAX_DIAM_MM = 22
    used to delete it."""
    reason, review = du.verdict_measured(23.24, 1561.0)
    assert reason == "", "a real 23 mm stone must NOT be rejected"
    assert review == "large_for_ureter"


def test_absurdly_large_is_rejected():
    reason, _ = du.verdict_measured(45.0, 900.0)
    assert reason == "too_large_for_ureter"


def test_too_small():
    reason, _ = du.verdict_measured(0.5, 900.0)
    assert reason == "too_small"


def test_ordinary_stone_passes_clean():
    reason, review = du.verdict_measured(6.0, 800.0)
    assert reason == "" and review == ""


def test_size_is_checked_before_density():
    """A sub-millimetre speck at 50 HU fails both. It must be reported as
    too_small, because that is the earlier and more specific reason."""
    reason, _ = du.verdict_measured(0.4, 50.0)
    assert reason == "too_small"


# --- tier 2: mimics -------------------------------------------------------

def test_tube_in_fat_rejects_on_its_own():
    assert du.verdict_mimic(["tube_in_fat"], True) == "extraureteric_calcification"


def test_tube_in_fat_beats_phlebolith_label():
    """Both conditions hold; the more specific reason must be recorded, or the
    audit trail cannot distinguish the two mimics."""
    assert du.verdict_mimic(["fatty_rim", "round", "tube_in_fat"], True) == \
        "extraureteric_calcification"


def test_two_cues_reject():
    assert du.verdict_mimic(["fatty_rim", "off_path"], False) == "phlebolith_likely"


def test_one_cue_does_not_reject():
    assert du.verdict_mimic(["fatty_rim"], False) == ""


def test_no_cues():
    assert du.verdict_mimic([], False) == ""


# --- the flag-only stent test must never reject ---------------------------

def test_stent_like_never_touches_a_real_stone():
    """Every confirmed true positive in the validation cohort, by (diameter,
    volume, elongation). None may be flagged as a stent."""
    for dmax, vol, elong in ((4.72, 40.0, 0.620),     # 8676809 L VUJ
                             (5.03, 45.0, 0.587),     # 8678618 R VUJ
                             (6.70, 60.0, 0.297),     # 8674625 R mid
                             (8.75, 120.0, 0.418),    # 8675824 R upper
                             (6.97, 80.0, 0.612),     # 8659576 L VUJ
                             (12.62, 199.0, 0.421),   # 8664459 L VUJ
                             (23.24, 900.0, 0.450)):  # 8659576 R mid, 21 mm
        assert du.stent_like(dmax, vol, elong) == "", (
            f"{dmax} mm / {vol} mm3 / elong {elong} is a real stone")


def test_stent_like_flags_a_tube():
    assert du.stent_like(120.0, 380.0, 0.15) == "stent_like"


# --- compactness: the replacement for what MAX_DIAM_MM used to do ---------

def test_real_21mm_stone_survives_compactness():
    """8659576's obstructing calculus: report says 21 mm / 1466 HU with SEVERE
    hydronephrosis. It measured 23.24 mm, 1561 HU, 951 mm3 -> fill 0.145.
    It must be ACCEPTED and flagged, never rejected."""
    reason, review = du.verdict_measured(23.24, 1561.0, 951.28)
    assert reason == "", "the most urgent stone in the cohort must not be rejected"
    assert review == "large_for_ureter"


@pytest.mark.parametrize("dmax,hu,vol", [
    (35.04, 580.0, 294.99),   # 8674941, fill 0.013
    (36.25, 563.0, 581.03),   # 8674941, fill 0.023
    (25.93, 659.0, 155.02),   # 8674941, fill 0.017
    (20.97, 478.0,  18.05),   # 8674941, fill 0.004
    (21.06, 547.0,  40.75),   # 8674941, fill 0.008
])
def test_large_tubular_objects_are_rejected(dmax, hu, vol):
    """The false positives the raised MAX_DIAM_MM admitted. Large, dense enough
    to clear the floor, but holding almost no volume -- vessels, not stones."""
    reason, _ = du.verdict_measured(dmax, hu, vol)
    assert reason == "tubular_not_stone"


def test_compactness_does_not_touch_small_stones():
    """A 3 mm stone spans a handful of voxels and its caliper is quantised, so
    its fill can be poor. Applying the test below LARGE_FOR_URETER_MM would
    delete microliths."""
    reason, review = du.verdict_measured(3.0, 900.0, 4.0)
    assert reason == "" and review == ""


def test_compactness_needs_a_volume():
    """With no volume the test cannot run, and must not reject by default."""
    reason, review = du.verdict_measured(25.0, 900.0, None)
    assert reason == "" and review == "large_for_ureter"


# --- physical plausibility flags ------------------------------------------

def test_measurement_flags():
    from calculus.kidney import detect_stones as ds
    # 8664459 #194: 30.4 mm caliper holding 217 mm3
    assert "caliper_suspect" in ds.measurement_flags(216.71, 30.41, 806.0)
    # 8674625: both wrong at once
    f = ds.measurement_flags(200.0, 21.8, 3071.0)
    assert "caliper_suspect" in f and "hu_implausible" in f
    # the real stone, and an ordinary one: clean
    assert ds.measurement_flags(951.28, 23.24, 1561.0) == ""
    assert ds.measurement_flags(50.0, 5.0, 800.0) == ""
