"""The size string's axis order and label, and the capture match that reads it.

Both bugs these cover were silent. The axis order was wrong for months without
anything failing -- our string put TR first while every report writes AP first,
so a reader comparing the two compared different axes and saw a size error that
was not there. And when the label was added to the report cell but not to the
capture index, every kidney capture vanished from the API response, because a
failed identity match yields no key rather than an error.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calculus.report.make_report import fmt_size                      # noqa: E402
from calculus.evaluate.score_cohort import parse_our_size_mm          # noqa: E402


def test_ap_comes_first_and_axes_are_named():
    # called (tr, ap, cc) -- the order every call site uses
    s = fmt_size(11.72, 7.60, 5.08)
    assert s == "7.6 x 11.7 x 5.1 (AP x TR x CC)"
    # AP first, matching the 328 reports that write "(AP x TR)" outright
    assert s.split(" x ")[0] == "7.6"


def test_harness_reads_the_largest_axis_through_the_label():
    assert parse_our_size_mm(fmt_size(5.55, 4.44, 7.22)) == 7.2


def test_a_numeric_label_cannot_corrupt_the_size():
    # the current label has no digits; a future one must not be able to win
    assert parse_our_size_mm("4.4 x 5.5 x 7.2 (AP x TR x CC, 3 axes)") == 7.2


def _attach(calculi, index, sid="S1"):
    sys.argv = ["x"]
    import app
    return app.Analyser._attach_captures(calculi, index, sid)


def test_capture_matches_despite_a_formatting_difference():
    """The exact regression: report cell labelled, index cell not."""
    calculi = {"left": [{"organ": "Kidney", "density_hu": 220,
                         "size_mm": "2.4 x 2.4 x 2.4 (AP x TR x CC)"}]}
    index = {"kidney": [{"file": "kidney_01.png", "side": "left",
                         "density_hu": 220, "size_mm": "2.4 x 2.4 x 2.4"}]}
    out = _attach(calculi, index)
    assert out["left"][0]["secondary_capture"].endswith("kidney_01.png")


def test_capture_matches_despite_axis_reordering():
    calculi = {"right": [{"organ": "Kidney", "density_hu": 683,
                          "size_mm": "2.2 x 3.6 x 4.9 (AP x TR x CC)"}]}
    index = {"kidney": [{"file": "kidney_07.png", "side": "right",
                         "density_hu": 683, "size_mm": "3.6 x 2.2 x 4.9"}]}
    out = _attach(calculi, index)
    assert out["right"][0]["secondary_capture"].endswith("kidney_07.png")


def test_a_genuinely_different_stone_still_gets_no_capture():
    """Loosening the match must not make it match anything."""
    calculi = {"left": [{"organ": "Kidney", "density_hu": 220,
                         "size_mm": "2.4 x 2.4 x 2.4 (AP x TR x CC)"}]}
    index = {"kidney": [{"file": "kidney_01.png", "side": "left",
                         "density_hu": 900, "size_mm": "9.9 x 9.9 x 9.9"}]}
    out = _attach(calculi, index)
    assert "secondary_capture" not in out["left"][0]


def test_ureteric_stones_carry_no_capture_key():
    calculi = {"left": [{"organ": "Ureter (VUJ)", "density_hu": 577,
                         "size_mm": "7.6 x 11.7 x 5.1 (AP x TR x CC)"}]}
    out = _attach(calculi, {"kidney": []})
    assert "secondary_capture" not in out["left"][0]
