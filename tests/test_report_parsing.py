"""The report parser decides our ground truth, so its edge cases are pinned here.

Both cases below are real bugs that reached results before being caught.
"""
from calculus.evaluate.compare_measurements import calculus_sizes


def test_obstructing_stone_in_a_hydronephrosis_sentence_is_not_discarded():
    """This exclusion silently dropped the most important ureteric stones."""
    s = ("Mild hydronephrosis of approximately 9 x 6 mm sized calculus noted "
         "in left upper ureter")
    got = calculus_sizes(s)
    assert 9.0 in got["ureteric"] and 6.0 in got["ureteric"]


def test_organ_and_cyst_measurements_are_not_read_as_stones():
    """'Right kidney: Measures 11.2 x 4.7 cm' once became a 116 mm stone."""
    s = ("Right kidney: Measures 11.2 x 4.7 cm. A 4.6 x 3.1 x 5.7 mm calculus "
         "is seen in the interpolar calyx")
    got = calculus_sizes(s)
    assert max(got["renal"]) < 10.0


def test_bladder_stones_are_filed_separately():
    s = ("A large hyperdense vesical calculus measuring approximately "
         "4.6 x 3.2 x 3.6 cm is noted within the urinary bladder")
    got = calculus_sizes(s)
    assert got["bladder"] and not got["renal"] and not got["ureteric"]
