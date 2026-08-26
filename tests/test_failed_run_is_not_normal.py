"""A failed analysis must never be reported as a normal study.

On 8583083 the report step crashed with AttributeError, infer_study discarded
the failure and exited 0, the API found no report CSV and shaped its answer from
nothing -- returning study_prediction "Normal" for a study whose pipeline log
listed seven detected ureteric calculi, one of them 9.1 mm at 1274 HU.

The absence of a report is not the absence of disease. These tests pin that.
"""
import os
import sys

import pandas as pd
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from calculus.report.make_report import _organ_str          # noqa: E402
from calculus.kidney.detect_stones import CAND_COLS          # noqa: E402


class Row:
    """An itertuples-like row that simply lacks the field."""
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_organ_survives_a_missing_compartment():
    # the exact crash: the attribute is absent, not NaN
    assert _organ_str(getattr(Row(side="left"), "compartment", None)) == "Kidney"


def test_organ_survives_a_nan_compartment():
    assert _organ_str(float("nan")) == "Kidney"


def test_organ_reads_a_real_compartment():
    assert _organ_str("kidney") == "Kidney"
    assert _organ_str("upper_ureter") == "Upper Ureter"


def test_api_refuses_to_shape_without_a_report(tmp_path):
    sys.argv = ["x"]
    import app
    run = str(tmp_path)
    os.makedirs(os.path.join(run, "reports"), exist_ok=True)
    a = app.Analyser.__new__(app.Analyser)
    with pytest.raises(RuntimeError) as e:
        a._shape("S1", "S1", run, [], "", env="prod")
    msg = str(e.value).lower()
    # it must say WHY, and must not have invented a prediction
    assert "no report" in msg or "no result" in msg
    assert "not a negative finding" in msg


def test_empty_study_csv_columns_are_a_real_subset():
    """CAND_COLS is a contract; it must not drift from the real schema."""
    sample = os.path.join(HERE, "final_check_deployment", "csv", "per_study",
                          "8677912_candidates.csv")
    if not os.path.exists(sample):
        pytest.skip("no sample candidates.csv on this box")
    real = set(pd.read_csv(sample, nrows=0).columns)
    missing = set(CAND_COLS) - real
    assert not missing, f"CAND_COLS names columns a real run does not have: {missing}"


def test_part1_only_excludes_every_other_compartment(tmp_path):
    """The kidney table must contain kidney rows and nothing else.

    detect_ureteric and detect_bladder write <id>_ureter_candidates.csv and
    <id>_bladder_candidates.csv into the same directory, and the glob
    "*_candidates.csv" matches both. The ureter case was found and fixed; the
    bladder case was left, invisible only because the bladder erosion kept that
    file empty. Fixing the erosion merged five bladder rows into
    baseline_stones.csv on 8583083.
    """
    import glob as _glob
    for name in ("81_candidates.csv", "81_ureter_candidates.csv",
                 "81_bladder_candidates.csv"):
        (tmp_path / name).write_text("study_id\n81\n")

    from calculus.kidney.detect_stones import main as _  # noqa: F401  import check
    import calculus.kidney.detect_stones as m
    FOREIGN = ("_ureter_", "_bladder_")
    got = [os.path.basename(f)
           for f in sorted(_glob.glob(str(tmp_path / "*_candidates.csv")))
           if not any(k in os.path.basename(f) for k in FOREIGN)]
    assert got == ["81_candidates.csv"], got
    # and the constant really is in the module, not just in this test
    src = open(m.__file__).read()
    assert 'FOREIGN = ("_ureter_", "_bladder_")' in src
