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
