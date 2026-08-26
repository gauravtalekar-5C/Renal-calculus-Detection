"""Findings must not disappear between detection and reporting.

This is the 8583083 failure reduced to an invariant. The detectors accepted
seven ureteric calculi; the API answered "Normal, 0 calculi"; the two records sat
on disk contradicting each other and nothing compared them.

The tests below fix the shape of the guard, not just its existence: it must fire
on the real failure, and it must NOT fire on the legitimate transformations that
sit between a detection and a reported finding -- otherwise it gets switched off
and the protection is worth nothing.
"""
import os
import sys

import pandas as pd
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from calculus.report.reconcile import accepted_counts, reconcile   # noqa: E402


def _run(tmp_path, kidney=None, ureter=None, bladder=None):
    per = tmp_path / "csv" / "per_study"
    per.mkdir(parents=True, exist_ok=True)
    for name, frame in (("candidates", kidney),
                        ("ureter_candidates", ureter),
                        ("bladder_candidates", bladder)):
        if frame is not None:
            pd.DataFrame(frame).to_csv(per / f"S1_{name}.csv", index=False)
    return str(tmp_path)


def test_the_8583083_state_is_rejected(tmp_path):
    """Seven accepted ureteric stones, a report counting zero."""
    run = _run(tmp_path, ureter=[{"is_stone": True, "report_this": True}] * 7)
    assert accepted_counts(run, "S1")["ureteric"] == 7
    ok, why, delta = reconcile(run, "S1", reported_total=0)
    assert not ok
    assert "cannot vanish" in why
    assert delta["accepted"]["ureteric"] == 7


def test_a_bladder_only_loss_is_rejected(tmp_path):
    run = _run(tmp_path, bladder=[{"is_stone": True}] * 3)
    ok, why, _ = reconcile(run, "S1", reported_total=0)
    assert not ok


def test_a_genuinely_normal_study_passes(tmp_path):
    """Nothing detected, nothing reported. Must NOT fire."""
    run = _run(tmp_path,
               kidney=[{"is_stone": False, "compartment": "kidney"}],
               ureter=[{"is_stone": False, "report_this": True}],
               bladder=[{"is_stone": False}])
    ok, why, delta = reconcile(run, "S1", reported_total=0)
    assert ok and why == ""
    assert delta["accepted"] == {"renal": 0, "ureteric": 0, "bladder": 0}


def test_a_study_with_no_csvs_at_all_passes(tmp_path):
    """A study the detectors never reached must not be called a bug here.

    That case is covered by the missing-report guard, which raises a different
    and more accurate error. This check must not pre-empt it with a wrong one.
    """
    ok, _, _ = reconcile(str(tmp_path), "S1", reported_total=0)
    assert ok


def test_deduplication_does_not_trip_the_guard(tmp_path):
    """drop_puj_duplicates legitimately reports fewer than were accepted."""
    run = _run(tmp_path,
               kidney=[{"is_stone": True, "compartment": "kidney"}],
               ureter=[{"is_stone": True, "report_this": True}] * 2)
    # 3 accepted, 2 reported after the PUJ duplicate is dropped
    ok, why, delta = reconcile(run, "S1", reported_total=2)
    assert ok, why
    assert delta["accepted"]["renal"] == 1
    assert delta["reported_total"] == 2


def test_ranked_out_ureteric_rows_are_not_counted(tmp_path):
    run = _run(tmp_path, ureter=[{"is_stone": True, "report_this": True},
                                 {"is_stone": True, "report_this": False}])
    assert accepted_counts(run, "S1")["ureteric"] == 1


def test_a_bladder_row_in_the_kidney_csv_is_not_double_counted(tmp_path):
    """The same object reported by two detectors must count once.

    This is the mirror of the FOREIGN-glob bug: bladder rows reaching the kidney
    table would otherwise inflate the accepted count and make the guard fire on
    a correct report.
    """
    run = _run(tmp_path,
               kidney=[{"is_stone": True, "compartment": "bladder_lumen"}],
               bladder=[{"is_stone": True}])
    assert accepted_counts(run, "S1")["renal"] == 0
    assert accepted_counts(run, "S1")["bladder"] == 1
    ok, _, _ = reconcile(run, "S1", reported_total=1)
    assert ok


def test_api_raises_on_the_inconsistent_state(tmp_path):
    """End to end through _shape, not just the pure function."""
    sys.argv = ["x"]
    import app
    run = str(tmp_path)
    rep = os.path.join(run, "reports")
    os.makedirs(rep, exist_ok=True)
    per = os.path.join(run, "csv", "per_study")
    os.makedirs(per, exist_ok=True)
    # a report table that exists but describes nothing...
    with open(os.path.join(rep, "S1_report.csv"), "w") as fh:
        fh.write("section,a,b,c,d,e,f\n")
        fh.write("HEADER,Study,CT KUB PLAIN,,,,\n")
    # ...while the ureteric detector accepted two stones
    pd.DataFrame([{"is_stone": True, "report_this": True}] * 2).to_csv(
        os.path.join(per, "S1_ureter_candidates.csv"), index=False)

    a = app.Analyser.__new__(app.Analyser)
    with pytest.raises(RuntimeError) as e:
        a._shape("S1", "S1", run, [], "", env="prod")
    assert "cannot vanish" in str(e.value)
