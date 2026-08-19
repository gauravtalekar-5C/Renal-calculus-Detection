"""Tests that need no patient data -- pure geometry and parsing.

Deliberately no test asserts a sensitivity or a threshold value: those are
measured on a cohort, they change when the cohort changes, and pinning them in a
test would turn a measurement into an assumption.
"""
import numpy as np
import pytest

from calculus.ureter import ureter_corridor as uc
from calculus.ureter import ureter_corridor_fast as ucf


def test_centreline_runs_between_its_landmarks():
    puj = np.array([100.0, 100.0, 300.0])
    ili = np.array([120.0, 90.0, 200.0])
    uvj = np.array([130.0, 110.0, 100.0])
    p = uc.centreline(puj, ili, uvj)
    # N_PATH segments means N_PATH+1 points: both endpoints are included
    assert len(p) == uc.N_PATH + 1
    # the smoothed curve starts near the PUJ and ends near the UVJ
    assert np.linalg.norm(p[0] - puj) < 25
    assert np.linalg.norm(p[-1] - uvj) < 25


def test_arclength_is_at_least_the_straight_line():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 50.0])
    c = np.array([0.0, 0.0, 100.0])
    p = uc.centreline(a, b, c)
    arc = uc.arclen_mm(p, (1.0, 1.0, 1.0))
    assert arc[-1] >= 99.0        # a curve is never shorter than the chord


def test_fast_corridor_box_contains_the_whole_radius():
    """The optimisation is only valid if the padded box cannot clip the tube."""
    shape = (60, 60, 60)
    pm = np.zeros(shape, bool)
    pm[30, 30, 20:40] = True
    spacing = (1.0, 1.0, 1.0)
    box = ucf._box(pm, shape, spacing, 8.0)
    assert box[0].start <= 30 - 8 and box[0].stop >= 30 + 8 + 1
    assert box[2].start <= 20 - 8 and box[2].stop >= 39 + 8 + 1


@pytest.mark.parametrize("mm_to_uvj,expect", [(5.0, "vuj"), (30.0, "lower"),
                                              (120.0, "mid")])
def test_zone_classification(mm_to_uvj, expect):
    bounds = {"sacrum_top": 10, "sacrum_bottom": 0}
    assert uc.classify_zone(5, bounds, mm_to_uvj) == expect
