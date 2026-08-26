"""merge_fragments: rejoin one calculus that partial volume broke into pieces.

These tests check the RETURN COUNT as well as the label content. The previous
splitter shipped with an off-by-one (`next_id` started at 1 and incremented
after assigning, so it returned count+1 while the caller looped `range(1, n+1)`)
and the tests did not catch it because they only counted distinct pieces.
"""
import numpy as np
import pytest

from calculus.kidney import detect_stones as ds

SP = (1.0, 1.0, 1.0)
VOX = 1.0


def _labelled(vol, grow=None):
    """Label vol >= GROW_HU the way the detector does, and build peak_of."""
    import cc3d
    grow = ds.GROW_HU if grow is None else grow
    lab, n = cc3d.connected_components(vol >= grow, connectivity=26, return_N=True)
    peak = {i: float(vol[lab == i].max()) for i in range(1, n + 1)}
    return lab.astype(np.int32), n, peak


def test_dense_bridge_merges():
    """Two lobes joined by material at 100 HU -- below GROW_HU (130) so they
    label separately, but well above urine. That is the staghorn neck."""
    v = np.zeros((24, 24, 24), np.float32)
    v[6:11, 10:15, 10:15] = 900.0          # lobe A, ends at x=10
    v[13:18, 10:15, 10:15] = 900.0         # lobe B, starts at x=13
    v[11:13, 11:14, 11:14] = 100.0         # 2-voxel partial-volume neck
    # centre-to-centre gap x=10 -> x=13 is 3 mm, i.e. exactly MERGE_MAX_GAP_MM
    lab, n, peak = _labelled(v)
    assert n == 2, "precondition: the neck must NOT connect them at GROW_HU"
    out, np_, m, nm = ds.merge_fragments(lab, n, peak, v, SP, VOX)
    assert nm == 1
    assert m == 1, f"returned count should be 1, got {m}"
    assert len(set(out[out > 0].ravel())) == 1
    assert sorted(np_) == [1] and np_[1] == 900.0


def test_urine_gap_does_not_merge():
    """Same geometry, but the gap holds urine at 10 HU. Two separate stones."""
    v = np.zeros((24, 24, 24), np.float32)
    v[6:11, 10:15, 10:15] = 900.0
    v[13:18, 10:15, 10:15] = 900.0
    v[11:13, 11:14, 11:14] = 10.0
    lab, n, peak = _labelled(v)
    assert n == 2
    out, np_, m, nm = ds.merge_fragments(lab, n, peak, v, SP, VOX)
    assert nm == 0 and m == 2
    assert len(set(out[out > 0].ravel())) == 2


def test_far_apart_does_not_merge():
    """Dense material between them, but the gap is wider than MERGE_MAX_GAP_MM.
    Two stones in different calyces must stay two."""
    v = np.zeros((40, 24, 24), np.float32)
    v[4:9, 10:15, 10:15] = 900.0
    v[30:35, 10:15, 10:15] = 900.0
    v[9:30, 11:14, 11:14] = 100.0          # a 21 mm "bridge" -- far too long
    lab, n, peak = _labelled(v)
    assert n == 2
    out, np_, m, nm = ds.merge_fragments(lab, n, peak, v, SP, VOX, gap_mm=3.0)
    assert nm == 0 and m == 2


def test_chain_merges_transitively():
    """A-B and B-C bridge; A and C never touch. A staghorn is a chain of lobes,
    so all three must end up as one calculus."""
    v = np.zeros((34, 24, 24), np.float32)
    for x0 in (4, 11, 18):
        v[x0:x0 + 5, 10:15, 10:15] = 900.0
    v[9:11, 11:14, 11:14] = 100.0          # bridges A-B
    v[16:18, 11:14, 11:14] = 100.0         # bridges B-C, A never touches C
    lab, n, peak = _labelled(v)
    assert n == 3
    out, np_, m, nm = ds.merge_fragments(lab, n, peak, v, SP, VOX)
    assert m == 1, f"chain should collapse to one, got {m}"
    assert nm == 2
    assert len(set(out[out > 0].ravel())) == 1


def test_single_blob_untouched():
    v = np.zeros((20, 20, 20), np.float32)
    v[8:13, 8:13, 8:13] = 800.0
    lab, n, peak = _labelled(v)
    assert n == 1
    out, np_, m, nm = ds.merge_fragments(lab, n, peak, v, SP, VOX)
    assert nm == 0 and m == 1
    assert np.array_equal(out, lab)


def test_peak_is_the_max_of_the_group():
    """The merged object's peak must be the brightest member's, not the first."""
    v = np.zeros((24, 24, 24), np.float32)
    v[6:11, 10:15, 10:15] = 400.0
    v[13:18, 10:15, 10:15] = 1200.0
    v[11:13, 11:14, 11:14] = 100.0
    lab, n, peak = _labelled(v)
    out, np_, m, nm = ds.merge_fragments(lab, n, peak, v, SP, VOX)
    assert m == 1 and np_[1] == 1200.0


def test_gap_limit_is_respected():
    """Same dense neck, one voxel wider. Beyond MERGE_MAX_GAP_MM it must NOT
    merge -- condition 2 is the safety belt that stops a leaking bridge mask
    fusing two separate calyceal stones."""
    v = np.zeros((26, 24, 24), np.float32)
    v[6:11, 10:15, 10:15] = 900.0          # ends x=10
    v[15:20, 10:15, 10:15] = 900.0         # starts x=15 -> 5 mm centre-to-centre
    v[11:15, 11:14, 11:14] = 100.0
    lab, n, peak = _labelled(v)
    assert n == 2
    out, np_, m, nm = ds.merge_fragments(lab, n, peak, v, SP, VOX, gap_mm=3.0)
    assert nm == 0 and m == 2
    # ...and it DOES merge once the gap allowance covers it
    out, np_, m, nm = ds.merge_fragments(lab, n, peak, v, SP, VOX, gap_mm=6.0)
    assert nm == 1 and m == 1
