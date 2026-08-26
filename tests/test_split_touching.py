"""Splitting stones fused to each other -- and NOT splitting one that isn't.

Both directions matter. Merging three stones into one under-counts and reports
the brightest density only; splitting one branched staghorn into three
over-counts and shrinks every size. The saddle-prominence rule is what separates
the two cases, so it is tested in both directions here.
"""
import numpy as np

from calculus.kidney import detect_stones as ds

SP = (0.8, 0.8, 1.0)


def _sphere(shape, c, r_mm, peak, sp=SP):
    g = np.ogrid[tuple(slice(0, n) for n in shape)]
    d2 = sum(((g[i] - c[i]) * sp[i]) ** 2 for i in range(3))
    v = np.zeros(shape, np.float32)
    v[d2 <= r_mm ** 2] = peak
    # a partial-volume rind, which is what bridges neighbouring stones
    v[(d2 > r_mm ** 2) & (d2 <= (r_mm + 1.2) ** 2)] = peak * 0.45
    return v


def _pieces(vol):
    comp = vol >= ds.GROW_HU
    lab = comp.astype(np.int32)
    out, _, _, _ = ds.split_touching_stones(
        lab, 1, {1: float(vol[comp].max())}, vol, SP, float(np.prod(SP)))
    return len([i for i in np.unique(out) if i])


def test_two_separate_stones_split():
    shape = (40, 40, 40)
    v = np.maximum(_sphere(shape, (20, 20, 15), 2.2, 700),
                   _sphere(shape, (20, 20, 25), 2.2, 650))
    assert _pieces(v) == 2


def test_two_stones_bridged_by_partial_volume_split():
    """The real failure: the bridge never drops below GROW_HU, so labelling
    fuses them. Seen on 8677121 -- three stones reported as one 12.3 mm blob."""
    shape = (40, 40, 40)
    v = np.maximum(_sphere(shape, (20, 20, 18), 2.2, 700),
                   _sphere(shape, (20, 20, 23), 2.2, 650))
    v[20, 20, 20:22] = 200.0            # the bridge, above 130
    assert _pieces(v) == 2


def test_branched_stone_is_not_split():
    """A staghorn has several lobes and several peaks but NO dark neck between
    them. It must survive as one stone."""
    shape = (40, 40, 40)
    v = np.maximum(_sphere(shape, (20, 20, 18), 2.4, 800),
                   _sphere(shape, (20, 20, 22), 2.4, 800))
    v[v > 0] = np.maximum(v[v > 0], 780)
    assert _pieces(v) == 1


def test_single_small_stone_is_not_split():
    assert _pieces(_sphere((40, 40, 40), (20, 20, 20), 1.6, 600)) == 1


def test_branched_calculus_is_not_split():
    """A parent that shatters into more pieces than TOUCH_MAX_PIECES is a
    branched calculus, not a cluster of touching stones, and must be left whole.

    Five lobes on a stalk, each a separate 900 HU peak, joined by 240 HU necks --
    the geometry measured on 8662768, where the splitter cut one staghorn the
    radiologist described as 22 x 31 x 29 mm into twelve 'stones'.
    """
    import cc3d
    v = np.zeros((44, 24, 24), np.float32)
    for x0 in (4, 11, 18, 25, 32):
        v[x0:x0 + 5, 10:15, 10:15] = 900.0
    for x0 in (9, 16, 23, 30):
        v[x0:x0 + 2, 10:15, 10:15] = 240.0      # necks ABOVE GROW_HU -> one blob
    lab, n = cc3d.connected_components(v >= ds.GROW_HU, connectivity=26,
                                       return_N=True)
    assert n == 1, "precondition: the necks must make this ONE component"
    peak = {1: float(v.max())}
    out, _, m, nsp = ds.split_touching_stones(lab.astype(np.int32), 1, peak,
                                              v, (1.0, 1.0, 1.0), 1.0)
    assert m == 1, f"a 5-lobe branched calculus must stay ONE object, got {m}"
    assert nsp == 0
    assert getattr(ds.split_touching_stones, "last_unsplit", 0) >= 1


def test_two_touching_stones_still_split():
    """The regression guard: TOUCH_MAX_PIECES must not undo the fix the splitter
    was built for. Two lobes -> two stones, as on 8677121."""
    import cc3d
    v = np.zeros((26, 24, 24), np.float32)
    v[5:11, 9:16, 9:16] = 900.0
    v[15:21, 9:16, 9:16] = 900.0
    v[11:15, 10:15, 10:15] = 240.0
    lab, n = cc3d.connected_components(v >= ds.GROW_HU, connectivity=26,
                                       return_N=True)
    assert n == 1
    out, _, m, nsp = ds.split_touching_stones(lab.astype(np.int32), 1,
                                              {1: 900.0}, v, (1.0, 1.0, 1.0), 1.0)
    assert m == 2, f"two touching stones must still split, got {m}"
    assert nsp == 1
