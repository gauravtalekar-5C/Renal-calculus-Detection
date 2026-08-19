"""Where the ureter runs, derived from organs we already segment.

WHY THIS EXISTS
---------------
`detect_stones.build_roi(kidney_only=False)` drew a STRAIGHT LINE from the
kidney centroid to the bladder centroid and called it the ureter. On the first
test study that produced 70 spurious detections, because a straight line cuts
through bowel, mesentery, sacrum and iliac vessels -- none of which the real
ureter goes near.

The real ureter is not straight. It:

  1. leaves the kidney at the PELVI-URETERIC JUNCTION (PUJ), on the medial
     side of the hilum;
  2. runs almost vertically down the anterior surface of psoas major, roughly
     over the tips of the lumbar transverse processes;
  3. crosses the COMMON ILIAC ARTERY at the pelvic brim -- a real, segmentable
     landmark, and the one that makes the curve a curve;
  4. swings posterolaterally along the pelvic sidewall;
  5. turns anteromedially to enter the bladder at the VESICO-URETERIC JUNCTION
     (UVJ), on the posterolateral corner of the bladder base.

Steps 1, 3 and 5 are all derivable from masks TotalSegmentator already gives
us (kidney Dice ~0.96, bladder ~0.90, iliac arteries). So we get a curved
corridor for free -- no ureter segmentation, no annotation.

WHAT THIS BUYS AND WHAT IT DOES NOT
-----------------------------------
It buys a SEARCH REGION and a DISTANCE REFERENCE. It does not buy a ureter
segmentation, and it is not accurate enough to be one. A ~20 mm corridor
around an interpolated curve is the right tool for "is this dense blob
plausibly in the ureter"; it is the wrong tool for "trace the lumen".

The UVJ rule below is GEOMETRIC AND UNVALIDATED. It places the landmark at the
posterolateral bladder base because that is where the trigone is, but nobody
has checked it against a radiologist's click. Until ~40 studies are annotated
and the error measured, every distance this module reports carries that
unknown. Say so in any output that reaches a clinician.

COORDINATES
-----------
Verified empirically on 8563509 (see the axis check in `orientation_ok`):

    axis 0 increasing -> patient LEFT
    axis 1 increasing -> POSTERIOR
    axis 2 increasing -> SUPERIOR

Every "medial", "lateral", "posterior" below is written against that.
"""



import numpy as np
from scipy import ndimage

# Radius of the search tube around the interpolated centreline. 20 mm is
# deliberately generous: the curve is an estimate, the ureter deviates from it,
# and a MISSED stone cannot be recovered downstream whereas a false positive
# can still be rejected on shape, density and bone/vessel proximity. Tighten
# this only after measuring what it costs in sensitivity.
CORRIDOR_MM = 20.0

UVJ_BASE_FRAC = 0.35

# Fraction of the kidney's height, measured from its top, below which the
# hilum is sought. The PUJ is at the lower half of the hilum.
PUJ_LOWER_FRAC = 0.45

# Points sampled along the centreline. Enough that consecutive points overlap
# at any realistic voxel size, so the drawn path has no gaps.
N_PATH = 400

# Gaussian smoothing (in path samples) applied to the piecewise-linear route
# through the three landmarks. Without it the path has sharp corners at the
# waypoints, which a real ureter does not.
PATH_SMOOTH = 18.0


def orientation_ok(masks):
    """Check the array really is (left+, posterior+, superior+).

    Cheap insurance. Every 'medial' and 'posterior' in this module is a raw
    index comparison, so a study stored in a different orientation would put
    the landmarks on the wrong side SILENTLY -- the corridor would still look
    like a plausible tube, just the wrong one.
    """
    need = ("kidney_left", "kidney_right", "urinary_bladder")
    if not all(k in masks and masks[k].any() for k in need):
        return None                        # cannot check; caller decides
    kl = ndimage.center_of_mass(masks["kidney_left"])
    kr = ndimage.center_of_mass(masks["kidney_right"])
    bl = ndimage.center_of_mass(masks["urinary_bladder"])
    left_ok = kl[0] > kr[0]                # left kidney at higher axis 0
    sup_ok = (kl[2] + kr[2]) / 2 > bl[2]   # kidneys above the bladder
    return bool(left_ok and sup_ok)
# WHAT THIS FUNCTION DOES: confirms the image is stored in the axis order this
# module assumes, by checking two facts that are true of every human: the left
# kidney is to the left of the right kidney, and both sit above the bladder.
# Returns None when the masks needed for the check are missing.


def _midline_x(masks, shape):
    """Axis-0 index of the body midline.

    Taken from the aorta or the vertebrae rather than the image centre: the
    patient is not always centred in the field of view, and 'medial' has to
    mean medial to the PATIENT.
    """
    for k in ("vertebrae_L5", "vertebrae_L1", "aorta"):
        if k in masks and masks[k].any():
            return float(ndimage.center_of_mass(masks[k])[0])
    return shape[0] / 2.0                  # last resort
# WHAT THIS FUNCTION DOES: finds where the patient's midline is along axis 0,
# preferring a spinal structure and falling back to the aorta, so that "medial"
# and "lateral" are measured from the patient rather than from the image.


def landmark_puj(kidney, midline_x):
    """PUJ: the medial point of the lower hilum, where the pelvis becomes ureter.

    The hilum is the medial concavity of the kidney. Restricting to the lower
    part of the kidney before taking the most medial voxel picks the PUJ rather
    than the renal artery/vein entry, which is higher.
    """
    idx = np.argwhere(kidney)
    if not len(idx):
        return None
    z = idx[:, 2]
    # keep the lower PUJ_LOWER_FRAC of the kidney's craniocaudal extent
    zcut = z.min() + (z.max() - z.min()) * PUJ_LOWER_FRAC
    lower = idx[z <= zcut]
    if not len(lower):
        lower = idx
    # medial = closest to the midline along axis 0
    medial = lower[np.argmin(np.abs(lower[:, 0] - midline_x))]
    return medial.astype(float)

# WHAT THIS FUNCTION DOES: returns the voxel coordinate of the pelvi-ureteric
# junction for one kidney, by taking the most medial voxel in the lower part of
# that kidney -- which is where the renal pelvis funnels into the ureter.


def landmark_uvj(bladder, side, midline_x):
    """UVJ: the posterolateral corner of the bladder base on one side.

    The ureters enter the bladder at the two upper corners of the trigone,
    which sits on the bladder BASE (floor), posteriorly, one on each side.

    UNVALIDATED -- see the module docstring. This is anatomy translated into a
    geometric rule, not a rule measured against radiologist clicks.
    """
    idx = np.argwhere(bladder)
    if not len(idx):
        return None
    z = idx[:, 2]
    # keep the lowest UVJ_BASE_FRAC of the bladder: its base, not its dome
    zcut = z.min() + (z.max() - z.min()) * UVJ_BASE_FRAC
    base = idx[z <= zcut]
    if not len(base):
        base = idx
    # restrict to this side of the bladder's own centre, so left and right
    # landmarks cannot collapse onto the same voxel
    bx = base[:, 0].mean()
    half = base[base[:, 0] >= bx] if side == "left" else base[base[:, 0] <= bx]
    if not len(half):
        half = base
    # score = posterior + lateral, in voxels, normalised so neither term
    # dominates purely because the volume is anisotropic
    post = half[:, 1] - half[:, 1].mean()
    lat = (half[:, 0] - midline_x) * (1.0 if side == "left" else -1.0)
    score = post / (post.std() + 1e-6) + lat / (lat.std() + 1e-6)
    return half[int(np.argmax(score))].astype(float)



# WHAT THIS FUNCTION DOES: returns the voxel coordinate where one ureter is
# expected to enter the bladder, by finding the most posterior-and-lateral
# voxel on that side of the bladder floor. This is the reference point that
# "distance to VUJ" is measured from.


def landmark_iliac(masks, side, kidney_pt, bladder_pt):
    """Where the ureter crosses the common iliac artery, at the pelvic brim.

    This is the waypoint that turns a straight line into an anatomical curve.
    The ureter passes immediately ANTERIOR to the artery, so the landmark is
    the artery's course shifted slightly forward (lower axis 1).

    Falls back to the midpoint of PUJ and UVJ when the artery is unavailable,
    which degrades the corridor to the old straight line for that study --
    worth logging, not worth failing on.
    """
    key = f"iliac_artery_{side}"
    if key not in masks or not masks[key].any():
        return (kidney_pt + bladder_pt) / 2.0, False
    idx = np.argwhere(masks[key])
    z = idx[:, 2]
    # the pelvic brim is the TOP of the common iliac, just below the aortic
    # bifurcation; take a thin band there rather than a single voxel so noise
    # in the mask's last slice cannot move the landmark
    band = idx[z >= z.max() - max(2, 0.08 * (z.max() - z.min()))]
    pt = band.mean(axis=0)
    pt[1] -= 4.0                           # ~ a few mm anterior to the artery
    return pt, True
# WHAT THIS FUNCTION DOES: returns the voxel coordinate of the point where the
# ureter crosses the iliac artery at the pelvic brim, taken from the top of the
# iliac artery mask and nudged forward. The second return value says whether a
# real artery mask was used or whether the function fell back to a midpoint.


def centreline(puj, iliac, uvj, n=N_PATH, smooth=PATH_SMOOTH):
    """A smooth path through the three landmarks, in voxel coordinates.

    Piecewise-linear first so the curve genuinely PASSES THROUGH every
    landmark (a Bezier would only be pulled toward the middle one), then
    Gaussian-smoothed to round off the two corners. The endpoints are pinned
    afterwards, because smoothing otherwise drags them inward and the UVJ is
    the very point we measure distances from.
    """
    legs = [(puj, iliac), (iliac, uvj)]
    pts = []
    for a, b in legs:
        for t in np.linspace(0, 1, n // 2, endpoint=False):
            pts.append(a + (b - a) * t)
    pts.append(np.asarray(uvj, float))
    p = np.array(pts, float)
    if smooth > 0 and len(p) > 8:
        # smooth each coordinate along the path; 'nearest' keeps the ends put
        p = np.stack([ndimage.gaussian_filter1d(p[:, k], smooth, mode="nearest")
                      for k in range(3)], axis=1)
    p[0], p[-1] = np.asarray(puj, float), np.asarray(uvj, float)
    return p
# WHAT THIS FUNCTION DOES: turns the three anatomical landmarks into a dense
# list of points tracing the expected course of the ureter, smoothed so it
# curves the way a ureter does instead of bending sharply at each waypoint.


def path_to_mask(path, shape):
    """Mark every voxel the path passes through."""
    m = np.zeros(shape, bool)
    idx = np.rint(path).astype(int)
    ok = np.all((idx >= 0) & (idx < np.array(shape)), axis=1)
    idx = idx[ok]
    if len(idx):
        m[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return m
# WHAT THIS FUNCTION DOES: converts the list of centreline points into a
# boolean volume with those voxels set, which is what the distance transform
# needs in order to build the corridor around it.


def arclen_mm(path, spacing):
    """Cumulative distance along the path, in mm, from its FIRST point.

    Needed because "3 cm above the VUJ" is a distance ALONG the ureter, not a
    straight line. Near the VUJ the two agree closely; over the pelvic brim
    they do not.
    """
    d = np.diff(path, axis=0) * np.asarray(spacing)
    step = np.linalg.norm(d, axis=1)
    return np.concatenate([[0.0], np.cumsum(step)])
# WHAT THIS FUNCTION DOES: measures how far along the ureter path each sampled
# point lies, in millimetres, so that a stone's position can be reported as a
# distance travelled along the tract rather than as a straight-line shortcut.


def build(masks, shape, spacing, radius_mm=CORRIDOR_MM):
    """Per-side corridor masks, centrelines and landmarks for one study.

    Returns {side: dict} with keys:
        corridor   bool volume, the search region
        path       (N,3) centreline in voxel coords, PUJ -> UVJ
        arclen     (N,)  mm along the path from the PUJ
        puj, iliac, uvj    landmark voxel coords
        iliac_real bool, False if the artery mask was missing
    Sides with no kidney or no bladder are omitted entirely.
    """
    out = {}
    bladder = masks.get("urinary_bladder")
    if bladder is None or not bladder.any():
        return out                          # no UVJ -> nothing to measure from
    mx = _midline_x(masks, shape)

    for side in ("left", "right"):
        kid = masks.get(f"kidney_{side}")
        if kid is None or not kid.any():
            continue
        puj = landmark_puj(kid, mx)
        uvj = landmark_uvj(bladder, side, mx)
        if puj is None or uvj is None:
            continue
        ili, real = landmark_iliac(masks, side, puj, uvj)
        path = centreline(puj, ili, uvj)
        # distance transform from the drawn path -> a tube of the given radius
        pm = path_to_mask(path, shape)
        if not pm.any():
            continue
        d = ndimage.distance_transform_edt(~pm, sampling=spacing)
        out[side] = {
            "corridor": d <= radius_mm,
            "dist_to_path": d,              # reused as a phlebolith cue
            "path": path,
            "arclen": arclen_mm(path, spacing),
            "puj": puj, "iliac": ili, "uvj": uvj,
            "iliac_real": real,
        }
    return out
# WHAT THIS FUNCTION DOES: the entry point. For each side it locates the three
# landmarks, traces a smooth curve between them, and inflates that curve into a
# tube -- the region in which a ureteric stone can plausibly lie. It also keeps
# the distance-to-centreline map, because how far a dense blob sits off the
# expected course is one of the better clues that it is a phlebolith.


def zone_bounds(masks):
    """Craniocaudal boundaries of the three ureteric thirds, as axis-2 indices.

    Radiologists divide the ureter by BONE, not by the ureter itself:

        upper / proximal : PUJ            -> top of sacrum
        mid              : top of sacrum  -> bottom of sacrum
        lower / distal   : bottom of sacrum -> UVJ

    So the zone label costs no annotation at all: it is a lookup against the
    sacrum mask, which TotalSegmentator already produces.
    """
    s = masks.get("sacrum")
    if s is None or not s.any():
        return None
    z = np.where(s.any(axis=(0, 1)))[0]
    return {"sacrum_top": int(z.max()), "sacrum_bottom": int(z.min())}
# WHAT THIS FUNCTION DOES: reads the top and bottom of the sacrum, which are
# the standard landmarks separating the upper, mid and lower thirds of the
# ureter. Returns None when the sacrum was not segmented, in which case the
# caller has to fall back on distance along the path.


# Distance along the tract from the UVJ, in mm, used for the lower two thirds.
#
# WHY NOT THE SACRUM FOR THE LOWER BOUNDARY
# TotalSegmentator's `sacrum` class includes the coccyx and runs BELOW the
# bladder: on 8563509 the sacrum spans z 56-139 while the bladder is z 57-79.
# So "below the bottom of the sacrum" is essentially never true, and the first
# version of this function returned "mid" for every distal stone -- both VUJ
# calculi in the test set came back as mid ureteric.
#
# The sacrum TOP is still used, because it is a real and stable landmark for the
# upper/mid boundary and it classified the two proximal stones correctly.



VUJ_MM = 10.0        # at the junction itself
DISTAL_MM = 50.0     # lower/distal ureter


def classify_zone(z_index, bounds, mm_to_uvj=None):
    """Where along the ureter a candidate lies, in the words a report uses.

    upper : above the top of the sacrum
    mid   : below that, more than DISTAL_MM along the tract from the UVJ
    lower : within DISTAL_MM of the UVJ
    vuj   : within VUJ_MM of the UVJ

    mm_to_uvj is the arc length along the interpolated centreline, not a
    straight line, because that is what "3 cm above the VUJ" means.
    """
    if bounds is not None and z_index > bounds["sacrum_top"]:
        return "upper"
    if mm_to_uvj is None:
        return "mid" if bounds is not None else "unknown"
    if mm_to_uvj <= VUJ_MM:
        return "vuj"
    if mm_to_uvj <= DISTAL_MM:
        return "lower"
    return "mid"
# WHAT THIS FUNCTION DOES: turns a stone's height in the volume into the words
# a radiologist uses -- upper, mid or lower ureteric -- using the sacrum as the
# dividing landmark.


