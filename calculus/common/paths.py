"""Where results go.

Large intermediate data (nifti/, seg/, dicoms/) always lives at the project
root and is SHARED between runs -- it is expensive to rebuild (TotalSegmentator
is ~2 min/study) and identical whichever analysis you are doing.

Results (csv/, overlays/) go under RUN, which defaults to the project root but
can be pointed at a per-run folder:

    CALCULUS_RUN=run_full44 ./run_part1.sh

so a new run never overwrites the numbers from an old one.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

# results root -- absolute, or relative to the project root
_run = os.environ.get("CALCULUS_RUN", "").strip()
RUN = ROOT if not _run else (
    _run if os.path.isabs(_run) else os.path.join(ROOT, _run))

CSV = os.path.join(RUN, "csv")
OVERLAYS = os.path.join(RUN, "overlays")

# Shared between runs by default -- rebuilding them is expensive and they do not
# depend on which analysis you are doing.
#
# A SECOND COHORT can override them, so a new dataset gets its own volumes and
# masks instead of mixing into the main cohort's folders:
#     CALCULUS_ZIPS=ureteric_stone_dataset/zips \
#     CALCULUS_NIFTI=ureteric_stone_dataset/nifti \
#     CALCULUS_SEG=ureteric_stone_dataset/seg  ...
# Defaults are unchanged, so every existing command behaves exactly as before.
def _dir(env, *default):
    v = os.environ.get(env, "").strip()
    if not v:
        return os.path.join(ROOT, *default)
    return v if os.path.isabs(v) else os.path.join(ROOT, v)


NIFTI = _dir("CALCULUS_NIFTI", "nifti")
SEG = _dir("CALCULUS_SEG", "seg")
ZIPS = _dir("CALCULUS_ZIPS", "dicoms", "zips")


def ensure():
    """Create the results directories for this run."""
    for d in (CSV, OVERLAYS):
        os.makedirs(d, exist_ok=True)
    return RUN
