"""Renal and ureteric calculus detection and measurement on non-contrast CT.

Two detectors over shared plumbing:
    calculus.kidney   stones inside the kidney   - validated
    calculus.ureter   stones in the ureter       - recall validated,
                                                   precision under work
The split is deliberate: they search different regions with different
rejection rules, and tuning the ureteric side must not be able to move the
kidney numbers.
"""

__version__ = "0.1.0"
