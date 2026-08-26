

def test_idose_is_not_junk():
    """Philips names its iterative recon "iDose". A bare `dose` token in
    JUNK_RE threw away every "PLAIN THIN, iDose (4)" series -- the thin
    non-contrast recon we most want -- as a dose-report screenshot. Seven
    studies lost their only usable series to this and scored verdict=skip."""
    from calculus.common.triage_series import JUNK_RE
    for desc in ["PLAIN THIN, iDose (4)", "NCCT THIN, iDose (5)",
                 "CONTRAST 3mm, iDose (3)", "iDose 4"]:
        assert not JUNK_RE.search(desc), f"{desc} must not be junk"
    for desc in ["Dose Report", "DOSE SCREEN", "Patient Protocol dose",
                 "Topogram 0.7", "Exam Summary", "MIP cor"]:
        assert JUNK_RE.search(desc), f"{desc} must be junk"
