# simulator/tests/test_pe_classification.py
"""
PE case classification tests — classify_pe_case (single region) +
classify_pe_case_multi (multi region).

⚠️ 2026-06-21 fix (R2-C2): return <bool expr> → assert (real validation under pytest).
"""

from ..impl.sequence_generator import (
    classify_pe_case,
    classify_pe_case_multi,
)


# =========================================================================== #
# classify_pe_case — single region
# =========================================================================== #

def test_r1_mixed_r2_host():
    """Virus inside the R1 region but not covering all of R1; R2 entirely host."""
    ct, r1vs, r1ve, r2vs, r2ve = classify_pe_case(60, 130, 150, 300)
    assert ct == "R1_MIXED_R2_HOST"


def test_r1_host_r2_mixed():
    """Virus inside the R2 region but not covering all of R2."""
    ct, r1vs, r1ve, r2vs, r2ve = classify_pe_case(160, 280, 150, 300)
    assert ct == "R1_HOST_R2_MIXED"


def test_r1_host_r2_virus():
    """R1 entirely host, R2 entirely virus."""
    ct, *_ = classify_pe_case(150, 300, 150, 300)
    assert ct == "R1_HOST_R2_VIRUS"


def test_r1_virus_r2_host():
    """R1 entirely virus, R2 entirely host."""
    ct, *_ = classify_pe_case(0, 150, 150, 300)
    assert ct == "R1_VIRUS_R2_HOST"


def test_r1_mixed_r2_mixed():
    """Virus spans the R1/R2 boundary; both ends are mixed."""
    ct, r1vs, r1ve, r2vs, r2ve = classify_pe_case(100, 200, 150, 300)
    assert ct == "R1_MIXED_R2_MIXED"


def test_r2_coordinate_transform():
    """R2 coordinate transform: virus at [200,280) in an insert of 300."""
    ct, r1vs, r1ve, r2vs, r2ve = classify_pe_case(200, 280, 150, 300)
    assert r2vs == 20 and r2ve == 100


def test_internal():
    """Virus falls inside the gap (insert=500, virus at [200,280))."""
    ct, *_ = classify_pe_case(200, 280, 150, 500)
    assert ct == "INTERNAL"


# =========================================================================== #
# classify_pe_case_multi — multi region
# =========================================================================== #

def test_multi_both_in_r1():
    """Both regions inside R1 -> R1_MIXED_R2_HOST."""
    ct, *_ = classify_pe_case_multi([(20, 60), (80, 120)], 150, 300)
    assert ct == "R1_MIXED_R2_HOST"


def test_multi_one_r1_one_r2():
    """One inside R1, one inside R2 -> R1_MIXED_R2_MIXED."""
    ct, *_ = classify_pe_case_multi([(20, 60), (200, 280)], 150, 300)
    assert ct == "R1_MIXED_R2_MIXED"


def test_multi_r1_fully_covered():
    """R1 fully covered -> R1_VIRUS_R2_HOST."""
    ct, *_ = classify_pe_case_multi([(0, 80), (80, 150)], 150, 300)
    assert ct == "R1_VIRUS_R2_HOST"


def test_multi_internal():
    """All inside the gap -> INTERNAL."""
    ct, *_ = classify_pe_case_multi([(200, 220), (250, 280)], 150, 500)
    assert ct == "INTERNAL"


def test_multi_single_region_degenerate():
    """A single region degenerates to classify_pe_case."""
    ct, *_ = classify_pe_case_multi([(150, 300)], 150, 300)
    assert ct == "R1_HOST_R2_VIRUS"


def run():
    """Standalone run entry point (pytest is the primary runner; this function is kept for backward-compatible exit codes)."""
    tests = [
        test_r1_mixed_r2_host, test_r1_host_r2_mixed, test_r1_host_r2_virus,
        test_r1_virus_r2_host, test_r1_mixed_r2_mixed, test_r2_coordinate_transform,
        test_internal, test_multi_both_in_r1, test_multi_one_r1_one_r2,
        test_multi_r1_fully_covered, test_multi_internal, test_multi_single_region_degenerate,
    ]
    results = []
    for t in tests:
        try:
            t()
            print(f" [PASS] {t.__name__}")
            results.append(True)
        except Exception as e:
            print(f" [FAIL] {t.__name__}: {e}")
            results.append(False)
    return all(results)


if __name__ == '__main__':
    import sys
    sys.exit(0 if run() else 1)
