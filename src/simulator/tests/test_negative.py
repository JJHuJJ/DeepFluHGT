# simulator/tests/test_negative.py
"""
Negative-sample generation tests — 4 types: NEG, NEG_VIRUS, NEG_HARD, NEG_PCR.

⚠️ 2026-06-21 fix (R2-C2): return True/False → assert (real validation under pytest).
"""

import random

from ..impl.sequence_generator import (
    generate_virus_only,
    generate_pcr_chimera,
    generate_negative_hard,
    generate_host_only,
    generate_long_sequence_from_fragments,
)


def _dummy_prov():
    return {
        'file': 'test.fa', 'seq_id': 'test_seq',
        'genome_start': 0, 'genome_end': 200, 'strand': '+',
    }


def test_long_sequence_from_fragments():
    """Assemble fragments into a sequence of arbitrary length."""
    random.seed(2)
    frags = [('ATCG' * 50, _dummy_prov()),
             ('GCTA' * 50, _dummy_prov()),
             ('TTAA' * 50, _dummy_prov())]
    seq = generate_long_sequence_from_fragments([s for s, _ in frags], 1000)
    assert len(seq) == 1000, f"Expected 1000, got {len(seq)}"
    seq = generate_long_sequence_from_fragments([s for s, _ in frags], 50_000)
    assert len(seq) == 50_000, f"Expected 50000, got {len(seq)}"


def test_generate_virus_only():
    """Pure-virus negative sample."""
    random.seed(2)
    frags = [('ATCG' * 50, _dummy_prov()),
             ('GCTA' * 50, _dummy_prov())]
    seq, segs = generate_virus_only(frags, 300)
    assert len(seq) == 300, f"len={len(seq)}"
    assert all(seg['seg_type'] == 'VIRUS' for seg in segs), \
        f"Expected all VIRUS, got {[s['seg_type'] for s in segs]}"


def test_generate_pcr_chimera():
    """PCR chimera molecule."""
    random.seed(2)
    pool = [('AAAAGGGG' * 37, _dummy_prov()),
            ('CCCCTTTT' * 37, _dummy_prov())]
    seq, segs = generate_pcr_chimera(pool, 300)
    assert 0 < len(seq) <= 300, f"len={len(seq)}"
    assert all(seg['seg_type'] == 'HOST' for seg in segs), \
        f"Expected all HOST, got {[s['seg_type'] for s in segs]}"


def test_generate_negative_hard():
    """Hard negative sample."""
    random.seed(2)
    bg = 'N' * 300
    frags = [('ATCG' * 50, _dummy_prov())]
    pool = [('AAAAGGGG' * 37, _dummy_prov()),
            ('CCCCTTTT' * 37, _dummy_prov())]
    seq, segs = generate_negative_hard(bg, _dummy_prov(), frags, pool)
    assert 0 < len(seq) <= 300, f"len={len(seq)}"


def test_generate_host_only():
    """Pure-host provenance."""
    prov = _dummy_prov()
    segs = generate_host_only(prov, 300)
    assert len(segs) == 1, f"Expected 1 seg, got {len(segs)}"
    assert segs[0]['seg_type'] == 'HOST', f"Expected HOST, got {segs[0]['seg_type']}"
    assert segs[0]['insert_end'] == 300, f"insert_end={segs[0]['insert_end']}"


def run():
    """Standalone run entry point (pytest is the primary runner; this function is kept for backward-compatible exit codes)."""
    tests = [
        test_long_sequence_from_fragments,
        test_generate_virus_only,
        test_generate_pcr_chimera,
        test_generate_negative_hard,
        test_generate_host_only,
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
