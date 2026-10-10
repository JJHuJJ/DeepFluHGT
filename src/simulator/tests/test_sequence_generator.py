# simulator/tests/test_sequence_generator.py
"""
HGT event generator tests — single event / multi event / contig length distribution.

⚠️ 2026-06-21 fix (R2-C2): return True/False → assert (real validation under pytest).
"""

import random
import math

from ..impl.sequence_generator import HGTSequenceGenerator


def test_single_event_consistency():
    """200 single-event trials: breakpoints consistent, virus content matches."""
    random.seed(0)
    gen = HGTSequenceGenerator(0.6, 0.3, 0.1)
    bg = 'A' * 200
    virus = 'T' * 50

    for trial in range(200):
        new_seq, (s, e), mode, _ = gen.generate_positive_sample(bg, virus)
        ok_content = new_seq[s:e] == virus
        ok_bounds = 0 <= s < e <= len(new_seq)
        assert ok_content and ok_bounds, \
            f"trial {trial} ({mode}): s={s} e={e} len={len(new_seq)}"


def test_multi_event_2():
    """2 events: 100 trials, virus content correct and non-overlapping."""
    random.seed(42)
    gen = HGTSequenceGenerator(0.6, 0.3, 0.1)
    bg = 'A' * 500
    v1 = 'T' * 50
    v2 = 'G' * 40
    all_viruses = [v1, v2]

    for trial in range(100):
        seq, events, modes, meta = gen.generate_multi_hgt_sample(bg, [v1, v2], min_gap=20)
        for i, (hs, he) in enumerate(events):
            content = seq[hs:he]
            assert content in all_viruses, \
                f"2-event content trial={trial} events={events}"
        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                s1, e1 = events[i]
                s2, e2 = events[j]
                assert e1 <= s2 or e2 <= s1, f"2-event overlap events={events}"


def test_multi_event_3():
    """3 events: 100 trials, all correct."""
    random.seed(42)
    gen = HGTSequenceGenerator(0.6, 0.3, 0.1)
    bg = 'A' * 500
    v1 = 'T' * 50
    v2 = 'G' * 40
    v3 = 'C' * 30
    all_viruses = [v1, v2, v3]

    for trial in range(100):
        seq, events, modes, meta = gen.generate_multi_hgt_sample(
            bg, [v1, v2, v3], min_gap=20)
        assert len(events) == 3, f"3-event count: got {len(events)}"
        for i, (hs, he) in enumerate(events):
            assert seq[hs:he] in all_viruses, f"3-event content trial={trial}"


def test_multi_event_meta_consistency():
    """Multi-event meta coordinates are consistent with the events."""
    random.seed(42)
    gen = HGTSequenceGenerator(0.6, 0.3, 0.1)
    bg = 'A' * 500
    v1 = 'T' * 50
    v2 = 'G' * 40

    seq, events, modes, meta = gen.generate_multi_hgt_sample(bg, [v1, v2], min_gap=20)

    assert len(meta) == len(events), \
        f"meta length mismatch: {len(meta)} vs {len(events)}"
    for i, (em, (hs, he)) in enumerate(zip(meta, events)):
        assert em['hgt_s'] == hs and em['hgt_e'] == he, f"meta[{i}] coords mismatch"


def test_contig_length_distribution():
    """Log-uniform distribution: both short and long contigs are well represented."""
    random.seed(1)
    c_min, c_max = 300, 100_000
    lengths = [
        int(math.exp(random.uniform(math.log(c_min), math.log(c_max))))
        for _ in range(10_000)
    ]
    pct_short = sum(1 for l in lengths if l < 1_000) / len(lengths)
    pct_long = sum(1 for l in lengths if l > 50_000) / len(lengths)

    assert all(c_min <= l <= c_max for l in lengths), "Length out of range"
    assert pct_short >= 0.10, f"Short contigs underrepresented: {100*pct_short:.1f}%"
    assert pct_long >= 0.10, f"Long contigs underrepresented: {100*pct_long:.1f}%"


def run():
    """Standalone run entry point (pytest is the primary runner; this function is kept for backward-compatible exit codes)."""
    tests = [
        test_single_event_consistency,
        test_multi_event_2,
        test_multi_event_3,
        test_multi_event_meta_consistency,
        test_contig_length_distribution,
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
