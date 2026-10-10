# simulator/tests/test_provenance.py
"""
Provenance tests — provenance segment decomposition, read coordinate projection,
composite host reconstruction.

⚠️ 2026-06-21 fix (R2-C2): return True/False → assert (real validation under pytest).
"""

import random

from ..impl.sequence_generator import (
    compute_insert_segments,
    compute_insert_segments_multi,
    compute_read_segments,
    build_composite_host,
)


# =========================================================================== #
# Mock Sampler
# =========================================================================== #

class MockSampler:
    """Mock sampler for testing without real genome files."""
    def __init__(self):
        self.max_len = 10000
        self._counter = 0

    def sample_fragment_with_provenance(self, length, use_coverage=False):
        self._counter += 1
        seq = f"{'A' * (length // 4)}{'T' * (length // 4)}" \
              f"{'G' * (length // 4)}{'C' * (length - 3 * (length // 4))}"
        prov = {
            'file': f'genome_{self._counter}.fa',
            'seq_id': f'seq_{self._counter}',
            'genome_start': 0,
            'genome_end': length,
            'strand': '+',
        }
        return seq, prov


# =========================================================================== #
# compute_insert_segments — single-event provenance
# =========================================================================== #

def test_insert_segments_single_event():
    """Single event: 3 segments (HOST, VIRUS, HOST)."""
    bg_prov = {
        'file': 'microbe.fa', 'seq_id': 'seq1',
        'genome_start': 1000, 'genome_end': 1500, 'strand': '+',
    }
    vprov = {
        'file': 'virus.fa', 'seq_id': 'v1',
        'genome_start': 200, 'genome_end': 250, 'strand': '+',
    }
    segs = compute_insert_segments(
        bg_prov, vprov,
        hgt_pos=100, bg_right_offset=100,
        hgt_start=100, hgt_end=150, insert_len=300)

    assert len(segs) == 3, f"Expected 3 segs, got {len(segs)}"
    types = [s['seg_type'] for s in segs]
    assert types == ['HOST', 'VIRUS', 'HOST'], f"Types: {types}"
    for i in range(len(segs) - 1):
        assert segs[i]['insert_end'] == segs[i + 1]['insert_start'], \
            f"Gap between seg {i} and {i+1}"


# =========================================================================== #
# compute_insert_segments_multi — multi-event provenance
# =========================================================================== #

def test_insert_segments_multi():
    """2 events: 5 segments (HOST, VIRUS, HOST, VIRUS, HOST)."""
    bg_prov = {
        'file': 'microbe.fa', 'seq_id': 'seq1',
        'genome_start': 1000, 'genome_end': 1500, 'strand': '+',
    }
    vprov1 = {
        'file': 'virus.fa', 'seq_id': 'v1',
        'genome_start': 200, 'genome_end': 250, 'strand': '+',
    }
    vprov2 = {
        'file': 'virus.fa', 'seq_id': 'v2',
        'genome_start': 300, 'genome_end': 340, 'strand': '-',
    }
    event_meta = [
        {'hgt_s': 60, 'hgt_e': 110, 'mode': 'insertion',
         'bg_left_end': 60, 'bg_right_offset': 60, 'v_len': 50},
        {'hgt_s': 170, 'hgt_e': 210, 'mode': 'insertion',
         'bg_left_end': 120, 'bg_right_offset': 120, 'v_len': 40},
    ]
    insert_len = 250

    segs = compute_insert_segments_multi(
        [bg_prov], [vprov1, vprov2], event_meta, insert_len)

    assert len(segs) == 5, f"Expected 5 segs, got {len(segs)}"
    types = [s['seg_type'] for s in segs]
    assert types == ['HOST', 'VIRUS', 'HOST', 'VIRUS', 'HOST'], f"Types: {types}"

    for i in range(len(segs) - 1):
        assert segs[i]['insert_start'] <= segs[i + 1]['insert_start'], \
            f"Not sorted at index {i}"
    for i in range(len(segs) - 1):
        assert segs[i]['insert_end'] == segs[i + 1]['insert_start'], \
            f"Gap at index {i}"

    assert segs[0]['insert_start'] == 0, "First seg doesn't start at 0"
    assert segs[-1]['insert_end'] == insert_len, \
        f"Last seg doesn't end at {insert_len}"


# =========================================================================== #
# compute_read_segments — read coordinate projection
# =========================================================================== #

def test_read_segments_r1():
    """R1 coordinate projection: slice insert[0:150]."""
    insert_segs = [
        {'seg_type': 'HOST', 'source_type': 'microbe',
         'source_file': 'm.fa', 'source_seq_id': 's1',
         'genome_start': 0, 'genome_end': 100, 'strand': '+',
         'insert_start': 0, 'insert_end': 100},
        {'seg_type': 'VIRUS', 'source_type': 'virus',
         'source_file': 'v.fa', 'source_seq_id': 'v1',
         'genome_start': 0, 'genome_end': 200, 'strand': '+',
         'insert_start': 100, 'insert_end': 300},
    ]
    r1_segs = compute_read_segments(insert_segs, 0, 150, is_rc=False)
    assert len(r1_segs) == 2, f"Expected 2 R1 segs, got {len(r1_segs)}"
    assert r1_segs[0]['read_end'] == r1_segs[1]['read_start'], "R1 gap"


def test_read_segments_r2():
    """R2 coordinate projection: reverse-complement transform."""
    insert_segs = [
        {'seg_type': 'HOST', 'source_type': 'microbe',
         'source_file': 'm.fa', 'source_seq_id': 's1',
         'genome_start': 0, 'genome_end': 150, 'strand': '+',
         'insert_start': 0, 'insert_end': 150},
        {'seg_type': 'VIRUS', 'source_type': 'virus',
         'source_file': 'v.fa', 'source_seq_id': 'v1',
         'genome_start': 0, 'genome_end': 150, 'strand': '-',
         'insert_start': 150, 'insert_end': 300},
    ]
    r2_segs = compute_read_segments(insert_segs, 150, 150, is_rc=True)
    assert len(r2_segs) == 1, f"Expected 1 R2 seg, got {len(r2_segs)}"
    assert r2_segs[0]['strand'] == '+', \
        f"R2 strand should be '+', got {r2_segs[0]['strand']}"


# =========================================================================== #
# build_composite_host
# =========================================================================== #

def test_composite_host():
    """3-source composite host: correct length, distinct sources, contiguous coverage."""
    random.seed(42)
    sampler = MockSampler()
    target_len = 600

    composite, bg_prov_list = build_composite_host(sampler, target_len, 3)

    assert len(composite) == target_len, f"len={len(composite)} expected {target_len}"
    assert len(bg_prov_list) == 3, f"prov count={len(bg_prov_list)}"

    sources = set(p['seq_id'] for p in bg_prov_list)
    assert len(sources) == 3, f"sources={len(sources)}"

    assert bg_prov_list[0]['composite_start'] == 0, "first start != 0"
    for i in range(1, len(bg_prov_list)):
        assert bg_prov_list[i]['composite_start'] == bg_prov_list[i - 1]['composite_end'], \
            f"discontinuous at index {i}"
    assert bg_prov_list[-1]['composite_end'] == target_len, "last end != target_len"


def run():
    """Standalone run entry point (pytest is the primary runner; this function is kept for backward-compatible exit codes)."""
    tests = [
        test_insert_segments_single_event,
        test_insert_segments_multi,
        test_read_segments_r1,
        test_read_segments_r2,
        test_composite_host,
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
