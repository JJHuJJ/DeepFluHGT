# data_pipeline/tests/test_tokenizer.py
"""
Single-nucleotide tokenizer tests.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from data_pipeline.impl.tokenizer import SingleNucleotideTokenizer


tok = SingleNucleotideTokenizer()


def test_vocab_constants():
    """validation vocab constant."""
    assert tok.PAD_ID == 0
    assert tok.CLS_ID == 1
    assert tok.SEP_ID == 2
    assert tok.MASK_ID == 3
    assert tok.UNK_ID == 4
    assert tok.A_ID == 5
    assert tok.T_ID == 6
    assert tok.C_ID == 7
    assert tok.G_ID == 8
    assert tok.N_ID == 9
    assert tok.VOCAB_SIZE == 10
    return True


def test_basic_tokenize():
    """Basic tokenize: ATCG → [CLS, A, T, C, G, SEP]."""
    ids = tok.tokenize("ATCG")
    assert ids == [1, 5, 6, 7, 8, 2], f"got {ids}"
    return True


def test_lowercase():
    """Lowercase input is auto-uppercased."""
    ids = tok.tokenize("atcg")
    assert ids == [1, 5, 6, 7, 8, 2]
    return True


def test_n_base():
    """N base is mapped correctly."""
    ids = tok.tokenize("AN")
    assert ids[1] == 5   # A
    assert ids[2] == 9   # N
    return True


def test_iupac_unk():
    """IUPAC ambiguity codes map to UNK."""
    for ch in "RYSWKMDHBV":
        ids = tok.tokenize(ch)
        assert ids[1] == tok.UNK_ID, f"{ch} should be UNK, got {ids[1]}"
    return True


def test_empty_sequence():
    """nullsequence: [CLS, SEP]."""
    ids = tok.tokenize("")
    assert ids == [1, 2]
    return True


def test_pe_tokenize():
    """PE tokenize: [CLS] R1 [SEP] R2 [SEP]."""
    r1 = "A" * 150
    r2 = "T" * 150
    ids = tok.tokenize_pe(r1, r2)

    # length = CLS + 150 + SEP + 150 + SEP = 303
    assert len(ids) == 303, f"len={len(ids)}"
    assert ids[0] == tok.CLS_ID
    assert ids[151] == tok.SEP_ID   # end of R1
    assert ids[152] == tok.T_ID     # first base of R2
    assert ids[-1] == tok.SEP_ID
    return True


def test_pe_different_lengths():
    """PE with R1/R2 of different lengths."""
    r1 = "ATCG"
    r2 = "GC"
    ids = tok.tokenize_pe(r1, r2)
    # CLS + 4 + SEP + 2 + SEP = 9
    assert len(ids) == 9
    assert ids == [1, 5, 6, 7, 8, 2, 8, 7, 2]  # G=8, C=7
    return True


def test_ntv3_remap():
    """NTv3 ID mapping."""
    # canonical [CLS=1, A=5] → NTv3 [cls=3, A=6]
    mapped = tok.remap_to_ntv3([1, 5])
    assert mapped == [3, 6], f"got {mapped}"
    return True


def test_caduceus_remap():
    """Caduceus ID mapping."""
    # canonical [CLS=1, A=5] → Caduceus [cls=0, A=7]
    mapped = tok.remap_to_caduceus([1, 5])
    assert mapped == [0, 7], f"got {mapped}"
    return True


def test_region_targets():
    """Virus-region targets: [CLS] + base_targets + [SEP]."""
    from data_pipeline.contracts import BreakpointEvent
    events = [BreakpointEvent(hgt_start=5, hgt_end=10)]
    targets = tok.create_region_targets(events, 20)

    assert len(targets) == 22  # CLS + 20 + SEP
    assert targets[0] == 0.0   # CLS
    assert targets[6] == 1.0   # pos 5
    assert targets[10] == 1.0  # pos 9
    assert targets[11] == 0.0  # pos 10
    assert targets[-1] == 0.0  # SEP
    return True


def test_region_targets_multi_event():
    """Multi-event region targets."""
    from data_pipeline.contracts import BreakpointEvent
    events = [
        BreakpointEvent(hgt_start=5, hgt_end=10),
        BreakpointEvent(hgt_start=20, hgt_end=25),
    ]
    targets = tok.create_region_targets(events, 30)
    assert targets[6] == 1.0   # pos 5
    assert targets[21] == 1.0  # pos 20
    assert targets[25] == 1.0  # pos 24
    return True


def test_breakpoint_targets():
    """Breakpoint boundary targets."""
    from data_pipeline.contracts import BreakpointEvent
    events = [BreakpointEvent(hgt_start=5, hgt_end=10)]
    targets = tok.create_breakpoint_targets(events, 20)

    # hgt_start=5 → idx 6 (left boundary)
    assert targets[6] == 1.0
    # hgt_end=10 → idx 10 (hgt_end-1=9)
    assert targets[10] == 1.0
    # Interior positions are not boundaries
    assert targets[7] == 0.0
    return True


def test_negative_targets():
    """Negative-sample targets are all zero."""
    from data_pipeline.contracts import BreakpointEvent
    events = [BreakpointEvent(hgt_start=-1, hgt_end=-1)]
    region = tok.create_region_targets(events, 20)
    bp = tok.create_breakpoint_targets(events, 20)
    assert all(v == 0.0 for v in region)
    assert all(v == 0.0 for v in bp)
    return True


def run():
    tests = [
        test_vocab_constants,
        test_basic_tokenize,
        test_lowercase,
        test_n_base,
        test_iupac_unk,
        test_empty_sequence,
        test_pe_tokenize,
        test_pe_different_lengths,
        test_ntv3_remap,
        test_caduceus_remap,
        test_region_targets,
        test_region_targets_multi_event,
        test_breakpoint_targets,
        test_negative_targets,
    ]
    results = []
    for t in tests:
        try:
            ok = t()
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {t.__name__}")
            results.append(ok)
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            results.append(False)
    return all(results)


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
