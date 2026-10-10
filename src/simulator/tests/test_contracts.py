# simulator/tests/test_contracts.py
"""
Data structure validation tests — SimulatorConfig, ProvenanceSegment, HGTEvent, etc.

⚠️ 2026-06-21 fix (R2-C2): the original `return True/False` silently passed under
pytest (zero validation).
Changed to pytest-style assert + pytest.raises so pytest performs real validation;
run()/__main__ remain backward compatible.
"""

import pytest
from ..contracts import (
    SimulatorConfig,
    ProvenanceSegment,
    HGTEvent,
    SampleMetadata,
    SimulationOutput,
)


def test_simulator_config_defaults():
    """Default configuration should pass validate (no raise)."""
    SimulatorConfig().validate() # no raise means pass


def test_simulator_config_invalid_mode():
    """Invalid mode should trigger AssertionError."""
    with pytest.raises(AssertionError):
        SimulatorConfig(mode="invalid").validate()


def test_simulator_config_invalid_probs():
    """HGT event probabilities not summing to 1 should trigger AssertionError."""
    with pytest.raises(AssertionError):
        SimulatorConfig(insertion_prob=0.5, replacement_prob=0.3, deletion_prob=0.3).validate()


def test_simulator_config_invalid_ratios():
    """Split ratios not summing to 1 should trigger AssertionError."""
    with pytest.raises(AssertionError):
        SimulatorConfig(do_split=True, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3).validate()


def test_provenance_segment():
    seg = ProvenanceSegment(
        seg_type="HOST", source_type="microbe",
        source_file="test.fa", source_seq_id="seq1",
        genome_start=0, genome_end=300,
        strand="+", insert_start=0, insert_end=300,
    )
    assert seg.seg_type == "HOST"
    assert seg.read_start == -1 # default


def test_hgt_event():
    ev = HGTEvent(hgt_start=100, hgt_end=150, virus_len=50, hgt_mode="insertion")
    assert ev.hgt_mode == "insertion"


def test_sample_metadata():
    meta = SampleMetadata(
        seq_id="pos_00000001", label=1, n_events=1,
        hgt_modes=["insertion"], pe_case="R1_MIXED_R2_HOST",
    )
    assert meta.r1_vs == -1 # default
    assert len(meta.events) == 0 # default empty list


def test_simulation_output():
    out = SimulationOutput(sample_count=100, pos_count=50, neg_count=50)
    assert out.fasta_files == {}
    assert out.breakpoints_bed == ""


def run():
    """Standalone run entry point (pytest is the primary runner; this function is kept for backward-compatible exit codes)."""
    tests = [
        test_simulator_config_defaults,
        test_simulator_config_invalid_mode,
        test_simulator_config_invalid_probs,
        test_simulator_config_invalid_ratios,
        test_provenance_segment,
        test_hgt_event,
        test_sample_metadata,
        test_simulation_output,
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
