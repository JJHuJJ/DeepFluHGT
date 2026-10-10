# data_pipeline/tests/test_dataset.py
"""
HGTDataset test.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from data_pipeline.contracts import HGTSample, BreakpointEvent
from data_pipeline.impl.dataset import HGTDataset


def _make_pe_samples():
    """create PE testsample."""
    pos = HGTSample(
        seq_id="pos_00000000",
        label=1,
        sequence="A" * 150,
        r2_sequence="T" * 150,
        events=[BreakpointEvent(
            hgt_start=50, hgt_end=150, virus_len=100,
            hgt_mode="insertion", insert_len=300,
            r1_vs=-1, r1_ve=-1, r2_vs=0, r2_ve=100,
            pe_case="R1_HOST_R2_MIXED",
        )],
        sample_type="POS",
        seq_len=150,
    )
    neg = HGTSample(
        seq_id="neg_00000000",
        label=0,
        sequence="C" * 150,
        r2_sequence="G" * 150,
        events=[BreakpointEvent(pe_case="NEG")],
        sample_type="NEG",
        seq_len=150,
    )
    return [pos, neg]


def _make_contig_samples():
    """create Contig testsample."""
    pos = HGTSample(
        seq_id="pos_00000000",
        label=1,
        sequence="ATCG" * 100,  # 400bp
        events=[BreakpointEvent(
            hgt_start=100, hgt_end=200, virus_len=100,
            hgt_mode="replacement",
        )],
        sample_type="POS",
        seq_len=400,
    )
    neg = HGTSample(
        seq_id="neg_00000000",
        label=0,
        sequence="GCTA" * 100,
        events=[BreakpointEvent()],
        sample_type="NEG",
        seq_len=400,
    )
    return [pos, neg]


def test_pe_token_count():
    """PE mode: 303 tokens (150+150+3)."""
    ds = HGTDataset(_make_pe_samples(), mode="paired_end")
    item = ds[0]
    assert len(item.input_ids) == 303, f"len={len(item.input_ids)}"
    return True


def test_pe_token_layout():
    """PE mode: [CLS] R1 [SEP] R2 [SEP]."""
    ds = HGTDataset(_make_pe_samples(), mode="paired_end")
    item = ds[0]  # pos sample
    assert item.input_ids[0] == 1       # CLS
    assert item.input_ids[1] == 5       # A
    assert item.input_ids[151] == 2     # SEP after R1
    assert item.input_ids[152] == 6     # T (R2 first base)
    assert item.input_ids[-1] == 2      # SEP
    return True


def test_pe_target_lengths():
    """PE mode: targets lengthwith input_ids consistent."""
    ds = HGTDataset(_make_pe_samples(), mode="paired_end")
    for i in range(len(ds)):
        item = ds[i]
        assert len(item.input_ids) == len(item.region_targets), \
            f"sample {i}: ids={len(item.input_ids)} region={len(item.region_targets)}"
        assert len(item.input_ids) == len(item.breakpoint_targets), \
            f"sample {i}: ids={len(item.input_ids)} bp={len(item.breakpoint_targets)}"
    return True


def test_pe_negative_targets_zero():
    """PE negative samples: targets are all zero."""
    ds = HGTDataset(_make_pe_samples(), mode="paired_end")
    item = ds[1]  # neg
    assert item.label == 0
    assert all(v == 0.0 for v in item.region_targets)
    assert all(v == 0.0 for v in item.breakpoint_targets)
    return True


def test_pe_bio_features():
    """PE mode: bio_features shape=(260,)."""
    ds = HGTDataset(_make_pe_samples(), mode="paired_end")
    item = ds[0]
    assert item.bio_features.shape == (260,)
    return True


def test_contig_token_count():
    """Contig mode: seq_len + 2 tokens."""
    ds = HGTDataset(_make_contig_samples(), mode="contig")
    item = ds[0]  # 400bp contig
    assert len(item.input_ids) == 402, f"len={len(item.input_ids)}"
    return True


def test_contig_region_targets():
    """Contig mode: virus region marked correctly."""
    ds = HGTDataset(_make_contig_samples(), mode="contig")
    item = ds[0]
    # hgt_start=100, hgt_end=200
    assert item.region_targets[101] == 1.0  # pos 100 (+1 for CLS)
    assert item.region_targets[200] == 1.0  # pos 199
    assert item.region_targets[201] == 0.0  # pos 200 (outside)
    assert item.region_targets[100] == 0.0  # pos 99 (before)
    return True


def test_contig_truncation():
    """Contig truncation: max_length limit."""
    long_seq = "ATCG" * 5000  # 20000bp
    samples = [HGTSample(
        seq_id="pos_long", label=1,
        sequence=long_seq,
        events=[BreakpointEvent(hgt_start=5000, hgt_end=5100,
                                hgt_mode="insertion")],
        sample_type="POS", seq_len=20000,
    )]
    ds = HGTDataset(samples, mode="contig", max_length=1000)
    item = ds[0]
    assert len(item.input_ids) == 1002  # 1000 + CLS + SEP
    return True


def test_attention_mask():
    """attention_mask is all 1 (no padding)."""
    ds = HGTDataset(_make_pe_samples(), mode="paired_end")
    item = ds[0]
    assert all(m == 1 for m in item.attention_mask)
    return True


def test_metadata():
    """Metadata fields are correct."""
    ds = HGTDataset(_make_pe_samples(), mode="paired_end")
    item = ds[0]
    assert item.seq_id == "pos_00000000"
    assert item.sample_type == "POS"
    assert item.label == 1
    return True


def run():
    tests = [
        test_pe_token_count,
        test_pe_token_layout,
        test_pe_target_lengths,
        test_pe_negative_targets_zero,
        test_pe_bio_features,
        test_contig_token_count,
        test_contig_region_targets,
        test_contig_truncation,
        test_attention_mask,
        test_metadata,
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
