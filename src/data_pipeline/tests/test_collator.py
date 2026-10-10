# data_pipeline/tests/test_collator.py
"""
HGTCollator batch tests.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from data_pipeline.contracts import HGTSampleTokenized
from data_pipeline.impl.collator import HGTCollator, HardNegativeBalancer


def _make_tokenized(seq_id, label, length, sample_type="POS"):
    """create mock tokenized sample."""
    return HGTSampleTokenized(
        input_ids=[5] * length,
        attention_mask=[1] * length,
        bio_features=np.zeros(260, dtype=np.float32),
        label=label,
        region_targets=[float(label)] * length,
        breakpoint_targets=[0.0] * length,
        pe_case=0,
        hgt_mode=0,
        n_events=1 if label == 1 else 0,
        seq_id=seq_id,
        sample_type=sample_type,
    )


def test_batch_shape():
    """Batch shape is correct [B, L]."""
    collator = HGTCollator()
    samples = [
        _make_tokenized("s1", 1, 10),
        _make_tokenized("s2", 0, 8),
    ]
    batch = collator(samples)

    assert batch.input_ids.shape == (2, 10), f"shape={batch.input_ids.shape}"
    assert batch.labels.shape == (2,), f"shape={batch.labels.shape}"
    assert batch.bio_features.shape == (2, 260)
    return True


def test_attention_mask_padding():
    """attention_mask correctly marks padding."""
    collator = HGTCollator()
    samples = [
        _make_tokenized("s1", 1, 10),
        _make_tokenized("s2", 0, 8),
    ]
    batch = collator(samples)

    # s2 is padded to 10; the last 2 positions are 0
    assert batch.attention_mask[1, :8].sum().item() == 8
    assert batch.attention_mask[1, 8:].sum().item() == 0
    return True


def test_target_shapes():
    """breakpoint_targets and region_targets shapes match input_ids."""
    collator = HGTCollator()
    samples = [
        _make_tokenized("s1", 1, 10),
        _make_tokenized("s2", 0, 5),
    ]
    batch = collator(samples)

    L = batch.input_ids.shape[1]
    assert batch.breakpoint_targets.shape == (2, L)
    assert batch.region_targets.shape == (2, L)
    return True


def test_labels():
    """labels are passed through correctly."""
    collator = HGTCollator()
    samples = [
        _make_tokenized("s1", 1, 10),
        _make_tokenized("s2", 0, 10),
    ]
    batch = collator(samples)
    assert batch.labels[0].item() == 1
    assert batch.labels[1].item() == 0
    return True


def test_metadata():
    """Metadata is passed through correctly."""
    collator = HGTCollator()
    samples = [
        _make_tokenized("s1", 1, 10, "POS"),
        _make_tokenized("s2", 0, 10, "NEG"),
    ]
    batch = collator(samples)
    assert batch.seq_ids == ["s1", "s2"]
    assert batch.sample_types == ["POS", "NEG"]
    return True


def test_single_sample():
    """Single-sample batch."""
    collator = HGTCollator()
    samples = [_make_tokenized("s1", 1, 5)]
    batch = collator(samples)
    assert batch.input_ids.shape == (1, 5)
    return True


def test_hard_neg_balancer():
    """HardNegativeBalancer balanced sampling."""
    balancer = HardNegativeBalancer(balanced_neg_types=True)

    samples = []
    for i in range(10):
        samples.append(_make_tokenized(f"pos_{i}", 1, 10, "POS"))
    for i in range(10):
        samples.append(_make_tokenized(f"neg_{i}", 0, 10, "NEG"))
    for i in range(5):
        samples.append(_make_tokenized(f"nv_{i}", 0, 10, "NEG_VIRUS"))
    for i in range(5):
        samples.append(_make_tokenized(f"nh_{i}", 0, 10, "NEG_HARD"))
    for i in range(3):
        samples.append(_make_tokenized(f"np_{i}", 0, 10, "NEG_PCR"))

    result = balancer.rebalance(samples, target_batch_size=8)

    assert len(result) == 8
    pos_count = sum(1 for s in result if s.label == 1)
    assert pos_count == 4  # 50% pos
    return True


def test_empty_balancer():
    """Empty list does not crash."""
    balancer = HardNegativeBalancer()
    result = balancer.rebalance([], 8)
    assert result == []
    return True


def run():
    tests = [
        test_batch_shape,
        test_attention_mask_padding,
        test_target_shapes,
        test_labels,
        test_metadata,
        test_single_sample,
        test_hard_neg_balancer,
        test_empty_balancer,
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
