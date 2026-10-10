# model/tests/test_forward_shape.py
"""
Forward shape test  —  validates the dimensions of the model's output tensors.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import torch

from model.contracts import HGTConfig
from model.api import build_model
from data_pipeline.contracts import HGTBatch


def _make_pe_batch(B=4, L=303):
    """Build a synthetic PE batch."""
    return HGTBatch(
        input_ids=torch.randint(0, 10, (B, L)),
        attention_mask=torch.ones(B, L, dtype=torch.long),
        bio_features=torch.randn(B, 260),
        labels=torch.tensor([1, 0, 1, 0]),
        breakpoint_targets=torch.zeros(B, L),
        region_targets=torch.zeros(B, L),
        pe_case=torch.zeros(B, dtype=torch.long),
        hgt_mode=torch.zeros(B, dtype=torch.long),
        n_events=torch.tensor([1, 0, 2, 0]),
    )


def _make_contig_batch(B=4, L=402):
    """Build a synthetic contig batch."""
    return HGTBatch(
        input_ids=torch.randint(0, 10, (B, L)),
        attention_mask=torch.ones(B, L, dtype=torch.long),
        bio_features=torch.randn(B, 260),
        labels=torch.tensor([1, 0, 1, 0]),
        breakpoint_targets=torch.zeros(B, L),
        region_targets=torch.zeros(B, L),
        pe_case=torch.zeros(B, dtype=torch.long),
        hgt_mode=torch.zeros(B, dtype=torch.long),
        n_events=torch.tensor([1, 0, 1, 0]),
    )


def test_pe_forward_shape():
    """PE mode forward shape test."""
    config = HGTConfig(
        data_mode="paired_end",
        _smoke_global_layers=2,
        _smoke_local_layers=2,
        pretrained_nt_path="",
        pretrained_caduceus_path="",
    )
    model = build_model(config)
    model.eval()

    B, L = 4, 303
    batch = _make_pe_batch(B, L)

    with torch.no_grad():
        out = model(batch)

    assert out.cls_logits.shape == (B, 1), \
        f"cls_logits: expected [{B}, 1], got {list(out.cls_logits.shape)}"
    assert out.bp_heatmaps.shape == (B, L), \
        f"bp_heatmaps: expected [{B}, {L}], got {list(out.bp_heatmaps.shape)}"
    assert out.role_logits.shape == (B, 1)
    assert out.direction_logits.shape == (B, 1)
    assert out.bp_refined.shape == (B, 100), \
        f"bp_refined: expected [{B}, 100], got {list(out.bp_refined.shape)}"
    assert out.split_gates.shape == (B, L)
    assert out.donor_pool.shape == (B, 768)
    assert out.recipient_pool.shape == (B, 768)
    assert out.hidden_states.shape == (B, L, 768)
    assert out.rc_cls_logits is not None
    assert out.rc_cls_logits.shape == (B, 1)
    assert out.early_exit_fusion is not None
    assert out.early_exit_graph is not None

    return True


def test_contig_forward_shape():
    """Contig mode forward shape test."""
    config = HGTConfig(
        data_mode="contig",
        _smoke_global_layers=2,
        _smoke_local_layers=2,
        pretrained_nt_path="",
        pretrained_caduceus_path="",
    )
    model = build_model(config)
    model.eval()

    B, L = 4, 402
    batch = _make_contig_batch(B, L)

    with torch.no_grad():
        out = model(batch)

    assert out.cls_logits.shape == (B, 1)
    assert out.bp_heatmaps.shape == (B, L), \
        f"bp_heatmaps: expected [{B}, {L}], got {list(out.bp_heatmaps.shape)}"
    assert out.role_logits.shape == (B, 1)
    assert out.direction_logits.shape == (B, 1)
    assert out.bp_refined.shape == (B, 100)
    assert out.split_gates.shape == (B, L)
    assert out.donor_pool.shape == (B, 768)
    assert out.recipient_pool.shape == (B, 768)
    assert out.hidden_states.shape == (B, L, 768)
    assert out.rc_cls_logits is None, \
        f"Contig mode: rc_cls_logits should be None"

    return True


def test_batch_size_1():
    """batch_size=1 test (graph interaction edge case)."""
    config = HGTConfig(
        data_mode="paired_end",
        _smoke_global_layers=2,
        _smoke_local_layers=2,
        pretrained_nt_path="",
        pretrained_caduceus_path="",
    )
    model = build_model(config)
    model.eval()

    batch = _make_pe_batch(B=1, L=303)
    with torch.no_grad():
        out = model(batch)

    assert out.cls_logits.shape == (1, 1)
    assert out.bp_heatmaps.shape == (1, 303)
    return True


def test_variable_length_contig():
    """Variable-length contig test."""
    config = HGTConfig(
        data_mode="contig",
        _smoke_global_layers=2,
        _smoke_local_layers=2,
        pretrained_nt_path="",
        pretrained_caduceus_path="",
    )
    model = build_model(config)
    model.eval()

    # L=502 < LocalEncoder max_position_embeddings=512
    batch = _make_contig_batch(B=2, L=502)
    with torch.no_grad():
        out = model(batch)

    assert out.cls_logits.shape == (2, 1)
    assert out.bp_heatmaps.shape == (2, 502)
    assert out.hidden_states.shape == (2, 502, 768)
    return True


def test_no_local_encoder():
    """No-local-encoder test."""
    config = HGTConfig(
        data_mode="paired_end",
        _smoke_global_layers=2,
        use_local_encoder=False,
        pretrained_nt_path="",
        pretrained_caduceus_path="",
    )
    model = build_model(config)
    model.eval()

    batch = _make_pe_batch()
    with torch.no_grad():
        out = model(batch)

    assert out.cls_logits.shape == (4, 1)
    return True


def run():
    tests = [
        test_pe_forward_shape,
        test_contig_forward_shape,
        test_batch_size_1,
        test_variable_length_contig,
        test_no_local_encoder,
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
