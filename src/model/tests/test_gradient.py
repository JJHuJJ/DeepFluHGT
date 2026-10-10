# model/tests/test_gradient.py
"""
Gradient propagation test  —  validates that gradients flow correctly through
all submodules, with no NaN/Inf.

Note: some parameters only participate in computation under specific conditions:
  - cross_attn.pair_attn_b: only when PE-mode R2 is present
  - bp_refiner.conv_stack: only when the peak-extraction window passes through Stage 2
  - cls_head.mlp: only when use_evidential=True
The test validates gradient integrity of the core paths and skips conditional parameters.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import torch

from model.contracts import HGTConfig
from model.api import build_model
from data_pipeline.contracts import HGTBatch

# Conditional parameters: these do not participate in computation under the default config
CONDITIONAL_PREFIXES = [
    "cross_attn.pair_attn_a",      # pair interaction only when PE R2 is present
    "cross_attn.pair_attn_b",      # same as above
    "cross_attn.pair_norm_a",      # same as above
    "cross_attn.pair_norm_b",      # same as above
    "cross_attn.local_b_to_donor",     # only when local_hidden_b is not None (PE)
    "cross_attn.local_b_to_recipient", # same as above
    "cross_attn.norm_b",               # same as above
    "bp_refiner.conv_stack",        # only the Stage 2 refinement path
    "bp_refiner.fine_mlp",          # same as above
    "cls_head.mlp",                 # only when use_evidential=True
    "early_exit_mgr",               # only when threshold < 1.0 and included in the loss
]


def _is_conditional(name: str) -> bool:
    """Determine whether a parameter is conditional."""
    return any(name.startswith(p) for p in CONDITIONAL_PREFIXES)


def _make_pe_batch(B=4, L=303):
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


def test_gradient_flow():
    """Validate that all core parameters have gradients (excluding conditional parameters)."""
    config = HGTConfig(
        data_mode="paired_end",
        _smoke_global_layers=2,
        _smoke_local_layers=2,
        pretrained_nt_path="",
        pretrained_caduceus_path="",
    )
    model = build_model(config)
    model.train()

    batch = _make_pe_batch()
    out = model(batch)

    loss = (
        out.cls_logits.sum()
        + out.bp_heatmaps.sum()
        + out.role_logits.sum()
        + out.direction_logits.sum()
    )
    loss.backward()

    # Check core parameter gradients
    no_grad_params = []
    for name, p in model.named_parameters():
        if p.requires_grad and not _is_conditional(name):
            if p.grad is None:
                no_grad_params.append(name)
            elif p.grad.abs().sum().item() == 0:
                no_grad_params.append(f"{name} (zero)")

    assert len(no_grad_params) == 0, \
        f"Core parameters without gradients: {no_grad_params[:10]}"
    return True


def test_no_nan_gradients():
    """Validate that gradients contain no NaN/Inf."""
    config = HGTConfig(
        data_mode="paired_end",
        _smoke_global_layers=2,
        _smoke_local_layers=2,
        pretrained_nt_path="",
        pretrained_caduceus_path="",
    )
    model = build_model(config)
    model.train()

    batch = _make_pe_batch()
    out = model(batch)

    loss = out.cls_logits.sum() + out.bp_heatmaps.sum()
    loss.backward()

    nan_params = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            if torch.isnan(p.grad).any():
                nan_params.append(name)
            if torch.isinf(p.grad).any():
                nan_params.append(f"{name} (inf)")

    assert len(nan_params) == 0, \
        f"Parameters with NaN/Inf gradients: {nan_params[:10]}"
    return True


def test_contig_gradient():
    """Contig-mode gradient test (excluding conditional parameters)."""
    config = HGTConfig(
        data_mode="contig",
        _smoke_global_layers=2,
        _smoke_local_layers=2,
        pretrained_nt_path="",
        pretrained_caduceus_path="",
    )
    model = build_model(config)
    model.train()

    batch = _make_contig_batch()
    out = model(batch)

    loss = (
        out.cls_logits.sum()
        + out.role_logits.sum()
        + out.direction_logits.sum()
        + out.bp_heatmaps.sum()
    )
    loss.backward()

    no_grad = []
    for name, p in model.named_parameters():
        if p.requires_grad and not _is_conditional(name):
            if p.grad is None:
                no_grad.append(name)

    assert len(no_grad) == 0, \
        f"Core parameters without gradients: {no_grad[:10]}"
    return True


def test_component_gradients():
    """Validate gradient flow through key submodules (backbone, splitter, fpn, graph)."""
    config = HGTConfig(
        data_mode="paired_end",
        _smoke_global_layers=2,
        _smoke_local_layers=2,
        pretrained_nt_path="",
        pretrained_caduceus_path="",
    )
    model = build_model(config)
    model.train()

    batch = _make_pe_batch()
    out = model(batch)

    loss = (
        out.cls_logits.sum()
        + out.role_logits.sum()
        + out.direction_logits.sum()
        + out.bp_heatmaps.sum()
    )
    loss.backward()

    # Check key components
    components = {
        "backbone": False,
        "splitter": False,
        "cross_attn": False,
        "fpn": False,
        "graph": False,
        "cls_head": False,
        "role_head": False,
        "direction_head": False,
        "bio_encoder": False,
    }

    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None and p.grad.abs().sum().item() > 0:
            for comp in components:
                if name.startswith(comp):
                    components[comp] = True

    missing = [c for c, v in components.items() if not v]
    assert len(missing) == 0, \
        f"Components without gradient flow: {missing}"
    return True


def run():
    tests = [
        test_gradient_flow,
        test_no_nan_gradients,
        test_contig_gradient,
        test_component_gradients,
    ]
    results = []
    for t in tests:
        try:
            ok = t()
            print(f"  [{'PASS' if ok else 'FAIL'}] {t.__name__}")
            results.append(ok)
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            results.append(False)
    return all(results)


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
