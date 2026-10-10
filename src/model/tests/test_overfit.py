# model/tests/test_overfit.py
"""
Small-data overfitting test  —  validates that the model can overfit on 5 samples.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import torch
import torch.nn as nn

from model.contracts import HGTConfig
from model.api import build_model
from data_pipeline.contracts import HGTBatch


def test_overfit_pe():
    """PE mode: 5 samples, 100 epochs, validate that loss decreases."""
    config = HGTConfig(
        data_mode="paired_end",
        _smoke_global_layers=2,
        _smoke_local_layers=2,
        pretrained_nt_path="",
        pretrained_caduceus_path="",
    )
    model = build_model(config)
    model.train()

    B, L = 5, 303
    batch = HGTBatch(
        input_ids=torch.randint(0, 10, (B, L)),
        attention_mask=torch.ones(B, L, dtype=torch.long),
        bio_features=torch.randn(B, 260),
        labels=torch.tensor([1, 0, 1, 0, 1]),
        breakpoint_targets=torch.zeros(B, L),
        region_targets=torch.zeros(B, L),
        pe_case=torch.zeros(B, dtype=torch.long),
        hgt_mode=torch.zeros(B, dtype=torch.long),
        n_events=torch.tensor([1, 0, 2, 0, 1]),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    losses = []
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out.cls_logits.squeeze(-1), batch.labels.float())
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Validate that loss decreases
    assert losses[-1] < losses[0], \
        f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

    # Validate that the final loss is small enough
    assert losses[-1] < 0.5, \
        f"Final loss too high: {losses[-1]:.4f}"

    # Validate that predictions point the right way
    with torch.no_grad():
        model.eval()
        out = model(batch)
        preds = torch.sigmoid(out.cls_logits.squeeze(-1))
        pred_labels = (preds > 0.5).long()
        accuracy = (pred_labels == batch.labels).float().mean().item()
        assert accuracy >= 0.6, \
            f"Accuracy too low: {accuracy:.2f}"

    return True


def test_overfit_contig():
    """Contig mode: 5 sample overfit test."""
    config = HGTConfig(
        data_mode="contig",
        _smoke_global_layers=2,
        _smoke_local_layers=2,
        pretrained_nt_path="",
        pretrained_caduceus_path="",
    )
    model = build_model(config)
    model.train()

    B, L = 5, 202
    batch = HGTBatch(
        input_ids=torch.randint(0, 10, (B, L)),
        attention_mask=torch.ones(B, L, dtype=torch.long),
        bio_features=torch.randn(B, 260),
        labels=torch.tensor([1, 0, 1, 0, 1]),
        breakpoint_targets=torch.zeros(B, L),
        region_targets=torch.zeros(B, L),
        pe_case=torch.zeros(B, dtype=torch.long),
        hgt_mode=torch.zeros(B, dtype=torch.long),
        n_events=torch.tensor([1, 0, 1, 0, 1]),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    losses = []
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out.cls_logits.squeeze(-1), batch.labels.float())
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], \
        f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
    assert losses[-1] < 0.5, f"Final loss too high: {losses[-1]:.4f}"

    return True


def run():
    tests = [
        test_overfit_pe,
        test_overfit_contig,
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
