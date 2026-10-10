# evaluation/api.py
"""
DeepFluHGT  —  Evaluation public interface
======================================
Provides:
  - evaluate(): model evaluation entry point
  - EvalResult: evaluation results
  - test(): self-test
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluation.contracts import EvalConfig, EvalResult
from evaluation.impl.metrics import (
    compute_classification_metrics,
    compute_breakpoint_distance,
    compute_direction_accuracy,
)
from evaluation.impl.visualization import (
    plot_pr_curve,
    plot_roc_curve,
    plot_genome_track,
)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    config: EvalConfig = None,
    loss_fn=None,
) -> EvalResult:
    """
    Model evaluation entry point.

    Parameters
    ----------
    model : nn.Module
    dataloader : DataLoader or list[HGTBatch]
    config : EvalConfig
    loss_fn : nn.Module (optional, used to compute eval loss)

    Returns
    -------
    EvalResult
    """
    if config is None:
        config = EvalConfig()

    device = config.get_device()
    model = model.to(device)
    model.eval()

    all_probs = []
    all_labels = []
    all_heatmaps = []
    all_regions = []
    all_direction_logits = []
    all_hgt_modes = []
    total_loss = 0.0
    n_steps = 0

    for batch in dataloader:
        # Move to device
        for field_name in batch.__dataclass_fields__:
            val = getattr(batch, field_name)
            if isinstance(val, torch.Tensor):
                setattr(batch, field_name, val.to(device))

        output = model(batch)

        # loss
        if loss_fn is not None:
            total, _ = loss_fn(output, batch)
            total_loss += total.item()
            n_steps += 1

        # Collect
        probs = torch.sigmoid(output.cls_logits.squeeze(-1))
        all_probs.append(probs.cpu())
        all_labels.append(batch.labels.cpu())
        all_heatmaps.append(output.bp_heatmaps.cpu())
        all_regions.append(batch.region_targets.cpu())

        if output.direction_logits is not None:
            all_direction_logits.append(output.direction_logits.cpu())
        all_hgt_modes.append(batch.hgt_mode.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy().astype(int)
    all_heatmaps = torch.cat(all_heatmaps)
    all_regions = torch.cat(all_regions)
    all_hgt_modes = torch.cat(all_hgt_modes)

    # Classification metrics
    cls_metrics = compute_classification_metrics(
        all_probs, all_labels, config.threshold,
    )

    result = EvalResult(
        accuracy=cls_metrics["accuracy"],
        precision=cls_metrics["precision"],
        recall=cls_metrics["recall"],
        f1=cls_metrics["f1"],
        auroc=cls_metrics["auroc"],
        auprc=cls_metrics["auprc"],
        specificity=cls_metrics["specificity"],
        tp=cls_metrics["tp"],
        fp=cls_metrics["fp"],
        fn=cls_metrics["fn"],
        tn=cls_metrics["tn"],
        all_probs=all_probs,
        all_labels=all_labels,
        all_preds=(all_probs >= config.threshold).astype(int),
    )

    # Breakpoint distance
    if config.compute_breakpoint:
        bp_labels = torch.tensor(all_labels)
        bp_metrics = compute_breakpoint_distance(
            all_heatmaps, all_regions, bp_labels,
        )
        result.breakpoint_distance_mean = bp_metrics["distance_mean"]
        result.breakpoint_distance_median = bp_metrics["distance_median"]
        result.breakpoint_recall_50 = bp_metrics["recall_at_50"]

    # Direction accuracy
    if config.compute_direction and all_direction_logits:
        direction_logits = torch.cat(all_direction_logits)
        dir_acc = compute_direction_accuracy(
            direction_logits, all_hgt_modes, torch.tensor(all_labels),
        )
        result.direction_accuracy = dir_acc

    # Visualization
    os.makedirs(config.output_dir, exist_ok=True)

    n_pos = int(all_labels.sum())
    n_neg = len(all_labels) - n_pos
    if n_pos > 0 and n_neg > 0:
        pr_path = os.path.join(config.output_dir, "pr_curve.png")
        plot_pr_curve(all_labels, all_probs, save_path=pr_path)

        roc_path = os.path.join(config.output_dir, "roc_curve.png")
        plot_roc_curve(all_labels, all_probs, save_path=roc_path)

    # Genome track (take the first 3 positive samples)
    if config.compute_genome_track:
        pos_indices = np.where(all_labels == 1)[0][:3]
        for idx, i in enumerate(pos_indices):
            gt_path = os.path.join(config.output_dir, f"genome_track_{idx}.png")
            plot_genome_track(
                all_heatmaps[i].numpy(),
                all_regions[i].numpy(),
                int(all_labels[i]),
                seq_id=f"sample_{i}",
                save_path=gt_path,
            )

    return result


def test() -> bool:
    """Evaluation self-test."""
    print("=" * 60)
    print("DeepFluHGT  —  Evaluation Self-Test")
    print("=" * 60)

    results = []
    PASS = "\033[92mPASS\033[0m"
    FAIL = "\033[91mFAIL\033[0m"

    def log(name, passed, detail=""):
        sym = PASS if passed else FAIL
        msg = f"  {sym} {name}"
        if detail:
            msg += f"  [{detail}]"
        print(msg)
        results.append(passed)

    from model.contracts import HGTConfig
    from model.api import build_model
    from loss.contracts import LossConfig
    from loss.api import build_loss
    from data_pipeline.contracts import HGTBatch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 1. metriccompute ──
    print("\n[1] Metrics")
    probs = np.array([0.9, 0.8, 0.3, 0.1, 0.7, 0.6, 0.2, 0.4, 0.95, 0.05])
    labels = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 0])
    m = compute_classification_metrics(probs, labels)
    log("accuracy > 0.8", m["accuracy"] > 0.8, f"acc={m['accuracy']:.4f}")
    log("precision > 0", m["precision"] > 0, f"prec={m['precision']:.4f}")
    log("recall > 0", m["recall"] > 0, f"rec={m['recall']:.4f}")
    log("f1 > 0", m["f1"] > 0, f"f1={m['f1']:.4f}")
    log("auroc > 0.5", m["auroc"] > 0.5, f"auroc={m['auroc']:.4f}")
    log("auprc > 0.5", m["auprc"] > 0.5, f"auprc={m['auprc']:.4f}")
    log("tp + fp + fn + tn == 10",
        m["tp"] + m["fp"] + m["fn"] + m["tn"] == 10)

    # ── 2. End-to-end evaluation ──
    print("\n[2] End-to-End Evaluate")
    model_config = HGTConfig(
        data_mode="paired_end",
        _smoke_global_layers=2,
        _smoke_local_layers=2,
        pretrained_nt_path="",
        pretrained_caduceus_path="",
    )
    model = build_model(model_config).to(DEVICE)
    loss_fn = build_loss(LossConfig()).to(DEVICE)

    # Build a batch
    B = 8
    batch = HGTBatch(
        input_ids=torch.randint(0, 10, (B, 303), device=DEVICE),
        attention_mask=torch.ones(B, 303, dtype=torch.long, device=DEVICE),
        bio_features=torch.randn(B, 260, device=DEVICE),
        labels=torch.tensor([1, 0, 1, 1, 0, 1, 0, 0], device=DEVICE),
        breakpoint_targets=torch.zeros(B, 303, device=DEVICE),
        region_targets=torch.zeros(B, 303, device=DEVICE),
        pe_case=torch.zeros(B, dtype=torch.long, device=DEVICE),
        hgt_mode=torch.tensor([0, 3, 1, 0, 3, 1, 3, 3], device=DEVICE),
        n_events=torch.tensor([1, 0, 2, 1, 0, 1, 0, 0], device=DEVICE),
    )

    eval_config = EvalConfig(device=DEVICE, output_dir="./eval_test_output")
    result = evaluate(model, [batch], eval_config, loss_fn=loss_fn)

    log("EvalResult created", result is not None)
    log("accuracy in [0,1]", 0 <= result.accuracy <= 1)
    log("auroc in [0,1]", 0 <= result.auroc <= 1)
    log("f1 in [0,1]", 0 <= result.f1 <= 1)
    log("direction_accuracy present", result.direction_accuracy is not None)
    log("all_probs shape", result.all_probs is not None and len(result.all_probs) == B)
    log("summary() works", len(result.summary()) > 0)

    print(result.summary())

    # ── 3. Visualization ──
    print("\n[3] Visualization")
    pr_path = plot_pr_curve(labels, probs, save_path="/tmp/test_pr.png")
    log("PR curve generated", os.path.exists(pr_path))

    roc_path = plot_roc_curve(labels, probs, save_path="/tmp/test_roc.png")
    log("ROC curve generated", os.path.exists(roc_path))

    # ── Summary ──
    total_n = len(results)
    passed = sum(results)
    failed = total_n - passed

    print("\n" + "=" * 60)
    print(f"TOTAL: {total_n}  |  PASSED: {passed}  |  FAILED: {failed}")
    print("=" * 60)

    return failed == 0
