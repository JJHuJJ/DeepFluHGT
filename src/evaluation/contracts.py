# evaluation/contracts.py
"""
DeepFluHGT  —  Evaluation configuration and output
========================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    output_dir: str = "./eval_output"
    threshold: float = 0.5
    breakpoint_distance_threshold: int = 50  # bp
    compute_breakpoint: bool = True
    compute_direction: bool = True
    compute_genome_track: bool = False
    device: str = "auto"

    def get_device(self) -> str:
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


@dataclass
class EvalResult:
    """Evaluation results."""
    # Classification metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    auroc: float = 0.0
    auprc: float = 0.0
    specificity: float = 0.0

    # Breakpoint metrics
    breakpoint_distance_mean: Optional[float] = None
    breakpoint_distance_median: Optional[float] = None
    breakpoint_recall_50: Optional[float] = None  # % breakpoints within 50bp

    # Direction metric
    direction_accuracy: Optional[float] = None

    # Confusion matrix
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    # Raw predictions
    all_probs: Optional[np.ndarray] = None
    all_labels: Optional[np.ndarray] = None
    all_preds: Optional[np.ndarray] = None

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "Evaluation Results",
            "=" * 50,
            f"  Accuracy    : {self.accuracy:.4f}",
            f"  Precision   : {self.precision:.4f}",
            f"  Recall      : {self.recall:.4f}",
            f"  F1-score    : {self.f1:.4f}",
            f"  AUROC       : {self.auroc:.4f}",
            f"  AUPRC       : {self.auprc:.4f}",
            f"  Specificity : {self.specificity:.4f}",
        ]
        if self.breakpoint_distance_mean is not None:
            lines.append(f"  BP distance (mean)  : {self.breakpoint_distance_mean:.2f} bp")
            lines.append(f"  BP distance (median): {self.breakpoint_distance_median:.2f} bp")
            lines.append(f"  BP recall@50bp      : {self.breakpoint_recall_50:.4f}")
        if self.direction_accuracy is not None:
            lines.append(f"  Direction acc : {self.direction_accuracy:.4f}")
        lines.append(f"\n  TP={self.tp}  FP={self.fp}  FN={self.fn}  TN={self.tn}")
        lines.append("=" * 50)
        return "\n".join(lines)
