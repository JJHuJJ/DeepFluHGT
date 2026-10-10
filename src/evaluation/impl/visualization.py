# evaluation/impl/visualization.py
"""
DeepFluHGT  —  Visualization
==========================
  - plot_pr_curve: Precision-Recall curve
  - plot_roc_curve: ROC curve
  - plot_genome_track: genome breakpoint heatmap
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np


def plot_pr_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Precision-Recall Curve",
) -> str:
    """Plot a PR curve (ASCII or matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve

        precision, recall, _ = precision_recall_curve(labels, probs)

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.plot(recall, precision, linewidth=2)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        if save_path is None:
            save_path = "pr_curve.png"
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path
    except ImportError:
        return _ascii_pr(labels, probs)


def _ascii_pr(labels: np.ndarray, probs: np.ndarray) -> str:
    """ASCII PR curve (no matplotlib)."""
    # Bucket by threshold
    n_bins = 10
    thresholds = np.linspace(0.1, 0.9, n_bins)
    lines = ["PR Curve (ASCII)"]
    lines.append("Precision │")
    for t in reversed(thresholds):
        preds = (probs >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        bar_len = int(prec * 30)
        lines.append(f"  {prec:.2f}    │{'█' * bar_len}")
    lines.append(f"          └{'─' * 30} Recall")
    return "\n".join(lines)


def plot_roc_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "ROC Curve",
) -> str:
    """Plot a ROC curve."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc

        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.plot(fpr, tpr, linewidth=2, label=f"AUROC = {roc_auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path is None:
            save_path = "roc_curve.png"
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path
    except ImportError:
        return "ROC: matplotlib not available"


def plot_genome_track(
    heatmap: np.ndarray,
    region_target: np.ndarray,
    label: int,
    seq_id: str = "",
    save_path: Optional[str] = None,
) -> str:
    """
    Plot a genome breakpoint heatmap (single sample).

    Parameters
    ----------
    heatmap : [L] Predicted heatmap probabilities
    region_target : [L] Ground-truth region labels
    label : int (0/1)
    seq_id : str
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        L = len(heatmap)
        x = np.arange(L)

        fig, axes = plt.subplots(2, 1, figsize=(14, 4), sharex=True)

        axes[0].fill_between(x, heatmap, alpha=0.7, color="steelblue")
        axes[0].set_ylabel("Predicted")
        axes[0].set_ylim(0, 1)
        axes[0].set_title(f"{seq_id} (label={'POS' if label else 'NEG'})")

        axes[1].fill_between(x, region_target, alpha=0.7, color="coral")
        axes[1].set_ylabel("Ground Truth")
        axes[1].set_ylim(0, 1)
        axes[1].set_xlabel("Position (bp)")

        plt.tight_layout()

        if save_path is None:
            save_path = "genome_track.png"
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path
    except ImportError:
        return "Genome track: matplotlib not available"
