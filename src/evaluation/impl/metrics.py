# evaluation/impl/metrics.py
"""
DeepFluHGT  —  evaluationmetriccompute
================================
  - compute_classification_metrics: Accuracy, Precision, Recall, F1, AUROC, AUPRC, Specificity
  - compute_breakpoint_distance: breakpoint localization distance
  - compute_direction_accuracy: direction prediction accuracy
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def compute_classification_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict:
    """
    Compute classification metrics.

    Parameters
    ----------
    probs : [N] Predicted probabilities (sigmoid output)
    labels : [N] Ground-truth labels (0/1)
    threshold : float

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, auroc, auprc, specificity, tp, fp, fn, tn
    """
    preds = (probs >= threshold).astype(int)
    labels = labels.astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    accuracy = (tp + tn) / max(1, tp + fp + fn + tn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    specificity = tn / max(1, tn + fp)

    # AUROC / AUPRC
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    auroc = 0.0
    auprc = 0.0

    if n_pos > 0 and n_neg > 0:
        from sklearn.metrics import roc_auc_score, average_precision_score
        auroc = float(roc_auc_score(labels, probs))
        auprc = float(average_precision_score(labels, probs))

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "specificity": float(specificity),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def compute_breakpoint_distance(
    pred_heatmaps: torch.Tensor,
    region_targets: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    threshold: float = 0.5,
) -> Dict:
    """
    Compute breakpoint localization distance.

    Strategy: take the predicted heatmap peaks vs the ground-truth region
    boundaries and compute the nearest distance.

    Parameters
    ----------
    pred_heatmaps : [B, L] Predicted heatmap (sigmoid)
    region_targets : [B, L] Ground-truth region labels (0/1)
    labels : [B] Sample labels
    attention_mask : [B, L] optional
    threshold : float

    Returns
    -------
    dict: distance_mean, distance_median, recall_at_50
    """
    pos_mask = (labels == 1)
    if pos_mask.sum() == 0:
        return {
            "distance_mean": None,
            "distance_median": None,
            "recall_at_50": None,
        }

    distances = []
    for i in range(len(labels)):
        if not pos_mask[i]:
            continue

        region = region_targets[i]
        heatmap = pred_heatmaps[i]

        # Ground-truth boundary points: positions where the region goes 0→1 or 1→0
        region_diff = torch.diff(region.float())
        true_starts = (region_diff == 1.0).nonzero(as_tuple=True)[0] + 1
        true_ends = (region_diff == -1.0).nonzero(as_tuple=True)[0] + 1
        true_points = torch.cat([true_starts, true_ends]).float()

        if len(true_points) == 0:
            continue

        # Predicted peaks: local maxima of the heatmap above the threshold
        above = (heatmap > threshold)
        pred_peaks = above.nonzero(as_tuple=True)[0].float()

        if len(pred_peaks) == 0:
            # No prediction → maximum distance
            distances.append(float(region.shape[0]))
            continue

        # For each ground-truth point, find the nearest predicted point
        for tp in true_points:
            min_dist = (pred_peaks - tp).abs().min().item()
            distances.append(min_dist)

    if not distances:
        return {
            "distance_mean": None,
            "distance_median": None,
            "recall_at_50": None,
        }

    distances = np.array(distances)
    # m-02 note: with single-nucleotide tokenization, 1 token = 1 bp,
    # so distances are already in bp and recall_at_50 = 50 bp tolerance is correct.
    return {
        "distance_mean": float(distances.mean()),
        "distance_median": float(np.median(distances)),
        "recall_at_50": float((distances <= 50).mean()),
    }


def compute_direction_accuracy(
    direction_logits: torch.Tensor,
    hgt_mode: torch.Tensor,
    labels: torch.Tensor,
) -> Optional[float]:
    """
    Compute direction prediction accuracy.

    hgt_mode == 0 (insertion) → direction = 1 (host→virus)
    hgt_mode == 1,2 → direction = 0

    Parameters
    ----------
    direction_logits : [B, 1] or [B]
    hgt_mode : [B]
    labels : [B]

    Returns
    -------
    accuracy (float) or None
    """
    pos_mask = (labels == 1)
    if pos_mask.sum() == 0 or direction_logits is None:
        return None

    logits = direction_logits[pos_mask].squeeze(-1)
    mode = hgt_mode[pos_mask]

    pred_dir = (torch.sigmoid(logits) > 0.5).long()
    true_dir = (mode == 0).long()

    if len(true_dir) == 0:
        return None

    return float((pred_dir == true_dir).float().mean().item())


# ═══════════════════════════════════════════════════════════════════════════════ #
#  Stratified evaluation metrics & temperature-scaling calibration
# ═══════════════════════════════════════════════════════════════════════════════ #

# H-01 fix (CODE_AUDIT_V7): align the hard/easy stratification with the parser's
# full negative-type set. Before the fix the parser only passed through 4 classes,
# so HOMOLOGY/CONSERVED/LOW_COMPLEXITY could never reach this code under their real
# identities, and GC_MATCHED_CHIMERIC/AMALGAMATED_HGT/PARTIAL_VIRUS_HIT (together
# ~34% of training negative contigs) collapsed into "NEG", polluting the easy pool.
# Attribution (approved 2026-08-20):
#   hard = simulated hard negative classes: classic hard + homology/conserved/low-complexity
#          + GC-matched chimera / merged HGT / partial virus hit (7 classes)
#   easy = pure random NEG (excluding any constructed hard type)
# NEG_VIRUS / NEG_PCR do not participate in the hard/easy stratified AUC — they still
# appear in full in per_type_counts and neg_specificity_by_type (per-type specificity
# is unaffected by this fix).
# This fix is purely an evaluation-caliber change: no weights change; historical
# numbers are recomputed offline by scripts/recompute_stratified.py (run after M0 completes).
_HARD_NEG_TYPES = {
    "NEG_HARD", "NEG_HOMOLOGY", "NEG_CONSERVED", "NEG_LOW_COMPLEXITY",
    "NEG_GC_MATCHED_CHIMERIC", "NEG_AMALGAMATED_HGT", "NEG_PARTIAL_VIRUS_HIT",
}
_EASY_NEG_TYPES = {"NEG"}


def compute_stratified_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    sample_types: list,
    threshold: float = 0.5,
) -> Dict:
    """
    Compute metrics stratified by sample type.

    Parameters
    ----------
    probs : np.ndarray [N]
        Predicted probabilities (sigmoid output).
    labels : np.ndarray [N]
        Ground-truth labels (0/1).
    sample_types : list of str
        Sample type for each instance.
    threshold : float
        Classification threshold.

    Returns
    -------
    dict with:
    - hard_neg_auc: AUC on hard negatives + all positives
    - easy_neg_auc: AUC on easy negatives + all positives
    - pos_recall: recall on positive samples
    - neg_specificity_by_type: specificity per negative type
    - per_type_counts: {type: count}
    """
    from sklearn.metrics import roc_auc_score

    labels = labels.astype(int)
    sample_types = list(sample_types)
    n = len(probs)
    preds = (probs >= threshold).astype(int)

    pos_mask = labels == 1
    neg_mask = labels == 0

    result: Dict = {}

    # ── Per-type counts ──
    per_type_counts: Dict[str, int] = {}
    for st in sample_types:
        per_type_counts[st] = per_type_counts.get(st, 0) + 1
    result["per_type_counts"] = per_type_counts

    # ── Positive recall ──
    n_pos = int(pos_mask.sum())
    if n_pos > 0:
        result["pos_recall"] = float(preds[pos_mask].sum() / n_pos)
    else:
        result["pos_recall"] = 0.0

    # ── Hard negative AUC (all positives + hard negatives) ──
    hard_neg_idx = np.array([
        i for i in range(n)
        if neg_mask[i] and sample_types[i] in _HARD_NEG_TYPES
    ])
    pos_idx = np.where(pos_mask)[0]
    hard_auc_idx = np.concatenate([pos_idx, hard_neg_idx])

    if len(pos_idx) > 0 and len(hard_neg_idx) > 0:
        result["hard_neg_auc"] = float(roc_auc_score(
            labels[hard_auc_idx], probs[hard_auc_idx]))
    else:
        result["hard_neg_auc"] = 0.0

    # ── Easy negative AUC (all positives + easy negatives) ──
    easy_neg_idx = np.array([
        i for i in range(n)
        if neg_mask[i] and sample_types[i] in _EASY_NEG_TYPES
    ])
    easy_auc_idx = np.concatenate([pos_idx, easy_neg_idx])

    if len(pos_idx) > 0 and len(easy_neg_idx) > 0:
        result["easy_neg_auc"] = float(roc_auc_score(
            labels[easy_auc_idx], probs[easy_auc_idx]))
    else:
        result["easy_neg_auc"] = 0.0

    # ── Per-negative-type FP rate (specificity complement) ──
    neg_specificity_by_type: Dict[str, float] = {}
    for st in sorted(set(sample_types)):
        type_mask = np.array([s == st for s in sample_types]) & neg_mask
        n_type = int(type_mask.sum())
        if n_type > 0:
            # FP rate = fraction of negatives of this type predicted as positive
            fp_count = int(preds[type_mask].sum())
            neg_specificity_by_type[st] = float(1.0 - fp_count / n_type)
    result["neg_specificity_by_type"] = neg_specificity_by_type

    return result


def compute_calibration_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Dict:
    """
    Compute ECE (Expected Calibration Error) and apply temperature scaling.

    Parameters
    ----------
    probs : np.ndarray [N]
        Predicted probabilities (sigmoid output).
    labels : np.ndarray [N]
        Ground-truth labels (0/1).
    n_bins : int
        Number of bins for ECE computation.

    Returns
    -------
    dict with:
    - ece: Expected Calibration Error (before calibration)
    - optimal_temperature: best T from grid search
    - ece_calibrated: ECE after temperature scaling with optimal T
    """
    labels = labels.astype(int)

    def _compute_ece(p: np.ndarray, y: np.ndarray, n_bins: int) -> float:
        """Compute Expected Calibration Error."""
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n_total = len(y)

        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            # Last bin is inclusive on both ends; others are [lo, hi)
            if i < n_bins - 1:
                mask = (p >= lo) & (p < hi)
            else:
                mask = (p >= lo) & (p <= hi)

            n_bin = int(mask.sum())
            if n_bin == 0:
                continue

            avg_confidence = float(p[mask].mean())
            avg_accuracy = float(y[mask].mean())
            ece += n_bin / n_total * abs(avg_confidence - avg_accuracy)

        return ece

    def _scale_temperature(p: np.ndarray, T: float) -> np.ndarray:
        """Apply temperature scaling: sigmoid(logit(p) / T)."""
        p_clipped = np.clip(p, 1e-7, 1.0 - 1e-7)
        logits = np.log(p_clipped / (1.0 - p_clipped))
        scaled_logits = logits / T
        # Numerically stable sigmoid
        scaled_probs = np.where(
            scaled_logits >= 0,
            1.0 / (1.0 + np.exp(-scaled_logits)),
            np.exp(scaled_logits) / (1.0 + np.exp(scaled_logits)),
        )
        return np.clip(scaled_probs, 1e-7, 1.0 - 1e-7)

    # ── ECE before calibration ──
    ece = _compute_ece(probs, labels, n_bins)

    # ── Grid search for optimal temperature ──
    temperature_grid = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    best_T = 1.0
    best_ece = ece

    for T in temperature_grid:
        scaled = _scale_temperature(probs, T)
        ece_T = _compute_ece(scaled, labels, n_bins)
        if ece_T < best_ece:
            best_ece = ece_T
            best_T = T

    return {
        "ece": float(ece),
        "optimal_temperature": float(best_T),
        "ece_calibrated": float(best_ece),
    }
