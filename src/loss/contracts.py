# loss/contracts.py
"""
DeepFluHGT  —  Loss configuration
=============================
Centrally manages all loss-function hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LossConfig:
    """Loss function configuration."""

    # ── Manual weights ──
    cls_weight: float = 1.0
    bp_weight: float = 0.3
    role_weight: float = 0.2
    direction_weight: float = 0.2
    disentangle_weight: float = 0.1
    rc_consistency_weight: float = 0.1
    refiner_weight: float = 0.5

    # ── Breakpoint-region positive weighting ──
    region_pos_weight: float = 10.0

    # ── Cascade gating ──
    cascade_gate_slope: float = 10.0

    # ── Refiner Gaussian sigma ──
    breakpoint_sigma: float = 5.0

    # ── Automatic weight learning ──
    use_uncertainty_weight: bool = False
    num_tasks: int = 4   # cls, bp, role, direction

    # ── Focal Loss (replaces BCE for cls) ──
    use_focal_loss: bool = False
    focal_alpha: float = 0.25    # positive-sample weight (0.25 = hard samples matter more)
    focal_gamma: float = 2.0     # focusing parameter (larger = more attention to hard samples)

    # ── Early-exit auxiliary loss weight ──
    early_exit_weight: float = 0.25

    # ── P1b: negative-sample breakpoint regularization (RFC-P1-CONTIG-CHEAT-001) ──
    # not cascade gated; forces the bp head to learn "no peak" on negatives (fixes C-06: negative breakpoint loss zeroed by the cascade)
    neg_bp_reg_weight: float = 0.1
