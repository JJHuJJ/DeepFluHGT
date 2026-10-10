# loss/api.py
"""
DeepFluHGT  —  Loss public interface
=================================
Provides:
  - CombinedLoss: multi-task loss aggregator (core)
  - build_loss(): build from a LossConfig
  - test(): self-test

Loss composition
--------
  1. cls_loss         —  BCEWithLogitsLoss (classification)
  2. bp_loss          —  WeightedBreakpointLoss + CascadeGating (breakpoint region)
  3. role_loss        —  BCEWithLogitsLoss + CascadeGating (role)
  4. direction_loss   —  BCEWithLogitsLoss + positive-only (direction)
  5. disentangle_loss —  SquaredCosine (cos² disentanglement)
  6. refiner_loss     —  GaussianRefinerLoss (refined distribution)
  7. rc_loss          —  RCConsistencyLoss (forward/reverse-strand consistency, PE only)
  8. early_exit_loss  —  auxiliary classification loss (optional)
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from loss.contracts import LossConfig
from loss.impl.cascade_gating import CascadeGatingLoss
from loss.impl.rc_consistency import RCConsistencyLoss
from loss.impl.uncertainty_weight import TaskUncertaintyWeight


# ═══════════════════════════════════════════════════════════════════════════════ #
#  FocalLoss  —  classification loss focusing on hard samples
# ═══════════════════════════════════════════════════════════════════════════════ #

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al. 2017) as a replacement for BCEWithLogitsLoss.

    FL(p) = -alpha * (1 - p)^gamma * log(p)

    Parameters
    ----------
    alpha : float
        positive samplesweights (default 0.25)
    gamma : float
        Focusing parameter (default 2.0). Larger gamma shrinks the loss on already-
        well-classified samples, concentrating gradient on hard samples.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : [B, 1] or [B] raw logits before sigmoid
        targets : [B] or [B, 1] float labels (mixup compatible)

        Returns
        -------
        scalar loss
        """
        # FOCAL-AMP-001 fix: at fp32 belowcompute focal factor
        # In fp16, (1 - p_t) underflows to 0 for high-confidence samples (min fp16 ~6e-8),
        # making the focal factor 0 instead of a tiny positive number, over-suppressing easy samples.
        device_type = logits.device.type if hasattr(logits.device, 'type') else "cuda"
        with torch.amp.autocast(device_type, enabled=False):
            logits_flat = logits.float().squeeze(-1) if logits.dim() > 1 else logits.float()
            targets_flat = targets.float()

            # Numerically stable sigmoid + BCE
            bce = F.binary_cross_entropy_with_logits(
                logits_flat, targets_flat, reduction='none')

            # p_t = p if y=1, else 1-p
            probs = torch.sigmoid(logits_flat)
            p_t = torch.where(targets_flat >= 0.5, probs, 1.0 - probs)

            # focal factor: (1 - p_t)^gamma
            focal_factor = (1.0 - p_t) ** self.gamma

            # alpha weighting
            alpha_t = torch.where(targets_flat >= 0.5, self.alpha, 1.0 - self.alpha)

            loss = alpha_t * focal_factor * bce
            return loss.mean()


# ═══════════════════════════════════════════════════════════════════════════════ #
#  WeightedBreakpointLoss  —  weighted breakpoint-region loss
# ═══════════════════════════════════════════════════════════════════════════════ #

class WeightedBreakpointLoss(nn.Module):
    """
    Per-token BCE loss; positive samples (virus tokens) get higher weights.

    Parameters
    ----------
    region_pos_weight : float
    """

    def __init__(self, region_pos_weight: float = 10.0):
        super().__init__()
        self.region_pos_weight = region_pos_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cascade_confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred               : [B, L] sigmoid probability
        target             : [B, L] 0/1 region labels
        attention_mask     : [B, L] optional
        cascade_confidence : [B] optional, cascade gating signal (detached)

        Returns
        -------
        scalar loss
        """
        # NaN defense: on NaN/Inf predictions return a NaN loss so step() skips the batch;
        # do not train on nan_to_num-falsified data, which would introduce systematic bias
        with torch.amp.autocast(device_type=pred.device.type, enabled=False):
            pred_fp32 = pred.float()
            if not torch.isfinite(pred_fp32).all():
                import warnings
                warnings.warn(
                    "[WeightedBreakpointLoss] NaN/Inf prediction detected; skipping this batch. "
                    "Check the backbone projection layer for overflow.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return torch.tensor(0.0, device=pred.device, requires_grad=True)

        weights = 1.0 + target * (self.region_pos_weight - 1.0)
        # AMP safety: binary_cross_entropy is incompatible with autocast; compute in fp32
        with torch.amp.autocast(device_type=pred.device.type, enabled=False):
            pred_fp32 = pred_fp32.clamp(1e-6, 1 - 1e-6)
            target_fp32 = target.float().clamp(0.0, 1.0)
            bce = F.binary_cross_entropy(pred_fp32, target_fp32, reduction="none")
        weighted = weights * bce

        if attention_mask is not None:
            weighted = weighted * attention_mask.float()
            n_valid = attention_mask.float().sum(dim=-1).clamp(min=1.0)
            per_sample = weighted.sum(dim=-1) / n_valid
        else:
            per_sample = weighted.mean(dim=-1)

        if cascade_confidence is not None:
            per_sample = per_sample * cascade_confidence

        return per_sample.mean()


# ═══════════════════════════════════════════════════════════════════════════════ #
#  GaussianRefinerLoss  —  Gaussian-target refinement loss
# ═══════════════════════════════════════════════════════════════════════════════ #

class GaussianRefinerLoss(nn.Module):
    """
    KL-divergence loss on the refiner output: KL(gaussian_target ∥ softmax(refined)).

    Parameters
    ----------
    sigma : float
    """

    def __init__(self, sigma: float = 5.0):
        super().__init__()
        self.sigma = sigma

    def forward(
        self,
        refined_output: torch.Tensor,
        breakpoint_offset: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        refined_output    : [B, W] softmax refined distribution
        breakpoint_offset : [B] ground-truth breakpoint offset within the window
        valid_mask        : [B] optional, 1=positive samples, 0=negative samples

        Returns
        -------
        scalar loss
        """
        B, W = refined_output.shape
        device = refined_output.device

        positions = torch.arange(W, device=device, dtype=refined_output.dtype)
        center = breakpoint_offset.float().unsqueeze(1)
        gauss = torch.exp(-0.5 * ((positions.unsqueeze(0) - center) / self.sigma) ** 2)
        gauss = gauss / gauss.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        log_pred = refined_output.clamp(min=1e-8).log()  # clamp to avoid gradient drift
        kl = F.kl_div(log_pred, gauss, reduction="none").sum(dim=-1)

        if valid_mask is not None:
            n_valid = valid_mask.float().sum().clamp(min=1.0)
            return (kl * valid_mask.float()).sum() / n_valid
        return kl.mean()


# ═══════════════════════════════════════════════════════════════════════════════ #
#  SquaredCosineDisentangleLoss  —  cos² disentanglement loss
# ═══════════════════════════════════════════════════════════════════════════════ #

class SquaredCosineDisentangleLoss(nn.Module):
    """cos² orthogonality constraint loss (defined inline to avoid a circular import)."""

    def forward(self, d: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        d = F.normalize(d, dim=-1)
        r = F.normalize(r, dim=-1)
        cos_sim = torch.sum(d * r, dim=-1)
        return (cos_sim ** 2).mean()


# ═══════════════════════════════════════════════════════════════════════════════ #
#  CombinedLoss  —  multi-task loss aggregator
# ═══════════════════════════════════════════════════════════════════════════════ #

class CombinedLoss(nn.Module):
    """
    DeepFluHGT multi-task loss aggregator.

    Reads HGTOutput + HGTBatch → computes all sub-losses → weighted aggregation → returns (total, loss_dict).

    Parameters
    ----------
    config : LossConfig
    """

    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config

        # ── Sub-loss modules ──
        self.bp_loss_fn = WeightedBreakpointLoss(
            region_pos_weight=config.region_pos_weight,
        )
        self.refiner_loss_fn = GaussianRefinerLoss(sigma=config.breakpoint_sigma)
        self.rc_loss_fn = RCConsistencyLoss()
        self.disentangle_fn = SquaredCosineDisentangleLoss()
        self.cascade_gate = CascadeGatingLoss(slope=config.cascade_gate_slope)

        # ── Focal Loss (optional, replaces BCE for cls) ──
        self.focal_loss_fn = None
        if config.use_focal_loss:
            self.focal_loss_fn = FocalLoss(
                alpha=config.focal_alpha,
                gamma=config.focal_gamma,
            )

        # ── Automatic weight learning (optional) ──
        self.uncertainty_weight = None
        if config.use_uncertainty_weight:
            self.uncertainty_weight = TaskUncertaintyWeight(
                num_tasks=config.num_tasks,
                task_names=["cls", "bp", "role", "direction"],
            )

        # ── Manual weights ──
        self.cls_weight = config.cls_weight
        self.bp_weight = config.bp_weight
        self.role_weight = config.role_weight
        self.direction_weight = config.direction_weight
        self.disentangle_weight = config.disentangle_weight
        self.rc_weight = config.rc_consistency_weight
        self.refiner_weight = config.refiner_weight
        self.early_exit_weight = config.early_exit_weight
        self.neg_bp_reg_weight = config.neg_bp_reg_weight  # P1b

    def forward(
        self,
        output,
        batch,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute all sub-losses and aggregate them.

        Parameters
        ----------
        output : HGTOutput
        batch  : HGTBatch

        Returns
        -------
        (total_loss, loss_dict)
            total_loss : scalar tensor (keeps the computation graph)
            loss_dict  : {name: float} detached values for logging
        """
        device = output.cls_logits.device
        loss_dict = {}
        live_losses = {}

        # ── 1. Cascade gating confidence ──
        confidence = self.cascade_gate(output.cls_logits)  # [B], detached

        # ── 2. Classification loss (Focal Loss or BCE) ──
        if self.focal_loss_fn is not None:
            cls_loss = self.focal_loss_fn(output.cls_logits, batch.labels.float())
        else:
            cls_loss = F.binary_cross_entropy_with_logits(
                output.cls_logits.squeeze(-1),
                batch.labels.float(),
            )
        loss_dict["cls"] = cls_loss.item()
        live_losses["cls"] = cls_loss

        # ── 3. Breakpoint-region loss (weighted + cascade gated) ──
        bp_loss = self.bp_loss_fn(
            output.bp_heatmaps,
            batch.region_targets,
            attention_mask=batch.attention_mask,
            cascade_confidence=confidence,
        )
        loss_dict["bp"] = bp_loss.item()
        live_losses["bp"] = bp_loss

        # ── 3.5 Negative-sample breakpoint regularization (P1b: not cascade gated, forcing the bp head to learn "no peak" on negatives) ──
        neg_bp_reg = torch.tensor(0.0, device=device)
        if self.neg_bp_reg_weight > 0 and output.bp_heatmaps is not None:
            neg_mask = (batch.labels <= 0.5)
            if neg_mask.any():
                neg_bp = output.bp_heatmaps[neg_mask]  # [n_neg, L]
                neg_bp_reg = neg_bp.max(dim=-1).values.mean()  # push down the bp peak on negatives
        loss_dict["neg_bp_reg"] = neg_bp_reg.item()

        # ── 4. role loss (cascade gated) ──
        # Note: use batch.labels.float() instead of (hgt_mode < 3).float()
        # Reason: after mixup labels are float mixtures (e.g. 0.7) while hgt_mode stays integer,
        #       so an auxiliary target would be inconsistent with the mixed primary classification label, causing gradient conflict.
        # Equivalence: without mixup label∈{0,1}; positives hgt_mode∈{0,1,2}→role=1=label,
        #         negatives hgt_mode=3→role=0=label; the two are fully equivalent.
        role_targets = batch.labels.float()
        # BUG-007 fix: per-sample cascade gating: mean(L*C) instead of mean(L)*mean(C)
        per_sample_role = F.binary_cross_entropy_with_logits(
            output.role_logits.squeeze(-1), role_targets, reduction='none',
        )
        role_loss = (per_sample_role * confidence).mean()
        loss_dict["role"] = role_loss.item()
        live_losses["role"] = role_loss

        # ── 5. Direction loss (positives only, cascade gated, NaN guarded) ──
        # BUG-010 fix: support post-mixup soft one-hot hgt_mode [B, 4] and
        #              un-mixed hard integer hgt_mode [B].
        pos_mask = (batch.labels > 0.5)  # >0.5 supports mixup float labels and hard labels
        if pos_mask.sum() > 0 and output.direction_logits is not None:
            dir_logits = output.direction_logits[pos_mask].squeeze(-1)
            # NaN defense: skip on NaN/Inf logits
            with torch.amp.autocast(device_type=dir_logits.device.type, enabled=False):
                if not torch.isfinite(dir_logits.float()).all():
                    dir_loss = torch.tensor(0.0, device=device)  # 0 not NaN, to avoid hurting the other losses
                else:
                    # BUG-010 fix: direction targets support soft one-hot and hard integer
                    _hgt = batch.hgt_mode[pos_mask]
                    if _hgt.dim() == 2 and _hgt.size(-1) >= 2:
                        # Post-mixup: soft one-hot, insertion probability = column 0
                        dir_targets = _hgt[:, 0]
                    else:
                        # Un-mixed: hard integer, insertion=0 → direction=1
                        dir_targets = (_hgt == 0).float()

                    # Note: use reduction='none' + label-purity weighting
                    # Without mixup: labels are all 1.0, behavior unchanged
                    # With mixup: a sample with label=0.7 contributes 70% of the gradient
                    per_sample_loss = F.binary_cross_entropy_with_logits(
                        dir_logits.float(),
                        dir_targets,
                        reduction='none',
                    )
                    sample_weights = batch.labels[pos_mask].float()
                    # BUG-007 fix: per-sample cascade gating + mixup purity weighting
                    gated = per_sample_loss * sample_weights * confidence[pos_mask]
                    dir_loss = gated.sum() / sample_weights.sum().clamp(min=1.0)
        else:
            dir_loss = torch.tensor(0.0, device=device)
        loss_dict["direction"] = dir_loss.item()
        live_losses["direction"] = dir_loss

        # ── 6. Disentanglement loss (cos²) ──
        if output.donor_pool is not None and output.recipient_pool is not None:
            dis_loss = self.disentangle_fn(output.donor_pool, output.recipient_pool)
        else:
            dis_loss = torch.tensor(0.0, device=device)
        loss_dict["disentangle"] = dis_loss.item()

        # ── 7. Refinement loss (optional) ──
        if output.bp_refined is not None:
            offsets = self._compute_refiner_offsets(output, batch)
            valid = (batch.labels > 0.5)  # >0.5 supports mixup float labels and hard labels
            refiner_loss = self.refiner_loss_fn(output.bp_refined, offsets, valid)
        else:
            refiner_loss = torch.tensor(0.0, device=device)
        loss_dict["refiner"] = refiner_loss.item()

        # ── 8. RC consistency loss (PE only) ──
        if output.rc_cls_logits is not None:
            rc_loss = self.rc_loss_fn(output.cls_logits, output.rc_cls_logits)
        else:
            rc_loss = torch.tensor(0.0, device=device)
        loss_dict["rc_consistency"] = rc_loss.item()

        # ── 9. Early-exit auxiliary loss (optional) ──
        ee_loss = torch.tensor(0.0, device=device)
        n_exits = 0
        if output.early_exit_fusion is not None:
            ee_loss = ee_loss + F.binary_cross_entropy_with_logits(
                output.early_exit_fusion.squeeze(-1), batch.labels.float(),
            )
            n_exits += 1
        if output.early_exit_graph is not None:
            ee_loss = ee_loss + F.binary_cross_entropy_with_logits(
                output.early_exit_graph.squeeze(-1), batch.labels.float(),
            )
            n_exits += 1
        if n_exits > 1:
            ee_loss = ee_loss / n_exits  # average the two branches, keeping early_exit_weight semantics consistent
        loss_dict["early_exit"] = ee_loss.item()

        # ── 10. Weighted aggregation ──
        if self.uncertainty_weight is not None:
            # Automatic weights: the four core tasks cls, bp, role, direction
            total_core, weighted_dict = self.uncertainty_weight(
                cls_loss, bp_loss, role_loss, dir_loss,
            )
            # Other losses use manual weights
            total = (
                total_core
                + self.disentangle_weight * dis_loss
                + self.refiner_weight * refiner_loss
                + self.rc_weight * rc_loss
                + self.early_exit_weight * ee_loss
                + self.neg_bp_reg_weight * neg_bp_reg  # P1b
            )
            # Log the automatic weights
            for name, w_loss in weighted_dict.items():
                loss_dict[f"weighted_{name}"] = w_loss.item()
            loss_dict["auto_weights"] = self.uncertainty_weight.get_weights()
        else:
            total = (
                self.cls_weight * cls_loss
                + self.bp_weight * bp_loss
                + self.role_weight * role_loss
                + self.direction_weight * dir_loss
                + self.disentangle_weight * dis_loss
                + self.refiner_weight * refiner_loss
                + self.rc_weight * rc_loss
                + self.early_exit_weight * ee_loss
                + self.neg_bp_reg_weight * neg_bp_reg  # P1b
            )

        # Defense-in-depth: double NaN guard (complements the check inside training_engine step())
        if not torch.isfinite(total):
            import warnings
            nan_tasks = [k for k, v in loss_dict.items()
                         if isinstance(v, (int, float)) and not math.isfinite(v)]
            warnings.warn(
                f"[NaN Loss] NaN/Inf total loss detected. "
                f"NaN tasks: {nan_tasks}. The training step is skipped (zero gradient).",
                RuntimeWarning, stacklevel=2)
            total = torch.tensor(0.0, device=device, requires_grad=True)
            loss_dict["_nan_detected"] = True

        loss_dict["total"] = total.item()

        return total, loss_dict

    @staticmethod
    def _compute_refiner_offsets(output, batch):
        """
        Compute the refinement offset within the window from the coarse heatmap peak and the true breakpoint boundary.

        Uses breakpoint_targets (boundary markers) rather than region_targets (region centroid),
        so the refiner learns to localize the exact virus-host boundary rather than the virus-region center.

        Returns
        -------
        offsets : [B] offset within the window (0 ~ W-1)
        """
        bp = batch.breakpoint_targets  # [B, L], 1.0 at boundary positions
        L = bp.size(-1)
        pos = torch.arange(L, device=bp.device, dtype=bp.dtype)

        coarse_peak = output.bp_heatmaps.argmax(dim=-1).float()  # [B]

        # Find the breakpoint boundary position closest to coarse_peak
        bp_sum = bp.sum(dim=-1)  # [B]
        has_bp = bp_sum > 0  # [B]

        # For each sample: if a breakpoint exists, take the nearest boundary; else fall back to the centroid
        # Nearest-breakpoint search (vectorized). pos[i]==i, so argmin gives the nearest position
        dist = (pos.unsqueeze(0) - coarse_peak.unsqueeze(-1)).abs()  # [B, L]
        dist = torch.where(bp > 0, dist, torch.full_like(dist, float('inf')))
        nearest_bp_pos = dist.argmin(dim=-1).float()  # [B]

        # Fallback: region centroid when no breakpoint exists
        region = batch.region_targets
        region_sum = region.sum(dim=-1).clamp(min=1.0)
        centroid = (region * pos.unsqueeze(0)).sum(dim=-1) / region_sum

        true_pos = torch.where(has_bp, nearest_bp_pos.float(), centroid)

        W = output.bp_refined.size(-1)
        offsets = true_pos - coarse_peak + W // 2
        offsets = offsets.clamp(0, W - 1)

        neg_mask = (batch.labels <= 0.5)  # <=0.5 supports mixup float labels and hard labels
        offsets[neg_mask] = W // 2
        # Positives with an all-zero region (breakpoint truncated by the window) → no valid target, fall back to window center
        no_target = (batch.labels > 0.5) & (batch.region_targets.sum(dim=-1) == 0)
        offsets[no_target] = W // 2

        return offsets.round().long()  # round instead of truncate, removing the ~0.5 position bias


# ═══════════════════════════════════════════════════════════════════════════════ #
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════ #

def build_loss(config: LossConfig = None) -> CombinedLoss:
    """Build the loss function."""
    if config is None:
        config = LossConfig()
    return CombinedLoss(config)


def test() -> bool:
    """Loss self-test."""
    print("=" * 60)
    print("DeepFluHGT  —  Loss Self-Test")
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

    from model.contracts import HGTConfig as ModelConfig, HGTOutput
    from data_pipeline.contracts import HGTBatch

    # Build mock data
    B, L = 4, 303
    batch = HGTBatch(
        input_ids=torch.randint(0, 10, (B, L)),
        attention_mask=torch.ones(B, L, dtype=torch.long),
        bio_features=torch.randn(B, 260),
        labels=torch.tensor([1, 0, 1, 0]),
        breakpoint_targets=torch.zeros(B, L),
        region_targets=torch.zeros(B, L),
        pe_case=torch.zeros(B, dtype=torch.long),
        hgt_mode=torch.tensor([0, 3, 1, 3]),
        n_events=torch.tensor([1, 0, 2, 0]),
    )

    output = HGTOutput(
        cls_logits=torch.randn(B, 1),
        bp_heatmaps=torch.sigmoid(torch.randn(B, L)),
        role_logits=torch.randn(B, 1),
        direction_logits=torch.randn(B, 1),
        bp_refined=torch.softmax(torch.randn(B, 100), dim=-1),
        split_gates=torch.sigmoid(torch.randn(B, L)),
        donor_pool=torch.randn(B, 768),
        recipient_pool=torch.randn(B, 768),
        hidden_states=torch.randn(B, L, 768),
        rc_cls_logits=torch.randn(B, 1),
        early_exit_fusion=torch.randn(B, 1),
        early_exit_graph=torch.randn(B, 1),
    )

    # ── 1. Manual weight mode ──
    print("\n[1] Manual Weight Mode")
    config = LossConfig(use_uncertainty_weight=False)
    loss_fn = build_loss(config)
    total, loss_dict = loss_fn(output, batch)

    log("total loss is scalar", total.dim() == 0)
    log("total loss finite", torch.isfinite(total).item())
    log("loss_dict has all keys",
        all(k in loss_dict for k in ["cls", "bp", "role", "direction",
                                       "disentangle", "refiner",
                                       "rc_consistency", "early_exit", "total"]))

    # ── 2. Automatic weight mode ──
    print("\n[2] Auto Weight Mode (TaskUncertaintyWeight)")
    config = LossConfig(use_uncertainty_weight=True)
    loss_fn = build_loss(config)
    total, loss_dict = loss_fn(output, batch)

    log("auto weight: total loss scalar", total.dim() == 0)
    log("auto weight: total finite", torch.isfinite(total).item())
    log("auto weight: has auto_weights",
        "auto_weights" in loss_dict and isinstance(loss_dict["auto_weights"], dict))
    log("auto weight: 4 task weights",
        len(loss_dict.get("auto_weights", {})) == 4)

    # ── 3. Gradient flow ──
    print("\n[3] Gradient Test")
    config = LossConfig(use_uncertainty_weight=True)
    loss_fn = build_loss(config)

    # Need an output with requires_grad  —  raw logits, full computation graph
    _bp_raw = torch.randn(B, L, requires_grad=True)
    _bp_sig = torch.sigmoid(_bp_raw)
    _ref_raw = torch.randn(B, 100, requires_grad=True)
    _ref_sm = torch.softmax(_ref_raw, dim=-1)
    output_grad = HGTOutput(
        cls_logits=torch.randn(B, 1, requires_grad=True),
        bp_heatmaps=_bp_sig,
        role_logits=torch.randn(B, 1, requires_grad=True),
        direction_logits=torch.randn(B, 1, requires_grad=True),
        bp_refined=_ref_sm,
        donor_pool=torch.randn(B, 768, requires_grad=True),
        recipient_pool=torch.randn(B, 768, requires_grad=True),
        rc_cls_logits=torch.randn(B, 1, requires_grad=True),
        early_exit_fusion=torch.randn(B, 1, requires_grad=True),
        early_exit_graph=torch.randn(B, 1, requires_grad=True),
    )
    total, _ = loss_fn(output_grad, batch)
    total.backward()

    log("cls_logits grad exists", output_grad.cls_logits.grad is not None)
    log("bp_heatmaps grad exists", _bp_raw.grad is not None)
    log("donor_pool grad exists", output_grad.donor_pool.grad is not None)

    # log_vars gradient
    log("uncertainty log_vars grad exists",
        loss_fn.uncertainty_weight.log_vars.grad is not None)

    # ── Summary ──
    total_n = len(results)
    passed = sum(results)
    failed = total_n - passed

    print("\n" + "=" * 60)
    print(f"TOTAL: {total_n}  |  PASSED: {passed}  |  FAILED: {failed}")
    print("=" * 60)

    return failed == 0
