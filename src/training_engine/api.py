# training_engine/api.py
"""
DeepFluHGT  —  Training Engine public interface
============================================
Provides:
  - TrainingEngine: training orchestrator (core)
  - train(): start training from a config (convenience entry point)
  - test(): self-test

trainingcomponent:
  1. Four-group discriminative learning-rate AdamW (global/local/heads/lora)
  2. Physical freezing (during the frozen phase the backbone is excluded from the optimizer, saving GPU memory)
  3. LoRA low-rank adaptation (per-layer-depth adaptivity)
  4. Mixup data augmentation (curriculum learning)
  5. AMP mixed precision
  6. Gradient accumulation + gradient clipping (grad_norm logging)
  7. Direction logits monitor (guards against saturation)
  8. Checkpoint save/load
"""

from __future__ import annotations

import json
import math
import os
import numpy as np
import sys
import time
from collections import defaultdict
from typing import Dict, Optional

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from training_engine.contracts import TrainingConfig
from training_engine.impl.lora import (
    LoRALinear,
    apply_lora_to_model,
    merge_lora_weights,
    get_lora_params,
)
from training_engine.impl.mixup import mixup_batch
from training_engine.impl.optimizer import (
    create_optimizer,
    create_frozen_optimizer,
    create_unfrozen_optimizer,
    build_scheduler,
    build_per_group_scheduler,
    migrate_optimizer_state_by_name,
)


# ═══════════════════════════════════════════════════════════════════════════════ #
#  AMP helpers  —  bf16/fp16 dtype with GradScaler decision (Contig fp16 overflow fix)
# ═══════════════════════════════════════════════════════════════════════════════ #

def needs_grad_scaler(amp_dtype: str, use_amp: bool) -> bool:
    """Whether a GradScaler is needed.

    bf16 has the fp32 dynamic range (±3.4e38) and rarely underflows → no scaler needed;
    fp16 (±65504) needs a scaler to prevent backward underflow. Never needed when use_amp=False.

    Background: Contig E5 79% fp16 forward overflow (memory contig-fp16-overflow-rootcause),
    switching to bf16 removed the forward overflow and needs no scaler.
    """
    if not use_amp:
        return False
    return (amp_dtype or "").lower() not in ("bfloat16", "bf16")


def amp_torch_dtype(amp_dtype: str) -> torch.dtype:
    """amp_dtype string → torch.dtype. Unknown/None defaults to fp16 (backward compatible)."""
    if (amp_dtype or "").lower() in ("bfloat16", "bf16"):
        return torch.bfloat16
    return torch.float16


# ═══════════════════════════════════════════════════════════════════════════════ #
#  TrainingEngine  —  training orchestrator
# ═══════════════════════════════════════════════════════════════════════════════ #

class TrainingEngine:
    """
    DeepFluHGT training orchestrator.

    Features:
      - Physical freezing (backbone excluded from the optimizer, saving AdamW-state GPU memory)
      - Discriminative unfreezing (global/local group LRs)
      - Mixup curriculum learning (disabled during the warmup phase)
      - LoRA injection and merge
      - Gradient accumulation + gradient clipping (grad_norm logging)
      - Direction logits monitor (guards against saturation)
      - AMP mixed precision
      - Checkpoint save/load

    Parameters
    ----------
    model : nn.Module
        DeepFluHGT model
    loss_fn : nn.Module
        CombinedLoss lossfunction
    config : TrainingConfig
        training configuration
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config: TrainingConfig,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.config = config

        # Device
        self.device = config.get_device()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0   # best model selected by AUPRC (higher is better)
        self.best_epoch = 0

        # Optimizer / scheduler / AMP
        self.optimizer = None
        self.scheduler = None
        self.scaler = None

        # early stopping
        self.patience_counter = 0
        self.optimal_threshold = 0.5  # searched/updated from the validation set during training

        # Training history
        self.history = {
            "train_loss": [],
            "eval_loss": [],
            "lr": [],
        }

        # ── [R3] grad_norm accumulator ──
        self._grad_norm_accum = 0.0
        self._grad_norm_count = 0

    def setup(self, steps_per_epoch: int = 0) -> None:
        """Initialize the optimizer, scheduler, LoRA, and AMP.

        steps_per_epoch: actual number of steps per epoch (0 = fall back to the legacy estimate)
        """
        cfg = self.config

        # LoRA injection
        if cfg.use_lora:
            apply_lora_to_model(
                self.model,
                rank=cfg.lora_rank,
                alpha=cfg.lora_alpha,
                dropout=cfg.lora_dropout,
                target_modules=cfg.lora_target_modules,
            )
            print(f"[LoRA] injection complete, trainable LoRA parameters: "
                  f"{sum(p.numel() for p in get_lora_params(self.model)):,}")

        # Device transfer
        self.model = self.model.to(self.device)
        self.loss_fn = self.loss_fn.to(self.device)

        # ── [R1] Physical freezing: during the frozen phase create only a heads+lora optimizer ──
        if cfg.freeze_epochs > 0 and self.current_epoch < cfg.freeze_epochs:
            self._freeze_backbone_physical()
            self.optimizer = create_frozen_optimizer(
                self.model,
                lr=cfg.learning_rate,
                lora_lr_multiplier=cfg.lora_lr_multiplier,
                weight_decay=cfg.weight_decay,
                adam_beta1=cfg.adam_beta1,
                adam_beta2=cfg.adam_beta2,
                adam_eps=cfg.adam_eps,
            )
        elif cfg.freeze_epochs > 0:
            # Resume: past the frozen phase; create the four-group discriminative optimizer (consistent with _unfreeze)
            from training_engine.impl.optimizer import create_unfrozen_optimizer
            self.optimizer = create_unfrozen_optimizer(
                self.model,
                lr=cfg.learning_rate,
                global_lr_ratio=cfg.global_lr_ratio,
                local_lr_ratio=cfg.local_lr_ratio,
                lora_lr_multiplier=cfg.lora_lr_multiplier,
                weight_decay=cfg.weight_decay,
                backbone_prefix=cfg.backbone_prefix,
                global_encoder_prefix=cfg.global_encoder_prefix,
                local_encoder_prefix=cfg.local_encoder_prefix,
                adam_beta1=cfg.adam_beta1,
                adam_beta2=cfg.adam_beta2,
                adam_eps=cfg.adam_eps,
            )
        else:
            self.optimizer = create_optimizer(
                self.model,
                lr=cfg.learning_rate,
                backbone_lr_ratio=cfg.backbone_lr_ratio,
                lora_lr_multiplier=cfg.lora_lr_multiplier,
                weight_decay=cfg.weight_decay,
                backbone_prefix=cfg.backbone_prefix,
                adam_beta1=cfg.adam_beta1,
                adam_beta2=cfg.adam_beta2,
                adam_eps=cfg.adam_eps,
            )

        # Scheduler: prefer max_steps, else actual steps_per_epoch, finally the legacy estimate
        if cfg.max_steps > 0:
            total_steps = cfg.max_steps
        elif steps_per_epoch > 0:
            # Divide by accumulation_steps: the scheduler advances inside optimizer_step(),
            # rather than on every micro-batch (BUG-006 fix)
            optimizer_steps_per_epoch = max(1, steps_per_epoch // cfg.accumulation_steps)
            total_steps = optimizer_steps_per_epoch * cfg.max_epochs
        else:
            total_steps = cfg.max_epochs * 1000  # legacy estimate (backward compatible)
        warmup_steps = int(total_steps * cfg.warmup_ratio)
        self._total_scheduler_steps = total_steps
        self._warmup_steps = warmup_steps
        self.scheduler = build_scheduler(
            self.optimizer, total_steps, warmup_steps, cfg.schedule_type,
        )

        # AMP
        if needs_grad_scaler(cfg.amp_dtype, cfg.use_amp) and self.device == "cuda":
            self.scaler = torch.amp.GradScaler("cuda")
        # bf16 (amp_dtype="bfloat16"): scaler stays None  —  fp32-like range, rarely underflows, no scaler needed

    # ── Backbone freeze / unfreeze ──

    @staticmethod
    def _is_embed_param(name: str) -> bool:
        """Determine whether a parameter is a token embedding (used for FGM adversarial perturbation)."""
        nl = name.lower()
        if 'pos' in nl:
            return False  # exclude positional encodings
        return any(kw in nl for kw in ('embed', 'wte', 'word_embedding', 'tok_emb'))

    def _freeze_backbone_physical(self) -> None:
        """
        Physical freezing: set requires_grad=False and exclude from the optimizer.
        Unlike a plain requires_grad=False, backbone parameters never enter AdamW,
        avoiding allocation of Adam momentum/variance states for ~685M frozen parameters (saves ~10GB of GPU memory).
        """
        for name, param in self.model.named_parameters():
            if self.config.backbone_prefix in name:
                if "lora_A" in name or "lora_B" in name:
                    continue
                param.requires_grad = False

    def _unfreeze_backbone_with_discriminative_lr(self) -> None:
        """
        [R2] Discriminative unfreezing: unfreeze the backbone and rebuild the optimizer.

        Rebuild strategy:
          - global_encoder: lr × global_lr_ratio (650M parameters, very conservative)
          - local_encoder:  lr × local_lr_ratio  (35M parameters, moderate)
          - heads:          lr × 1.0  (normal)
          - lora:           lr × lora_lr_multiplier  (fast adaptation)

        LR-DEAD-001 fix (architecture-level fix):
          - use build_per_group_scheduler() instead of the unified build_scheduler()
          - backbone group: initial_lr set to the target LR, warmup_from_zero=True (linear ramp from 0)
          - heads/LoRA group: initial_lr continues from the current value, warmup_from_zero=False (skip warmup)
          - LambdaLR effective_lr = initial_lr × lambda(step)
            backbone: initial_lr × (step+1)/warmup_steps → linear ramp to the target LR
            heads:    initial_lr × cosine(step) → cosine decay starting from the continued LR

        LR-DROP-001 fix:
          - heads/LoRA group warmup_steps=0, lambda(step) goes straight into cosine decay
          - no more 300× LR crash
        """
        cfg = self.config

        # 1. Unfreeze all backbone parameters
        for name, param in self.model.named_parameters():
            if self.config.backbone_prefix in name:
                param.requires_grad = True

        # 2. Save old optimizer learning rates (for heads-group LR continuation)
        old_lrs = {g.get("name", f"group_{i}"): g["lr"]
                   for i, g in enumerate(self.optimizer.param_groups)}

        # 3. Rebuild the optimizer (four-group discriminative LRs)
        self.optimizer = create_unfrozen_optimizer(
            self.model,
            lr=cfg.learning_rate,
            global_lr_ratio=cfg.global_lr_ratio,
            local_lr_ratio=cfg.local_lr_ratio,
            lora_lr_multiplier=cfg.lora_lr_multiplier,
            weight_decay=cfg.weight_decay,
            backbone_prefix=cfg.backbone_prefix,
            global_encoder_prefix=cfg.global_encoder_prefix,
            local_encoder_prefix=cfg.local_encoder_prefix,
            adam_beta1=cfg.adam_beta1,
            adam_beta2=cfg.adam_beta2,
            adam_eps=cfg.adam_eps,
        )

        # 4. Set initial_lr  —  LR-DEAD-001 root fix
        #    LambdaLR: effective_lr = initial_lr × lr_lambda(step)
        #    initial_lr must be > 0 for the backbone to actually learn
        remaining_steps = max(1, self._total_scheduler_steps - self.global_step)
        warmup_steps_unfreeze = max(1, int(0.05 * remaining_steps))
        old_heads_lr = old_lrs.get("heads", cfg.learning_rate)
        old_lora_lr = old_lrs.get("lora", cfg.learning_rate * cfg.lora_lr_multiplier)

        warmup_configs = []
        for g in self.optimizer.param_groups:
            name = g.get("name", "")
            if "global_encoder" in name:
                target_lr = cfg.learning_rate * cfg.global_lr_ratio
                g["lr"] = target_lr
                g["initial_lr"] = target_lr
                warmup_configs.append({
                    "warmup_steps": warmup_steps_unfreeze,
                    "warmup_from_zero": True,
                })
            elif "local_encoder" in name:
                target_lr = cfg.learning_rate * cfg.local_lr_ratio
                g["lr"] = target_lr
                g["initial_lr"] = target_lr
                warmup_configs.append({
                    "warmup_steps": warmup_steps_unfreeze,
                    "warmup_from_zero": True,
                })
            elif "heads" in name:
                # Heads: continue the decayed LR, no warmup (LR-DROP-001 fix)
                g["lr"] = old_heads_lr
                g["initial_lr"] = old_heads_lr
                warmup_configs.append({
                    "warmup_steps": 0,
                    "warmup_from_zero": False,
                })
            elif "lora" in name:
                g["lr"] = old_lora_lr
                g["initial_lr"] = old_lora_lr
                warmup_configs.append({
                    "warmup_steps": 0,
                    "warmup_from_zero": False,
                })
            else:
                # Unknown group: keep the LR from creation
                g["initial_lr"] = g["lr"]
                warmup_configs.append({
                    "warmup_steps": 0,
                    "warmup_from_zero": False,
                })

        # 5. Rebuild the scheduler  —  independent per-group warmup
        self.scheduler = build_per_group_scheduler(
            self.optimizer, remaining_steps, warmup_configs, cfg.schedule_type,
        )

        # 6. Diagnostic output: LR information for each param group
        print(f"[Epoch {self.current_epoch}] backbone unfrozen  —  four-group discriminative LR "
              f"(warmup={warmup_steps_unfreeze} steps, remaining={remaining_steps}):")
        for i, g in enumerate(self.optimizer.param_groups):
            n = sum(p.numel() for p in g["params"])
            nm = g.get("name", "?")
            wd = g.get("weight_decay", -1)
            base_lr = self.scheduler.base_lrs[i]
            print(f"  {nm:>25s}: initial_lr={base_lr:.2e}, current_lr={g['lr']:.2e}, "
                  f"params={n:,}, wd={wd}, warmup_from_zero={warmup_configs[i].get('warmup_from_zero', False)}")

    def _should_freeze(self) -> bool:
        return self.current_epoch < self.config.freeze_epochs

    # ── Mixup curriculum learning ──

    def _should_use_mixup(self) -> bool:
        return (
            self.config.mixup_alpha > 0
            and self.current_epoch >= self.config.mixup_warmup_epochs
        )

    # ── Single training step ──

    def step(self, batch) -> Dict[str, float]:
        """
        Execute a single training step.

        Parameters
        ----------
        batch : HGTBatch

        Returns
        -------
        loss_dict : {name: float}
        """
        cfg = self.config
        self.model.train()

        # Mixup
        lam = 1.0
        if self._should_use_mixup():
            batch, lam = mixup_batch(batch, alpha=cfg.mixup_alpha)

        # AMP
        use_amp = cfg.use_amp                       # autocast switch (on for both bf16/fp16)
        use_scaler = self.scaler is not None        # GradScaler only for fp16; bf16=None

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_torch_dtype(cfg.amp_dtype)):
            output = self.model(batch)
            total_loss, loss_dict = self.loss_fn(output, batch)

        # ── [R4] Direction logits monitor (guards against saturation) ──
        if output.direction_logits is not None:
            with torch.no_grad():
                dl = output.direction_logits.detach().float()
                # Sync optimization: 4× .item() → 1× .cpu() (merge GPU→CPU sync points, reduce GPU idle)
                _dl_stats = torch.stack([dl.mean(), dl.std(), dl.max(), dl.min()]).cpu()
                loss_dict["dir_mean"] = _dl_stats[0].item()
                loss_dict["dir_std"] = _dl_stats[1].item()
                loss_dict["dir_max"] = _dl_stats[2].item()
                loss_dict["dir_min"] = _dl_stats[3].item()

        # NaN guard: if the loss is NaN/Inf, skip backward for this batch
        # m-12 note: persistent NaN yields many zero-gradient steps; the trainer should watch log warnings
        if not torch.isfinite(total_loss):
            print(f"[WARNING] NaN/Inf loss detected at step {self.global_step}, "
                  f"skipping backward. (persistent NaN may indicate training issues)")
            loss_dict["total"] = float("nan")
            loss_dict["mixup_lam"] = lam
            loss_dict["_skipped"] = True
            return loss_dict

        # Scale the loss for gradient accumulation
        scaled_loss = total_loss / cfg.accumulation_steps

        # Backward pass
        if use_scaler:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        # ── FGM adversarial training ──
        # m-04 note: FGM acts on the post-mixup batch (a linear combination of two samples),
        # so the adversarial perturbation target is the mixed representation. This is standard practice,
        # but semantically it differs from adversarial perturbation of the raw data.
        if cfg.use_fgm and self.current_epoch >= cfg.fgm_start_epoch:
            embed_params = []
            for name, param in self.model.named_parameters():
                if param.requires_grad and self._is_embed_param(name):
                    embed_params.append(param)

            if embed_params:
                backup = [p.data.clone() for p in embed_params]

                try:
                    for p in embed_params:
                        if p.grad is not None:
                            grad_norm = p.grad.norm()
                            if grad_norm > 0:
                                perturbation = cfg.fgm_epsilon * p.grad / grad_norm
                                p.data.add_(perturbation)

                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_torch_dtype(cfg.amp_dtype)):
                        output_adv = self.model(batch)
                        total_loss_adv, _ = self.loss_fn(output_adv, batch)

                    if torch.isfinite(total_loss_adv):
                        # FGM-ACC-001 fix: divide by accumulation_steps
                        # FGM fires on every micro-step, not once per accumulation window.
                        # Without scaling, accumulation_steps=2 would double the effective epsilon.
                        fgm_scaled_loss = total_loss_adv / cfg.accumulation_steps
                        if use_scaler:
                            self.scaler.scale(fgm_scaled_loss).backward()
                        else:
                            fgm_scaled_loss.backward()
                finally:
                    # FGM-SAFE-001 fix: try/finally guarantees parameter restoration
                    for p, b in zip(embed_params, backup):
                        p.data.copy_(b)

        loss_dict["total"] = total_loss.item()
        loss_dict["mixup_lam"] = lam

        return loss_dict

    def optimizer_step(self) -> Optional[float]:
        """
        Execute an optimizer step (NaN gradient check + gradient clipping + step + scheduler).

        [R3] Returns grad_norm for logging; None means the step was skipped.
        """
        # If there is no gradient at all (the whole accumulation window was skipped), skip directly
        has_any_grad = any(
            p.grad is not None for p in self.model.parameters() if p.requires_grad
        )
        if not has_any_grad:
            return None

        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        # NaN gradient check: after unscale, check whether any parameter gradient contains NaN/Inf
        has_nan_grad = False
        for p in self.model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                has_nan_grad = True
                break

        if has_nan_grad:
            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler is not None:
                self.scaler.update()
            return None

        # ── [R3] Gradient clipping + grad_norm logging ──
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable, self.config.gradient_clip_norm,
        )
        grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)

        # Accumulate grad_norm for epoch-level logging
        if math.isfinite(grad_norm_val):
            self._grad_norm_accum += grad_norm_val
            self._grad_norm_count += 1

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1

        return grad_norm_val

    # ── Epoch training ──

    def train_epoch(self, dataloader) -> Dict[str, float]:
        """
        Execute one epoch of training.

        Parameters
        ----------
        dataloader : DataLoader

        Returns
        -------
        epoch_metrics : {name: mean_loss}
        """
        cfg = self.config

        # ── [R1/R2] Freeze/unfreeze control ──
        if self._should_freeze():
            self._freeze_backbone_physical()
        elif cfg.freeze_epochs > 0 and self.current_epoch == cfg.freeze_epochs:
            self._unfreeze_backbone_with_discriminative_lr()

        # Reset the grad_norm accumulator
        self._grad_norm_accum = 0.0
        self._grad_norm_count = 0

        epoch_losses = {}
        n_steps = 0
        steps_in_window = 0
        last_grad_norm = 0.0

        self.optimizer.zero_grad(set_to_none=True)

        for step_idx, batch in enumerate(dataloader):
            # Device transfer
            batch = _batch_to_device(batch, self.device)

            # Single step
            loss_dict = self.step(batch)

            # Accumulate metrics (NaN batches are not accumulated)
            for k, v in loss_dict.items():
                if isinstance(v, (int, float)) and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + v
            if not loss_dict.get("_skipped", False):
                n_steps += 1

            # Gradient accumulation: run the optimizer step at the window boundary
            steps_in_window += 1
            if steps_in_window >= cfg.accumulation_steps:
                gn = self.optimizer_step()
                if gn is not None:
                    last_grad_norm = gn
                steps_in_window = 0

            # Log  —  output every log_interval steps
            if (step_idx + 1) % cfg.log_interval == 0:
                avg = {k: v / max(1, n_steps) for k, v in epoch_losses.items()
                       if isinstance(v, (int, float))}
                lr = self.optimizer.param_groups[0]["lr"]
                frozen = "FROZEN" if self._should_freeze() else "UNFROZEN"
                total_steps = len(dataloader)
                progress = f"{step_idx+1}/{total_steps}"
                pct = (step_idx + 1) / total_steps * 100

                # ── [R3] grad_norm + [R4] direction logits ──
                print(f"  [Epoch {self.current_epoch+1}  {progress} ({pct:.0f}%)  "
                      f"backbone={frozen}]  "
                      f"loss={avg.get('total', 0):.4f}  "
                      f"cls={avg.get('cls', 0):.4f}  "
                      f"bp={avg.get('bp', 0):.4f}  "
                      f"role={avg.get('role', 0):.4f}  "
                      f"dir={avg.get('direction', 0):.4f}  "
                      f"lr={lr:.2e}  "
                      f"gn={last_grad_norm:.2f}  "
                      f"dl=[{avg.get('dir_mean', 0):.2f},{avg.get('dir_std', 0):.2f},"
                      f"{avg.get('dir_max', 0):.1f},{avg.get('dir_min', 0):.1f}]")

        # Trailing incomplete accumulation window
        if steps_in_window > 0:
            gn = self.optimizer_step()
            if gn is not None:
                last_grad_norm = gn

        epoch_metrics = {k: v / max(1, n_steps) for k, v in epoch_losses.items()
                         if isinstance(v, (int, float))}
        epoch_metrics["epoch"] = self.current_epoch
        epoch_metrics["lr"] = self.optimizer.param_groups[0]["lr"]

        # [R3] Mean grad_norm
        if self._grad_norm_count > 0:
            epoch_metrics["grad_norm"] = self._grad_norm_accum / self._grad_norm_count

        self.current_epoch += 1

        return epoch_metrics

    # ── evaluation ──

    @torch.no_grad()
    def eval_epoch(self, dataloader, aggregate_contig: bool = False, threshold: float = 0.5,
                   find_optimal_threshold: bool = True, early_exit: bool = True) -> Dict[str, float]:
        """
        Gradient-free evaluation.

        Parameters
        ----------
        dataloader : DataLoader
        aggregate_contig : bool
            In sliding-window mode, aggregate window predictions per contig before computing metrics.
            Aggregation strategy: contig_prob = adaptive top-K mean (of window probabilities), contig_label = max (of window labels).
            The loss stays window-level (used for early stopping); acc/auc/f1 etc. use the contig level.
        early_exit : bool
            Whether to enable early-exit inference (default True). Set to False to obtain baseline metrics from the full forward pass.
            M-05 note: early_exit=True introduces selection bias; when reporting official metrics,
            provide the early_exit=False baseline for comparison.

        Returns
        -------
        eval_metrics : {eval_loss, accuracy, ...}
        """
        self.model.eval()

        total_loss = 0.0
        n_steps = 0
        all_probs = []
        all_labels = []
        all_seq_ids = []
        all_sample_types = []
        # R2-M5 fix (2026-06-22): collect direction logits + hgt_mode for direction accuracy
        # METRIC-DIR-001 (2026-08-14): also collect a window-level label snapshot; the direction metric
        # uses window-level triplets end-to-end, decoupled from the contig aggregation path (evaluation-only)
        all_dir_logits = []
        all_hgt_modes = []
        all_dir_labels = []

        for batch in dataloader:
            batch = _batch_to_device(batch, self.device)

            # NEW-06 fix: the early_exit parameter is configurable, allowing baseline comparison
            output = self.model(batch, early_exit=early_exit)
            total, loss_dict = self.loss_fn(output, batch)

            total_loss += total.item()
            n_steps += 1

            probs = torch.sigmoid(output.cls_logits.squeeze(-1))
            all_probs.append(probs.cpu())
            all_labels.append(batch.labels.cpu())

            # R2-M5 fix: collect direction logits + hgt_mode
            # METRIC-DIR-001: append logits/mode/label as aligned columns (only when all three
            # are available), guaranteeing per-window alignment; contig aggregation replacing all_labels does not ripple here
            _batch_hgt_mode = getattr(batch, 'hgt_mode', None)
            if output.direction_logits is not None and _batch_hgt_mode is not None:
                all_dir_logits.append(output.direction_logits.detach().cpu())
                all_hgt_modes.append(_batch_hgt_mode.detach().cpu())
                all_dir_labels.append(batch.labels.detach().cpu())

            # Collect sample_types (for stratified evaluation)
            batch_sample_types = getattr(batch, 'sample_types', None)
            if batch_sample_types is not None:
                all_sample_types.extend(list(batch_sample_types))
            else:
                all_sample_types.extend(['UNKNOWN'] * probs.shape[0])

            # Collect contig IDs (for sliding-window aggregation)
            if aggregate_contig:
                seq_ids = getattr(batch, 'original_seq_ids', None) or []
                if len(seq_ids) < probs.shape[0]:
                    seq_ids = getattr(batch, 'seq_ids', None) or \
                              [f"unk_{i}" for i in range(probs.shape[0])]
                all_seq_ids.extend(seq_ids)

        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy().astype(int)

        # NaN guard: replace NaN probabilities with 0.5
        nan_mask = np.isnan(all_probs)
        if nan_mask.any():
            all_probs = np.where(nan_mask, 0.5, all_probs)

        # ── Sliding window → contig-level aggregation (RFC-CONTIG-STRAT-001 change B) ──
        _contig_aggregated = False
        # Back up window-level (pre-aggregation) data for the supplementary window-level report; _aggregate returns new arrays, zero-copy reference here
        win_probs, win_labels = all_probs, all_labels
        win_types = list(all_sample_types)
        if aggregate_contig and all_seq_ids:
            all_probs, all_labels, all_sample_types = self._aggregate_contig_predictions(
                all_probs, all_labels, all_seq_ids, all_sample_types)
            _contig_aggregated = True   # change A already syncs sample_types, no need to clear

        preds = (all_probs >= threshold).astype(int)
        total_correct = (preds == all_labels).sum()
        total_samples = len(all_labels)

        metrics = {
            "eval_loss": total_loss / max(1, n_steps),
            "accuracy": float(total_correct / max(1, total_samples)),
        }
        # M-12 fix: explicitly mark the aggregation state to prevent misleading downstream .get("cal_ece", 0.0)
        if _contig_aggregated:
            metrics["_contig_aggregated"] = True

        # sklearn metric
        n_pos = int(all_labels.sum())
        n_neg = total_samples - n_pos
        if n_pos > 0 and n_neg > 0:
            from sklearn.metrics import (
                roc_auc_score, average_precision_score,
                f1_score, precision_score, recall_score,
                confusion_matrix,
            )
            metrics["auc"] = float(roc_auc_score(all_labels, all_probs))
            metrics["auprc"] = float(average_precision_score(all_labels, all_probs))
            metrics["f1"] = float(f1_score(all_labels, preds, zero_division=0))
            metrics["precision"] = float(precision_score(all_labels, preds, zero_division=0))
            metrics["recall"] = float(recall_score(all_labels, preds, zero_division=0))
            tn, fp, fn, tp = confusion_matrix(all_labels, preds, labels=[0, 1]).ravel()
            metrics["specificity"] = float(tn / max(1, tn + fp))

            # Search the F1-maximizing optimal threshold (validation phase only)
            if find_optimal_threshold:
                optimal_thr = self._find_optimal_threshold(all_labels, all_probs)
                metrics["optimal_threshold"] = optimal_thr
                optimal_preds = (all_probs >= optimal_thr).astype(int)
                metrics["f1_at_optimal"] = float(
                    f1_score(all_labels, optimal_preds, zero_division=0))
                metrics["precision_at_optimal"] = float(
                    precision_score(all_labels, optimal_preds, zero_division=0))
                metrics["recall_at_optimal"] = float(
                    recall_score(all_labels, optimal_preds, zero_division=0))

            # Contract (frozen-period evaluation enhancement by design): audit metric  —  recall at a fixed
            # precision. Under class imbalance AUPRC is the primary metric; this metric characterizes working-point reachability;
            # stored into the checkpoint alongside the metrics dict.
            from sklearn.metrics import precision_recall_curve
            _pr_prec, _pr_rec, _ = precision_recall_curve(all_labels, all_probs)
            for _pt in (0.90, 0.95):
                _m = _pr_prec >= _pt
                metrics[f"recall_at_p{int(_pt * 100)}"] = (
                    float(_pr_rec[_m].max()) if _m.any() else 0.0)

        # ── Stratified evaluation & temperature-scaling calibration (RFC-CONTIG-STRAT-001 change B: dual report + ECE decoupling) ──
        # M-08 note: temperature scaling is fit on the current data, so the ECE is an optimistic estimate.
        # Official reports should use a held-out calibration set or report only the uncalibrated ECE.
        metrics.update(TrainingEngine._compute_layered_metrics(
            win_probs, win_labels, win_types,
            all_probs, all_labels, all_sample_types,
            threshold=threshold, layered=_contig_aggregated))

        metrics["_all_probs"] = all_probs
        metrics["_all_labels"] = all_labels

        # ── direction_accuracy (R2-M5; METRIC-DIR-001 bugfix 2026-08-14) ──
        # fixbefore bug: dir_logits forwindow-level, but aggregate_contig pathbelow all_labels
        # was already replaced with contig level → shape guard always False → metric never written → the
        # `.get(..., 0) or 0` at the persistence site silently filled in 0.
        # Fix: compute (window-level) from all_dir_labels / all_hgt_modes collected as per-window pairs,
        # positives only, without affecting the primary metric or training. Output protocol: see _compute_direction_metrics.
        metrics.update(TrainingEngine._compute_direction_metrics(
            all_dir_logits, all_hgt_modes, all_dir_labels))

        return metrics

    @staticmethod
    def _compute_direction_metrics(dir_logits_list, hgt_modes_list, labels_list):
        """METRIC-DIR-001 (2026-08-14): window-level direction_accuracy, evaluation-only.

        Input: the window-level triplet lists collected batch-by-batch in eval_epoch (logits/mode/label
        appended as aligned columns), fully decoupled from the contig-level label replacement in aggregate_contig.
        Semantics consistent with evaluation/impl/metrics.compute_direction_accuracy:
        Positives only (label==1), hgt_mode==0 (insertion) → direction=1.

        Three-field output protocol (0.0 forbidden as a stand-in for "does not exist"):
          direction_accuracy   float | None  (None = not evaluable this run)
          direction_evaluable  bool
          direction_n / direction_correct  int (positive window count / correct prediction count)
          direction_status     "OK" | "NOT_APPLICABLE" | "SHAPE_MISMATCH"
        """
        out = {
            "direction_accuracy": None,
            "direction_evaluable": False,
            "direction_n": 0,
            "direction_correct": 0,
            "direction_status": "NOT_APPLICABLE",
        }
        if not (dir_logits_list and hgt_modes_list and labels_list):
            return out  # model has no direction head / data has no hgt_mode → does not exist, not 0.0
        dir_logits = torch.cat(dir_logits_list).squeeze(-1)
        hgt_modes = torch.cat(hgt_modes_list)
        win_labels = torch.cat(labels_list).long()
        if not (dir_logits.shape[0] == win_labels.shape[0] == hgt_modes.shape[0]):
            out["direction_status"] = "SHAPE_MISMATCH"
            return out
        pos_mask = win_labels == 1
        out["direction_n"] = int(pos_mask.sum().item())
        if out["direction_n"] == 0:
            return out  # no positive windows → NOT_APPLICABLE / None
        pred_dir = (torch.sigmoid(dir_logits[pos_mask]) > 0.5).long()
        true_dir = (hgt_modes[pos_mask] == 0).long()  # insertion=0 → direction=1
        out["direction_correct"] = int((pred_dir == true_dir).sum().item())
        out["direction_accuracy"] = out["direction_correct"] / out["direction_n"]
        out["direction_evaluable"] = True
        out["direction_status"] = "OK"
        return out

    @staticmethod
    def _find_optimal_threshold(labels: np.ndarray, probs: np.ndarray,
                                min_recall: float = 0.5) -> float:
        """Exact search for the F1-maximizing threshold on the precision-recall curve.

        Parameters
        ----------
        min_recall : float
            Minimum recall constraint, avoiding meaningless thresholds with tiny R but P=1.0.
        """
        from sklearn.metrics import precision_recall_curve
        precisions, recalls, thresholds = precision_recall_curve(labels, probs)
        # thresholds is 1 shorter than precisions/recalls (the last entry precisions=1, recalls=0 is the boundary point)
        f1_scores = 2 * precisions[:-1] * recalls[:-1] / \
            np.maximum(precisions[:-1] + recalls[:-1], 1e-8)
        # Apply the min_recall constraint: zero out F1 for thresholds that miss the recall floor
        valid_mask = recalls[:-1] >= min_recall
        if not valid_mask.any():
            # No threshold meets the recall floor; fall back to the highest-recall one
            best_idx = int(np.argmax(recalls[:-1]))
        else:
            masked_f1 = np.where(valid_mask, f1_scores, 0.0)
            best_idx = int(np.argmax(masked_f1))
        return float(thresholds[best_idx])

    @staticmethod
    def _aggregate_contig_predictions(
        probs: np.ndarray, labels: np.ndarray, seq_ids: list,
        sample_types: list = None,
    ) -> tuple:
        """
        Aggregate window-level predictions to the contig level.

        Strategy (operating on sigmoid probabilities):
          - Short contig (≤3 windows): max-pooling (degenerates to the traditional strategy)
          - Medium contig (4-10 windows): top-3 mean
          - Long contig (>10 windows): max(top-3 mean, P90)
            top-K is the mean of the K highest-probability windows, robust to single-window misclassification.
            P90 ensures the top 10% of windows carry enough signal to call the contig positive.
          - contig_label = max over all window labels of the contig

        sample_types : list, optional
            Per-window sample_type. Within a contig, window types are consistent; aggregation takes the first window as representative.
            If None, contig_sample_types are all None (backward compatible).

        Returns
        -------
        (contig_probs, contig_labels, contig_sample_types) : tuple
            The three are order-aligned (same contig_data.values() iteration). RFC-CONTIG-STRAT-001 change A.
        """
        import itertools
        contig_data = defaultdict(lambda: {"probs": [], "label": 0, "type": None})
        types_iter = (iter(sample_types) if sample_types is not None
                      else itertools.repeat(None))
        for prob, label, sid, stype in zip(probs, labels, seq_ids, types_iter):
            d = contig_data[sid]
            d["probs"].append(float(prob))
            d["label"] = max(d["label"], int(label))
            if d["type"] is None:
                d["type"] = stype

        contig_probs = []
        for d in contig_data.values():
            window_probs = sorted(d["probs"], reverse=True)
            n = len(window_probs)

            if n <= 3:
                # Short contig: max-pooling
                contig_probs.append(window_probs[0])
            elif n <= 10:
                # Medium contig: top-3 mean
                k = min(3, n)
                contig_probs.append(np.mean(window_probs[:k]))
            else:
                # Long contig: max(top-3 mean, P90)
                k = min(3, n)
                top_k_mean = np.mean(window_probs[:k])
                p90 = np.percentile(d["probs"], 90)
                contig_probs.append(max(top_k_mean, p90))

        contig_probs = np.array(contig_probs, dtype=np.float32)
        contig_labels = np.array(
            [d["label"] for d in contig_data.values()], dtype=int)
        contig_sample_types = [d["type"] for d in contig_data.values()]

        n_contigs = len(contig_probs)
        n_pos = int(contig_labels.sum())
        n_neg = n_contigs - n_pos
        print(f"  [Contig Aggregation] contigs={n_contigs}, "
              f"pos={n_pos}, neg={n_neg}, "
              f"ratio=1:{n_neg / max(1, n_pos):.1f}")

        return contig_probs, contig_labels, contig_sample_types

    @staticmethod
    def _compute_layered_metrics(
        win_probs, win_labels, win_types,
        main_probs, main_labels, main_types,
        threshold: float, layered: bool,
    ) -> dict:
        """Compute stratified (hard/easy neg) + ECE metrics, dual report (RFC-CONTIG-STRAT-001 change B).

        Primary report strat_*/cal_*: when layered=True use main (post contig-level aggregation);
            else use win (PE/sample level).
        Supplementary window-level window_strat_*/window_cal_*: produced only when layered=True (pre-aggregation, clean,
            free of the M-01 aggregation bias), for dual-report comparison.
        ECE decoupled from stratification: ECE needs only probs+labels, independent of sample_types length.

        Parameters
        ----------
        win_probs, win_labels, win_types : window-level (pre-aggregation).
        main_probs, main_labels, main_types : primary report level (post-aggregation contig level; equals win when not aggregating).
        threshold : classification threshold.
        layered : whether to emit the supplementary window-level report (i.e. aggregation occurred).
        """
        from evaluation.impl.metrics import (
            compute_stratified_metrics, compute_calibration_metrics)

        out = {}

        def _add(prefix: str, probs, labels, types) -> None:
            n_pos = int(np.asarray(labels).sum())
            n_neg = len(labels) - n_pos
            if n_pos > 0 and n_neg > 0:
                # ECE decoupling: needs only probs+labels (independent of sample_types)
                cal = compute_calibration_metrics(probs, labels)
                out.update({f"{prefix}cal_{k}": v for k, v in cal.items()})
                # Stratified: requires sample_types length alignment
                if len(types) == len(probs):
                    strat = compute_stratified_metrics(probs, labels, types, threshold)
                    out.update({f"{prefix}strat_{k}": v for k, v in strat.items()})

        # primaryreport (contig level or samplelevel)
        _add("", main_probs, main_labels, main_types)
        # Supplementary window-level report (aggregation path only, dual report)
        if layered:
            _add("window_", win_probs, win_labels, win_types)
        return out

    # ── Checkpoint ──

    def save_checkpoint(self, path: str, metrics: Optional[Dict] = None) -> None:
        """savetrainingcheckpoint.

        Includes:
          - model weights, optimizer state, scheduler state, scaler state
          - RNG state (M-06 fix: reproducible resume)
          - param_name_order (RESUME-001 fix: parameter-name migration)
          - optimal_threshold top-level field (CHKPT-THRESHOLD-001 fix)
          - best_metric_mode (CHKPT-BEST-001 fix: explicit metadata)
          - _total_scheduler_steps (internal scheduler step-count persistence)
        """
        import random as _random
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        # RESUME-001 fix: save the parameter-name order for cross-stage Adam state migration
        param_name_order = []
        name_to_param = {}
        for n, p in self.model.named_parameters():
            name_to_param[id(p)] = n
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_name_order.append(
                    name_to_param.get(id(p), f"unknown_{len(param_name_order)}")
                )

        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "best_metric_mode": "max",  # CHKPT-BEST-001 fix: explicitly declare the optimization direction
            "patience_counter": self.patience_counter,
            "early_stop_patience": self.config.early_stop_patience,
            "optimal_threshold": self.optimal_threshold,  # CHKPT-THRESHOLD-001 fix: top-level field
            "param_name_order": param_name_order,  # RESUME-001 fix
            "_total_scheduler_steps": getattr(self, "_total_scheduler_steps", 0),
            "_warmup_steps": getattr(self, "_warmup_steps", 0),
            "config": {
                "learning_rate": self.config.learning_rate,
                "max_epochs": self.config.max_epochs,
                "batch_size": self.config.batch_size,
            },
            # M-06 fix: save all RNG states to ensure reproducibility after resume
            "rng_state": {
                "python": _random.getstate(),
                "numpy": __import__("numpy").random.get_state(),
                "torch_cpu": torch.random.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        if metrics:
            state["metrics"] = metrics

        # Contract (frozen-period logging enhancement by design): contract metadata  —  a response letter
        # can cite it directly, proving "frozen representation evaluation" is not an after-the-fact claim.
        state["contract"] = self._contract_metadata()

        torch.save(state, path)
        print(f"[Checkpoint] save: {path}")

    def _contract_metadata(self) -> dict:
        """Validatable snapshot of the encoder-frozen contract (stored in every checkpoint)."""
        import hashlib
        bb = {n: p for n, p in self.model.named_parameters()
              if n.startswith(self.config.backbone_prefix)}
        h = hashlib.sha256()
        for n in sorted(bb):
            t = bb[n].detach().cpu().contiguous()
            h.update(n.encode())
            h.update(t.view(torch.uint8).numpy().tobytes()
                     if t.dtype in (torch.bfloat16,) else t.numpy().tobytes())
        return {
            "encoder_frozen": all(not p.requires_grad for p in bb.values()),
            "backbone_n_params": int(sum(p.numel() for p in bb.values())),
            "backbone_sha256": h.hexdigest(),
            "trainable_n_params": int(sum(p.numel() for p in self.model.parameters()
                                          if p.requires_grad)),
            "optimizer_n_params": int(sum(p.numel() for g in self.optimizer.param_groups
                                          for p in g["params"])),
            "freeze_epochs": self.config.freeze_epochs,
            "max_epochs": self.config.max_epochs,
            "config_sha256": getattr(self.config, "config_sha256", ""),
            "dataset_manifest_sha256": getattr(self.config,
                                               "dataset_manifest_sha256", ""),
        }

    def load_checkpoint(self, path: str) -> Dict:
        """loadtrainingcheckpoint.

        Includes the RESUME-001 fix (parameter-name-matched Adam state migration),
        SCHED-CORRUPT-001 fix (explicit scheduler group-count validation),
        CHKPT-THRESHOLD-001 fix (top-level optimal_threshold),
        CHKPT-BEST-001 fix (explicit best_metric_mode).
        """
        state = torch.load(path, map_location=self.device, weights_only=False)
        # strict=False: skip non-model parameters such as the rotary embedding cache
        missing, unexpected = self.model.load_state_dict(state["model"], strict=False)
        if unexpected:
            print(f"  [Checkpoint] skip {len(unexpected)} extra keys "
                  f"(rotary embedding cache etc.)")
        if missing:
            print(f"  [Checkpoint] missing {len(missing)} keys: {missing[:3]}...")

        # ── RESUME-001 fix: optimizer load + parameter-name-matched migration ──
        optimizer_loaded = False
        try:
            self.optimizer.load_state_dict(state["optimizer"])
            optimizer_loaded = True
        except ValueError:
            saved_groups = len(state.get("optimizer", {}).get("param_groups", []))
            current_groups = len(self.optimizer.param_groups)
            print(f"  [Checkpoint] Optimizer param-group count mismatch "
                  f"(saved={saved_groups}, current={current_groups})")
            # RESUME-001 fix: migrate Adam m/v state by parameter name
            if "param_name_order" in state:
                migrated = migrate_optimizer_state_by_name(
                    old_opt_state=state["optimizer"],
                    old_param_names=state["param_name_order"],
                    new_optimizer=self.optimizer,
                    model=self.model,
                )
                print(f"  [Checkpoint] Adam state migrated by parameter name: "
                      f"{migrated} parameters matched successfully")
            else:
                print(f"  [Checkpoint] old checkpoint has no param_name_order, "
                      f"using a fresh optimizer (Adam state lost)")

        # ── SCHED-CORRUPT-001 fix: explicit scheduler group-count validation ──
        if optimizer_loaded and self.scheduler and state.get("scheduler"):
            saved_sched = state["scheduler"]
            # LambdaLR.state_dict() has no param_groups; use base_lrs length to judge the group count
            saved_base_lrs = saved_sched.get("base_lrs", [])
            current_base_lrs = self.scheduler.base_lrs if hasattr(self.scheduler, "base_lrs") else []
            if len(saved_base_lrs) == len(current_base_lrs) and len(saved_base_lrs) > 0:
                self.scheduler.load_state_dict(saved_sched)
            else:
                print(f"  [Checkpoint] Scheduler group count mismatch "
                      f"(saved={len(saved_base_lrs)}, current={len(current_base_lrs)}), "
                      f"using a fresh scheduler")

        # GradScaler
        if self.scaler and state.get("scaler"):
            self.scaler.load_state_dict(state["scaler"])

        # Epoch / step
        self.current_epoch = state.get("epoch", 0)
        self.global_step = state.get("global_step", 0)

        # ── CHKPT-BEST-001 fix: best_metric_mode ──
        state_best = state.get("best_metric", 0.0)
        metric_mode = state.get("best_metric_mode", None)
        if metric_mode is None:
            # Legacy compat: old checkpoints lack the metric_mode field
            # Old-version best_metric is loss (<3.0); the new version is AUPRC (0~1.0)
            if state_best > 1.0:
                state_best = 0.0
        self.best_metric = state_best
        self.best_epoch = state.get("best_epoch", 0)
        self.patience_counter = state.get("patience_counter", 0)

        # ── CHKPT-THRESHOLD-001 fix: top-level optimal_threshold field ──
        self.optimal_threshold = state.get(
            "optimal_threshold",
            state.get("metrics", {}).get("optimal_threshold", 0.5),
        )

        # Internal scheduler step counts (if saved)
        if "_total_scheduler_steps" in state:
            self._total_scheduler_steps = state["_total_scheduler_steps"]
        if "_warmup_steps" in state:
            self._warmup_steps = state["_warmup_steps"]

        # M-06 fix: restore RNG state (ensures reproducibility)
        rng_state = state.get("rng_state")
        if rng_state:
            import random as _random
            import numpy as np
            _random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            # BUG-033 fix: PyTorch 2.10+ requires ByteTensor for set_rng_state
            # BUG-035 fix: map_location=cuda moves RNG tensors to GPU, set_rng_state requires CPU
            _cpu_rng = rng_state["torch_cpu"].to(torch.uint8).cpu()
            torch.random.set_rng_state(_cpu_rng)
            if rng_state.get("torch_cuda") and torch.cuda.is_available():
                _cuda_rng = [t.to(torch.uint8).cpu() for t in rng_state["torch_cuda"]]
                torch.cuda.set_rng_state_all(_cuda_rng)
            print(f"  [Checkpoint] RNG state restored")

        print(f"[Checkpoint] load: {path} (epoch={self.current_epoch}, "
              f"global_step={self.global_step}, "
              f"best_metric={self.best_metric:.4f}, "
              f"optimal_threshold={self.optimal_threshold:.4f})")
        return state


# ═══════════════════════════════════════════════════════════════════════════════ #
#  Utility functions
# ═══════════════════════════════════════════════════════════════════════════════ #

def _batch_to_device(batch, device):
    """Move an HGTBatch to the specified device."""
    for field_name in batch.__dataclass_fields__:
        val = getattr(batch, field_name)
        if isinstance(val, torch.Tensor):
            setattr(batch, field_name, val.to(device))
    return batch


# ═══════════════════════════════════════════════════════════════════════════════ #
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════ #

def train(
    model: nn.Module,
    loss_fn: nn.Module,
    train_dataloader,
    eval_dataloader=None,
    config: TrainingConfig = None,
) -> TrainingEngine:
    """
    Training entry point.

    Parameters
    ----------
    model : nn.Module
    loss_fn : nn.Module
    train_dataloader : DataLoader
    eval_dataloader : DataLoader | None
    config : TrainingConfig

    Returns
    -------
    TrainingEngine (with training history)
    """
    if config is None:
        config = TrainingConfig()

    config.validate()
    device = config.get_device()

    print("=" * 60)
    print("DeepFluHGT  —  Training")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print(f"Epochs: {config.max_epochs}  LR: {config.learning_rate}")
    print(f"LoRA: {config.use_lora}  Mixup α: {config.mixup_alpha}")
    print(f"Freeze: {config.freeze_epochs} epochs → 4-group discriminative LR")
    print("=" * 60)

    engine = TrainingEngine(model, loss_fn, config)
    engine.setup(steps_per_epoch=len(train_dataloader))

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    for epoch in range(config.max_epochs):
        t0 = time.time()
        train_metrics = engine.train_epoch(train_dataloader)
        elapsed = time.time() - t0

        engine.history["train_loss"].append(train_metrics.get("total", 0))
        engine.history["lr"].append(train_metrics.get("lr", 0))

        print(f"\n[Epoch {epoch}] "
              f"total={train_metrics.get('total', 0):.4f}  "
              f"cls={train_metrics.get('cls', 0):.4f}  "
              f"bp={train_metrics.get('bp', 0):.4f}  "
              f"time={elapsed:.1f}s  "
              f"grad_norm={train_metrics.get('grad_norm', 0):.2f}")

        # evaluation
        if eval_dataloader and (epoch + 1) % config.eval_interval == 0:
            eval_metrics = engine.eval_epoch(eval_dataloader)
            engine.history["eval_loss"].append(eval_metrics.get("eval_loss", 0))

            print(f"  [Eval] loss={eval_metrics.get('eval_loss', 0):.4f}  "
                  f"acc={eval_metrics.get('accuracy', 0):.4f}  "
                  f"auc={eval_metrics.get('auc', 0):.4f}")

            # Best model (selected by AUPRC, must exceed min_delta)
            eval_auprc = eval_metrics.get("auprc", 0.0)
            if eval_auprc > engine.best_metric + 0.001:
                engine.best_metric = eval_auprc
                engine.best_epoch = epoch
                engine.patience_counter = 0
                if config.save_best:
                    path = os.path.join(config.checkpoint_dir, "best_model.pt")
                    engine.save_checkpoint(path, eval_metrics)
            else:
                engine.patience_counter += 1

            # early stopping
            if engine.patience_counter >= config.early_stop_patience:
                print(f"[Early Stop] no AUPRC improvement for {config.early_stop_patience} consecutive epochs "
                      f"(best={engine.best_metric:.4f})")
                break

        # Periodic save
        if config.save_last and (epoch + 1) % config.save_epochs == 0:
            path = os.path.join(config.checkpoint_dir, f"epoch_{epoch + 1}.pt")
            engine.save_checkpoint(path)

    # Save the final model
    if config.save_last:
        path = os.path.join(config.checkpoint_dir, "last_model.pt")
        engine.save_checkpoint(path)

    # merge LoRA weights (inferenceuse)
    if config.use_lora:
        merge_lora_weights(engine.model)
        path = os.path.join(config.checkpoint_dir, "merged_model.pt")
        engine.save_checkpoint(path)

    return engine


def build_training_engine(
    model: nn.Module,
    loss_fn: nn.Module,
    config: TrainingConfig = None,
) -> TrainingEngine:
    """Build the training engine (without starting training)."""
    if config is None:
        config = TrainingConfig()
    engine = TrainingEngine(model, loss_fn, config)
    engine.setup()
    return engine


def test() -> bool:
    """Training Engine self-test."""
    print("=" * 60)
    print("DeepFluHGT  —  Training Engine Self-Test")
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

    from model.contracts import HGTConfig, HGTOutput
    from model.api import build_model
    from loss.contracts import LossConfig
    from loss.api import build_loss
    from data_pipeline.contracts import HGTBatch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 1. LoRA test ──
    print("\n[1] LoRA")
    from training_engine.impl.lora import LoRALinear

    linear = nn.Linear(768, 256)
    lora = LoRALinear(linear, rank=4, alpha=8.0)
    x = torch.randn(2, 768)
    y = lora(x)
    log("LoRALinear forward shape", y.shape == (2, 256))
    log("LoRALinear has grad on A", lora.lora_A.requires_grad)
    log("LoRALinear original frozen", not lora.original.weight.requires_grad)

    # mergeweights
    lora.merge_weights()
    log("merge_weights: B is zero", (lora.lora_B == 0).all().item())

    # apply_lora_to_model
    model_config = HGTConfig(
        data_mode="paired_end",
        _smoke_global_layers=2,
        _smoke_local_layers=2,
        pretrained_nt_path="",
        pretrained_caduceus_path="",
    )
    model = build_model(model_config)
    n_lora = apply_lora_to_model(model, rank=4, alpha=8.0)
    n_lora_modules = sum(1 for m in model.modules() if isinstance(m, LoRALinear))
    log(f"apply_lora: {n_lora_modules} LoRA layers", n_lora_modules > 0,
        f"got {n_lora_modules}")

    lora_params = get_lora_params(model)
    log(f"get_lora_params: {len(lora_params)} params", len(lora_params) > 0)

    # ── 2. Mixup test ──
    print("\n[2] Mixup")
    B = 4
    batch = HGTBatch(
        input_ids=torch.randint(0, 10, (B, 303)),
        attention_mask=torch.ones(B, 303, dtype=torch.long),
        bio_features=torch.randn(B, 260),
        labels=torch.tensor([1, 0, 1, 0]),
        breakpoint_targets=torch.zeros(B, 303),
        region_targets=torch.zeros(B, 303),
        pe_case=torch.zeros(B, dtype=torch.long),
        hgt_mode=torch.tensor([0, 3, 1, 3]),
        n_events=torch.tensor([1, 0, 2, 0]),
    )
    mixed, lam = mixup_batch(batch, alpha=0.2)
    log("mixup: lam >= 0.5", lam >= 0.5, f"lam={lam:.4f}")
    log("mixup: labels mixed (float)",
        mixed.labels.is_floating_point(), f"dtype={mixed.labels.dtype}")
    log("mixup: region_targets mixed",
        mixed.region_targets.is_floating_point())
    log("mixup: input_ids unchanged (int)",
        not mixed.input_ids.is_floating_point())

    # ── 3. Optimizer test ──
    print("\n[3] Optimizer (4-group LR)")
    model2 = build_model(model_config)
    apply_lora_to_model(model2, rank=4)

    optimizer = create_optimizer(
        model2, lr=3e-5, backbone_lr_ratio=0.1, lora_lr_multiplier=2.0,
    )
    n_groups = len(optimizer.param_groups)
    log(f">=2 param groups (decay/no_decay split)", n_groups >= 2, f"got {n_groups}")

    # test frozen optimizer
    model2f = build_model(model_config)
    apply_lora_to_model(model2f, rank=4)
    # Manually freeze the backbone
    for n, p in model2f.named_parameters():
        if "backbone" in n and "lora" not in n:
            p.requires_grad = False
    frozen_opt = create_frozen_optimizer(model2f, lr=3e-5, lora_lr_multiplier=2.0)
    log("frozen optimizer: >=2 groups (decay/no_decay split)",
        len(frozen_opt.param_groups) >= 2,
        f"got {len(frozen_opt.param_groups)} groups")

    # test unfrozen optimizer (4-group)
    for n, p in model2f.named_parameters():
        if "backbone" in n:
            p.requires_grad = True
    unfrozen_opt = create_unfrozen_optimizer(model2f, lr=3e-5)
    log("unfrozen optimizer: >=4 groups (decay/no_decay split)",
        len(unfrozen_opt.param_groups) >= 4,
        f"got {len(unfrozen_opt.param_groups)} groups")

    if len(unfrozen_opt.param_groups) >= 4:
        lrs = [g["lr"] for g in unfrozen_opt.param_groups]
        names = [g.get("name", "?") for g in unfrozen_opt.param_groups]
        log("global < local < heads (decay groups)",
            lrs[0] <= lrs[2] < lrs[4],
            f"LRs: {list(zip(names, [f'{l:.2e}' for l in lrs]))}")

    # Scheduler
    scheduler = build_scheduler(optimizer, total_steps=1000, warmup_steps=60)
    log("scheduler created", scheduler is not None)

    # ── 4. TrainingEngine step test ──
    print("\n[4] TrainingEngine Step")
    model3 = build_model(model_config).to(DEVICE)
    loss_fn = build_loss(LossConfig()).to(DEVICE)

    train_config = TrainingConfig(
        use_lora=True,
        lora_rank=4,
        use_amp=(DEVICE == "cuda"),
        device=DEVICE,
        mixup_alpha=0.0,
        freeze_epochs=0,
    )

    engine = TrainingEngine(model3, loss_fn, train_config)
    engine.setup()

    # Single step
    batch_dev = HGTBatch(
        input_ids=torch.randint(0, 10, (B, 303), device=DEVICE),
        attention_mask=torch.ones(B, 303, dtype=torch.long, device=DEVICE),
        bio_features=torch.randn(B, 260, device=DEVICE),
        labels=torch.tensor([1, 0, 1, 0], device=DEVICE),
        breakpoint_targets=torch.zeros(B, 303, device=DEVICE),
        region_targets=torch.zeros(B, 303, device=DEVICE),
        pe_case=torch.zeros(B, dtype=torch.long, device=DEVICE),
        hgt_mode=torch.tensor([0, 3, 1, 3], device=DEVICE),
        n_events=torch.tensor([1, 0, 2, 0], device=DEVICE),
    )
    loss_dict = engine.step(batch_dev)
    log("step: loss_dict has total", "total" in loss_dict)
    log("step: total loss finite", isinstance(loss_dict["total"], float)
        and math.isfinite(loss_dict["total"]))

    # [R4] direction logits monitor
    log("step: direction logits monitored",
        "dir_mean" in loss_dict and "dir_max" in loss_dict,
        f"mean={loss_dict.get('dir_mean', 0):.2f}")

    # optimizer step
    gn = engine.optimizer_step()
    log("optimizer_step: global_step incremented", engine.global_step == 1)
    log("optimizer_step: returns grad_norm", gn is not None and isinstance(gn, float))

    # ── 5. Checkpoint test ──
    print("\n[5] Checkpoint")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")
        engine.save_checkpoint(ckpt_path, {"accuracy": 0.9})
        log("save_checkpoint: file exists", os.path.exists(ckpt_path))

        engine2 = TrainingEngine(
            build_model(model_config).to(DEVICE),
            build_loss(LossConfig()).to(DEVICE),
            train_config,
        )
        engine2.setup()
        state = engine2.load_checkpoint(ckpt_path)
        log("load_checkpoint: epoch restored", state["epoch"] == engine.current_epoch)
        log("load_checkpoint: metrics present", "metrics" in state)

    # ── 6. Single-epoch training test ──
    print("\n[6] Single Epoch Training")
    model4 = build_model(model_config).to(DEVICE)
    loss_fn4 = build_loss(LossConfig()).to(DEVICE)
    config4 = TrainingConfig(
        use_lora=True,
        lora_rank=4,
        use_amp=(DEVICE == "cuda"),
        device=DEVICE,
        max_epochs=1,
        mixup_alpha=0.0,
        log_interval=2,
        freeze_epochs=0,
    )
    engine4 = TrainingEngine(model4, loss_fn4, config4)
    engine4.setup()

    # Simple dataloader
    from torch.utils.data import DataLoader, TensorDataset

    N = 20
    labels = torch.tensor([1, 0] * 10, device=DEVICE)
    pos_pattern = torch.tensor([5, 6] * 5, device=DEVICE)
    neg_pattern = torch.tensor([7, 8] * 5, device=DEVICE)
    input_ids = torch.zeros(N, 303, dtype=torch.long, device=DEVICE)
    for i in range(N):
        pat = pos_pattern if labels[i] == 1 else neg_pattern
        input_ids[i] = pat.repeat((303 + 9) // 10)[:303]

    batches = []
    for start in range(0, N, B):
        end = min(start + B, N)
        batches.append(HGTBatch(
            input_ids=input_ids[start:end],
            attention_mask=torch.ones(end - start, 303, dtype=torch.long, device=DEVICE),
            bio_features=torch.randn(end - start, 260, device=DEVICE),
            labels=labels[start:end],
            breakpoint_targets=torch.zeros(end - start, 303, device=DEVICE),
            region_targets=torch.zeros(end - start, 303, device=DEVICE),
            pe_case=torch.zeros(end - start, dtype=torch.long, device=DEVICE),
            hgt_mode=torch.tensor([0, 3] * ((end - start) // 2 + 1), device=DEVICE)[:end - start],
            n_events=torch.tensor([1, 0] * ((end - start) // 2 + 1), device=DEVICE)[:end - start],
        ))

    epoch_metrics = engine4.train_epoch(batches)
    log("train_epoch: returns metrics", isinstance(epoch_metrics, dict))
    log("train_epoch: has total", "total" in epoch_metrics)
    log("train_epoch: loss finite",
        math.isfinite(epoch_metrics.get("total", float("nan"))))
    log("train_epoch: has grad_norm", "grad_norm" in epoch_metrics)

    # ── Summary ──
    total_n = len(results)
    passed = sum(results)
    failed = total_n - passed

    print("\n" + "=" * 60)
    print(f"TOTAL: {total_n}  |  PASSED: {passed}  |  FAILED: {failed}")
    print("=" * 60)

    return failed == 0
