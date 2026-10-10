# training_engine/contracts.py
"""
DeepFluHGT  —  Training Engine configuration
========================================
Centrally manages all training-engine hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Training engine configuration."""

    # ── Optimizer ──
    learning_rate: float = 3e-5
    backbone_lr_ratio: float = 0.1
    lora_lr_multiplier: float = 2.0
    weight_decay: float = 0.05
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8

    # ── Learning-rate schedule ──
    warmup_ratio: float = 0.06
    schedule_type: str = "cosine"  # "cosine" | "linear" | "constant"

    # ── Training control ──
    batch_size: int = 16
    max_epochs: int = 30
    max_steps: int = -1           # -1 = follow epochs
    gradient_clip_norm: float = 5.0
    accumulation_steps: int = 1
    early_stop_patience: int = 5

    # ── Backbone frozen ──
    freeze_epochs: int = 0        # freeze backbone for the first N epochs
    backbone_prefix: str = "backbone"
    global_encoder_prefix: str = "backbone.global_encoder"
    local_encoder_prefix: str = "backbone.local_encoder"
    global_lr_ratio: float = 0.1    # global encoder LR = base_lr * 0.1
    local_lr_ratio: float = 0.2      # local encoder LR = base_lr * 0.2

    # ── Mixup ──
    mixup_alpha: float = 0.2      # 0 = disable
    mixup_warmup_epochs: int = 2  # no mixup for the first N epochs

    # ── LoRA ──
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_target_modules: Optional[list] = None  # None = default

    # ── FGM adversarial training ──
    use_fgm: bool = False
    fgm_epsilon: float = 1.0
    fgm_start_epoch: int = 3

    # ── AMP ──
    use_amp: bool = True
    amp_dtype: str = "float16"   # "float16" | "bfloat16"; bf16 range = fp32, eliminates forward overflow

    # ── Checkpoint ──
    checkpoint_dir: str = "./checkpoints"
    save_best: bool = True
    save_last: bool = True
    save_epochs: int = 5          # save every N epochs

    # ── Contract provenance hashes (logging level; stored in every checkpoint) ──
    config_sha256: str = ""            # sha256 of the training YAML
    dataset_manifest_sha256: str = ""  # joint sha256 of train/val/test breakpoints.bed

    # ── log ──
    log_interval: int = 50        # print every N steps
    eval_interval: int = 1        # evaluate every N epochs

    # ── Device ──
    device: str = "auto"          # "auto" | "cuda" | "cpu"

    # ── seed ──
    # BUG-018 fix: the seed is controlled globally by DeepFluHGT.py via seed_all(),
    # TrainingConfig no longer defines this field.
    # seed: int = 42

    def get_device(self) -> str:
        """Parse the device."""
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def validate(self):
        """Validate configuration sanity."""
        assert self.learning_rate > 0
        assert 0 < self.backbone_lr_ratio <= 1.0
        assert self.lora_lr_multiplier > 0
        assert self.weight_decay >= 0
        assert self.gradient_clip_norm > 0
        assert self.accumulation_steps >= 1
        assert self.max_epochs > 0 or self.max_steps > 0
        assert self.mixup_alpha >= 0
        assert self.schedule_type in ("cosine", "linear", "constant")
