# training_engine/impl/lora.py
"""
DeepFluHGT  —  LoRA low-rank adaptation
===================================
  - LoRALinear: frozen original weights + low-rank A/B matrices
  - apply_lora_to_model(): depth-adaptive LoRA injection
  - merge_lora_weights(): weight merge at inference
  - get_lora_params(): get the trainable LoRA parameters
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    LoRA low-rank adaptation: y = W_frozen @ x + dropout(x) @ A @ B * (alpha / rank)

    Parameters
    ----------
    original_linear : nn.Linear
        the wrapped original linear layer (weights frozen)
    rank : int
        LoRA rank
    alpha : float
        scaling factor
    dropout : float
        LoRA dropout rate
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        in_dim = original_linear.in_features
        out_dim = original_linear.out_features
        device = original_linear.weight.device

        # freeze the original weights
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

        # LoRA low-rank matrices: A [rank, in], B [out, rank]
        self.lora_A = nn.Parameter(torch.randn(rank, in_dim, device=device) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank, device=device))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def __getattr__(self, name: str):
        """Proxy access to the original Linear properties (weight, bias, in_features, etc.)."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "original":
                raise
            return getattr(self.original, name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.original(x)
        dropped = self.lora_dropout(x)
        lora_out = (dropped @ self.lora_A.T @ self.lora_B.T) * self.scale
        return base + lora_out

    def merge_weights(self) -> None:
        """Merge LoRA weights into the original linear layer (for inference)."""
        with torch.no_grad():
            self.original.weight.add_(self.lora_B @ self.lora_A * self.scale)
            self.lora_B.zero_()
            self.lora_A.zero_()


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Inject LoRA with depth adaptivity.

    Deeper layers get a larger rank: shallow r/2, middle r, deep 2r (clamped to at least 1).

    Parameters
    ----------
    model : nn.Module
    rank : int
        base LoRA rank
    alpha : float
        LoRA scaling factor
    dropout : float
        LoRA dropout
    target_modules : list[str] | None
        module-name keywords to replace (default ["q_proj", "v_proj", "dense"])

    Returns
    -------
    model (modified in place)
    """
    if target_modules is None:
        target_modules = ["linear1", "linear2", "out_proj"]

    # Collect all target Linear layers and their depths
    linear_info = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            is_target = "Linear" in target_modules or any(t in name for t in target_modules)
            if is_target:
                depth = name.count(".")
                linear_info.append((name, module, depth))

    if not linear_info:
        return model

    max_depth = max(d for _, _, d in linear_info)
    min_depth = min(d for _, _, d in linear_info)
    depth_range = max_depth - min_depth if max_depth > min_depth else 1

    for name, module, depth in linear_info:
        norm_depth = (depth - min_depth) / depth_range

        if norm_depth < 0.33:
            adaptive_rank = max(1, rank // 2)
        elif norm_depth < 0.67:
            adaptive_rank = rank
        else:
            adaptive_rank = min(rank * 2, module.in_features)

        adaptive_rank = min(adaptive_rank, module.in_features, module.out_features)
        if adaptive_rank < 1:
            continue

        lora_layer = LoRALinear(module, rank=adaptive_rank, alpha=alpha, dropout=dropout)

        *path, attr_name = name.split(".")
        parent = model
        for p in path:
            parent = getattr(parent, p)
        setattr(parent, attr_name, lora_layer)

    return model


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Recursively merge the weights of all LoRALinear in the model into the original Linear."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge_weights()
    return model


def get_lora_params(model: nn.Module) -> List[nn.Parameter]:
    """Return all trainable LoRA parameters in the model."""
    params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.extend([module.lora_A, module.lora_B])
    return params
