# training_engine/impl/optimizer.py
"""
DeepFluHGT  —  optimizer and learning-rate scheduling
======================================
  - create_optimizer(): param-group construction (frozen-phase / unfrozen-phase)
  - create_frozen_optimizer(): frozen phase includes only heads + lora (saves GPU memory)
  - create_unfrozen_optimizer(): unfrozen phase, four-group discriminative learning rates
  - build_scheduler(): cosine warmup scheduler (unified lambda, used in the frozen phase)
  - build_per_group_scheduler(): per-group independent warmup scheduler (used on unfreeze)
  - migrate_optimizer_state_by_name(): migrate Adam state by parameter name
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from training_engine.impl.lora import LoRALinear, get_lora_params


# ═══════════════════════════════════════════════════════════════════════════════ #
#  Parameter grouping utilities
# ═══════════════════════════════════════════════════════════════════════════════ #

def _collect_lora_param_ids(model: nn.Module) -> set:
    """Collect the id set of all LoRA parameters."""
    return {id(p) for p in get_lora_params(model)}


def _split_decay_groups(params: list, **group_kwargs) -> list:
    """Split a parameter list into decay and no_decay AdamW param groups.

    no_decay: 1-D parameters (bias, LayerNorm weight, scalar params, etc.)
    Keeps name; the no_decay group gets the suffix "(no_wd)"
    """
    decay, no_decay = [], []
    for p in params:
        if p.ndim < 2:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = []
    base_name = group_kwargs.pop("name", None)
    if decay:
        groups.append({"params": decay, **group_kwargs})
        if base_name:
            groups[-1]["name"] = base_name
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0, **group_kwargs})
        if base_name:
            groups[-1]["name"] = f"{base_name} (no_wd)"
    return groups


def _classify_params(
    model: nn.Module,
    backbone_prefix: str = "backbone",
    global_encoder_prefix: str = "backbone.global_encoder",
    local_encoder_prefix: str = "backbone.local_encoder",
) -> dict:
    """
    Split model parameters into four groups: global_encoder, local_encoder, heads, lora.

    Returns
    -------
    dict with keys: 'global', 'local', 'heads', 'lora'
        each value is list[param]
    """
    lora_param_ids = _collect_lora_param_ids(model)

    groups = {"global": [], "local": [], "heads": [], "lora": []}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in lora_param_ids:
            groups["lora"].append(param)
        elif global_encoder_prefix and name.startswith(global_encoder_prefix):
            groups["global"].append(param)
        elif local_encoder_prefix and name.startswith(local_encoder_prefix):
            groups["local"].append(param)
        elif backbone_prefix in name:
            # other parameters under backbone (projection etc.) go to the global group
            groups["global"].append(param)
        else:
            groups["heads"].append(param)

    return groups


# ═══════════════════════════════════════════════════════════════════════════════ #
#  Optimizer construction
# ═══════════════════════════════════════════════════════════════════════════════ #

def create_optimizer(
    model: nn.Module,
    lr: float = 3e-5,
    backbone_lr_ratio: float = 0.1,
    lora_lr_multiplier: float = 2.0,
    weight_decay: float = 0.05,
    backbone_prefix: str = "backbone",
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_eps: float = 1e-8,
) -> torch.optim.AdamW:
    """
    Create a three-group discriminative learning-rate AdamW optimizer (legacy-interface compatible).

    Groups:
      1. backbone: lr × backbone_lr_ratio
      2. heads / loss (task): lr
      3. lora (LoRA): lr × lora_lr_multiplier
    """
    backbone_params = []
    head_params = []
    lora_params = []
    lora_param_ids = _collect_lora_param_ids(model)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in lora_param_ids:
            lora_params.append(param)
        elif backbone_prefix in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = []

    if backbone_params:
        param_groups.append({
            "params": backbone_params,
            "lr": lr * backbone_lr_ratio,
            "name": "backbone",
        })
    if head_params:
        param_groups.append({
            "params": head_params,
            "lr": lr,
            "name": "heads",
        })
    if lora_params:
        param_groups.append({
            "params": lora_params,
            "lr": lr * lora_lr_multiplier,
            "name": "lora",
        })

    if not param_groups:
        param_groups = [{"params": list(model.parameters()), "lr": lr, "name": "all"}]

    # Split bias/LayerNorm into a no_decay group
    split_groups = []
    for g in param_groups:
        params = g.pop("params")
        split_groups.extend(_split_decay_groups(params, **g))

    return torch.optim.AdamW(
        split_groups,
        weight_decay=weight_decay,  # global default; no_decay groups explicitly override to 0
        betas=(adam_beta1, adam_beta2),
        eps=adam_eps,
    )


def create_frozen_optimizer(
    model: nn.Module,
    lr: float = 3e-5,
    lora_lr_multiplier: float = 2.0,
    weight_decay: float = 0.05,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_eps: float = 1e-8,
) -> torch.optim.AdamW:
    """
    Frozen-phase optimizer: includes only heads + lora parameters.
    Backbone parameters are not added to the optimizer, avoiding useless AdamW-state GPU memory.
    """
    groups = _classify_params(model)

    param_groups = []
    if groups["heads"]:
        param_groups.append({
            "params": groups["heads"],
            "lr": lr,
            "name": "heads",
        })
    if groups["lora"]:
        param_groups.append({
            "params": groups["lora"],
            "lr": lr * lora_lr_multiplier,
            "name": "lora",
        })

    if not param_groups:
        param_groups = [{"params": list(model.parameters()), "lr": lr, "name": "all"}]

    # Split bias/LayerNorm into a no_decay group
    split_groups = []
    for g in param_groups:
        params = g.pop("params")
        split_groups.extend(_split_decay_groups(params, **g))

    n_params = sum(len(g["params"]) for g in split_groups)
    total_params = sum(p.numel() for g in split_groups for p in g["params"])
    print(f"[Frozen Optimizer] {n_params} tensors, {total_params:,} params "
          f"(heads + lora only, backbone excluded)")

    return torch.optim.AdamW(
        split_groups,
        weight_decay=weight_decay,
        betas=(adam_beta1, adam_beta2),
        eps=adam_eps,
    )


def create_unfrozen_optimizer(
    model: nn.Module,
    lr: float = 3e-5,
    global_lr_ratio: float = 0.05,
    local_lr_ratio: float = 0.1,
    lora_lr_multiplier: float = 2.0,
    weight_decay: float = 0.05,
    backbone_prefix: str = "backbone",
    global_encoder_prefix: str = "backbone.global_encoder",
    local_encoder_prefix: str = "backbone.local_encoder",
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_eps: float = 1e-8,
) -> torch.optim.AdamW:
    """
    Unfrozen-phase optimizer: four-group discriminative learning rates.

      global_encoder: lr × global_lr_ratio  (default 0.05×, very conservative)
      local_encoder:  lr × local_lr_ratio   (default 0.1×,  moderate)
      heads:          lr × 1.0              (normal)
      lora:           lr × lora_lr_multiplier (default 2.0×, fast adaptation)
    """
    groups = _classify_params(
        model, backbone_prefix, global_encoder_prefix, local_encoder_prefix,
    )

    param_groups = []

    if groups["global"]:
        param_groups.append({
            "params": groups["global"],
            "lr": lr * global_lr_ratio,
            "name": "global_encoder",
        })
    if groups["local"]:
        param_groups.append({
            "params": groups["local"],
            "lr": lr * local_lr_ratio,
            "name": "local_encoder",
        })
    if groups["heads"]:
        param_groups.append({
            "params": groups["heads"],
            "lr": lr,
            "name": "heads",
        })
    if groups["lora"]:
        param_groups.append({
            "params": groups["lora"],
            "lr": lr * lora_lr_multiplier,
            "name": "lora",
        })

    if not param_groups:
        param_groups = [{"params": list(model.parameters()), "lr": lr, "name": "all"}]

    # Split bias/LayerNorm into a no_decay group
    split_groups = []
    for g in param_groups:
        params = g.pop("params")
        split_groups.extend(_split_decay_groups(params, **g))

    print(f"[Unfrozen Optimizer] 4-group discriminative LR:")
    for g in split_groups:
        n = sum(p.numel() for p in g["params"])
        wd = g.get("weight_decay", weight_decay)
        print(f"  lr={g['lr']:.2e}, params={n:,}, wd={wd}")

    return torch.optim.AdamW(
        split_groups,
        weight_decay=weight_decay,
        betas=(adam_beta1, adam_beta2),
        eps=adam_eps,
    )


# ═══════════════════════════════════════════════════════════════════════════════ #
#  Learning-rate schedulers
# ═══════════════════════════════════════════════════════════════════════════════ #

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    schedule_type: str = "cosine",
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Build a learning-rate scheduler.

    Parameters
    ----------
    optimizer : Optimizer
    total_steps : int
    warmup_steps : int
    schedule_type : str
        "cosine" | "linear" | "constant"

    Returns
    -------
    LambdaLR
    """
    if schedule_type == "cosine":

        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)  # +1 avoids LR=0 on the first step
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    elif schedule_type == "linear":

        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)  # +1 avoids LR=0 on the first step
            return max(0.0, 1.0 - (step - warmup_steps) / max(1, total_steps - warmup_steps))

    else:  # constant

        def lr_lambda(step):
            return min(1.0, (step + 1) / max(1, warmup_steps))  # +1 avoids LR=0 on the first step

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def build_per_group_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_configs: List[Dict[str, Any]],
    schedule_type: str = "cosine",
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Build a learning-rate scheduler with per-group independent warmup.

    Resolves LR-DEAD-001 (backbone LR permanently zero) and LR-DROP-001 (heads LR 300x crash):
    different parameter groups get independent lambda functions implementing differentiated warmup strategies.
    The backbone group ramps linearly from 0 to initial_lr; heads/LoRA groups continue directly without warmup.

    Parameters
    ----------
    optimizer : Optimizer
        an already-created optimizer (param_groups should already have correct initial_lr)
    total_steps : int
        total number of steps
    warmup_configs : List[Dict]
        one config dict per parameter group; length must equal len(optimizer.param_groups).
        Each dict:
            warmup_steps (int): warmup steps for that group
            warmup_from_zero (bool): True=linear ramp from 0; False=directly 1.0, skip warmup
    schedule_type : str
        "cosine" | "linear" | "constant"

    Returns
    -------
    LambdaLR
        a scheduler with per-group lambdas

    Raises
    ------
    ValueError
        warmup_configs length does not match optimizer param_groups
    """
    if len(warmup_configs) != len(optimizer.param_groups):
        raise ValueError(
            f"warmup_configs length ({len(warmup_configs)}) != "
            f"optimizer param_groups length ({len(optimizer.param_groups)})"
        )

    def make_lambda(warmup_steps: int, from_zero: bool):
        """Build the lambda function for a single parameter group."""
        if schedule_type == "cosine":
            def lr_lambda(step: int) -> float:
                if from_zero and step < warmup_steps:
                    # Backbone: linear ramp from 0 to 1 (× initial_lr = target LR)
                    return (step + 1) / max(1, warmup_steps)
                if not from_zero:
                    # Heads/LoRA: skip warmup, cosine decay straight from initial_lr
                    pass
                # Cosine decay phase (or direct cosine when from_zero=False)
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                progress = max(0.0, min(1.0, progress))
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            return lr_lambda

        elif schedule_type == "linear":
            def lr_lambda(step: int) -> float:
                if from_zero and step < warmup_steps:
                    return (step + 1) / max(1, warmup_steps)
                return max(0.0, 1.0 - (step - warmup_steps) / max(1, total_steps - warmup_steps))
            return lr_lambda

        else:  # constant
            def lr_lambda(step: int) -> float:
                if from_zero:
                    return min(1.0, (step + 1) / max(1, warmup_steps))
                return 1.0
            return lr_lambda

    lambdas = [
        make_lambda(c.get("warmup_steps", 0), c.get("warmup_from_zero", False))
        for c in warmup_configs
    ]
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)


def migrate_optimizer_state_by_name(
    old_opt_state: Dict[str, Any],
    old_param_names: List[str],
    new_optimizer: torch.optim.Optimizer,
    model: nn.Module,
) -> int:
    """
    Migrate Adam state (exp_avg, exp_avg_sq, step) to the new optimizer by parameter name.

    Resolves RESUME-001: on frozen→unfrozen resume the optimizer param-group count changes,
    and load_state_dict raises ValueError, losing all Adam state.

    Strategy:
    1. Build a name → adam_state mapping from old_opt_state + old_param_names
    2. Iterate the new optimizer parameters and migrate by name match
    3. Unmatched parameters (newly unfrozen backbone) keep default initialization

    Parameters
    ----------
    old_opt_state : Dict[str, Any]
        the old optimizer state_dict()
    old_param_names : List[str]
        names of parameters in the old optimizer, ordered by param_groups[*].params
        (provided by the param_name_order field in save_checkpoint)
    new_optimizer : torch.optim.Optimizer
        the new optimizer (already created, with correct param groups)
    model : nn.Module
        the current model (used to build the name → param mapping)

    Returns
    -------
    int
        number of successfully migrated parameters
    """
    # 1. Rebuild from the old state: param_name → adam_state
    name_to_adam_state: Dict[str, Dict[str, Any]] = {}
    flat_idx = 0
    for group in old_opt_state.get("param_groups", []):
        for param_idx in group.get("params", []):
            if param_idx in old_opt_state.get("state", {}) and flat_idx < len(old_param_names):
                name_to_adam_state[old_param_names[flat_idx]] = old_opt_state["state"][param_idx]
            flat_idx += 1

    if not name_to_adam_state:
        return 0

    # 2. Build the new model mapping: param → name
    param_to_name: Dict[int, str] = {id(p): n for n, p in model.named_parameters()}

    # 3. Iterate the new optimizer parameters and migrate by name match
    migrated = 0
    for group in new_optimizer.param_groups:
        for p in group["params"]:
            pid = id(p)
            name = param_to_name.get(pid)
            if name and name in name_to_adam_state:
                old_adam = name_to_adam_state[name]
                new_optimizer.state[p] = {
                    "step": old_adam.get("step", torch.tensor(0.0, device=p.device)),
                    "exp_avg": old_adam["exp_avg"].to(device=p.device),
                    "exp_avg_sq": old_adam["exp_avg_sq"].to(device=p.device),
                }
                migrated += 1

    return migrated
