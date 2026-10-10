# model/contracts.py
"""
DeepFluHGT  —  Model configuration and output data structures
============================================
Defines the model's input configuration (HGTConfig) and output structure (HGTOutput).
Interfaces with the data_pipeline module's HGTBatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch


@dataclass
class HGTConfig:
    """
    DeepFluHGT global configuration.

    All hyperparameters are collected here; model components receive them
    via constructor parameters.
    """

    # ── data ──
    data_mode: str = "paired_end"            # "paired_end" | "contig"
    read_length: int = 150                   # PE read length (bp)
    max_contig_length: int = 10000           # Maximum contig length (bp)
    vocab_size: int = 10                     # Single-nucleotide canonical vocab

    # ── Architecture ──
    hidden_dim: int = 768                    # Global encoder hidden dimension
    local_hidden_dim: int = 256              # Local encoder hidden dimension
    num_attention_heads: int = 12            # Number of attention heads
    num_fpn_dilations: Tuple[int, ...] = (1, 2, 4)  # FPN dilation rates
    dropout: float = 0.15
    graph_top_k: int = 5                     # Top-k neighbors for batch graph interaction
    early_exit_threshold: float = 0.95       # Early-exit confidence threshold

    # ── Bio features ──
    bio_feature_dim: int = 260
    bio_encoder_hidden: int = 128

    # ── Refiner ──
    refiner_window: int = 50                 # Half-width of the breakpoint refinement window
    # BUG-025 fix: breakpoint_sigma has been moved to LossConfig; the model does not use this field
    # breakpoint_sigma: float = 5.0

    # ── Evidential ──
    use_evidential: bool = False             # Whether to use the Dirichlet evidential head
    evidential_hidden_dim: int = 256

    # ── pretrainedweightspath ──
    pretrained_nt_path: str = ""
    pretrained_caduceus_path: str = ""

    # ── v7: Evo2 offline embedding (third foundation-model modality) ──
    use_evo: bool = True                  # Whether to enable Evo2 offline embedding
    pretrained_evo_dim: int = 4096        # evo2_7b hidden_size
    evo_hidden: int = 512                 # EvoEncoder MLP intermediate dimension

    # ── Hardware ──
    use_flash_attention: bool = True
    # BUG-019 fix: use_amp has been moved to TrainingConfig; the model does not use this field
    # use_amp: bool = True
    # BUG-017 fix: seed is controlled globally by DeepFluHGT.py; the model does not use this field
    # seed: int = 42

    # ── Smoke test use (fewer layers for speed) ──
    _smoke_global_layers: int = 0            # 0 = makeusedefault (12)
    _smoke_local_layers: int = 0             # 0 = makeusedefault (16)

    # ── Local encoder switch ──
    use_local_encoder: bool = True

    # BUG-016 fix: local_window had no code reference, already removed
    # local_window: int = 320

    def validate(self):
        """Validate configuration legality."""
        assert self.data_mode in ("paired_end", "contig"), \
            f"data_mode must be 'paired_end' or 'contig', got '{self.data_mode}'"
        assert self.hidden_dim > 0
        assert self.vocab_size == 10, \
            f"v6.0 uses single-nucleotide tokenization, vocab_size must be 10"
        assert self.num_attention_heads > 0
        assert 0.0 <= self.dropout < 1.0
        assert 0.0 < self.early_exit_threshold <= 1.0
        assert self.read_length > 0


@dataclass
class HGTOutput:
    """
    DeepFluHGT modeloutput.

    Tensor dimension notation
    ----------------
    B : batch size
    L : sequence token length
    D : hidden_dim (default 768)
    W : 2 * refiner_window (default 100)
    """

    # ── Core outputs (always present) ──
    cls_logits: torch.Tensor = field(
        default_factory=lambda: torch.empty(0))
        # [B, 1]  HGT classification logit
    bp_heatmaps: torch.Tensor = field(
        default_factory=lambda: torch.empty(0))
        # [B, L]  Per-base breakpoint heatmap (sigmoid)
    role_logits: torch.Tensor = field(
        default_factory=lambda: torch.empty(0))
        # [B, 1]  Donor/recipient role affinity

    # ── Optional outputs ──
    direction_logits: Optional[torch.Tensor] = None
        # [B, 1]  Transfer direction logit
    bp_refined: Optional[torch.Tensor] = None
        # [B, 2W]  Refined breakpoint distribution (softmax)
    rc_cls_logits: Optional[torch.Tensor] = None
        # [B, 1]  R2 consistency classification logit in PE mode

    # ── Early-exit outputs ──
    early_exit_fusion: Optional[torch.Tensor] = None
        # [B, 1]  Fusion-layer early-exit logit
    early_exit_graph: Optional[torch.Tensor] = None
        # [B, 1]  Graph-interaction-layer early-exit logit

    # ── Interpretability outputs ──
    split_gates: Optional[torch.Tensor] = None
        # [B, L]  Donor/recipient gating values
    donor_pool: Optional[torch.Tensor] = None
        # [B, D]  Donor pooled representation
    recipient_pool: Optional[torch.Tensor] = None
        # [B, D]  Recipient pooled representation
    hidden_states: Optional[torch.Tensor] = None
        # [B, L, D]  FPN output hidden states (for visualization)
    exited_at: Optional[int] = None
        # Early-exit layer index (None = full forward pass)
