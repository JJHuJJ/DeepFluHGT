# model/impl/fusion.py
"""
DeepFluHGT  —  core disentangling and multi-dimensional information interaction modules
==================================================
Implements the model's "information processing hub":
  1. FeatureSplitBlock          —  scalar-gated donor/recipient disentangling
  2. RoleGatedCrossAttention    —  role-aware four-way cross-attention
  3. SequenceFPN                —  multi-dilation 1D feature pyramid
  4. BatchGraphInteraction      —  in-batch dynamic graph neural network interaction

Ported directly from v5.0, no token-size dependency.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════ #
#  FeatureSplitBlock  —  scalar-gated donor/recipient disentangling
# ═══════════════════════════════════════════════════════════════════════════════ #

class FeatureSplitBlock(nn.Module):
    """
    Scalar-gated feature splitting module.

    Decomposes the token sequence representation x into donor and recipient streams.
    Core constraint: donor + recipient = x (information conservation).

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension D
    num_heads : int
        Number of self-attention heads
    dropout : float
        Dropout rate
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # ── Gating MLP: x → scalar gate ──
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        # ── Donor stream ──
        self.donor_self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.donor_norm = nn.LayerNorm(hidden_dim)

        # ── Recipient stream ──
        self.recipient_self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.recipient_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Scalar-gated split + independent stream processing.

        Parameters
        ----------
        x : [B, L, D]
        attention_mask : [B, L], optional

        Returns
        -------
        dict: donor_hidden, recipient_hidden, donor_pooled, recipient_pooled, gates
        """
        gates = torch.sigmoid(self.gate_mlp(x))  # [B, L, 1]

        donor = gates * x
        recipient = (1.0 - gates) * x

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)

        d_attn_out, _ = self.donor_self_attn(
            donor, donor, donor,
            key_padding_mask=key_padding_mask,
        )
        donor_hidden = self.donor_norm(donor + d_attn_out)

        r_attn_out, _ = self.recipient_self_attn(
            recipient, recipient, recipient,
            key_padding_mask=key_padding_mask,
        )
        recipient_hidden = self.recipient_norm(recipient + r_attn_out)

        donor_pooled = F.normalize(donor_hidden[:, 0, :], dim=-1)
        recipient_pooled = F.normalize(recipient_hidden[:, 0, :], dim=-1)

        return {
            "donor_hidden": donor_hidden,
            "recipient_hidden": recipient_hidden,
            "donor_pooled": donor_pooled,
            "recipient_pooled": recipient_pooled,
            "gates": gates.squeeze(-1),  # [B, L]
        }


# DC-02 fix: SquaredCosineDisentangleLoss already has its canonical definition in loss/api.py;
# the duplicate copy here has been removed.

# ═══════════════════════════════════════════════════════════════════════════════ #
#  RoleGatedCrossAttention  —  role-aware four-way cross-attention
# ═══════════════════════════════════════════════════════════════════════════════ #

class RoleGatedCrossAttention(nn.Module):
    """
    Role-aware cross-attention fusion module.

    Four-way cross-attention:
      local_a → donor, local_a → recipient,
      local_b → donor, local_b → recipient

    Fuses using the sigmoid of the role logit as the gate:
      fused = σ(role) × donor_view + (1-σ(role)) × recipient_view

    Parameters
    ----------
    hidden_dim : int
    num_heads : int
    dropout : float
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # ── Four-way cross-attention ──
        self.local_a_to_donor = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.local_a_to_recipient = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.local_b_to_donor = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.local_b_to_recipient = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )

        self.norm_a = nn.LayerNorm(hidden_dim)
        self.norm_b = nn.LayerNorm(hidden_dim)

        # ── PairInteraction ──
        self.pair_attn_a = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.pair_attn_b = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.pair_norm_a = nn.LayerNorm(hidden_dim)
        self.pair_norm_b = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        local_hidden: torch.Tensor,
        donor_hidden: torch.Tensor,
        recipient_hidden: torch.Tensor,
        role_logit: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        local_hidden_b: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Four-way cross-attention fusion.

        Parameters
        ----------
        local_hidden : [B, L_local, D]
        donor_hidden : [B, L_global, D]
        recipient_hidden : [B, L_global, D]
        role_logit : [B, 1]
        attention_mask : [B, L_global], optional
        local_hidden_b : [B, L_local, D], optional (PE R2)

        Returns
        -------
        dict: fused_a, fused_b, donor_view_a, recipient_view_a
        """
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)

        role_weight = torch.sigmoid(role_logit)  # [B, 1]

        # ── local_a four-way ──
        donor_view_a, _ = self.local_a_to_donor(
            local_hidden, donor_hidden, donor_hidden,
            key_padding_mask=key_padding_mask,
        )
        recipient_view_a, _ = self.local_a_to_recipient(
            local_hidden, recipient_hidden, recipient_hidden,
            key_padding_mask=key_padding_mask,
        )
        fused_a = role_weight.unsqueeze(1) * donor_view_a + \
                  (1.0 - role_weight.unsqueeze(1)) * recipient_view_a
        fused_a = self.norm_a(fused_a + local_hidden)

        result = {
            "fused_a": fused_a,
            "fused_b": None,
            "donor_view_a": donor_view_a,
            "recipient_view_a": recipient_view_a,
        }

        # ── local_b four-way (PE R2) ──
        if local_hidden_b is not None:
            donor_view_b, _ = self.local_b_to_donor(
                local_hidden_b, donor_hidden, donor_hidden,
                key_padding_mask=key_padding_mask,
            )
            recipient_view_b, _ = self.local_b_to_recipient(
                local_hidden_b, recipient_hidden, recipient_hidden,
                key_padding_mask=key_padding_mask,
            )
            fused_b = role_weight.unsqueeze(1) * donor_view_b + \
                      (1.0 - role_weight.unsqueeze(1)) * recipient_view_b
            fused_b = self.norm_b(fused_b + local_hidden_b)

            # ── PairInteraction ──
            pair_a_out, _ = self.pair_attn_a(fused_a, fused_b, fused_b)
            pair_b_out, _ = self.pair_attn_b(fused_b, fused_a, fused_a)
            result["fused_a"] = self.pair_norm_a(fused_a + pair_a_out)
            result["fused_b"] = self.pair_norm_b(fused_b + pair_b_out)

        return result


# ═══════════════════════════════════════════════════════════════════════════════ #
#  SequenceFPN  —  multi-dilation 1D feature pyramid
# ═══════════════════════════════════════════════════════════════════════════════ #

class SequenceFPN(nn.Module):
    """
    Multi-scale sequence feature pyramid network.

    Three parallel dilated 1D convolution branches (dilation = [1, 2, 4]),
    concatenated and fused by a 1×1 convolution, with a residual connection.

    Parameters
    ----------
    hidden_dim : int
    dilations : tuple of int
    kernel_size : int
    dropout : float
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        dilations: Tuple[int, ...] = (1, 2, 4),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dilations = dilations
        n_branches = len(dilations)

        branch_dim = hidden_dim // n_branches
        self.branch_dim = branch_dim

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    hidden_dim, branch_dim,
                    kernel_size=kernel_size,
                    padding=self._compute_padding(kernel_size, d),
                    dilation=d,
                ),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for d in dilations
        ])

        total_branch_dim = branch_dim * n_branches
        self.fuse_conv = nn.Conv1d(total_branch_dim, hidden_dim, kernel_size=1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _compute_padding(kernel_size: int, dilation: int) -> int:
        return (kernel_size - 1) * dilation // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, L, D]

        Returns
        -------
        [B, L, D] hidden states after multi-scale fusion (including residual)
        """
        residual = x
        x_conv = x.transpose(1, 2)  # [B, D, L]

        branch_outputs = [branch(x_conv) for branch in self.branches]
        concat = torch.cat(branch_outputs, dim=1)
        fused = self.fuse_conv(concat)
        fused = fused.transpose(1, 2)  # [B, L, D]

        fused = self.norm(self.dropout(fused) + residual)
        return fused


# ═══════════════════════════════════════════════════════════════════════════════ #
#  BatchGraphInteraction  —  in-batch graph neural network interaction
# ═══════════════════════════════════════════════════════════════════════════════ #

class BatchGraphInteraction(nn.Module):
    """
    In-batch dynamic graph neural network interaction module.

    1. Cosine similarity → Top-k adjacency matrix
    2. GraphSAGE message passing
    3. Residual + FFN + LayerNorm

    Parameters
    ----------
    hidden_dim : int
    top_k : int
    dropout : float
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        top_k: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.top_k = top_k

        self.msg_linear = nn.Linear(hidden_dim, hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pooled : [B, D]

        Returns
        -------
        [B, D] graph-enhanced sample representations
        """
        B = pooled.size(0)
        if B <= 1:
            # When B=1 skip graph interaction, but still apply FFN+LayerNorm
            # to keep representations consistent
            return self.norm(pooled + self.ffn(pooled))

        # NEW-01 fix: guard against zero vectors to prevent NaN
        norms = pooled.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normed = pooled / norms
        sim_matrix = torch.mm(normed, normed.t())

        k = min(self.top_k, B - 1)
        if k <= 0:
            return pooled

        sim_matrix.fill_diagonal_(float("-inf"))
        topk_vals, topk_indices = sim_matrix.topk(k, dim=-1)

        # Numerically safe softmax: replace invalid positions with a tiny value → exp(tiny)≈0
        # When all are -inf, all_invalid forces zero contribution
        mask = torch.isfinite(topk_vals)
        all_invalid = ~mask.any(dim=-1, keepdim=True)
        safe_vals = torch.where(mask, topk_vals, torch.finfo(topk_vals.dtype).min)
        attn_weights = F.softmax(safe_vals, dim=-1)
        attn_weights = torch.where(all_invalid, torch.zeros_like(attn_weights), attn_weights)

        msg_features = self.msg_linear(pooled)
        neighbor_features = msg_features[topk_indices]
        aggregated = torch.sum(
            attn_weights.unsqueeze(-1) * neighbor_features, dim=1,
        )

        output = pooled + aggregated
        output = output + self.ffn(output)
        output = self.norm(output)

        return output
