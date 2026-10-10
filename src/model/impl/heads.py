# model/impl/heads.py
"""
DeepFluHGT  —  multi-task prediction heads and base-level refinement module
===============================================
  1. EvidentialHead     —  Dirichlet evidential classification head
  2. DirectionHead      —  transfer direction prediction head
  3. RoleHead           —  donor/recipient role recognition head
  4. BreakpointRefiner  —  base-level breakpoint refiner

Ported directly from v5.0, no token-size dependency.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════ #
#  EvidentialHead  —  Dirichlet evidential classification head
# ═══════════════════════════════════════════════════════════════════════════════ #

class EvidentialHead(nn.Module):
    """
    Evidential classification head based on the Dirichlet distribution.

    Degrades to a simple linear classification head when use_evidential=False.

    Parameters
    ----------
    hidden_dim : int
    evidential_hidden_dim : int
    num_classes : int
    dropout : float
    use_evidential : bool
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        evidential_hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.1,
        use_evidential: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_evidential = use_evidential

        if use_evidential:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, evidential_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(evidential_hidden_dim, num_classes),
            )
        else:
            self.simple_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        x : [B, D]

        Returns
        -------
        dict: cls_logit, evidence, alpha, belief, uncertainty
        """
        if not self.use_evidential:
            logit = self.simple_head(x)
            return {
                "cls_logit": logit,
                "evidence": None,
                "alpha": None,
                "belief": None,
                "uncertainty": None,
            }

        evidence = F.softplus(self.mlp(x))  # [B, K]
        alpha = evidence + 1.0
        S = alpha.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        belief = evidence / S
        uncertainty = self.num_classes / S.squeeze(-1)

        cls_logit = (evidence[:, 1:2] - evidence[:, 0:1])

        return {
            "cls_logit": cls_logit,
            "evidence": evidence,
            "alpha": alpha,
            "belief": belief,
            "uncertainty": uncertainty,
        }


# ═══════════════════════════════════════════════════════════════════════════════ #
#  DirectionHead  —  transfer direction prediction head
# ═══════════════════════════════════════════════════════════════════════════════ #

class DirectionHead(nn.Module):
    """
    Predicts gene transfer direction using the asymmetry of donor/recipient pooled features.

    v6.0 fix: added LayerNorm + Dropout to prevent logit explosion.

    Parameters
    ----------
    hidden_dim : int
    dropout : float
    """

    def __init__(self, hidden_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim * 2, 1)

    def forward(
        self,
        donor_pooled: torch.Tensor,
        recipient_pooled: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        donor_pooled : [B, D]
        recipient_pooled : [B, D]

        Returns
        -------
        direction_logits : [B, 1]
        """
        feat = torch.cat([
            donor_pooled - recipient_pooled,
            donor_pooled * recipient_pooled,
        ], dim=-1)
        feat = self.norm(feat)
        feat = self.drop(feat)
        return self.head(feat)


# ═══════════════════════════════════════════════════════════════════════════════ #
#  RoleHead  —  donor/recipient role recognition head
# ═══════════════════════════════════════════════════════════════════════════════ #

class RoleHead(nn.Module):
    """
    Role recognition head: outputs a donor-affinity logit.

    Two stages:
      1. forward_pre_graph: before graph interaction, provides the initial role_logit for cross-attention
      2. forward_post_graph: after graph interaction, refined prediction combining graph + bio features

    Parameters
    ----------
    hidden_dim : int
    """

    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.pre_graph_proj = nn.Linear(hidden_dim, 1)
        self.post_graph_norm = nn.LayerNorm(hidden_dim * 2)
        self.post_graph_head = nn.Linear(hidden_dim * 2, 1)

    def forward_pre_graph(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pooled : [B, D]

        Returns
        -------
        role_logit : [B, 1]
        """
        return self.pre_graph_proj(pooled)

    def forward_post_graph(
        self,
        graph_pooled: torch.Tensor,
        bio_features_encoded: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        graph_pooled : [B, D]
        bio_features_encoded : [B, D]

        Returns
        -------
        role_logits : [B, 1]
        """
        combined = torch.cat([graph_pooled, bio_features_encoded], dim=-1)
        combined = self.post_graph_norm(combined)
        return self.post_graph_head(combined)


# ═══════════════════════════════════════════════════════════════════════════════ #
#  BreakpointRefiner  —  base-level breakpoint refiner
# ═══════════════════════════════════════════════════════════════════════════════ #

class BreakpointRefiner(nn.Module):
    """
    Two-stage breakpoint localizer.

    Stage 1: generate a coarse token-level heatmap from the FPN output
    Stage 2: extract a hidden-state window of ±refiner_window around the peak value, refine with a CNN

    Parameters
    ----------
    hidden_dim : int
    refiner_window : int
    max_peaks : int
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        refiner_window: int = 50,
        max_peaks: int = 3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.refiner_window = refiner_window
        self.max_peaks = max_peaks

        # Stage 1: coarse heatmap
        self.coarse_proj = nn.Linear(hidden_dim, 1)

        # Stage 2: stacked 1D CNN
        d2 = hidden_dim // 2
        d4 = hidden_dim // 4
        d8 = hidden_dim // 8
        self.conv_stack = nn.Sequential(
            nn.Conv1d(hidden_dim, d2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d2, d4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d4, d8, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.fine_mlp = nn.Linear(d8, 1)

    def forward(
        self,
        fpn_output: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        n_events: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Two-stage breakpoint localization.

        Parameters
        ----------
        fpn_output : [B, L, D]
        attention_mask : [B, L], optional
        n_events : [B], optional

        Returns
        -------
        dict: bp_heatmaps, bp_refined, all_refined, peak_positions
        """
        B, L, D = fpn_output.shape
        W = 2 * self.refiner_window

        # Stage 1
        coarse_logits = self.coarse_proj(fpn_output).squeeze(-1)

        if attention_mask is not None:
            coarse_logits = coarse_logits.masked_fill(
                attention_mask == 0, float("-inf"),
            )

        bp_heatmaps = torch.sigmoid(coarse_logits)

        # Stage 2
        if n_events is not None:
            effective_k = n_events.clamp(min=1).max().item()
        else:
            effective_k = self.max_peaks
        effective_k = min(int(effective_k), self.max_peaks, L)

        if effective_k <= 0:
            return {
                "bp_heatmaps": bp_heatmaps,
                "bp_refined": torch.zeros(B, W, device=fpn_output.device),
                "all_refined": [],
                "peak_positions": torch.zeros(B, 0, dtype=torch.long, device=fpn_output.device),
            }

        topk_vals, topk_indices = coarse_logits.topk(effective_k, dim=-1)

        all_refined = []
        best_confidences = torch.full((B,), float("-inf"), device=fpn_output.device)
        best_refined = torch.zeros(B, W, device=fpn_output.device)

        for peak_idx in range(effective_k):
            peak_positions = topk_indices[:, peak_idx]

            windows = self._extract_windows(
                fpn_output, peak_positions, self.refiner_window, attention_mask,
            )

            conv_out = self.conv_stack(windows.transpose(1, 2)).transpose(1, 2)
            fine_logits = self.fine_mlp(conv_out).squeeze(-1)
            # Numerically safe: replace invalid positions with a tiny value → exp(tiny)≈0
            # When all logits are invalid, all_invalid forces zero contribution
            mask = torch.isfinite(fine_logits)
            all_invalid = ~mask.any(dim=-1, keepdim=True)
            safe_logits = torch.where(mask, fine_logits, torch.finfo(fine_logits.dtype).min)
            fine_probs = F.softmax(safe_logits, dim=-1)
            fine_probs = torch.where(
                all_invalid, torch.zeros_like(fine_probs), fine_probs
            )

            all_refined.append(fine_probs)

            confidence = topk_vals[:, peak_idx]
            update_mask = confidence > best_confidences
            best_refined[update_mask] = fine_probs[update_mask]
            best_confidences = torch.maximum(best_confidences, confidence)

        return {
            "bp_heatmaps": bp_heatmaps,
            "bp_refined": best_refined,
            "all_refined": all_refined,
            "peak_positions": topk_indices,
        }

    def _extract_windows(
        self,
        fpn_output: torch.Tensor,
        centers: torch.Tensor,
        half_window: int,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Extract a hidden-state window of ±half_window, zero-padded at the boundaries."""
        B, L, D = fpn_output.shape
        W = 2 * half_window

        offsets = torch.arange(-half_window, half_window, device=centers.device)
        indices = centers.unsqueeze(1) + offsets.unsqueeze(0)

        valid_mask = (indices >= 0) & (indices < L)
        clamped_indices = indices.clamp(0, L - 1)

        indices_expanded = clamped_indices.unsqueeze(-1).expand(-1, -1, D)
        windows = torch.gather(fpn_output, 1, indices_expanded)

        if attention_mask is not None:
            gathered_masks = torch.gather(attention_mask, 1, clamped_indices)
            valid_mask = valid_mask & (gathered_masks > 0)

        windows = windows * valid_mask.unsqueeze(-1).float()
        return windows
