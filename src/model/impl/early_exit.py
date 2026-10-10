# model/impl/early_exit.py
"""
DeepFluHGT  —  early-exit adaptive mechanism
==================================
Lightweight EarlyExitClassifiers are deployed after the fusion layer and the
graph interaction layer. At inference time, when confidence > threshold, the
remaining expensive computation is skipped.

Deployment points:
  - exit_1: after the fusion layer (FeatureSplitBlock + SequenceFPN)
  - exit_2: after the graph interaction layer (BatchGraphInteraction)

Ported directly from v5.0.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════ #
#  EarlyExitClassifier  —  lightweight classification head
# ═══════════════════════════════════════════════════════════════════════════════ #

class EarlyExitClassifier(nn.Module):
    """
    Lightweight early-exit classifier.

    Architecture: pooled [B, D] → Linear(D, D//4) → GELU → Dropout → Linear(D//4, 1)

    Parameters
    ----------
    hidden_dim : int
    dropout : float
    """

    def __init__(self, hidden_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pooled : [B, D]

        Returns
        -------
        logits : [B, 1]
        """
        return self.classifier(pooled)


# ═══════════════════════════════════════════════════════════════════════════════ #
#  EarlyExitManager  —  multi-point early-exit coordinator
# ═══════════════════════════════════════════════════════════════════════════════ #

class EarlyExitManager(nn.Module):
    """
    Manages multiple early-exit classifiers.

    Parameters
    ----------
    hidden_dim : int
    threshold : float
    dropout : float
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        threshold: float = 0.95,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.threshold = threshold

        self.exit_fusion = EarlyExitClassifier(hidden_dim, dropout)
        self.exit_graph = EarlyExitClassifier(hidden_dim, dropout)

    def check_exit(
        self,
        logits: torch.Tensor,
        early_exit: bool = False,
    ) -> torch.Tensor:
        """
        Decide whether each sample should early-exit.

        Returns
        -------
        should_exit : [B] bool
        """
        if not early_exit:
            return torch.zeros(logits.size(0), dtype=torch.bool, device=logits.device)
        confidence = torch.sigmoid(logits.abs()).squeeze(-1)
        return confidence > self.threshold

    def compute_exit_loss(
        self,
        exit_logits: torch.Tensor,
        labels: torch.Tensor,
        weight: float = 0.25,
    ) -> torch.Tensor:
        """Compute the auxiliary training loss for the early-exit classifier."""
        return weight * F.binary_cross_entropy_with_logits(
            exit_logits.squeeze(-1), labels.float(),
        )

    def forward_fusion(
        self,
        pooled: torch.Tensor,
        early_exit: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Early-exit check after the fusion layer."""
        exit_logits = self.exit_fusion(pooled)
        should_exit = self.check_exit(exit_logits, early_exit)
        return exit_logits, should_exit, pooled

    def forward_graph(
        self,
        pooled: torch.Tensor,
        early_exit: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Early-exit check after the graph interaction layer."""
        exit_logits = self.exit_graph(pooled)
        should_exit = self.check_exit(exit_logits, early_exit)
        return exit_logits, should_exit, pooled

    def build_early_exit_output(
        self,
        exit_logits: torch.Tensor,
        exit_layer: int,
        seq_len: int = 1,
    ) -> Dict:
        """Build a minimal HGTOutput-compatible dict from the early-exit logits."""
        B = exit_logits.size(0)
        return {
            "cls_logits": exit_logits,
            "bp_heatmaps": torch.zeros(
                B, seq_len, device=exit_logits.device,
            ),
            "role_logits": torch.zeros(
                B, 1, device=exit_logits.device,
            ),
            "direction_logits": None,
            "bp_refined": None,
            "split_gates": None,
            "donor_pool": None,
            "recipient_pool": None,
            "hidden_states": None,
            "exited_at": exit_layer,
        }
