# model/impl/bio_features.py
"""
DeepFluHGT  —  biological prior feature encoder
========================================
Maps the 260-dimensional hand-crafted biological features (including GC skew,
4-mer frequencies, Shannon entropy, etc.) into the same hidden space as the
foundation models.

Architecture: Linear(260, 128) → GELU → LayerNorm(128) → Dropout → Linear(128, 768)

Ported directly from v5.0, no token-size dependency.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class BiologicalFeatureEncoder(nn.Module):
    """
    Learnable biological feature encoder.

    Parameters
    ----------
    bio_feature_dim : int
        Input feature dimension (default 260)
    hidden_dim : int
        Target hidden dimension (default 768)
    bio_encoder_hidden : int
        MLP intermediate dimension (default 128)
    dropout : float
        Dropout rate (default 0.1)
    """

    def __init__(
        self,
        bio_feature_dim: int = 260,
        hidden_dim: int = 768,
        bio_encoder_hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bio_feature_dim = bio_feature_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(bio_feature_dim, bio_encoder_hidden),
            nn.GELU(),
            nn.LayerNorm(bio_encoder_hidden),
            nn.Dropout(dropout),
            nn.Linear(bio_encoder_hidden, hidden_dim),
        )

    def forward(
        self,
        bio_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Encode biological features.

        Parameters
        ----------
        bio_features : [B, 260] or None

        Returns
        -------
        [B, hidden_dim]
        """
        if bio_features is None or bio_features.numel() == 0:
            B = 1
            device = next(self.parameters()).device
            return torch.zeros(B, self.hidden_dim, device=device)

        if bio_features.dtype != torch.float32:
            bio_features = bio_features.float()

        return self.encoder(bio_features)


class EvoEncoder(nn.Module):
    """
    Evo2 offline embedding encoder (v7).

    Maps precomputed Evo2 pooled embeddings [B, D_evo] into the same hidden
    space as the foundation models. Structure mirrors BiologicalFeatureEncoder
    (Linear→GELU→LayerNorm→Dropout→Linear). Evo2 acts as a third, offline-precomputed
    foundation-model modality (global prior) and does not participate in the online forward pass.

    Parameters
    ----------
    evo_dim : int
        Evo2 embedding dimension (evo2_7b = 4096)
    hidden_dim : int
        Target hidden dimension (default 768)
    evo_hidden : int
        MLP intermediate dimension (default 512)
    dropout : float
    """

    def __init__(
        self,
        evo_dim: int = 4096,
        hidden_dim: int = 768,
        evo_hidden: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.evo_dim = evo_dim
        self.hidden_dim = hidden_dim
        # Input LayerNorm: the Evo2 mean-pooled embedding norm is ~1e4 (elements ~1e2),
        # so it must be normalized before entering the MLP (bio_features input ~O(1) has no such issue).
        self.encoder = nn.Sequential(
            nn.LayerNorm(evo_dim),
            nn.Linear(evo_dim, evo_hidden),
            nn.GELU(),
            nn.LayerNorm(evo_hidden),
            nn.Dropout(dropout),
            nn.Linear(evo_hidden, hidden_dim),
        )

    def forward(self, evo_embeddings: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        evo_embeddings : [B, D_evo] or None

        Returns
        -------
        [B, hidden_dim]
        """
        if evo_embeddings is None or evo_embeddings.numel() == 0:
            B = 1
            device = next(self.parameters()).device
            return torch.zeros(B, self.hidden_dim, device=device)

        if evo_embeddings.dtype != torch.float32:
            evo_embeddings = evo_embeddings.float()

        return self.encoder(evo_embeddings)
