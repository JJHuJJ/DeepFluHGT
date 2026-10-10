# loss/impl/cascade_gating.py
"""
DeepFluHGT  —  CascadeGatingLoss (Innovation 9)
====================================================
Cascade gating: gate auxiliary-task gradients with the classification confidence.

confidence = sigmoid(slope * cls_logit.detach())
- no learnable parameters
- detach() ensures the gating does not affect the classification head
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CascadeGatingLoss(nn.Module):
    """
    Cascade gating signal generator.

    Parameters
    ----------
    slope : float
        sharpening slope (default 10.0)
    """

    def __init__(self, slope: float = 10.0):
        super().__init__()  # DC-12 fix: inherit nn.Module for PyTorch module-system compatibility
        self.slope = slope

    def __call__(self, cls_logits: torch.Tensor) -> torch.Tensor:
        """
        m-01 note: for negative samples confidence ≈ 0, zeroing all auxiliary-loss gradients.
        This is correct for bp/direction (negatives have no breakpoint/direction), but the role head
        only sees positives during training while inference must process all samples  —  a known design trade-off.

        Parameters
        Parameters
        ----------
        cls_logits : [B, 1] raw classification logit

        Returns
        -------
        confidence : [B] detached, valueat (0, 1)
        """
        return torch.sigmoid(self.slope * cls_logits.detach()).squeeze(-1)
