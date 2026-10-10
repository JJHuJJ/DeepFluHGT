# loss/impl/rc_consistency.py
"""
DeepFluHGT  —  RCConsistencyLoss (Innovation 10)
=====================================================
Symmetric KL divergence between the forward- and reverse-strand classification outputs, enforcing strand invariance.
loss = 0.5 * (KL(p∥q) + KL(q∥p))
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RCConsistencyLoss(nn.Module):
    """
    Forward/reverse-strand consistency loss.

    Applies a symmetric KL constraint to the R1 (forward) and R2 (RC) classification outputs in PE mode,
    ensuring the model gives consistent predictions on both strands.
    """

    def forward(
        self,
        forward_logits: torch.Tensor,
        rc_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        forward_logits : [B, 1] forward-strand cls logit
        rc_logits      : [B, 1] reverse-strand cls logit

        Returns
        -------
        scalar loss
        """
        with torch.amp.autocast(forward_logits.device.type, enabled=False):
            fwd = forward_logits.float()
            rc = rc_logits.float()
            p = torch.sigmoid(fwd).clamp(1e-6, 1 - 1e-6)
            q = torch.sigmoid(rc).clamp(1e-6, 1 - 1e-6)

            kl_pq = p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))
            kl_qp = q * torch.log(q / p) + (1 - q) * torch.log((1 - q) / (1 - p))

            return 0.5 * (kl_pq + kl_qp).mean()
