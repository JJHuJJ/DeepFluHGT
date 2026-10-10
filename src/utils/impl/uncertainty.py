# utils/impl/uncertainty.py
"""
DeepFluHGT  —  uncertainty estimation
================================
Monte Carlo Dropout / Deep Ensemble utilities.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def mc_dropout_predict(
    model: nn.Module,
    batch,
    n_forward: int = 10,
    device: str = "cpu",
) -> dict:
    """
    Monte Carlo Dropout uncertainty estimation.

    Enable dropout and run multiple forward passes, then summarize the predictive distribution.

    Parameters
    ----------
    model : nn.Module
    batch : HGTBatch
    n_forward : int (number of forward passes)
    device : str

    Returns
    -------
    dict: mean_probs, std_probs, entropy
    """
    model.train()  # keep dropout active
    all_probs = []

    with torch.no_grad():
        for _ in range(n_forward):
            output = model(batch)
            probs = torch.sigmoid(output.cls_logits.squeeze(-1))
            all_probs.append(probs.cpu().numpy())

    all_probs = np.stack(all_probs)  # [n_forward, B]
    mean_probs = all_probs.mean(axis=0)
    std_probs = all_probs.std(axis=0)

    # Entropy
    eps = 1e-8
    entropy = -(
        mean_probs * np.log(mean_probs + eps)
        + (1 - mean_probs) * np.log(1 - mean_probs + eps)
    )

    return {
        "mean_probs": mean_probs,
        "std_probs": std_probs,
        "entropy": entropy,
        "predictions": all_probs,
    }


def predictive_entropy(mean_probs: np.ndarray) -> np.ndarray:
    """Compute predictive entropy."""
    eps = 1e-8
    return -(mean_probs * np.log(mean_probs + eps) +
             (1 - mean_probs) * np.log(1 - mean_probs + eps))


def mutual_information(all_probs: np.ndarray) -> np.ndarray:
    """
    Mutual information (MI): total entropy - mean conditional entropy.
    Measures the part of model uncertainty beyond "data uncertainty" (i.e. epistemic uncertainty).
    """
    mean_probs = all_probs.mean(axis=0)
    total_entropy = predictive_entropy(mean_probs)

    eps = 1e-8
    cond_entropy = -np.mean(
        all_probs * np.log(all_probs + eps) +
        (1 - all_probs) * np.log(1 - all_probs + eps),
        axis=0,
    )

    return total_entropy - cond_entropy
