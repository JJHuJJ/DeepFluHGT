# loss/impl/uncertainty_weight.py
"""
DeepFluHGT  —  TaskUncertaintyWeight (Innovation 12)
========================================================
Learnable homoscedastic-uncertainty parameters automatically balance the multi-task loss (Kendall et al. 2018).

loss_total = Σ_i [ exp(-log_var_i) * loss_i + log_var_i ]

- log_var_i is a learnable parameter, initialized to 0
- exp(-log_var_i) acts as inverse-variance weighting
- the log_var_i regularizer prevents weights from growing unbounded
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class TaskUncertaintyWeight(nn.Module):
    """
    Learnable automatic multi-task weighting.

    Parameters
    ----------
    num_tasks : int
        number of tasks (default 4: cls, bp, role, direction)
    task_names : list of str, optional
        task name list (for logging)
    """

    def __init__(
        self,
        num_tasks: int = 4,
        task_names: list = None,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_names = task_names or [f"task_{i}" for i in range(num_tasks)]
        assert len(self.task_names) == num_tasks

        # learnable log-variance parameters, initialized to 0 → initial weights are all 1.0
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, *losses: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        *losses : variable number of scalar loss tensors (one per task)

        Returns
        -------
        (total_weighted_loss, dict of per-task weighted losses)
        """
        assert len(losses) == self.num_tasks, (
            f"Expected {self.num_tasks} losses, got {len(losses)}"
        )

        weighted = {}
        total = torch.tensor(0.0, device=losses[0].device)

        for i, loss in enumerate(losses):
            lv = self.log_vars[i].clamp(-10.0, 6.0)  # tightened upper bound: exp(-6)≈0.0025, prevents task weights collapsing to zero
            precision = torch.exp(-lv)
            w_loss = 0.5 * precision * loss + 0.5 * lv  # Kendall 2018 standard form
            name = self.task_names[i] if i < len(self.task_names) else f"task_{i}"
            weighted[name] = w_loss
            total = total + w_loss

        return total, weighted

    def get_weights(self) -> Dict[str, float]:
        """Return the current effective weight of each task (exp(-log_var))."""
        return {
            name: torch.exp(-self.log_vars[i]).item()
            for i, name in enumerate(self.task_names)
        }
