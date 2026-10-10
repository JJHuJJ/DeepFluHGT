# utils/impl/checkpoint.py
"""
DeepFluHGT  —  Checkpoint manager
====================================
Pluggable checkpoint save/load with best-model tracking.
"""

from __future__ import annotations

import glob
import os
import json
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class CheckpointManager:
    """
    Pluggable Checkpoint manager.

    Features:
      - save() / load() full training state
      - save_best() best-model tracking
      - save_last() latest model
      - list_checkpoints() lists all checkpoints
      - cleanup() keeps the most recent N checkpoints

    Parameters
    ----------
    save_dir : str
    max_keep : int  (keep at most N checkpoints, 0 = unlimited)
    """

    def __init__(self, save_dir: str = "./checkpoints", max_keep: int = 5):
        self.save_dir = save_dir
        self.max_keep = max_keep
        self.best_metric = float("inf")
        self.best_path = None
        os.makedirs(save_dir, exist_ok=True)

    def save(
        self,
        state: Dict[str, Any],
        epoch: int,
        metric: Optional[float] = None,
        is_best: bool = False,
        tag: str = "",
    ) -> str:
        """
        savecheckpoint.

        Parameters
        ----------
        state : dict  (model, optimizer, scheduler, ...)
        epoch : int
        metric : float (evaluation metric, lower is better)
        is_best : bool
        tag : str (filename tag)

        Returns
        -------
        path : str
        """
        tag_str = f"_{tag}" if tag else ""
        filename = f"ckpt_epoch{epoch:03d}{tag_str}.pt"
        path = os.path.join(self.save_dir, filename)

        state["_epoch"] = epoch
        state["_timestamp"] = time.time()
        if metric is not None:
            state["_metric"] = metric

        torch.save(state, path)

        # Best model
        if is_best or (metric is not None and metric < self.best_metric):
            self.best_metric = metric
            self.best_path = path
            best_path = os.path.join(self.save_dir, "best_model.pt")
            torch.save(state, best_path)

        # Cleanup
        if self.max_keep > 0:
            self.cleanup()

        return path

    def load(self, path: str, map_location=None) -> Dict:
        """loadcheckpoint."""
        if map_location is None:
            map_location = "cpu"
        return torch.load(path, map_location=map_location, weights_only=False)

    def load_best(self) -> Optional[Dict]:
        """Load the best model."""
        best_path = os.path.join(self.save_dir, "best_model.pt")
        if os.path.exists(best_path):
            return self.load(best_path)
        return None

    def load_latest(self) -> Optional[Dict]:
        """Load the latest checkpoint."""
        ckpts = sorted(glob.glob(os.path.join(self.save_dir, "ckpt_*.pt")))
        if ckpts:
            return self.load(ckpts[-1])
        return None

    def list_checkpoints(self):
        """List all checkpoints."""
        return sorted(glob.glob(os.path.join(self.save_dir, "ckpt_*.pt")))

    def cleanup(self):
        """Keep the most recent N checkpoints."""
        ckpts = sorted(glob.glob(os.path.join(self.save_dir, "ckpt_*.pt")))
        while len(ckpts) > self.max_keep:
            os.remove(ckpts.pop(0))
