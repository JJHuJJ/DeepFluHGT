# utils/impl/logging.py
"""
DeepFluHGT  —  unified logging interface
================================
Pluggable logging supporting print / file / wandb / tensorboard.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional


class Logger:
    """
    Unified logging interface.

    Supported backends:
      - "print": standard output
      - "file": write JSONL file
      - "none": silent

    Parameters
    ----------
    backend : str
    log_dir : str
    project : str (wandb use)
    """

    def __init__(
        self,
        backend: str = "print",
        log_dir: str = "./logs",
        project: str = "DeepFluHGT",
    ):
        self.backend = backend
        self.log_dir = log_dir
        self.project = project
        self._file = None
        self._history = []

        if backend == "file":
            os.makedirs(log_dir, exist_ok=True)
            self._file = open(
                os.path.join(log_dir, "training.jsonl"), "a"
            )

    def log(self, metrics: Dict[str, Any], step: int = 0, prefix: str = ""):
        """Log metrics."""
        entry = {
            "step": step,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            **{f"{prefix}{k}": v for k, v in metrics.items()},
        }
        self._history.append(entry)

        if self.backend == "print":
            parts = [f"[Step {step}]"]
            for k, v in metrics.items():
                if isinstance(v, float):
                    parts.append(f"{k}={v:.4f}")
                else:
                    parts.append(f"{k}={v}")
            print("  ".join(parts))

        elif self.backend == "file":
            self._file.write(json.dumps(entry) + "\n")
            self._file.flush()

    def info(self, msg: str):
        """Print an informational message."""
        if self.backend in ("print", "file"):
            print(msg)

    def close(self):
        """closelog."""
        if self._file is not None:
            self._file.close()

    def get_history(self) -> list:
        """Return all records."""
        return self._history
