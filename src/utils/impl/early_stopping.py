# utils/impl/early_stopping.py
# NOTE: TrainingEngine already implements early stopping inline; this class is used only by the utils/api.py self-test.
"""
DeepFluHGT  —  Early Stopping
==================================
Pluggable early-stopping strategy.
"""

from __future__ import annotations

from typing import Optional


class EarlyStopping:
    """
    Pluggable early-stopping controller.

    Parameters
    ----------
    patience : int
        number of tolerated epochs (default 5)
    min_delta : float
        minimum improvement (default 0.0)
    mode : str
        "min" (lower metric is better) or "max"
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def step(self, metric: float) -> bool:
        """
        Update the early-stopping state.

        Parameters
        ----------
        metric : float

        Returns
        -------
        should_stop : bool
        """
        if self.best_score is None:
            self.best_score = metric
            return False

        improved = (
            (metric < self.best_score - self.min_delta) if self.mode == "min"
            else (metric > self.best_score + self.min_delta)
        )

        if improved:
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def reset(self):
        """Reset the state."""
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    @property
    def state_dict(self) -> dict:
        return {
            "counter": self.counter,
            "best_score": self.best_score,
            "should_stop": self.should_stop,
        }

    @state_dict.setter
    def state_dict(self, d: dict):
        self.counter = d.get("counter", 0)
        self.best_score = d.get("best_score")
        self.should_stop = d.get("should_stop", False)
