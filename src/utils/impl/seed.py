# utils/impl/seed.py
"""
DeepFluHGT.8  —  global seed control
================================
Set deterministic seeds for Python / NumPy / PyTorch / CUDA to ensure reproducible experiments.

usage:
    from utils.impl.seed import seed_all
    seed_all(42)
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


def seed_all(seed: int, *, deterministic: bool = True) -> None:
    """
    Set all RNG seeds to ensure reproducibility.

    Parameters
    ----------
    seed : int
        global random seed value.
    deterministic : bool, optional
        whether to enable cuDNN deterministic mode (default True). Setting False can speed up training,
        at the cost of reproducibility.
    """
    # Python built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # cuDNN deterministic
    if deterministic and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # M-07 fix: enable PyTorch global deterministic-algorithms mode
    # controls CUDA non-determinism of torch.gather, scatter_add, attention, etc.
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        # warn_only=True: operations without deterministic support only warn instead of erroring,
        # avoiding blocked training (some CUDA kernels have no deterministic implementation)


def worker_init_fn(worker_id: int, base_seed: int = 42) -> None:
    """
    DataLoader worker init function; sets an independent seed per worker.

    Parameters
    ----------
    worker_id : int
        worker index passed automatically by DataLoader.
    base_seed : int
        base seed value.
    """
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
