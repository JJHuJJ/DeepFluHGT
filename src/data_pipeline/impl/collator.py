# data_pipeline/impl/collator.py
"""
DeepFluHGT  —  batch collator
================================
Collates variable-length lists of HGTSampleTokenized into equal-length HGTBatch tensors.

Core functionality
--------
1. Dynamic padding: pad to the maximum length within the batch
2. Hard-negative balancing: optionally balance NEG/NEG_VIRUS/NEG_HARD/NEG_PCR
3. Return an HGTBatch tensor
"""

from __future__ import annotations

import random
from typing import List, Optional

# m-13 fix: use an explicit Random instance, isolating global random state
_rng = random.Random(42)

import numpy as np
import torch

from data_pipeline.contracts import HGTSampleTokenized, HGTBatch


# ═══════════════════════════════════════════════════════════════════════════════ #
#  Helper functions
# ═══════════════════════════════════════════════════════════════════════════════ #

def _pad_1d(
    sequences: List[List[int]],
    pad_value: int = 0,
) -> torch.Tensor:
    """Pad a variable-length list of integer lists into an equal-length tensor [B, max_len]."""
    max_len = max(len(s) for s in sequences)
    batch_size = len(sequences)
    padded = torch.full((batch_size, max_len), pad_value, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
    return padded


def _pad_1d_float(
    sequences: List[List[float]],
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Pad a variable-length list of float lists into an equal-length tensor [B, max_len]."""
    max_len = max(len(s) for s in sequences)
    batch_size = len(sequences)
    padded = torch.full((batch_size, max_len), pad_value, dtype=torch.float32)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.float32)
    return padded


# ═══════════════════════════════════════════════════════════════════════════════ #
#  Hard-negative balancing sampler
# ═══════════════════════════════════════════════════════════════════════════════ #

class HardNegativeBalancer:
    """
    Controls the sampling ratio of hard negatives within a batch.

    Parameters
    ----------
    hard_neg_ratio : float
        Target ratio of hard negatives among all negatives
    balanced_neg_types : bool
        Whether to distribute evenly across the four negative types (25% each)
    """

    POSITIVE_TYPES = {"POS"}
    EASY_NEG_TYPES = {"NEG"}
    HARD_NEG_TYPES = {"NEG_VIRUS", "NEG_HARD", "NEG_PCR"}

    def __init__(
        self,
        hard_neg_ratio: float = 0.5,
        balanced_neg_types: bool = True,
    ):
        self.hard_neg_ratio = hard_neg_ratio
        self.balanced_neg_types = balanced_neg_types

    def rebalance(
        self,
        samples: List[HGTSampleTokenized],
        target_batch_size: int,
    ) -> List[HGTSampleTokenized]:
        """Apply balanced resampling to the input sample list."""
        if not samples:
            return samples

        groups = {
            "pos": [], "easy_neg": [],
            "neg_virus": [], "neg_hard": [], "neg_pcr": [],
        }

        for s in samples:
            if s.label == 1:
                groups["pos"].append(s)
            elif s.sample_type == "NEG":
                groups["easy_neg"].append(s)
            elif s.sample_type == "NEG_VIRUS":
                groups["neg_virus"].append(s)
            elif s.sample_type == "NEG_HARD":
                groups["neg_hard"].append(s)
            elif s.sample_type == "NEG_PCR":
                groups["neg_pcr"].append(s)
            else:
                groups["easy_neg"].append(s)

        n_pos_target = target_batch_size // 2
        n_neg_target = target_batch_size - n_pos_target

        if self.balanced_neg_types:
            n_each = n_neg_target // 4
            n_easy, n_virus, n_hard = n_each, n_each, n_each
            n_pcr = n_neg_target - 3 * n_each
        else:
            n_easy = int(n_neg_target * (1 - self.hard_neg_ratio))
            n_hard_total = n_neg_target - n_easy
            n_virus = n_hard_total // 3
            n_hard = n_hard_total // 3
            n_pcr = n_hard_total - 2 * (n_hard_total // 3)

        def _sample_group(group, n):
            if not group or n <= 0:
                return []
            return _rng.choices(group, k=n)

        result = (
            _sample_group(groups["pos"], n_pos_target)
            + _sample_group(groups["easy_neg"], n_easy)
            + _sample_group(groups["neg_virus"], n_virus)
            + _sample_group(groups["neg_hard"], n_hard)
            + _sample_group(groups["neg_pcr"], n_pcr)
        )

        _rng.shuffle(result)
        return result[:target_batch_size]


# ═══════════════════════════════════════════════════════════════════════════════ #
#  HGTCollator  —  main collator
# ═══════════════════════════════════════════════════════════════════════════════ #

class HGTCollator:
    """
    Batch collator: HGTSampleTokenized list → HGTBatch.

    Parameters
    ----------
    pad_id : int
        Token padding ID (default 0)
    rebalance_hard_neg : bool
        Whether to enable hard-negative balanced resampling
    hard_neg_ratio : float
        Hard-negative ratio
    balanced_neg_types : bool
        Whether to balance the four negative sample types
    target_batch_size : Optional[int]
        Target batch size for resampling
    """

    def __init__(
        self,
        pad_id: int = 0,
        rebalance_hard_neg: bool = False,
        hard_neg_ratio: float = 0.5,
        balanced_neg_types: bool = True,
        target_batch_size: Optional[int] = None,
    ):
        self.pad_id = pad_id
        self.rebalance_hard_neg = rebalance_hard_neg

        if rebalance_hard_neg:
            self.balancer = HardNegativeBalancer(
                hard_neg_ratio=hard_neg_ratio,
                balanced_neg_types=balanced_neg_types,
            )
        else:
            self.balancer = None

        self.target_batch_size = target_batch_size

    def __call__(
        self,
        samples: List[HGTSampleTokenized],
    ) -> HGTBatch:
        """Collate one batch."""
        assert len(samples) > 0, "cannot collate an empty list"

        # Optional: hard-negative balanced resampling
        if self.rebalance_hard_neg and self.balancer is not None:
            bs = self.target_batch_size or len(samples)
            samples = self.balancer.rebalance(samples, bs)

        # Step 1: Padding primarysequence
        input_ids = _pad_1d(
            [s.input_ids for s in samples], pad_value=self.pad_id)
        attention_mask = _pad_1d(
            [s.attention_mask for s in samples], pad_value=0)

        # Step 2: Pad breakpoint targets
        region_targets = _pad_1d_float(
            [s.region_targets for s in samples], pad_value=0.0)
        breakpoint_targets = _pad_1d_float(
            [s.breakpoint_targets for s in samples], pad_value=0.0)

        # Step 3: Classification labels
        labels = torch.tensor([s.label for s in samples], dtype=torch.long)

        # Step 4: Auxiliary task tags
        pe_case = torch.tensor([s.pe_case for s in samples], dtype=torch.long)
        hgt_mode = torch.tensor([s.hgt_mode for s in samples], dtype=torch.long)
        n_events = torch.tensor([s.n_events for s in samples], dtype=torch.long)

        # Step 5: Biological features
        bio_features = torch.from_numpy(
            np.stack([s.bio_features for s in samples]))

        # Step 5b (v7): Evo2 offline embeddings (fixed D_evo, no padding needed)
        # If no sample has an evo cache entry (length 0) → batch.evo_embeddings = None; the model skips fusion.
        # Note: samples within a batch must either all have cache entries or all lack them
        # (precompute must cover all seq_ids).
        if {len(s.evo_embeddings) for s in samples} == {0}:
            evo_embeddings = None
        else:
            evo_embeddings = torch.from_numpy(
                np.stack([s.evo_embeddings for s in samples]))

        # Step 6: Metadata
        seq_ids = [s.seq_id for s in samples]
        sample_types = [s.sample_type for s in samples]

        # Step 7: Window metadata (sliding-window mode; fill defaults in other modes)
        window_starts = []
        window_ends = []
        original_seq_lens = []
        original_seq_ids = []

        for s in samples:
            if hasattr(s, 'window_meta') and s.window_meta is not None:
                wm = s.window_meta
                window_starts.append(wm.window_start)
                window_ends.append(wm.window_end)
                original_seq_lens.append(wm.original_seq_len)
                original_seq_ids.append(wm.original_seq_id)
            else:
                window_starts.append(0)
                window_ends.append(max(0, len(s.input_ids) - 2))
                original_seq_lens.append(max(0, len(s.input_ids) - 2))
                original_seq_ids.append(s.seq_id)

        return HGTBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bio_features=bio_features,
            evo_embeddings=evo_embeddings,
            labels=labels,
            breakpoint_targets=breakpoint_targets,
            region_targets=region_targets,
            pe_case=pe_case,
            hgt_mode=hgt_mode,
            n_events=n_events,
            seq_ids=seq_ids,
            sample_types=sample_types,
            window_starts=window_starts,
            window_ends=window_ends,
            original_seq_lens=original_seq_lens,
            original_seq_ids=original_seq_ids,
        )
