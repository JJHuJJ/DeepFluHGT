# data_pipeline/impl/dataset.py
"""
DeepFluHGT  —  HGTDataset (PyTorch Dataset)
================================================
Converts a list of HGTSample into HGTSampleTokenized (tokenized + targets).
"""

from __future__ import annotations

import os
import random
import sys
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from schema import PE_CASE_MAP, HGT_MODE_MAP

from data_pipeline.contracts import (
    HGTSample, HGTSampleTokenized, BreakpointEvent, WindowMeta,
)
from data_pipeline.impl.tokenizer import SingleNucleotideTokenizer
from data_pipeline.impl.bio_features import BioFeatureExtractor


def _load_evo_cache(evo_cache_path: Optional[str]) -> Optional[dict]:
    """Load the Evo2 offline embedding cache {seq_id: np.ndarray[D_evo]} (v7).

    The cache is a dict saved by scripts/precompute_evo2_embeddings.py via torch.save.
    Returning None means the feature is disabled (model forward will skip Evo2
    fusion and fall back to the dual-encoder configuration).
    """
    if not evo_cache_path:
        return None
    if not os.path.exists(evo_cache_path):
        print(f"[EvoCache] warning: evo_cache_path does not exist ({evo_cache_path}), disabling Evo2 fusion")
        return None
    # weights_only=False: the cache is a {seq_id: np.ndarray} dict (self-produced, trusted file);
    # torch>=2.6 defaults to weights_only=True which rejects numpy arrays, so it must be explicitly relaxed.
    cache = torch.load(evo_cache_path, map_location="cpu", weights_only=False)
    print(f"[EvoCache] load Evo2 embedding cache: {evo_cache_path} ({len(cache)} item)")
    return cache


# ── v7: RC data augmentation (replaces the RC equivariance inductive bias of the removed Caduceus-PS) ──
# Implemented at the sequence level: first revcomp the entire sequence, then remap event
# coordinates; tokens/targets/bio features all stay consistent automatically. Applied only
# during training with probability rc_aug_prob; contig mode is supported first
# (including sliding windows). PE-mode fragment-level RC = R1/R2 swap, which involves
# pe_case semantics, so it is not enabled for now.
_RC_TRANS = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _revcomp(seq: str) -> str:
    """DNA reverse complement."""
    return seq.translate(_RC_TRANS)[::-1]


def _rc_events(events: List[BreakpointEvent], seq_len: int) -> List[BreakpointEvent]:
    """Remap event coordinates to reverse-complement space: [s, e) → [L-e, L-s). Negative-sample markers (-1) are kept as-is."""
    out = []
    for e in events:
        if e.hgt_start < 0:
            out.append(e)
            continue
        out.append(BreakpointEvent(
            event_idx=e.event_idx,
            hgt_start=seq_len - e.hgt_end,
            hgt_end=seq_len - e.hgt_start,
            virus_len=e.virus_len,
            hgt_mode=e.hgt_mode,
        ))
    return out


class HGTDataset(Dataset):
    """
    DeepFluHGT dataset.

    Parameters
    ----------
    samples : List[HGTSample]
        List of parsed samples (produced by BEDParser)
    mode : str
        "paired_end" or "contig"
    max_length : int
        Maximum sequence length (bp); longer sequences are truncated (contig mode only)
    read_len : int
        PE mode read length (default 150)
    extract_bio : bool
        Whether to extract biological features
    adaptive_window : bool
        Whether to use adaptive window sizing in contig mode (default True)
    """

    # Adaptive window tier thresholds (bp)
    _WINDOW_TIERS = [
        (500, 500),      # ≤500bp: keep original length
        (1000, 800),     # ≤1kb: 800bp window
        (2000, 1200),    # ≤2kb: 1200bp window
        (5000, 2000),    # ≤5kb: 2000bp window
        (10000, 3000),   # ≤10kb: 3000bp window
        (20000, 4000),   # ≤20kb: 4000bp window
    ]
    _DEFAULT_WINDOW = 5000  # default window for >20kb

    def __init__(
        self,
        samples: List[HGTSample],
        mode: str = "paired_end",
        max_length: int = 10000,
        read_len: int = 150,
        extract_bio: bool = True,
        adaptive_window: bool = True,
        evo_cache_path: Optional[str] = None,
        rc_aug_prob: float = 0.0,
    ):
        self.samples = samples
        self.mode = mode
        self.max_length = max_length
        self.read_len = read_len
        self.extract_bio = extract_bio
        self.adaptive_window = adaptive_window and (mode == "contig")
        self.rc_aug_prob = rc_aug_prob

        self.tokenizer = SingleNucleotideTokenizer()
        self.bio_extractor = BioFeatureExtractor(kmer_size=4) \
            if extract_bio else None

        # v7: Evo2 offline embedding cache (per seq_id)
        self._evo_cache = _load_evo_cache(evo_cache_path)
        self._evo_dim = (len(next(iter(self._evo_cache.values())))
                         if self._evo_cache else 0)

        print(f"[HGTDataset] mode={mode}, n={len(samples)}, "
              f"max_length={max_length}"
              f"{', adaptive_window=True' if self.adaptive_window else ''}"
              f"{f', evo_cache=on(dim={self._evo_dim})' if self._evo_cache else ', evo_cache=off'}")

    def _evo_vec(self, seq_id: str) -> np.ndarray:
        """Fetch the Evo2 embedding by seq_id (v7).
        No cache → empty array (the collator detects the all-empty case and disables fusion);
        cache present but this sample missing → zero vector (graceful)."""
        if not self._evo_cache:
            return np.zeros(0, dtype=np.float32)
        vec = self._evo_cache.get(seq_id)
        if vec is None:
            return np.zeros(self._evo_dim, dtype=np.float32)
        return np.asarray(vec, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> HGTSampleTokenized:
        """Fetch a single tokenized sample."""
        sample = self.samples[idx]

        if self.mode == "paired_end":
            return self._get_pe_item(sample)
        else:
            return self._get_contig_item(sample)

    # ──────────────────────────────────────────────────────────────────────── #
    #  PE mode
    # ──────────────────────────────────────────────────────────────────────── #

    def _get_pe_item(self, sample: HGTSample) -> HGTSampleTokenized:
        """PE mode: [CLS] R1 [SEP] R2 [SEP]"""
        r1_seq = sample.sequence.upper()
        r2_seq = (sample.r2_sequence or "").upper()

        r1_len = len(r1_seq)
        r2_len = len(r2_seq)

        # Tokenize
        input_ids = self.tokenizer.tokenize_pe(r1_seq, r2_seq)
        attention_mask = [1] * len(input_ids)

        # Generate targets
        events = sample.events if sample.events else []
        region_targets, breakpoint_targets = self.tokenizer.create_pe_targets(
            events, r1_len, r2_len)

        # Length consistency
        assert len(input_ids) == len(region_targets), \
            (f"[{sample.seq_id}] input_ids({len(input_ids)}) "
             f"!= region({len(region_targets)})")
        assert len(input_ids) == len(breakpoint_targets), \
            (f"[{sample.seq_id}] input_ids({len(input_ids)}) "
             f"!= bp({len(breakpoint_targets)})")

        # Auxiliary tags
        pe_case_str = events[0].pe_case if events else ""
        hgt_mode_str = events[0].hgt_mode if events else "none"
        pe_case_int = PE_CASE_MAP.get(pe_case_str, 7)
        hgt_mode_int = HGT_MODE_MAP.get(hgt_mode_str, 3)

        n_events = len([e for e in events if e.hgt_start >= 0])

        # Biological features (based on R1+R2 concatenation)
        if self.extract_bio and self.bio_extractor is not None:
            combined_seq = r1_seq + r2_seq
            bio_features = self.bio_extractor.extract(combined_seq)
        else:
            bio_features = np.zeros(260, dtype=np.float32)

        return HGTSampleTokenized(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bio_features=bio_features,
            evo_embeddings=self._evo_vec(sample.seq_id),
            label=sample.label,
            region_targets=region_targets,
            breakpoint_targets=breakpoint_targets,
            pe_case=pe_case_int,
            hgt_mode=hgt_mode_int,
            n_events=n_events,
            seq_id=sample.seq_id,
            sample_type=sample.sample_type,
        )

    # ──────────────────────────────────────────────────────────────────────── #
    #  Contig mode
    # ──────────────────────────────────────────────────────────────────────── #

    def _get_contig_item(self, sample: HGTSample) -> HGTSampleTokenized:
        """Contig mode: [CLS] contig [SEP]"""
        seq = sample.sequence.upper()
        events = sample.events if sample.events else []

        # Adaptive window truncation
        if self.adaptive_window:
            window_size = self._adaptive_window_size(len(seq), events)
        else:
            window_size = self.max_length

        if len(seq) > window_size:
            seq, events = self._truncate_contig(seq, events, window_size)

        seq_len = len(seq)

        # v7: RC data augmentation (contig mode, training only)
        if self.rc_aug_prob > 0 and random.random() < self.rc_aug_prob:
            seq = _revcomp(seq)
            events = _rc_events(events, seq_len)

        # Tokenize
        input_ids = self.tokenizer.tokenize(seq)
        attention_mask = [1] * len(input_ids)

        # Generate targets
        region_targets = self.tokenizer.create_region_targets(events, seq_len)
        breakpoint_targets = self.tokenizer.create_breakpoint_targets(
            events, seq_len)

        # Length consistency
        assert len(input_ids) == len(region_targets), \
            (f"[{sample.seq_id}] input_ids({len(input_ids)}) "
             f"!= region({len(region_targets)})")
        assert len(input_ids) == len(breakpoint_targets), \
            (f"[{sample.seq_id}] input_ids({len(input_ids)}) "
             f"!= bp({len(breakpoint_targets)})")

        # Auxiliary tags
        hgt_mode_str = events[0].hgt_mode if events else "none"
        hgt_mode_int = HGT_MODE_MAP.get(hgt_mode_str, 3)

        n_events = len([e for e in events if e.hgt_start >= 0])

        # Biological features
        if self.extract_bio and self.bio_extractor is not None:
            bio_features = self.bio_extractor.extract(seq)
        else:
            bio_features = np.zeros(260, dtype=np.float32)

        return HGTSampleTokenized(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bio_features=bio_features,
            evo_embeddings=self._evo_vec(sample.seq_id),
            label=sample.label,
            region_targets=region_targets,
            breakpoint_targets=breakpoint_targets,
            pe_case=7,  # contigs have no PE case
            hgt_mode=hgt_mode_int,
            n_events=n_events,
            seq_id=sample.seq_id,
            sample_type=sample.sample_type,
        )

    def _adaptive_window_size(
        self, seq_len: int, events: List[BreakpointEvent],
    ) -> int:
        """
        Adaptively select the window size based on contig length.

        M-02 fix: do not use ground-truth event coordinates to compute the window size.
        The window is selected uniformly by contig-length tier, identically for
        positive and negative samples.

        Returns
        -------
        window_size : int
        """
        max_cap = self.max_length

        # M-02 fix: tier uniformly by contig length; no positive/negative distinction
        for threshold, tier_window in self._WINDOW_TIERS:
            if seq_len <= threshold:
                return min(tier_window, seq_len, max_cap)

        return min(self._DEFAULT_WINDOW, seq_len, max_cap)

    def _truncate_contig(
        self, seq: str, events: List[BreakpointEvent], window_size: int,
    ) -> tuple:
        """
        Truncate a long contig to window_size.

        M-02 fix: for positive samples, the start position is randomly offset by
        ±50% of the window width around the breakpoint. This keeps the breakpoint
        inside the window with high probability (removing the centering bias) while
        the position stays unfixed (mimicking inference, where the breakpoint location
        is unknown). Negative samples still use a random start position.

        Returns
        -------
        (truncated_seq, adjusted_events)
        """
        max_len = window_size

        sample_events_pos = [e for e in events if e.hgt_start >= 0]

        if sample_events_pos:
            # M-02 fix: positive samples  —  random offset near the breakpoint
            first = sample_events_pos[0]
            center = (first.hgt_start + first.hgt_end) // 2
            # Max offset = 50% of the window width, keeping the breakpoint inside the window with high probability
            max_offset = max_len // 2
            offset = random.randint(-max_offset, max_offset)
            ideal_start = center - max_len // 2 + offset
            # Clamp to [0, seq_len - max_len]
            window_start = max(0, min(ideal_start, len(seq) - max_len))
        else:
            # negative samples: fully random position
            window_start = random.randint(
                0, max(0, len(seq) - max_len))

        window_end = min(window_start + max_len, len(seq))
        window_start = max(0, window_end - max_len)

        truncated_seq = seq[window_start:window_end]

        # Adjust event coordinates
        adjusted_events = []
        for e in events:
            new_start = e.hgt_start - window_start
            new_end = e.hgt_end - window_start
            # Keep only events overlapping the window
            if new_end > 0 and new_start < max_len:
                adjusted_events.append(BreakpointEvent(
                    event_idx=e.event_idx,
                    hgt_start=max(0, new_start),
                    hgt_end=min(new_end, max_len),
                    virus_len=e.virus_len,
                    hgt_mode=e.hgt_mode,
                ))

        return truncated_seq, adjusted_events


# ═══════════════════════════════════════════════════════════════════════════════ #
#  Sliding-window contig dataset
# ═══════════════════════════════════════════════════════════════════════════════ #

class SlidingWindowContigDataset(Dataset):
    """
    Sliding-window contig dataset.

    Splits each contig into multiple overlapping windows; each window serves as an
    independent training sample. All coordinates are mapped back to the original
    sequence via WindowMeta.

    Parameters
    ----------
    samples : List[HGTSample]
        List of parsed samples
    window_size : int
        Window size (bp), default 1500 (fits A100 80GB at bs=16)
    overlap : int
        Window overlap (bp), default 300
    is_training : bool
        Whether in training mode (affects positive-sample breakpoint centering
        and negative-sample sampling)
    max_neg_windows : int
        Maximum number of windows per negative contig during training
    extract_bio : bool
        Whether to extract biological features
    """

    def __init__(
        self,
        samples: List[HGTSample],
        window_size: int = 1500,
        overlap: int = 300,
        is_training: bool = True,
        max_neg_windows: int = 2,
        max_eval_windows: int = 10,
        extract_bio: bool = True,
        evo_cache_path: Optional[str] = None,
        rc_aug_prob: float = 0.0,
    ):
        self.samples = samples
        self.window_size = window_size
        self.overlap = overlap
        self.is_training = is_training
        self.max_neg_windows = max_neg_windows
        self.max_eval_windows = max_eval_windows
        self.extract_bio = extract_bio
        self.rc_aug_prob = rc_aug_prob

        self.tokenizer = SingleNucleotideTokenizer()
        self.bio_extractor = BioFeatureExtractor(kmer_size=4) \
            if extract_bio else None

        # v7: Evo2 offline embedding cache (per contig seq_id; all windows of a contig share it)
        self._evo_cache = _load_evo_cache(evo_cache_path)
        self._evo_dim = (len(next(iter(self._evo_cache.values())))
                         if self._evo_cache else 0)

        # Precompute the window index
        self._window_index: List[tuple] = []
        self._build_window_index()

        # statistics
        n_pos_win = sum(1 for _, _, _, lb, _ in self._window_index if lb == 1)
        n_neg_win = len(self._window_index) - n_pos_win
        n_hard_neg = sum(1 for _, _, _, _, st in self._window_index
                         if st == "NEG_HARD")
        print(f"[SlidingWindowContigDataset] "
              f"contigs={len(samples)}, windows={len(self._window_index)}, "
              f"pos={n_pos_win}, neg={n_neg_win}, "
              f"hard_neg={n_hard_neg}, "
              f"window_size={window_size}, overlap={overlap}")

    def __len__(self) -> int:
        return len(self._window_index)

    def _evo_vec(self, seq_id: str) -> np.ndarray:
        """Fetch the Evo2 embedding by seq_id (v7). All windows of the same contig share one vector."""
        if not self._evo_cache:
            return np.zeros(0, dtype=np.float32)
        vec = self._evo_cache.get(seq_id)
        if vec is None:
            return np.zeros(self._evo_dim, dtype=np.float32)
        return np.asarray(vec, dtype=np.float32)

    def __getitem__(self, idx: int) -> HGTSampleTokenized:
        """Fetch a single window sample."""
        sample_idx, win_start, win_end, win_label, win_sample_type = \
            self._window_index[idx]
        sample = self.samples[sample_idx]

        # Extract the window subsequence
        window_seq = sample.sequence[win_start:win_end].upper()
        window_len = len(window_seq)

        # Remap event coordinates to window space
        window_events = self._remap_events(sample.events, win_start, win_end)

        # v7: RC data augmentation (window-level, training only)
        if self.is_training and self.rc_aug_prob > 0 and random.random() < self.rc_aug_prob:
            window_seq = _revcomp(window_seq)
            window_events = _rc_events(window_events, window_len)

        # Tokenize: [CLS] + seq + [SEP]
        input_ids = self.tokenizer.tokenize(window_seq)
        attention_mask = [1] * len(input_ids)

        # Generate targets
        region_targets = self.tokenizer.create_region_targets(
            window_events, window_len)
        breakpoint_targets = self.tokenizer.create_breakpoint_targets(
            window_events, window_len)

        # Length consistency
        assert len(input_ids) == len(region_targets), \
            (f"[{sample.seq_id}] input_ids({len(input_ids)}) "
             f"!= region({len(region_targets)})")
        assert len(input_ids) == len(breakpoint_targets), \
            (f"[{sample.seq_id}] input_ids({len(input_ids)}) "
             f"!= bp({len(breakpoint_targets)})")

        # Auxiliary tags
        valid_events = [e for e in window_events if e.hgt_start >= 0]
        hgt_mode_str = valid_events[0].hgt_mode if valid_events else "none"
        hgt_mode_int = HGT_MODE_MAP.get(hgt_mode_str, 3)
        n_events = len(valid_events)

        # Biological features (based on the window subsequence)
        if self.extract_bio and self.bio_extractor is not None:
            bio_features = self.bio_extractor.extract(window_seq)
        else:
            bio_features = np.zeros(260, dtype=np.float32)

        # Window metadata  —  O(1) lookup (precomputed in _build_window_index)
        meta = self._window_meta.get(sample_idx, {"first_idx": idx, "count": 1})
        total_windows = meta["count"]
        window_idx = idx - meta["first_idx"]

        window_meta = WindowMeta(
            original_seq_id=sample.seq_id,
            window_start=win_start,
            window_end=win_end,
            original_seq_len=sample.seq_len,
            window_idx=window_idx,
            total_windows=total_windows,
        )

        return HGTSampleTokenized(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bio_features=bio_features,
            evo_embeddings=self._evo_vec(sample.seq_id),
            label=win_label,
            region_targets=region_targets,
            breakpoint_targets=breakpoint_targets,
            pe_case=7,  # contigs have no PE case
            hgt_mode=hgt_mode_int,
            n_events=n_events,
            seq_id=sample.seq_id,
            sample_type=win_sample_type,
            window_meta=window_meta,
        )

    # ──────────────────────────────────────────────────────────────────────── #
    #  Window index construction
    # ──────────────────────────────────────────────────────────────────────── #

    def _build_window_index(self):
        """Precompute the window index for all contigs plus per-sample metadata (O(1) lookup)."""
        self._window_index = []
        self._window_meta = {}  # sample_idx → {first_idx, count}
        for sample_idx, sample in enumerate(self.samples):
            first_idx = len(self._window_index)
            windows = self._generate_windows(sample)
            for win_start, win_end, win_label, win_sample_type in windows:
                self._window_index.append(
                    (sample_idx, win_start, win_end, win_label, win_sample_type))
            count = len(self._window_index) - first_idx
            if count > 0:
                self._window_meta[sample_idx] = {"first_idx": first_idx, "count": count}

    def _generate_windows(
        self, sample: HGTSample,
    ) -> List[tuple]:
        """
        Generate the sliding-window list for a single contig.

        Strategy D (training mode):
          - Positive contig: breakpoint-centered window (POS) + randomly sampled hard_neg windows
          - Negative contig: randomly sampled windows (limited by max_neg_windows)

        validation/test/inference mode:
          - Full systematic stride coverage (no sampling limit)

        Returns
        -------
        List of (window_start, window_end, label, sample_type)
        """
        seq_len = sample.seq_len
        events = sample.events if sample.events else []
        valid_events = [e for e in events if e.hgt_start >= 0]

        # Short contig: no splitting
        if seq_len <= self.window_size:
            return [(0, seq_len, sample.label, sample.sample_type)]

        stride = self.window_size - self.overlap

        # ── validation/test/inference: full systematic stride coverage ──
        if not self.is_training:
            return self._generate_full_coverage(
                sample, seq_len, stride, valid_events)

        # ── training mode: strategy D ──
        if sample.label == 1 and valid_events:
            return self._generate_positive_training(
                sample, seq_len, stride, valid_events)
        elif sample.label == 1:
            # Positive contig but no valid events (edge case) → treat as negative
            return self._generate_negative_training(
                sample, seq_len, stride)
        else:
            return self._generate_negative_training(
                sample, seq_len, stride)

    def _generate_full_coverage(
        self, sample, seq_len, stride, valid_events,
    ) -> List[tuple]:
        """
        validation/test/inference: full systematic stride coverage, with a window cap.

        When max_eval_windows is exceeded:
          - Keep all POS windows (ensures breakpoint coverage)
          - Uniformly sample from NEG windows to fill the remaining quota
        """
        starts = list(range(0, seq_len - self.window_size + 1, stride))
        if not starts or starts[-1] + self.window_size < seq_len:
            starts.append(seq_len - self.window_size)
        starts = sorted(set(starts))

        pos_windows = []
        neg_windows = []
        for start in starts:
            end = start + self.window_size
            has_bp = any(e.hgt_start < end and e.hgt_end > start
                         for e in valid_events)
            if has_bp:
                pos_windows.append((start, end, 1, "POS"))
            elif sample.label == 1:
                neg_windows.append((start, end, 0, "NEG_HARD"))
            else:
                neg_windows.append((start, end, 0, sample.sample_type))

        # Window cap sampling
        cap = self.max_eval_windows
        if cap > 0 and len(pos_windows) + len(neg_windows) > cap:
            # Keep all POS windows; uniformly sample from NEG windows
            neg_budget = max(0, cap - len(pos_windows))
            if neg_budget == 0:
                neg_windows = []
            elif len(neg_windows) > neg_budget:
                step = len(neg_windows) / neg_budget
                indices = [int(i * step) for i in range(neg_budget)]
                neg_windows = [neg_windows[i] for i in indices]

        return pos_windows + neg_windows

    def _generate_positive_training(
        self, sample, seq_len, stride, valid_events,
    ) -> List[tuple]:
        """
        Training mode - positive contig: breakpoint-centered windows + limited hard_neg.

        For each breakpoint event:
          1. Generate the breakpoint-centered window (POS)
          2. Additionally offset 1-2 windows for the same event (also POS, for diversity)

        Then randomly sample max_hard_neg non-breakpoint windows as NEG_HARD.
        """
        pos_starts = []
        for event in valid_events:
            # Breakpoint-centered window
            for bp_pos in [event.hgt_start, event.hgt_end]:
                if bp_pos < 0:
                    continue
                center = bp_pos - self.window_size // 2
                center = max(0, min(center, seq_len - self.window_size))
                if center not in pos_starts:
                    pos_starts.append(center)

            # Additional offset windows: increase diversity near the breakpoint
            mid = (event.hgt_start + event.hgt_end) // 2
            for offset in [-stride, stride]:
                s = mid - self.window_size // 2 + offset
                s = max(0, min(s, seq_len - self.window_size))
                end = s + self.window_size
                # Ensure the offset window still covers the breakpoint
                if event.hgt_start < end and event.hgt_end > s:
                    if s not in pos_starts:
                        pos_starts.append(s)

        # Deduplicate + build POS windows
        pos_starts = sorted(set(pos_starts))
        pos_windows = []
        for start in pos_starts:
            end = start + self.window_size
            pos_windows.append((start, end, 1, "POS"))

        # Randomly sample hard_neg: select from non-breakpoint regions
        max_hard_neg = self.max_neg_windows  # Reuse the same parameter
        hard_neg_candidates = []
        # Generate systematic strides, excluding the already-selected POS regions
        all_starts = set(range(0, seq_len - self.window_size + 1, stride))
        if seq_len > self.window_size:
            all_starts.add(seq_len - self.window_size)
        pos_set = set(pos_starts)
        neg_starts = all_starts - pos_set

        for start in neg_starts:
            end = start + self.window_size
            has_bp = any(e.hgt_start < end and e.hgt_end > start
                         for e in valid_events)
            if not has_bp:
                hard_neg_candidates.append((start, end, 0, "NEG_HARD"))

        hard_neg = random.sample(
            hard_neg_candidates,
            min(max_hard_neg, len(hard_neg_candidates)),
        ) if hard_neg_candidates else []

        return pos_windows + hard_neg

    def _generate_negative_training(
        self, sample, seq_len, stride,
    ) -> List[tuple]:
        """Training mode - negative contig: randomly sampled windows."""
        all_starts = list(range(0, seq_len - self.window_size + 1, stride))
        if seq_len > self.window_size:
            all_starts.append(seq_len - self.window_size)
        all_starts = sorted(set(all_starts))

        all_windows = [
            (s, s + self.window_size, 0, sample.sample_type)
            for s in all_starts
        ]

        if len(all_windows) > self.max_neg_windows:
            return random.sample(all_windows, self.max_neg_windows)
        return all_windows

    # ──────────────────────────────────────────────────────────────────────── #
    #  Event coordinate remapping
    # ──────────────────────────────────────────────────────────────────────── #

    @staticmethod
    def _remap_events(
        events: List[BreakpointEvent],
        win_start: int,
        win_end: int,
    ) -> List[BreakpointEvent]:
        """Remap event coordinates from original contig space to window space."""
        window_len = win_end - win_start
        remapped = []
        for e in events:
            new_start = e.hgt_start - win_start
            new_end = e.hgt_end - win_start
            # Keep only events overlapping the window
            if new_end > 0 and new_start < window_len:
                remapped.append(BreakpointEvent(
                    event_idx=e.event_idx,
                    hgt_start=max(0, new_start),
                    hgt_end=min(new_end, window_len),
                    virus_len=e.virus_len,
                    hgt_mode=e.hgt_mode,
                ))
        return remapped
