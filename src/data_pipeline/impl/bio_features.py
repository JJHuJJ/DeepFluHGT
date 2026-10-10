# data_pipeline/impl/bio_features.py
"""
DeepFluHGT  —  biological feature extractor
====================================
260-dim global biological features: 4 global statistics + 256 4-mer frequencies.

Directly adopts the v5.0 BioFeatureExtractor implementation.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict

import numpy as np


class BioFeatureExtractor:
    """
    Global biological feature extractor.

    Output dimension: 4 (global statistics) + 4^k (k-mer frequencies) = 260 (default k=4)

    Global statistics (4 dims):
      1. GC content (0~1)
      2. AT/GC ratio
      3. Shannon entropy (base diversity)
      4. Normalized sequence length log(L+1)/20

    k-mer frequencies (256 dims):
      Normalized 4-mer occurrence frequencies.

    Parameters
    ----------
    kmer_size : int
        k value for k-mer statistics (default 4 → 256 dims)
    """

    _BASE_MAP: Dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3}

    def __init__(self, kmer_size: int = 4):
        self.kmer_size = kmer_size
        self.kmer_dim = 4 ** kmer_size   # 256
        self.feature_dim = 4 + self.kmer_dim  # 260

    def extract(self, sequence: str) -> np.ndarray:
        """
        Extract 260-dim global biological features.

        Parameters
        ----------
        sequence : str
            DNA sequence (uppercase)

        Returns
        -------
        np.ndarray
            shape=(260,), dtype=float32
        """
        seq = sequence.upper()
        L = len(seq)
        if L == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)

        # ── 1. GC content ──
        gc = sum(1 for b in seq if b in "GC") / L

        # ── 2. AT/GC ratio ──
        n_at = sum(1 for b in seq if b in "AT")
        n_gc = max(sum(1 for b in seq if b in "GC"), 1)
        at_gc_ratio = n_at / n_gc

        # ── 3. Shannon entropy ──
        freq = Counter(seq)
        entropy = -sum(
            (c / L) * math.log2(c / L) for c in freq.values() if c > 0
        )

        # ── 4. Normalized sequence length ──
        norm_len = math.log(L + 1) / 20.0

        global_stats = np.array(
            [gc, at_gc_ratio, entropy, norm_len], dtype=np.float32
        )

        # ── 5. k-mer frequencies (O(L) scan) ──
        kmer_counts = np.zeros(self.kmer_dim, dtype=np.float32)
        for i in range(L - self.kmer_size + 1):
            kmer = seq[i: i + self.kmer_size]
            idx = self._kmer_to_idx(kmer)
            if idx >= 0:
                kmer_counts[idx] += 1

        # ── Normalize to frequencies ──
        total = kmer_counts.sum()
        if total > 0:
            kmer_counts /= total

        return np.concatenate([global_stats, kmer_counts])

    def _kmer_to_idx(self, kmer: str) -> int:
        """k-mer → index. Returns -1 if it contains N."""
        idx = 0
        for b in kmer:
            val = self._BASE_MAP.get(b, -1)
            if val < 0:
                return -1
            idx = idx * 4 + val
        return idx
