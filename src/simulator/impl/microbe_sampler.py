# simulator/impl/microbe_sampler.py
"""
Microbe genome sampler
- Lazily loads pyfaidx Fasta handles (LRU cache, avoids opening too many files at once)
- Length-weighted random fragment sampling, O(1) index lookup
- Supports abundance weights (lognormal distribution)
- safe_sample_fragment automatically truncates requests exceeding max_len
"""

import os
import random
import numpy as np
from collections import OrderedDict
from pyfaidx import Fasta

# Default max number of cached handles (avoids exhausting file descriptors)
_DEFAULT_MAX_HANDLES = 512


class MicrobeSampler:
    """
    Iterates over all .fa / .fasta / .fna files under genome_dir,
    builds a sequence index, and supports length-weighted random fragment sampling.
    File handles use a lazy-load + LRU cache strategy.
    """

    def __init__(self, genome_dir: str,
                 use_coverage: bool = False,
                 coverage_scale=None,
                 max_handles: int = _DEFAULT_MAX_HANDLES,
                 allowed_files: list = None):
        self.records: list = [] # (path, seq_id, length)
        self.base_weights: list = []
        self.cover_weights: list = []
        self._fasta_cache: OrderedDict = OrderedDict()
        self._max_handles: int = max_handles
        self.max_len: int = 0
        self.use_coverage = use_coverage
        self._allowed_set: set = set(allowed_files) if allowed_files is not None else None

        # First pass: build the index only (no files kept open)
        for fname in sorted(os.listdir(genome_dir)):
            if not fname.endswith(('.fa', '.fasta', '.fna')):
                continue
            if self._allowed_set is not None and fname not in self._allowed_set:
                continue
            path = os.path.join(genome_dir, fname)
            # pyfaidx builds the index but does not cache handles: close right after use (fault-tolerant: skip corrupted/invalid fasta)
            try:
                fasta = Fasta(path)
            except Exception as _e:
                print(f"[MicrobeSampler] skipping corrupted fasta {fname}: {_e}", flush=True)
                continue
            for seq_id in fasta.keys():
                length = len(fasta[seq_id])
                if length < 10:
                    continue
                self.records.append((path, seq_id, length))
                self.base_weights.append(length)
                if length > self.max_len:
                    self.max_len = length
            # Build then close immediately; reopen lazily later when needed
            fasta.close()

        if not self.records:
            raise RuntimeError(f"No valid sequences found in {genome_dir}")

        if use_coverage:
            if coverage_scale is not None:
                self.cover_weights = list(coverage_scale)
            else:
                lognorm = np.random.lognormal(0.0, 0.5, len(self.records))
                self.cover_weights = [w * f for w, f in
                                      zip(self.base_weights, lognorm)]
        else:
            self.cover_weights = self.base_weights.copy()

        print(f"[MicrobeSampler] Indexed {len(self.records)} sequences, "
              f"max_len={self.max_len:,} bp, "
              f"max_handles={self._max_handles}")

    # -- Lazy load + LRU ---------------------------------------- #

    def _get_fasta(self, path: str) -> Fasta:
        """Get a Fasta handle; the LRU cache evicts the least recently used."""
        if path in self._fasta_cache:
            # Cache hit: move to the end (most recently used)
            self._fasta_cache.move_to_end(path)
            return self._fasta_cache[path]
        # Cache miss: open the file
        fasta = Fasta(path)
        self._fasta_cache[path] = fasta
        self._fasta_cache.move_to_end(path)
        # Evict: when over the limit, close the least recently used
        while len(self._fasta_cache) > self._max_handles:
            oldest_path, oldest_fasta = self._fasta_cache.popitem(last=False)
            oldest_fasta.close()
        return fasta

    # -- sampling ------------------------------------------------ #

    def sample_fragment(self, length: int, use_coverage: bool = True) -> str:
        """
        Randomly extract a `length` bp fragment (returns only the sequence string).
        """
        seq, _ = self.sample_fragment_with_provenance(length, use_coverage)
        return seq

    def sample_fragment_with_provenance(
        self, length: int, use_coverage: bool = True
    ):
        """
        Randomly extract a `length` bp fragment and also return provenance.

        Returns
        -------
        seq : str
        prov : dict with fields:
                 file - source FASTA file path
                 source_seq_id - source sequence ID
                 genome_start - start coordinate in that sequence (0-based)
                 genome_end - end coordinate (half-open interval)
                 strand - always '+'
        """
        if length > self.max_len:
            raise ValueError(
                f"Requested fragment length {length:,} > max available "
                f"{self.max_len:,} bp")
        weights = (self.cover_weights
                   if (self.use_coverage and use_coverage)
                   else self.base_weights)
        # Preflight: check whether any sequence is long enough
        max_available = max(seq_len for _, _, seq_len in self.records)
        if length > max_available:
            raise ValueError(
                f"No sequence long enough for {length:,} bp "
                f"(max available: {max_available:,} bp)")
        attempts = 0
        while True:
            path, seq_id, seq_len = random.choices(
                self.records, weights=weights, k=1)[0]
            if seq_len < length:
                attempts += 1
                if attempts > 10000:
                    raise RuntimeError(
                        f"Failed to find {length:,} bp fragment after 10000 attempts")
                continue
            start = random.randint(0, seq_len - length)
            fasta = self._get_fasta(path)
            seq = str(fasta[seq_id][start:start + length]).upper()
            prov = {
                'file': path,
                'seq_id': seq_id,
                'genome_start': start,
                'genome_end': start + length,
                'strand': '+',
            }
            return seq, prov

    def safe_sample_fragment(self, length: int,
                             use_coverage: bool = True) -> str:
        """
        Same as sample_fragment, but truncates automatically when length > max_len (for very large contigs).
        """
        actual_len = min(length, self.max_len)
        return self.sample_fragment(actual_len, use_coverage=use_coverage)

    def safe_sample_fragment_with_provenance(
        self, length: int, use_coverage: bool = True
    ):
        actual_len = min(length, self.max_len)
        return self.sample_fragment_with_provenance(actual_len,
                                                    use_coverage=use_coverage)
