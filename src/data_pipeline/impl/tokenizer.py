# data_pipeline/impl/tokenizer.py
"""
DeepFluHGT  —  unified single-nucleotide tokenizer
============================================
Encodes each base of a DNA sequence as an independent token,
so that the NTv3 and Caduceus models share the same tokenization.

Token layout
----------
PE mode   : [CLS] R1[0..L1-1] [SEP] R2[0..L2-1] [SEP]
Contig    : [CLS] seq[0..L-1] [SEP]

Vocab (canonical, internal to the data pipeline)
  PAD=0, CLS=1, SEP=2, MASK=3, UNK=4, A=5, T=6, C=7, G=8, N=9

The model layer maps canonical IDs to model-specific IDs via REMAP tables.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from data_pipeline.contracts import BreakpointEvent


class SingleNucleotideTokenizer:
    """Unified single-nucleotide DNA tokenizer."""

    # ── Canonical vocab ──
    PAD_ID: int = 0
    CLS_ID: int = 1
    SEP_ID: int = 2
    MASK_ID: int = 3
    UNK_ID: int = 4

    A_ID: int = 5
    T_ID: int = 6
    C_ID: int = 7
    G_ID: int = 8
    N_ID: int = 9

    VOCAB_SIZE: int = 10

    _CHAR_MAP: Dict[str, int] = {
        'A': 5, 'T': 6, 'C': 7, 'G': 8, 'N': 9,
    }

    # ── Model-specific remapping tables ──
    # canonical_id → model_id
    NTV3_REMAP: Dict[int, int] = {
        0: 1, 1: 3, 2: 4, 3: 2, 4: 0,
        5: 6, 6: 7, 7: 8, 8: 9, 9: 10,
    }
    CADUCEUS_REMAP: Dict[int, int] = {
        0: 4, 1: 0, 2: 1, 3: 3, 4: 6,
        5: 7, 6: 10, 7: 8, 8: 9, 9: 11,
    }

    # ──────────────────────────────────────────────────────────────────────── #
    #  Tokenize
    # ──────────────────────────────────────────────────────────────────────── #

    def tokenize(self, sequence: str) -> List[int]:
        """
        Tokenize a DNA sequence into a list of token IDs.

        Layout: [CLS] + base_tokens + [SEP]

        Parameters
        ----------
        sequence : str
            DNA sequence (may contain N)

        Returns
        -------
        List[int]
            token ID list, length = len(sequence) + 2
        """
        tokens = [self.CLS_ID]
        for ch in sequence.upper():
            tokens.append(self._CHAR_MAP.get(ch, self.UNK_ID))
        tokens.append(self.SEP_ID)
        return tokens

    def tokenize_pe(self, r1: str, r2: str) -> List[int]:
        """
        PE mode tokenization: [CLS] R1 [SEP] R2 [SEP]

        Parameters
        ----------
        r1 : str
            R1 read sequence
        r2 : str
            R2 read sequence

        Returns
        -------
        List[int]
            token ID list, length = len(r1) + len(r2) + 3
        """
        tokens = [self.CLS_ID]
        for ch in r1.upper():
            tokens.append(self._CHAR_MAP.get(ch, self.UNK_ID))
        tokens.append(self.SEP_ID)
        for ch in r2.upper():
            tokens.append(self._CHAR_MAP.get(ch, self.UNK_ID))
        tokens.append(self.SEP_ID)
        return tokens

    # ──────────────────────────────────────────────────────────────────────── #
    #  Target generation
    # ──────────────────────────────────────────────────────────────────────── #

    def create_region_targets(
        self,
        events: List[BreakpointEvent],
        seq_len: int,
    ) -> List[float]:
        """
        Create a per-base binary mask of the virus region.

        Layout: [CLS]=0 + base_targets[0..seq_len-1] + [SEP]=0

        Parameters
        ----------
        events : List[BreakpointEvent]
            List of breakpoint events
        seq_len : int
            Sequence length (bp)

        Returns
        -------
        List[float]
            length = seq_len + 2, value ∈ {0.0, 1.0}
        """
        full_len = seq_len + 2  # CLS + bases + SEP
        targets = [0.0] * full_len

        for event in events:
            if event.hgt_start < 0 or event.hgt_end < 0:
                continue
            s = max(0, event.hgt_start)
            e = min(event.hgt_end, seq_len)
            for i in range(s, e):
                targets[i + 1] = 1.0  # +1 skip [CLS]

        return targets

    def create_breakpoint_targets(
        self,
        events: List[BreakpointEvent],
        seq_len: int,
    ) -> List[float]:
        """
        Create a per-base binary mask of breakpoint boundaries.

        Marks 1.0 at every virus-host boundary position:
          - hgt_start (if > 0)
          - hgt_end (if < seq_len and > 0)

        Layout: [CLS]=0 + base_targets[0..seq_len-1] + [SEP]=0

        Parameters
        ----------
        events : List[BreakpointEvent]
            List of breakpoint events
        seq_len : int
            Sequence length (bp)

        Returns
        -------
        List[float]
            length = seq_len + 2
        """
        full_len = seq_len + 2
        targets = [0.0] * full_len

        for event in events:
            if event.hgt_start < 0 or event.hgt_end < 0:
                continue
            # Left boundary (host→virus transition)
            if 0 < event.hgt_start < seq_len:
                targets[event.hgt_start + 1] = 1.0
            # Right boundary (virus→host transition)
            if 0 < event.hgt_end <= seq_len:
                targets[event.hgt_end + 1 - 1] = 1.0  # hgt_end-1 position

        return targets

    def create_pe_targets(
        self,
        events: List[BreakpointEvent],
        r1_len: int,
        r2_len: int,
    ) -> Tuple[List[float], List[float]]:
        """
        Generate full region_targets and breakpoint_targets for PE mode.

        Layout: [CLS]=0 R1[0..r1_len-1] [SEP]=0 R2[0..r2_len-1] [SEP]=0
        Total length = r1_len + r2_len + 3

        Coordinates: only read-space coordinates (r1_vs/r1_ve, r2_vs/r2_ve) are used.
        BUG-008 fix: the redundant insert-space coordinate path (hgt_start/hgt_end)
        was removed, avoiding duplicate breakpoint marks from two coordinate
        systems labeling the same position.

        Parameters
        ----------
        events : List[BreakpointEvent]
            List of breakpoint events (usually just 1)
        r1_len : int
            R1 read length (bp)
        r2_len : int
            R2 read length (bp)

        Returns
        -------
        region_targets : List[float]
        breakpoint_targets : List[float]
        """
        total_len = r1_len + r2_len + 3  # CLS + R1 + SEP + R2 + SEP
        region = [0.0] * total_len
        bp = [0.0] * total_len

        # [CLS] at index 0 → 0.0
        # R1 at indices [1, 1+r1_len)
        # [SEP] at index 1+r1_len → 0.0
        # R2 at indices [2+r1_len, 2+r1_len+r2_len)
        # [SEP] at index 2+r1_len+r2_len → 0.0

        r1_offset = 1       # start offset of R1 in the token sequence
        r2_offset = 2 + r1_len  # start offset of R2 in the token sequence

        for event in events:
            # ── R1 region targets ──
            if event.r1_vs >= 0 and event.r1_ve >= 0:
                s = max(0, event.r1_vs)
                e = min(event.r1_ve, r1_len)
                for i in range(s, e):
                    region[r1_offset + i] = 1.0

                # R1 breakpoint targets
                if s > 0:
                    bp[r1_offset + s] = 1.0
                if e < r1_len and e > 0:
                    bp[r1_offset + e - 1] = 1.0

            # ── R2 region targets ──
            # r2_vs/r2_ve are positions in the R2 RC coordinate system
            if event.r2_vs >= 0 and event.r2_ve >= 0:
                s = max(0, event.r2_vs)
                e = min(event.r2_ve, r2_len)
                for i in range(s, e):
                    region[r2_offset + i] = 1.0

                # R2 breakpoint targets
                if s > 0:
                    bp[r2_offset + s] = 1.0
                if e < r2_len and e > 0:
                    bp[r2_offset + e - 1] = 1.0

            # ── Block 2 deleted (BUG-008 fix) ──
            # It originally used hgt_start/hgt_end (insert coordinate system) mapped back
            # to read space, which marked the same physical positions as Block 1
            # (r1_vs/r1_ve, r2_vs/r2_ve, directly in read space), causing duplicate
            # breakpoint annotations and possible offsets in RC-flip edge cases.

        return region, bp

    # ──────────────────────────────────────────────────────────────────────── #
    #  Decode (invertibility validation)
    # ──────────────────────────────────────────────────────────────────────── #

    _ID_TO_CHAR: Dict[int, str] = {
        0: '<PAD>', 1: '<CLS>', 2: '<SEP>', 3: '<MASK>', 4: '<UNK>',
        5: 'A', 6: 'T', 7: 'C', 8: 'G', 9: 'N',
    }

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """
        Decode a list of token IDs into a DNA sequence string.

        Parameters
        ----------
        token_ids : List[int]
            List of token IDs
        skip_special : bool
            Whether to skip special tokens (CLS, SEP, PAD, etc.)

        Returns
        -------
        str
            Decoded sequence (uppercase)
        """
        special_ids = {self.PAD_ID, self.CLS_ID, self.SEP_ID, self.MASK_ID}
        result = []
        for tid in token_ids:
            if skip_special and tid in special_ids:
                continue
            result.append(self._ID_TO_CHAR.get(tid, '<UNK>'))
        return ''.join(result)

    def decode_pe(self, token_ids: List[int], r1_len: int) -> Tuple[str, str]:
        """
        decode PE token sequencefor (R1, R2).

        Parameters
        ----------
        token_ids : List[int]
            PE token sequence [CLS] R1 [SEP] R2 [SEP]
        r1_len : int
            R1 length (bp)

        Returns
        -------
        Tuple[str, str]
            (r1_sequence, r2_sequence)
        """
        # R1: indices [1, 1+r1_len)
        r1_ids = token_ids[1:1 + r1_len]
        # R2: indices [2+r1_len, 2+r1_len+r2_len)
        # r2_len = len(token_ids) - r1_len - 3
        r2_start = 2 + r1_len
        r2_ids = token_ids[r2_start:-1]  # excluding the final SEP

        r1 = ''.join(self._ID_TO_CHAR.get(t, 'N') for t in r1_ids)
        r2 = ''.join(self._ID_TO_CHAR.get(t, 'N') for t in r2_ids)
        return r1, r2

    # ──────────────────────────────────────────────────────────────────────── #
    #  Helper methods
    # ──────────────────────────────────────────────────────────────────────── #

    @classmethod
    def remap_to_ntv3(cls, token_ids: List[int]) -> List[int]:
        """Map canonical token IDs to NTv3 model IDs."""
        return [cls.NTV3_REMAP.get(t, 0) for t in token_ids]

    @classmethod
    def remap_to_caduceus(cls, token_ids: List[int]) -> List[int]:
        """Map canonical token IDs to Caduceus model IDs."""
        return [cls.CADUCEUS_REMAP.get(t, 4) for t in token_ids]
