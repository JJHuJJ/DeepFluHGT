# data_pipeline/contracts.py
"""
DeepFluHGT  —  Data Pipeline data structures
==========================================
Defines all input/output data structures of the data_pipeline module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch


# ═══════════════════════════════════════════════════════════════════════════════ #
#  Level 0: sliding-window coordinate metadata
# ═══════════════════════════════════════════════════════════════════════════════ #

@dataclass
class WindowMeta:
    """
    Sliding-window coordinate metadata.

    Used to map in-window coordinates back to original sequence coordinates
    during training/inference.

    Coordinate mapping formula (1 bp = 1 token):
      original_pos = window_start + (token_idx - 1)   # -1 accounts for CLS

    Property descriptions
    --------
    original_seq_id : original contig sequence ID
    window_start    : window start position in the original sequence (0-based bp)
    window_end      : window end position in the original sequence (half-open)
    original_seq_len: total length of the original contig (bp)
    window_idx      : index of this window within the contig (0-based)
    total_windows   : total number of windows of this contig
    """
    original_seq_id: str = ""
    window_start: int = 0
    window_end: int = 0
    original_seq_len: int = 0
    window_idx: int = 0
    total_windows: int = 0


# ═══════════════════════════════════════════════════════════════════════════════ #
#  Level 1: single breakpoint event (from BED parse)
# ═══════════════════════════════════════════════════════════════════════════════ #

@dataclass
class BreakpointEvent:
    """
    A single HGT breakpoint event.

    Property descriptions
    --------
    event_idx  : event index (0-based, increasing for multi-event samples)
    hgt_start  : start position of the virus region in the sequence (0-based, half-open);
                 set to -1 for negative samples
    hgt_end    : end position of the virus region in the sequence
    virus_len  : length of the inserted virus fragment (bp)
    hgt_mode   : integration mode ("insertion" | "replacement" | "deletion_associated")
    insert_len : total PE insert length (PE mode only)
    r1_vs/r1_ve : coordinate interval of the virus in R1 (PE only, -1 means absent)
    r2_vs/r2_ve : coordinate interval of the virus in R2 (RC coordinate system; PE only)
    pe_case    : PE scenario string (PE only)
    """
    event_idx: int = 0
    hgt_start: int = -1
    hgt_end: int = -1
    virus_len: int = 0
    hgt_mode: str = "none"
    # PE-mode-specific fields
    insert_len: Optional[int] = None
    r1_vs: int = -1
    r1_ve: int = -1
    r2_vs: int = -1
    r2_ve: int = -1
    pe_case: str = ""


# ═══════════════════════════════════════════════════════════════════════════════ #
#  Level 2: single sample (Parser output)
# ═══════════════════════════════════════════════════════════════════════════════ #

@dataclass
class HGTSample:
    """
    Complete parse result for a single sample (from BED + FASTA merge).

    Field descriptions
    --------
    seq_id        : unique sample identifier (e.g. "pos_00000001", "neg_00000005")
    label         : classification label (0=negative, 1=positive)
    sequence      : primary DNA sequence (R1 in PE mode; the full contig in contig mode)
    r2_sequence   : R2-end sequence in PE mode (None in contig mode)
    events        : list of all HGT breakpoint events contained in this sample
    sample_type   : sample type ("POS", "NEG", "NEG_VIRUS", "NEG_HARD", "NEG_PCR")
    seq_len       : primary sequence length (bp)
    """
    seq_id: str = ""
    label: int = 0
    sequence: str = ""
    r2_sequence: Optional[str] = None
    events: List[BreakpointEvent] = field(default_factory=list)
    sample_type: str = "NEG"
    seq_len: int = 0


# ═══════════════════════════════════════════════════════════════════════════════ #
#  Level 3: single tokenized sample (Dataset output)
# ═══════════════════════════════════════════════════════════════════════════════ #

@dataclass
class HGTSampleTokenized:
    """
    Return type of Dataset.__getitem__().

    PE mode: input_ids = [CLS] R1 [SEP] R2 [SEP] (303 tokens for 150bp)
    Contig mode: input_ids = [CLS] contig [SEP]
    """
    # ── Tokenized sequence ──
    input_ids: List[int]                    # [CLS] + bases + [SEP]
    attention_mask: List[int]               # 1=valid, 0=pad

    # ── Biological features ──
    bio_features: np.ndarray = field(
        default_factory=lambda: np.zeros(260, dtype=np.float32))

    # ── v7: Evo2 offline embedding (per whole sequence, precomputed cache, looked up by seq_id) ──
    evo_embeddings: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32))
        # [D_evo] float32; when missing, an empty array (length 0) — collator/model skips fusion accordingly

    # ── Classification label ──
    label: int = 0                          # 0=negative, 1=positive

    # ── Breakpoint targets (single-nucleotide resolution) ──
    region_targets: List[float] = field(default_factory=list)
        # length = number of input_ids; each value ∈ {0.0, 1.0}
        # 1.0 = the base at this position belongs to the virus region
    breakpoint_targets: List[float] = field(default_factory=list)
        # length = number of input_ids; each value ∈ {0.0, 1.0}
        # 1.0 = this position is a virus-host boundary (hgt_start or hgt_end)

    # ── Auxiliary task tags ──
    pe_case: int = 7                        # PE_CASE_MAP encode (default NEG)
    hgt_mode: int = 3                       # HGT_MODE_MAP encode (default none)
    n_events: int = 0                       # total number of HGT events

    # ── Metadata ──
    seq_id: str = ""
    sample_type: str = "NEG"

    # ── Sliding-window metadata (contig mode; non-None when sliding windows are used) ──
    window_meta: Optional[WindowMeta] = None


# ═══════════════════════════════════════════════════════════════════════════════ #
#  Level 4: modelinputbatch (Collator output → modelinput)
# ═══════════════════════════════════════════════════════════════════════════════ #

@dataclass
class HGTBatch:
    """
    Output of the batch collator; the input to the model's forward pass.

    Tensor dimension conventions
    ------------
    B : batch size
    L : sequence token length (including [CLS], [SEP], padding)
    F : biological feature dimension (260)
    """
    # ── Core inputs ──
    input_ids: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long))
        # [B, L]  int64  —  token IDs
    attention_mask: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long))
        # [B, L]  int64  —  1=valid, 0=padding

    # ── Biological features ──
    bio_features: Optional[torch.Tensor] = None
        # [B, F]  float32  —  260-dim global biological features

    # ── v7: Evo2 offline embedding ──
    evo_embeddings: Optional[torch.Tensor] = None
        # [B, D_evo]  float32  —  Evo2 7B precomputed pooled embedding (per seq_id); None=disabled

    # ── Classification label ──
    labels: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long))
        # [B]  int64  —  0=negative, 1=positive

    # ── Breakpoint targets (single-nucleotide resolution) ──
    breakpoint_targets: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.float32))
        # [B, L]  float32  —  boundary-position mask
    region_targets: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.float32))
        # [B, L]  float32  —  virus-region mask

    # ── Auxiliary task tags ──
    # NOTE (BUG-014): In contig mode pe_case is hard-coded to 7 (NEG placeholder value);
    # model forward and loss do not consume this field — it only keeps the HGTBatch shape consistent.
    pe_case: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long))
        # [B]  int64 (PE) or [B, 8] float (Mixup after one-hot)
    hgt_mode: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long))
        # [B]  int64
    n_events: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long))
        # [B]  int64

    # ── Metadata (not involved in gradient computation) ──
    seq_ids: List[str] = field(default_factory=list)
    sample_types: List[str] = field(default_factory=list)

    # ── Sliding-window metadata (contig mode, non-tensor) ──
    window_starts: List[int] = field(default_factory=list)
        # window start bp in the original sequence (0-based), length = B
    window_ends: List[int] = field(default_factory=list)
        # window end bp in the original sequence (half-open), length = B
    original_seq_lens: List[int] = field(default_factory=list)
        # total length of the original contig (bp), length = B
    original_seq_ids: List[str] = field(default_factory=list)
        # original contig ID (may differ from seq_ids), length = B

    def to(self, device: torch.device) -> "HGTBatch":
        """Move all tensors to the specified device."""
        tensor_fields = [
            "input_ids", "attention_mask",
            "labels", "breakpoint_targets", "region_targets",
            "pe_case", "hgt_mode", "n_events",
        ]
        for fname in tensor_fields:
            t = getattr(self, fname)
            if isinstance(t, torch.Tensor) and t.numel() > 0:
                setattr(self, fname, t.to(device))

        optional_fields = ["bio_features", "evo_embeddings"]
        for fname in optional_fields:
            t = getattr(self, fname)
            if isinstance(t, torch.Tensor):
                setattr(self, fname, t.to(device))

        return self
