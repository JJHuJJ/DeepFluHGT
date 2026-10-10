# data_pipeline/impl/parser.py
"""
DeepFluHGT  —  BED / FASTA data parser
============================================
Parses the simulator's breakpoints.bed and FASTA output into a list of HGTSample.

Key design
--------
1. Auto-detect PE (13-column) / Contig (8-column) formats
2. Handle the one-to-many relationship where one contig seq_id contains multiple event_idx entries
3. Preserve negative sample types from the BED (H-01 fix: 10 contig types + 3 PE-specific types;
   unknown NEG_* types are passed through instead of collapsing to "NEG"  —  —  collapsing
   would contaminate the hard/easy stratification)
4. Support both a direct directory and train/val/test subdirectories

BED format reference
------------
PE format (13 columns):
  seq_id | event_idx | label | insert_len | hgt_start | hgt_end |
  virus_len | hgt_mode | pe_case | r1_vs | r1_ve | r2_vs | r2_ve

Contig format (8 columns):
  seq_id | event_idx | label | seq_len | hgt_start | hgt_end |
  virus_len | hgt_mode
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Optional

from Bio import SeqIO

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from schema import PE_CASE_MAP, HGT_MODE_MAP

from data_pipeline.contracts import BreakpointEvent, HGTSample


# ═══════════════════════════════════════════════════════════════════════════════ #
#  Complete set of negative sample types (H-01 fix, CODE_AUDIT_V7)
# ═══════════════════════════════════════════════════════════════════════════════ #
# In contig-mode BEDs the 8th column (hgt_mode) stores neg_type for negative samples;
# likewise the 9th column (pe_case) in PE-mode BEDs.
# Measured (simulated_data/Contig_500k/{train,val,test}/breakpoints.bed 8th column,
#       per-contig counts on 2026-08-20): 10 types, of which 6 were previously
#       collapsed into "NEG" during training
#       (55% of negative contigs: GC_MATCHED_CHIMERIC 29975 / AMALGAMATED_HGT 19956 /
#        PARTIAL_VIRUS_HIT 18063 / CONSERVED 16105 / HOMOLOGY 15816 /
#        LOW_COMPLEXITY 10102).
# PE data (measured on PE_10k) has 3 additional PE-specific types.
# Unknown NEG_* prefixed types are passed through (visible in per_type_counts, not
# placed into hard/easy buckets), preventing another silent collapse if new negative
# types are added in the future.
NEG_TYPE_MAP = {
    # ── 10 shared types (measured from contig BEDs) ──
    "NEG": "NEG",
    "NEG_VIRUS": "NEG_VIRUS",
    "NEG_HARD": "NEG_HARD",
    "NEG_PCR": "NEG_PCR",
    "NEG_CONSERVED": "NEG_CONSERVED",
    "NEG_HOMOLOGY": "NEG_HOMOLOGY",
    "NEG_LOW_COMPLEXITY": "NEG_LOW_COMPLEXITY",
    "NEG_GC_MATCHED_CHIMERIC": "NEG_GC_MATCHED_CHIMERIC",
    "NEG_AMALGAMATED_HGT": "NEG_AMALGAMATED_HGT",
    "NEG_PARTIAL_VIRUS_HIT": "NEG_PARTIAL_VIRUS_HIT",
    # ── 3 PE-specific types (measured from PE_10k BEDs) ──
    "NEG_PE_HOST_DIVERSITY": "NEG_PE_HOST_DIVERSITY",
    "NEG_PE_GC_SHIFT_REGION": "NEG_PE_GC_SHIFT_REGION",
    "NEG_PE_CODON_BOUNDARY": "NEG_PE_CODON_BOUNDARY",
}


# ═══════════════════════════════════════════════════════════════════════════════ #
#  BED parser
# ═══════════════════════════════════════════════════════════════════════════════ #

class BEDParser:
    """
    Parse the simulator's breakpoints.bed file.

    Auto-detects PE (13-column) / Contig (8-column) formats and
    aggregates multiple events of the same seq_id into one record.

    Parameters
    ----------
    bed_path : str
        Path to the breakpoints.bed file
    fasta_dir : str
        Path to the directory containing the FASTA files
    mode : str or None
        "paired_end" or "contig"; None triggers auto-detection
    """

    def __init__(
        self,
        bed_path: str,
        fasta_dir: str,
        mode: Optional[str] = None,
    ):
        assert os.path.isfile(bed_path), f"BED filedoes not exist: {bed_path}"
        assert os.path.isdir(fasta_dir), f"FASTA directorydoes not exist: {fasta_dir}"

        self.bed_path = bed_path
        self.fasta_dir = fasta_dir

        if mode is not None:
            assert mode in ("paired_end", "contig"), \
                f"mode must be 'paired_end' or 'contig', got '{mode}'"
            self.mode = mode
        else:
            self.mode = self._detect_mode(bed_path)

    # ──────────────────────────────────────────────────────────────────────── #
    #  Public interface
    # ──────────────────────────────────────────────────────────────────────── #

    def parse(self) -> List[HGTSample]:
        """parse BED + FASTA, return HGTSample list."""
        if self.mode == "paired_end":
            grouped = self._parse_pe_bed()
        else:
            grouped = self._parse_contig_bed()

        fasta_seqs = self._load_fasta_sequences()
        samples = self._merge(grouped, fasta_seqs)
        self._print_stats(samples)
        return samples

    # ──────────────────────────────────────────────────────────────────────── #
    #  Mode detection
    # ──────────────────────────────────────────────────────────────────────── #

    @staticmethod
    def _detect_mode(bed_path: str) -> str:
        """Determine the data mode automatically from the BED column count."""
        with open(bed_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                n_cols = len(line.split("\t"))
                if n_cols >= 12:
                    return "paired_end"
                elif n_cols >= 7:
                    return "contig"
                else:
                    raise ValueError(
                        f"Unrecognized BED format: n_cols={n_cols}, "
                        f"expected 8 (contig) or 13 (paired_end)")

        raise ValueError(f"BED file is empty or has no valid data rows: {bed_path}")

    # ──────────────────────────────────────────────────────────────────────── #
    #  PE BED parse (13column)
    # ──────────────────────────────────────────────────────────────────────── #

    def _parse_pe_bed(self) -> Dict[str, List[BreakpointEvent]]:
        """
        Parse the PE-mode BED file.

        PE BED (13column):
          seq_id \t event_idx \t label \t insert_len \t hgt_start \t hgt_end
          \t virus_len \t hgt_mode \t pe_case
          \t r1_vs \t r1_ve \t r2_vs \t r2_ve
        """
        grouped: Dict[str, List[BreakpointEvent]] = defaultdict(list)

        with open(self.bed_path, "r") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                cols = line.split("\t")
                if len(cols) < 12:
                    continue

                seq_id    = cols[0]
                event_idx = int(cols[1])
                label     = int(cols[2])
                insert_len = int(cols[3])
                hgt_start = int(cols[4])
                hgt_end   = int(cols[5])
                virus_len = int(cols[6])
                hgt_mode  = cols[7]
                pe_case   = cols[8]
                r1_vs     = int(cols[9])
                r1_ve     = int(cols[10])
                r2_vs     = int(cols[11])
                r2_ve     = int(cols[12]) if len(cols) > 12 else -1

                event = BreakpointEvent(
                    event_idx=event_idx,
                    hgt_start=hgt_start,
                    hgt_end=hgt_end,
                    virus_len=virus_len,
                    hgt_mode=hgt_mode,
                    insert_len=insert_len,
                    r1_vs=r1_vs,
                    r1_ve=r1_ve,
                    r2_vs=r2_vs,
                    r2_ve=r2_ve,
                    pe_case=pe_case,
                )
                grouped[seq_id].append(event)

        return dict(grouped)

    # ──────────────────────────────────────────────────────────────────────── #
    #  Contig BED parse (8column)
    # ──────────────────────────────────────────────────────────────────────── #

    def _parse_contig_bed(self) -> Dict[str, List[BreakpointEvent]]:
        """
        Parse the contig-mode BED file (8 columns).

        Contig BED:
          seq_id \t event_idx \t label \t seq_len \t hgt_start \t hgt_end
          \t virus_len \t hgt_mode

        For negative samples the hgt_mode column stores the neg_type.
        """
        grouped: Dict[str, List[BreakpointEvent]] = defaultdict(list)

        with open(self.bed_path, "r") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                cols = line.split("\t")
                if len(cols) < 7:
                    continue

                seq_id    = cols[0]
                event_idx = int(cols[1])
                label     = int(cols[2])
                seq_len   = int(cols[3])
                hgt_start = int(cols[4])
                hgt_end   = int(cols[5])
                virus_len = int(cols[6])
                hgt_mode  = cols[7] if len(cols) > 7 else "none"

                event = BreakpointEvent(
                    event_idx=event_idx,
                    hgt_start=hgt_start,
                    hgt_end=hgt_end,
                    virus_len=virus_len,
                    hgt_mode=hgt_mode,
                )
                grouped[seq_id].append(event)

        return dict(grouped)

    # ──────────────────────────────────────────────────────────────────────── #
    #  FASTA load
    # ──────────────────────────────────────────────────────────────────────── #

    def _load_fasta_sequences(self) -> Dict[str, Dict[str, str]]:
        """
        Load all sequences from the FASTA files.

        Returns
        -------
        PE: {"R1": {id: seq}, "R2": {id: seq}}
        Contig: {"main": {id: seq}}
        """
        result: Dict[str, Dict[str, str]] = {}

        if self.mode == "paired_end":
            r1_path = os.path.join(self.fasta_dir, "R1.fasta")
            r2_path = os.path.join(self.fasta_dir, "R2.fasta")
            assert os.path.isfile(r1_path), f"R1.fasta does not exist: {r1_path}"
            assert os.path.isfile(r2_path), f"R2.fasta does not exist: {r2_path}"

            result["R1"] = self._read_fasta(r1_path)
            result["R2"] = self._read_fasta(r2_path)
        else:
            ds_path = os.path.join(self.fasta_dir, "dataset.fasta")
            assert os.path.isfile(ds_path), f"dataset.fasta does not exist: {ds_path}"
            result["main"] = self._read_fasta(ds_path)

        return result

    @staticmethod
    def _read_fasta(path: str) -> Dict[str, str]:
        """read FASTA file, return {seq_id: sequence}."""
        seqs: Dict[str, str] = {}
        for rec in SeqIO.parse(path, "fasta"):
            seqs[rec.id] = str(rec.seq).upper()
        return seqs

    # ──────────────────────────────────────────────────────────────────────── #
    #  merge BED + FASTA → HGTSample
    # ──────────────────────────────────────────────────────────────────────── #

    def _merge(
        self,
        grouped: Dict[str, List[BreakpointEvent]],
        fasta_seqs: Dict[str, Dict[str, str]],
    ) -> List[HGTSample]:
        """Merge grouped events with FASTA sequences into a list of HGTSample."""
        samples: List[HGTSample] = []

        if self.mode == "paired_end":
            r1_seqs = fasta_seqs.get("R1", {})
            r2_seqs = fasta_seqs.get("R2", {})
        else:
            main_seqs = fasta_seqs.get("main", {})

        for seq_id, events in grouped.items():
            label = 1 if events[0].hgt_start >= 0 else 0

            if self.mode == "paired_end":
                sequence = r1_seqs.get(seq_id, "")
                r2_seq = r2_seqs.get(seq_id, None)
            else:
                sequence = main_seqs.get(seq_id, "")
                r2_seq = None

            if not sequence:
                continue

            sample_type = "POS" if label == 1 else \
                self._infer_neg_type(seq_id, events)

            samples.append(HGTSample(
                seq_id=seq_id,
                label=label,
                sequence=sequence,
                r2_sequence=r2_seq,
                events=events,
                sample_type=sample_type,
                seq_len=len(sequence),
            ))

        return samples

    # ──────────────────────────────────────────────────────────────────────── #
    #  Utility methods
    # ──────────────────────────────────────────────────────────────────────── #

    def _infer_neg_type(
        self, seq_id: str, events: List[BreakpointEvent]
    ) -> str:
        """Infer the negative sample type (H-01 fix: all types preserved; see the module-level NEG_TYPE_MAP).

        Contig mode reads events[0].hgt_mode (BED column 8),
        PE mode reads events[0].pe_case (BED column 9)  —  —  negative-sample rows store neg_type in that column.
        Unknown NEG_* prefixed types are passed through; other unknown values keep the old behavior of falling back to "NEG".
        """
        if self.mode == "paired_end" and events:
            raw = events[0].pe_case
        elif self.mode == "contig" and events:
            raw = events[0].hgt_mode
        else:
            return "NEG"

        if raw in NEG_TYPE_MAP:
            return NEG_TYPE_MAP[raw]
        if raw.startswith("NEG"):
            return raw
        return "NEG"

    @staticmethod
    def _print_stats(samples: List[HGTSample]) -> None:
        """Print parse statistics."""
        n_pos = sum(1 for s in samples if s.label == 1)
        n_neg = sum(1 for s in samples if s.label == 0)
        event_counts = defaultdict(int)
        for s in samples:
            if s.label == 1:
                event_counts[len(s.events)] += 1

        print(f"[BEDParser] mode=parsed, total={len(samples)}, "
              f"pos={n_pos}, neg={n_neg}")


# ═══════════════════════════════════════════════════════════════════════════════ #
#  Path-resolution helper
# ═══════════════════════════════════════════════════════════════════════════════ #

def resolve_data_paths(data_path: str) -> Dict[str, str]:
    """
    Resolve the data path, returning a dict with bed_path and fasta_dir.

    Supports three layout forms:
      1. Direct output directory: data_path/ containing breakpoints.bed + FASTA files
      2. Split subdirectory: data_path/train/ (or val/, test/)
      3. Split parent directory: data_path/ containing train/, val/, test/ subdirectories

    Returns
    -------
    dict with keys: bed_path, fasta_dir, mode
    """
    bed_path = os.path.join(data_path, "breakpoints.bed")

    if os.path.isfile(bed_path):
        # Direct directory
        return {
            "bed_path": bed_path,
            "fasta_dir": data_path,
        }

    # Check the train subdirectory
    train_bed = os.path.join(data_path, "train", "breakpoints.bed")
    if os.path.isfile(train_bed):
        return {
            "bed_path": train_bed,
            "fasta_dir": os.path.join(data_path, "train"),
        }

    raise FileNotFoundError(
        f"breakpoints.bed not found: tried {bed_path} and {train_bed}")
