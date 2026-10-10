# data_pipeline/impl/input_reader.py
"""
DeepFluHGT  —  multi-format input reader
====================================
Supports multiple sequence file formats for prediction mode:
  - PE FASTQ compressed:     R1.fastq.gz + R2.fastq.gz
  - PE FASTQ uncompressed:   R1.fastq + R2.fastq
  - PE FASTA compressed:     R1.fasta.gz + R2.fasta.gz
  - PE FASTA uncompressed:   R1.fasta + R2.fasta
  - Contig FASTA uncompressed: .fasta / .fa
  - Contig FASTA compressed:   .fasta.gz / .fa.gz
  - Contig FASTQ uncompressed: .fastq / .fq
  - Contig FASTQ compressed:   .fastq.gz / .fq.gz
"""

from __future__ import annotations

import gzip
import os
from typing import Dict, Generator, List, Optional, Tuple

from data_pipeline.contracts import HGTSample, BreakpointEvent


# ═══════════════════════════════════════════════════════════════════════════════ #
#  Helper functions
# ═══════════════════════════════════════════════════════════════════════════════ #

def _open_maybe_gz(path: str):
    """Auto-detect gzip and open the file accordingly."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def _detect_mode(input_path: str) -> str:
    """
    Auto-detect the input mode (paired_end / contig).

    Rules:
      - If the input is a directory containing R1.* files → paired_end
      - If the input is a directory containing dataset.fasta → contig
      - If the input is a file → decide by filename (R1 → PE, otherwise contig)
    """
    if os.path.isdir(input_path):
        for name in os.listdir(input_path):
            if name.startswith("R1"):
                return "paired_end"
        return "contig"

    # filepath
    basename = os.path.basename(input_path)
    if "R1" in basename:
        return "paired_end"
    return "contig"


def _detect_read_len(path: str, is_fastq: bool) -> int:
    """Read the first sequence to obtain the read length."""
    f = _open_maybe_gz(path)
    try:
        if is_fastq:
            # FASTQ: @header\nSEQ\n+\nQUAL
            next(f)  # @header
            seq = next(f).strip()
            return len(seq)
        else:
            # FASTA: >header\nSEQ
            next(f)  # >header
            seq = next(f).strip()
            return len(seq)
    finally:
        f.close()


# ═══════════════════════════════════════════════════════════════════════════════ #
#  FASTA read
# ═══════════════════════════════════════════════════════════════════════════════ #

def read_fasta(path: str) -> Generator[Tuple[str, str], None, None]:
    """
    Read a FASTA file (.gz supported).

    Yields
    ------
    (seq_id, sequence) : Tuple[str, str]
    """
    f = _open_maybe_gz(path)
    try:
        header = ""
        seq_parts = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header:
                    yield header[1:].split()[0], "".join(seq_parts)
                header = line
                seq_parts = []
            else:
                seq_parts.append(line.upper())
        if header:
            yield header[1:].split()[0], "".join(seq_parts)
    finally:
        f.close()


# ═══════════════════════════════════════════════════════════════════════════════ #
#  FASTQ read
# ═══════════════════════════════════════════════════════════════════════════════ #

def read_fastq(path: str) -> Generator[Tuple[str, str, str], None, None]:
    """
    Read a FASTQ file (.gz supported).

    Yields
    ------
    (seq_id, sequence, quality) : Tuple[str, str, str]
    """
    f = _open_maybe_gz(path)
    try:
        while True:
            header = f.readline()
            if not header:
                break
            header = header.strip()
            if not header or not header.startswith("@"):
                continue
            seq = f.readline().strip().upper()
            f.readline()  # +
            qual = f.readline().strip()
            seq_id = header[1:].split()[0]
            yield seq_id, seq, qual
    finally:
        f.close()


# ═══════════════════════════════════════════════════════════════════════════════ #
#  Input path resolution
# ═══════════════════════════════════════════════════════════════════════════════ #

class InputSpec:
    """Parsed input file specification."""

    def __init__(
        self,
        mode: str,           # "paired_end" | "contig"
        r1_path: str = "",   # PE: R1 filepath
        r2_path: str = "",   # PE: R2 filepath
        seq_path: str = "",  # Contig: sequencefilepath
        read_len: int = 150,
    ):
        self.mode = mode
        self.r1_path = r1_path
        self.r2_path = r2_path
        self.seq_path = seq_path
        self.read_len = read_len

    def is_fastq(self) -> bool:
        """Determine whether the format is FASTQ."""
        if self.mode == "paired_end":
            return self.r1_path.endswith(".fastq") or self.r1_path.endswith(".fq") or \
                self.r1_path.endswith(".fastq.gz") or self.r1_path.endswith(".fq.gz")
        return self.seq_path.endswith(".fastq") or self.seq_path.endswith(".fq") or \
            self.seq_path.endswith(".fastq.gz") or self.seq_path.endswith(".fq.gz")

    def __repr__(self):
        if self.mode == "paired_end":
            return f"InputSpec(PE, R1={self.r1_path}, R2={self.r2_path})"
        return f"InputSpec(contig, path={self.seq_path})"


def resolve_input(
    input_path: str,
    mode: str = "auto",
    read_len: int = 0,
) -> InputSpec:
    """
    Resolve the input path, auto-detecting the format.

    Supported inputs:
      1. Directory: containing R1/R2 files (PE) or a dataset.* file (contig)
      2. Single file: FASTA/FASTQ in contig mode
      3. PE pair: two files (R1, R2) separated by a comma

    Parameters
    ----------
    input_path : str
        Input path (directory, file, or comma-separated R1,R2)
    mode : str
        "auto", "paired_end", "contig"
    read_len : int
        PE read length (0 = auto-detect)

    Returns
    -------
    InputSpec
    """
    detected_mode = mode if mode != "auto" else _detect_mode(input_path)

    # ── PE mode: comma-separated R1,R2 ──
    if "," in input_path:
        parts = input_path.split(",")
        r1_path, r2_path = parts[0].strip(), parts[1].strip()
        if not os.path.isfile(r1_path) or not os.path.isfile(r2_path):
            raise FileNotFoundError(f"PE files not found: {r1_path}, {r2_path}")
        rl = read_len if read_len > 0 else _detect_read_len(r1_path,
            r1_path.endswith(".fastq") or r1_path.endswith(".fq") or
            r1_path.endswith(".fastq.gz") or r1_path.endswith(".fq.gz"))
        return InputSpec("paired_end", r1_path=r1_path, r2_path=r2_path, read_len=rl)

    # ── PE mode: directory ──
    if detected_mode == "paired_end" and os.path.isdir(input_path):
        r1_path, r2_path = _find_pe_files(input_path)
        rl = read_len if read_len > 0 else _detect_read_len(r1_path,
            r1_path.endswith(".fastq") or r1_path.endswith(".fq") or
            r1_path.endswith(".fastq.gz") or r1_path.endswith(".fq.gz"))
        return InputSpec("paired_end", r1_path=r1_path, r2_path=r2_path, read_len=rl)

    # ── Contig mode: single file ──
    if os.path.isfile(input_path):
        return InputSpec("contig", seq_path=input_path, read_len=read_len or 150)

    # ── Contig mode: directory ──
    if os.path.isdir(input_path):
        seq_path = _find_contig_file(input_path)
        return InputSpec("contig", seq_path=seq_path, read_len=read_len or 150)

    raise FileNotFoundError(f"Cannot parse input: {input_path}")


def _find_pe_files(directory: str) -> Tuple[str, str]:
    """Find the R1/R2 file pair in the directory."""
    candidates_r1 = []
    candidates_r2 = []
    for name in os.listdir(directory):
        if name.startswith("R1"):
            candidates_r1.append(os.path.join(directory, name))
        elif name.startswith("R2"):
            candidates_r2.append(os.path.join(directory, name))

    # Priority: .fastq.gz > .fastq > .fasta.gz > .fasta
    priority = [".fastq.gz", ".fq.gz", ".fastq", ".fq", ".fasta.gz", ".fa.gz", ".fasta", ".fa"]

    def _pick(candidates, priority_list):
        for ext in priority_list:
            for c in candidates:
                if c.endswith(ext):
                    return c
        return candidates[0] if candidates else None

    r1 = _pick(candidates_r1, priority)
    r2 = _pick(candidates_r2, priority)

    if not r1 or not r2:
        raise FileNotFoundError(
            f"PE file pair not found in {directory}: "
            f"R1={candidates_r1}, R2={candidates_r2}")

    return r1, r2


def _find_contig_file(directory: str) -> str:
    """Find the contig sequence file in the directory."""
    priority = [
        "dataset.fasta", "dataset.fa",
        "dataset.fasta.gz", "dataset.fa.gz",
        "dataset.fastq", "dataset.fq",
        "dataset.fastq.gz", "dataset.fq.gz",
    ]
    for name in priority:
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            return path

    # Try any fasta/fastq
    for name in sorted(os.listdir(directory)):
        for ext in [".fasta", ".fa", ".fasta.gz", ".fa.gz",
                    ".fastq", ".fq", ".fastq.gz", ".fq.gz"]:
            if name.endswith(ext):
                return os.path.join(directory, name)

    raise FileNotFoundError(f"No sequence file found in {directory}")


# ═══════════════════════════════════════════════════════════════════════════════ #
#  Unified read interface
# ═══════════════════════════════════════════════════════════════════════════════ #

def read_sequences(
    input_path: str,
    mode: str = "auto",
    read_len: int = 0,
) -> Tuple[List[HGTSample], str]:
    """
    Unified reader for input sequence files; returns a list of HGTSample.

    Parameters
    ----------
    input_path : str
        Input path (directory, file, or comma-separated R1,R2)
    mode : str
        "auto", "paired_end", "contig"
    read_len : int
        PE read length (0 = auto-detect)

    Returns
    -------
    (samples, mode) : Tuple[List[HGTSample], str]
        samples: unlabeled HGTSample list
        mode: "paired_end" or "contig"
    """
    spec = resolve_input(input_path, mode, read_len)
    samples = []

    if spec.mode == "paired_end":
        is_fq = spec.is_fastq()
        if is_fq:
            r1_iter = read_fastq(spec.r1_path)
            r2_iter = read_fastq(spec.r2_path)
            for (r1_id, r1_seq, _), (r2_id, r2_seq, _) in zip(r1_iter, r2_iter):
                # Use the R1 ID as seq_id
                seq_id = r1_id
                samples.append(HGTSample(
                    seq_id=seq_id,
                    label=-1,  # unknown label
                    sequence=r1_seq,
                    r2_sequence=r2_seq,
                    events=[],
                    sample_type="UNKNOWN",
                    seq_len=len(r1_seq),
                ))
        else:
            r1_iter = read_fasta(spec.r1_path)
            r2_iter = read_fasta(spec.r2_path)
            for (r1_id, r1_seq), (r2_id, r2_seq) in zip(r1_iter, r2_iter):
                seq_id = r1_id
                samples.append(HGTSample(
                    seq_id=seq_id,
                    label=-1,
                    sequence=r1_seq,
                    r2_sequence=r2_seq,
                    events=[],
                    sample_type="UNKNOWN",
                    seq_len=len(r1_seq),
                ))
    else:
        # Contig mode
        is_fq = spec.is_fastq()
        if is_fq:
            for seq_id, seq, _ in read_fastq(spec.seq_path):
                samples.append(HGTSample(
                    seq_id=seq_id,
                    label=-1,
                    sequence=seq,
                    events=[],
                    sample_type="UNKNOWN",
                    seq_len=len(seq),
                ))
        else:
            for seq_id, seq in read_fasta(spec.seq_path):
                samples.append(HGTSample(
                    seq_id=seq_id,
                    label=-1,
                    sequence=seq,
                    events=[],
                    sample_type="UNKNOWN",
                    seq_len=len(seq),
                ))

    print(f"[input_reader] mode={spec.mode}, n_samples={len(samples)}")
    return samples, spec.mode
