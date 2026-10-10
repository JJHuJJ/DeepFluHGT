# data_pipeline/api.py
"""
DeepFluHGT  —  Data Pipeline external interface
==========================================
The single entry point used across modules.

API
---
load_dataset(path) -> HGTDataset
collate(batch) -> HGTBatch
test() -> bool
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.contracts import (
    HGTSample, HGTSampleTokenized, HGTBatch,
)
from data_pipeline.impl.parser import BEDParser, resolve_data_paths
from data_pipeline.impl.dataset import HGTDataset, SlidingWindowContigDataset
from data_pipeline.impl.collator import HGTCollator


def _resolve_evo_cache(evo_cache_path: Optional[str], split_name: str = None) -> Optional[str]:
    """Resolve the (per-split) Evo2 embedding cache path (v7).

    - contains '{split}' placeholder → formatted with the split (e.g. 'evo_cache/evo_{split}.pt')
    - is a directory → '{dir}/{split}.pt'
    - otherwise → returned as-is (single dataset, used by load_dataset)
    """
    if not evo_cache_path:
        return None
    if split_name and "{split}" in evo_cache_path:
        return evo_cache_path.format(split=split_name)
    if split_name and os.path.isdir(evo_cache_path):
        return os.path.join(evo_cache_path, f"{split_name}.pt")
    return evo_cache_path


def load_dataset(
    data_path: str,
    mode: str = "auto",
    max_length: int = 10000,
    read_len: int = 150,
    extract_bio: bool = True,
    evo_cache_path: Optional[str] = None,
    rc_aug_prob: float = 0.0,
) -> HGTDataset:
    """
    Load simulator output as an HGTDataset.

    Parameters
    ----------
    data_path : str
        Simulator output directory (containing breakpoints.bed and FASTA files)
        or a split subdirectory (e.g. train/)
    mode : str
        "paired_end", "contig", or "auto" (auto-detected)
    max_length : int
        Maximum sequence length (bp); longer contigs are truncated in contig mode
    read_len : int
        PE read length (default 150)
    extract_bio : bool
        Whether to extract biological features

    Returns
    -------
    HGTDataset
    """
    paths = resolve_data_paths(data_path)

    parser = BEDParser(
        bed_path=paths["bed_path"],
        fasta_dir=paths["fasta_dir"],
        mode=None if mode == "auto" else mode,
    )

    samples = parser.parse()
    actual_mode = parser.mode

    return HGTDataset(
        samples=samples,
        mode=actual_mode,
        max_length=max_length,
        read_len=read_len,
        extract_bio=extract_bio,
        evo_cache_path=_resolve_evo_cache(evo_cache_path),
        rc_aug_prob=rc_aug_prob,
    )


def load_splits(
    data_dir: str,
    mode: str = "auto",
    max_length: int = 10000,
    read_len: int = 150,
    extract_bio: bool = True,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    evo_cache_path: Optional[str] = None,
    rc_aug_prob: float = 0.0,
) -> dict:
    """
    Auto-detect train/val/test subdirectories and load the data splits.

    Logic:
      1. if data_dir/train/breakpoints.bed exists:
         → load train/, val/, test/ each as its own HGTDataset
      2. if data_dir/breakpoints.bed exists (no subdirectories):
         → persistently split to disk following the v5.0 split_dataset.py logic,
           then load from the split output directory

    Parameters
    ----------
    data_dir : str
        Data root directory
    mode : str
        "paired_end", "contig", "auto"
    max_length : int
        Maximum sequence length
    read_len : int
        PE read length
    extract_bio : bool
        Whether to extract biological features
    train_ratio : float
        Training-set ratio (only used for automatic splitting)
    val_ratio : float
        Validation-set ratio (only used for automatic splitting)
    seed : int
        Random seed for splitting

    Returns
    -------
    dict: {"train": HGTDataset, "val": HGTDataset, "test": HGTDataset}
    """
    train_bed = os.path.join(data_dir, "train", "breakpoints.bed")
    val_bed = os.path.join(data_dir, "val", "breakpoints.bed")
    test_bed = os.path.join(data_dir, "test", "breakpoints.bed")
    root_bed = os.path.join(data_dir, "breakpoints.bed")

    # ── mode 1: train/val/test subdirectories already exist → load directly ──
    if os.path.isfile(train_bed):
        splits = {}
        for split_name, split_bed in [("train", train_bed), ("val", val_bed), ("test", test_bed)]:
            split_dir = os.path.join(data_dir, split_name)
            if os.path.isfile(split_bed):
                parser = BEDParser(
                    bed_path=split_bed,
                    fasta_dir=split_dir,
                    mode=None if mode == "auto" else mode,
                )
                samples = parser.parse()
                actual_mode = parser.mode
                splits[split_name] = HGTDataset(
                    samples=samples,
                    mode=actual_mode,
                    max_length=max_length,
                    read_len=read_len,
                    extract_bio=extract_bio,
                    evo_cache_path=_resolve_evo_cache(evo_cache_path, split_name),
                    rc_aug_prob=(rc_aug_prob if split_name == "train" else 0.0),
                )
            else:
                print(f"[WARN] {split_name}/breakpoints.bed not found, skipping")
                splits[split_name] = None

        return splits

    # ── mode 2: breakpoints.bed in the root directory → persistent split ──
    if os.path.isfile(root_bed):
        split_output_dir = _split_dataset_to_disk(
            input_dir=data_dir,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
        # load from the split output directory
        return load_splits(
            data_dir=split_output_dir,
            mode=mode,
            max_length=max_length,
            read_len=read_len,
            extract_bio=extract_bio,
            evo_cache_path=evo_cache_path,
            rc_aug_prob=rc_aug_prob,
        )

    raise FileNotFoundError(
        f"No data found: tried {train_bed} and {root_bed}")


def _split_dataset_to_disk(
    input_dir: str,
    output_dir: str = "",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> str:
    """
    Persistently split the data to disk following the v5.0 split_dataset.py logic.

    Improvements over v5.0:
      - ratio normalization check
      - deduplication by seq_id
      - automatic PE/contig mode detection
      - output directory defaults to input_dir/splits/

    Returns
    -------
    str: split output directory (containing train/, val/, test/ subdirectories)
    """
    # m-8 fix (2026-06-22): warn that this is a sample-level split (not genome isolation)
    import warnings
    warnings.warn(
        "_split_dataset_to_disk performs a sample-level split (by seq_id), not genome-level isolation —  — "
        "the same genome may appear across splits. It is not invoked when genome_isolation=True."
        "For formal training data, use DeepFluHGT.py simulate --genome_isolation instead.",
        UserWarning, stacklevel=2,
    )
    import gzip
    import random as rng
    import json

    rng.seed(seed)
    SPLITS = ["train", "val", "test"]

    if not output_dir:
        output_dir = os.path.join(input_dir, "splits")

    # ── 1. Scan the BED for all seq_ids ──
    in_bed = os.path.join(input_dir, "breakpoints.bed")
    all_ids = []
    with open(in_bed) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            all_ids.append(line.split("\t")[0])

    # Deduplicate while preserving order
    all_ids = list(dict.fromkeys(all_ids))
    rng.shuffle(all_ids)
    total = len(all_ids)
    print(f"[split] Total unique sequences: {total}")

    # ── 2. Ratio normalization ──
    test_ratio = 1.0 - train_ratio - val_ratio
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-5:
        train_ratio /= ratio_sum
        val_ratio /= ratio_sum
        test_ratio /= ratio_sum

    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = total - n_train - n_val

    # ── 3. Assign seq_ids to splits ──
    id_to_split = {}
    for i, seq_id in enumerate(all_ids):
        if i < n_train:
            id_to_split[seq_id] = "train"
        elif i < n_train + n_val:
            id_to_split[seq_id] = "val"
        else:
            id_to_split[seq_id] = "test"

    # ── 4. Create output directories ──
    for sp in SPLITS:
        os.makedirs(os.path.join(output_dir, sp), exist_ok=True)

    # ── 5. Copy files ──
    def _copy_bed_by_id(in_path, handles_dict, mapping):
        with open(in_path) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    for h in handles_dict.values():
                        h.write(line)
                    continue
                seq_id = line.split("\t")[0]
                sp = mapping.get(seq_id)
                if sp:
                    handles_dict[sp].write(line)

    def _copy_fasta_by_id(in_path, handles_dict, mapping):
        try:
            from Bio import SeqIO
        except ImportError:
            # Manual parse when Biopython is unavailable
            _copy_fasta_by_id_manual(in_path, handles_dict, mapping)
            return
        for rec in SeqIO.parse(in_path, "fasta"):
            sp = mapping.get(rec.id)
            if sp:
                SeqIO.write(rec, handles_dict[sp], "fasta")

    def _copy_fasta_by_id_manual(in_path, handles_dict, mapping):
        with open(in_path) as f:
            header = ""
            seq_lines = []
            for line in f:
                if line.startswith(">"):
                    if header:
                        _write_fasta_record(header, seq_lines, handles_dict, mapping)
                    header = line
                    seq_lines = []
                else:
                    seq_lines.append(line)
            if header:
                _write_fasta_record(header, seq_lines, handles_dict, mapping)

    def _write_fasta_record(header, seq_lines, handles_dict, mapping):
        seq_id = header[1:].split()[0]
        sp = mapping.get(seq_id)
        if sp:
            handles_dict[sp].write(header)
            for sl in seq_lines:
                handles_dict[sp].write(sl)

    def _copy_tsv_by_id(in_path, handles_dict, mapping):
        if not os.path.exists(in_path):
            return
        with open(in_path) as f:
            header = f.readline()
            for h in handles_dict.values():
                h.write(header)
            for line in f:
                if not line.strip():
                    continue
                seq_id = line.split("\t")[0]
                sp = mapping.get(seq_id)
                if sp:
                    handles_dict[sp].write(line)

    # Detect mode
    is_pe = os.path.exists(os.path.join(input_dir, "R1.fasta"))

    # BED
    bed_handles = {sp: open(os.path.join(output_dir, sp, "breakpoints.bed"), "w")
                   for sp in SPLITS}
    _copy_bed_by_id(in_bed, bed_handles, id_to_split)

    # FASTA
    fasta_handles = {}
    if is_pe:
        for r_file in ["R1.fasta", "R2.fasta"]:
            in_path = os.path.join(input_dir, r_file)
            if os.path.exists(in_path):
                handles = {sp: open(os.path.join(output_dir, sp, r_file), "w")
                           for sp in SPLITS}
                _copy_fasta_by_id(in_path, handles, id_to_split)
                fasta_handles.update(handles)
    else:
        in_path = os.path.join(input_dir, "dataset.fasta")
        if os.path.exists(in_path):
            handles = {sp: open(os.path.join(output_dir, sp, "dataset.fasta"), "w")
                       for sp in SPLITS}
            _copy_fasta_by_id(in_path, handles, id_to_split)
            fasta_handles.update(handles)

    # Provenance
    prov_handles = {sp: open(os.path.join(output_dir, sp, "provenance.tsv"), "w")
                    for sp in SPLITS}
    _copy_tsv_by_id(os.path.join(input_dir, "provenance.tsv"), prov_handles, id_to_split)

    # Metadata
    orig_meta = os.path.join(input_dir, "metadata.json")
    if os.path.exists(orig_meta):
        with open(orig_meta) as f:
            orig = json.load(f)
        meta_by_split = {sp: [] for sp in SPLITS}
        for rec in orig.get("positive", []):
            sp = id_to_split.get(rec.get("seq_id"))
            if sp:
                meta_by_split[sp].append(rec)
        for sp in SPLITS:
            with open(os.path.join(output_dir, sp, "metadata.json"), "w") as f:
                json.dump({"mode": orig.get("mode"), "positive": meta_by_split[sp]},
                          f, indent=2)

    # Close all file handles
    for h in list(fasta_handles.values()) + list(bed_handles.values()) + list(prov_handles.values()):
        h.close()

    print(f"[split] Done: train={n_train}, val={n_val}, test={n_test}")
    print(f"[split] Output: {output_dir}")

    return output_dir


def load_splits_sliding_window(
    data_dir: str,
    window_size: int = 1500,
    overlap: int = 300,
    max_neg_windows: int = 2,
    max_eval_windows: int = 10,
    read_len: int = 150,
    extract_bio: bool = True,
    evo_cache_path: Optional[str] = None,
    rc_aug_prob: float = 0.0,
) -> dict:
    """
    Load contig data as sliding-window datasets.

    Same auto-detection logic as load_splits(), but uses SlidingWindowContigDataset
    instead of HGTDataset. Each long contig is cut into multiple overlapping windows.

    Parameters
    ----------
    data_dir : str
        Data root directory (must contain train/val/test subdirectories)
    window_size : int
        Window size (bp)
    overlap : int
        Window overlap (bp)
    max_neg_windows : int
        Maximum number of windows per negative contig during training
    max_eval_windows : int
        Maximum number of windows per contig during validation/test (0=unlimited)
    read_len : int
        PE read length (unused in contig mode; kept for interface compatibility)
    extract_bio : bool
        Whether to extract biological features

    Returns
    -------
    dict: {"train": SlidingWindowContigDataset, "val": ..., "test": ...}
    """
    train_bed = os.path.join(data_dir, "train", "breakpoints.bed")
    val_bed = os.path.join(data_dir, "val", "breakpoints.bed")
    test_bed = os.path.join(data_dir, "test", "breakpoints.bed")

    splits = {}
    for split_name, split_bed in [
        ("train", train_bed), ("val", val_bed), ("test", test_bed),
    ]:
        split_dir = os.path.join(data_dir, split_name)
        if not os.path.isfile(split_bed):
            print(f"[WARN] {split_name}/breakpoints.bed not found, skipping")
            splits[split_name] = None
            continue

        parser = BEDParser(
            bed_path=split_bed,
            fasta_dir=split_dir,
            mode="contig",
        )
        samples = parser.parse()

        is_train = (split_name == "train")
        splits[split_name] = SlidingWindowContigDataset(
            samples=samples,
            window_size=window_size,
            overlap=overlap,
            is_training=is_train,                # train: Strategy D; val/test: full coverage + cap
            max_neg_windows=max_neg_windows,
            max_eval_windows=max_eval_windows,    # window cap during eval
            extract_bio=extract_bio,
            evo_cache_path=_resolve_evo_cache(evo_cache_path, split_name),
            rc_aug_prob=(rc_aug_prob if is_train else 0.0),
        )

    return splits


def collate(
    samples: List[HGTSampleTokenized],
    pad_id: int = 0,
    rebalance_hard_neg: bool = False,
    hard_neg_ratio: float = 0.5,
    balanced_neg_types: bool = True,
    target_batch_size: Optional[int] = None,
) -> HGTBatch:
    """
    Collate a list of tokenized samples into an HGTBatch.

    Parameters
    ----------
    samples : List[HGTSampleTokenized]
        List of samples returned by the Dataset
    pad_id : int
        Padding token ID
    rebalance_hard_neg : bool
        Whether to enable hard-negative rebalancing
    hard_neg_ratio : float
        Hard-negative ratio
    balanced_neg_types : bool
        Whether to balance negative sample types
    target_batch_size : Optional[int]
        Target batch size for resampling

    Returns
    -------
    HGTBatch
    """
    collator = HGTCollator(
        pad_id=pad_id,
        rebalance_hard_neg=rebalance_hard_neg,
        hard_neg_ratio=hard_neg_ratio,
        balanced_neg_types=balanced_neg_types,
        target_batch_size=target_batch_size,
    )
    return collator(samples)


def test() -> bool:
    """Run the data_pipeline module self-test."""
    results = []

    # ── Tokenizer test ──
    from data_pipeline.impl.tokenizer import SingleNucleotideTokenizer
    tok = SingleNucleotideTokenizer()

    # Basic tokenize
    ids = tok.tokenize("ATCG")
    assert ids == [1, 5, 6, 7, 8, 2], f"Basic tokenize failed: {ids}"
    results.append(("tokenizer.basic", True))

    # PE tokenize (150bp → 303 tokens)
    r1 = "A" * 150
    r2 = "T" * 150
    pe_ids = tok.tokenize_pe(r1, r2)
    assert len(pe_ids) == 303, f"PE tokenize length: {len(pe_ids)}"
    assert pe_ids[0] == 1  # CLS
    assert pe_ids[151] == 2  # SEP after R1
    assert pe_ids[-1] == 2  # SEP at end
    results.append(("tokenizer.pe_length", True))

    # N → UNK
    ids_n = tok.tokenize("AN")
    assert ids_n == [1, 5, 9, 2], f"N mapping failed: {ids_n}"
    results.append(("tokenizer.n_unk", True))

    # IUPAC → UNK
    ids_r = tok.tokenize("R")
    assert ids_r == [1, 4, 2], f"IUPAC UNK failed: {ids_r}"
    results.append(("tokenizer.iupac_unk", True))

    # Remapping
    ntv3_ids = tok.remap_to_ntv3([1, 5, 6])  # CLS, A, T
    assert ntv3_ids == [3, 6, 7], f"NTv3 remap failed: {ntv3_ids}"
    results.append(("tokenizer.ntv3_remap", True))

    cad_ids = tok.remap_to_caduceus([1, 5, 6])  # CLS, A, T
    assert cad_ids == [0, 7, 10], f"Caduceus remap failed: {cad_ids}"
    results.append(("tokenizer.caduceus_remap", True))

    # ── Region targets ──
    from data_pipeline.contracts import BreakpointEvent
    events = [BreakpointEvent(hgt_start=5, hgt_end=10)]
    region = tok.create_region_targets(events, 20)
    assert len(region) == 22, f"Region length: {len(region)}"
    assert region[0] == 0.0  # CLS
    assert region[6] == 1.0  # pos 5 (+1 for CLS)
    assert region[10] == 1.0  # pos 9
    assert region[11] == 0.0  # pos 10 (outside)
    assert region[-1] == 0.0  # SEP
    results.append(("targets.region", True))

    # ── Breakpoint targets ──
    bp = tok.create_breakpoint_targets(events, 20)
    assert len(bp) == 22
    assert bp[6] == 1.0  # hgt_start=5 → idx 6
    assert bp[10] == 1.0  # hgt_end-1=9 → idx 10
    results.append(("targets.breakpoint", True))

    # ── Bio features ──
    from data_pipeline.impl.bio_features import BioFeatureExtractor
    bio = BioFeatureExtractor()
    feats = bio.extract("ATCGATCG" * 10)
    assert feats.shape == (260,), f"Bio features shape: {feats.shape}"
    results.append(("bio_features.shape", True))

    # ── Print results ──
    print("\n[data_pipeline] Self-test results:")
    all_ok = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if not ok:
            all_ok = False

    return all_ok
