# simulator/impl/dataset_splitter.py
"""
Dataset splitter — splits train/val/test by ratio.
Refactored from v4.0 split_dataset.py, CLI dependency removed.
"""

import json
import os
import random
from collections import defaultdict

from Bio import SeqIO


def detect_mode(input_dir: str):
    """Detect the data mode of the input directory."""
    files = set(os.listdir(input_dir))
    has_r1 = "R1.fasta" in files
    has_r2 = "R2.fasta" in files
    has_ds = "dataset.fasta" in files
    has_bed = "breakpoints.bed" in files
    has_meta = "metadata.json" in files

    if not has_bed:
        raise FileNotFoundError("breakpoints.bed not found")

    if has_r1 and has_r2:
        return "paired_end", "R1.fasta", "R2.fasta", has_meta
    if has_ds:
        return "contig", "dataset.fasta", None, has_meta
    raise ValueError("Cannot detect mode (need dataset.fasta or R1/R2.fasta)")


def collect_ids(fasta_path: str) -> list:
    return [rec.id for rec in SeqIO.parse(fasta_path, "fasta")]


def load_metadata(meta_path: str) -> dict:
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path) as f:
        data = json.load(f)
    return {r['seq_id']: r for r in data.get('positive', [])}


def copy_fasta_by_id(in_fasta: str, out_handles: dict, id_to_split: dict):
    for rec in SeqIO.parse(in_fasta, "fasta"):
        sp = id_to_split.get(rec.id)
        if sp:
            out_handles[sp].write(f">{rec.id}\n{str(rec.seq)}\n")


def copy_bed_by_id(in_bed: str, out_handles: dict, id_to_split: dict):
    with open(in_bed) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            sid = line.split('\t', 1)[0]
            sp = id_to_split.get(sid)
            if sp:
                out_handles[sp].write(line)


class DatasetSplitter:
    """Split the simulated data into train/val/test by ratio."""

    def __init__(self, train_ratio: float = 0.8,
                 val_ratio: float = 0.1,
                 test_ratio: float = 0.1,
                 seed: int = 42):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def split(self, input_dir: str, output_dir: str) -> str:
        """
        Run the split and return the output directory path.
        """
        random.seed(self.seed)

        mode, f1_name, f2_name, has_meta = detect_mode(input_dir)
        print(f"[DatasetSplitter] Detected {mode} mode")

        all_ids = collect_ids(os.path.join(input_dir, f1_name))
        total = len(all_ids)
        print(f"[DatasetSplitter] Total samples: {total:,}")

        tr_sz = int(round(total * self.train_ratio))
        va_sz = int(round(total * self.val_ratio))
        te_sz = total - tr_sz - va_sz
        print(f"[DatasetSplitter] train={tr_sz:,} val={va_sz:,} test={te_sz:,}")

        shuffled = all_ids.copy()
        random.shuffle(shuffled)
        tr_ids = shuffled[:tr_sz]
        va_ids = shuffled[tr_sz:tr_sz + va_sz]
        te_ids = shuffled[tr_sz + va_sz:tr_sz + va_sz + te_sz]

        id_to_split: dict = {}
        for sid in tr_ids: id_to_split[sid] = 'train'
        for sid in va_ids: id_to_split[sid] = 'val'
        for sid in te_ids: id_to_split[sid] = 'test'

        # Positive-sample distribution statistics
        pos_meta = load_metadata(os.path.join(input_dir, 'metadata.json'))
        for sp_name, sp_ids in [('train', tr_ids), ('val', va_ids), ('test', te_ids)]:
            n_pos = sum(1 for sid in sp_ids if sid in pos_meta)
            print(f" {sp_name}: {len(sp_ids):,} samples "
                  f"({n_pos:,} positive, {len(sp_ids)-n_pos:,} negative)")

        # -- Create output directories and files --
        SPLITS = ['train', 'val', 'test']
        for sp in SPLITS:
            os.makedirs(os.path.join(output_dir, sp), exist_ok=True)

        fasta_handles: dict = {}
        bed_handles: dict = {}

        for sp in SPLITS:
            sp_dir = os.path.join(output_dir, sp)
            if mode == 'paired_end':
                fasta_handles[(sp, 'r1')] = open(
                    os.path.join(sp_dir, 'R1.fasta'), 'w')
                fasta_handles[(sp, 'r2')] = open(
                    os.path.join(sp_dir, 'R2.fasta'), 'w')
            else:
                fasta_handles[sp] = open(
                    os.path.join(sp_dir, 'dataset.fasta'), 'w')
            bed_handles[sp] = open(
                os.path.join(sp_dir, 'breakpoints.bed'), 'w')

        # -- Write BED header --
        in_bed = os.path.join(input_dir, 'breakpoints.bed')
        with open(in_bed) as f:
            header_line = f.readline()
        for sp in SPLITS:
            bed_handles[sp].write(header_line)

        # -- Assign FASTA --
        if mode == 'paired_end':
            r1_handles = {sp: fasta_handles[(sp, 'r1')] for sp in SPLITS}
            r2_handles = {sp: fasta_handles[(sp, 'r2')] for sp in SPLITS}
            copy_fasta_by_id(os.path.join(input_dir, 'R1.fasta'),
                             r1_handles, id_to_split)
            copy_fasta_by_id(os.path.join(input_dir, 'R2.fasta'),
                             r2_handles, id_to_split)
        else:
            copy_fasta_by_id(os.path.join(input_dir, 'dataset.fasta'),
                             fasta_handles, id_to_split)

        # -- Assign BED --
        copy_bed_by_id(in_bed, bed_handles, id_to_split)

        # -- Close files --
        for fh in fasta_handles.values(): fh.close()
        for fh in bed_handles.values(): fh.close()

        # -- Split metadata.json --
        if has_meta:
            meta_by_split: dict = {sp: [] for sp in SPLITS}
            orig_meta_path = os.path.join(input_dir, 'metadata.json')
            with open(orig_meta_path) as f:
                orig = json.load(f)
            orig_mode = orig.get('mode', 'unknown')
            for rec in orig.get('positive', []):
                sp = id_to_split.get(rec['seq_id'])
                if sp:
                    meta_by_split[sp].append(rec)
            for sp in SPLITS:
                out_meta = os.path.join(output_dir, sp, 'metadata.json')
                with open(out_meta, 'w') as f:
                    json.dump({'mode': orig_mode,
                               'positive': meta_by_split[sp]}, f, indent=2)

        print(f"[DatasetSplitter] Done -> {output_dir}")
        return output_dir
