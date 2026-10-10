# data_pipeline/tests/test_integration.py
"""
End-to-end integration tests: simulator output → data_pipeline load → batch.
Requires realistic simulator output data.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from data_pipeline.impl.parser import BEDParser
from data_pipeline.impl.dataset import HGTDataset
from data_pipeline.impl.collator import HGTCollator


def _generate_synthetic_pe_data(tmpdir):
    """Generate synthetic PE simulator output."""
    bed_path = os.path.join(tmpdir, "breakpoints.bed")
    r1_path = os.path.join(tmpdir, "R1.fasta")
    r2_path = os.path.join(tmpdir, "R2.fasta")
    meta_path = os.path.join(tmpdir, "metadata.json")

    n_pos, n_neg = 20, 20

    with open(bed_path, 'w') as bf, \
         open(r1_path, 'w') as f1, \
         open(r2_path, 'w') as f2:

        bf.write("#seq_id\tevent_idx\tlabel\tinsert_len\thgt_start\thgt_end\t"
                 "virus_len\thgt_mode\tpe_case\tr1_vs\tr1_ve\tr2_vs\tr2_ve\n")

        for i in range(n_pos):
            sid = f"pos_{i:08d}"
            r1 = "A" * 150
            r2 = "T" * 150
            f1.write(f">{sid} HGT case=R1_HOST_R2_MIXED\n{r1}\n")
            f2.write(f">{sid} HGT case=R1_HOST_R2_MIXED\n{r2}\n")
            bf.write(f"{sid}\t0\t1\t300\t50\t150\t100\tinsertion\t"
                     f"R1_HOST_R2_MIXED\t-1\t-1\t0\t100\n")

        for i, ntype in enumerate(["NEG"] * 7 + ["NEG_VIRUS"] * 5 +
                                   ["NEG_HARD"] * 4 + ["NEG_PCR"] * 4):
            sid = f"neg_{i:08d}"
            r1 = "C" * 150
            r2 = "G" * 150
            f1.write(f">{sid} {ntype}\n{r1}\n")
            f2.write(f">{sid} {ntype}\n{r2}\n")
            bf.write(f"{sid}\t0\t0\t300\t-1\t-1\t0\tnone\t{ntype}\t"
                     f"-1\t-1\t-1\t-1\n")

    with open(meta_path, 'w') as f:
        json.dump({"positive": [], "negative": []}, f)

    return tmpdir


def _generate_synthetic_contig_data(tmpdir):
    """Generate synthetic contig simulator output."""
    bed_path = os.path.join(tmpdir, "breakpoints.bed")
    ds_path = os.path.join(tmpdir, "dataset.fasta")
    meta_path = os.path.join(tmpdir, "metadata.json")

    n_pos, n_neg = 20, 20

    with open(bed_path, 'w') as bf, \
         open(ds_path, 'w') as fa:

        bf.write("#seq_id\tevent_idx\tlabel\tseq_len\thgt_start\thgt_end\t"
                 "virus_len\thgt_mode\n")

        for i in range(n_pos):
            sid = f"pos_{i:08d}"
            seq = "ATCG" * 100  # 400bp
            fa.write(f">{sid} HGT len=400\n{seq}\n")
            bf.write(f"{sid}\t0\t1\t400\t100\t200\t100\tinsertion\n")

        for i in range(n_neg):
            sid = f"neg_{i:08d}"
            seq = "GCTA" * 100
            fa.write(f">{sid} NEG\n{seq}\n")
            bf.write(f"{sid}\t0\t0\t400\t-1\t-1\t0\tNEG\n")

    with open(meta_path, 'w') as f:
        json.dump({"positive": [], "negative": []}, f)

    return tmpdir


def test_pe_pipeline():
    """PE end-to-end: parse → Dataset → Batch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _generate_synthetic_pe_data(tmpdir)

        parser = BEDParser(
            os.path.join(tmpdir, "breakpoints.bed"),
            tmpdir, mode="paired_end")
        samples = parser.parse()
        assert len(samples) == 40

        ds = HGTDataset(samples, mode="paired_end")
        assert len(ds) == 40

        item = ds[0]
        assert len(item.input_ids) == 303
        assert len(item.region_targets) == 303
        assert item.bio_features.shape == (260,)

        # Batch
        collator = HGTCollator()
        batch_items = [ds[i] for i in range(8)]
        batch = collator(batch_items)

        assert batch.input_ids.shape == (8, 303)
        assert batch.labels.shape == (8,)
        assert batch.breakpoint_targets.shape == (8, 303)
        assert batch.region_targets.shape == (8, 303)
    return True


def test_contig_pipeline():
    """Contig end-to-end: parse → Dataset → Batch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _generate_synthetic_contig_data(tmpdir)

        parser = BEDParser(
            os.path.join(tmpdir, "breakpoints.bed"),
            tmpdir, mode="contig")
        samples = parser.parse()
        assert len(samples) == 40

        ds = HGTDataset(samples, mode="contig")
        assert len(ds) == 40

        item = ds[0]
        assert len(item.input_ids) == 402  # 400 + CLS + SEP
        assert item.label == 1
        assert item.region_targets[101] == 1.0  # pos 100

        # Batch (variable length)
        collator = HGTCollator()
        batch_items = [ds[i] for i in range(4)]
        batch = collator(batch_items)

        assert batch.input_ids.shape[0] == 4
        assert batch.labels.shape == (4,)
    return True


def test_pe_data_alignment():
    """PE mode: region_targets fully aligned with input_ids."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _generate_synthetic_pe_data(tmpdir)

        parser = BEDParser(
            os.path.join(tmpdir, "breakpoints.bed"),
            tmpdir, mode="paired_end")
        samples = parser.parse()

        ds = HGTDataset(samples, mode="paired_end")
        for i in range(len(ds)):
            item = ds[i]
            assert len(item.input_ids) == len(item.region_targets), \
                f"sample {i}: mismatch"
            assert len(item.input_ids) == len(item.breakpoint_targets), \
                f"sample {i}: bp mismatch"
    return True


def test_contig_data_alignment():
    """Contig mode: targets alignment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _generate_synthetic_contig_data(tmpdir)

        parser = BEDParser(
            os.path.join(tmpdir, "breakpoints.bed"),
            tmpdir, mode="contig")
        samples = parser.parse()

        ds = HGTDataset(samples, mode="contig")
        for i in range(len(ds)):
            item = ds[i]
            assert len(item.input_ids) == len(item.region_targets)
            assert len(item.input_ids) == len(item.breakpoint_targets)
    return True


def test_hard_neg_balancing():
    """Hard-negative balancing integration test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _generate_synthetic_pe_data(tmpdir)

        parser = BEDParser(
            os.path.join(tmpdir, "breakpoints.bed"),
            tmpdir, mode="paired_end")
        samples = parser.parse()

        ds = HGTDataset(samples, mode="paired_end")
        batch_items = [ds[i] for i in range(len(ds))]

        collator = HGTCollator(
            rebalance_hard_neg=True,
            balanced_neg_types=True,
            target_batch_size=8,
        )
        batch = collator(batch_items)
        assert batch.input_ids.shape[0] == 8
    return True


def run():
    tests = [
        test_pe_pipeline,
        test_contig_pipeline,
        test_pe_data_alignment,
        test_contig_data_alignment,
        test_hard_neg_balancing,
    ]
    results = []
    for t in tests:
        try:
            ok = t()
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {t.__name__}")
            results.append(ok)
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            results.append(False)
    return all(results)


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
