# data_pipeline/tests/test_parser.py
"""
BED / FASTA parser tests.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from data_pipeline.impl.parser import BEDParser, resolve_data_paths


def _write_test_data_pe(tmpdir):
    """create PE testdata."""
    bed_path = os.path.join(tmpdir, "breakpoints.bed")
    r1_path = os.path.join(tmpdir, "R1.fasta")
    r2_path = os.path.join(tmpdir, "R2.fasta")

    # PE BED (13 columns)
    with open(bed_path, 'w') as f:
        f.write("#seq_id\tevent_idx\tlabel\tinsert_len\thgt_start\thgt_end\t"
                "virus_len\thgt_mode\tpe_case\tr1_vs\tr1_ve\tr2_vs\tr2_ve\n")
        f.write("pos_00000000\t0\t1\t300\t50\t150\t100\tinsertion\t"
                "R1_HOST_R2_MIXED\t-1\t-1\t0\t100\n")
        f.write("neg_00000000\t0\t0\t300\t-1\t-1\t0\tnone\tNEG\t"
                "-1\t-1\t-1\t-1\n")

    with open(r1_path, 'w') as f:
        f.write(">pos_00000000 HGT case=R1_HOST_R2_MIXED\n")
        f.write("A" * 150 + "\n")
        f.write(">neg_00000000 NEG\n")
        f.write("T" * 150 + "\n")

    with open(r2_path, 'w') as f:
        f.write(">pos_00000000 HGT case=R1_HOST_R2_MIXED\n")
        f.write("C" * 150 + "\n")
        f.write(">neg_00000000 NEG\n")
        f.write("G" * 150 + "\n")

    return tmpdir


def _write_test_data_contig(tmpdir):
    """create Contig testdata."""
    bed_path = os.path.join(tmpdir, "breakpoints.bed")
    ds_path = os.path.join(tmpdir, "dataset.fasta")

    # Contig BED (8 columns)
    with open(bed_path, 'w') as f:
        f.write("#seq_id\tevent_idx\tlabel\tseq_len\thgt_start\thgt_end\t"
                "virus_len\thgt_mode\n")
        f.write("pos_00000000\t0\t1\t500\t100\t200\t100\tinsertion\n")
        f.write("pos_00000000\t1\t1\t500\t300\t350\t50\treplacement\n")
        f.write("neg_00000000\t0\t0\t500\t-1\t-1\t0\tNEG\n")

    with open(ds_path, 'w') as f:
        f.write(">pos_00000000 HGT len=500\n")
        f.write("ATCG" * 125 + "\n")
        f.write(">neg_00000000 NEG\n")
        f.write("GCTA" * 125 + "\n")

    return tmpdir


def test_parse_pe():
    """PE modeparse."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_test_data_pe(tmpdir)
        parser = BEDParser(
            os.path.join(tmpdir, "breakpoints.bed"),
            tmpdir, mode="paired_end")
        samples = parser.parse()

        assert len(samples) == 2, f"Expected 2, got {len(samples)}"

        pos = [s for s in samples if s.label == 1][0]
        neg = [s for s in samples if s.label == 0][0]

        assert pos.seq_id == "pos_00000000"
        assert pos.r2_sequence is not None
        assert len(pos.events) == 1
        assert pos.events[0].pe_case == "R1_HOST_R2_MIXED"
        assert neg.sample_type == "NEG"
    return True


def test_parse_contig():
    """Contig modeparse."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_test_data_contig(tmpdir)
        parser = BEDParser(
            os.path.join(tmpdir, "breakpoints.bed"),
            tmpdir, mode="contig")
        samples = parser.parse()

        assert len(samples) == 2

        pos = [s for s in samples if s.label == 1][0]
        assert pos.seq_id == "pos_00000000"
        assert pos.r2_sequence is None
        assert len(pos.events) == 2  # multi-event
        assert pos.events[0].hgt_mode == "insertion"
        assert pos.events[1].hgt_mode == "replacement"
    return True


def test_auto_detect_pe():
    """Auto-detect PE mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_test_data_pe(tmpdir)
        parser = BEDParser(
            os.path.join(tmpdir, "breakpoints.bed"), tmpdir)
        assert parser.mode == "paired_end"
    return True


def test_auto_detect_contig():
    """Auto-detect contig mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_test_data_contig(tmpdir)
        parser = BEDParser(
            os.path.join(tmpdir, "breakpoints.bed"), tmpdir)
        assert parser.mode == "contig"
    return True


def test_resolve_paths_direct():
    """Path resolution: direct directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        open(os.path.join(tmpdir, "breakpoints.bed"), 'w').close()
        paths = resolve_data_paths(tmpdir)
        assert paths["bed_path"] == os.path.join(tmpdir, "breakpoints.bed")
    return True


def test_resolve_paths_subdir():
    """Path resolution: train subdirectory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "train"))
        open(os.path.join(tmpdir, "train", "breakpoints.bed"), 'w').close()
        paths = resolve_data_paths(tmpdir)
        assert "train" in paths["bed_path"]
    return True


def run():
    tests = [
        test_parse_pe,
        test_parse_contig,
        test_auto_detect_pe,
        test_auto_detect_contig,
        test_resolve_paths_direct,
        test_resolve_paths_subdir,
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
