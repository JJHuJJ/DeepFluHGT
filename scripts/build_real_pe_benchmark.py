#!/usr/bin/env python3
"""Build a real-sequence PE benchmark (L1) — uses IslandViewer real microbial genomes + experimentally validated GI annotations.

Real sequences (IslandViewer) + simulated reads (wgsim). Fills the "PE x real sequence" evidence gap.
See REAL_PE_VALIDATION_PLAN.md §3 for details.

Pipeline:
  1. iterate over benchmark/islandviewer4/{NC,NZ}_*/ : genome + _islands.fasta (GI coordinates in the header)
  2. GI regions → pos sources; flank (non-GI) windows → neg sources, GC-matched to pos
  3. wgsim samples PE reads from the pos / neg sources separately (150bp, insert 350±50, Illumina errors)
  4. merge R1/R2.fasta + pe_labels_real.tsv (read_id, label, source)

usage:
  python scripts/build_real_pe_benchmark.py \
      --islandviewer benchmark/islandviewer4 \
      --out audit_output/real_pe_bench \
      --n_pos 2000 --n_neg 2000 --window 3000 --gc_tol 3.0
"""
import os, sys, argparse, subprocess, tempfile, shutil
from pathlib import Path
from Bio import SeqIO

ROOT = Path(__file__).resolve().parent.parent
WGSIM = shutil.which("wgsim") or "/usr/bin/wgsim"


def gc(seq: str) -> float:
    s = seq.upper()
    g = s.count("G") + s.count("C")
    return 100.0 * g / max(1, len(s))


def collect_genomes(iv_dir: Path):
    """Yield (gid, genome_seq, [(gi_start, gi_end), ...])."""
    for d in sorted(iv_dir.iterdir()):
        if not d.is_dir():
            continue
        gids = list(d.glob("*.fasta"))
        gids = [g for g in gids if "islands" not in g.name and "merged" not in g.name.lower()]
        isl = list(d.glob("*_islands.fasta"))
        if not gids or not isl:
            continue
        genome_rec = next(SeqIO.parse(str(gids[0]), "fasta"))
        gi_coords = []
        for rec in SeqIO.parse(str(isl[0]), "fasta"):
            parts = rec.id.split("_GI_")
            if len(parts) == 2:
                try:
                    s, e = parts[1].split("_")[0], parts[1].split("_")[1]
                    gi_coords.append((int(s), int(e)))
                except (ValueError, IndexError):
                    continue
        yield gids[0].stem, str(genome_rec.seq).upper(), gi_coords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--islandviewer", default="benchmark/islandviewer4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_pos", type=int, default=2000)
    ap.add_argument("--n_neg", type=int, default=2000)
    ap.add_argument("--window", type=int, default=3000, help="flank window size")
    ap.add_argument("--gc_tol", type=float, default=3.0, help="GC matching tolerance (%)")
    ap.add_argument("--read_len", type=int, default=150)
    ap.add_argument("--insert", type=int, default=350)
    ap.add_argument("--insert_sd", type=int, default=50)
    args = ap.parse_args()

    iv_dir = Path(args.islandviewer)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp())

    # -- 1. collect GIs (pos) + flank windows (neg candidates) --
    gi_records = [] # pos source: SeqRecord list
    flank_windows = [] # (gc, seq) neg candidate
    n_genomes = n_gi = 0
    for gid, genome, gi_coords in collect_genomes(iv_dir):
        n_genomes += 1
        glen = len(genome)
        gi_set = []
        for (s, e) in gi_coords:
            s, e = max(0, s), min(glen, e)
            if e - s >= args.read_len * 2:
                gi_set.append((s, e))
                gi_records.append((f"{gid}_GI_{s}_{e}", genome[s:e]))
                n_gi += 1
        # flank = genome - GI; slice windows
        gi_sorted = sorted(gi_set)
        cursor = 0
        for (s, e) in gi_sorted:
            # flank before the GI
            while cursor + args.window <= s:
                w = genome[cursor:cursor + args.window]
                if "N" not in w:
                    flank_windows.append((gc(w), w))
                cursor += args.window
            cursor = max(cursor, e)
        while cursor + args.window <= glen:  # trailing flank
            w = genome[cursor:cursor + args.window]
            if "N" not in w:
                flank_windows.append((gc(w), w))
            cursor += args.window

    print(f"[Data] genomes={n_genomes} GI(pos records)={n_gi} flank windows={len(flank_windows)}")

    # -- 2. GC-matched flank sampling --
    pos_gcs = [gc(s) for _, s in gi_records]
    pos_gc_mean = sum(pos_gcs) / max(1, len(pos_gcs))
    # bucket by GC; prefer flank windows whose GC is close to the pos mean
    flank_windows.sort(key=lambda x: abs(x[0] - pos_gc_mean))
    # also ensure a GC-distribution match: tolerance filter + equal-quantity sampling
    matched = [(g, s) for g, s in flank_windows if abs(g - pos_gc_mean) <= args.gc_tol]
    if len(matched) < 50:  # tolerance too strict, relax
        matched = flank_windows[:200]
    print(f"[GC] pos mean GC={pos_gc_mean:.1f}% GC-matched flank windows={len(matched)} "
          f"(tol ±{args.gc_tol}%)")

    # wgsim limitation: with more source sequences than -N, some get 0 reads (per-sequence floor assignment).
    # subsample to a representative set <= N/2 so every source sequence gets >=2 reads and wgsim output is stable.
    import random
    random.seed(42)
    max_pos_src = max(args.n_pos // 2, 100)
    max_neg_src = max(args.n_neg // 2, 100)
    if len(gi_records) > max_pos_src:
        gi_records = random.sample(gi_records, max_pos_src)
    if len(matched) > max_neg_src:
        matched = random.sample(matched, max_neg_src)
    print(f"[src] pos source seqs={len(gi_records)} neg source seqs={len(matched)} (capped for wgsim)")

    # write pos / neg source fasta
    pos_src = tmp / "pos_source.fasta"
    neg_src = tmp / "neg_source.fasta"
    with open(pos_src, "w") as f:
        for i, (sid, s) in enumerate(gi_records):
            f.write(f">{sid}\n{s}\n")
    with open(neg_src, "w") as f:
        for i, (g, s) in enumerate(matched):
            f.write(f">flank_{i:06d}_gc{g:.1f}\n{s}\n")

    # -- 3. sample PE reads with wgsim --
    pos_r1, pos_r2 = tmp / "pos_R1.fq", tmp / "pos_R2.fq"
    neg_r1, neg_r2 = tmp / "neg_R1.fq", tmp / "neg_R2.fq"
    common = ["-1", str(args.read_len), "-2", str(args.read_len),
              "-d", str(args.insert), "-s", str(args.insert_sd), "-e", "0.02", "-r", "0", "-R", "0", "-X", "0"]
    print(f"[wgsim] pos -N {args.n_pos} ...")
    subprocess.run([WGSIM] + common + ["-N", str(args.n_pos), str(pos_src), str(pos_r1), str(pos_r2)],
                   check=True, capture_output=True)
    print(f"[wgsim] neg -N {args.n_neg} ...")
    subprocess.run([WGSIM] + common + ["-N", str(args.n_neg), str(neg_src), str(neg_r1), str(neg_r2)],
                   check=True, capture_output=True)

    # -- 4. merge R1/R2.fasta + rename + labels.tsv --
    r1_out, r2_out = out_dir / "pe_real_R1.fasta", out_dir / "pe_real_R2.fasta"
    labels_out = out_dir / "pe_labels_real.tsv"

    def parse_fq(path):
        recs = []
        with open(path) as f:
            while True:
                h = f.readline()
                if not h:
                    break
                seq = f.readline().strip()
                f.readline(); f.readline() # +, qual
                recs.append(seq)
        return recs

    pos1, pos2 = parse_fq(pos_r1), parse_fq(pos_r2)
    neg1, neg2 = parse_fq(neg_r1), parse_fq(neg_r2)
    assert len(pos1) == len(pos2) and len(neg1) == len(neg2), "R1/R2 count mismatch"

    with open(r1_out, "w") as f1, open(r2_out, "w") as f2, open(labels_out, "w") as fl:
        fl.write("read_id\tlabel\tsource\tpe_case\n")
        for i, (s1, s2) in enumerate(zip(pos1, pos2)):
            rid = f"pos_{i:06d}"
            f1.write(f">{rid}/1\n{s1}\n"); f2.write(f">{rid}/2\n{s2}\n")
            fl.write(f"{rid}\t1\treal_gi\tPOS\n")
        for i, (s1, s2) in enumerate(zip(neg1, neg2)):
            rid = f"neg_{i:06d}"
            f1.write(f">{rid}/1\n{s1}\n"); f2.write(f">{rid}/2\n{s2}\n")
            fl.write(f"{rid}\t0\treal_flank_gc_matched\tNEG\n")

    n_total = len(pos1) + len(neg1)
    print(f"\n{'='*60}")
    print(f"truereal PE benchmark buildcompleted (L1)")
    print(f"{'='*60}")
    print(f" pos (real GI) : {len(pos1)}")
    print(f" neg (GC-matched flank): {len(neg1)}")
    print(f" total read-pairs : {n_total}")
    print(f" pos mean GC={pos_gc_mean:.1f}% (neg GC-matched ±{args.gc_tol}%)")
    print(f" R1: {r1_out}")
    print(f" R2: {r2_out}")
    print(f" labels: {labels_out}")
    print(f"\nnext steps:")
    print(f" DeepFluHGT.py predict {r1_out},{r2_out} --mode paired_end \\")
    print(f" --checkpoint checkpoints/PE_2M/best_model_slim.pt \\")
    print(f" --config configs/pe_2m.yaml --read_len {args.read_len} --batch_size 8 \\")
    print(f" --output_dir {out_dir}/predict")

    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
