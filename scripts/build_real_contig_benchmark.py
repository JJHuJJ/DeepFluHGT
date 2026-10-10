#!/usr/bin/env python3
"""Control experiment: evaluate real IslandViewer GIs in Contig mode (3000bp windows).

Uses the same data source as build_real_pe_benchmark.py (PE 150bp) to distinguish:
  - whether PE 0.45 means "the 150bp read signal is too weak" or "the model does not generalize to real bacterial GIs" (domain shift)

Outputs a benbow-format contig FASTA → eval_external_benbow.py (Contig mode).
"""
import sys, argparse
from pathlib import Path
from Bio import SeqIO
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_real_pe_benchmark import collect_genomes, gc  # reuse extraction logic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--islandviewer", default="benchmark/islandviewer4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=3000)
    ap.add_argument("--gc_tol", type=float, default=3.0)
    ap.add_argument("--max_per_side", type=int, default=300)
    args = ap.parse_args()

    iv_dir = Path(args.islandviewer)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    pos_contigs, neg_contigs = [], []  # (id, seq)
    for gid, genome, gi_coords in collect_genomes(iv_dir):
        glen = len(genome)
        gi_sorted = sorted((max(0, s), min(glen, e)) for s, e in gi_coords)
        # pos = GI sequence (only segments >= window become contigs)
        for (s, e) in gi_sorted:
            if e - s >= args.window:
                pos_contigs.append((f"{gid}_GI_{s}_{e}", genome[s:e]))
        # neg = flank windows (GC-matched to the overall pos GC)
        cursor = 0
        for (s, e) in gi_sorted:
            while cursor + args.window <= s:
                w = genome[cursor:cursor + args.window]
                if "N" not in w:
                    neg_contigs.append((f"{gid}_flank_{cursor}", w))
                cursor += args.window
            cursor = max(cursor, e)
        while cursor + args.window <= glen:
            w = genome[cursor:cursor + args.window]
            if "N" not in w:
                neg_contigs.append((f"{gid}_flank_{cursor}", w))
            cursor += args.window

    # GC-matched sampling of negs so their GC is close to that of pos
    pos_gcs = [gc(s) for _, s in pos_contigs]
    pos_gc_mean = sum(pos_gcs) / max(1, len(pos_gcs))
    neg_contigs.sort(key=lambda x: abs(gc(x[1]) - pos_gc_mean))
    neg_matched = [(i, s) for i, s in neg_contigs if abs(gc(s) - pos_gc_mean) <= args.gc_tol]
    if len(neg_matched) < 50:
        neg_matched = neg_contigs[:200]
    import random; random.seed(42)
    pos_s = random.sample(pos_contigs, min(len(pos_contigs), args.max_per_side))
    neg_s = random.sample(neg_matched, min(len(neg_matched), args.max_per_side))

    fasta = out_dir / "islandviewer_contigs.fasta"
    with open(fasta, "w") as f:
        for i, (sid, s) in enumerate(pos_s):
            f.write(f">{sid}|1|{sid} real_gi\n{s}\n")
        for i, (sid, s) in enumerate(neg_s):
            f.write(f">{sid}|0|{sid} real_flank_gc_matched\n{s}\n")
    pos_gc = sum(gc(s) for _, s in pos_s) / len(pos_s)
    neg_gc = sum(gc(s) for _, s in neg_s) / len(neg_s)
    print(f"[Data] pos contigs={len(pos_s)} neg contigs={len(neg_s)}")
    print(f"[GC]  pos GC={pos_gc:.1f}%  neg GC={neg_gc:.1f}%  |Δ|={abs(pos_gc-neg_gc):.1f}%")
    print(f"FASTA: {fasta}")
    print(f"\nNext step: eval_external_benbow.py --benchmark {fasta} --ckpt <Contig> ...")


if __name__ == "__main__":
    main()
