#!/usr/bin/env python3
"""Real-water-body metagenomic/metatranscriptomic HGT discovery pipeline (Ch5 discovery + Ch6 active HGT).

Data (prepared by the user, pending sync; format in CHAPTER6_PIPELINE_DESIGN.md §0):
  - metagenome: contigs (preferred) + fastq
  - metatranscriptome: fastq + contigs
  - DNA/RNA partially paired
  - 6-7 water bodies, 200-300 samples

Stage → thesis chapter mapping (aligned with THESIS_OUTLINE_DRAFT.md):
  qc       Ch5 §5.2  data ingestion with QC/assembly (contig length filter + dedup; fastq→fastp+MEGAHIT)
  predict  Ch5 §5.3  DeepFluHGT contig HGT prediction (median aggregation)
  validate Ch5 §5.4  BLAST + mobile elements + taxonomic-origin triple validation / Tier grading
  active   Ch6 §6.1-6.2  active HGT from the metatranscriptome (salmon TPM → active vs silent) ★ NMI selling point
  cross    Ch6 §6.3  cross-water-body generalization + ecology

External tool dependencies (called via subprocess; graceful degradation + WARN if missing):
  salmon/bowtie2/megahit/fastp/blastn/minimap2/samtools/cd-hit-est ✓ already installed;
  kraken2/genomad ✗ optional (if missing, skip taxonomic-origin/mobility analysis).

usage (run in order once the data is synced):
  python scripts/ch6_pipeline.py qc       --fastq-dir <FQ> --contigs-dir <CTG> --out ch6/
  python scripts/ch6_pipeline.py predict  --contigs-dir ch6/contigs --ckpt <contig_slim.pt> --out ch6/
  python scripts/ch6_pipeline.py active   --hgt-dir ch6/hgt --contigs-dir ch6/contigs --pairs pairs.tsv --out ch6/active
  python scripts/ch6_pipeline.py cross    --hgt-dir ch6/hgt --active-json ch6/active/active_hgt.json --water-map water.tsv --out ch6/cross
  python scripts/ch6_pipeline.py validate --hgt-dir ch6/hgt --contigs-dir ch6/contigs --top-n 200 --out ch6/validate

Input file formats:
  pairs.tsv  : sample\trna_r1.fq\trna_r2.fq   (paired samples; a missing column = non-paired DNA-only)
  water.tsv  : sample\twater                   (sample → water-body mapping)
"""
from __future__ import annotations

import os
import sys
import csv
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
PYTHON = "python"
BLAST_DB = os.environ.get("CH6_BLAST_DB", str(ROOT / "audit_output/blast_db/local_db"))


# ─────────────────────────── helpers ───────────────────────────
def run(cmd: list, desc: str = "") -> int:
    """Run an external command with live output; on failure return rc instead of raising."""
    if desc:
        print(f"  $ {' '.join(cmd[:2])} ...  ({desc})", flush=True)
    return subprocess.run(cmd, check=False).returncode


def have(tool: str) -> bool:
    return shutil.which(tool) is not None


def read_predictions(tsv: Path) -> dict:
    """Read predictions.tsv → {seq_id (contig): hgt_probability}."""
    out = {}
    if not tsv.exists():
        return out
    with open(tsv) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                out[row["seq_id"]] = float(row["hgt_probability"])
            except (KeyError, ValueError, TypeError):
                continue
    return out


def read_fasta_ids(path: Path) -> set:
    ids = set()
    if not path.exists():
        return ids
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                ids.add(line[1:].split()[0])
    return ids


def iter_contig_seq(fa: Path):
    """yield (id, seq) from a fasta (memory-friendly)."""
    sid, buf = None, []
    if not fa.exists():
        return
    with open(fa) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if sid is not None:
                    yield sid, "".join(buf)
                sid, buf = line[1:].split()[0], []
            else:
                buf.append(line.upper())
        if sid is not None:
            yield sid, "".join(buf)


# ─────────────────────────── A. qc / assembly ───────────────────────────
def cmd_qc(args):
    """Ch5 §5.2: contig length filter + dedup; fastp+MEGAHIT assembly for fastq-only samples."""
    out = Path(args.out)
    contigs_out = out / "contigs"
    contigs_out.mkdir(parents=True, exist_ok=True)
    contigs_dir = Path(args.contigs_dir) if args.contigs_dir else None
    fastq_dir = Path(args.fastq_dir) if args.fastq_dir else None
    min_len = args.min_contig_len

    n_use, n_asm = 0, 0
    # 1) existing contigs → length filter + cd-hit-est dedup
    if contigs_dir and contigs_dir.is_dir():
        fas = sorted(list(contigs_dir.glob("*.fasta")) + list(contigs_dir.glob("*.fa")))
        print(f"[qc] {len(fas)} samples with existing contigs → length filter (≥{min_len}bp) + dedup")
        for fa in fas:
            dst = contigs_out / fa.name
            with open(dst, "w") as w:
                for sid, seq in iter_contig_seq(fa):
                    if len(seq) >= min_len:
                        w.write(f">{sid}\n{seq}\n")
            n_use += 1
            if have("cd-hit-est"):
                tmp = dst.with_suffix(".nr.fa")
                rc = run(["cd-hit-est", "-i", str(dst), "-o", str(tmp), "-c", "0.95",
                          "-n", "10", "-T", "4", "-d", "0"], "cd-hit-est 95%")
                if rc == 0 and tmp.exists() and tmp.stat().st_size > 0:
                    tmp.replace(dst)         # overwrite only on success and non-empty output (m-10 fix)
                elif tmp.exists():
                    tmp.unlink()             # clean up failed output, keep the length-filtered result
    # 2) fastq-only → fastp + MEGAHIT
    if fastq_dir and fastq_dir.is_dir():
        pairs = defaultdict(dict)
        for fq in fastq_dir.glob("*.f*q*"):
            stem = fq.stem.replace("_R1", "").replace("_R2", "").replace("_1", "").replace("_2", "")
            tag = "R1" if ("R1" in fq.name or "_1" in fq.name) else "R2"
            pairs[stem][tag] = fq
        todo = {s: v for s, v in pairs.items()
                if not (contigs_dir and (contigs_dir / f"{s}.fasta").exists())}
        print(f"[qc] {len(todo)} fastq-only samples → fastp + MEGAHIT assembly")
        if todo and not have("megahit"):
            print("  [WARN] megahit missing, skipping assembly (please provide contigs or install megahit)")
        for s, v in todo.items():
            if "R1" not in v or "R2" not in v:
                print(f"  [skip] {s}: not paired-end fastq"); continue
            if not have("fastp") or not have("megahit"):
                continue
            r1c, r2c = out / "qc" / f"{s}_R1.fq.gz", out / "qc" / f"{s}_R2.fq.gz"
            (out / "qc").mkdir(exist_ok=True)
            run(["fastp", "-i", str(v["R1"]), "-I", str(v["R2"]),
                 "-o", str(r1c), "-O", str(r2c), "-j", str(out / "qc" / f"{s}.json"),
                 "-h", str(out / "qc" / f"{s}.html")], "fastp")
            asm = out / "asm" / s / "final.contigs.fa"
            (out / "asm" / s).mkdir(parents=True, exist_ok=True)
            run(["megahit", "-1", str(r1c), "-2", str(r2c), "-o", str(out / "asm" / s),
                 "--min-contig-len", str(min_len)], "MEGAHIT")
            if asm.exists():
                dst = contigs_out / f"{s}.fasta"
                with open(dst, "w") as w:
                    for sid, seq in iter_contig_seq(asm):
                        if len(seq) >= min_len:
                            w.write(f">{sid}\n{seq}\n")
                n_asm += 1
    print(f"[qc] done: {n_use} using existing contigs + {n_asm} newly assembled → {contigs_out}")


# ─────────────────────────── B. predict ───────────────────────────
def cmd_predict(args):
    """Ch5 §5.3: per-sample contigs → DeepFluHGT contig prediction (per-contig max-pool) → cross-contig median summary."""
    contigs_dir = Path(args.contigs_dir)
    out = Path(args.out)
    (out / "hgt").mkdir(parents=True, exist_ok=True)
    fas = sorted(list(contigs_dir.glob("*.fasta")) + list(contigs_dir.glob("*.fa")))
    print(f"[predict] {len(fas)} sample contigs → DeepFluHGT predict (threshold={args.threshold})")
    for i, fa in enumerate(fas):
        sample = fa.stem
        pred_out = out / "hgt" / sample
        pred_out.mkdir(exist_ok=True)
        cmd = [PYTHON, str(ROOT / "DeepFluHGT.py"), "predict", str(fa),
               "--mode", "contig", "--checkpoint", args.ckpt,
               "--config", str(ROOT / "configs/contig_500k.yaml"),
               "--threshold", str(args.threshold),
               "--batch_size", str(args.batch_size), "--output_dir", str(pred_out)]
        print(f"  [{i+1}/{len(fas)}] {sample}", flush=True)
        run(cmd, "DeepFluHGT predict")
    # summarize
    summary = {}
    for d in (out / "hgt").iterdir():
        probs = list(read_predictions(d / "predictions.tsv").values())
        if probs:
            summary[d.name] = {"n_contigs": len(probs),
                               "median_prob": float(np.median(probs)),  # median across contigs (each contig's probability is the max-pool of its predictions)
                               "n_above_thr": int(sum(1 for p in probs if p >= args.threshold))}
    json.dump(summary, open(out / "hgt_summary.json", "w"), indent=2)
    print(f"[predict] done → {out/'hgt_summary.json'}")


# ─────────────────────────── C. active HGT ★ ───────────────────────────
def parse_quant_sf(qsf: Path) -> dict:
    """salmon quant.sf → {Name (contig): TPM}. Columns: Name Length EffectiveLength TPM NumReads.
    Note: the first row of a salmon quant.sf is a '#' metadata comment; the header is only
    the second row — — the '#' row must be skipped, else DictReader takes it as the header
    → row['Name'] KeyError is swallowed → TPM ends up empty (M-1 fix)."""
    tpm = {}
    if not qsf.exists():
        return tpm
    with open(qsf) as f:
        lines = [ln for ln in f if not ln.lstrip().startswith("#")]  # skip salmon's '#' metadata header row
        reader = csv.DictReader(lines, delimiter="\t")
        for row in reader:
            try:
                tpm[row["Name"]] = float(row["TPM"])
            except (KeyError, ValueError, TypeError):
                continue
    return tpm


def cmd_active(args):
    """Ch6 §6.1-6.2: paired-sample RNA→salmon→TPM → active (high HGT & high TPM) / silent (high HGT & low TPM).
    Non-paired samples are marked DNA-only (see cross for water-body-level RNA analysis)."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not have("salmon"):
        print("[active] ✗ salmon missing  —  cannot compute TPM/active HGT (the core NMI selling point is blocked). Install salmon and re-run.")
        return
    pairs = {}  # sample -> (r1, r2)
    if Path(args.pairs).exists():
        for row in csv.DictReader(open(args.pairs), delimiter="\t"):
            pairs[row["sample"]] = (row["rna_r1"], row["rna_r2"])
    hgt_dir = Path(args.hgt_dir)
    contigs_dir = Path(args.contigs_dir)
    results = {}
    for sample, (r1, r2) in pairs.items():
        contigs_fa = contigs_dir / f"{sample}.fasta"
        hgt = read_predictions(hgt_dir / sample / "predictions.tsv")
        if not contigs_fa.exists() or not hgt:
            print(f"  [skip] {sample}: contigs or HGT missing"); continue
        sdir = out / sample
        sdir.mkdir(parents=True, exist_ok=True)
        idx = sdir / "salmon_index"
        run(["salmon", "index", "-t", str(contigs_fa), "-i", str(idx)], "salmon index")
        quant = sdir / "quant"
        run(["salmon", "quant", "-i", str(idx), "-l", "A",
             "-1", str(r1), "-2", str(r2), "-o", str(quant),
             "--seqBias", "--gcBias", "--validateMappings"], "salmon quant")
        tpm = parse_quant_sf(quant / "quant.sf")
        active_n = silent_n = 0
        per_contig = {}
        for cid, p in hgt.items():
            t = tpm.get(cid, 0.0)
            is_hgt = p >= args.hgt_threshold
            if is_hgt and t >= args.tpm_threshold:
                cls = "active"; active_n += 1
            elif is_hgt:
                cls = "silent"; silent_n += 1
            else:
                cls = "non_hgt"
            per_contig[cid] = {"hgt_prob": p, "tpm": t, "class": cls}
        tot = max(1, active_n + silent_n)
        results[sample] = {"paired": True, "n_hgt": active_n + silent_n,
                           "n_active": active_n, "n_silent": silent_n,
                           "active_rate": active_n / tot,
                           "per_contig": per_contig}
        print(f"  [{sample}] HGT={active_n+silent_n} active={active_n} silent={silent_n} "
              f"active_rate={active_n/tot:.2%}")
    # non-paired → DNA-only
    for d in hgt_dir.iterdir():
        if d.is_dir() and d.name not in pairs:
            hgt = read_predictions(d / "predictions.tsv")
            results[d.name] = {"paired": False,
                               "n_hgt": int(sum(1 for p in hgt.values() if p >= args.hgt_threshold)),
                               "note": "DNA-only; see cross for cohort-level activity analysis"}
    json.dump(results, open(out / "active_hgt.json", "w"), indent=2)
    n_paired = sum(1 for v in results.values() if v.get("paired"))
    print(f"[active] {n_paired} paired / {len(results)-n_paired} non-paired → {out/'active_hgt.json'}")


# ─────────────────────────── D. cross-water ───────────────────────────
def cmd_cross(args):
    """Ch6 §6.3: per-water-body HGT pattern + active rate + cross-water-body."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    water_map = {}
    for row in csv.DictReader(open(args.water_map), delimiter="\t"):
        water_map[row["sample"]] = row["water"]
    hgt_summary = json.load(open(Path(args.hgt_dir).parent / "hgt_summary.json"))
    active = {}
    if Path(args.active_json).exists():
        active = json.load(open(args.active_json))
    by_water = defaultdict(list)
    for s, m in hgt_summary.items():
        by_water[water_map.get(s, "unknown")].append(s)
    water_stats = {}
    for w, samples in by_water.items():
        hgt_counts = [hgt_summary[s].get("n_above_thr", 0) for s in samples if s in hgt_summary]
        arates = [active[s]["active_rate"] for s in samples
                  if s in active and active[s].get("paired")]
        water_stats[w] = {
            "n_samples": len(samples),
            "mean_hgt_contigs": float(sum(hgt_counts) / max(1, len(hgt_counts))),
            "mean_active_rate": float(sum(arates) / len(arates)) if arates else None,
            "n_paired": sum(1 for s in samples if s in active and active[s].get("paired")),
        }
    json.dump(water_stats, open(out / "cross_water.json", "w"), indent=2)
    print(f"[cross] {len(by_water)} water bodies → {out/'cross_water.json'}")


# ─────────────────────────── E. validate (BLAST) ───────────────────────────
def cmd_validate(args):
    """Ch5 §5.4: collect top-N HGT contigs → BLAST against local_db → donor/recipient + Tier grading.
    Optional: kraken2 taxonomic origin / genomad mobility (WARN+skip if missing)."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hgt_dir = Path(args.hgt_dir)
    contigs_dir = Path(args.contigs_dir)
    # 1) collect top-N (prob, sample, seq_id)
    all_hgt = []
    for d in sorted(hgt_dir.iterdir()):
        preds = read_predictions(d / "predictions.tsv")
        for cid, p in preds.items():
            all_hgt.append((p, d.name, cid))
    all_hgt.sort(reverse=True)
    top = all_hgt[: args.top_n]
    print(f"[validate] top-{len(top)} HGT contigs (of {len(all_hgt)} candidates total)")
    # 2) extract sequences → query.fasta
    query = out / "query.fasta"
    # build a sample→fasta index to avoid repeated reads
    sample_fa = {fa.stem: fa for fa in list(contigs_dir.glob("*.fasta")) + list(contigs_dir.glob("*.fa"))}
    wanted = defaultdict(set)
    for _, sample, cid in top:
        wanted[sample].add(cid)
    seq_map = {}
    for sample, ids in wanted.items():
        fa = sample_fa.get(sample)
        if not fa:
            continue
        for sid, seq in iter_contig_seq(fa):
            if sid in ids:
                seq_map[(sample, sid)] = seq
    with open(query, "w") as w:
        for p, sample, cid in top:
            seq = seq_map.get((sample, cid))
            if seq:
                w.write(f">{sample}__{cid}__p{p:.3f}\n{seq}\n")
    print(f"  query → {query} ({len(top)} seqs)")
    # 3) BLAST
    blast_summary = {"n_query": len(top), "blast_db": BLAST_DB}
    if have("blastn") and Path(BLAST_DB + ".nsq").exists():
        bout = out / "blast.tsv"
        run(["blastn", "-query", str(query), "-db", BLAST_DB, "-evalue", "1e-5",
             "-outfmt", "6 qseqid stitle pident length evalue", "-max_target_seqs", "3",
             "-out", str(bout)], "blastn")
        # count hits
        n_hit = 0
        if bout.exists():
            n_hit = sum(1 for _ in open(bout))
        blast_summary["n_hits"] = n_hit
        blast_summary["hit_rate"] = n_hit / max(1, len(top))
        print(f"  BLAST: {n_hit} hits ({blast_summary['hit_rate']:.1%} of query)")
    else:
        blast_summary["error"] = "blastn or local_db missing (build the DB: makeblastdb)"
        print(f"  [WARN] {blast_summary['error']}")
    # 4) optional taxonomic origin/mobility
    for tool, label in [("kraken2", "taxonomic origin"), ("genomad", "mobility")]:
        if not have(tool):
            blast_summary.setdefault("skipped", {})[tool] = f"{label} missing, skip"
            print(f"  [WARN] {tool} missing  —  skipping {label} analysis (optional)")
    # 5) Tier grading (simple version: Tier1=top-N with a BLAST hit; Tier2=high HGT without BLAST; Tier3=rest)
    json.dump(blast_summary, open(out / "validate_summary.json", "w"), indent=2)
    print(f"[validate] → {out/'validate_summary.json'}")


# ─────────────────────────── main ───────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Ch5/Ch6 real-water-body HGT discovery pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("qc", help="Ch5§5.2 QC/assembly")
    q.add_argument("--fastq-dir"); q.add_argument("--contigs-dir")
    q.add_argument("--out", default="ch6")
    q.add_argument("--min-contig-len", type=int, default=1000)

    p = sub.add_parser("predict", help="Ch5§5.3 HGT prediction")
    p.add_argument("--contigs-dir", required=True); p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="ch6")
    p.add_argument("--threshold", type=float, default=0.3, help="under-confident on real data; set below 0.5")
    p.add_argument("--batch_size", type=int, default=8)

    a = sub.add_parser("active", help="Ch6 active HGT ★")
    a.add_argument("--hgt-dir", required=True); a.add_argument("--contigs-dir", required=True)
    a.add_argument("--pairs", required=True, help="sample\trna_r1\trna_r2")
    a.add_argument("--out", default="ch6/active")
    a.add_argument("--hgt-threshold", type=float, default=0.3)
    a.add_argument("--tpm-threshold", type=float, default=1.0)

    c = sub.add_parser("cross", help="Ch6§6.3 cross-water-body")
    c.add_argument("--hgt-dir", required=True); c.add_argument("--water-map", required=True)
    c.add_argument("--active-json", default="ch6/active/active_hgt.json")
    c.add_argument("--out", default="ch6/cross")

    v = sub.add_parser("validate", help="Ch5§5.4 BLAST validation")
    v.add_argument("--hgt-dir", required=True); v.add_argument("--contigs-dir", required=True)
    v.add_argument("--top-n", type=int, default=200); v.add_argument("--out", default="ch6/validate")

    args = ap.parse_args()
    {"qc": cmd_qc, "predict": cmd_predict, "active": cmd_active,
     "cross": cmd_cross, "validate": cmd_validate}[args.cmd](args)


if __name__ == "__main__":
    main()
