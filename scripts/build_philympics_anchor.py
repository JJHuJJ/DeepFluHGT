#!/usr/bin/env python3
"""build_philympics_anchor.py — the fourth V7 real anchor (Philympics 2021) data build.

verbatim replication 8 protocol (benchmark/philympics/v6.8_usage/ of
phase0_7_sequence_holdout.py + build_v5_4_dataset.py), with only two replacements:
  1. GBK_DIR → this repo's benchmark/philympics/source/.../genbank (78 gb.gz local files);
  2. minimap2 training recipient target library → **V7 trainingset recipient**
     (simulated_data/Contig_500k/genome_split_manifest.json of
      microbe.train.files, 9,242 genome; 8 use 11,552 allquantity).
filterstrand: 275 event → minimap2 asm5 best-hit identity <50% retain →
same-genome GC (|ΔGC|<0.05) + length-exact-matched non-prophage control →
control_len>0 ofpairedevent (v6.8: 275→85→81).

CPU-only (minimap2 multi-thread), never touches the GPU.
output benchmark/philympics/v7_anchor/:
  v7_all_prophages.fasta 275 extracted events (reconciled against 8.0's phase0_7/all_prophages.fasta)
  v7_recipients_train.fa concatenated target library of 9,242 training recipients (deletable; --keep_target keeps it)
  minimap2.paf originalratiotowardoutput
  v7_filter_summary.json identity stratified + with 8 control
  v7_independent_prophage.fasta / v7_control.fasta / v7_paired.fasta (evaluationinput)
  v7_matching_stats.json per-event pairing metadata (for paired analysis / shortcut checks)

usage: python scripts/build_philympics_anchor.py [--threads 32]
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from Bio import SeqIO

ROOT = Path(__file__).resolve().parent.parent
PHIL = ROOT / "benchmark/philympics"
GBK_DIR = PHIL / "source/ProphagePredictionComparisons/genbank"
COORDS = GBK_DIR / "prophageCoords.tsv"
V68_ALL = PHIL / "v6.8_usage/phase0_7/all_prophages.fasta"
MG_DIR = Path("./microbe_genomes")
SPLIT_MANIFEST = ROOT / "simulated_data/Contig_500k/genome_split_manifest.json"
OUT_DIR = PHIL / "v7_anchor"

IDENTITY_THRESHOLD = 0.50 # <50% = sequence-level independent (v6.8 samevalue)


def log(msg):
    print(msg, flush=True)


def gc(seq):
    s = seq.upper()
    n = sum(s.count(b) for b in "ACGT")
    return (s.count("G") + s.count("C")) / n if n else 0.0


# -- 1. event extraction (v6.8 phase0_7.extract_all_prophages verbatim) ------------------ #

def parse_all_coords():
    events = []
    with open(COORDS) as f:
        next(f)
        for line in f:
            p = line.strip().split("\t")
            if len(p) < 6:
                continue
            events.append({"file": p[0], "org": p[1], "acc": p[2],
                           "start": int(p[4].lstrip(">")), "end": int(p[5].lstrip(">")),
                           "n_genes": int(p[6]) if len(p) > 6 else 0})
    return events


def extract_all_prophages(events):
    by_file = defaultdict(list)
    for e in events:
        by_file[e["file"]].append(e)
    sequences, event_meta = {}, []
    for fname, file_events in by_file.items():
        path = GBK_DIR / fname
        if not path.exists():
            for cand in GBK_DIR.glob(f"*{fname.split('_')[0]}*"):
                if cand.name.endswith(".gz"):
                    path = cand
                    break
        try:
            with gzip.open(path, "rt") as fh:
                records = {r.id: str(r.seq) for r in SeqIO.parse(fh, "genbank")}
        except Exception as ex:
            log(f" ERROR loading {fname}: {ex}")
            continue
        for e in file_events:
            acc = e["acc"]
            seq = None
            for rid, rseq in records.items():
                if acc in rid or rid in acc or acc == rid:
                    seq = rseq
                    break
            if seq is None and records:
                seq = list(records.values())[0]
            if seq is None:
                continue
            start = max(0, e["start"] - 1)
            end = min(len(seq), e["end"])
            prophage_seq = seq[start:end]
            if len(prophage_seq) < 100:
                continue
            idx = len(event_meta)
            header = f"prophage_{idx}_{e['org'].replace(' ', '_')[:30]}_{acc}_{e['start']}_{e['end']}"
            sequences[idx] = (header, prophage_seq)
            event_meta.append({**e, "idx": idx, "length": len(prophage_seq),
                               "gc": round(gc(prophage_seq), 4)})
    return sequences, event_meta


def verify_vs_v68(sequences):
    """reconcile per-item against 8.0's all_prophages.fasta (headers + sequences)."""
    v68 = {}
    name = None
    for ln in open(V68_ALL):
        ln = ln.strip()
        if ln.startswith(">"):
            name = ln[1:]
            v68[name] = []
        elif name:
            v68[name].append(ln)
    v68 = {k: "".join(v) for k, v in v68.items()}
    mine = {h: s for h, s in sequences.values()}
    n_common = len(set(mine) & set(v68))
    seq_eq = sum(1 for h in set(mine) & set(v68) if mine[h].upper() == v68[h].upper())
    log(f"[verify] 8 all_prophages={len(v68)} localextract={len(mine)} "
        f"totalsame header={n_common} sequencealletc.={seq_eq}")
    assert n_common == len(mine) == len(v68) and seq_eq == n_common, \
        "extraction inconsistent with 8.0 — stop (replication fidelity hard assert)"
    return True


# -- 2. V7 recipient target library + minimap2 (v6.8 phase0_7 sameparameters) ------------------ #

def build_recipient_target(threads: int):
    manifest = json.loads(SPLIT_MANIFEST.read_text())
    files = manifest["microbe"]["train"]["files"]
    log(f"[target] V7 training recipient: {len(files)} genome "
        f"(genome_split_manifest microbe.train)")
    target = OUT_DIR / "v7_recipients_train.fa"
    if target.exists() and target.stat().st_size > 1e8:
        log(f"[target] alreadyat ({target.stat().st_size//1e9:.1f} GB), re-use")
        return target, len(files)
    missing = [f for f in files if not (MG_DIR / f).exists()]
    assert not missing, f"recipient genomemissing {len(missing)}: {missing[:5]}"
    t0 = time.time()
    with open(target, "w") as out:
        for i, f in enumerate(files):
            with open(MG_DIR / f) as fh:
                out.write(fh.read())
            if (i + 1) % 1000 == 0:
                log(f" ...{i+1}/{len(files)} ({time.time()-t0:.0f}s)")
    log(f"[target] built: {len(files)} genomes, "
        f"{target.stat().st_size/1e9:.1f} GB ({time.time()-t0:.0f}s)")
    return target, len(files)


def run_minimap2(query_fasta, target_fasta, threads: int):
    paf = OUT_DIR / "minimap2.paf"
    cmd = ["minimap2", "-t", str(threads), "-x", "asm5", "--secondary=no",
           "-p", "0.1", "-N", "1", str(target_fasta), str(query_fasta)]
    log(f"[minimap2] {' '.join(cmd[:8])} ... (target {target_fasta.stat().st_size/1e9:.1f}GB)")
    t0 = time.time()
    with open(paf, "w") as out:
        r = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        log(f" minimap2 stderr: {r.stderr[:500]}")
    log(f"[minimap2] done in {(time.time()-t0)/60:.1f} min → {paf.name}")

    best_hit = {}
    for line in open(paf):
        p = line.strip().split("\t")
        if len(p) < 12:
            continue
        qname, nmatch, block_len = p[0], int(p[9]), int(p[10])
        if block_len < 500: # 8 samethreshold: ignore <500bp ratiotoward
            continue
        identity = nmatch / block_len if block_len > 0 else 0
        try:
            qidx = int(qname.split("_")[1])
        except (ValueError, IndexError):
            continue
        if qidx not in best_hit or identity > best_hit[qidx][0]:
            best_hit[qidx] = (identity, block_len, p[5])
    return best_hit, paf


# -- 3. GC/lengthmatch control (v6.8 build_v5_4_dataset.find_control verbatim) ----─ #

def find_control(seq, prophage_coords, target_len, target_gc, tol=0.05):
    contig_len = len(seq)
    if contig_len < target_len * 2:
        return None
    step = max(target_len // 4, 500)
    best = None
    best_diff = 999
    for pos in range(0, contig_len - target_len, step):
        end = pos + target_len
        if any(not (end <= ps or pos >= pe) for ps, pe in prophage_coords):
            continue
        region = str(seq[pos:end])
        if len(region) < target_len // 2:
            continue
        g = gc(region)
        d = abs(g - target_gc)
        if d <= tol and d < best_diff:
            best = (region, g, pos)
            best_diff = d
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--keep_target", action="store_true",
                    help="retain ~39GB target library (defaultratiotowardafterdelete)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1) extract the 275 events + reconcile against 8.0
    events = parse_all_coords()
    log(f"[coords] eventtotalcount: {len(events)}")
    sequences, event_meta = extract_all_prophages(events)
    log(f"[extract] extract: {len(sequences)} sequence")
    query = OUT_DIR / "v7_all_prophages.fasta"
    with open(query, "w") as f:
        for idx, (header, seq) in sequences.items():
            f.write(f">{header}\n{seq}\n")
    verify_vs_v68(sequences)

    # 2) V7 recipient target library + minimap2
    target, n_recip = build_recipient_target(args.threads)
    best_hit, paf = run_minimap2(query, target, args.threads)
    log(f"[minimap2] ≥500bp hit withinevent: {len(best_hit)}/{len(event_meta)}")

    for m in event_meta:
        if m["idx"] in best_hit:
            ident, blen, tgt = best_hit[m["idx"]]
            m.update(best_identity=round(ident, 4), best_block_len=blen,
                     best_target=tgt)
        else:
            m.update(best_identity=0.0, best_block_len=0, best_target="none")

    tiers = defaultdict(list)
    for m in event_meta:
        i = m["best_identity"]
        k = (">95%" if i > 0.95 else "70-95%" if i > 0.70 else
             "<70%" if i > 0.50 else "<50%")
        tiers[k].append(m["idx"])
    tiers["no_hit"] = [m["idx"] for m in event_meta if m["best_target"] == "none"
                       and m["best_identity"] == 0.0]
    n_ind = sum(1 for m in event_meta if m["best_identity"] < IDENTITY_THRESHOLD)
    log("[tiers] " + " ".join(f"{k}:{len(v)}" for k, v in tiers.items()
                               if k != "no_hit"))
    log(f"[filter] independent (<{IDENTITY_THRESHOLD:.0%}): {n_ind} "
        f"(v6.8 toward 11,552 recipients for 85)")

    # 3) control pairing (verbatim replication of v6.8 build_v5_4_dataset's core logic)
    random.seed(42)
    independent = [m for m in event_meta if m["best_identity"] < IDENTITY_THRESHOLD]
    by_file = defaultdict(list)
    for e in independent:
        by_file[e["file"]].append(e)

    prophage_records, control_records, stats = [], [], []
    skipped = 0
    for fname, file_events in by_file.items():
        path = GBK_DIR / fname
        if not path.exists():
            for cand in GBK_DIR.glob(f"*{fname.split('_')[0]}*"):
                if cand.name.endswith(".gz"):
                    path = cand
                    break
        try:
            with gzip.open(path, "rt") as fh:
                records = {r.id: str(r.seq) for r in SeqIO.parse(fh, "genbank")}
        except Exception as ex:
            log(f" ERROR {fname}: {ex}")
            skipped += len(file_events)
            continue
        pp_by_contig = defaultdict(list)
        for e in file_events:
            pp_by_contig[e["acc"]].append((e["start"], e["end"]))
        for e in file_events:
            acc = e["acc"]
            seq = None
            for rid, rseq in records.items():
                if acc in rid or rid in acc or acc == rid:
                    seq = rseq
                    break
            if seq is None and records:
                seq = list(records.values())[0]
            if seq is None:
                skipped += 1
                continue
            start = max(0, e["start"] - 1)
            end = min(len(seq), e["end"])
            prophage_seq = seq[start:end]
            if len(prophage_seq) < 200:
                skipped += 1
                continue
            p_gc = gc(prophage_seq)
            sid = f"prophage_{e['org'].replace(' ', '_')[:30]}_{acc}_{e['start']}_{e['end']}"
            prophage_records.append((sid, prophage_seq))
            ctrl = find_control(seq, pp_by_contig[acc], len(prophage_seq), p_gc)
            if ctrl:
                cseq, cgc, cpos = ctrl
                cid = f"control_{e['org'].replace(' ', '_')[:30]}_{acc}_{cpos}"
                control_records.append((cid, cseq))
                stats.append({"org": e["org"], "acc": acc,
                              "identity": round(e["best_identity"], 4),
                              "prophage_len": len(prophage_seq),
                              "prophage_gc": round(p_gc, 4),
                              "control_len": len(cseq),
                              "control_gc": round(cgc, 4),
                              "gc_diff": round(abs(p_gc - cgc), 4),
                              "prophage_id": sid, "control_id": cid})
            else:
                stats.append({"org": e["org"], "acc": acc,
                              "identity": round(e["best_identity"], 4),
                              "prophage_len": len(prophage_seq),
                              "prophage_gc": round(p_gc, 4),
                              "control_len": 0, "control_gc": 0, "gc_diff": 999,
                              "prophage_id": sid, "control_id": None})

    (OUT_DIR / "v7_independent_prophage.fasta").write_text(
        "".join(f">{s}\n{q}\n" for s, q in prophage_records))
    (OUT_DIR / "v7_control.fasta").write_text(
        "".join(f">{c}\n{q}\n" for c, q in control_records))
    (OUT_DIR / "v7_paired.fasta").write_text(
        "".join(f">{s}\n{q}\n" for s, q in prophage_records + control_records))
    (OUT_DIR / "v7_matching_stats.json").write_text(json.dumps(stats, indent=2))

    n_paired = sum(1 for s in stats if s["control_len"] > 0)
    summary = {
        "protocol": "v6.8 phase0_7+build_v5_4 replicated; recipients swapped to the V7 train split",
        "n_recipients": n_recip, "recipient_source": str(SPLIT_MANIFEST),
        "minimap2": {"preset": "asm5", "threads": args.threads,
                     "min_block": 500, "identity_threshold": IDENTITY_THRESHOLD},
        "n_events": len(event_meta), "n_extracted": len(sequences),
        "n_independent": len(independent), "n_prophage_written": len(prophage_records),
        "n_control": len(control_records), "n_paired": n_paired,
        "n_species": len({s["org"] for s in stats if s["control_len"] > 0}),
        "n_genera": len({s["org"].split()[0] for s in stats if s["control_len"] > 0}),
        "skipped": skipped,
        "v68_reference": {"n_independent": 85, "n_paired": 81, "n_species": 33,
                          "auroc_mean_pool": 0.616},
        "tier_counts": {k: len(v) for k, v in tiers.items()},
        "elapsed_s": time.time() - t0,
    }
    (OUT_DIR / "v7_filter_summary.json").write_text(json.dumps(summary, indent=2))

    log(f"\n[build] prophage={len(prophage_records)} control={len(control_records)} "
        f"paired={n_paired} species={summary['n_species']} "
        f"(v6.8: 85/81/33)")
    log(f"[build] done in {(time.time()-t0)/60:.1f} min → {OUT_DIR}")

    if not args.keep_target:
        target.unlink()
        log(f"[cleanup] deletetarget library {target.name} "
            f"(paf alreadyarchived, verifiable)")


if __name__ == "__main__":
    sys.exit(main())
