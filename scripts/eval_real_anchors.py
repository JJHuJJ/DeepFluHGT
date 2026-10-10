#!/usr/bin/env python3
# scripts/eval_real_anchors.py
"""DeepFluHGT  —  CPU inference evaluation of the M0 checkpoint on three real-data anchors.

Faithfully replicates the v6.8 scripts/eval_external_benbow.py evaluation protocol:
  sliding window (window=3000, overlap=300) → sigmoid probability per window →
  contig-level aggregation (max/mean/median/top3_mean; top3_mean acts as a soft NMS) →
  AUC/AUPRC/F1/ECE with positive/negative stratification.

Three anchors:
  1. benbow   —  benchmark/literature_gi/repo/test_data/benbow_test_data.fasta (566 item)
  2. iv4      —  benchmark/islandviewer4/, replicating the 300 pos + 300 neg
               GC-matched sampling of v6.8 build_real_contig_benchmark.py (seed=42)
  3. ice      —  benchmark/iceberg2/raw_downloads/ICE_seq_experimental.fas (207 item
               experimentally validated ICEs, all positive); negatives = 207 intervals
               sampled from non-GI regions of real IV4 genomes, matched by exact 1:1
               length + GC (±3%) (seed=42)

Hard constraints (CPU-only, coexisting with M0 training on the GPU):
  CUDA_VISIBLE_DEVICES="" / OMP_NUM_THREADS=4 / torch.set_num_threads(4)

usage:
  python scripts/eval_real_anchors.py --anchor benbow --win_cap 4
"""
from __future__ import annotations

import os
# ── CPU hard constraint: must be set before importing torch ──
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import sys
import json
import time
import random
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

IV4_DIR = ROOT / "benchmark/islandviewer4"
BENBOW_FASTA = ROOT / "benchmark/literature_gi/repo/test_data/benbow_test_data.fasta"
ICE_FASTA = ROOT / "benchmark/iceberg2/raw_downloads/ICE_seq_experimental.fas"

DEFAULT_CKPT = ROOT / "checkpoints/Contig_500k_ntv3only/best_model.pt"
DEFAULT_CFG = ROOT / "configs/contig_500k_ntv3only.yaml"

NT_FALLBACK = ROOT / "model/InstaDeepAI/NTv3_650M_pre"


# ═══════════════════════════════════════════════════════════════════════════ #
#  dataread (v6.8 verbatim replication)
# ═══════════════════════════════════════════════════════════════════════════ #

def read_benbow(fasta_path) -> dict:
    """Read the Benbow FASTA → {seq_id: (label, sequence)}.  (v6.8 eval_external_benbow.py verbatim replication)

    Header format: >accession_GI_n|label|accession:start-end organism
    """
    samples = {}
    cur_id, cur_label, buf = None, None, []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if cur_id:
                    samples[cur_id] = (cur_label, "".join(buf))
                parts = line[1:].split("|")
                cur_id = parts[2].split()[0] if len(parts) > 2 and parts[2].strip() else parts[0]
                cur_label = int(parts[1]) if len(parts) > 1 else 0
                buf = []
            else:
                buf.append(line.upper())
        if cur_id:
            samples[cur_id] = (cur_label, "".join(buf))
    return samples


def gc(seq: str) -> float:
    """(v6.8 build_real_pe_benchmark.py verbatim replication)"""
    s = seq.upper()
    g = s.count("G") + s.count("C")
    return 100.0 * g / max(1, len(s))


def collect_genomes(iv_dir: Path):
    """Yield (gid, genome_seq, [(gi_start, gi_end), ...]).  (v6.8 verbatim replication)"""
    from Bio import SeqIO
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


def build_iv4_300_300(iv_dir: Path, window: int = 3000, gc_tol: float = 3.0,
                      max_per_side: int = 300) -> dict:
    """Replicates the 300+300 sampling of v6.8 build_real_contig_benchmark.py (verbatim logic, seed=42).

    return {seq_id: (label, sequence)}  —  same structure as Benbow.
    """
    pos_contigs, neg_contigs = [], []
    n_genomes = 0
    for gid, genome, gi_coords in collect_genomes(iv_dir):
        n_genomes += 1
        glen = len(genome)
        gi_sorted = sorted((max(0, s), min(glen, e)) for s, e in gi_coords)
        for (s, e) in gi_sorted:
            if e - s >= window:
                pos_contigs.append((f"{gid}_GI_{s}_{e}", genome[s:e]))
        cursor = 0
        for (s, e) in gi_sorted:
            while cursor + window <= s:
                w = genome[cursor:cursor + window]
                if "N" not in w:
                    neg_contigs.append((f"{gid}_flank_{cursor}", w))
                cursor += window
            cursor = max(cursor, e)
        while cursor + window <= glen:
            w = genome[cursor:cursor + window]
            if "N" not in w:
                neg_contigs.append((f"{gid}_flank_{cursor}", w))
            cursor += window

    pos_gcs = [gc(s) for _, s in pos_contigs]
    pos_gc_mean = sum(pos_gcs) / max(1, len(pos_gcs))
    neg_contigs.sort(key=lambda x: abs(gc(x[1]) - pos_gc_mean))
    neg_matched = [(i, s) for i, s in neg_contigs if abs(gc(s) - pos_gc_mean) <= gc_tol]
    if len(neg_matched) < 50:
        neg_matched = neg_contigs[:200]
    random.seed(42)
    pos_s = random.sample(pos_contigs, min(len(pos_contigs), max_per_side))
    neg_s = random.sample(neg_matched, min(len(neg_matched), max_per_side))

    samples = {}
    for sid, s in pos_s:
        samples[sid] = (1, s)
    for sid, s in neg_s:
        samples[sid] = (0, s)
    pos_gc = sum(gc(s) for _, s in pos_s) / len(pos_s)
    neg_gc = sum(gc(s) for _, s in neg_s) / len(neg_s)
    print(f"[IV4] genomes={n_genomes} pos_pool={len(pos_contigs)} "
          f"neg_pool={len(neg_contigs)} neg_matched={len(neg_matched)}")
    print(f"[IV4] sampled: pos={len(pos_s)} neg={len(neg_s)}  "
          f"pos GC={pos_gc:.1f}% neg GC={neg_gc:.1f}% |Δ|={abs(pos_gc-neg_gc):.1f}%")
    return samples


# ═══════════════════════════════════════════════════════════════════════════ #
#  ICEberg3 anchor: 207 experimentally validated ICEs + GC/length-matched real-genome negative controls
# ═══════════════════════════════════════════════════════════════════════════ #

def read_iceberg(fasta_path) -> list:
    """>ICEberg|id|name|GenBank|acc|start..end desc → [(iceberg_id, accession, seq)]"""
    out, cur, buf = [], None, []
    for ln in open(fasta_path):
        ln = ln.strip()
        if ln.startswith(">"):
            if cur is not None:
                out.append(cur + ("".join(buf).upper(),))
            parts = ln[1:].split("|")
            iid = parts[1] if len(parts) > 1 else ln[1:]
            acc = parts[4] if len(parts) > 4 else ""
            cur = (iid, acc)
            buf = []
        else:
            buf.append(ln)
    if cur is not None:
        out.append(cur + ("".join(buf).upper(),))
    return out


def non_gi_intervals(glen: int, gi_coords):
    """Genome GI coordinates → merged list of non-GI intervals [(start,end)]."""
    merged = []
    for s, e in sorted((max(0, s), min(glen, e)) for s, e in gi_coords):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out, cur = [], 0
    for s, e in merged:
        if s > cur:
            out.append((cur, s))
        cur = max(cur, e)
    if cur < glen:
        out.append((cur, glen))
    return out


def build_ice_anchor(ice_records, iv_dir: Path, n_neg: int, seed: int = 42):
    """ICEs all positive + non-GI regions of real IV4 genomes as negative controls.

    Negative-sampling strategy (as documented in the report):
      - length: exactly 1:1 with the paired ICE element length (the same L bp)
      - GC: candidate-region GC within ≤ 3% of the paired ICE GC; if none of the
        8 attempts qualifies, the best tried candidate is recorded as a fallback
      - source: non-GI (non-predicted-island) intervals from real complete
        bacterial/archaeal genomes (50 items) in IV4, ordered by distance to each
        genome's overall non-GI GC and cycled through in turn; already-used
        intervals never overlap or get reused
      - no ambiguous bases (ACGT only); fully deterministic with seed=42
    """
    rng = random.Random(seed)
    genomes = []  # (gid, seq, intervals, non_gi_gc)
    for gid, seq, gi in collect_genomes(iv_dir):
        iv = non_gi_intervals(len(seq), gi)
        total = sum(e - s for s, e in iv)
        if total < 10000:
            continue
        g = sum(seq[s:e].count("G") + seq[s:e].count("C") for s, e in iv)
        genomes.append((gid, seq, iv, 100.0 * g / total))
    total_bp = sum(e - s for _, _, iv, _ in genomes for s, e in iv)
    print(f"[ICE-neg] genome pool: {len(genomes)} IV4 genomes, "
          f"non-GI bp total={total_bp}")

    occupied = defaultdict(list)  # gid → [(s,e)] already occupied
    pos_samples, neg_samples, neg_meta = {}, {}, []
    pos_sorted = list(ice_records)  # file order (deterministic)

    used_pos_len = []
    for iid, acc, seq in pos_sorted:
        pos_samples[f"ICE_{iid}"] = (1, seq)
        used_pos_len.append((iid, len(seq), gc(seq)))

    # pair negatives with only the first n_neg ICEs (default: all 207)
    n_neg = min(n_neg, len(pos_sorted))
    for iid, L, gc_p in used_pos_len[:n_neg]:
        # genomes sorted by non-GI GC distance (deterministic); equal-distance ones rotated randomly
        cand = sorted(genomes, key=lambda g: (abs(g[3] - gc_p), g[0]))
        best = None  # (gc_delta, gid, start)
        found = False
        for gid, seq, iv, _g in cand:
            big = [(s, e) for s, e in iv if e - s >= L]
            if not big:
                continue
            occ = occupied[gid]
            for _ in range(8):
                s0, e0 = big[rng.randrange(len(big))]
                start = rng.randrange(s0, e0 - L + 1)
                end = start + L
                if any(start < oe and end > os_ for os_, oe in occ):
                    continue
                region = seq[start:end]
                if region.count("A") + region.count("C") + region.count("G") + region.count("T") != L:
                    continue
                d = abs(gc(region) - gc_p)
                if best is None or d < best[0]:
                    best = (d, gid, start)
                if d <= 3.0:
                    occupied[gid].append((start, end))
                    neg_samples[f"NEG_{iid}_{gid}_{start}"] = (0, region)
                    neg_meta.append({"pos_id": iid, "genome": gid, "start": start,
                                     "end": end, "len": L, "gc": gc(region),
                                     "pos_gc": gc_p, "gc_delta": d, "fallback": False})
                    found = True
                    break
            if found:
                break
        if not found and best is not None:
            d, gid, start = best
            # fallback: use the tried candidate with the smallest GC distance (even if > 3%)
            seq_g = next(s for g, s, _, _ in genomes if g == gid)
            region = seq_g[start:start + L]
            if region.count("A") + region.count("C") + region.count("G") + region.count("T") != L:
                region = region.replace("N", "A")
            occupied[gid].append((start, start + L))
            neg_samples[f"NEG_{iid}_{gid}_{start}"] = (0, region)
            neg_meta.append({"pos_id": iid, "genome": gid, "start": start,
                             "end": start + L, "len": L, "gc": gc(region),
                             "pos_gc": gc_p, "gc_delta": d, "fallback": True})

    samples = dict(pos_samples)
    samples.update(neg_samples)
    pl = [len(s) for _, s in pos_samples.values()]
    nl = [len(s) for _, s in neg_samples.values()]
    pg = [gc(s) for _, s in pos_samples.values()]
    ng = [gc(s) for _, s in neg_samples.values()]
    print(f"[ICE] pos={len(pos_samples)} neg={len(neg_samples)} "
          f"(fallback={sum(1 for m in neg_meta if m['fallback'])})")
    print(f"[ICE] len bp pos med={int(np.median(pl))} neg med={int(np.median(nl))}; "
          f"GC pos mean={np.mean(pg):.1f}% neg mean={np.mean(ng):.1f}% "
          f"|Δ|={abs(np.mean(pg)-np.mean(ng)):.1f}%")
    return samples, neg_meta


# ═══════════════════════════════════════════════════════════════════════════ #
#  model construction + window prediction (v6.8 protocol, CPU)
# ═══════════════════════════════════════════════════════════════════════════ #

def load_model_cpu(ckpt: Path, config: Path):
    import torch
    import yaml
    from model.api import build_model
    from model.contracts import HGTConfig

    torch.set_num_threads(4)
    assert not torch.cuda.is_available(), "CPU-only: CUDA_VISIBLE_DEVICES must be empty"

    mc = yaml.safe_load(open(config))["model"]
    cfg = HGTConfig(
        data_mode="contig", max_contig_length=3000,
        hidden_dim=mc["hidden_dim"],
        num_attention_heads=mc["num_attention_heads"],
        dropout=mc["dropout"],
        graph_top_k=mc["graph_top_k"],
        bio_feature_dim=mc["bio_feature_dim"],
        pretrained_nt_path=mc.get("pretrained_nt_path", str(NT_FALLBACK)),
        pretrained_caduceus_path="",
        use_local_encoder=False,
        use_evo=bool(mc.get("use_evo", False)),
        pretrained_evo_dim=int(mc.get("pretrained_evo_dim", 4096)),
        early_exit_threshold=0.95,
    )
    model = build_model(cfg)

    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    ckpt_epoch = st.get("epoch")
    contract = st.get("contract", {})
    if ckpt_epoch != 2:  # best_model may have been overwritten by a later better epoch → roll back to the ep2 archive
        alt = ckpt.parent / "epoch_2.pt"
        print(f"[WARN] {ckpt.name} epoch={ckpt_epoch} ≠ 2 → fallback {alt.name}")
        st = torch.load(alt, map_location="cpu", weights_only=False)
        ckpt_epoch = st.get("epoch")
        contract = st.get("contract", {})
    missing, unexpected = model.load_state_dict(st["model"], strict=False)
    model.eval()

    contract_info = {
        "ckpt": str(ckpt),
        "ckpt_epoch": ckpt_epoch,
        "encoder_frozen": contract.get("encoder_frozen"),
        "backbone_sha256": contract.get("backbone_sha256"),
        "config_sha256": contract.get("config_sha256"),
        "dataset_manifest_sha256": contract.get("dataset_manifest_sha256"),
        "state_dict_missing": len(missing),
        "state_dict_unexpected": len(unexpected),
    }
    assert contract.get("encoder_frozen") is True, "contract: encoder_frozen must be True"
    assert contract.get("backbone_sha256"), "contract: backbone_sha256 missing"
    print(f"[Model] epoch={ckpt_epoch} load: missing={len(missing)} "
          f"unexpected={len(unexpected)} (rotary lazy-cache keys)")
    print(f"[Contract] encoder_frozen={contract_info['encoder_frozen']} "
          f"backbone_sha256={contract_info['backbone_sha256'][:16]}...")
    return model, contract_info, mc


def predict_windows(model, samples: dict, window: int, overlap: int, win_cap: int):
    """v6.8 eval_external_benbow.py window-prediction protocol (CPU version).

    window=3000 / overlap=300 / is_training=False / batch=1 / sigmoid(cls_logits)
    Sole deviation: max_eval_windows changed from the config's 10 → win_cap (CPU budget; see report).
    """
    import torch
    from torch.utils.data import DataLoader
    from data_pipeline.impl.dataset import SlidingWindowContigDataset
    from data_pipeline.impl.collator import HGTCollator
    from data_pipeline.contracts import HGTSample

    hgt_samples, labels_map = [], {}
    for sid, (label, seq) in samples.items():
        hgt_samples.append(HGTSample(
            seq_id=sid, label=label, sequence=seq, r2_sequence=None,
            events=[], sample_type="POS" if label == 1 else "NEG",
            seq_len=len(seq)))
        labels_map[sid] = label

    ds = SlidingWindowContigDataset(
        samples=hgt_samples, window_size=window, overlap=overlap,
        is_training=False, max_neg_windows=9999, max_eval_windows=win_cap,
        extract_bio=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=HGTCollator(pad_id=0), num_workers=2)
    print(f"[Data] {len(hgt_samples)} contigs, {len(ds)} windows (win_cap={win_cap})")

    contig_windows = defaultdict(list)
    t0 = time.time()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            output = model(batch, early_exit=False)
            prob = torch.sigmoid(output.cls_logits.squeeze(-1)).item()
            sids = getattr(batch, "original_seq_ids", None) or batch.seq_ids
            contig_windows[sids[0]].append(prob)
            if (i + 1) % 100 == 0:
                el = time.time() - t0
                print(f"  [{i+1}/{len(ds)}] {el:.0f}s ({el/(i+1):.2f}s/win, "
                      f"ETA {(len(ds)-i-1)*el/(i+1)/3600:.2f}h)", flush=True)
    infer_s = time.time() - t0
    print(f"[Predict] {len(ds)} windows in {infer_s:.0f}s "
          f"({infer_s/len(ds):.2f}s/win)")
    return contig_windows, labels_map, len(ds), infer_s


# ═══════════════════════════════════════════════════════════════════════════ #
#  aggregation + metrics (v6.8 aggregate_and_eval verbatim replication + optimal-threshold F1)
# ═══════════════════════════════════════════════════════════════════════════ #

def aggregate_and_eval(contig_windows, labels_map, threshold=0.5):
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score,
        precision_score, recall_score, confusion_matrix,
    )

    sid_list = sorted(contig_windows.keys())
    y_true = np.array([labels_map.get(s, 0) for s in sid_list])

    strategies = {}
    for name in ["max", "mean", "median", "top3_mean"]:
        if name == "max":
            vals = [max(contig_windows[s]) for s in sid_list]
        elif name == "mean":
            vals = [np.mean(contig_windows[s]) for s in sid_list]
        elif name == "median":
            vals = [np.median(contig_windows[s]) for s in sid_list]
        elif name == "top3_mean":
            vals = [np.mean(sorted(contig_windows[s], reverse=True)[:3]) for s in sid_list]
        strategies[name] = np.array(vals)

    results = {}
    for name, scores in strategies.items():
        preds = (scores >= threshold).astype(int)
        n_pos = int(y_true.sum())
        n_neg = len(y_true) - n_pos

        entry = {
            "n_samples": len(y_true), "n_positive": n_pos, "n_negative": n_neg,
            "threshold": threshold,
        }

        if n_pos > 0 and n_neg > 0:
            entry["auc"] = float(roc_auc_score(y_true, scores))
            entry["auprc"] = float(average_precision_score(y_true, scores))
            entry["f1"] = float(f1_score(y_true, preds))
            entry["precision"] = float(precision_score(y_true, preds, zero_division=0))
            entry["recall"] = float(recall_score(y_true, preds, zero_division=0))
            tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
            entry["tp"], entry["fp"], entry["fn"], entry["tn"] = int(tp), int(fp), int(fn), int(tn)
            entry["specificity"] = float(tn / max(1, tn + fp))
            bins = min(10, n_pos, n_neg)
            if bins > 1:
                bin_edges = np.linspace(0.0, 1.0, bins + 1)
                bin_idx = np.clip(np.digitize(scores, bin_edges[1:-1]), 0, bins - 1)
                ece = 0.0
                for b in range(bins):
                    mask = bin_idx == b
                    if mask.sum() > 0:
                        acc = float(y_true[mask].mean())
                        conf = float(scores[mask].mean())
                        ece += mask.sum() / len(scores) * abs(acc - conf)
                entry["ece"] = float(ece)
            pos_scores = scores[y_true == 1]
            neg_scores = scores[y_true == 0]
            entry["pos_mean_prob"] = float(pos_scores.mean())
            entry["neg_mean_prob"] = float(neg_scores.mean())
            entry["neg_spec_at_0.5"] = float((neg_scores < 0.5).mean())

            # ── optimal-threshold F1 (same metric definition as epoch_metrics.json optimal_threshold) ──
            best_t, best_f1 = threshold, entry["f1"]
            for t in np.arange(0.01, 1.00, 0.01):
                f1_t = f1_score(y_true, (scores >= t).astype(int), zero_division=0)
                if f1_t > best_f1:
                    best_t, best_f1 = float(t), float(f1_t)
            entry["optimal_threshold"] = best_t
            entry["f1_at_optimal"] = float(best_f1)

        results[name] = entry
    return results


def window_stats(contig_windows, labels_map):
    sid_list = sorted(contig_windows.keys())
    y = np.array([labels_map.get(s, 0) for s in sid_list])
    n_win = np.array([len(contig_windows[s]) for s in sid_list])
    return {
        "n_contigs": len(sid_list),
        "n_windows": int(n_win.sum()),
        "win_per_contig_pos_mean": float(n_win[y == 1].mean()) if (y == 1).any() else None,
        "win_per_contig_neg_mean": float(n_win[y == 0].mean()) if (y == 0).any() else None,
        "win_per_contig_max": int(n_win.max()),
    }


# ═══════════════════════════════════════════════════════════════════════════ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", required=True, choices=["benbow", "iv4", "ice"])
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--win_cap", type=int, default=10,
                    help="max eval windows per contig (v6.8 config=10; lower it for CPU budget)")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke run: keep only the first N contigs per class (0=all)")
    ap.add_argument("--tag_suffix", default="")
    ap.add_argument("--out_dir", default=str(ROOT / "docs/evidence"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"real_anchor_{args.anchor}_m0ep2{args.tag_suffix}"

    t_start = time.time()
    # ── 1. data ──
    neg_meta = None
    if args.anchor == "benbow":
        samples = read_benbow(BENBOW_FASTA)
        n_pos = sum(1 for _, (l, _) in samples.items() if l == 1)
        print(f"[Benbow] {len(samples)} samples: {n_pos} pos + {len(samples)-n_pos} neg")
    elif args.anchor == "iv4":
        samples = build_iv4_300_300(IV4_DIR)
    else:
        ice = read_iceberg(ICE_FASTA)
        print(f"[ICE] experimental ICE records: {len(ice)}")
        samples, neg_meta = build_ice_anchor(ice, IV4_DIR, n_neg=len(ice))

    if args.limit > 0:  # smoke: first N items per class
        pos = {k: v for k, v in samples.items() if v[0] == 1}
        neg = {k: v for k, v in samples.items() if v[0] == 0}
        samples = dict(list(pos.items())[:args.limit])
        samples.update(dict(list(neg.items())[:args.limit]))
        print(f"[Smoke] limited to {len(samples)} contigs")

    # ── 2. model (contractassert) ──
    model, contract_info, mc = load_model_cpu(Path(args.ckpt), Path(args.config))
    import yaml as _yaml
    _dc = _yaml.safe_load(open(args.config))["data"]

    # ── 3. window prediction ──
    contig_windows, labels_map, n_windows, infer_s = predict_windows(
        model, samples, window=_dc["sw_window_size"], overlap=_dc["sw_overlap"],
        win_cap=args.win_cap)

    # ── 4. aggregation + metric ──
    results = aggregate_and_eval(contig_windows, labels_map)
    wstats = window_stats(contig_windows, labels_map)

    # ── 5. save artifacts ──
    sid_list = sorted(contig_windows.keys())

    def _agg(kind, w):
        if kind == "max":
            return max(w)
        if kind == "mean":
            return float(np.mean(w))
        if kind == "median":
            return float(np.median(w))
        return float(np.mean(sorted(w, reverse=True)[:3]))

    np.savez(
        out_dir / f"{tag}.npz",
        contig_ids=np.array(sid_list, dtype=object),
        labels=np.array([labels_map[s] for s in sid_list], dtype=np.int64),
        n_windows=np.array([len(contig_windows[s]) for s in sid_list], dtype=np.int64),
        window_probs=np.array([np.asarray(contig_windows[s], dtype=np.float32)
                               for s in sid_list], dtype=object),
        agg_max=np.array([_agg("max", contig_windows[s]) for s in sid_list], dtype=np.float32),
        agg_mean=np.array([_agg("mean", contig_windows[s]) for s in sid_list], dtype=np.float32),
        agg_median=np.array([_agg("median", contig_windows[s]) for s in sid_list], dtype=np.float32),
        agg_top3_mean=np.array([_agg("top3_mean", contig_windows[s]) for s in sid_list], dtype=np.float32),
    )

    payload = {
        "anchor": args.anchor,
        "protocol": {
            "source": "replication of v6.8 scripts/eval_external_benbow.py",
            "window_size": 3000, "overlap": 300, "batch_size": 1,
            "max_eval_windows_config_v68": 10,
            "max_eval_windows_used": args.win_cap,
            "aggregations": ["max", "mean", "median", "top3_mean"],
            "device": "cpu", "torch_threads": 4,
        },
        "contract": contract_info,
        "window_stats": wstats,
        "neg_sample_meta": neg_meta,
        "infer_seconds": infer_s,
        "total_seconds": time.time() - t_start,
        "results": results,
    }
    with open(out_dir / f"{tag}.json", "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*62}")
    print(f"[{args.anchor.upper()}] n={wstats['n_contigs']} "
          f"(pos={results['max']['n_positive']}/neg={results['max']['n_negative']}) "
          f"windows={wstats['n_windows']}")
    for name in ["max", "mean", "median", "top3_mean"]:
        m = results[name]
        print(f"  [{name:9s}] AUC={m.get('auc', float('nan')):.4f} "
              f"AUPRC={m.get('auprc', float('nan')):.4f} "
              f"F1@0.5={m.get('f1', float('nan')):.4f} "
              f"F1@opt({m.get('optimal_threshold', 0):.2f})={m.get('f1_at_optimal', float('nan')):.4f}")
    print(f"  infer={infer_s/3600:.2f}h total={(time.time()-t_start)/3600:.2f}h")
    print(f"Saved: {out_dir}/{tag}.json + .npz")


if __name__ == "__main__":
    main()
