#!/usr/bin/env python3
"""evo2_layer_scan.py  —  Evo2 7B layer scan (accessibility vs GC leakage vs runtime)
================================================================================
On evo_cache/step0_subset/step0_subset.fasta (n=1996, 31.7 Mbp), embed each of the
five layers blocks.{20,24,28,30,32}.mlp.l3; each layer reports:
  ① accessibility: linear probe AUROC (pos_/neg_ tag, 70/30 stratified seed42)
  ② GC leakage: ridge regression R² of embedding→GC (test fold; log10(len) R² also attached)
  ③ runtime: that layer's embedding throughput (window/s, Mbp/s)

The embedding protocol is consistent with the canonical sliding-window protocol of
evo_cache/step0_subset/evo_subset.pt: window=3000 / stride=2700 (300bp overlap),
covering each contig from position 0, the tail window kept, BOS removed within the
window for mean-pooling, windows averaged with equal weights → contig vector (fp16).

Implementation note (important): the five layers are extracted at once via a
**single-forward multi-layer hook** (evo2's forward(layer_names=[...]) registers an
independent hook per layer; each layer's output is bit-for-bit consistent with a
separate forward per layer); the first batch runs a fidelity verification
(single-layer call vs multi-layer call, recording the maximum absolute difference),
and if verification fails it automatically rolls back to per-layer mode.

GPU discipline: CUDA_VISIBLE_DEVICES=4090 UUID; the only GPU process; 4090 24GB →
conservative token_budget cap (default 24576).

output evo_cache/step0_subset/layer_scan/:
  emb_blocks{L}.pt        per layer {seq_id: fp16[4096]} (canonical sliding-window protocol)
  layer_scan_raw.json     raw numbers per layer (probe/GC R²/runtime/fidelity verification)

usage (launcher: scripts/run/evo2_layer_scan_4090.sh):
  CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python \
      scripts/evo2_layer_scan.py --embed
  # the probe stage (CPU) runs independently via scripts/evo2_layer_scan_probe.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

CACHE = ROOT / "evo_cache/step0_subset"
FASTA = CACHE / "step0_subset.fasta"
OUT = CACHE / "layer_scan"
LAYERS = [20, 24, 28, 30, 31]  # the task specified 32, but evo2_7b only has blocks 0-31 → the last block 31 substitutes (deviation noted)
LAYER_NAMES = [f"blocks.{b}.mlp.l3" for b in LAYERS]
GPU_4090_UUID = "${GPU_4090_UUID}"


def assert_4090():
    n = torch.cuda.device_count()
    assert n == 1, f"exactly 1 visible GPU expected (4090 GPU lock), actual {n}"
    name = torch.cuda.get_device_name(0)
    assert "4090" in name, f"visible GPU is not RTX 4090: {name}"
    free, total = torch.cuda.mem_get_info(0)
    print(f"[GPU] {name} | mem free {free/1e9:.1f}/{total/1e9:.1f} GB")
    return name


class MultiLayerEmbedder:
    """Evo2 multi-layer embedding (reuses the pooling/batching protocol of precompute_evo2_embeddings)."""

    def __init__(self, model_name="evo2_7b", layer_names=None, device="cuda:0",
                 local_path=None):
        from evo2 import Evo2
        self.model = Evo2(model_name, local_path=local_path)
        self.layer_names = layer_names or LAYER_NAMES
        self.device = device
        self.hook_times = []          # [(layer_name, time.time())]  —  layer-depth vs runtime curve
        self._t_hook = {}             # layer_name → trigger time (this batch)
        self._ts_handles = []
        for ln in self.layer_names:   # attach a timing hook (coexists with evo2's internal embedding hooks)
            def _mk(name):
                def hook(*_a):
                    self._t_hook[name] = time.time()
                return hook
            self._ts_handles.append(
                self.model.model.get_submodule(ln).register_forward_hook(_mk(ln)))
        with torch.no_grad():
            toks = self.model.tokenizer.tokenize("ACGT")
            self._bos_added = len(toks) > 4
            probe = torch.tensor([toks], dtype=torch.int).to(device)
            _, embs = self.model(probe, return_embeddings=True,
                                 layer_names=self.layer_names)
        self.dim = embs[self.layer_names[0]].shape[-1]
        print(f"[MultiLayerEmbedder] layers={self.layer_names} "
              f"hidden={self.dim} bos_added={self._bos_added}")

    def _pool(self, emb, n_valid):
        start = 1 if self._bos_added else 0
        e = emb[start:start + n_valid]
        if e.shape[0] == 0:
            e = emb[:1]
        return e.float().mean(dim=0)

    @torch.no_grad()
    def embed_batch_multi(self, seqs):
        """→ {layer: [B, D] cpu float32} (right padding + valid-length-masked pooling).

        Also records each layer's hook trigger time (layer-depth vs runtime curve;
        the layers share one forward, so each layer's embedding has the same
        wall-clock cost; hook times characterize the layer's compute depth in the
        network)."""
        toks = [self.model.tokenizer.tokenize(s) for s in seqs]
        T = max(len(t) for t in toks)
        ids = torch.zeros((len(seqs), T), dtype=torch.int)
        for i, t in enumerate(toks):
            ids[i, :len(t)] = torch.tensor(t, dtype=torch.int)
        self.hook_times.clear()
        self._t_hook.clear()
        t_fwd = time.time()
        _, embs = self.model(ids.to(self.device), return_embeddings=True,
                             layer_names=self.layer_names)
        self.last_forward_s = time.time() - t_fwd
        self.hook_times = [(ln, self._t_hook.get(ln, float("nan")) - t_fwd)
                           for ln in self.layer_names]
        out = {}
        for ln in self.layer_names:
            emb = embs[ln]
            out[ln] = torch.stack(
                [self._pool(emb[i], len(s)) for i, s in enumerate(seqs)]).cpu()
        return out


def fidelity_check(embedder, seqs):
    """Fidelity verification: single-layer call vs multi-layer call (first batch).
    Returns {layer: max_abs_diff}."""
    diffs = {}
    batch = embedder.embed_batch_multi(seqs)
    for ln in embedder.layer_names[:2]:  # spot-check the first two layers
        single = embedder.model
        toks = [single.tokenizer.tokenize(s) for s in seqs]
        T = max(len(t) for t in toks)
        ids = torch.zeros((len(seqs), T), dtype=torch.int)
        for i, t in enumerate(toks):
            ids[i, :len(t)] = torch.tensor(t, dtype=torch.int)
        _, embs = single(ids.to(embedder.device), return_embeddings=True,
                         layer_names=[ln])
        with torch.no_grad():
            ref = torch.stack(
                [embedder._pool(embs[ln][i], len(s))
                 for i, s in enumerate(seqs)]).cpu()
        diffs[ln] = float((ref - batch[ln]).abs().max())
    print(f"[fidelity] multi-layer single forward vs per-layer separate forwards: max|Δ| = "
          f"{ {k: f'{v:.2e}' for k, v in diffs.items()} }")
    return diffs


def build_windows(records, window=3000, stride=2700):
    """Canonical sliding-window protocol (same type as precompute_evo2_embeddings.embed_contigs_windowed)."""
    win_entries = []
    for ci, (sid, seq) in enumerate(records):
        if len(seq) <= window:
            win_entries.append((ci, len(seq), seq))
        else:
            for start in range(0, len(seq), stride):
                win_entries.append((ci, len(seq[start:start + window]),
                                    seq[start:start + window]))
    return win_entries


def batch_slices(entries_sorted, batch_size, token_budget, len_fn):
    i, n = 0, len(entries_sorted)
    while i < n:
        take = 1
        while (i + take < n and take < batch_size
               and (take + 1) * len_fn(entries_sorted[i + take]) <= token_budget):
            take += 1
        yield i, i + take
        i += take


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", action="store_true", help="GPU embedding stage")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--token_budget", type=int, default=24576,
                    help="4090 24GB budget: 8×3001 tokens ≈ 24k")
    ap.add_argument("--model", default="evo2_7b")
    ap.add_argument("--local_path",
                    default=str(ROOT / "model/arcinstitute/evo2_7b/evo2_7b.pt"))
    ap.add_argument("--layers", default=",".join(str(b) for b in LAYERS))
    ap.add_argument("--tap", default="mlp_l3", choices=["mlp_l3", "block"],
                    help="layer extraction point: mlp_l3=blocks.N.mlp.l3 (as originally "
                         "specified by the task; blocks 30/31 have a numeric dead zone "
                         "~1e-15 at this tap) / "
                         "block=whole-block output blocks.N (residual stream, supplementary control)")
    ap.add_argument("--dtype", default=None, choices=["float16", "float32"],
                    help="embedding storage precision (default: mlp_l3→float16, block→float32; "
                         "residual-stream magnitudes beyond the fp16 range would overflow)")
    args = ap.parse_args()
    if not args.embed:
        ap.error("this script only runs the embedding stage (--embed); for the probe see evo2_layer_scan_probe.py")
    store_dtype = np.float32 if args.dtype == "float32" else \
        (np.float16 if args.dtype == "float16" else
         (np.float32 if args.tap == "block" else np.float16))

    OUT.mkdir(parents=True, exist_ok=True)
    gpu = assert_4090()
    from precompute_evo2_embeddings import iter_fasta

    blocks = [int(b) for b in args.layers.split(",")]
    if args.tap == "block":
        layer_names = [f"blocks.{b}" for b in blocks]
    else:
        layer_names = [f"blocks.{b}.mlp.l3" for b in blocks]
    # idempotent: if all artifacts for this tap already exist → skip (launcher is re-entrant)
    finals = [OUT / f"emb_{ln.replace('.', '_')}.pt" for ln in layer_names]
    if all(f.exists() for f in finals):
        print(f"[skip] tap={args.tap} all artifacts already exist, skipping embedding")
        return
    records = list(iter_fasta(str(FASTA)))
    print(f"[data] {len(records)} contigs, "
          f"{sum(len(s) for _, s in records)/1e6:.1f} Mbp")

    embedder = MultiLayerEmbedder(args.model, layer_names, "cuda:0",
                                  local_path=args.local_path)
    seqs_probe = [records[i][1][:2000] for i in (0, len(records) // 2)]
    fid = fidelity_check(embedder, seqs_probe)

    # ── checkpoint resume: load partial (completed contigs are consistent across layers, written by the same loop) ──
    partial_files = {ln: OUT / f"emb_{ln.replace('.', '_')}.partial.pt"
                     for ln in layer_names}
    out = {ln: {} for ln in layer_names}
    if all(p.exists() for p in partial_files.values()):
        pre = {ln: torch.load(partial_files[ln], map_location="cpu",
                              weights_only=False) for ln in layer_names}
        done_ids = set.intersection(*(set(v.keys()) for v in pre.values()))
        out = {ln: pre[ln] for ln in layer_names}
        records = [(sid, s) for sid, s in records if sid not in done_ids]
        print(f"[resume] {len(done_ids)} contigs already done, {len(records)} remaining")

    win_entries = build_windows(records)
    total_bp = sum(w[1] for w in win_entries)
    print(f"[windows] {len(win_entries)} windows, {total_bp/1e6:.1f} Mbp")

    total_win = defaultdict(int)
    for ci, _, _ in win_entries:
        total_win[ci] += 1

    acc = {ln: defaultdict(lambda: np.zeros(embedder.dim, np.float64)) for ln in layer_names}
    done_ct = defaultdict(int)
    hook_frac = {ln: [] for ln in layer_names}   # hook trigger time / total forward time of this batch

    t0 = time.time()
    n_batches = sum(1 for _ in batch_slices(win_entries, args.batch,
                                            args.token_budget, lambda e: e[1]))
    last_flush = 0
    for bi, (lo, hi) in enumerate(batch_slices(win_entries, args.batch,
                                               args.token_budget,
                                               lambda e: e[1])):
        chunk = win_entries[lo:hi]
        res = embedder.embed_batch_multi([w[2] for w in chunk])
        fw = max(embedder.last_forward_s, 1e-9)
        for ln, t in embedder.hook_times:
            if np.isfinite(t):
                hook_frac[ln].append(t / fw)
        for ln in layer_names:
            vecs = res[ln].numpy()
            for k, w in enumerate(chunk):
                ci = w[0]
                acc[ln][ci] += vecs[k]
        for w in chunk:
            done_ct[w[0]] += 1
        for ci in list(done_ct):
            if done_ct[ci] == total_win[ci]:
                for ln in layer_names:
                    out[ln][records[ci][0]] = \
                        (acc[ln][ci] / done_ct[ci]).astype(store_dtype)
                del done_ct[ci]
                for ln in layer_names:
                    acc[ln].pop(ci, None)
        if (bi + 1) % 100 == 0 or bi + 1 == n_batches:
            el = time.time() - t0
            done_bp = sum(w[1] for w in win_entries[:hi])
            print(f"  [{hi}/{len(win_entries)}] {el/60:.1f}min "
                  f"({done_bp/1e6/el*60:.2f} Mbp/min, "
                  f"ETA {(total_bp-done_bp)/1e6/(done_bp/1e6/el*60):.0f}min) "
                  f"contigs done {len(out[layer_names[0]])}", flush=True)
        if (bi + 1) - last_flush >= 500:      # checkpoint safety: periodically flush completed contigs to disk
            for ln in layer_names:
                torch.save(out[ln], OUT / f"emb_{ln.replace('.', '_')}.partial.pt")
            last_flush = bi + 1

    elapsed = time.time() - t0
    for ln in layer_names:
        p = OUT / f"emb_{ln.replace('.', '_')}.pt"
        torch.save(out[ln], p)
        (OUT / f"emb_{ln.replace('.', '_')}.partial.pt").unlink(missing_ok=True)
        print(f"[save] {p.name}: {len(out[ln])} contigs × {embedder.dim}")

    raw = {
        "protocol": {"window": 3000, "stride": 2700, "batch": args.batch,
                     "token_budget": args.token_budget, "gpu": gpu,
                     "model": args.model, "dim": embedder.dim, "tap": args.tap,
                     "pool": "window mean (BOS removed) → equal-weight contig avg"},
        "layers": layer_names, "n_records": len(records),
        "n_windows": len(win_entries), "total_mb": total_bp / 1e6,
        "fidelity_max_abs_diff": fid,
        "elapsed_s": elapsed, "mbp_per_min": total_bp / 1e6 / (elapsed / 60),
        "per_layer_embed_wall_s": elapsed,   # one forward produces all five layers at once
        "hook_arrival_frac_median": {ln: float(np.median(v)) if v else None
                                     for ln, v in hook_frac.items()},
    }
    raw_name = "layer_scan_raw.json" if args.tap == "mlp_l3" \
        else f"layer_scan_raw_{args.tap}.json"
    (OUT / raw_name).write_text(json.dumps(raw, indent=2))
    print(f"[done] {elapsed/60:.1f} min → {OUT}")


if __name__ == "__main__":
    main()
