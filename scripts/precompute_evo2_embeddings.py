#!/usr/bin/env python
# scripts/precompute_evo2_embeddings.py
"""
DeepFluHGT  —  Evo2 offline embedding precompute
================================================
For each FASTA record (seq_id) in a dataset, compute the Evo2 7B mean-pooled embedding
and save a {seq_id: np.ndarray[D_evo, float16]} dict (torch.save).
The data_pipeline loads it by seq_id lookup (HGTDataset / SlidingWindowContigDataset
evo_cache_path); all windows/truncations of the same sequence share one vector.

Design points (v7 contract)
----------------------
- model: evo2_7b (hidden_size=4096, single-nucleotide tokenization, runs on A100 bf16;
  20b needs Hopper). The local checkpoint is loaded via local_path, bypassing the firewalled HuggingFace.
- layer: blocks.28.mlp.l3 (~88% depth within 32 layers; paper finding: intermediate-layer representations beat the final layer).
- **Contig sliding-window protocol (canonical, deterministic)**: 3000 bp window / 2700 bp stride
  (300 bp overlap, aligned with training-side sw_window_size=3000 / sw_overlap=300),
  covering the whole contig from position 0, keeping the trailing partial window, no random/centered crop
  and no max-N-window truncation; each window is mean-pooled, then equally averaged → contig-level vector.
  sensitivity alternative: stride=3000 (no overlap)  —  switch with --evo_stride.
- Within-window pooling: mean-pool over tokens after dropping BOS.
- Batching: all windows are collected together, sorted by length and split into token-budget batches, right padding +
  valid-length masked pooling (causal model, right-side padding does not affect preceding positions).
- PE data: reads are only 150bp, no sliding window; R1/R2 are pooled separately and then averaged,
  weighted by read length (to avoid spurious junctions from concatenation).
- output: {output_dir}/{split}.pt  —  consistent with the directory convention of
  data_pipeline._resolve_evo_cache (dir → dir/{split}.pt).

Environment (separate venv, needs Python >=3.11,<3.13; use system Python 3.11):
    python3 -m venv ~/evo2_venv
    ~/evo2_venv/bin/pip install evo2

usage
----
# training/evaluation data (including train/val/test subdirectories):
CUDA_VISIBLE_DEVICES=0 python \
    scripts/precompute_evo2_embeddings.py \
    --data_dir simulated_data/Contig_500k \
    --output_dir evo_cache/Contig_500k

# Any single FASTA (e.g. the Benbow benchmark):
CUDA_VISIBLE_DEVICES=0 python \
    scripts/precompute_evo2_embeddings.py \
    --fasta benchmark/some_set.fasta \
    --output evo_cache/some_set.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch


# ── Minimal FASTA parser (no Biopython dependency) ──

def iter_fasta(path):
    """Yield (seq_id, sequence). seq_id = the header up to the first whitespace (= BED col0)."""
    header, chunks = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:].split()[0] if len(line) > 1 else ""
                chunks = []
            else:
                chunks.append(line.upper())
    if header is not None:
        yield header, "".join(chunks)


# ── Evo2 embedding ──

class Evo2Embedder:
    """Evo2 model wrapper: sequence → mean-pooled embedding [hidden_size]."""

    def __init__(self, model_name: str, layer_name: str, device: str = "cuda:0",
                 chunk_size: int = 32768, local_path: str = None):
        from evo2 import Evo2  # lazy import, to give a clear installation hint
        # local_path: local checkpoint (.pt)  —  bypass the HuggingFace download (HF is firewalled on this machine)
        self.model = Evo2(model_name, local_path=local_path)
        self.layer_name = layer_name
        self.device = device
        self.chunk_size = chunk_size
        # probe the hidden size and whether BOS is prepended
        with torch.no_grad():
            toks = self.model.tokenizer.tokenize("ACGT")
            self._bos_added = len(toks) > 4
            probe = torch.tensor([toks], dtype=torch.int).to(device)
            _, embs = self.model(probe, return_embeddings=True,
                                 layer_names=[layer_name])
        self.dim = embs[layer_name].shape[-1]
        print(f"[Evo2Embedder] model={model_name} layer={layer_name} "
              f"hidden={self.dim} bos_added={self._bos_added} "
              f"chunk={chunk_size}")

    def _pool(self, emb: torch.Tensor, n_valid: int) -> torch.Tensor:
        """Mean-pool the first n_valid tokens after dropping BOS → cpu float32 vector."""
        start = 1 if self._bos_added else 0
        e = emb[start:start + n_valid]
        if e.shape[0] == 0:
            e = emb[:1]
        return e.float().mean(dim=0).cpu()

    @torch.no_grad()
    def embed_batch(self, seqs: list) -> list:
        """Batch-embed short sequences (right padding + valid-length masked pooling).
        Requires all len(seq) <= chunk_size."""
        toks = [self.model.tokenizer.tokenize(s) for s in seqs]
        T = max(len(t) for t in toks)
        ids = torch.zeros((len(seqs), T), dtype=torch.int)
        for i, t in enumerate(toks):
            ids[i, :len(t)] = torch.tensor(t, dtype=torch.int)
        ids = ids.to(self.device)
        _, embs = self.model(ids, return_embeddings=True,
                             layer_names=[self.layer_name])
        emb = embs[self.layer_name]          # [B, T, D]
        return [self._pool(emb[i], len(s)).numpy() for i, s in enumerate(seqs)]

    @torch.no_grad()
    def _embed_chunk(self, seq: str) -> torch.Tensor:
        ids = torch.tensor([self.model.tokenizer.tokenize(seq)],
                           dtype=torch.int).to(self.device)
        _, embs = self.model(ids, return_embeddings=True,
                             layer_names=[self.layer_name])
        return self._pool(embs[self.layer_name][0], len(seq))

    @torch.no_grad()
    def embed(self, seq: str) -> np.ndarray:
        """Single sequence → [D]. Long sequences are chunked and length-weight averaged (equivalent to a global mean-pool)."""
        seq = seq.upper()
        if not seq:
            return np.zeros(self.dim, dtype=np.float32)
        if len(seq) <= self.chunk_size:
            return self._embed_chunk(seq).numpy()
        vecs, ws = [], []
        for i in range(0, len(seq), self.chunk_size):
            sub = seq[i:i + self.chunk_size]
            vecs.append(self._embed_chunk(sub))
            ws.append(len(sub))
        vecs = torch.stack(vecs)
        w = torch.tensor(ws, dtype=torch.float32)
        return ((vecs * w.unsqueeze(1)).sum(0) / w.sum()).numpy()


# ── Dataset processing ──

def embed_contigs_windowed(embedder: Evo2Embedder, records: list, batch_size: int,
                           token_budget: int, window: int, stride: int,
                           tag: str = "", partial_path: str = None,
                           flush_every_batches: int = 1000) -> dict:
    """Contig sliding-window protocol (v7 contract, deterministic):

    - window bp window / stride bp step, covering the entire contig from position 0.
      canonical: window=3000, stride=2700 (i.e. 300 bp overlap,
      aligned with training-side sw_window_size=3000 / sw_overlap=300).
    - The trailing partial window (shorter than window) is kept (avoids systematically losing the contig tail).
    - No random crop / single center crop / max-N-window truncation (no positional bias, reproducible).
    - aggregation: each window is mean-pooled, then **equally averaged** (simple mean pooling, canonical).
      Overlap regions are counted once by each of the two adjacent windows (slightly higher weight, accepted);
      coverage-weighted aggregation is left as a future option.

    records: list[(sid, seq)] → {sid: np.float16 vec}

    Checkpoint resume (v7): if partial_path exists → load the already-completed contig vectors and skip them;
    atomically flush to disk every flush_every_batches batches (write tmp then rename).
    Note: after a resume, the batch composition of the remaining contigs differs from the first run → bf16 cross-batch jitter
    (cos>=0.9999, documented in contract section 6); the manifest marks it as resumed.
    """
    # 0) Checkpoint resume: load already-completed contigs
    out = {}
    if partial_path and os.path.isfile(partial_path):
        out = torch.load(partial_path, map_location="cpu", weights_only=False)
        print(f"    [{tag}] RESUME: {len(out)} contigs already completed, skipped")
    records = [(sid, s) for sid, s in records if sid not in out]

    # 1) Generate all windows (deterministic, no sampling)
    win_entries = []   # (contig_idx, window_len, sub_seq)
    for ci, (sid, seq) in enumerate(records):
        if len(seq) <= window:
            win_entries.append((ci, len(seq), seq))
        else:
            for start in range(0, len(seq), stride):
                sub = seq[start:start + window]
                win_entries.append((ci, len(sub), sub))
    total_bp = sum(w[1] for w in win_entries)
    print(f"    [{tag}] {len(records)} contigs → {len(win_entries)} windows "
          f"(window={window}, stride={stride}, total={total_bp/1e6:.1f} Mbp)")

    # Total windows per contig (progress tracking)
    total_win = {}
    for ci, _, _ in win_entries:
        total_win[ci] = total_win.get(ci, 0) + 1

    # 2) Batch-process all windows together (length sort + token budget)
    win_entries.sort(key=lambda x: x[1])
    n_batches = sum(1 for _ in _batch_slices(win_entries, batch_size, token_budget, lambda e: e[1]))
    acc, done_ct = {}, {}
    for bi, (lo, hi) in enumerate(_batch_slices(win_entries, batch_size, token_budget, lambda e: e[1])):
        outs = embedder.embed_batch([w[2] for w in win_entries[lo:hi]])
        for k, v in enumerate(outs):
            ci = win_entries[lo + k][0]
            acc[ci] = acc.get(ci, 0) + v
            done_ct[ci] = done_ct.get(ci, 0) + 1
        # Aggregate completed contigs immediately
        for ci in list(done_ct):
            if done_ct[ci] == total_win[ci]:
                out[records[ci][0]] = (acc[ci] / done_ct[ci]).astype(np.float16)
                del acc[ci], done_ct[ci]
        if (bi + 1) % 100 == 0 or bi + 1 == n_batches:
            print(f"    [{tag}] windows {hi}/{len(win_entries)} "
                  f"(contigs done {len(out)})")
        if partial_path and (bi + 1) % flush_every_batches == 0:
            _tmp = partial_path + ".tmp"
            torch.save(out, _tmp)
            os.replace(_tmp, partial_path)
            print(f"    [{tag}] flush: {len(out)} contigs → {os.path.basename(partial_path)}")

    return out

def _batch_slices(entries_sorted: list, batch_size: int, token_budget: int,
                  len_fn):
    """Split into batches by token budget. entries_sorted must be ascending by len_fn (in-batch max = last element).

    Constraint: take × longest length in batch ≤ token_budget and take ≤ batch_size.
    Avoids mixing wildly different lengths in one batch (padding waste/OOM): a mixed 291bp~100kb batch would pad to 100kb.
    len_fn: entry → sequence length (bp).
    """
    i, n = 0, len(entries_sorted)
    while i < n:
        take = 1
        while (i + take < n and take < batch_size
               and (take + 1) * len_fn(entries_sorted[i + take]) <= token_budget):
            take += 1
        yield i, i + take
        i += take


def process_pe_dir(embedder: Evo2Embedder, split_dir: str, batch_size: int,
                   token_budget: int = 98304):
    """PE split: collect all of R1/R2, batch-process, then combine weighted by read length."""
    r1_path = os.path.join(split_dir, "R1.fasta")
    r2_path = os.path.join(split_dir, "R2.fasta")
    r2_map = dict(iter_fasta(r2_path)) if os.path.isfile(r2_path) else {}
    pairs = [(sid, r1, r2_map.get(sid, ""))
             for sid, r1 in iter_fasta(r1_path)]

    # Collect all reads (R1+R2) together, sort by length, then batch-process
    flat = []
    for pi, (sid, r1, r2) in enumerate(pairs):
        if r1:
            flat.append((pi, r1))
        if r2:
            flat.append((pi, r2))
    flat.sort(key=lambda x: len(x[1]))
    print(f"    [PE] {len(pairs)} pairs → {len(flat)} reads to embed")

    pair_acc = [None] * len(pairs)
    pair_w = [0.0] * len(pairs)
    n_batches = sum(1 for _ in _batch_slices(flat, batch_size, token_budget, lambda e: len(e[1])))
    for bi, (lo, hi) in enumerate(_batch_slices(flat, batch_size, token_budget, lambda e: len(e[1]))):
        chunk = flat[lo:hi]
        outs = embedder.embed_batch([s for _, s in chunk])
        for (pi, s), v in zip(chunk, outs):
            L = len(s)
            acc = pair_acc[pi]
            if acc is None:
                pair_acc[pi] = v * L
                pair_w[pi] = L
            else:
                pair_acc[pi] = acc + v * L
                pair_w[pi] += L
        if (bi + 1) % 200 == 0 or bi + 1 == n_batches:
            print(f"    [PE] reads {hi}/{len(flat)}")

    return {pairs[i][0]: (pair_acc[i] / pair_w[i]).astype(np.float16)
            for i in range(len(pairs)) if pair_acc[i] is not None}


def process_split_dir(embedder: Evo2Embedder, split_dir: str, batch_size: int,
                      token_budget: int = 98304, evo_window: int = 3000,
                      evo_stride: int = 2700, out_path: str = None):
    """Process one split directory (PE: R1/R2.fasta; Contig: dataset.fasta sliding-window protocol)."""
    r1_path = os.path.join(split_dir, "R1.fasta")
    if os.path.isfile(r1_path):
        return process_pe_dir(embedder, split_dir, batch_size, token_budget)
    fasta_path = os.path.join(split_dir, "dataset.fasta")
    if not os.path.isfile(fasta_path):
        print(f"[WARN] {fasta_path} does not exist, skip")
        return None
    partial = (out_path + ".partial.pt") if out_path else None
    result = embed_contigs_windowed(embedder, list(iter_fasta(fasta_path)),
                                    batch_size, token_budget,
                                    window=evo_window, stride=evo_stride,
                                    tag=os.path.basename(split_dir),
                                    partial_path=partial)
    if partial and os.path.isfile(partial):
        os.remove(partial)
        print(f"    [resume] removing {os.path.basename(partial)} (split complete)")
    return result


def main():
    ap = argparse.ArgumentParser(description="Evo2 offline embedding precompute (v7)")
    ap.add_argument("--data_dir", default=None,
                    help="dataset root directory (including train/val/test subdirectories)")
    ap.add_argument("--output_dir", default=None, help="output directory → {split}.pt")
    ap.add_argument("--fasta", default=None, help="a single FASTA file (e.g. a benchmark)")
    ap.add_argument("--output", default=None, help="output .pt path for --fasta mode")
    ap.add_argument("--model", default="evo2_7b",
                    help="Evo2 model name (default evo2_7b; 20b needs Hopper)")
    ap.add_argument("--local_path",
                    default="./model/arcinstitute/evo2_7b/evo2_7b.pt",
                    help="local checkpoint path (bypasses the HF download; empty string pulls from HF)")
    ap.add_argument("--layer", default="blocks.28.mlp.l3",
                    help="layer to take embeddings from (default blocks.28.mlp.l3, ~88% depth of 7b)")
    ap.add_argument("--chunk_size", type=int, default=32768,
                    help="batching/chunking threshold (bp, default 32768)")
    ap.add_argument("--batch_size", type=int, default=32,
                    help="batch size cap for short sequences (default 32)")
    ap.add_argument("--token_budget", type=int, default=98304,
                    help="padded token budget per batch (default 98304 ≈ 42GB GPU memory)")
    ap.add_argument("--evo_window", type=int, default=3000,
                    help="contig sliding-window window size bp (default 3000, aligned with sw_window_size)")
    ap.add_argument("--evo_stride", type=int, default=2700,
                    help="contig sliding-window stride bp (default 2700 = 300bp overlap, aligned with sw_overlap)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"],
                    help="splits to process (default train val test)")
    args = ap.parse_args()

    if not args.data_dir and not args.fasta:
        ap.error("--data_dir or --fasta is required")
    if args.fasta and not args.output:
        ap.error("--fasta mode requires --output")

    t0 = time.time()
    local_path = args.local_path or None
    if local_path and not os.path.isfile(local_path):
        print(f"[ERROR] local_path does not exist: {local_path}")
        sys.exit(1)
    embedder = Evo2Embedder(args.model, args.layer, args.device, args.chunk_size,
                            local_path=local_path)

    if args.fasta:
        print(f"[Evo2Precompute] {args.fasta} → {args.output}")
        # --fasta direct mode: a single file, same sliding-window protocol
        out_path_f = args.output
        partial = out_path_f + ".partial.pt"
        out = embed_contigs_windowed(embedder, list(iter_fasta(args.fasta)),
                                     args.batch_size, args.token_budget,
                                     window=args.evo_window, stride=args.evo_stride,
                                     tag="fasta", partial_path=partial)
        if os.path.isfile(partial):
            os.remove(partial)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        torch.save(out, args.output)
        print(f"[Evo2Precompute] saved {len(out)} vectors → {args.output} "
              f"({time.time()-t0:.0f}s)")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    for split in args.splits:
        split_dir = os.path.join(args.data_dir, split)
        if not os.path.isdir(split_dir):
            print(f"[WARN] {split_dir} does not exist, skip")
            continue
        print(f"[Evo2Precompute] {split_dir} → {args.output_dir}/{split}.pt")
        out = process_split_dir(embedder, split_dir, args.batch_size,
                                args.token_budget,
                                evo_window=args.evo_window,
                                evo_stride=args.evo_stride,
                                out_path=os.path.join(args.output_dir, f"{split}.pt"))
        if out:
            torch.save(out, os.path.join(args.output_dir, f"{split}.pt"))
            print(f"[Evo2Precompute] {split}: {len(out)} vectors, "
                  f"dim={embedder.dim} ({time.time()-t0:.0f}s)")

    print(f"[Evo2Precompute] done, total runtime {time.time()-t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
