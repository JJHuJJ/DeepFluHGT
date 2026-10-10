#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B1 cross-task RAP  —  NTv3-pre CPU embedding (DeepFluHGT; the Evo2 half waits for a GPU window)
================================================================================
For the 5 public Feng 2025 (Nat Commun 16:10780) datasets under benchmark/crosstask/,
run CPU forward passes with the frozen NTv3-pre (model/InstaDeepAI/NTv3_650M_pre) and
output mean-pooled sequence embeddings in two calibers → {dataset}/{dataset}_ntv3_pooled.npz.

The two calibers (keep them distinct; already noted in the report):
  pooled_1536     NTv3-pre hidden_states[-1] (deconv tower output, full resolution
                  restored, 1536d), masked mean-pool over real tokens
                  ([CLS]+seq+[SEP], no pads).
                   —  —  pure frozen-NTv3 representation; the primary probe caliber.
  pooled_768_proj the same hidden state passed through GlobalEncoder._proj
                  (LayerNorm(1536) → Linear(1536,768), **randomly initialized, seed42, untrained**)
                  then mean-pooled  —  —  a "full-model-path caliber" archive: remap→pad→NTv3→_proj→pool,
                  step-by-step consistent with model/impl/backbone.py GlobalEncoder._forward_nt,
                  but the _proj weights come from the pre-training random init (no checkpoint);
                  it serves only as a pipeline-caliber control, not as evidence of
                  representation quality.

Processing chain (fully mirrors GlobalEncoder._forward_nt, 0):
  canonical tokenize ([CLS] seq [SEP]; A=5,T=6,C=7,G=8,N=9; see
  data_pipeline/impl/tokenizer.py) → NTV3_REMAP {5:6,6:7,7:8,8:9,9:10,1:3,2:4}
  → pad length to a multiple of 128 (NTv3 conv tower; pad token id=1, attention_mask filled with 0;
    40bp→[CLS]+40+[SEP]=42→128, 41bp→43→128, 200bp→202→256, normal processing)
  → NTv3-pre forward(output_hidden_states=True) → take hidden_states[-1][:, :L, :]

Run discipline: CPU-only (CUDA_VISIBLE_DEVICES=""), torch/BLAS threads ≤4 (hard constraint),
touch no files of any in-flight task (only write the new {dataset}_ntv3_pooled.npz + manifest).

usage:
  python scripts/embed_ntv3_cpu.py --datasets all --split test
  python scripts/embed_ntv3_cpu.py --datasets ecoli_4mc --limit 64   # smoke
"""
from __future__ import annotations

import os
# Hard constraint: CPU-only + threads ≤4 (must be set before importing torch/numpy)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # reuse rap_probes
DATASETS = ["promoter_b_amyloliquefaciens", "ecoli_4mc", "promoter_r_capsulatus",
            "human_vs_worm", "celegans_4mc"]               # small→large, incremental output
DEFAULT_NT = str(REPO / "model" / "InstaDeepAI" / "NTv3_650M_pre")


def main() -> None:
    ap = argparse.ArgumentParser(description="B1 NTv3-only CPU embedding (cross-task RAP)")
    ap.add_argument("--datasets", default="all",
                    help="all or a comma-separated list of directory names (under benchmark/crosstask/)")
    ap.add_argument("--split", default="test", choices=["test", "train"],
                    help="test → rap_input.fasta (probe pool, default); train → train.fasta")
    ap.add_argument("--nt-path", default=DEFAULT_NT)
    ap.add_argument("--batch-size", type=int, default=0,
                    help="0=auto-select by pad length (≤160→64, ≤320→32, otherwise→8)")
    ap.add_argument("--limit", type=int, default=0, help=">0 forward only the first n items (smoke/speed test)")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    from data_pipeline.impl.tokenizer import SingleNucleotideTokenizer
    from model.impl.backbone import GlobalEncoder
    import rap_probes as rap                       # reuse fasta loading/statistics (already validated)

    names = DATASETS if args.datasets == "all" else args.datasets.split(",")
    root = REPO / "benchmark" / "crosstask"

    t_start = time.time()
    torch.manual_seed(42)                             # makes the _proj random init reproducible
    enc = GlobalEncoder(pretrained_path=args.nt_path, use_flash_attention=False)
    enc.eval()
    if enc._use_fallback:
        raise SystemExit("FATAL: NTv3-pre failed to load and fell back to random init, aborting (check nt-path)")
    tok = SingleNucleotideTokenizer()

    for name in names:
        ddir = root / name
        fasta = ddir / ("rap_input.fasta" if args.split == "test" else "train.fasta")
        seqs = rap.load_fasta(fasta)
        ids = list(seqs.keys())
        if args.limit:
            ids = ids[: args.limit]
        n = len(ids)
        Ls = np.array([len(seqs[i]) for i in ids])
        print(f"\n[embed] {name} split={args.split} n={n} bp[{Ls.min()},{Ls.max()}]",
              flush=True)

        # tokenize (canonical: [CLS]+seq+[SEP])  —  —  same type as data_pipeline
        token_lists = [tok.tokenize(seqs[i]) for i in ids]
        L = max(len(t) for t in token_lists)
        pad_to = ((L + 127) // 128) * 128
        bs = args.batch_size or (64 if pad_to <= 160 else 32 if pad_to <= 320 else 8)

        pool1536 = np.empty((n, 1536), dtype=np.float16)
        pool768 = np.empty((n, 768), dtype=np.float16)
        t0 = time.time()
        n_done = 0
        with torch.no_grad():
            for b0 in range(0, n, bs):
                chunk = token_lists[b0: b0 + bs]
                lb = max(len(t) for t in chunk)
                ids_t = torch.zeros((len(chunk), lb), dtype=torch.long)   # PAD=0
                for r, t in enumerate(chunk):
                    ids_t[r, : len(t)] = torch.tensor(t, dtype=torch.long)
                #  —  —  mirrors GlobalEncoder._forward_nt (backbone.py 0)  —  —
                mapped = enc._remap_ids(ids_t)                            # canonical→NTv3
                attn = (ids_t != 0).long()                                # real tokens=1
                B, Lb = mapped.shape
                if Lb % 128 != 0:                                         # conv tower pad
                    pad_len = 128 - (Lb % 128)
                    mapped = torch.cat(
                        [mapped, torch.full((B, pad_len), 1, dtype=mapped.dtype)], 1)
                    attn = torch.cat(
                        [attn, torch.zeros((B, pad_len), dtype=attn.dtype)], 1)
                out = enc._nt_model(input_ids=mapped, output_hidden_states=True,
                                    attention_mask=attn)
                h_last = out.hidden_states[-1][:, :Lb, :].float()         # [B,L,1536]
                mask = (attn[:, :Lb] != 0).unsqueeze(-1).float()
                p1536 = (h_last * mask).sum(1) / mask.sum(1).clamp(min=1)
                with torch.amp.autocast(device_type="cpu", enabled=False):
                    h768 = enc._proj(h_last)
                p768 = (h768 * mask).sum(1) / mask.sum(1).clamp(min=1)

                pool1536[b0: b0 + len(chunk)] = p1536.numpy().astype(np.float16)
                pool768[b0: b0 + len(chunk)] = p768.numpy().astype(np.float16)
                n_done += len(chunk)
                if (b0 // bs) % 20 == 0 or b0 + bs >= n:
                    el = time.time() - t0
                    eta = el / n_done * (n - n_done)
                    print(f"  {n_done}/{n} ({n_done/el:.1f} seq/s, ETA {eta/60:.1f} min)",
                          flush=True)

        el = time.time() - t0
        st = np.array([rap.seq_stats(seqs[i]) for i in ids])              # len/gc/n%
        y = np.array([1 if i.startswith("pos_") else 0 for i in ids], dtype=np.int8)
        assert np.isfinite(pool1536.astype(np.float32)).all(), "1536 embeddingincluding NaN/Inf"
        assert np.isfinite(pool768.astype(np.float32)).all(), "768 embeddingincluding NaN/Inf"

        suffix = "" if args.split == "test" else "_train"
        out_npz = ddir / f"{name}_ntv3_pooled{suffix}.npz"
        manifest = dict(
            dataset=name, split=args.split, n=n, seq_bp=[int(Ls.min()), int(Ls.max())],
            token_len_with_cls_sep=int(L), pad_to=int(pad_to), batch_size=bs,
            model_path=args.nt_path, caliber_1536="NTv3-pre hidden_states[-1] "
            "(deconv full resolution) masked mean-pool over [CLS]+seq+[SEP]",
            caliber_768="the same hidden state through GlobalEncoder._proj (LayerNorm1536+Linear1536x768, "
            "randomly initialized with torch.manual_seed(42), untrained) then mean-pooled  —  —  full-model-path caliber archive",
            proj_weights="random-init (seed42), NOT from any checkpoint",
            dtype="fp16 (cast to fp32 at probe time, same practice as evo_cache)",
            threads=args.threads, elapsed_s=round(el, 1),
            seq_per_s=round(n / el, 2), script="scripts/embed_ntv3_cpu.py",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
        np.savez_compressed(
            out_npz, ids=np.array(ids), labels=y,
            pooled_1536=pool1536, pooled_768_proj=pool768,
            length=st[:, 0].astype(np.float32), gc=st[:, 1].astype(np.float32),
            n_pct=st[:, 2].astype(np.float32),
            manifest=np.array(json.dumps(manifest)))
        print(f"[embed] {name} done -> {out_npz.name} ({el/60:.1f} min, "
              f"{n/el:.1f} seq/s, pos={int(y.sum())})", flush=True)

    print(f"\n[embed] ALL done, total runtime {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
