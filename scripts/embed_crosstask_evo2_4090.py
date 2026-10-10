#!/usr/bin/env python3
"""embed_crosstask_evo2_4090.py — Evo2 embeddings for the B1 cross-task RAP (4090).

toward benchmark/crosstask/ 5 datasetof test split (rap_input.fasta,
40-200bp short sequence) compute Evo2 7B embedding:
  - re-use scripts/precompute_evo2_embeddings.py of Evo2Embedder
    (evo2_7b local checkpoint, layer blocks.28.mlp.l3, mean-pool after dropping BOS);
  - short sequence (≤200bp ≤ chunk_size) directly embed_batch, lengthsort + token budgetsplitbatch
    (with precompute ofbatchprocessprotocolconsistent), notsliding window;
  - output {dataset}/{dataset}_evo.npy — float16 [N, 4096], roworder == rap_input.fasta
    == the id order of {dataset}_ntv3_pooled.npz (the probe side aligns by row_idx;
    the 644 all-N records of human_vs_worm are excluded uniformly on the probe side);
  - plus {dataset}_evo_manifest.json (model/layer/batch/elapsed time).

GPU discipline: CUDA_VISIBLE_DEVICES=4090 UUID; assert the only visible device is the 4090;
serialized with other GPU tasks (one process of this script at a time).

usage (launcher: scripts/run/crosstask_evo2_4090.sh):
  CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python \
      scripts/embed_crosstask_evo2_4090.py --datasets all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

CROSS = ROOT / "benchmark/crosstask"
DATASETS = ["human_vs_worm", "promoter_b_amyloliquefaciens",
            "promoter_r_capsulatus", "ecoli_4mc", "celegans_4mc"]

GPU_4090_UUID = "${GPU_4090_UUID}"


def assert_4090():
    n = torch.cuda.device_count()
    assert n == 1, f"exactly 1 visible GPU expected (4090 GPU lock), actual {n}"
    name = torch.cuda.get_device_name(0)
    assert "4090" in name, f"visible GPU is not RTX 4090: {name}"
    free, total = torch.cuda.mem_get_info(0)
    print(f"[GPU] {name} | mem free {free/1e9:.1f}/{total/1e9:.1f} GB")
    return name


def embed_dataset(embedder, records, batch_size, token_budget):
    """short sequencedirectlybatchembedding (lengthsort + token budget; with precompute protocolconsistent)."""
    from precompute_evo2_embeddings import _batch_slices
    order = sorted(range(len(records)), key=lambda i: len(records[i][1]))
    out = np.zeros((len(records), embedder.dim), dtype=np.float16)
    t0 = time.time()
    n_batches = sum(1 for _ in _batch_slices(order, batch_size, token_budget,
                                             lambda i: len(records[i][1])))
    for bi, (lo, hi) in enumerate(_batch_slices(order, batch_size, token_budget,
                                                lambda i: len(records[i][1]))):
        idxs = order[lo:hi]
        vecs = embedder.embed_batch([records[i][1] for i in idxs])
        for k, i in enumerate(idxs):
            out[i] = vecs[k].astype(np.float16)
        if (bi + 1) % 20 == 0 or bi + 1 == n_batches:
            el = time.time() - t0
            print(f" batches {bi+1}/{n_batches} ({hi}/{len(records)} seqs, "
                  f"{el:.0f}s, ETA {(n_batches-bi-1)*el/(bi+1):.0f}s)", flush=True)
    return out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--token_budget", type=int, default=32768,
                    help="4090 24GB budget (A100 cachetaskuse 98304)")
    ap.add_argument("--model", default="evo2_7b")
    ap.add_argument("--layer", default="blocks.28.mlp.l3")
    ap.add_argument("--local_path",
                    default=str(ROOT / "model/arcinstitute/evo2_7b/evo2_7b.pt"))
    args = ap.parse_args()

    names = DATASETS if args.datasets == "all" else args.datasets.split(",")
    gpu = assert_4090()

    from precompute_evo2_embeddings import Evo2Embedder, iter_fasta
    embedder = Evo2Embedder(args.model, args.layer, "cuda:0",
                            local_path=args.local_path)

    t_all = time.time()
    for ds in names:
        fasta = CROSS / ds / "rap_input.fasta"
        npz_path = CROSS / ds / f"{ds}_ntv3_pooled.npz"
        out_path = CROSS / ds / f"{ds}_evo.npy"
        if out_path.exists():
            print(f"[evo2] {ds}: {out_path.name} alreadyat, skip")
            continue
        records = list(iter_fasta(str(fasta)))
        ids = [r[0] for r in records]
        y = np.array([1 if i.startswith("pos_") else 0 for i in ids])
        # alignment assert against the NTv3 npz (same pool, same order)
        z = np.load(npz_path, allow_pickle=False)
        assert [str(s) for s in z["ids"]] == ids, f"{ds}: npz ids with fasta orderinconsistent"
        assert (z["labels"].astype(int) == y).all()
        n_pos = int(y.sum())
        print(f"\n[evo2] {ds}: n={len(records)} pos={n_pos} "
              f"len {min(len(r[1]) for r in records)}-"
              f"{max(len(r[1]) for r in records)}bp")

        t0 = time.time()
        emb, elapsed = embed_dataset(embedder, records, args.batch,
                                     args.token_budget)
        np.save(out_path, emb)
        manifest = {
            "model": args.model, "layer": args.layer, "dim": int(embedder.dim),
            "dtype": "float16", "pool": "mean over tokens (BOS removed)",
            "row_order": "rap_input.fasta == ntv3 npz ids",
            "n": len(records), "n_pos": n_pos,
            "batch_size": args.batch, "token_budget": args.token_budget,
            "gpu": gpu, "elapsed_s": round(elapsed, 1),
            "local_path": args.local_path,
            "mean_norm": float(np.linalg.norm(emb.astype(np.float32), axis=1).mean()),
        }
        (Cross := CROSS / ds).mkdir(exist_ok=True)
        (Cross / f"{ds}_evo_manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"[evo2] {ds}: saved {emb.shape} → {out_path.name} "
              f"({elapsed:.0f}s, mean norm {manifest['mean_norm']:.2f})")

    print(f"\n[evo2] allallcompleted ({(time.time()-t_all)/60:.1f} min)")


if __name__ == "__main__":
    main()
