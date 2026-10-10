#!/usr/bin/env python3
"""Strip optimizer/scheduler/scaler/rng → a slim checkpoint retaining only model weights.

Why needed: DeepFluHGT.py predict/eval uses `torch.load(ckpt, map_location=device)` to load
the entire checkpoint onto the GPU (DeepFluHGT.py:992/1266). The PE ckpt is 8.4GB, mostly optimizer
state (Adam m+v ~5.8GB); the model is only ~2.9GB. On a shared A100 fully occupied by training,
an 8.4GB GPU memory peak can OOM the training job. This script loads on CPU and saves only
state["model"] + key fields → ~2.9GB, safe for predict.

usage:
  python scripts/slim_checkpoint.py --in checkpoints/PE_2M/best_model.pt \
      --out checkpoints/PE_2M/best_model_slim.pt
  # Then: DeepFluHGT.py predict ... --checkpoint <slim.pt>  (weights fully identical)
"""
import os, argparse
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="original checkpoint path")
    ap.add_argument("--out", required=True, help="slim checkpoint outputpath")
    args = ap.parse_args()

    sd = torch.load(args.inp, map_location="cpu", weights_only=False)
    slim = {
        "model": sd["model"],
        "metrics": sd.get("metrics", {}),
        "optimal_threshold": sd.get("optimal_threshold"),
        "epoch": sd.get("epoch"),
        "best_metric": sd.get("best_metric"),
        "_slim_note": "optimizer/scheduler/scaler/rng stripped for memory-safe predict; weights identical to original",
    }
    torch.save(slim, args.out)
    si, so = os.path.getsize(args.inp) / 1e9, os.path.getsize(args.out) / 1e9
    print(f"{args.inp}: {si:.2f} GB  ->  {args.out}: {so:.2f} GB  ({(1 - so / si) * 100:.0f}% smaller)")
    print(f"  optimal_threshold={slim['optimal_threshold']}  epoch={slim['epoch']}  best_metric={slim['best_metric']}")
    print(f"\nusage: DeepFluHGT.py predict ... --checkpoint {args.out}  (weights identical, GPU peak ~3GB)")


if __name__ == "__main__":
    main()
