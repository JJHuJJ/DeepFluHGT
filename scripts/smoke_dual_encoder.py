#!/usr/bin/env python
# scripts/smoke_dual_encoder.py
"""
DeepFluHGT  —  dual-encoder (NTv3-pre + offline Evo2) smoke test
==============================================================
Before starting the full Evo2 cache build (≈10 days of A100), validate the
architecture/data contracts on a small dataset. Does not touch the already-frozen
paper version; this is a production-readiness test.

QA Gates
--------
G1 shape        : Evo2 embedding dim == expected (4096) == HGTConfig.pretrained_evo_dim
G2 alignment    : cache keys ⊇ dataset seq_ids; a spot re-computed embedding matches the cache
G3 finite       : cache vectors / forward logits / loss all finite (no NaN/Inf)
G4 grad contract: encoder frozen (grad=None/0); fuse_alpha.grad>0; heads.grad>0;
                  evo_encoder.grad>0 after the gate is unlocked
G5 determinism  : two evals of the same batch fully consistent; re-computed embedding
                  consistent with the cache (fp16 tolerance)
G6 reload       : a few training steps → save checkpoint → rebuild model and load →
                  eval consistent with before saving

Full mini-cycle: raw seq → Evo2 cache → NTv3 → fusion → forward → loss → backward
→ optimizer steps → checkpoint → reload → inference

usage
----
python scripts/smoke_dual_encoder.py \
    --data_dir simulated_data/Contig_500_p1test \
    --evo_cache evo_cache/contig_smoke --mode contig
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.api import load_splits, load_splits_sliding_window
from data_pipeline.impl.collator import HGTCollator
from model.api import build_model
from model.contracts import HGTConfig


class Gates:
    def __init__(self):
        self.results = []

    def check(self, name: str, ok: bool, detail: str = ""):
        sym = "PASS" if ok else "FAIL"
        print(f"  [{sym}] {name}" + (f"  ({detail})" if detail else ""))
        self.results.append((name, ok))
        return ok

    def summary(self) -> bool:
        n_pass = sum(1 for _, ok in self.results if ok)
        print(f"\nGATES: {n_pass}/{len(self.results)} passed")
        return n_pass == len(self.results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--evo_cache", required=True)
    ap.add_argument("--mode", default="contig", choices=["contig", "paired_end"])
    ap.add_argument("--nt_path",
                    default="./model/InstaDeepAI/NTv3_650M_pre")
    ap.add_argument("--n_steps", type=int, default=3)
    ap.add_argument("--evo2_python", default="python",
                    help="evo2 venv python (used by the G5a recompute subprocess)")
    ap.add_argument("--evo2_gpu_uuid",
                    default="${GPU_A100_UUID}",
                    help="GPU UUID pinned for the G5a subprocess")
    args = ap.parse_args()

    torch.manual_seed(42)
    gates = Gates()
    # ── data + cache ──
    # contig mode mirrors the production training path (sliding windows, consistent
    # with contig_smoke.yaml); whole-contig mode with 4×5kb cross-attention would OOM.
    if args.mode == "contig":
        splits = load_splits_sliding_window(
            data_dir=args.data_dir, window_size=1500, overlap=300,
            max_neg_windows=2, max_eval_windows=10,
            evo_cache_path=args.evo_cache)
    else:
        splits = load_splits(data_dir=args.data_dir, mode=args.mode,
                             evo_cache_path=args.evo_cache)
    ds = splits["train"]
    collator = HGTCollator(pad_id=0)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    batches = [collator([ds[i] for i in range(k, min(k + 4, len(ds)))])
               for k in range(0, min(12, len(ds)), 4)]
    dev_batches = [b.to(dev) for b in batches]

    # ── G1 shape ──
    print("\n== G1 shape ==")
    s0 = ds[0]
    dim = len(s0.evo_embeddings)
    gates.check("G1a per-sample evo dim == 4096", dim == 4096, f"dim={dim}")
    gates.check("G1b batch evo shape [B,4096]",
                dev_batches[0].evo_embeddings is not None
                and dev_batches[0].evo_embeddings.shape[1] == 4096)

    # ── G2 alignment: cache coverage + distribution statistics (short/mid/long, pos/neg) ──
    print("\n== G2 alignment ==")
    cache = torch.load(os.path.join(args.evo_cache, "train.pt"),
                       map_location="cpu", weights_only=False)
    ds_ids = {s.seq_id for s in ds.samples}
    missing = ds_ids - set(cache.keys())
    gates.check("G2a cache keys ⊇ dataset seq_ids", not missing,
                f"missing={len(missing)}")
    lens = np.array([s.seq_len for s in ds.samples])
    labels = np.array([s.label for s in ds.samples])
    print(f"    dataset: n={len(lens)}, len min/med/p90/max="
          f"{lens.min():.0f}/{np.median(lens):.0f}/"
          f"{np.percentile(lens,90):.0f}/{lens.max():.0f}, "
          f"pos={int(labels.sum())}, neg={int((labels==0).sum())}")
    gates.check("G2b covers short(<1k)/mid(1-10k)/long(>10k) contigs",
                (lens < 1000).any() and ((lens >= 1000) & (lens <= 10000)).any()
                and (lens > 10000).any())
    gates.check("G2c both positive and negative samples present",
                labels.sum() > 0 and (labels == 0).sum() > 0)

    # ── G5 (cache part): re-computed embedding consistent with the cache ──
    # Evo2 needs the evo2 venv (the thesis env has no evo2 package); run the
    # production precompute script itself as a subprocess.
    print("\n== G5 cache determinism (spot recompute, evo2 venv subprocess) ==")
    import subprocess
    spot = list(ds.samples)[:2]
    spot_fasta = os.path.join(tempfile.gettempdir(), "smoke_spot.fasta")
    with open(spot_fasta, "w") as f:
        for s in spot:
            f.write(f">{s.seq_id}\n")
            seq = s.sequence.upper()
            for i in range(0, len(seq), 80):
                f.write(seq[i:i + 80] + "\n")
    spot_out = os.path.join(tempfile.gettempdir(), "smoke_spot.pt")
    env = dict(os.environ)
    if args.evo2_gpu_uuid:
        env["CUDA_VISIBLE_DEVICES"] = args.evo2_gpu_uuid
    r = subprocess.run(
        [args.evo2_python,
         os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "precompute_evo2_embeddings.py"),
         "--fasta", spot_fasta, "--output", spot_out],
        capture_output=True, text=True, env=env, timeout=1200)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        gates.check("G5a recompute subprocess (evo2 venv)", False, "subprocess failed")
    else:
        recompute = torch.load(spot_out, map_location="cpu", weights_only=False)
        # bf16 has numeric jitter across batch compositions (the kernel reduction path
        # varies with batch shape); absolute differences are meaningless, so use cosine
        # similarity + relative L2 as the criterion.
        worst_cos, worst_rel = 1.0, 0.0
        for s in spot:
            a = cache[s.seq_id].astype(np.float32)
            b = recompute[s.seq_id].astype(np.float32)
            worst_cos = min(worst_cos, float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
            worst_rel = max(worst_rel, float(np.linalg.norm(a - b) / np.linalg.norm(a)))
        gates.check("G5a re-computed embedding ≈ cache (cos≥0.9999, rel_L2≤1%)",
                    worst_cos >= 0.9999 and worst_rel <= 0.01,
                    f"cos={worst_cos:.6f}, rel_L2={worst_rel:.4%}")
        del recompute
        torch.cuda.empty_cache()

    # ── model (real NTv3-pre) ──
    cfg = HGTConfig(
        data_mode=args.mode, pretrained_nt_path=args.nt_path,
        pretrained_caduceus_path="", use_local_encoder=False,
        use_evo=True, pretrained_evo_dim=4096,
        early_exit_threshold=0.95,
    )
    model = build_model(cfg).to(dev).train()

    # ── G3 finite: forward + loss ──
    out0 = model(dev_batches[0])
    loss0 = out0.cls_logits.float().sum() + out0.role_logits.float().sum()
    gates.check("G3a logits/loss finite",
                bool(torch.isfinite(out0.cls_logits).all()
                     and torch.isfinite(loss0)))
    vecs = np.stack([cache[k] for k in list(cache.keys())[:64]])
    gates.check("G3b cache vectors finite", bool(np.isfinite(vecs).all()))

    # ── G4 grad contract (encoder frozen, fusion/heads trainable) ──
    print("\n== G4 grad contract ==")
    model.zero_grad(set_to_none=True)
    # production contract: encoder (NTv3) frozen
    for p in model.backbone.parameters():
        p.requires_grad = False
    out = model(dev_batches[0])
    loss = out.cls_logits.float().sum() + out.role_logits.float().sum() \
        + out.bp_heatmaps.float().sum()
    loss.backward()
    enc_none = all(p.grad is None for p in model.backbone.parameters())
    heads_g = sum((p.grad.abs().sum().item() if p.grad is not None else 0)
                  for n, p in model.named_parameters()
                  if n.startswith(("cls_head", "role_head", "direction_head",
                                   "evo_encoder", "tri_fusion", "splitter",
                                   "cross_attn", "fpn", "graph", "fuse_alpha")))
    alpha_g = model.fuse_alpha.grad
    gates.check("G4a encoder grad = None (frozen)", enc_none)
    gates.check("G4b heads/fusion grad > 0", heads_g > 0, f"sum|g|={heads_g:.1f}")
    gates.check("G4c fuse_alpha.grad present and non-zero",
                alpha_g is not None and float(alpha_g.abs()) > 0)
    # after the gate is unlocked, the evo branch should receive gradients
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.fuse_alpha.fill_(0.5)
    out = model(batches[1] if len(batches) > 1 else batches[0])
    (out.cls_logits.float().sum() + out.role_logits.float().sum()).backward()
    evo_g = sum((p.grad.abs().sum().item() if p.grad is not None else 0)
                for p in model.evo_encoder.parameters())
    gates.check("G4d evo_encoder grad > 0 after the gate is unlocked", evo_g > 0,
                f"sum|g|={evo_g:.1f}")
    with torch.no_grad():
        model.fuse_alpha.fill_(0.0)   # restore zero initialization
    for p in model.backbone.parameters():
        p.requires_grad = True

    # ── optimizer steps (under the frozen-encoder production contract) ──
    print(f"\n== optimizer steps ({args.n_steps}) ==")
    for p in model.backbone.parameters():
        p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=2e-5)
    for step in range(args.n_steps):
        opt.zero_grad(set_to_none=True)
        out = model(batches[step % len(batches)])
        l = out.cls_logits.float().sum() + out.role_logits.float().sum()
        l.backward()
        opt.step()
        print(f"    step {step}: loss={float(l):.4f}")

    # ── G6 reload ──
    print("\n== G6 checkpoint reload ==")
    model.eval()
    with torch.no_grad():
        pre = model(dev_batches[0]).cls_logits.clone()
    ckpt_path = os.path.join(tempfile.gettempdir(), "smoke_dual.pt")
    torch.save({"model": model.state_dict()}, ckpt_path)
    cfg2 = HGTConfig(
        data_mode=args.mode, pretrained_nt_path=args.nt_path,
        pretrained_caduceus_path="", use_local_encoder=False,
        use_evo=True, pretrained_evo_dim=4096,
        early_exit_threshold=0.95,
    )
    model2 = build_model(cfg2).to(dev).eval()
    state = torch.load(ckpt_path, map_location=dev, weights_only=False)
    # strict=False: NTv3-pre's rotary cos/sin_cached are non-persistent buffers lazily
    # generated after a forward (present in state_dict, absent in a fresh model);
    # ignored on load, recomputed at inference.
    model2.load_state_dict(state["model"], strict=False)
    with torch.no_grad():
        post = model2(batches[0]).cls_logits
    diff = float((pre - post).abs().max())
    # Tolerance 1e-4: rebuilding the model across processes is non-deterministic in
    # fp32 kernels (~1e-5-scale logit differences, probability differences after
    # sigmoid <1e-5, no effect on predictions); in-process determinism is guaranteed by G5b.
    gates.check("G6a predictions consistent after reload (fp32 tolerance)", diff <= 1e-4,
                f"max_diff={diff:.2e}")

    # ── G5 inference determinism ──
    print("\n== G5 inference determinism ==")
    with torch.no_grad():
        a = model2(dev_batches[1]).cls_logits
        b_ = model2(dev_batches[1]).cls_logits
    gates.check("G5b two evals of the same input consistent", float((a - b_).abs().max()) <= 1e-6)

    ok = gates.summary()
    print("\nSMOKE:", "PASS  —  safe to start the full Evo2 cache build" if ok
          else "FAIL  —  fix the gate(s) above first")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
