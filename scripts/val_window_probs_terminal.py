#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""val_window_probs_terminal.py  —  val window-level probabilities for the M2(ep9)/M3(ep7) terminal checkpoints.

Purpose: the M2/M3 decision-layer residual re-audit (SF7C/ST1-C2 commitments) requires val
window probabilities in exactly the same metric caliber as M0. The M0 version (residual_corr.py)
consumes m0_trajectory/trainval_gap_epoch_2.npz (an engine val-monitor artifact, including
event-aware window selection); the M2/M3 engine does not yet persist window-level artifacts,
so this script fills the gap:
  window set = BEDParser(val, mode=contig) + SlidingWindowContigDataset(
      is_training=False, 3000/300, max_neg_windows=2, max_eval_windows=10,
      rc_aug_prob=0.0)   ← verbatim-identical parameters to residual_corr.rebuild_val_windows
      (extract_bio=True + evo_cache=val.pt only because the model input requires them)
  output npz keys consistent with m0 trainval_gap: val_probs/val_labels/val_types/val_ids
  (window order = ds._window_index order; positional alignment with the M0 npz can be asserted).

Model loading = verbatim replication of interim_eval.load_model_interim (hard-asserted epoch contract:
M2 best_model.pt=ep9, M3=ep7).

usage (launcher: scripts/run/val_window_probs_{m2,m3}.sh):
  python scripts/val_window_probs_terminal.py --model m2 --gpu_uuid <A100-UUID>
"""
from __future__ import annotations

import os
import sys
import time
import json
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

NT_FALLBACK = ROOT / "model/InstaDeepAI/NTv3_650M_pre"
VAL_DIR = ROOT / "simulated_data/Contig_500k/val"
GPU_A100_UUID = "${GPU_A100_UUID}"

MODELS = {
    "m2": dict(ckpt=ROOT / "checkpoints/Contig_500k/best_model.pt",
               config=ROOT / "configs/contig_500k.yaml",
               evo_val=ROOT / "evo_cache/Contig_500k/val.pt",
               expect_epoch=9,
               out=ROOT / "docs/evidence/terminal/M2/val_window_probs_final.npz"),
    "m3": dict(ckpt=ROOT / "checkpoints/Contig_500k_m3/best_model.pt",
               config=ROOT / "configs/contig_500k_m3.yaml",
               evo_val=ROOT / "evo_cache/Contig_500k_shuffle/val.pt",
               expect_epoch=7,
               out=ROOT / "docs/evidence/terminal/M3/val_window_probs_final.npz"),
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--gpu_uuid", default=GPU_A100_UUID)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--min_free_gb", type=float, default=20.0)
    return ap.parse_args()


def load_model(ckpt: Path, config: Path, expect_epoch: int):
    """interim_eval.load_model_interim verbatim replication."""
    import torch
    import yaml
    from model.api import build_model
    from model.contracts import HGTConfig

    device = torch.device("cuda:0")
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
        evo_hidden=int(mc.get("evo_hidden", 512)),
        early_exit_threshold=0.95,
    )
    model = build_model(cfg)
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    ckpt_epoch = st.get("epoch")
    assert ckpt_epoch == expect_epoch, \
        f"contract: {ckpt.name} epoch={ckpt_epoch} != expected {expect_epoch}"
    contract = st.get("contract", {})
    missing, unexpected = model.load_state_dict(st["model"], strict=False)
    model.to(device).eval()
    assert contract.get("encoder_frozen") is True
    assert contract.get("backbone_sha256")
    print(f"[Model] {ckpt.name} ep={ckpt_epoch} use_evo={bool(mc.get('use_evo'))} "
          f"missing={len(missing)} unexpected={len(unexpected)} → {device}",
          flush=True)
    return model, device


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    import torch
    from torch.utils.data import DataLoader
    from data_pipeline.impl.parser import BEDParser
    from data_pipeline.impl.dataset import SlidingWindowContigDataset
    from data_pipeline.impl.collator import HGTCollator

    m = MODELS[args.model]
    n_gpu = torch.cuda.device_count()
    assert n_gpu == 1, f"expected exactly one visible GPU, actual {n_gpu}"
    name = torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info(0)
    assert free / 1e9 >= args.min_free_gb, f"insufficient free GPU memory ({free/1e9:.1f}GB)"
    print(f"[GPU] {name} | free {free/1e9:.1f}/{total/1e9:.1f} GB", flush=True)

    t0 = time.time()
    # window set: verbatim-identical parameters to residual_corr.rebuild_val_windows (event-aware)
    val_samples = BEDParser(
        bed_path=str(VAL_DIR / "breakpoints.bed"),
        fasta_dir=str(VAL_DIR), mode="contig").parse()
    ds = SlidingWindowContigDataset(
        samples=val_samples, window_size=3000, overlap=300,
        is_training=False, max_neg_windows=2, max_eval_windows=10,
        extract_bio=True, evo_cache_path=str(m["evo_val"]), rc_aug_prob=0.0)
    n = len(ds)
    print(f"[win] {n} val windows (BEDParser event-aware, isomorphic to the M0 npz)", flush=True)

    model, device = load_model(m["ckpt"], m["config"], m["expect_epoch"])
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        collate_fn=HGTCollator(pad_id=0), num_workers=4)

    probs = np.empty(n, dtype=np.float64)
    t1 = time.time()
    done = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            out = model(batch, early_exit=False)
            p = torch.sigmoid(out.cls_logits.squeeze(-1)).float().cpu().numpy()
            probs[done:done + len(p)] = p
            done += len(p)
            if (i + 1) % 200 == 0:
                el = time.time() - t1
                print(f"  [{done}/{n}] {el/60:.1f}min "
                      f"({el/(i+1):.2f}s/batch, ETA {(n-done)*el/done/60:.0f}min)",
                      flush=True)
    assert done == n
    infer_s = time.time() - t1

    # window metadata (keys/order consistent with the m0 trainval_gap npz)
    ids = np.array([ds.samples[w[0]].seq_id for w in ds._window_index],
                   dtype=object)
    labels = np.array([w[3] for w in ds._window_index], dtype=int)
    types = np.array([w[4] for w in ds._window_index], dtype=object)
    np.savez_compressed(
        m["out"], val_probs=probs, val_labels=labels, val_types=types,
        val_ids=ids)
    meta = {"model": args.model, "ckpt": str(m["ckpt"]),
            "expect_epoch": m["expect_epoch"], "n_windows": int(n),
            "n_pos": int(labels.sum()), "gpu": name, "batch": args.batch,
            "infer_seconds": round(infer_s, 1),
            "total_seconds": round(time.time() - t0, 1),
            "window_protocol": ("BEDParser(val)+SlidingWindow(3000/300, eval, "
                               "max_neg=2, cap=10, rc_aug=0)  —  same "
                               "parameters as residual_corr.rebuild_val_windows"),
            "evo_cache": str(m["evo_val"])}
    json.dump(meta, open(str(m["out"]).replace(".npz", "_meta.json"), "w"),
              indent=1)
    print(f"[done] {args.model}: n={n} pos={int(labels.sum())} "
          f"{infer_s/60:.1f}min -> {m['out']}", flush=True)


if __name__ == "__main__":
    main()
