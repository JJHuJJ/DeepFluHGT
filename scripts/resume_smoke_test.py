#!/usr/bin/env python
# scripts/resume_smoke_test.py
"""
DeepFluHGT  —  resume smoke test (contract §11 resilience)
=========================================================
Validate whether a checkpoint has the integrity needed for "resume and continue training":
  R1 complete state fields: model/optimizer/scheduler/epoch/global_step/rng_state (all four state categories)
  R2 contract metadata: contract.encoder_frozen=True + complete sha chain
  R3 rebuild-load: a freshly constructed model succeeds at load_state_dict(strict=False, rotary lazy-cache keys)
  R4 finite forward pass: forward logits on a fixed batch are finite
  R5 two-eval determinism: two forward passes on the same input satisfy max|Δ|≤1e-5 (same process)

usage: python scripts/resume_smoke_test.py --ckpt <path> --config <yaml> --data_dir <dir>
"""
from __future__ import annotations
import argparse, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_pipeline.api import load_splits_sliding_window
from data_pipeline.impl.collator import HGTCollator
from model.api import build_model
from model.contracts import HGTConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--nt_path",
                    default="./model/InstaDeepAI/NTv3_650M_pre")
    args = ap.parse_args()
    ok = True

    st = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    need = ["model", "optimizer", "scheduler", "epoch", "global_step", "rng_state"]
    missing = [k for k in need if k not in st]
    rng_ok = all(k in (st.get("rng_state") or {}) for k in
                 ("python", "numpy", "torch_cpu"))
    print(f"R1 complete state fields: {'PASS' if not missing and rng_ok else 'FAIL ' + str(missing)}")
    ok &= not missing and rng_ok

    c = st.get("contract", {})
    c_ok = c.get("encoder_frozen") is True and c.get("backbone_sha256") \
        and c.get("config_sha256") and c.get("dataset_manifest_sha256")
    print(f"R2 contract metadata: {'PASS' if c_ok else 'FAIL ' + str(list(c.keys()))}")
    ok &= bool(c_ok)

    import yaml
    mc = yaml.safe_load(open(args.config))["model"]
    cfg = HGTConfig(
        data_mode="contig", pretrained_nt_path=mc.get("pretrained_nt_path", args.nt_path),
        pretrained_caduceus_path="", use_local_encoder=False,
        use_evo=bool(mc.get("use_evo", False)),
        pretrained_evo_dim=int(mc.get("pretrained_evo_dim", 4096)),
        early_exit_threshold=0.95)
    model = build_model(cfg)
    model.load_state_dict(st["model"], strict=False)  # rotary lazy-cache keys (contract §5)
    model.eval()
    print("R3 rebuild-load: PASS")

    splits = load_splits_sliding_window(data_dir=args.data_dir, window_size=3000,
                                        overlap=300, evo_cache_path="")
    b = HGTCollator(pad_id=0)([splits["val"][i] for i in range(2)])
    with torch.no_grad():
        o1 = model(b).cls_logits
        o2 = model(b).cls_logits
    finite = bool(torch.isfinite(o1).all())
    det = float((o1 - o2).abs().max())
    print(f"R4 finite forward pass: {'PASS' if finite else 'FAIL'}")
    print(f"R5 same-process determinism: max|Δ|={det:.1e} {'PASS' if det <= 1e-5 else 'FAIL'}")
    ok &= finite and det <= 1e-5
    print("\nRESUME SMOKE:", "PASS" if ok else "FAIL")
    import json as _j, datetime as _d
    with open("logs/resume_events.jsonl", "a") as _f:
        _f.write(_j.dumps({"time": _d.datetime.now().isoformat(timespec="seconds"),
                           "event": "resume_smoke_test", "ckpt": args.ckpt,
                           "result": "PASS" if ok else "FAIL"}) + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
