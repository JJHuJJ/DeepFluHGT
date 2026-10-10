#!/usr/bin/env python
# scripts/batch_sweep.py  —  V7 batch OOM/throughput sweep (frozen-contract regime)
# usage: python scripts/batch_sweep.py --batches 2 4 8 --steps 15
# output: peak GPU memory per batch + seconds/step + windows/second; stop immediately on OOM.
# Note: v8.0's b8/b20 are unfrozen-regime reference values; after V7 freezing there is no
# gradient graph / no Adam state for the encoder, so the memory profile differs and must
# be measured empirically.
import argparse, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.api import build_model
from model.contracts import HGTConfig
from data_pipeline.api import load_splits_sliding_window
from data_pipeline.impl.collator import HGTCollator
from torch.utils.data import DataLoader

ap = argparse.ArgumentParser()
ap.add_argument("--batches", type=int, nargs="+", default=[2, 4, 8])
ap.add_argument("--steps", type=int, default=15)
ap.add_argument("--nt_path", default="./model/InstaDeepAI/NTv3_650M_pre")
ap.add_argument("--data_dir", default="simulated_data/Contig_500k")
args = ap.parse_args()

splits = load_splits_sliding_window(data_dir=args.data_dir, window_size=3000,
                                    overlap=300, evo_cache_path="")
val_ds = splits["val"]
print(f"val windows: {len(val_ds)}")

for b in args.batches:
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        model = build_model(HGTConfig(
            data_mode="contig", pretrained_nt_path=args.nt_path,
            pretrained_caduceus_path="", use_local_encoder=False,
            use_evo=False, early_exit_threshold=0.95)).cuda()
        for p in model.backbone.parameters():
            p.requires_grad = False   # contract regime
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5)
        scaler = torch.amp.GradScaler("cuda")
        dl = DataLoader(val_ds, batch_size=b, collate_fn=HGTCollator(pad_id=0), num_workers=2)
        it = iter(dl); ts = []
        for step in range(args.steps):
            t0 = time.time()
            batch = next(it).to("cuda")
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(batch)
                loss = (out.cls_logits.float().sum() + out.role_logits.float().sum()
                        + out.bp_heatmaps.float().sum())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            torch.cuda.synchronize(); ts.append(time.time() - t0)
        ts = ts[2:]  # drop the first two warmup steps
        peak = torch.cuda.max_memory_allocated() / 2**30
        sp = sum(ts) / len(ts)
        print(f"RESULT b={b}: peak={peak:.1f}GB  {sp:.2f}s/step  {b/sp:.2f} win/s")
        del model, opt, dl, it, scaler
    except torch.OutOfMemoryError:
        print(f"RESULT b={b}: OOM")
        break
    except StopIteration:
        print(f"RESULT b={b}: insufficient data for the requested steps")
        break
print("SWEEP DONE")
