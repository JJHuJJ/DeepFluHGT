#!/usr/bin/env python
# scripts/speed_probe_cpu.py — CPU speed probe for NTv3-650M window forward.
# Hard constraint: CUDA invisible, 4 torch threads, 4 OMP threads.
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import torch
torch.set_num_threads(4)
print(f"torch={torch.__version__} cuda_avail={torch.cuda.is_available()} threads={torch.get_num_threads()}")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import yaml
from model.api import build_model
from model.contracts import HGTConfig

CKPT = os.path.join(ROOT, "checkpoints/Contig_500k_ntv3only/best_model.pt")
CFG = os.path.join(ROOT, "configs/contig_500k_ntv3only.yaml")

mc = yaml.safe_load(open(CFG))["model"]
t0 = time.time()
cfg = HGTConfig(
    data_mode="contig",
    max_contig_length=3000,
    hidden_dim=mc["hidden_dim"],
    num_attention_heads=mc["num_attention_heads"],
    dropout=mc["dropout"],
    graph_top_k=mc["graph_top_k"],
    bio_feature_dim=mc["bio_feature_dim"],
    pretrained_nt_path=mc["pretrained_nt_path"],
    pretrained_caduceus_path="",
    use_local_encoder=False,
    use_evo=bool(mc.get("use_evo", False)),
    pretrained_evo_dim=int(mc.get("pretrained_evo_dim", 4096)),
    early_exit_threshold=0.95,
)
model = build_model(cfg)
print(f"build_model: {time.time()-t0:.1f}s")

t0 = time.time()
st = torch.load(CKPT, map_location="cpu", weights_only=False)
print(f"ckpt load: {time.time()-t0:.1f}s  epoch={st.get('epoch')} keys={sorted(st.keys())[:8]}")
c = st.get("contract", {})
print("contract:", {k: c.get(k) for k in ("encoder_frozen", "backbone_sha256", "config_sha256", "dataset_manifest_sha256", "epoch")})
missing, unexpected = model.load_state_dict(st["model"], strict=False)
print(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
print("missing (first 5):", missing[:5])
model.eval()

# Build a fake 3000bp window batch through the real pipeline
from data_pipeline.impl.dataset import SlidingWindowContigDataset
from data_pipeline.impl.collator import HGTCollator
from data_pipeline.contracts import HGTSample
import random
random.seed(0)
seq = "".join(random.choice("ACGT") for _ in range(3000))
samples = [HGTSample(seq_id="probe_0", label=0, sequence=seq, r2_sequence=None,
                     events=[], sample_type="NEG", seq_len=3000)]
ds = SlidingWindowContigDataset(samples=samples, window_size=3000, overlap=300,
                                is_training=False, max_neg_windows=9999,
                                max_eval_windows=10, extract_bio=True)
print("windows:", len(ds))
batch = HGTCollator(pad_id=0)([ds[0]])
print("batch L:", batch.input_ids.shape)

with torch.no_grad():
    # warmup (lazy rotary cache, first kernel compile)
    t0 = time.time(); o = model(batch, early_exit=False); t1 = time.time()
    print(f"warmup forward: {t1-t0:.2f}s  logit={o.cls_logits.item():.4f}")
    times = []
    for i in range(3):
        t0 = time.time(); o = model(batch, early_exit=False); times.append(time.time()-t0)
    print(f"repeat forwards: {[f'{t:.2f}' for t in times]} s  median={sorted(times)[1]:.2f}s")
    o2 = model(batch, early_exit=False)
    print(f"determinism max|d|={abs(o.cls_logits.item()-o2.cls_logits.item()):.2e}")
print(f"RSS check done. threads={torch.get_num_threads()}")
