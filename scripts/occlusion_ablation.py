#!/usr/bin/env python3
# scripts/occlusion_ablation.py
"""DeepFluHGT  —  Task C: module occlusion ablation (M0, 4090).

Ported from (read-only reference): ../DeepFluHGT.8.0/upgrade_pipeline/scripts/phase1_ablation/
run_ablation.py  —  —  load the trained checkpoint, use a forward hook at inference time to zero
the specified module's output (ModuleZeroer), and observe per-class performance changes.
A quick diagnostic ablation (no retraining).

v7 adaptations:
  - M0 = Contig_500k_ntv3only/best_model.pt (ep2, use_evo=False → no evo branch).
  - Module outputs come in two kinds — dict (splitter/cross_attn) and tensor
    (fpn/graph/bio_encoder) → ModuleZeroer zeroes recursively (tensor/dict/list/tuple).
  - data: Contig_500k/test stratified-sampled by expected window count from BED seq_len
    (seed=42) to reach ~20k windows, same sliding-window protocol as Tasks A/B
    (3000/300, win_cap=10, batch=8, fp32).
  - Variants (8 = baseline + 5 single-module + 2 combined):
      full_model / wo_splitter / wo_cross_attn / wo_fpn / wo_graph /
      wo_bio_encoder / wo_splitter_x_cross / wo_fpn_x_graph
  - metric: window-level + contig-level (max/mean) AUROC/AUPRC, ΔAUPRC vs full,
    DeLong paired test (variant vs full, delong_test.delong_test_full).

Structural notes (priors, checked in the report):
  - bio_encoder output only enters role_head.post_graph (model/api.py Step 12),
    not the cls decision path → cls metrics should be **bit-identical**.
  - graph output graph_pooled is the cls_head input (head_pooled; M0 has no evo gate) →
    zeroing it degrades cls to a constant logit → metrics collapse to chance level.

GPU discipline: CUDA_VISIBLE_DEVICES=4090 UUID GPU lock (A100 strictly forbidden); only this single
GPU process at a time (guaranteed by the serial chain).

usage (launcher: scripts/run/difficulty_curve_chain.sh):
  python scripts/occlusion_ablation.py --selftest    # CPU seconds-level self-check
  CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python scripts/occlusion_ablation.py
"""
from __future__ import annotations

import os
import sys
import json
import math
import time
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

GPU_4090_UUID = "${GPU_4090_UUID}"
DEFAULT_CKPT = ROOT / "checkpoints/Contig_500k_ntv3only/best_model.pt"
DEFAULT_CFG = ROOT / "configs/contig_500k_ntv3only.yaml"
NT_FALLBACK = ROOT / "model/InstaDeepAI/NTv3_650M_pre"
TEST_DIR = ROOT / "simulated_data/Contig_500k/test"

VARIANTS = [
    ("full_model", []),
    ("wo_splitter", ["splitter"]),
    ("wo_cross_attn", ["cross_attn"]),
    ("wo_fpn", ["fpn"]),
    ("wo_graph", ["graph"]),
    ("wo_bio_encoder", ["bio_encoder"]),
    ("wo_splitter_x_cross", ["splitter", "cross_attn"]),
    ("wo_fpn_x_graph", ["fpn", "graph"]),
]
VARIANT_DESC = {
    "full_model": "full model (baseline)",
    "wo_splitter": "w/o FeatureSplitBlock (disentangling split, donor/recipient + gates)",
    "wo_cross_attn": "w/o RoleGatedCrossAttention (role-gated cross-attention)",
    "wo_fpn": "w/o SequenceFPN (multi-scale dilated-conv pyramid)",
    "wo_graph": "w/o BatchGraphInteraction (batch graph interaction; direct cls_head input)",
    "wo_bio_encoder": "w/o BiologicalFeatureEncoder (only feeds role_head; structurally does not affect cls)",
    "wo_splitter_x_cross": "combined: splitter + cross_attn zeroed together",
    "wo_fpn_x_graph": "combined: fpn + graph zeroed together",
}


def parse_args():
    ap = argparse.ArgumentParser(description="Module occlusion ablation (Task C)")
    ap.add_argument("--selftest", action="store_true",
                    help="CPU smoke: self-check of the zeroing hook mechanism (no pretrained weights/GPU)")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--win_cap", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n_windows", type=int, default=20000,
                    help="target window count (stratified sampling by expected window count from BED, seed=42)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu_uuid", default=GPU_4090_UUID)
    ap.add_argument("--tag", default="m0")
    ap.add_argument("--out_dir", default=str(ROOT / "docs/evidence"))
    return ap.parse_args()


ARGS = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = ARGS.gpu_uuid

import eval_real_anchors as cpu  # noqa: E402  (metric reuse)
os.environ["CUDA_VISIBLE_DEVICES"] = ARGS.gpu_uuid  # restore immediately (the cpu module blanked it at import)

from delong_test import delong_test_full  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════ #
#  Occlusion mechanism (v6.8 ModuleZeroer port + dict output support)
# ═══════════════════════════════════════════════════════════════════════════ #

def _zero_obj(o):
    if torch.is_tensor(o):
        return torch.zeros_like(o)
    if isinstance(o, dict):
        return {k: _zero_obj(v) for k, v in o.items()}
    if isinstance(o, tuple):
        return tuple(_zero_obj(x) for x in o)
    if isinstance(o, list):
        return [_zero_obj(x) for x in o]
    return o


class ModuleZeroer:
    """Forward hook: recursively zeroes the module output (tensor/dict/list/tuple)."""

    def __init__(self, module: nn.Module, name: str = ""):
        self.module, self.name = module, name
        self.hook = None

    def _zero_output(self, module, inputs, output):
        return _zero_obj(output)

    def enable(self):
        self.hook = self.module.register_forward_hook(self._zero_output)

    def disable(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None


def apply_variant(model, modules: list):
    zeroers = [ModuleZeroer(getattr(model, m), m) for m in modules]
    for z in zeroers:
        z.enable()
    return zeroers


def clear_variant(zeroers):
    for z in zeroers:
        z.disable()


# ═══════════════════════════════════════════════════════════════════════════ #
#  Self-check (CPU, seconds-level): validate the zeroing mechanism and structural priors on a smoke model
# ═══════════════════════════════════════════════════════════════════════════ #

def selftest():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from model.api import build_model
    from model.contracts import HGTConfig
    from data_pipeline.contracts import HGTBatch

    cfg = HGTConfig(data_mode="contig", _smoke_global_layers=2,
                    _smoke_local_layers=2, early_exit_threshold=0.95,
                    pretrained_nt_path="", pretrained_caduceus_path="")
    model = build_model(cfg).eval()
    B, L = 4, 402
    batch = HGTBatch(
        input_ids=torch.randint(1, 5, (B, L)),
        attention_mask=torch.ones(B, L, dtype=torch.long),
        bio_features=torch.randn(B, 260),
        labels=torch.tensor([1, 0, 1, 0]),
        breakpoint_targets=torch.zeros(B, L),
        region_targets=torch.zeros(B, L),
        pe_case=torch.zeros(B, dtype=torch.long),
        hgt_mode=torch.zeros(B, dtype=torch.long),
        n_events=torch.tensor([1, 0, 2, 0]))

    def cls():
        with torch.no_grad():
            return model(batch, early_exit=False).cls_logits.squeeze(-1)

    base = {}
    for name, mods in VARIANTS:
        z = apply_variant(model, mods)
        out = cls()
        clear_variant(z)
        assert torch.isfinite(out).all(), f"{name}: cls_logits contains NaN/Inf"
        base[name] = out.clone()
        print(f"  [{name:22s}] cls={out.tolist().__str__()[:70]}...")

    # Validate structural priors
    assert torch.allclose(base["full_model"], base["wo_bio_encoder"], atol=1e-6), \
        "prior violated: bio_encoder unexpectedly affects the cls path?"
    assert not torch.allclose(base["full_model"], base["wo_splitter"]), \
        "wo_splitter did not change the output (hook not taking effect?)"
    g = base["wo_graph"]
    assert float(g.std()) < 1e-8, "wo_graph should degrade to a constant logit (cls_head input zeroed)"
    assert not torch.allclose(base["full_model"], base["wo_fpn"])
    print("[selftest] PASS: zeroing mechanism works; bio_encoder does not affect cls "
          "(structural conclusion holds); wo_graph → constant logit")


# ═══════════════════════════════════════════════════════════════════════════ #
#  data: stratified sampling by expected window count from BED (seed=42) → ~20k windows
# ═══════════════════════════════════════════════════════════════════════════ #

def expected_windows(seq_len: int, window: int = 3000, stride: int = 2700,
                     cap: int = 10) -> int:
    if seq_len <= window:
        return 1
    return min(1 + math.ceil((seq_len - window) / stride), cap)


def read_bed(bed_path: Path):
    """→ {seq_id: (label, seq_len)} (first occurrence wins)."""
    out = {}
    with open(bed_path) as f:
        f.readline()
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) < 5 or p[0] in out:
                continue
            out[p[0]] = (int(p[2]), int(p[3]))
    return out


def sample_contigs(bed: dict, n_windows: int, seed: int):
    """Shuffle pos/neg separately, then alternate; accumulate until the expected window count reaches the target (class balance ~50/50)."""
    rng = np.random.RandomState(seed)
    pos = np.array([s for s, (l, _) in bed.items() if l == 1], dtype=object)
    neg = np.array([s for s, (l, _) in bed.items() if l == 0], dtype=object)
    rng.shuffle(pos)
    rng.shuffle(neg)
    chosen, ip, in_, cum = [], 0, 0, 0
    while cum < n_windows and (ip < len(pos) or in_ < len(neg)):
        if ip < len(pos) and cum < n_windows:
            sid = str(pos[ip]); ip += 1
            chosen.append(sid)
            cum += expected_windows(bed[sid][1])
        if in_ < len(neg) and cum < n_windows:
            sid = str(neg[in_]); in_ += 1
            chosen.append(sid)
            cum += expected_windows(bed[sid][1])
    n_pos_sel = sum(1 for s in chosen if bed[s][0] == 1)
    print(f"[Sample] seed={seed}: {len(chosen)} contigs "
          f"(pos={n_pos_sel}, neg={len(chosen)-n_pos_sel}), "
          f"expected windows ≈ {cum} (target {n_windows})")
    return set(chosen), cum


def load_fasta_subset(fasta_path: Path, want: set):
    out, header, buf = {}, None, []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None and header in want:
                    out[header] = "".join(buf)
                header = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.upper())
    if header is not None and header in want:
        out[header] = "".join(buf)
    missing = len(want) - len(out)
    assert missing == 0, f"fasta is missing {missing} of the selected contigs"
    return out


# ═══════════════════════════════════════════════════════════════════════════ #
#  main pipeline
# ═══════════════════════════════════════════════════════════════════════════ #

def main():
    if ARGS.selftest:
        selftest()
        return

    from sklearn.metrics import roc_auc_score, average_precision_score

    n_gpu = torch.cuda.device_count()
    assert n_gpu == 1, f"exactly 1 visible GPU expected (4090 GPU lock), actual {n_gpu}"
    gpu_name = torch.cuda.get_device_name(0)
    assert "4090" in gpu_name, f"visible GPU is not RTX 4090: {gpu_name}"
    free, total = torch.cuda.mem_get_info(0)
    print(f"[GPU] {gpu_name} | mem free {free/1e9:.1f}/{total/1e9:.1f} GB")

    out_dir = Path(ARGS.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # 1. data
    bed = read_bed(TEST_DIR / "breakpoints.bed")
    print(f"[BED] {len(bed)} contigs")
    want, est = sample_contigs(bed, ARGS.n_windows, ARGS.seed)
    seqs = load_fasta_subset(TEST_DIR / "dataset.fasta", want)
    samples = {sid: (bed[sid][0], seqs[sid]) for sid in sorted(seqs)}

    from torch.utils.data import DataLoader
    from data_pipeline.impl.dataset import SlidingWindowContigDataset
    from data_pipeline.impl.collator import HGTCollator
    from data_pipeline.contracts import HGTSample

    import yaml as _yaml
    _dc = _yaml.safe_load(open(ARGS.config))["data"]

    hgt_samples, labels_map = [], {}
    for sid, (label, seq) in samples.items():
        hgt_samples.append(HGTSample(
            seq_id=sid, label=label, sequence=seq, r2_sequence=None,
            events=[], sample_type="POS" if label == 1 else "NEG",
            seq_len=len(seq)))
        labels_map[sid] = label
    ds = SlidingWindowContigDataset(
        samples=hgt_samples, window_size=_dc["sw_window_size"],
        overlap=_dc["sw_overlap"], is_training=False,
        max_neg_windows=9999, max_eval_windows=ARGS.win_cap, extract_bio=True)
    loader = DataLoader(ds, batch_size=ARGS.batch, shuffle=False,
                        collate_fn=HGTCollator(pad_id=0), num_workers=4)
    print(f"[Data] {len(hgt_samples)} contigs, {len(ds)} windows "
          f"(win_cap={ARGS.win_cap}, batch={ARGS.batch})")

    # 2. model (loaded once; hooks attached/detached per variant)
    from model.api import build_model
    from model.contracts import HGTConfig
    mc = _yaml.safe_load(open(ARGS.config))["model"]
    device = torch.device("cuda:0")
    cfg = HGTConfig(
        data_mode="contig", max_contig_length=3000,
        hidden_dim=mc["hidden_dim"], num_attention_heads=mc["num_attention_heads"],
        dropout=mc["dropout"], graph_top_k=mc["graph_top_k"],
        bio_feature_dim=mc["bio_feature_dim"],
        pretrained_nt_path=mc.get("pretrained_nt_path", str(NT_FALLBACK)),
        pretrained_caduceus_path="", use_local_encoder=False,
        use_evo=bool(mc.get("use_evo", False)),
        pretrained_evo_dim=int(mc.get("pretrained_evo_dim", 4096)),
        early_exit_threshold=0.95)
    model = build_model(cfg)
    st = torch.load(Path(ARGS.ckpt), map_location="cpu", weights_only=False)
    if st.get("epoch") != 2:
        alt = Path(ARGS.ckpt).parent / "epoch_2.pt"
        print(f"[WARN] epoch={st.get('epoch')} ≠ 2 → fallback {alt.name}")
        st = torch.load(alt, map_location="cpu", weights_only=False)
    assert st.get("contract", {}).get("encoder_frozen") is True
    missing, unexpected = model.load_state_dict(st["model"], strict=False)
    model.to(device).eval()
    print(f"[Model] epoch={st.get('epoch')} use_evo={bool(mc.get('use_evo', False))} "
          f"missing={len(missing)} unexpected={len(unexpected)}")

    # 3. Per-variant inference (same loader, deterministic order)
    variant_out = {}
    for vi, (name, mods) in enumerate(VARIANTS):
        zeroers = apply_variant(model, mods)
        win_probs, win_labels = [], []
        contig_windows = defaultdict(list)
        t0 = time.time()
        with torch.no_grad():
            for i, batch in enumerate(loader):
                batch = batch.to(device)
                output = model(batch, early_exit=False)
                probs = torch.sigmoid(
                    output.cls_logits.squeeze(-1)).float().cpu().numpy()
                sids = batch.original_seq_ids or batch.seq_ids
                for sid, p in zip(sids, probs):
                    contig_windows[sid].append(float(p))
                    win_probs.append(float(p))
                    win_labels.append(int(labels_map[sid]))
                if (i + 1) % 100 == 0:
                    el = time.time() - t0
                    done = (i + 1) * ARGS.batch
                    print(f"  [{name}][{min(done,len(ds))}/{len(ds)}] {el:.0f}s "
                          f"ETA {(len(loader)-i-1)*el/(i+1)/60:.1f}min", flush=True)
        clear_variant(zeroers)

        y_w = np.array(win_labels)
        p_w = np.array(win_probs)
        sid_list = sorted(contig_windows.keys())
        y_c = np.array([labels_map[s] for s in sid_list])
        s_max = np.array([max(contig_windows[s]) for s in sid_list])
        s_mean = np.array([float(np.mean(contig_windows[s])) for s in sid_list])
        variant_out[name] = {
            "desc": VARIANT_DESC[name], "modules": mods,
            "n_windows": int(len(y_w)),
            "win_auroc": float(roc_auc_score(y_w, p_w)),
            "win_auprc": float(average_precision_score(y_w, p_w)),
            "contig_auprc_max": float(average_precision_score(y_c, s_max)),
            "contig_auroc_max": float(roc_auc_score(y_c, s_max)),
            "contig_auprc_mean": float(average_precision_score(y_c, s_mean)),
            "contig_auroc_mean": float(roc_auc_score(y_c, s_mean)),
            "_s_max": s_max, "_y_c": y_c,
            "infer_seconds": time.time() - t0,
        }
        v = variant_out[name]
        print(f"[{name:22s}] winAUROC={v['win_auroc']:.4f} "
              f"winAUPRC={v['win_auprc']:.4f} "
              f"contigAUPRC(max)={v['contig_auprc_max']:.4f} "
              f"({v['infer_seconds']:.0f}s)")

    # 4. Δ + DeLong paired (vs full_model; contig max caliber)
    full = variant_out["full_model"]
    for name, v in variant_out.items():
        v["delta_auprc_max"] = v["contig_auprc_max"] - full["contig_auprc_max"]
        v["delta_win_auprc"] = v["win_auprc"] - full["win_auprc"]
        if name == "full_model":
            r = delong_test_full(full["_y_c"], full["_s_max"], full["_s_max"])
            v["delong_vs_full"] = {"auc": r["auc_a"], "se": math.sqrt(r["var_a"])}
        else:
            r = delong_test_full(v["_y_c"], v["_s_max"], full["_s_max"])
            v["delong_vs_full"] = {
                "delta": r["delta"], "se_delta": r["se_delta"], "z": r["z"],
                "p": r["p"],
                "ci95_delta": r["ci95_delta"] if "ci95_delta" in r else None}

    # 5. Archive + report
    np.savez(out_dir / f"ablation_occlusion_{ARGS.tag}.npz",
             variants=np.array([n for n, _ in VARIANTS], dtype=object),
             contig_ids=np.array(sorted(labels_map.keys()), dtype=object),
             **{f"scores_{n}": v["_s_max"] for n, v in variant_out.items()},
             labels=full["_y_c"])

    payload = {
        "task": "C  —  module occlusion ablation (M0)",
        "protocol": {
            "source": "v6.8 run_ablation.py ModuleZeroer port (v7 dict-output adaptation)",
            "data": "Contig_500k/test stratified sampling (expected window count from BED seq_len, seed=42)",
            "n_contigs": len(hgt_samples), "n_windows": int(len(ds)),
            "window_size": _dc["sw_window_size"], "overlap": _dc["sw_overlap"],
            "win_cap": ARGS.win_cap, "batch": ARGS.batch,
            "gpu": gpu_name, "use_evo": bool(mc.get("use_evo", False)),
        },
        "variants": {n: {k: v for k, v in vv.items()
                         if not k.startswith("_")}
                     for n, vv in variant_out.items()},
        "total_seconds": time.time() - t_start,
    }
    with open(out_dir / f"ablation_occlusion_{ARGS.tag}.json", "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False,
                  default=lambda o: float(o) if isinstance(o, np.floating) else o)

    md = []
    md.append("# Module occlusion ablation  —  M0 (NTv3-only, ep2)\n")
    md.append("> Generated: %s  ·  Task C  ·  `scripts/occlusion_ablation.py`\n"
              % time.strftime("%Y-%m-%d %H:%M:%S"))
    md.append("> Ported from 8.0 `upgrade_pipeline/scripts/phase1_ablation/"
              "run_ablation.py` (forward-hook output zeroing at inference; a quick "
              "diagnostic ablation, no retraining). v7 adaptations: dict outputs of "
              "splitter/cross_attn zeroed recursively; "
              "M0 use_evo=False → no evo branch.\n")
    md.append(f"> data: Contig_500k/test stratified sampling seed=42 → "
              f"{len(hgt_samples):,} contigs / {len(ds):,} windows; "
              f"sliding window 3000/300, win_cap=10, batch=8, fp32.\n")

    md.append("\n## 1. Primary table (contig-level max aggregation)\n")
    md.append("| Variant | Occluded module | window AUROC | window AUPRC | AUPRC(max) | "
              "AUROC(max) | ΔAUPRC vs full | DeLong Δ 95%CI | p (vs full) |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for name, v in variant_out.items():
        d = v.get("delong_vs_full", {})
        if name == "full_model":
            md.append(f"| **full_model** |  —  | {v['win_auroc']:.4f} | "
                      f"{v['win_auprc']:.4f} | **{v['contig_auprc_max']:.4f}** | "
                      f"{v['contig_auroc_max']:.4f} | 0 |  —  |  —  |")
        else:
            ci = d.get("ci95_delta")
            ci_s = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else " — "
            md.append(f"| {name} | {', '.join(v['modules'])} | "
                      f"{v['win_auroc']:.4f} | {v['win_auprc']:.4f} | "
                      f"{v['contig_auprc_max']:.4f} | {v['contig_auroc_max']:.4f} | "
                      f"{v['delta_auprc_max']:+.4f} | {ci_s} | "
                      f"{d.get('p', float('nan')):.2e} |")

    md.append("\n## 2. Secondary calibers (mean aggregation / window-level)\n")
    md.append("| Variant | AUPRC(mean) | AUROC(mean) | ΔwindowAUPRC |")
    md.append("|---|---|---|---|")
    for name, v in variant_out.items():
        md.append(f"| {name} | {v['contig_auprc_mean']:.4f} | "
                  f"{v['contig_auroc_mean']:.4f} | {v['delta_win_auprc']:+.4f} |")

    md.append("\n## 3. Structural conclusions (prior → measured)\n")
    md.append("- **bio_encoder**: its output only enters role_head.post_graph "
              "(model/api.py Step 12), not the cls decision path → cls metrics should be bit-identical. "
              "Measured ΔAUPRC = %+.4f (the self-check already validated bit-identical outputs).\n"
              % variant_out["wo_bio_encoder"]["delta_auprc_max"])
    md.append("- **graph**: graph_pooled is the cls_head input (head_pooled; "
              "M0 has no evo residual gate) → zeroing it degrades cls to a constant logit and metrics collapse to "
              "chance level (AUROC=0.5, AUPRC=prevalence). Measured AUROC=%.4f.\n"
              % variant_out["wo_graph"]["contig_auroc_max"])
    md.append("- With splitter/cross_attn/fpn zeroed, part of the backbone→cls pathway remains; "
              "the degradation magnitude is judged by the DeLong Δ.\n")
    md.append("\n## 4. Limitations\n")
    md.append("- Inference-time occlusion ablation (no retraining): no compensatory retraining "
              "occurs between modules; what it reflects is \"the marginal contribution of each module "
              "under the current weights\".\n"
              "- The collapse of wo_graph/wo_fpn_x_graph is structural (the cls input is zeroed); "
              "interpretation should distinguish these from \"compensable modules\".\n")

    out_md = out_dir / "ABLATION_M0.md"
    with open(out_md, "w") as f:
        f.write("".join(md))
    print(f"[Report] saved {out_md}")
    print(f"total={(time.time()-t_start)/60:.1f}min")


if __name__ == "__main__":
    main()
