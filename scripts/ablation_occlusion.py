#!/usr/bin/env python3
# scripts/ablation_occlusion.py
"""DeepFluHGT — task C: M0 moduleocclusion ablation (occlusion ablation, 4090).

Ported from (read-only) ../DeepFluHGT.8.0/upgrade_pipeline/scripts/phase1_ablation/
run_ablation.py:
  - ModuleZeroer (forward hook willmoduleoutputzeroed) verbatim port;
  - 8 variantset splitter/cross_attn/fpn/graph/bio_encoder (+bp_refiner) →
    same-named modules in the API (model/api.py: FeatureSplitBlock / RoleGatedCrossAttention /
    SequenceFPN / BatchGraphInteraction / BiologicalFeatureEncoder);
  - data/inference protocol upgraded to the contig sliding-window pipeline (v6.8 was whole-sequence direct pass):
    Contig_500k/test sample ~20k window (seed=42 stratifieddeterministic), withtask A/B same type
    windowing/aggregation (window=3000/overlap=300/win_cap=10, contig level max/mean aggregation),
    8 variants share the same window set (pairedcomparable) + paired DeLong (vs full_model).

Scope (same caveat as in 8.0): rapid diagnostic ablation (inference-time occlusion, no retraining) — identifies key modules;
formal paper ablation requires retraining.

M0 = use_evo=False (evo_encoder=None, fuse_alpha=None) → no-evo variant.

GPU discipline: CUDA_VISIBLE_DEVICES=4090 UUID GPU lock (A100 strictly forbidden); at any moment only this one
GPU process (guaranteed by the chain script's serialization).

usage (launcher: scripts/run/difficulty_curve_chain.sh):
  CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python scripts/ablation_occlusion.py
  python scripts/ablation_occlusion.py --plan # CPU pilot (samplewindowstatistics)
  CUDA_VISIBLE_DEVICES=... python scripts/ablation_occlusion.py \
      --n_windows 400 --variants full_model wo_graph --out_dir /tmp/x # smoke
"""
from __future__ import annotations

import os
import sys
import json
import math
import time
import random
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
TEST_FASTA = ROOT / "simulated_data/Contig_500k/test/dataset.fasta"
TEST_PROV = ROOT / "simulated_data/Contig_500k/test/provenance.tsv"


def parse_args():
    ap = argparse.ArgumentParser(description="M0 moduleocclusion ablation (task C)")
    ap.add_argument("--n_windows", type=int, default=20000,
                    help="sampled-window budget (approximate; truncated at contig granularity; seed=42 deterministic)")
    ap.add_argument("--variants", nargs="*", default=[],
                    help="run only the given variants (default: all 8)")
    ap.add_argument("--plan", action="store_true",
                    help="CPU pilot: samplingwithwindowstatistics, notloadmodel")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--win_cap", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu_uuid", default=GPU_4090_UUID)
    ap.add_argument("--tag", default="ablation_m0")
    ap.add_argument("--out_dir", default=str(ROOT / "docs/evidence"))
    return ap.parse_args()


ARGS = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = ARGS.gpu_uuid

import eval_real_anchors as cpu # noqa: E402 (windowing/aggregation/metricverbatim reuse)
os.environ["CUDA_VISIBLE_DEVICES"] = ARGS.gpu_uuid  # re-assert right after import (the cpu module clears it)

from delong_test import delong_test_full # noqa: E402

import torch # noqa: E402
import torch.nn as nn # noqa: E402


def assert_4090():
    n = torch.cuda.device_count()
    assert n == 1, f"exactly 1 visible GPU expected (4090 GPU lock), actual {n}"
    name = torch.cuda.get_device_name(0)
    assert "4090" in name, f"visible GPU is not RTX 4090: {name}"
    free, total = torch.cuda.mem_get_info(0)
    print(f"[GPU] {name} | mem free {free/1e9:.1f}/{total/1e9:.1f} GB | "
          f"CUDA {torch.version.cuda} | torch {torch.__version__}")
    return name


# =========================================================================== #
# ModuleZeroer (v6.8 run_ablation.py verbatim port: forward-hook outputzeroed)
# =========================================================================== #

class ModuleZeroer:
    def __init__(self, module):
        self.module = module
        self.hook = None

    def zero_output(self, module, inp, output):
        if isinstance(output, tuple):
            return tuple(torch.zeros_like(o) if isinstance(o, torch.Tensor) else o
                         for o in output)
        elif isinstance(output, torch.Tensor):
            return torch.zeros_like(output)
        return output

    def enable(self):
        self.hook = self.module.register_forward_hook(self.zero_output)

    def disable(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None


VARIANT_DEFS = {
    "full_model": ([], "completemodel (baseline)"),
    "wo_splitter": (["splitter"], "w/o FeatureSplitBlock (disentangled split)"),
    "wo_cross_attn": (["cross_attn"], "w/o RoleGatedCrossAttention (role-gated cross attention)"),
    "wo_fpn": (["fpn"], "w/o SequenceFPN (multi-scale pyramid)"),
    "wo_graph": (["graph"], "w/o BatchGraphInteraction (batchfigureinteract)"),
    "wo_bio_encoder": (["bio_encoder"], "w/o BiologicalFeatureEncoder (260-dim biological features)"),
    "wo_fpn__graph": (["fpn", "graph"], "w/o FPN+Graph (twomodulejoint occlusion)"),
    "wo_splitter__cross_attn": (["splitter", "cross_attn"], "w/o Splitter+CrossAttn (joint occlusion)"),
}


# =========================================================================== #
# sampling: Contig_500k/test → ~N window (seed=42, stratifieddeterministic, allvarianttotaluse)
# =========================================================================== #

def n_eval_windows(seq_len: int, window: int = 3000, stride: int = 300,
                   cap: int = 10) -> int:
    """verbatim replication SlidingWindowContigDataset._generate_full_coverage ofwindowcount
    (events=[] → all windows become NEG type, uniformly subject to the cap)."""
    if seq_len <= window:
        return 1
    starts = list(range(0, seq_len - window + 1, stride))
    if not starts or starts[-1] + window < seq_len:
        starts.append(seq_len - window)
    n = len(sorted(set(starts)))
    return min(n, cap) if cap > 0 else n


def read_labels(prov_path: Path) -> dict:
    labels = {}
    with open(prov_path) as f:
        header = f.readline().rstrip("\n").split("\t")
        i_sid, i_lab = header.index("seq_id"), header.index("label")
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) > max(i_sid, i_lab) and p[i_sid] not in labels:
                labels[p[i_sid]] = int(p[i_lab])
    return labels


def fasta_lengths(fasta: Path) -> dict:
    out, header, n = {}, None, 0
    with open(fasta) as f:
        for line in f:
            if line.startswith(">"):
                if header is not None:
                    out[header] = n
                header = line[1:].split()[0]
                n = 0
            else:
                n += len(line.strip())
    if header is not None:
        out[header] = n
    return out


def fasta_selected(fasta: Path, wanted: set) -> dict:
    out, header, buf = {}, None, []
    with open(fasta) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None and header in wanted:
                    out[header] = "".join(buf)
                header = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.upper())
    if header is not None and header in wanted:
        out[header] = "".join(buf)
    return out


def sample_contigs(target_windows: int, seed: int, window=3000, stride=300, cap=10):
    """→ (chosen_ids ordered, labels, total_windows); pos/neg alternate until the budget is reached."""
    labels = read_labels(TEST_PROV)
    print(f"[Prov] {len(labels)} contigs (pos={sum(labels.values())})")
    lens = fasta_lengths(TEST_FASTA)
    print(f"[FASTA] {len(lens)} records, lengths pass done "
          f"({TEST_FASTA.stat().st_size/1e6:.0f} MB)")

    rng = random.Random(seed)
    pos = [s for s in labels if labels[s] == 1 and s in lens]
    neg = [s for s in labels if labels[s] == 0 and s in lens]
    rng.shuffle(pos)
    rng.shuffle(neg)
    print(f"[Pool] pos={len(pos)} neg={len(neg)} (seed={seed})")

    chosen, total, i_pos, i_neg = [], 0, 0, 0
    while total < target_windows and (i_pos < len(pos) or i_neg < len(neg)):
        if i_pos < len(pos):
            s = pos[i_pos]; i_pos += 1
            chosen.append(s); total += n_eval_windows(lens[s], window, stride, cap)
        if total >= target_windows:
            break
        if i_neg < len(neg):
            s = neg[i_neg]; i_neg += 1
            chosen.append(s); total += n_eval_windows(lens[s], window, stride, cap)

    n_p = sum(1 for s in chosen if labels[s] == 1)
    lens_ch = np.array([lens[s] for s in chosen])
    print(f"[Sample] {len(chosen)} contigs → ~{total} windows "
          f"(pos={n_p} neg={len(chosen)-n_p}, len p50={np.median(lens_ch):.0f}bp "
          f"p90={np.percentile(lens_ch,90):.0f}bp max={lens_ch.max()}bp)")
    return chosen, labels, total, lens


# =========================================================================== #
# model + splitvariantinference (modelbuildverbatim replication eval_real_anchors_gpu)
# =========================================================================== #

def load_model_gpu(ckpt: Path, config: Path):
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
        early_exit_threshold=0.95,
    )
    model = build_model(cfg)

    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    ckpt_epoch = st.get("epoch")
    contract = st.get("contract", {})
    if ckpt_epoch != 2:
        alt = ckpt.parent / "epoch_2.pt"
        print(f"[WARN] {ckpt.name} epoch={ckpt_epoch} ≠ 2 → fallback {alt.name}")
        st = torch.load(alt, map_location="cpu", weights_only=False)
        ckpt_epoch = st.get("epoch")
        contract = st.get("contract", {})
    missing, unexpected = model.load_state_dict(st["model"], strict=False)
    model.to(device).eval()

    assert contract.get("encoder_frozen") is True, "contract: encoder_frozen must be True"
    for mod in ["splitter", "cross_attn", "fpn", "graph", "bio_encoder"]:
        assert hasattr(model, mod) and getattr(model, mod) is not None, \
            f"modelmissingcanablationmodule: {mod}"
    print(f"[Model] epoch={ckpt_epoch} use_evo={bool(mc.get('use_evo', False))} "
          f"load: missing={len(missing)} unexpected={len(unexpected)} → {device}")
    return model, device


def predict_variant(model, ds, labels_map, batch_size: int, device,
                    variant: str, modules: list):
    """install zeroing hooks → run the loader inference → remove hooks. Returns contig_windows."""
    zeroers = []
    for mname in modules:
        z = ModuleZeroer(getattr(model, mname))
        z.enable()
        zeroers.append(z)

    from torch.utils.data import DataLoader
    from data_pipeline.impl.collator import HGTCollator
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=HGTCollator(pad_id=0), num_workers=4)

    contig_windows = defaultdict(list)
    n_nan = 0
    t0 = time.time()
    try:
        with torch.no_grad():
            for i, batch in enumerate(loader):
                batch = batch.to(device)
                output = model(batch, early_exit=False)
                probs = (torch.sigmoid(output.cls_logits.squeeze(-1))
                         .float().cpu().numpy())
                n_nan += int(np.isnan(probs).sum())
                probs = np.nan_to_num(probs, nan=0.5)
                sids = batch.original_seq_ids or batch.seq_ids
                for sid, p in zip(sids, probs):
                    contig_windows[sid].append(float(p))
                if (i + 1) % 50 == 0:
                    el = time.time() - t0
                    print(f" [{(i+1)*batch_size}/{len(ds)}] {el:.0f}s "
                          f"(ETA {(len(loader)-i-1)*el/(i+1)/60:.1f}min)",
                          flush=True)
    finally:
        for z in zeroers:
            z.disable()
    infer_s = time.time() - t0
    print(f" [{variant}] {len(ds)} windows in {infer_s:.0f}s "
          f"({infer_s/max(1,len(ds)):.3f}s/win, nan→0.5: {n_nan})")
    return contig_windows, infer_s, n_nan


def _agg(kind, w):
    if kind == "max":
        return max(w)
    return float(np.mean(w))


# =========================================================================== #
# report
# =========================================================================== #

def write_md(out_dir: Path, tag, records, meta):
    base = records[0] if records else None
    md = []
    md.append("# moduleocclusion ablation — M0 (NTv3-only, ep2)\n\n")
    md.append(f"> generate: {time.strftime('%Y-%m-%d %H:%M:%S')} · task C · "
              "`scripts/ablation_occlusion.py`\n\n")
    md.append("> method: forward-hook **outputzeroed** (inferenceocclusion, no retraining; rapid diagnostic ablation, "
              "formal paper ablation requires retraining). Ported from 8.0 "
              "`upgrade_pipeline/scripts/phase1_ablation/run_ablation.py` "
              "(read-only), data/inferenceprotocolupgraded for the contig sliding-window pipeline "
              "(window=3000/overlap=300/win_cap=10, contig levelaggregation), "
              "withtask A/B same type; 8 variants share the same window set (pairedcomparable).\n\n")
    md.append(f"> sample: Contig_500k/test stratifieddeterministicsampling seed={meta['seed']} → "
              f"**{meta['n_windows']} window** / {meta['n_contigs']:,} contigs "
              f"(pos={meta['n_pos']:,} / neg={meta['n_neg']:,}); "
              f"batch={meta['batch']}, fp32, 4090.\n\n")
    md.append("> model: `checkpoints/Contig_500k_ntv3only/best_model.pt` (ep2, "
              "use_evo=False → no-evo variant; encoder (NTv3) frozennotablation).\n\n")

    md.append("## 1. variant x performance (contig-level max aggregation as the primary protocol)\n\n")
    md.append("| variant | occlusionmodule | AUPRC | AUROC | F1@opt | ΔAUPRC | ΔAUROC | "
              "DeLong p (vs full) |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in records:
        mods = "+".join(r["modules"]) or " — "
        md.append(f"| **{r['variant']}** | {mods} | {r['auprc_max']:.4f} | "
                  f"{r['auroc_max']:.4f} | {r['f1_at_optimal']:.4f} | "
                  f"{r['delta_auprc']:+.4f} | {r['delta_auroc']:+.4f} | "
                  f"{r['delong_p_vs_full']:.2e} |")
    md.append("")
    md.append("## 2. mean aggregationcontrol\n\n")
    md.append("| variant | AUPRC(mean) | AUROC(mean) | ΔAUPRC(mean) |")
    md.append("|---|---|---|---|")
    for r in records:
        md.append(f"| {r['variant']} | {r['auprc_mean']:.4f} | "
                  f"{r['auroc_mean']:.4f} | {r['delta_auprc_mean']:+.4f} |")
    md.append("")
    md.append("## 3. How to read + notes\n\n")
    md.append("- the Δ<0 magnitude = that module's marginal contribution in the M0 inference path; DeLong is a **paired** test "
              "(same window set, `delong_test.delong_test_full`, reporting the p of the delta).\n")
    md.append("- zeroed outputs propagate through downstream LayerNorm/gating; joint occlusion (fpn+graph, "
              "splitter+cross_attn) probes path redundancy.\n")
    md.append("- bio_encoder in the graph only feeds role_head (post_graph), not cls_head → "
              "if ΔAUPRC≈0 this is **structurally expected** (the biological-feature branch does not touch the split-class path), not a failure.\n")
    md.append("- inference-time occlusion underestimates the post-retraining contribution (the other modules cannot compensate); the qualitative ranking is trustworthy, "
              "the magnitudes are conservative (same statement as the original v6.8 script).\n")
    md.append("")
    md.append(f"## 4. artifact\n\n- `{tag}.json` / `{tag}.npz` (eachvariantper- contig splitcount, verifiable)\n")

    out = out_dir / "ABLATION_M0.md"
    with open(out, "w") as f:
        f.write("\n".join(md))
    print(f"[Report] saved {out}")


# =========================================================================== #

def main():
    t_start = time.time()
    out_dir = Path(ARGS.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = ARGS.tag

    chosen, labels, est_windows, lens_ch = sample_contigs(
        ARGS.n_windows, ARGS.seed, window=3000, stride=300, cap=ARGS.win_cap)

    if ARGS.plan:
        print(f"[Plan] CPU pilotend: ~{est_windows} window / {len(chosen)} contigs")
        return

    gpu_name = assert_4090()
    seqs = fasta_selected(TEST_FASTA, set(chosen))
    print(f"[FASTA] fetched {len(seqs)} contig sequences "
          f"({sum(len(s) for s in seqs.values())/1e6:.1f} Mbp)")

    from data_pipeline.impl.dataset import SlidingWindowContigDataset
    from data_pipeline.contracts import HGTSample
    hgt_samples, labels_map = [], {}
    for sid in chosen:
        seq = seqs[sid]
        hgt_samples.append(HGTSample(
            seq_id=sid, label=labels[sid], sequence=seq, r2_sequence=None,
            events=[], sample_type="POS" if labels[sid] == 1 else "NEG",
            seq_len=len(seq)))
        labels_map[sid] = labels[sid]
    ds = SlidingWindowContigDataset(
        samples=hgt_samples, window_size=3000, overlap=300,
        is_training=False, max_neg_windows=9999, max_eval_windows=ARGS.win_cap,
        extract_bio=True)
    print(f"[Data] {len(hgt_samples)} contigs, {len(ds)} windows "
          f"(win_cap={ARGS.win_cap}, batch={ARGS.batch})")

    model, device = load_model_gpu(Path(ARGS.ckpt), Path(ARGS.config))

    variants = ARGS.variants or list(VARIANT_DEFS.keys())
    for v in variants:
        assert v in VARIANT_DEFS, f"not yetknownvariant {v}: {list(VARIANT_DEFS)}"

    records, scores_archive = [], {}
    for v in variants:
        modules, desc = VARIANT_DEFS[v]
        print(f"\n[Variant] {v}: {desc} (zero: {modules or 'none'})")
        cw, infer_s, n_nan = predict_variant(
            model, ds, labels_map, ARGS.batch, device, v, modules)

        res = cpu.aggregate_and_eval(cw, labels_map)
        sid_list = sorted(cw.keys())
        y = np.array([labels_map[s] for s in sid_list], dtype=np.int64)
        s_max = np.array([_agg("max", cw[s]) for s in sid_list], dtype=np.float32)
        s_mean = np.array([_agg("mean", cw[s]) for s in sid_list], dtype=np.float32)
        scores_archive[f"{v}__ids"] = np.array(sid_list, dtype=object)
        scores_archive[f"{v}__y"] = y
        scores_archive[f"{v}__max"] = s_max
        scores_archive[f"{v}__mean"] = s_mean

        records.append({
            "variant": v, "modules": modules, "desc": desc,
            "n_contigs": len(sid_list), "n_windows_infer": len(ds),
            "nan_count": n_nan,
            "auprc_max": res["max"]["auprc"], "auroc_max": res["max"]["auc"],
            "f1_at_optimal": res["max"]["f1_at_optimal"],
            "auprc_mean": res["mean"]["auprc"], "auroc_mean": res["mean"]["auc"],
            "infer_seconds": infer_s,
            "agg_max_scores": s_max, "y": y, "_sid": sid_list,
        })

    # baselinecontrol + paired DeLong
    base = next((r for r in records if r["variant"] == "full_model"), None)
    for r in records:
        if base is not None:
            r["delta_auprc"] = r["auprc_max"] - base["auprc_max"]
            r["delta_auroc"] = r["auroc_max"] - base["auroc_max"]
            r["delta_auprc_mean"] = r["auprc_mean"] - base["auprc_mean"]
            if r is not base and len(r["y"]) == len(base["y"]) and \
                    np.array_equal(r["_sid"], base["_sid"]):
                d = delong_test_full(r["y"], base["agg_max_scores"],
                                     r["agg_max_scores"])
                r["delong_p_vs_full"] = d["p"]
                r["delong_delta"] = d["delta"]
            else:
                r["delong_p_vs_full"] = float("nan")
        else:
            r["delta_auprc"] = r["delta_auroc"] = r["delta_auprc_mean"] = 0.0
            r["delong_p_vs_full"] = float("nan")

    meta = {"n_windows": len(ds), "n_contigs": len(chosen),
            "n_pos": sum(labels[s] for s in chosen), "n_neg": len(chosen) - sum(labels[s] for s in chosen),
            "seed": ARGS.seed, "batch": ARGS.batch, "win_cap": ARGS.win_cap,
            "gpu": gpu_name, "target_windows": ARGS.n_windows}
    np.savez(out_dir / f"{tag}.npz", **scores_archive)

    dump = [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
             for k, v in r.items() if not k.startswith("_")} for r in records]
    with open(out_dir / f"{tag}.json", "w") as f:
        json.dump({"task": "C_ablation_occlusion",
                   "meta": meta,
                   "method": "forward-hook output zeroing (ported from v6.8 run_ablation_v2; "
                             "inferenceocclusionno retraining)",
                   "variants": dump},
                  f, indent=2, ensure_ascii=False,
                  default=lambda o: float(o) if isinstance(o, (np.floating,)) else o)

    write_md(out_dir, tag, records, meta)

    print(f"\n{'='*70}")
    print(f"[Ablation] {len(records)} variants, {len(ds)} windows each")
    for r in records:
        print(f" {r['variant']:<26} AUPRC={r['auprc_max']:.4f} "
              f"AUROC={r['auroc_max']:.4f} ΔAUPRC={r['delta_auprc']:+.4f} "
              f"p={r['delong_p_vs_full']:.1e}")
    print(f" total={(time.time()-t_start)/60:.1f}min")
    print(f"Saved: {out_dir}/{tag}.json + .npz + ABLATION_M0.md")


if __name__ == "__main__":
    main()
