#!/usr/bin/env python3
# scripts/difficulty_eval.py
"""DeepFluHGT  —  B2 difficulty ladder D1-D3 M0 inference evaluation (4090) + D0-D3 curve report.

Task A (V7.5 evidence module #5, 2026-08-26):
  inference: M0 = checkpoints/Contig_500k_ntv3only/best_model.pt (best=ep2,
        use_evo=False → inference does not depend on the evo cache); for each
        difficulty_ladder D1/D2/D3 test/ tier, run windowing inference + contig aggregation + DeLong.
        windowing/aggregation/metric reused verbatim from scripts/eval_real_anchors.py (import that module);
        GPU model loading / batch inference replicates scripts/eval_real_anchors_gpu.py
        (protocol: window=3000, overlap=300, win_cap=10, batch=8, fp32,
         deterministic sliding-window override + max_eval_windows cap, same as the real-anchor evaluation).
  report: --report aggregates D0 (=Contig_500k/test same-protocol pooled reference, max aggregation
        AUPRC 0.8142 / AUROC 0.8202, from docs/evidence/OOD_CROSSTAXA_M0.md
        Task B pooled; after the DA-H3 caliber fix, no longer using the training-side mixed aggregation 0.827) + D1-3 →
        docs/evidence/DIFFICULTY_CURVE_M0.md.

GPU discipline: CUDA_VISIBLE_DEVICES=4090 UUID GPU lock (A100 strictly forbidden); the script asserts that exactly
  one device, the 4090, is visible; only this single GPU process at a time (guaranteed by the serial chain script).

usage (launcher: scripts/run/difficulty_curve_chain.sh):
  CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python scripts/difficulty_eval.py --tier D1
  python scripts/difficulty_eval.py --report          # CPU, seconds-level
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

# Ladder tier definitions (configs/difficulty_ladder/*.yaml; generator: scripts/run/gen_difficulty_variants.sh)
TIERS = {
    "D1": ("D1_gi_length", "realistic GI length distribution (island span / virus payload shifted upward)"),
    "D2": ("D2_donor_sim", "donor similarity constraint (evolutionary distance <=2% for 85%, GC adaptation disabled)"),
    "D3": ("D3_fragmentation", "assembly fragmentation (artifact rate 0.40, contig cap 12kb)"),
}
# D0 reference (no re-inference): pooled inference under the same protocol (max aggregation
# + events=[] windowing + win_cap=10/b=8/fp32), docs/evidence/OOD_CROSSTAXA_M0.md section 1 "pooled reference (50k under this protocol)"
#  —  —  DA-H3 caliber fix (2026-08-31): the first version used the training-side mixed aggregation 0.827 as D0,
# which does not match the caliber of the fresh max-aggregation inference for D1-D3; now switched to the same-protocol reference 0.8142/0.8202.
# (the original caliber 0.8269/0.8272 = mixed aggregation + early_exit=True, see REAL_ANCHORS_M0final.md)
D0_REF = {"tier": "D0", "name": "Contig_500k/test (same-protocol pooled reference)",
          "auprc": 0.8142063460190857, "auroc": 0.8202116838735494,
          "auprc_mean": 0.7621671833186684, "n_contigs": 50000,
          "n_windows": 199707,
          "source": "docs/evidence/OOD_CROSSTAXA_M0.md (Task B pooled, "
                    "same checkpoint / same pool / same windowing-aggregation implementation)"}


def parse_args():
    ap = argparse.ArgumentParser(description="Difficulty ladder M0 evaluation (Task A)")
    ap.add_argument("--tier", choices=list(TIERS.keys()),
                    help="D1|D2|D3  —  run inference on that tier's test/")
    ap.add_argument("--report", action="store_true",
                    help="Aggregate D0-D3 → docs/evidence/DIFFICULTY_CURVE_M0.md (CPU)")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--win_cap", type=int, default=10,
                    help="Max eval windows per contig (full protocol = 10)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="smoke: take the first N items per class")
    ap.add_argument("--gpu_uuid", default=GPU_4090_UUID)
    ap.add_argument("--tag", default="difficulty_m0")
    ap.add_argument("--out_dir", default=str(ROOT / "docs/evidence"))
    return ap.parse_args()


ARGS = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = ARGS.gpu_uuid

import eval_real_anchors as cpu  # noqa: E402  (windowing/aggregation/metric reused verbatim)
os.environ["CUDA_VISIBLE_DEVICES"] = ARGS.gpu_uuid  # restore immediately (the cpu module blanked it at import)

from delong_test import delong_test_full  # noqa: E402

import torch  # noqa: E402


def assert_4090():
    n = torch.cuda.device_count()
    assert n == 1, f"exactly 1 visible GPU expected (4090 GPU lock), actual {n}"
    name = torch.cuda.get_device_name(0)
    assert "4090" in name, f"visible GPU is not RTX 4090: {name}"
    free, total = torch.cuda.mem_get_info(0)
    print(f"[GPU] {name} | mem free {free/1e9:.1f}/{total/1e9:.1f} GB | "
          f"CUDA {torch.version.cuda} | torch {torch.__version__}")
    return name


# ═══════════════════════════════════════════════════════════════════════════ #
#  Model construction (verbatim replication of eval_real_anchors_gpu.load_model_gpu  —  protocol-consistent)
# ═══════════════════════════════════════════════════════════════════════════ #

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
    if ckpt_epoch != 2:  # same rule as the real-anchor evaluation: best_model must be ep2
        alt = ckpt.parent / "epoch_2.pt"
        print(f"[WARN] {ckpt.name} epoch={ckpt_epoch} ≠ 2 → fallback {alt.name}")
        st = torch.load(alt, map_location="cpu", weights_only=False)
        ckpt_epoch = st.get("epoch")
        contract = st.get("contract", {})
    missing, unexpected = model.load_state_dict(st["model"], strict=False)
    model.to(device).eval()

    contract_info = {
        "ckpt": str(ckpt),
        "ckpt_epoch": ckpt_epoch,
        "encoder_frozen": contract.get("encoder_frozen"),
        "backbone_sha256": contract.get("backbone_sha256"),
        "use_evo": bool(mc.get("use_evo", False)),
        "state_dict_missing": len(missing),
        "state_dict_unexpected": len(unexpected),
    }
    assert contract.get("encoder_frozen") is True, "contract: encoder_frozen must be True"
    print(f"[Model] epoch={ckpt_epoch} use_evo={contract_info['use_evo']} "
          f"load: missing={len(missing)} unexpected={len(unexpected)} → {device}")
    return model, contract_info, device


# ═══════════════════════════════════════════════════════════════════════════ #
#  windowinference (verbatim replication eval_real_anchors_gpu.predict_windows_gpu)
# ═══════════════════════════════════════════════════════════════════════════ #

def predict_windows_gpu(model, samples: dict, window: int, overlap: int,
                        win_cap: int, batch_size: int, device):
    from torch.utils.data import DataLoader
    from data_pipeline.impl.dataset import SlidingWindowContigDataset
    from data_pipeline.impl.collator import HGTCollator
    from data_pipeline.contracts import HGTSample

    hgt_samples, labels_map = [], {}
    for sid, (label, seq) in samples.items():
        hgt_samples.append(HGTSample(
            seq_id=sid, label=label, sequence=seq, r2_sequence=None,
            events=[], sample_type="POS" if label == 1 else "NEG",
            seq_len=len(seq)))
        labels_map[sid] = label

    ds = SlidingWindowContigDataset(
        samples=hgt_samples, window_size=window, overlap=overlap,
        is_training=False, max_neg_windows=9999, max_eval_windows=win_cap,
        extract_bio=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=HGTCollator(pad_id=0), num_workers=4)
    print(f"[Data] {len(hgt_samples)} contigs, {len(ds)} windows "
          f"(win_cap={win_cap}, batch={batch_size})")

    contig_windows = defaultdict(list)
    t0 = time.time()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            output = model(batch, early_exit=False)
            probs = torch.sigmoid(output.cls_logits.squeeze(-1)).float().cpu().numpy()
            sids = batch.original_seq_ids or batch.seq_ids
            for sid, p in zip(sids, probs):
                contig_windows[sid].append(float(p))
            if (i + 1) % 50 == 0:
                el = time.time() - t0
                done = (i + 1) * batch_size
                print(f"  [{min(done,len(ds))}/{len(ds)}] {el:.0f}s "
                      f"({el/(i+1):.2f}s/batch, "
                      f"ETA {(len(loader)-i-1)*el/(i+1)/60:.1f}min)", flush=True)
    infer_s = time.time() - t0
    print(f"[Predict] {len(ds)} windows in {infer_s:.0f}s "
          f"({infer_s/max(1,len(ds)):.3f}s/win)")
    return contig_windows, labels_map, len(ds), infer_s


def delong_single(y: np.ndarray, scores: np.ndarray) -> dict:
    """Single-predictor DeLong (var_a → SE/CI/z-vs-0.5); same style as the real-anchor evaluation."""
    r = delong_test_full(y, scores, scores)
    se = math.sqrt(r["var_a"])
    auc = r["auc_a"]
    z = (auc - 0.5) / se if se > 0 else math.inf
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return {"auc": auc, "se": se,
            "ci95": [auc - 1.959963984540054 * se,
                     auc + 1.959963984540054 * se],
            "z_vs_half": z, "p_vs_half": p, "n": int(len(y)),
            "n_pos": int(np.sum(y == 1)), "var": r["var_a"]}


# ═══════════════════════════════════════════════════════════════════════════ #
#  Tier data reading: dataset.fasta + provenance.tsv (seq_id → label)
# ═══════════════════════════════════════════════════════════════════════════ #

def read_labels_from_provenance(prov_path: Path) -> dict:
    labels = {}
    with open(prov_path) as f:
        header = f.readline().rstrip("\n").split("\t")
        i_sid, i_lab = header.index("seq_id"), header.index("label")
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) <= max(i_sid, i_lab):
                continue
            labels[p[i_sid]] = int(p[i_lab])
    return labels


def read_tier_samples(test_dir: Path, limit: int = 0) -> dict:
    """{seq_id: (label, sequence)}  —  keep only records tagged in provenance."""
    labels = read_labels_from_provenance(test_dir / "provenance.tsv")
    samples, header, buf = {}, None, []
    fasta = test_dir / "dataset.fasta"
    with open(fasta) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None and header in labels:
                    samples[header] = (labels[header], "".join(buf))
                header = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.upper())
    if header is not None and header in labels:
        samples[header] = (labels[header], "".join(buf))

    n_pos = sum(1 for v in samples.values() if v[0] == 1)
    print(f"[Tier] {test_dir}: {len(samples)} contigs "
          f"(pos={n_pos}, neg={len(samples)-n_pos})")
    assert n_pos > 0 and len(samples) - n_pos > 0, "both positive and negative samples must be present"

    if limit > 0:
        pos = {k: v for k, v in samples.items() if v[0] == 1}
        neg = {k: v for k, v in samples.items() if v[0] == 0}
        samples = dict(list(pos.items())[:limit])
        samples.update(dict(list(neg.items())[:limit]))
        print(f"[Smoke] limited to {len(samples)} contigs")
    return samples


# ═══════════════════════════════════════════════════════════════════════════ #
#  Main pipeline: single-tier inference
# ═══════════════════════════════════════════════════════════════════════════ #

def run_tier(tier: str, args):
    gpu_name = assert_4090()
    name, axis = TIERS[tier]
    test_dir = ROOT / f"simulated_data/difficulty_ladder/{name}/test"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"difficulty_{tier}_{args.tag}"
    t_start = time.time()

    samples = read_tier_samples(test_dir, args.limit)
    model, contract_info, device = load_model_gpu(Path(args.ckpt), Path(args.config))

    import yaml as _yaml
    _dc = _yaml.safe_load(open(args.config))["data"]
    contig_windows, labels_map, n_windows, infer_s = predict_windows_gpu(
        model, samples, window=_dc["sw_window_size"], overlap=_dc["sw_overlap"],
        win_cap=args.win_cap, batch_size=args.batch, device=device)
    del model
    torch.cuda.empty_cache()

    results = cpu.aggregate_and_eval(contig_windows, labels_map)
    wstats = cpu.window_stats(contig_windows, labels_map)

    sid_list = sorted(contig_windows.keys())
    y_sorted = np.array([labels_map[s] for s in sid_list], dtype=np.int64)

    def _agg(kind, w):
        if kind == "max":
            return max(w)
        if kind == "mean":
            return float(np.mean(w))
        if kind == "median":
            return float(np.median(w))
        return float(np.mean(sorted(w, reverse=True)[:3]))

    agg_scores = {k: np.array([_agg(k, contig_windows[s]) for s in sid_list],
                              dtype=np.float32)
                  for k in ["max", "mean", "median", "top3_mean"]}
    delong = {k: delong_single(y_sorted, v) for k, v in agg_scores.items()}

    np.savez(out_dir / f"{tag}.npz",
             contig_ids=np.array(sid_list, dtype=object),
             labels=y_sorted,
             n_windows=np.array([len(contig_windows[s]) for s in sid_list],
                                dtype=np.int64),
             window_probs=np.array(
                 [np.asarray(contig_windows[s], dtype=np.float32) for s in sid_list],
                 dtype=object),
             **{f"agg_{k}": v for k, v in agg_scores.items()})

    payload = {
        "tier": tier, "tier_dir": name, "difficulty_axis": axis,
        "protocol": {
            "source": "Task A: reuse of eval_real_anchors windowing/aggregation (GPU variant protocol)",
            "window_size": _dc["sw_window_size"], "overlap": _dc["sw_overlap"],
            "batch_size": args.batch, "max_eval_windows": args.win_cap,
            "aggregations": ["max", "mean", "median", "top3_mean"],
            "device": "cuda", "gpu": gpu_name, "dtype": "fp32",
            "note": "events=[] (same as the real-anchor protocol): eval windows deterministically "
                    "override the sliding window; under the win_cap cap, POS windows are not prioritized "
                    "(same metric caliber as D0/B/C)",
        },
        "contract": contract_info,
        "window_stats": wstats,
        "delong": delong,
        "infer_seconds": infer_s,
        "total_seconds": time.time() - t_start,
        "results": results,
    }
    with open(out_dir / f"{tag}.json", "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False,
                  default=lambda o: float(o) if isinstance(o, np.floating) else o)

    print(f"\n{'='*70}")
    print(f"[{tier}] {axis}")
    print(f"  n={wstats['n_contigs']} windows={wstats['n_windows']}")
    for name2 in ["max", "mean", "median", "top3_mean"]:
        m, d = results[name2], delong[name2]
        print(f"  [{name2:9s}] AUC={m.get('auc', float('nan')):.4f} "
              f"[{d['ci95'][0]:.4f},{d['ci95'][1]:.4f}] | "
              f"AUPRC={m.get('auprc', float('nan')):.4f} "
              f"F1@opt={m.get('f1_at_optimal', float('nan')):.4f}")
    print(f"  infer={infer_s/60:.1f}min total={(time.time()-t_start)/60:.1f}min")
    print(f"Saved: {out_dir}/{tag}.json + .npz")


# ═══════════════════════════════════════════════════════════════════════════ #
#  Report: D0-D3 difficulty-AUPRC curve
# ═══════════════════════════════════════════════════════════════════════════ #

def island_span_stats(test_dir: Path) -> dict:
    """Generation acceptance statistics (island span distribution; reuses the gen_difficulty_variants.sh stats logic)."""
    import numpy as np
    meta_path = test_dir / "metadata.json"
    if not meta_path.is_file():
        return {}
    try:
        meta = json.load(open(meta_path))
        pos = meta["positive"] if isinstance(meta, dict) else \
            [m for m in meta if m.get("label") == 1]
        spans = []
        for m in pos:
            ev = m.get("events", [])
            if ev:
                spans.append(max(e["hgt_end"] for e in ev)
                             - min(e["hgt_start"] for e in ev))
        a = np.array(spans)
        return {"n_pos": len(pos),
                "span_min": int(a.min()), "span_p50": float(np.median(a)),
                "span_p90": float(np.percentile(a, 90)), "span_max": int(a.max())}
    except Exception as e:  # metadata format differences → do not block the report
        return {"error": str(e)}


def write_report(args):
    out_dir = Path(args.out_dir)
    rows = []
    for tier in ["D1", "D2", "D3"]:
        name, axis = TIERS[tier]
        jPath = out_dir / f"difficulty_{tier}_{args.tag}.json"
        if not jPath.is_file():
            rows.append({"tier": tier, "axis": axis, "missing": True})
            continue
        j = json.load(open(jPath))
        rows.append({
            "tier": tier, "axis": axis, "missing": False,
            "n_contigs": j["window_stats"]["n_contigs"],
            "n_windows": j["window_stats"]["n_windows"],
            "auprc_max": j["results"]["max"].get("auprc"),
            "auroc_max": j["results"]["max"].get("auc"),
            "auprc_mean": j["results"]["mean"].get("auprc"),
            "auprc_top3": j["results"]["top3_mean"].get("auprc"),
            "f1_max": j["results"]["max"].get("f1_at_optimal"),
            "delong_max": j["delong"]["max"],
            "span": island_span_stats(
                ROOT / f"simulated_data/difficulty_ladder/{name}/test"),
            "infer_seconds": j.get("infer_seconds"),
        })

    md = []
    md.append("# Difficulty ladder D0-D3  —  M0 (NTv3-only, ep2) difficulty-AUPRC curve\n")
    md.append("> Generated: %s  ·  Task A (V7.5 evidence module #5)  ·  "
              "`scripts/difficulty_eval.py --report`\n"
              % time.strftime("%Y-%m-%d %H:%M:%S"))
    md.append("> model: `checkpoints/Contig_500k_ntv3only/best_model.pt` (ep2, "
              "encoder frozen, **use_evo=False** → inference does not depend on the evo cache; "
              "the D-tier evo2 embeddings are written into the chain by a separate task, prepared solely for M2 reuse).\n")
    md.append("> window protocol: 3000bp / 300bp overlap / win_cap=10 / batch=8 / fp32; "
              "contig aggregation max · mean · median · top3_mean; DeLong single-predictor SE "
              "(`delong_test`), same style as the real-anchor evaluation (`eval_real_anchors_gpu`).\n")

    md.append("\n## 0. Caliber fix (DEEP_AUDIT_FINAL DA-H3)\n")
    md.append("**Problem**: the first version's D0 row \"0.827 (reused)\" was taken from the **mixed-aggregation** "
              "training-side metric in epoch_metrics, while D1-D3 use fresh max-aggregation inference (events=[], win_cap=10, "
              "b=8, fp32, early_exit=False)  —  —  an aggregation-caliber mismatch that mixed "
              "~0.013 of caliber difference into ΔAUPRC.\n")
    md.append("**Fix**: D0 now uses the same-protocol pooled reference  —  —  `OOD_CROSSTAXA_M0.md` Task B "
              "(same checkpoint M0 ep2, same test pool of 50,000 contigs, same windowing/aggregation "
              f"implementation, {D0_REF['n_windows']:,} windows): **AUPRC(max) = "
              f"{D0_REF['auprc']:.4f} / AUROC(max) = {D0_REF['auroc']:.4f}**.\n")
    md.append("**Effect**: the ΔAUPRC sign for D2 flips (−0.0043 → +0.0085); the qualitative conclusion is strengthened "
              "(under a consistent caliber all three tiers are ≥ D0, no performance cliff). The original 0.827 mixed-aggregation caliber is kept "
              "as a reference note (REAL_ANCHORS_M0final.md, including the early_exit protocol difference).\n")

    md.append("\n## 1. Difficulty-AUPRC curve (primary metric: max aggregation; D0 same-protocol reference)\n")
    md.append("| Tier | Difficulty axis | n_contigs | n_windows | AUPRC | AUROC | "
              "ΔAUPRC vs D0 | DeLong 95% CI (AUROC) |\n")
    md.append("|---|---|---|---|---|---|---|---|\n")
    d0 = D0_REF
    md.append(f"| **D0** | {d0['name']} (seed=42 split) | {d0['n_contigs']:,} | "
              f"{d0['n_windows']:,} | **{d0['auprc']:.4f}** | "
              f"{d0['auroc']:.4f} | 0 |  —  (pooled reference) |\n")
    for r in rows:
        if r.get("missing"):
            md.append(f"| {r['tier']} | {r['axis']} | _MISSING_ | | | | | |\n")
            continue
        dl = r["delong_max"]
        md.append(f"| **{r['tier']}** | {r['axis']} | {r['n_contigs']:,} | "
                  f"{r['n_windows']:,} | **{r['auprc_max']:.4f}** | "
                  f"{r['auroc_max']:.4f} | {r['auprc_max']-d0['auprc']:+.4f} | "
                  f"[{dl['ci95'][0]:.4f}, {dl['ci95'][1]:.4f}] |\n")
    md.append("\n> D0 source: `OOD_CROSSTAXA_M0.md` section 1 pooled reference  —  —  same checkpoint / same pool / "
              "same windowing-aggregation implementation; the only difference is the evaluation script (`crosstaxa_eval.py` vs "
              "`difficulty_eval.py`), both sharing the same `eval_real_anchors` windowing code path.\n")
    md.append("> **D3 note (a mechanism component unrelated to difficulty)**: D3 fragmentation → shorter contigs "
              "(1.80 windows/contig vs D1 4.21 / D2 3.98) → max aggregation suppresses the noise maximum "
              "of negative contigs, while a single window of a positive contig already contains the breakpoint → the window-count/aggregation mechanical variation contributes part of "
              "the +0.055; when citing the D3 number, report the window count alongside it.\n")
    md.append("> Residual caliber difference (declared): the D0 pool has 50k contigs (pos 25,500) vs 10k each for D1-D3 "
              "(pos 5,000)  —  —  pool sizes are fixed by the ladder generation design (each tier moves only one difficulty axis) "
              "and are outside the scope of this fix.\n")

    md.append("\n## 2. All aggregation calibers per tier\n")
    md.append("| Tier | AUPRC(max) | AUPRC(mean) | AUPRC(top3_mean) | "
              "F1@opt(max) | AUROC(max) | z vs 0.5 (max) |\n")
    md.append("|---|---|---|---|---|---|---|\n")
    md.append(f"| D0 | {d0['auprc']:.4f} (same-protocol reference) | {d0['auprc_mean']:.4f} | "
              f" —  |  —  | {d0['auroc']:.4f} |  —  |\n")
    for r in rows:
        if r.get("missing"):
            md.append(f"| {r['tier']} | _MISSING_ | | | | | |\n")
            continue
        dl = r["delong_max"]
        md.append(f"| {r['tier']} | {r['auprc_max']:.4f} | {r['auprc_mean']:.4f} | "
                  f"{r['auprc_top3']:.4f} | {r['f1_max']:.4f} | "
                  f"{r['auroc_max']:.4f} | {dl['z_vs_half']:+.2f} |\n")

    md.append("\n## 3. Generation acceptance check (island span, aligned to the real IV4 GI distribution)\n")
    md.append("| Tier | n_pos | span min | p50 | p90 | max | IV4 reference |\n")
    md.append("|---|---|---|---|---|---|---|\n")
    for r in rows:
        if r.get("missing") or not r.get("span"):
            md.append(f"| {r['tier']} |  —  | | | | | p50=8,019 / p90=28,155 |\n")
            continue
        s = r["span"]
        md.append(f"| {r['tier']} | {s.get('n_pos',' — ')} | {s.get('span_min',' — ')} | "
                  f"{s.get('span_p50',0):.0f} | {s.get('span_p90',0):.0f} | "
                  f"{s.get('span_max',' — ')} | p50=8,019 / p90=28,155 |\n")

    md.append("\n## 4. Comparability design notes\n")
    md.append("- Common to all three tiers: seed=42 + genome_isolation ratios identical to D0 → "
              "the test genome pool is fully consistent with Contig_500k "
              "(microbe 1155 / virus 8 strains); each tier moves only one difficulty axis.\n"
              "- Each tier has n_pos=n_neg=5000, train_ratio=val_ratio=0 (everything lands in test/); "
              "generator `scripts/run/gen_difficulty_variants.sh` (idempotent).\n"
              "- M0 inference uses use_evo=False: the D-tier evo2 embeddings (block28) take no part in these results; "
              "the tail task of the chain writes `evo_cache/D{1,2,3}/test.pt` for M2 reuse.\n")
    md.append("\n## 5. artifact\n")
    for tier in ["D1", "D2", "D3"]:
        md.append(f"- `{out_dir.name}/difficulty_{tier}_{args.tag}.json` + `.npz`"
                  " (per-contig window probability archive, verifiable/re-plottable)\n")
    md.append("- **D0 same-protocol reference**: `evidence/ood_crosstaxa_m0.json` + `.npz` "
              f"(`pooled.auprc_max={D0_REF['auprc']:.7f}` / "
              f"`pooled.auroc_max={D0_REF['auroc']:.7f}`, "
              f"{D0_REF['n_contigs']:,} contig / {D0_REF['n_windows']:,} window)\n")

    out = out_dir / "DIFFICULTY_CURVE_M0.md"
    with open(out, "w") as f:
        f.write("".join(md))
    print(f"[Report] saved {out}")
    for r in rows:
        if not r.get("missing"):
            print(f"  {r['tier']}: AUPRC(max)={r['auprc_max']:.4f} "
                  f"AUPRC(mean)={r['auprc_mean']:.4f}")


def main():
    if ARGS.report:
        write_report(ARGS)
        return
    if not ARGS.tier:
        raise SystemExit("--tier {D1,D2,D3} or --report is required")
    run_tier(ARGS.tier, ARGS)


if __name__ == "__main__":
    main()
