#!/usr/bin/env python3
"""Interpretability analysis: FeatureSplit gating / donor-recipient stream probing /
breakpoint-heatmap localization of the contig-mode model on real GIs (IslandViewer,
GC-matched pos/neg).

Pure inference (no gradients, no model changes); reuses the safe shared-GPU loading
of eval_external_benbow.py (load_state_dict(state["model"]) loads weights only,
CUDA_VISIBLE_DEVICES=1=A100, batch_size=1).
HGTOutput already directly exposes split_gates / donor_pool / recipient_pool / bp_heatmaps.

Three-part analysis (corresponding to THESIS_CH4_DRAFT.md §4.2):
  (A) FeatureSplit gating: gating-statistics differences between pos (real GIs) and
      neg (GC-matched flanks) + correlation of gating structure with classification
      confidence. Tests whether the decoupling responds to real insertions.
  (B) donor/recipient stream probing: per-stream RF probes predicting class / GC  —  —
      validating whether the two streams carry complementary information (rather than
      redundancy), and whether composition (GC) is concentrated in one ablatable
      stream (echoing ΔAUC=−0.001).
  (C) breakpoint heatmap localization: whether pos has sharp peaks and neg is flat  —  —
      testing whether the breakpoint head produces meaningful localization on real GIs.

usage:
  CUDA_VISIBLE_DEVICES=1 python \
      scripts/interpret_contig_real_gi.py \
      --ckpt checkpoints/Contig_500k/best_model.pt \
      --config configs/contig_500k.yaml \
      --benchmark audit_output/real_contig_bench/islandviewer_contigs.fasta \
      --out audit_output/12_interpret_contig_real_gi.json
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from collections import defaultdict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # A100 (reversemapping)

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.stats import mannwhitneyu, spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from DeepFluHGT import load_config
from model.contracts import HGTConfig
from model.api import build_model
from data_pipeline.impl.dataset import SlidingWindowContigDataset
from data_pipeline.impl.collator import HGTCollator
from data_pipeline.contracts import HGTSample


def read_fasta_labeled(fasta_path: str) -> dict:
    """Read the IslandViewer real-contig fasta → {seq_id: (label, sequence)}.
    Header: >id|label|id type  (label=1 real_gi, label=0 real_flank_gc_matched)
    """
    samples, cur_id, cur_label, buf = {}, None, None, []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if cur_id is not None:
                    samples[cur_id] = (cur_label, "".join(buf))
                parts = line[1:].split("|")
                cur_id = parts[2].split()[0] if len(parts) > 2 and parts[2].strip() else parts[0]
                cur_label = int(parts[1]) if len(parts) > 1 else 0
                buf = []
            else:
                buf.append(line.upper())
        if cur_id is not None:
            samples[cur_id] = (cur_label, "".join(buf))
    return samples


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation, fault-tolerant (constant array → nan)."""
    if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(spearmanr(a, b)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_windows", type=int, default=1500,
                    help="maximum number of sampled windows per group (controls GPU time)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    mc, dc = cfg["model"], cfg["data"]
    sw = dc["sw_window_size"]

    # ── load model (safe: loads state["model"] weights only) ──
    model = build_model(HGTConfig(
        data_mode="contig", max_contig_length=sw,
        hidden_dim=mc["hidden_dim"], local_hidden_dim=mc["local_hidden_dim"],
        num_attention_heads=mc["num_attention_heads"], dropout=mc["dropout"],
        graph_top_k=mc["graph_top_k"], bio_feature_dim=mc["bio_feature_dim"],
        pretrained_nt_path=mc["pretrained_nt_path"],
        pretrained_caduceus_path=mc["pretrained_caduceus_path"])).to("cuda")
    state = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    print(f"[Model] loaded ckpt epoch={state.get('epoch')}", flush=True)

    # ── data ──
    samples_raw = read_fasta_labeled(args.benchmark)
    hgt_samples = []
    for sid, (label, seq) in samples_raw.items():
        if len(seq) < 50:
            continue
        hgt_samples.append(HGTSample(
            seq_id=sid, label=label, sequence=seq, r2_sequence=None,
            events=[], sample_type="POS" if label == 1 else "NEG", seq_len=len(seq)))
    ds = SlidingWindowContigDataset(
        samples=hgt_samples, window_size=sw, overlap=dc["sw_overlap"],
        is_training=False, max_neg_windows=9999,
        max_eval_windows=dc["sw_max_eval_windows"], extract_bio=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=HGTCollator(pad_id=0), num_workers=2)
    print(f"[Data] {len(hgt_samples)} contigs → {len(ds)} windows", flush=True)

    # ── inference: collect per-window gating/streams/heatmap/confidence/GC/labels ──
    rec = defaultdict(list)  # cls_prob, gate_mean, gate_std, gate_frac_high,
    #                        bp_max, bp_prominence, donor_pool[], recipient_pool[], gc, label
    n_done = 0
    rng = np.random.RandomState(42)
    # pre-sample a window-index subset to control runtime (balanced across groups)
    pos_seen, neg_seen = 0, 0
    with torch.no_grad():
        for batch in loader:
            sids = getattr(batch, "original_seq_ids", None) or batch.seq_ids
            # use the original contig label to assign the group (sliding windows inherit the contig label)
            label = samples_raw.get(sids[0], (0, ""))[0]
            if label == 1 and pos_seen >= args.max_windows:
                continue
            if label == 0 and neg_seen >= args.max_windows:
                continue
            batch = batch.to("cuda")
            out = model(batch, early_exit=False)

            mask = batch.attention_mask.float()  # [1, L]
            valid = mask.squeeze(0) > 0  # [L] bool
            L_valid = int(valid.sum().item())

            cls_prob = float(torch.sigmoid(out.cls_logits).item())
            gates = out.split_gates.squeeze(0)  # [L]
            gates_v = gates[valid].cpu().numpy() if L_valid > 0 else np.array([float("nan")])
            bp = out.bp_heatmaps.squeeze(0)  # [L] sigmoid probs
            bp_v = bp[valid].cpu().numpy() if L_valid > 0 else np.array([float("nan")])

            # GC (bio_features[:,0])
            gc = float(batch.bio_features[0, 0].cpu().item()) if batch.bio_features is not None else float("nan")

            rec["cls_prob"].append(cls_prob)
            rec["label"].append(label)
            rec["gc"].append(gc)
            rec["gate_mean"].append(float(np.mean(gates_v)))
            rec["gate_std"].append(float(np.std(gates_v)))
            rec["gate_frac_high"].append(float(np.mean(gates_v > 0.5)))
            rec["bp_max"].append(float(np.max(bp_v)))
            rec["bp_prominence"].append(float(np.max(bp_v) - np.mean(bp_v)))
            rec["donor_pool"].append(out.donor_pool.squeeze(0).cpu().numpy())
            rec["recipient_pool"].append(out.recipient_pool.squeeze(0).cpu().numpy())

            if label == 1:
                pos_seen += 1
            else:
                neg_seen += 1
            n_done += 1
            if n_done % 100 == 0:
                print(f"  [infer] {n_done} windows (pos={pos_seen}, neg={neg_seen})", flush=True)

    print(f"[Infer] done: {n_done} windows (pos={pos_seen}, neg={neg_seen})", flush=True)

    # ── organize ──
    labels = np.array(rec["label"])
    cls = np.array(rec["cls_prob"])
    gc = np.array(rec["gc"])
    pos_m = labels == 1
    neg_m = labels == 0
    donor = np.stack(rec["donor_pool"])       # [N, D]
    recipient = np.stack(rec["recipient_pool"])  # [N, D]

    result = {
        "meta": {
            "ckpt": args.ckpt, "n_windows": int(n_done),
            "n_pos": int(pos_seen), "n_neg": int(neg_seen),
            "model_epoch": state.get("epoch"),
        },
    }

    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def group_stats(name: str, arr: np.ndarray) -> dict:
        p, n = arr[pos_m], arr[neg_m]
        d = {
            "pos_mean": float(np.mean(p)), "pos_std": float(np.std(p)),
            "neg_mean": float(np.mean(n)), "neg_std": float(np.std(n)),
            "pos_minus_neg": float(np.mean(p) - np.mean(n)),
        }
        # Mann-Whitney U (significance of between-group differences)
        if len(p) > 3 and len(n) > 3 and not np.isnan(arr).any():
            try:
                u, pval = mannwhitneyu(p, n, alternative="two-sided")
                d["mwu_pvalue"] = float(pval)
            except Exception:
                pass
        d["corr_with_clsprob"] = safe_corr(arr, cls)
        return d

    # ── (A) FeatureSplit gating ──
    result["A_split_gates"] = {
        "gate_mean": group_stats("gate_mean", np.array(rec["gate_mean"])),
        "gate_std": group_stats("gate_std", np.array(rec["gate_std"])),
        "gate_frac_high": group_stats("gate_frac_high", np.array(rec["gate_frac_high"])),
        "note": "gate_mean=mean donor allocation; gate_std=spatial structure of the gating "
                "(higher = less uniform); gate_frac_high=fraction of positions assigned to the donor. "
                "Tests the gating difference between real GIs and GC-matched flanks.",
    }

    # ── (B) donor/recipient stream probing ──
    def probe(name: str, X: np.ndarray) -> dict:
        d = {}
        # per-class (pos vs neg) AUC
        if pos_m.sum() > 10 and neg_m.sum() > 10:
            rf_c = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=4)
            d["group_auc"] = float(cross_val_score(
                rf_c, X, labels, cv=cv5, scoring="roc_auc").mean())
        # GC R² (whether composition is encoded in this stream)
        if not np.isnan(gc).any() and len(gc) > 20:
            rf_r = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=4)
            d["gc_r2"] = float(cross_val_score(rf_r, X, gc, cv=5, scoring="r2").mean())
        return d

    result["B_stream_probing"] = {
        "donor_pool": probe("donor", donor),
        "recipient_pool": probe("recipient", recipient),
        "note": "group_auc=this stream's standalone ability to distinguish real GIs from flanks; "
                "gc_r2=the degree to which this stream encodes GC. If decoupling is effective: "
                "the two streams complement each other; composition (GC) should be concentrated "
                "in one ablatable stream (echoing ΔAUC=−0.001).",
    }

    # ── (C) breakpoint heatmap localization ──
    result["C_breakpoint_heatmap"] = {
        "bp_max": group_stats("bp_max", np.array(rec["bp_max"])),
        "bp_prominence": group_stats("bp_prominence", np.array(rec["bp_prominence"])),
        "note": "bp_max=heatmap peak value; bp_prominence=how much the peak stands above the mean. "
                "Expected: pos (real GIs) has sharper peaks (localization), neg (flanks) is flat.",
    }

    # ── print ──
    print("\n" + "=" * 60)
    print("Interpretability analysis results (contig-mode model, real IslandViewer GIs)")
    print("=" * 60)
    print(f"\n[Meta] n_windows={n_done} (pos={pos_seen}, neg={neg_seen})")

    def gs(d):
        return (f"pos={d['pos_mean']:.3f}±{d['pos_std']:.3f} "
                f"neg={d['neg_mean']:.3f}±{d['neg_std']:.3f} "
                f"Δ={d['pos_minus_neg']:+.3f} p={d.get('mwu_pvalue', float('nan')):.2e} "
                f"ρ(cls)={d['corr_with_clsprob']:.3f}")

    print("\n(A) FeatureSplit gating")
    for k in ["gate_mean", "gate_std", "gate_frac_high"]:
        print(f"  {k}: {gs(result['A_split_gates'][k])}")

    print("\n(B) donor/recipient stream probing")
    for s in ["donor_pool", "recipient_pool"]:
        b = result["B_stream_probing"][s]
        print(f"  {s}: group_auc={b.get('group_auc', float('nan')):.3f} "
              f"gc_r2={b.get('gc_r2', float('nan')):.3f}")

    print("\n(C) breakpoint heatmap localization")
    for k in ["bp_max", "bp_prominence"]:
        print(f"  {k}: {gs(result['C_breakpoint_heatmap'][k])}")

    # save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
