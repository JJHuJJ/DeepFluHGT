#!/usr/bin/env python3
"""External benchmark evaluation: evaluate DeepFluHGT on real GI data (Benbow, 566 items).

Benbow provides experimentally validated Genomic Island positive/negative samples — the external evidence reviewers trust most.

Workflow:
1. Read the Benbow FASTA (headers include |label|)
2. Extract sequences + labels
3. Predict with the full model (contig sliding windows)
4. Aggregate window predictions → contig level
5. Compute AUC / AUPRC / F1 / stratified stats / ECE

usage:
  python scripts/eval_external_benbow.py \
      --ckpt checkpoints/Contig_500k/best_model.pt \
      --config configs/contig_500k.yaml \
      --benchmark benchmark/literature_gi/repo/test_data/benbow_test_data.fasta \
      --out audit_output/benbow_results.json
"""
import os, sys, json, argparse, subprocess, tempfile, re
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def read_benbow(fasta_path: str) -> dict:
    """Read the Benbow FASTA → {seq_id: (label, sequence)}.

    Header format: >accession_GI_n|label|accession:start-end organism
    label=1 → positive (GI), label=0 → negative (non-GI)
    """
    samples = {}
    cur_id, cur_label, buf = None, None, []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if cur_id:
                    samples[cur_id] = (cur_label, "".join(buf))
                parts = line[1:].split("|")
                # Unique key = coordinate (accession:start-end); accession alone collides
                # across multiple regions of the same genome (566 seqs → 20 accessions).
                cur_id = parts[2].split()[0] if len(parts) > 2 and parts[2].strip() else parts[0]
                cur_label = int(parts[1]) if len(parts) > 1 else 0
                buf = []
            else:
                buf.append(line.upper())
        if cur_id:
            samples[cur_id] = (cur_label, "".join(buf))
    return samples


def extract_and_predict(samples: dict, ckpt: str, config: str, out_dir: str):
    """Extract FASTA → DeepFluHGT predict → parse results."""
    import torch
    from DeepFluHGT import load_config
    from model.contracts import HGTConfig
    from model.api import build_model
    from data_pipeline.impl.dataset import SlidingWindowContigDataset
    from data_pipeline.impl.collator import HGTCollator
    from data_pipeline.contracts import HGTSample
    from torch.utils.data import DataLoader

    cfg = load_config(config)
    mc, dc = cfg["model"], cfg["data"]
    sw = dc["sw_window_size"]

    # Build HGTSample objects
    hgt_samples = []
    labels_map = {}
    for sid, (label, seq) in samples.items():
        hgt_samples.append(HGTSample(
            seq_id=sid, label=label, sequence=seq, r2_sequence=None,
            events=[], sample_type="POS" if label == 1 else "NEG",
            seq_len=len(seq)))
        labels_map[sid] = label

    # loadmodel
    model = build_model(HGTConfig(data_mode="contig", max_contig_length=sw,
        hidden_dim=mc["hidden_dim"], local_hidden_dim=mc["local_hidden_dim"],
        num_attention_heads=mc["num_attention_heads"], dropout=mc["dropout"],
        graph_top_k=mc["graph_top_k"], bio_feature_dim=mc["bio_feature_dim"],
        pretrained_nt_path=mc["pretrained_nt_path"],
        pretrained_caduceus_path=mc["pretrained_caduceus_path"])).to("cuda")
    state = torch.load(ckpt, map_location="cuda", weights_only=False)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    print(f"[Model] epoch={state.get('epoch')}", flush=True)

    # Sliding windows
    ds = SlidingWindowContigDataset(samples=hgt_samples, window_size=sw,
        overlap=dc["sw_overlap"], is_training=False,
        max_neg_windows=9999, max_eval_windows=dc["sw_max_eval_windows"],
        extract_bio=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=HGTCollator(pad_id=0), num_workers=2)
    print(f"[Data] {len(hgt_samples)} contigs, {len(ds)} windows", flush=True)

    # Collect window predictions
    from collections import defaultdict
    contig_windows = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to("cuda")
            output = model(batch, early_exit=False)
            prob = torch.sigmoid(output.cls_logits.squeeze(-1)).item()
            sids = getattr(batch, "original_seq_ids", None) or batch.seq_ids
            contig_windows[sids[0]].append(prob)

    return contig_windows, labels_map


def aggregate_and_eval(contig_windows, labels_map, threshold=0.5):
    """aggregationwindow → contig levelmetric."""
    from collections import defaultdict

    sid_list = sorted(contig_windows.keys())
    y_true = np.array([labels_map.get(s, 0) for s in sid_list])

    # Aggregation strategies
    strategies = {}
    for name in ["max", "mean", "median", "top3_mean"]:
        if name == "max":
            vals = [max(contig_windows[s]) for s in sid_list]
        elif name == "mean":
            vals = [np.mean(contig_windows[s]) for s in sid_list]
        elif name == "median":
            vals = [np.median(contig_windows[s]) for s in sid_list]
        elif name == "top3_mean":
            vals = [np.mean(sorted(contig_windows[s], reverse=True)[:3]) for s in sid_list]
        strategies[name] = np.array(vals)

    results = {}
    for name, scores in strategies.items():
        preds = (scores >= threshold).astype(int)
        n_pos = int(y_true.sum())
        n_neg = len(y_true) - n_pos

        entry = {
            "n_samples": len(y_true),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "threshold": threshold,
        }

        if n_pos > 0 and n_neg > 0:
            entry["auc"] = float(roc_auc_score(y_true, scores))
            entry["auprc"] = float(average_precision_score(y_true, scores))
            entry["f1"] = float(f1_score(y_true, preds))
            entry["precision"] = float(precision_score(y_true, preds, zero_division=0))
            entry["recall"] = float(recall_score(y_true, preds, zero_division=0))
            tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
            entry["tp"] = int(tp)
            entry["fp"] = int(fp)
            entry["fn"] = int(fn)
            entry["tn"] = int(tn)
            entry["specificity"] = float(tn / max(1, tn + fp))
            # ECE  —  equal-width binning, robust to empty bins
            # (calibration_curve drops empty bins → length mismatch vs np.histogram;
            #  compute manually so bin counts always align.)
            bins = min(10, n_pos, n_neg)
            if bins > 1:
                bin_edges = np.linspace(0.0, 1.0, bins + 1)
                bin_idx = np.clip(np.digitize(scores, bin_edges[1:-1]), 0, bins - 1)
                ece = 0.0
                for b in range(bins):
                    mask = bin_idx == b
                    if mask.sum() > 0:
                        acc = float(y_true[mask].mean())   # fraction positive
                        conf = float(scores[mask].mean())  # mean predicted prob
                        ece += mask.sum() / len(scores) * abs(acc - conf)
                entry["ece"] = float(ece)
            # stratified (positive = GI, negative = non-GI)
            pos_scores = scores[y_true == 1]
            neg_scores = scores[y_true == 0]
            entry["pos_mean_prob"] = float(pos_scores.mean())
            entry["neg_mean_prob"] = float(neg_scores.mean())
            entry["neg_spec_at_0.5"] = float((neg_scores < 0.5).mean())

        results[name] = entry

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--benchmark", default="benchmark/literature_gi/repo/test_data/benbow_test_data.fasta")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("=" * 60)
    print("External benchmark evaluation (Benbow real GI)")
    print("=" * 60)

    # 1. Read Benbow
    samples = read_benbow(args.benchmark)
    n_pos = sum(1 for _, (l, _) in samples.items() if l == 1)
    n_neg = len(samples) - n_pos
    print(f"[Data] {len(samples)} samples: {n_pos} positive (GI) + {n_neg} negative")

    # 2. Predict
    contig_windows, labels_map = extract_and_predict(samples, args.ckpt, args.config, os.path.dirname(args.out))

    # 2b. Persist raw window predictions  —  insulate the expensive GPU inference
    #     from any downstream metric-computation bugs (re-runs then need no GPU).
    raw_path = args.out.replace(".json", "_raw_windows.json")
    with open(raw_path, "w") as f:
        json.dump({
            "contig_windows": {k: list(v) for k, v in contig_windows.items()},
            "labels_map": {str(k): int(v) for k, v in labels_map.items()},
        }, f)
    print(f"Raw window predictions saved: {raw_path}")

    # 3. evaluation
    results = aggregate_and_eval(contig_windows, labels_map)

    # Print
    print(f"\n{'='*60}")
    print("Benbow external benchmark results (real GI data)")
    print(f"{'='*60}")
    for name, m in results.items():
        print(f"\n[aggregation: {name}]")
        for k, v in m.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    # save
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.out}")

    # Key takeaway
    best = results.get("max", {})
    print(f"\n{'='*60}")
    print(f"Key: real-data max AUC={best.get('auc', 'N/A')} AUPRC={best.get('auprc', 'N/A')} (simulated data 0.96)")
    print(f"     neg specificity={best.get('neg_spec_at_0.5', 'N/A')} (simulated hard_neg spec=0.03)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
