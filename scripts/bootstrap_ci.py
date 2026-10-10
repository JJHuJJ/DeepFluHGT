#!/usr/bin/env python3
"""Bootstrap AUC/AUPRC confidence intervals  —  Phase 0 statistical rigor.

A single-point AUC is not enough for the small real-data benchmarks (Benbow n=566 / IslandViewer n=600);
top journals require CIs. This script estimates 95% CIs by bootstrap resampling.

usage:
  python scripts/bootstrap_ci.py --labels <labels.tsv> --scores <scores.tsv> --n 1000 [--metric auc|auprc]
  # labels.tsv: id<TAB>label ; scores.tsv: id<TAB>score

Can also be applied to aggregated contig-level scores (from the _raw_windows.json export).
"""
import csv, argparse, json
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def bootstrap(y, s, metric, n, seed):
    rng = np.random.RandomState(seed)
    vals = []
    N = len(y)
    for _ in range(n):
        idx = rng.randint(0, N, N)
        if len(set(y[idx])) < 2:
            continue
        vals.append(metric(y[idx], s[idx]))
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def from_raw_windows(raw_json, agg="median"):
    """Load (labels, scores) from eval_external_benbow's _raw_windows.json export."""
    d = json.load(open(raw_json))
    cw, lm = d["contig_windows"], d["labels_map"]
    common = sorted(set(cw) & set(lm))
    y = np.array([int(lm[c]) for c in common])
    if agg == "max":
        s = np.array([max(cw[c]) for c in common])
    elif agg == "mean":
        s = np.array([np.mean(cw[c]) for c in common])
    else:
        s = np.array([np.median(cw[c]) for c in common])
    return y, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", help="labels.tsv (id\\tlabel)")
    ap.add_argument("--scores", help="scores.tsv (id\\tscore)")
    ap.add_argument("--raw-windows", help="or read eval_external_benbow's _raw_windows.json directly")
    ap.add_argument("--agg", default="median", choices=["max", "mean", "median", "top3"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--metric", choices=["auc", "auprc"], default="auc")
    args = ap.parse_args()

    if args.raw_windows:
        y, s = from_raw_windows(args.raw_windows, args.agg if args.agg != "top3" else "median")
    else:
        lab = {r["id"]: float(r["label"]) for r in csv.DictReader(open(args.labels), delimiter="\t")}
        sc = {r["id"]: float(r["score"]) for r in csv.DictReader(open(args.scores), delimiter="\t")}
        common = sorted(set(lab) & set(sc))
        y = np.array([lab[i] for i in common]); s = np.array([sc[i] for i in common])

    metric = roc_auc_score if args.metric == "auc" else average_precision_score
    pt = metric(y, s)
    mean, lo, hi = bootstrap(y, s, metric, args.n, 42)
    print(f"N={len(y)}  point {args.metric}={pt:.4f}  bootstrap mean={mean:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]  (n_boot={args.n})")


if __name__ == "__main__":
    main()
