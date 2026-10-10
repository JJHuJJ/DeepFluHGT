#!/usr/bin/env python3
"""Systematic aggregation-strategy comparison  —  recompute multiple aggregations from raw window predictions to decide which one the paper reports at Contig level.

Pure CPU (reuses the already-saved raw windows; no model re-run). Aggregations: max/mean/median/top3/top5/P90.
For each dataset, report AUC + AUPRC + neg specificity + pos recall per aggregation.

usage:
  python scripts/compare_aggregation.py \
      --raw audit_output/05_benbow_results_raw_windows.json:Benbow \
      --raw audit_output/08_real_contig_results_raw_windows.json:IslandViewer \
      --out audit_output/09_aggregation_compare.json
"""
import json, argparse, sys
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def aggregates(scores):
    s = sorted(scores, reverse=True)
    n = len(s)
    return {
        "max": max(scores),
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "top3_mean": float(np.mean(s[:3])),
        "top5_mean": float(np.mean(s[:5])),
        "P90": float(np.percentile(scores, 90)),
    }


def evaluate(raw_path, name):
    with open(raw_path) as f:
        d = json.load(f)
    cw = d["contig_windows"]
    lm = d["labels_map"]
    common = sorted(set(cw) & set(lm))
    y = np.array([int(lm[c]) for c in common])
    rows = {}
    for agg_name in ["max", "mean", "median", "top3_mean", "top5_mean", "P90"]:
        scores = np.array([aggregates(cw[c])[agg_name] for c in common])
        n_pos = int(y.sum()); n_neg = len(y) - n_pos
        if n_pos > 0 and n_neg > 0:
            auc = float(roc_auc_score(y, scores))
            auprc = float(average_precision_score(y, scores))
        else:
            auc = auprc = float("nan")
        neg_s = scores[y == 0]
        pos_s = scores[y == 1]
        rows[agg_name] = {
            "auc": round(auc, 4), "auprc": round(auprc, 4),
            "neg_spec@0.5": round(float((neg_s < 0.5).mean()), 4),
            "pos_recall@0.5": round(float((pos_s >= 0.5).mean()), 4),
            "neg_mean": round(float(neg_s.mean()), 3), "pos_mean": round(float(pos_s.mean()), 3),
        }
    return {"dataset": name, "n": len(common), "n_pos": int(y.sum()), "n_neg": len(y) - int(y.sum()),
            "by_agg": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="append", required=True, help="path:name (repeatable)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results = []
    for spec in args.raw:
        path, name = spec.rsplit(":", 1)
        results.append(evaluate(path, name))

    print(f"\n{'='*78}")
    print(f"Systematic aggregation-strategy comparison (Contig level, raw windows recomputed on pure CPU)")
    print(f"{'='*78}")
    for r in results:
        print(f"\n[{r['dataset']}] n={r['n']} (pos={r['n_pos']}, neg={r['n_neg']})")
        print(f"  {'aggregation':<12} {'AUC':>7} {'AUPRC':>7} {'neg_spec@0.5':>13} {'pos_recall@0.5':>15} {'neg_mean':>9} {'pos_mean':>9}")
        for agg, m in r["by_agg"].items():
            print(f"  {agg:<12} {m['auc']:>7} {m['auprc']:>7} {m['neg_spec@0.5']:>13} {m['pos_recall@0.5']:>15} {m['neg_mean']:>9} {m['pos_mean']:>9}")

    # Recommendation: the aggregation with the highest AUC
    print(f"\n{'='*78}")
    print("Recommended (by AUC):")
    for r in results:
        best = max(r["by_agg"].items(), key=lambda kv: kv[1]["auc"])
        print(f"  {r['dataset']}: {best[0]} (AUC={best[1]['auc']}, spec={best[1]['neg_spec@0.5']})")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
