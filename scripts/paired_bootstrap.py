#!/usr/bin/env python3
"""paired_bootstrap.py — Philympics paired-event bootstrap (release version).

Reproduces the published Philympics paired-event confidence intervals from the
shipped per-contig scores and pair records:

    inputs : docs/evidence/real_anchor_philympics_m0final.npz   (per-contig mean scores)
             benchmark/philympics/v7_anchor/v7_matching_stats.json (pair records)
    method : resample PAIR indices (B = 10,000, seed 42), rebuild the pooled
             array from the sampled pairs, recompute AUROC
    output : point estimate, wins, and per-aggregation pair-preserving 95% CIs

History: the original evaluator (eval_real_anchors_gpu.py, pre-2026-10-04)
resampled the concatenated contigs i.i.d., which destroys the prophage-control
pairing. The intervals were recomputed with pair-preserving resampling on
2026-09-29 (point estimates unchanged); this script is the shipped, runnable
record of that correction, and eval_real_anchors_gpu.py now uses the same
pair-preserving method.

Usage:
    python scripts/paired_bootstrap.py [--npz PATH] [--pairs PATH] [--seed 42] [--B 10000]
"""
import argparse
import json

import numpy as np
from sklearn.metrics import roc_auc_score


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default="docs/evidence/real_anchor_philympics_m0final.npz")
    ap.add_argument("--pairs", default="benchmark/philympics/v7_anchor/v7_matching_stats.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--B", type=int, default=10000)
    args = ap.parse_args()

    npz = np.load(args.npz, allow_pickle=True)
    ids = list(npz["contig_ids"].astype(str))
    scores = {k: float(v) for k, v in zip(ids, npz["agg_mean"])}
    pairs_rec = json.load(open(args.pairs))
    pairs = [(r["prophage_id"], r["control_id"]) for r in pairs_rec
             if r["prophage_id"] in scores and r["control_id"] in scores]
    print(f"evaluable pairs (both members scored): {len(pairs)} "
          f"(records={len(pairs_rec)}, scored contigs={len(ids)})")

    ps = np.array([scores[p] for p, c in pairs])
    cs = np.array([scores[c] for p, c in pairs])
    y = np.concatenate([np.ones(len(ps)), np.zeros(len(cs))])
    s = np.concatenate([ps, cs])
    point = float(roc_auc_score(y, s))
    wins = int((ps > cs).sum())

    rng = np.random.RandomState(args.seed)
    n_pairs = len(pairs)
    boot = []
    for _ in range(args.B):
        pi = rng.randint(0, n_pairs, n_pairs)
        yb = np.concatenate([np.ones(n_pairs), np.zeros(n_pairs)])
        sb = np.concatenate([ps[pi], cs[pi]])
        if len(set(yb)) < 2:
            continue
        boot.append(roc_auc_score(yb, sb))
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

    out = {
        "method": "pair-index bootstrap (prophage-control pairs kept intact)",
        "agg": "mean", "n_pairs": n_pairs, "seed": args.seed, "B": args.B,
        "point_auroc_pooled": round(point, 4), "wins": wins,
        "ci95_paired": [round(x, 4) for x in ci],
        "excludes_chance": ci[0] > 0.5,
    }
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
