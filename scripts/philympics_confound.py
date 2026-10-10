#!/usr/bin/env python3
"""philympics_confound.py  —  fourth-anchor GC/length confound audit (replica of v6.8 analyze_v5_4_confound)

Reads docs/evidence/real_anchor_philympics_m0final.npz (mean-aggregation scores) +
benchmark/philympics/v7_anchor/v7_matching_stats.json:
  - OLS-residualized mean-pool scores (raw / minus GC / minus length / minus GC+length) → AUROC;
  - permutation test (10k, seed 42) on the GC+length-residualized scores.
output benchmark/philympics/v7_anchor/v7_confound_results.json
(the report generator write_real_anchors_m0final_report.py references it).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
NPZ = ROOT / "docs/evidence/real_anchor_philympics_m0final.npz"
MATCH = ROOT / "benchmark/philympics/v7_anchor/v7_matching_stats.json"
OUT = ROOT / "benchmark/philympics/v7_anchor/v7_confound_results.json"
N_PERM, SEED = 10000, 42


def ols_residual(y, X):
    Xc = np.column_stack([X, np.ones(len(X))])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    return y - Xc @ beta


def main():
    match = json.loads(MATCH.read_text())
    valid = [i for i, m in enumerate(match) if m.get("control_len", 0) > 0]
    n = len(valid)

    z = np.load(NPZ, allow_pickle=True)
    ids = [str(s) for s in z["contig_ids"]]
    mean_scores = dict(zip(ids, z["agg_mean"].astype(float)))

    p_ids = [match[i]["prophage_id"] for i in valid]
    c_ids = [match[i]["control_id"] for i in valid]
    p_score = np.array([mean_scores[s] for s in p_ids])
    c_score = np.array([mean_scores[s] for s in c_ids])
    p_gc = np.array([match[i]["prophage_gc"] for i in valid])
    c_gc = np.array([match[i]["control_gc"] for i in valid])
    p_len = np.array([match[i]["prophage_len"] for i in valid], dtype=float)
    c_len = np.array([match[i]["control_len"] for i in valid], dtype=float)

    score = np.concatenate([p_score, c_score])
    gc = np.concatenate([p_gc, c_gc])
    length = np.concatenate([p_len, c_len])
    y = np.concatenate([np.ones(n), np.zeros(n)])

    raw = float(roc_auc_score(y, score))
    auc_gc = float(roc_auc_score(y, ols_residual(score, gc.reshape(-1, 1))))
    auc_len = float(roc_auc_score(y, ols_residual(score, length.reshape(-1, 1))))
    auc_both = float(roc_auc_score(y, ols_residual(
        score, np.column_stack([gc, length]))))

    rng = np.random.RandomState(SEED)
    ge = 0
    for _ in range(N_PERM):
        yp = rng.permutation(y)
        if roc_auc_score(yp, ols_residual(
                score, np.column_stack([gc, length]))) >= auc_both:
            ge += 1
    perm_p = ge / N_PERM

    res = {"n": n, "raw_auc": raw,
           "residualized_auc": {"after_gc": auc_gc, "after_length": auc_len,
                                "after_gc_length": auc_both},
           "permutation_p": perm_p, "n_permutations": N_PERM, "seed": SEED,
           "significant_p05": bool(perm_p < 0.05)}
    OUT.write_text(json.dumps(res, indent=2))
    print(f"raw={raw:.3f} |GC={auc_gc:.3f} |len={auc_len:.3f} "
          f"|GC+len={auc_both:.3f} perm_p={perm_p:.4f} → {OUT}")


if __name__ == "__main__":
    main()
