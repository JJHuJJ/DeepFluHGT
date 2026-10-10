#!/usr/bin/env python3
"""bootstrap_decomposition.py  —  Genome-cluster bootstrap for decomposition CIs.

Canonical implementation for EDT5 intervals. Extracted from P0 pipeline (Phase 3.6).
Algorithm: cluster bootstrap on AUPRC differences; B=5000; seed=42.
"""
import json
import numpy as np
from sklearn.metrics import average_precision_score

import os
R = os.environ.get("DEEPFLUHGT_ROOT", ".")
M0 = np.load(f"{R}/docs/evidence/ood_crosstaxa_m0.npz", allow_pickle=True)
M2 = np.load(f"{R}/docs/evidence/terminal/M2/test_eval_m2final.npz", allow_pickle=True)
M3 = np.load(f"{R}/docs/evidence/terminal/M3/test_eval_m3final.npz", allow_pickle=True)
assert (M0["contig_ids"] == M2["contig_ids"]).all() and (M0["contig_ids"] == M3["contig_ids"]).all(), "contig order is not consistent"
y_eval = M0["labels"].astype(int)
genome = M0["genome"].astype(str)
print("sample unique genome keys:", sorted(set(genome))[:6], "... total", len(set(genome)), "clusters; NO_HOST class:", sum(1 for g in set(genome) if 'NO_HOST' in g or g==''))

# ── canonical tag (constructed from the provenance.tsv label column) ──
canon = {}
with open(f"{R}/simulated_data/Contig_500k/test/provenance.tsv") as f:
    next(f)
    for line in f:
        p = line.split("\t")
        if p[0] not in canon: canon[p[0]] = int(p[1])
y_canon = np.array([canon[i] for i in M0["contig_ids"]])
flips = int((y_canon != y_eval).sum())
print(f"canonical {y_canon.sum()}/50000 vs eval {y_eval.sum()}/50000  ·  flips {flips}  ·  direction: neg→pos = {int(((y_canon==0)&(y_eval==1)).sum())}, pos→neg = {int(((y_canon==1)&(y_eval==0)).sum())}")

def ap(s, y): return float(average_precision_score(y, s))
out = {"experiment": "EDT5_test_genome_cluster_recompute", "date": "2026-09-23",
       "inputs": {"M0": "docs/evidence/ood_crosstaxa_m0.npz", "M2": "docs/evidence/terminal/M2/test_eval_m2final.npz",
                  "M3": "docs/evidence/terminal/M3/test_eval_m3final.npz", "unit": "genome_cluster", "B": 5000, "seed": 42},
       "reconstruction_check": {}, "point_estimates": {}, "label_noise": {}, "cis": {}}

for agg in ["max", "mean"]:
    s0, s2, s3 = M0[f"agg_{agg}"], M2[f"agg_{agg}"], M3[f"agg_{agg}"]
    a0, a2, a3 = ap(s0, y_eval), ap(s2, y_eval), ap(s3, y_eval)
    out["point_estimates"][agg] = {"M0": round(a0,4), "M2": round(a2,4), "M3": round(a3,4),
                                    "cap": round(a3-a0,4), "res": round(a2-a3,4), "share": round((a3-a0)/(a2-a0),4)}
    out["reconstruction_check"][agg] = {"M0": round(a0,4), "M2": round(a2,4), "M3": round(a3,4)}
    # label-noise sensitivity (canonical vs eval under this aggregation)
    out["label_noise"][agg] = {m: {"eval": round(ap(s,y_eval),4), "canonical": round(ap(s,y_canon),4),
                                    "delta_pp": round(10000*(ap(s,y_eval)-ap(s,y_canon))/1)} for m, s in [("M0",s0),("M2",s2),("M3",s3)]}
out["label_noise"]["flips_neg2pos"] = int(((y_canon==0)&(y_eval==1)).sum())
out["label_noise"]["realized_rate"] = round(flips/50000, 5)

# ── genome-cluster bootstrap (seed 42, B 5000) ──
gids, inverse = np.unique(genome, return_inverse=True)
# NO_HOST/synthetic buckets → singleton clusters (lesson learned)
is_synth = np.array([("NO_HOST" in g) or (g == "") for g in gids])
rng = np.random.default_rng(42)
n = len(y_eval)
B = 5000
for agg in ["max", "mean"]:
    s0, s2, s3 = M0[f"agg_{agg}"], M2[f"agg_{agg}"], M3[f"agg_{agg}"]
    # cluster index: true genomes follow gids; each synthetic bucket is its own singleton
    synth_pos = set(np.where(is_synth)[0])
    cluster_of_contig = np.empty(n, dtype=np.int64)
    next_cid = len(gids)
    for pos in range(len(gids)):
        pass
    # vectorized: start from inverse, then assign synthetic clusters unique ids
    cluster_of_contig = inverse.astype(np.int64)
    for i in range(n):
        if is_synth[inverse[i]]:
            cluster_of_contig[i] = next_cid; next_cid += 1
    uniq_clusters = np.unique(cluster_of_contig)
    members = [np.where(cluster_of_contig == c)[0] for c in uniq_clusters]
    caps, ress, shares = np.empty(B), np.empty(B), np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[c] for c in pick])
        ya, y0, y2, y3 = y_eval[idx], s0[idx], s2[idx], s3[idx]
        if ya.min() == ya.max(): caps[b]=np.nan; ress[b]=np.nan; shares[b]=np.nan; continue
        a0, a2, a3 = ap(y0,ya), ap(y2,ya), ap(y3,ya)
        caps[b] = a3-a0; ress[b] = a2-a3; shares[b] = (a3-a0)/(a2-a0)
    def ci(a): a=a[~np.isnan(a)]; return [round(float(np.percentile(a,2.5)),4), round(float(np.percentile(a,97.5)),4)]
    out["cis"][f"test_{agg}"] = {"unit":"genome_cluster","B":B,"seed":42,
        "cap_ci":ci(caps),"res_ci":ci(ress),"share_ci":[round(float(np.percentile(shares[~np.isnan(shares)],2.5)),3), round(float(np.percentile(shares[~np.isnan(shares)],97.5)),3)]}
    print(f"test/{agg} genome-cluster: cap{out['cis'][f'test_{agg}']['cap_ci']} res{out['cis'][f'test_{agg}']['res_ci']} share{out['cis'][f'test_{agg}']['share_ci']}")

# control: archived CIs (unit not recorded)
arch = json.load(open(f"{R}/docs/evidence/homology_audit/test_decomposition_bootstrap.json"))
out["archived_reference"] = {a: {"cap_ci": arch[a]["cap_ci"], "res_ci": arch[a]["res_ci"], "unit_recorded": arch[a].get("unit","UNRECORDED")} for a in ["max","mean"]}
json.dump(out, open(f"{R}/REVISION_EVIDENCE_CLOSURE/01_SCIENCE_CLOSURE/edt5_test_genome_cluster.json","w"), indent=2, default=str)
print("\n[OK] edt5_test_genome_cluster.json")
