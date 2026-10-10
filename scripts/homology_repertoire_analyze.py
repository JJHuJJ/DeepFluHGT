#!/usr/bin/env python3
"""homology_repertoire_analyze.py — B1-v4 statistical analysis (internal protocol name retained for provenance) (archived standalone form).

B7-2 (2026-10-04): the v4 analysis was originally run as inline Python; this
script is its archived reconstruction. It reads only the archived per-arm
contig scores (predictions/<arm>/scores.json) and recomputes every quantity in
B1v4_RESULT.json: primary signed-Δ distribution, placebo and negative controls,
secondary AUROC deltas, and supplementary gates A/B/C.

Validation contract: outputs must reproduce B1v4_RESULT.json (medians/means
exact; seeded bootstrap CI within Monte-Carlo tolerance).
"""
import argparse
import json

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def load_arm(root, arm):
    d = json.load(open(f"{root}/predictions/{arm}/scores.json"))
    return dict(zip(d["ids"], d["agg_mean"]))




def auc_dicts(pos_d, neg_d):
    ps = np.array(list(pos_d.values())); ns = np.array(list(neg_d.values()))
    y = np.concatenate([np.ones(len(ps)), np.zeros(len(ns))])
    return float(roc_auc_score(y, np.concatenate([ps, ns])))

def paired_delta(root, arm, ref):
    a, r = load_arm(root, arm), load_arm(root, ref)
    common = sorted(set(a) & set(r))
    ids = np.array(common)
    delta = np.array([a[i] - r[i] for i in common])
    return ids, delta


def auroc_pair(root, pos_arm, neg_arm):
    p, n = load_arm(root, pos_arm), load_arm(root, neg_arm)
    common_ref = None  # independent arms: use each arm's own contigs
    ps = np.array(list(p.values())); ns = np.array(list(n.values()))
    y = np.concatenate([np.ones(len(ps)), np.zeros(len(ns))])
    s = np.concatenate([ps, ns])
    return float(roc_auc_score(y, s))


def alignment_gates(d1, pl_ids, p1, p2, orig_pos):
    """R1-TierB alignment (2026-10-04): the two declared B1-v4 analyses the original
    inline script omitted; verified against SI S6.8 (TOST p<0.001; Wilcoxon 8e-71;
    rank-biserial 0.35 — standard matched-pairs formula)."""
    from scipy import stats as st
    pa = st.ttest_1samp(d1, -0.01, alternative="greater").pvalue
    pb = st.ttest_1samp(d1, 0.01, alternative="less").pvalue
    tost_p = max(pa, pb)
    a = np.array([abs(p1[i] - orig_pos[i]) for i in pl_ids])
    b = np.array([abs(p2[i] - orig_pos[i]) for i in pl_ids])
    w = st.wilcoxon(a, b)
    n = len(pl_ids)
    r_rb = abs(4 * w.statistic / (n * (n + 1)) - 1)
    return {"gate_TOST": {"bounds": "+/-0.01", "p": round(tost_p, 8), "si_claim": "p<0.001"},
            "gate_placebo": {"wilcoxon_p": float(f"{w.pvalue:.4g}"), "si_claim": "8e-71",
                             "rank_biserial": round(r_rb, 4), "si_claim_rb": "0.35", "n": n}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--B", type=int, default=10000)
    args = ap.parse_args()
    R = args.root

    # baseline
    baseline_auroc = auroc_pair(R, "orig_pos", "orig_neg")

    # primary: P1 junction patch vs original positives
    _, d1 = paired_delta(R, "P1", "orig_pos")
    primary = {
        "median_abs": float(np.median(np.abs(d1))),
        "mean_abs": float(np.mean(np.abs(d1))),
        "mean_signed": float(np.mean(d1)),
        # Tail fields on |delta| (clean-room F1 2026-10-04: the archived values
        # are percentiles of |delta| — verified exactly to 6dp; basis now matches).
        "P90": float(np.percentile(np.abs(d1), 90)),
        "P95": float(np.percentile(np.abs(d1), 95)),
        "P99": float(np.percentile(np.abs(d1), 99)),
        "max": float(np.max(np.abs(d1))),
        "min": float(np.min(d1)),
        "fraction_decrease": float((d1 < 0).mean()),
        "fraction_increase": float((d1 > 0).mean()),
    }

    # placebo + negative controls
    _, d2 = paired_delta(R, "P2", "orig_pos")
    _, dn = paired_delta(R, "N1", "orig_neg")
    placebo = {"median_abs": float(np.median(np.abs(d2))), "n": int(len(d2))}
    negative_control = {"median_abs": float(np.median(np.abs(dn)))}

    # same-subset probe (P1 restricted to contigs with both P1 and P2 arms)
    p1, p2 = load_arm(R, "P1"), load_arm(R, "P2")
    both = sorted(set(p1) & set(p2))
    d1_sub = np.array([p1[i] for i in both]) - np.array(
        [load_arm(R, "orig_pos")[i] for i in both])

    # secondary AUROC deltas — executed-arm caliber (validated: reproduces
    # archived delta_P1 = -0.013358 exactly). Positives/negatives are the
    # union dicts with patched scores overwriting originals where present.
    orig_pos_d, orig_neg_d = load_arm(R, "orig_pos"), load_arm(R, "orig_neg")
    n1_d = load_arm(R, "N1")
    neg_exec = {**orig_neg_d, **n1_d}
    a_orig = auroc_pair(R, "orig_pos", "orig_neg")
    a_p1 = auc_dicts({**orig_pos_d, **load_arm(R, "P1")}, neg_exec)
    a_p2 = auc_dicts({**orig_pos_d, **load_arm(R, "P2")}, neg_exec)
    # caliber note: archived delta_P2 = -0.007826; this caliber yields -0.007315
    # (diff 5e-4, secondary non-claim quantity; exact inline caliber of the
    # archived P2 value is unresolved — documented, not recomputed silently).

    # alignment gates (R1 TierB)
    _p1d, _p2d, _opd = load_arm(R, "P1"), load_arm(R, "P2"), load_arm(R, "orig_pos")
    _pl = sorted(set(_p1d) & set(_p2d) & set(_opd))
    align = alignment_gates(d1, _pl, _p1d, _p2d, _opd)

    # gates
    rng = np.random.RandomState(args.seed)
    means = []
    for _ in range(args.B):
        idx = rng.randint(0, len(d1), len(d1))
        means.append(d1[idx].mean())
    ci = [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
    t_p = float(stats.ttest_1samp(d1, 0.0).pvalue)
    try:
        w_p = float(stats.wilcoxon(d1).pvalue)
    except ValueError:
        w_p = 1.0
    gate_A = {"mean_signed": primary["mean_signed"], "ci_95": ci,
              "t_p": t_p, "wilcoxon_p": w_p}
    gate_B = {"cohens_d": float(primary["mean_signed"] / np.std(d1, ddof=1))}
    gate_C = {
        "frac>0.01": float((np.abs(d1) > 0.01).mean()),
        "frac>0.05": float((np.abs(d1) > 0.05).mean()),
        "frac>0.10": float((np.abs(d1) > 0.10).mean()),
        "frac>0.20": float((np.abs(d1) > 0.20).mean()),
        # tail_direction on raw signed fractions; the archived file's 0.513/0.487
        # is a normalized caliber (clean-room F3: internally inconsistent with the
        # archived primary fractions 0.504/0.475; documented, not redefined).
        "tail_direction": {"dec": float((d1 < 0).mean()),
                           "inc": float((d1 > 0).mean())},
    }

    out = {
        "analysis": "B1_v4_boundary_complete_reconstruction",
        "baseline_auroc": round(baseline_auroc, 6),
        "primary": {k: round(v, 6) if isinstance(v, float) else v
                    for k, v in primary.items()},
        "placebo": placebo,
        "negative_control": negative_control,
        "secondary_auroc": {"delta_P1": round(a_p1 - a_orig, 6),
                            "delta_P2": round(a_p2 - a_orig, 6)},
        "same_subset_probe": {
            "n": len(both),
            "p1_median_abs_same_subset": round(float(np.median(np.abs(d1_sub))), 6),
            "ratio_vs_placebo": round(float(np.median(np.abs(d1_sub))) / placebo["median_abs"], 4),
        },
        "alignment_gates": align,
        "supplementary_gates": {
            "gate_A": {**{k: (round(v, 6) if isinstance(v, float) else v)
                          for k, v in gate_A.items()},
                       "ci_method": f"percentile bootstrap of mean, seed={args.seed}, B={args.B}"},
            "gate_B": {k: round(v, 6) for k, v in gate_B.items()},
            "gate_C": {k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in gate_C.items()},
        },
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
