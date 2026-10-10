#!/usr/bin/env python3
"""Aggregate ablation-variant results → the paper's ablation table (RFC_ABLATION_FLAG.md §10.3).

readeachvariantattruereal benchmark (Benbow / IslandViewer) ofevaluation JSON
(produced by eval_external_benbow.py, format {strategy: {auc, ...}}) plus the optional
_raw_windows.json (contig_windows + labels_map), compute:
  - per-benchmark AUC (Benbow=median, IslandViewer=max, following the audit convention)
  - bootstrap 95% CI (resampling contigs from _raw_windows.json, when present)
  - ΔAUC vs Full (primarymode Contig, truereal benchmark)
  - optional val AUC (val_metrics.json)

No dependency on model code; once training is done and each variant is evaluated, this can run directly.

usage:
  python scripts/aggregate_ablation.py \
      --variants-dir audit_output/ablation \
      --full-benbow audit_output/05_benbow_results.json \
      --full-iv audit_output/08_real_contig_results.json \
      --out audit_output/ablation_table.json
"""
from __future__ import annotations

import os
import json
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
NAN3 = (float("nan"), float("nan"), float("nan"))


def load_auc(path: str, strategy: str) -> float:
    """read the AUC of a given aggregation strategy from eval_external_benbow.py output."""
    if not os.path.exists(path):
        return float("nan")
    with open(path) as f:
        d = json.load(f)
    if strategy in d and isinstance(d[strategy], dict) and "auc" in d[strategy]:
        return float(d[strategy]["auc"])
    if strategy in d and isinstance(d[strategy], (int, float)):
        return float(d[strategy])
    return float("nan")


def bootstrap_auc(raw_path: str, n: int = 1000, seed: int = 42) -> tuple:
    """from _raw_windows.json (contig_windows + labels_map) bootstrap contig level AUC.

    contig-level score = median(window probs) (the audit's primary recommendation); contigs are resampled to estimate the AUC distribution.
    return (point_auc, ci_low, ci_high); none raw filereturn NAN3.
    """
    if not raw_path or not os.path.exists(raw_path):
        return NAN3
    with open(raw_path) as f:
        d = json.load(f)
    cw = d.get("contig_windows", {})
    lm = d.get("labels_map", {})
    sids = sorted(cw.keys())
    if len(sids) < 5:
        return NAN3
    scores = np.array([np.median(cw[s]) for s in sids], dtype=float)
    labels = np.array([int(lm.get(s, 0)) for s in sids])
    pos = int(labels.sum())
    neg = len(labels) - pos
    if pos < 2 or neg < 2:
        return NAN3
    rng = np.random.RandomState(seed)
    aucs, idx = [], np.arange(len(sids))
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        lb, sb = labels[b], scores[b]
        if lb.sum() < 1 or (len(lb) - lb.sum()) < 1:
            continue
        aucs.append(roc_auc_score(lb, sb))
    if not aucs:
        return NAN3
    a = np.array(aucs)
    return (float(a.mean()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))


def fmt(point: float, ci: tuple) -> str:
    if np.isnan(point):
        return " — "
    if np.isnan(ci[1]) or np.isnan(ci[2]):
        return f"{point:.3f}"
    return f"{point:.3f} [{ci[1]:.3f}, {ci[2]:.3f}]"


def raw_for(result_json: str) -> str:
    """eval_external_benbow.py saves the raw data as <out>_raw_windows.json."""
    return result_json.replace(".json", "_raw_windows.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants-dir", required=True,
                    help="directory containing one subdirectory per variant (each with *benbow*.json / *real_contig*.json)")
    ap.add_argument("--full-benbow", default="audit_output/05_benbow_results.json")
    ap.add_argument("--full-iv", default="audit_output/08_real_contig_results.json")
    ap.add_argument("--benbow-strategy", default="median")
    ap.add_argument("--iv-strategy", default="max")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Full baseline
    full_bw = load_auc(args.full_benbow, args.benbow_strategy)
    full_iv = load_auc(args.full_iv, args.iv_strategy)
    full_bw_ci = bootstrap_auc(raw_for(args.full_benbow), args.n_bootstrap)
    full_iv_ci = bootstrap_auc(raw_for(args.full_iv), args.n_bootstrap)

    def gather(vpath: Path) -> dict:
        bw = next(vpath.glob("*benbow*.json"), None)
        iv = next((p for p in vpath.glob("*.json")
                   if "real_contig" in p.name or "island" in p.name.lower()), None)
        bw_auc = load_auc(str(bw), args.benbow_strategy) if bw else float("nan")
        iv_auc = load_auc(str(iv), args.iv_strategy) if iv else float("nan")
        bw_ci = bootstrap_auc(raw_for(str(bw)), args.n_bootstrap) if bw else NAN3
        iv_ci = bootstrap_auc(raw_for(str(iv)), args.n_bootstrap) if iv else NAN3
        val_auc = float("nan")
        vp = vpath / "val_metrics.json"
        if vp.exists():
            try:
                val_auc = float(json.load(open(vp)).get("best_val_auc", float("nan")))
            except Exception:
                pass
        return dict(benbow_auc=bw_auc, benbow_ci=bw_ci, iv_auc=iv_auc, iv_ci=iv_ci, val_auc=val_auc)

    rows = [dict(variant="Full", mode="Contig",
                 benbow_auc=full_bw, benbow_ci=full_bw_ci,
                 iv_auc=full_iv, iv_ci=full_iv_ci, val_auc=float("nan"),
                 dAUC_iv=0.0)]

    vdir = Path(args.variants_dir)
    if vdir.exists():
        for sub in sorted(vdir.iterdir()):
            if not sub.is_dir():
                continue
            g = gather(sub)
            d = float(g["iv_auc"] - full_iv) if not (np.isnan(g["iv_auc"]) or np.isnan(full_iv)) else float("nan")
            rows.append(dict(variant=sub.name,
                             mode="Contig" if "pe" not in sub.name.lower() else "PE",
                             **g, dAUC_iv=d))

    # print
    print("\n" + "=" * 90)
    print("ablationtable (RFC §10.3) — truereal benchmark AUC + bootstrap 95% CI")
    print("=" * 90)
    hdr = f"{'Variant':<24} {'Mode':<7} {'Val':<7} {'Benbow AUC [CI]':<26} {'IslandViewer [CI]':<26} {'ΔAUC(IV)'}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        v = f"{r['val_auc']:.3f}" if not np.isnan(r.get("val_auc", float("nan"))) else " — "
        d = f"{r['dAUC_iv']:+.3f}" if not np.isnan(r.get("dAUC_iv", float("nan"))) else " — "
        print(f"{r['variant']:<24} {r['mode']:<7} {v:<7} "
              f"{fmt(r['benbow_auc'], r['benbow_ci']):<26} "
              f"{fmt(r['iv_auc'], r['iv_ci']):<26} {d}")
    print("\nnote: Benbow=median, IslandViewer=max (audit convention); "
          f"CI estimated by contig bootstrap (n={args.n_bootstrap}). ΔAUC(IV) is relative to Full.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"rows": rows,
                   "full_baseline": {"benbow": full_bw, "iv": full_iv},
                   "strategies": {"benbow": args.benbow_strategy, "iv": args.iv_strategy},
                   "n_bootstrap": args.n_bootstrap}, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
