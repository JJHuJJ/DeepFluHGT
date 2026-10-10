#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B1 cross-task RAP  —  NTv3-only probe (DeepFluHGT; the Evo2 half awaits a GPU window)
================================================================================
For the 5 Feng 2025 datasets under benchmark/crosstask/, run RAP probes on the
{dataset}_ntv3_pooled.npz produced by scripts/embed_ntv3_cpu.py (frozen NTv3-pre;
convention documented in the npz manifest):
**reuse the linear/ridge protocol of scripts/rap_probes.py** (same split / same
standardization / same probe hyperparameters / same bootstrap) + the paired DeLong
from scripts/delong_test.py.

Feature groups (3+1 per dataset):
  ntv3_1536    frozen NTv3-pre hidden mean-pool 1536d   —  —  primary setting
  len+gc       log10(len)+GC (2d)                       —  —  shortcut baseline
  kmer7_rc     RC-canonical 7-mer relative frequency (8192d)  —  —  baseline
  ntv3_768_proj the 768d output of GlobalEncoder._proj (archived random-projection
                variant, appendix only)                  —  —  supplementary group

protocol (consistent with step0 rap_probes):
  70/30 stratified split (seed42) within the test pool (rap_input.fasta)
  + 5-seed repeats (42-46) → SD
  + label-permutation null (ridge: batched primal exact solution; linear: fit one
  at a time, count tiered by n)
  + bootstrap CI + paired DeLong (same test fold, pairwise across feature groups).
  Note: embeddings cover only the test split (consistent with step0 RAP's
  "single-pool 70/30" convention);
  train→test (Feng's reported protocol) is deferred to a GPU-window task (to be
  done together with the Evo2 half).

ecoli_4mc is imbalanced (pos≈16.7%): primary metric AUPRC (AUROC also reported,
noting the risk of inflated values).

output (benchmark/crosstask/):
  crosstask_ntv3_results.csv / crosstask_ntv3_delong.csv / CROSTASK_NTV3_RAP.md

usage: python scripts/crosstask_ntv3_probe.py [--datasets all] [--seeds 42,43,44,45,46]
"""
from __future__ import annotations

import os
# Hard constraint: CPU-only + BLAS threads ≤4 (must be set before importing numpy/sklearn)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))      # reuse already-validated implementations
import rap_probes as rap                                    # load_fasta/kmer/split/probe
from delong_test import delong_test_full                    # paired DeLong

CROSS = REPO / "benchmark" / "crosstask"
DATASETS = ["human_vs_worm", "promoter_b_amyloliquefaciens", "promoter_r_capsulatus",
            "ecoli_4mc", "celegans_4mc"]
GROUPS = ["ntv3_1536", "len+gc", "kmer7_rc", "ntv3_768_proj"]   # last entry = supplementary group
PROBES = ["linear", "ridge"]
DELONG_PAIRS = [("ntv3_1536", "len+gc"), ("ntv3_1536", "kmer7_rc"),
                ("len+gc", "kmer7_rc"), ("ntv3_1536", "ntv3_768_proj")]
TASK = {  # README §2 summary (used by the report)
    "human_vs_worm": ("species discrimination (200bp window)", "H. sapiens vs C. elegans", "AUROC"),
    "promoter_b_amyloliquefaciens": ("promoter recognition (40bp)", "B. amyloliquefaciens", "AUROC"),
    "promoter_r_capsulatus": ("promoter recognition (40bp)", "R. capsulatus", "AUROC"),
    "ecoli_4mc": ("4mC methylation (41bp, imbalanced 16.7%)", "E. coli K-12", "AUPRC (primary)"),
    "celegans_4mc": ("4mC methylation (41bp, balanced)", "C. elegans", "AUROC"),
}


def n_perm_ladder(n_train: int, probe: str) -> int:
    """Permutation count tiered by problem size (CPU budget; actual n_perm is
    recorded in the CSV). Ridge is very cheap when batched. Measured: for n>5k an
    lbfgs permutation fit takes 35-45s each (n_iter 1400/278), so B=200 is
    infeasible → for n>5k: B=100 + early-stopped null fits at tol=1e-3 (see
    probe_linear_null); observed metrics still use the full protocol."""
    if probe == "ridge":
        return 1000
    return 1000 if n_train <= 5000 else 100


def probe_linear_null(xtr, ytr, xte):
    """Linear probe dedicated to the permutation null: same hyperparameters as
    rap.probe_linear, the only difference being tol=1e-3 (early stopping; measured
    3-4x speedup). The null distribution sits at the chance level under permuted
    labels and is insensitive to the optimizer tolerance; observed metrics still
    use tol=1e-4, full protocol unchanged."""
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m = LogisticRegression(C=1.0, max_iter=2000, tol=1e-3, solver="lbfgs")
        m.fit(xtr, ytr)
    return m.predict_proba(xte)[:, 1]


def ridge_null_batched(xtr, ytr_perm, xte, alpha=1.0):
    """Batched primal ridge null  —  —  point-wise equivalent to the rap.dual
    formula (normalization + α=1), solving for m permuted label vectors at once;
    avoids the n×n dual kernel matrix for large n."""
    mu = xtr.mean(0, keepdims=True); sd = xtr.std(0) + 1e-12
    a = (xtr - mu) / sd; b = (xte - mu) / sd
    yc = ytr_perm - ytr_perm.mean(0, keepdims=True)
    d = a.shape[1]
    g = a.T @ a + alpha * np.eye(d)
    g[np.diag_indices(d)] += 1e-9 * max(1.0, float(np.trace(g)) / d)  # numerical stability
    chol = np.linalg.cholesky(g)
    w = np.linalg.solve(chol.T, np.linalg.solve(chol, a.T @ yc))
    return b @ w + ytr_perm.mean(0, keepdims=True)


def _collect_meta(ds: str, extra: dict) -> tuple:
    """Load fasta+npz and fill the extra metadata (also called for datasets skipped
    on resume; needed by the report)."""
    ddir = CROSS / ds
    seqs = rap.load_fasta(ddir / "rap_input.fasta")
    z = np.load(ddir / f"{ds}_ntv3_pooled.npz", allow_pickle=False)
    ids = [str(s) for s in z["ids"]]
    assert ids == list(seqs.keys()), f"{ds}: npz ids do not match fasta order"
    y = z["labels"].astype(int)
    y_pre = np.array([1 if i.startswith("pos_") else 0 for i in ids])
    assert (y == y_pre).all(), f"{ds}: labels inconsistent with the pos_/neg_ prefix"
    st = np.array([rap.seq_stats(seqs[i]) for i in ids])
    # Exclude all-N sequences (no ACGT at all, GC=NaN)  —  —  htw has 644 such items
    # (hg38/ce11 gaps), which also make up the 643 within-pool duplicates recorded
    # in the README; excluded uniformly for all feature groups to guarantee
    # same-pool comparability.
    acgt_ok = ~np.isnan(st[:, 1])
    n_alln = int((~acgt_ok).sum())
    row_idx = np.where(acgt_ok)[0]          # kept row indices within the npz (original pool)
    if n_alln:
        print(f"[probe] {ds}: exclude {n_alln} all-N (no ACGT, GC=NaN) sequences", flush=True)
        seqs = {i: seqs[i] for i, k in zip(ids, acgt_ok) if k}
        ids = [i for i, k in zip(ids, acgt_ok) if k]
        y = y[acgt_ok]; st = st[acgt_ok]
    seen = set()
    keep = np.ones(len(ids), bool)
    for i, s in enumerate(ids):
        if seqs[s] in seen:
            keep[i] = False
        else:
            seen.add(seqs[s])
    extra[ds] = dict(n=len(ids), n_pool=int(len(z["ids"])), pos=int(y.sum()),
                     n_dup=int((~keep).sum()), n_allN_dropped=n_alln,
                     gc_mean=float(np.nanmean(st[:, 1])),
                     n_pct=float(np.nanmean(st[:, 2])),
                     manifest=json.loads(str(z["manifest"])),
                     task=TASK[ds][0], org=TASK[ds][1], prim=TASK[ds][2],
                     prev=float(y.mean()))
    return seqs, ids, y, st, keep, row_idx


def run_dataset(ds: str, seeds: list[int], rows: list, drows: list,
                extra: dict) -> dict:
    seqs, ids, y, st, keep, row_idx = _collect_meta(ds, extra)
    n_dup = extra[ds]["n_dup"]
    _z = np.load(CROSS / ds / f"{ds}_ntv3_pooled.npz")
    p1536 = _z["pooled_1536"].astype(np.float32)[row_idx]
    p768 = _z["pooled_768_proj"].astype(np.float32)[row_idx]
    feats = {
        "ntv3_1536": p1536,
        "len+gc": np.stack([np.log10(st[:, 0]), st[:, 1]], 1).astype(np.float32),
        "kmer7_rc": rap.kmer_feature(seqs, ids),
        "ntv3_768_proj": p768,
    }
    print(f"\n[probe] {ds}: n={len(ids)} pos={int(y.sum())} within-pool dups={n_dup} "
          f"GC={extra[ds]['gc_mean']:.3f} N%={extra[ds]['n_pct']:.4f}", flush=True)

    tr, te = rap.stratified_split(y, seed=42)
    yte = y[te]
    n_boot = 1000 if len(te) <= 10000 else 500
    rng = np.random.default_rng(42)
    perm_ys_full = None
    preds = {}

    for g in GROUPS:
        X = feats[g]
        for pname in PROBES:
            t0 = time.time()
            xtr, xte = rap.standardize(X[tr], X[te])
            pfn = rap.PROBES[pname]
            s = pfn(xtr, y[tr], xte)
            auroc, auprc = rap.metrics(yte, s)
            (al, ah), (pl, ph) = rap.bootstrap_ci(yte, s, n_boot, seed=42)
            preds[(g, pname)] = np.asarray(s, dtype=np.float64)

            rep_a, rep_p = [auroc], [auprc]
            for sd_ in seeds[1:]:
                tr2, te2 = rap.stratified_split(y, seed=sd_)
                xtr2, xte2 = rap.standardize(X[tr2], X[te2])
                s2 = pfn(xtr2, y[tr2], xte2)
                a2, p2 = rap.metrics(y[te2], s2)
                rep_a.append(a2); rep_p.append(p2)

            n_perm = n_perm_ladder(len(tr), pname)
            if perm_ys_full is None or perm_ys_full.shape[0] < n_perm:
                perm_ys_full = np.stack(
                    [rng.permutation(y) for _ in range(max(
                        n_perm, max(n_perm_ladder(len(tr), p) for p in PROBES)))])
            null_a, null_p = [], []
            if pname == "ridge":
                sc = ridge_null_batched(xtr, perm_ys_full[:n_perm][:, tr].T, xte)
                null_a = list(rap.auroc_cols(perm_ys_full[:n_perm][:, te].T, sc))
                from sklearn.metrics import average_precision_score
                null_p = [average_precision_score(perm_ys_full[j, te], sc[:, j])
                          for j in range(n_perm)]
            else:
                for j in range(n_perm):
                    sj = probe_linear_null(xtr, perm_ys_full[j, tr], xte) \
                        if n_perm < 1000 else pfn(xtr, perm_ys_full[j, tr], xte)
                    a_, p_ = rap.metrics(perm_ys_full[j, te], sj)
                    null_a.append(a_); null_p.append(p_)
            p_au = (1 + int(np.sum(np.asarray(null_a) >= auroc))) / (1 + n_perm)
            p_ap = (1 + int(np.sum(np.asarray(null_p) >= auprc))) / (1 + n_perm)

            rows.append(dict(
                dataset=ds, feature_group=g, probe=pname,
                role={"linear": "Readability", "ridge": "Stability"}[pname],
                is_supplementary=int(g == "ntv3_768_proj"),
                n_train=int(len(tr)), n_test=int(len(te)), prevalence=float(yte.mean()),
                primary_metric=TASK[ds][2],
                auroc=auroc, auroc_ci95=f"[{al:.4f},{ah:.4f}]",
                auprc=auprc, auprc_ci95=f"[{pl:.4f},{ph:.4f}]",
                auroc_seed_mean=float(np.mean(rep_a)), auroc_seed_sd=float(np.std(rep_a, ddof=1)),
                auprc_seed_mean=float(np.mean(rep_p)), auprc_seed_sd=float(np.std(rep_p, ddof=1)),
                null_auroc_mean=float(np.mean(null_a)), null_auroc_sd=float(np.std(null_a)),
                null_auroc_p=p_au, null_auprc_mean=float(np.mean(null_p)),
                null_auprc_sd=float(np.std(null_p)), null_auprc_p=p_ap,
                n_perm=n_perm, n_boot=n_boot, runtime_s=round(time.time() - t0, 1)))
            r = rows[-1]
            print(f"  {g:14s} {pname:7s} AUROC={auroc:.3f} AUPRC={auprc:.3f} "
                  f"(null {np.mean(null_a):.3f}, pA={p_au:.1e}, pP={p_ap:.1e}) "
                  f"[{r['runtime_s']}s]", flush=True)

    # ── paired DeLong (same seed42 test fold) ──
    for pname in PROBES:
        for ga, gb in DELONG_PAIRS:
            d_ = delong_test_full(yte, preds[(ga, pname)], preds[(gb, pname)])
            drows.append(dict(dataset=ds, probe=pname, group_a=ga, group_b=gb,
                              auc_a=d_["auc_a"], auc_b=d_["auc_b"],
                              delta_auc=d_["delta"], z=d_["z"], p=d_["p"],
                              n=d_["n"], n_pos=d_["n_pos"]))

    # ── dedup sensitivity (only datasets with dups>0; seed42 linear primary groups) ──
    if n_dup:
        kte = keep[te]
        for g in ("ntv3_1536", "kmer7_rc"):
            X = feats[g]
            xtr, xte = rap.standardize(X[tr], X[te])
            s = rap.PROBES["linear"](xtr, y[tr], xte)
            a_, p_ = rap.metrics(yte[kte], s[kte])
            rows.append(dict(
                dataset=ds, feature_group=g, probe="linear_dedup",
                role="Dedup sensitivity", is_supplementary=1,
                n_train=int(len(tr)), n_test=int(kte.sum()),
                prevalence=float(yte[kte].mean()), primary_metric=TASK[ds][2],
                auroc=a_, auroc_ci95="", auprc=p_, auprc_ci95="",
                auroc_seed_mean=a_, auroc_seed_sd=0.0,
                auprc_seed_mean=p_, auprc_seed_sd=0.0,
                null_auroc_mean=np.nan, null_auroc_sd=np.nan, null_auroc_p=np.nan,
                null_auprc_mean=np.nan, null_auprc_sd=np.nan, null_auprc_p=np.nan,
                n_perm=0, n_boot=0, runtime_s=0.0))
            print(f"  [dedup] {g}: AUROC={a_:.3f} AUPRC={p_:.3f} "
                  f"(n_test {len(te)}→{int(kte.sum())})", flush=True)
    return extra[ds]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--resume", action="store_true",
                    help="reuse rows of already-completed datasets from the existing CSV (checkpoint resume at dataset granularity)")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    names = DATASETS if args.datasets == "all" else args.datasets.split(",")

    import pandas as pd
    t0 = time.time()
    rows: list = []
    drows: list = []
    extra: dict = {}
    if args.resume:
        for path, sink, nm in ((CROSS / "crosstask_ntv3_results.csv", "rows", "results"),
                               (CROSS / "crosstask_ntv3_delong.csv", "drows", "delong")):
            if path.exists():
                old = pd.read_csv(path).to_dict("records")
                (rows if sink == "rows" else drows).extend(old)
                print(f"[probe] resume: loaded {len(old)} existing {nm} rows", flush=True)
    done = {r.get("dataset") for r in rows}

    for ds in names:
        if ds in done:
            print(f"[probe] {ds} has complete results, skipped (--resume); metadata only", flush=True)
            _collect_meta(ds, extra)
            continue
        run_dataset(ds, seeds, rows, drows, extra)
        # incremental flush (dataset-granularity checkpoint safety): rewrite the full CSV after each completed dataset
        pd.DataFrame(rows).drop_duplicates(
            subset=["dataset", "feature_group", "probe"], keep="last").to_csv(
            CROSS / "crosstask_ntv3_results.csv", index=False)
        pd.DataFrame(drows).drop_duplicates(
            subset=["dataset", "probe", "group_a", "group_b"], keep="last").to_csv(
            CROSS / "crosstask_ntv3_delong.csv", index=False)

    # metadata backfill (for datasets skipped on resume)
    for ds in names:
        if ds not in extra:
            _collect_meta(ds, extra)

    df = pd.DataFrame(rows).drop_duplicates(
        subset=["dataset", "feature_group", "probe"], keep="last")
    df.to_csv(CROSS / "crosstask_ntv3_results.csv", index=False)
    ddf = pd.DataFrame(drows).drop_duplicates(
        subset=["dataset", "probe", "group_a", "group_b"], keep="last")
    ddf.to_csv(CROSS / "crosstask_ntv3_delong.csv", index=False)
    print(f"\n[probe] done: {len(df)} rows, {len(ddf)} delong pairs, "
          f"{(time.time()-t0)/60:.1f} min -> crosstask_ntv3_results.csv")

    from crosstask_ntv3_report import write_report          # report module in the same directory
    write_report(CROSS, df, ddf, extra, time.time() - t0)


if __name__ == "__main__":
    main()
