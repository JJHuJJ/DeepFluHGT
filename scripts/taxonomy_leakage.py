#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Taxonomy leakage audit upgrade (full-volume cache)  -  formal version of the suspended RVM-6 item  -  DeepFluHGT.

Limitation of the step0 version (`scripts/shortcut_audit.py` ③): one sample per genome, only
30 samples / 9 test genomes in ≥5-sample groups → insufficient statistical power, conclusion suspended. This script re-runs on the full embedding cache:

  (a) seq_id → source_genome: `simulated_data/Contig_500k/train/provenance.tsv`
      takes each sample's **first** seg_type=HOST ∧ source_type=microbe row's source_file
      (GCF accession; same metric caliber as step0). Samples without a single host (chimeric / contaminated segments) are excluded and counted.
  (b) groups with ≥5 samples/genome → from evo_cache/Contig_500k/train.pt (400k × 4096,
      fp16) stratified sampling of ≤60k samples (seed42, memory-safe: peak <10 GB).
  (c) genome-level linear probe (embedding→genome-group multi-class, macro-F1):
        - G-full : all ≥5-sample groups (K≈9,000), equal per-group cap (cap c so that Σ≤60k)
                   → estimator = ridge closed-form solution (LogisticRegression /
                     LinearSVC OvR for K≈9000 at 6 CPU threads × 200 permutations is infeasible  —  noted);
                     permutation null n=200 batched (one factorization, per-permutation solve).
        - G-bal  : random 800 groups × cap 20 (seed4242, class-balanced) → primary estimator =
                   sklearn LogisticRegression (lbfgs multinomial; saga converges more slowly,
                   lbfgs chosen for speed  —  noted) obs + 200 permutation refits (if the 4h budget is
                   exceeded the null degrades to ridge, noted); ridge closed-form as a consistency cross-check.
  (d) phylum level: GCF→phylum local mapping (DeepFluHGT.8.0/audit_output/
      taxonomy_manifest.json  —  covers 9232/9232 training host genomes; genus→phylum
      approximation, ~74% Unknown → known 12-phylum subset) probed the same way (K=12, proportionally
      stratified ≤60k). If the manifest is unavailable, the genome-level result stands (noted).

  Feature preprocessing: normalization (train statistics) → PCA-512 (fit on train, randomized SVD)  —
  the probe remains linear in the original embedding space (linear∘linear), and the PCA rank constraint makes
  K classes × 200 permutations feasible on CPU; the explained variance ratio is reported.

Interpretation (RVM-6): (macro-F1 − null mean) ≤ 0.10 LOW / 0.10–0.25 MODERATE / > 0.25 HIGH.
Expected = the features encode lineage information (composition-adjacent framing, see report §3).

Output: docs/evidence/TAXONOMY_LEAKAGE_FULL.{json,md} + .EXIT marker
Run discipline: CPU-only (CUDA_VISIBLE_DEVICES=""), OMP/BLAS ≤6 threads, nohup background.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")          # CPU-only
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "6")                        # limit to 6 threads

import argparse
import csv
import gc
import json
import sys
import time
import traceback
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_PROV = REPO / "simulated_data" / "Contig_500k" / "train" / "provenance.tsv"
DEFAULT_CACHE = REPO / "evo_cache" / "Contig_500k" / "train.pt"
DEFAULT_MANIFEST = Path(
    "./DeepFluHGT.8.0/audit_output/"
    "taxonomy_manifest.json")
EVID = REPO / "docs" / "evidence"

UNKNOWN_PHYLA = {"Unknown", "Bacteria_Unknown", "Archaea_Unknown", "Eukarya_Unknown"}
RIDGE_ALPHA = 1.0            # same as step0 shortcut_audit
LR_TIME_BUDGET_S = 4 * 3600  # time budget for the LR permutation null (if exceeded → null uses ridge, noted)
N_TRAIN_SAMPLES = 400_000    # Contig_500k train split


# ══════════════════════════ (a) provenance → genome mapping ══════════════════════════ #

def first_host_genome(prov_path: Path) -> dict[str, str]:
    """seq_id → GCF accession of the first seg_type=HOST ∧ source_type=microbe row."""
    out: dict[str, str] = {}
    with open(prov_path) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            sid = r["seq_id"]
            if sid in out:
                continue
            if r.get("seg_type") == "HOST" and r.get("source_type") == "microbe":
                base = os.path.basename(r.get("source_file", "") or "")
                if base:
                    out[sid] = base.replace("_genomic.fna", "")
    return out


# ════════════════════ Estimators: closed-form ridge + sklearn LogisticRegression ════════════════════ #

def _std_train_test(xtr, xte):
    mu = xtr.mean(0, keepdims=True)
    sd = xtr.std(0) + 1e-12
    return (xtr - mu) / sd, (xte - mu) / sd


def ridge_obs(xtr, ytr, xte, alpha=RIDGE_ALPHA):
    """Ridge→one-hot argmax closed-form classifier (observed; same family as step0 ridge_clf_fit)."""
    a, b = _std_train_test(xtr, xte)
    classes = np.unique(ytr)
    Y = np.zeros((len(a), len(classes)), dtype=np.float64)
    Y[np.arange(len(a)), np.searchsorted(classes, ytr)] = 1.0
    W = np.linalg.solve(a.T @ a + alpha * np.eye(a.shape[1]), a.T @ Y)
    return classes[np.argmax(b @ W + Y.mean(0), 1)]


def ridge_null_batch(xtr, y_perms, xte, n_classes, alpha=RIDGE_ALPHA, tag="",
                     log=print):
    """Permutation null: normalize/factor the GRAM once, solve one-hot per permutation → [n_perm] prediction arrays.

    y_perms: [n_train, n_perm] permuted labels (each column an independent permutation, already sliced to train rows).
    The column space stays fixed at 0..n_classes-1: after permutation a class may fall entirely into test (absent from train) →
    that class's one-hot column is all 0, the ridge output scores it constantly low → argmax never selects it, and macro-F1 records 0
    (consistent with LR's behavior when train lacks a class; a self-consistent null caliber).
    """
    a, b = _std_train_test(xtr, xte)
    G = a.T @ a + alpha * np.eye(a.shape[1])
    chol = np.linalg.cholesky(G)
    K = int(n_classes)
    rows = np.arange(len(a))
    out = []
    Y = np.empty((len(a), K), dtype=np.float64)   # pre-allocate, avoiding GB-level churn per perm
    t0 = time.time()
    for j in range(y_perms.shape[1]):
        yc = y_perms[:, j]
        Y.fill(0.0)
        Y[rows, yc] = 1.0
        W = np.linalg.solve(chol.T, np.linalg.solve(chol, a.T @ Y))
        out.append(np.argmax(b @ W + Y.mean(0), 1))
        if (j + 1) % 20 == 0:
            el = time.time() - t0
            log(f"[{tag}] ridge null {j+1}/{y_perms.shape[1]} ({el/(j+1):.1f}s/perm, "
                f"ETA {(y_perms.shape[1]-j-1)*el/(j+1)/60:.0f}min)", flush=True)
    return out


def macro_f1(y_true, y_pred, n_classes):
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, labels=np.arange(n_classes),
                          average="macro", zero_division=0))


def _lr():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=1.0, solver="lbfgs", max_iter=200, tol=1e-3)


# ══════════════════════════ Single probe run ══════════════════════════ #

def run_probe(name: str, X: np.ndarray, y: np.ndarray, n_perm: int, pca_dim: int,
              lr_primary: bool, log=print) -> dict:
    """X float32 [n,4096]; normalization+PCA (fit on train) → 70/30 stratified split → probe + null.

    lr_primary=True: sklearn LogisticRegression (lbfgs multinomial) as the primary estimator
    (obs + per-permutation refit; the first few fits are timed, and if LR_TIME_BUDGET_S is
    exceeded the null degrades to ridge, noted  —  the p-value caliber automatically rolls back
    to ridge obs vs ridge null).
    """
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split

    t0 = time.time()
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.30, stratify=y, random_state=42)
    n_classes = int(y.max()) + 1

    mu = X[tr].mean(0, keepdims=True)
    sd = X[tr].std(0) + 1e-12
    pca = PCA(n_components=min(pca_dim, X.shape[1], len(tr) - 1),
              svd_solver="randomized", random_state=42)
    pca.fit((X - mu) / sd)
    Z = pca.transform((X - mu) / sd).astype(np.float64)
    ztr, zte = Z[tr], Z[te]
    ytr, yte = y[tr], y[te]
    res = {
        "n_samples": int(len(y)), "n_train": int(len(tr)), "n_test": int(len(te)),
        "n_classes": n_classes, "pca_dim": int(Z.shape[1]),
        "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
        "split": "70/30 stratified (random_state=42); normalization+PCA both fit on train",
    }
    maj = np.bincount(ytr, minlength=n_classes).argmax()
    res["majority_baseline_macro_f1"] = macro_f1(
        yte, np.full_like(yte, maj), n_classes)

    # ── ridge (closed-form): obs + batched permutation null ──
    t1 = time.time()
    res["ridge_macro_f1"] = macro_f1(yte, ridge_obs(ztr, ytr, zte), n_classes)
    log(f"[{name}] ridge obs macroF1={res['ridge_macro_f1']:.4f} "
        f"({time.time()-t1:.0f}s; K={n_classes}, d={Z.shape[1]}, "
        f"PCA var={res['pca_variance_explained']:.3f})", flush=True)
    t1 = time.time()
    rng = np.random.default_rng(42)
    perms = np.stack([rng.permutation(y) for _ in range(n_perm)])  # [m, n]
    null_preds = ridge_null_batch(ztr, perms[:, tr].T, zte, n_classes,
                                  tag=name, log=log)
    null_r = np.array([macro_f1(perms[j][te], null_preds[j], n_classes)
                       for j in range(n_perm)])
    res["ridge_null_macro_f1_mean"] = float(null_r.mean())
    res["ridge_null_macro_f1_sd"] = float(null_r.std())
    res["ridge_null_macro_f1_p95"] = float(np.percentile(null_r, 95))
    res["ridge_perm_p_value"] = float(
        (1 + int((null_r >= res["ridge_macro_f1"]).sum())) / (1 + n_perm))
    res["ridge_null_runtime_s"] = round(time.time() - t1, 1)
    log(f"[{name}] ridge null: {res['ridge_null_macro_f1_mean']:.4f}±"
        f"{res['ridge_null_macro_f1_sd']:.4f} "
        f"({res['ridge_null_runtime_s']}s, {n_perm} perms)", flush=True)

    # ── sklearn LogisticRegression (multinomial lbfgs; depends on K and speed) ──
    lr_res: dict = {"status": "SKIPPED"}
    if lr_primary:
        t1 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = _lr().fit(ztr, ytr)
        lr_res = {
            "status": "OK",
            "estimator": "sklearn LogisticRegression(C=1, lbfgs multinomial, "
                         "max_iter=200, tol=1e-3; standardized+PCA feature)",
            "solver_note": "saga and lbfgs are both solvers for multinomial L2; saga converges"
                           "more slowly, so lbfgs was chosen for speed (LinearSVC is OvR with K binary"
                           "classifications, slower at this scale; not used)",
            "fit_time_s": round(time.time() - t1, 1),
            "n_iter": int(np.max(clf.n_iter_)),
            "macro_f1": macro_f1(yte, clf.predict(zte), n_classes),
        }
        log(f"[{name}] LR obs macroF1={lr_res['macro_f1']:.4f} "
            f"(fit {lr_res['fit_time_s']}s, n_iter={lr_res['n_iter']})", flush=True)
        # permutation null: run 3 first to measure speed; if the projection exceeds the budget → degrade to ridge (noted)
        t1 = time.time()
        probe_n = min(3, n_perm)
        warm = []
        for j in range(probe_n):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cj = _lr().fit(ztr, perms[j][tr])
            warm.append(cj)
        s_per = (time.time() - t1) / probe_n
        if s_per * n_perm <= LR_TIME_BUDGET_S:
            null_l = np.array([macro_f1(perms[j][te], cj.predict(zte), n_classes)
                               for j, cj in enumerate(warm)])
            t1 = time.time()
            for j in range(probe_n, n_perm):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cj = _lr().fit(ztr, perms[j][tr])
                null_l = np.append(
                    null_l, macro_f1(perms[j][te], cj.predict(zte), n_classes))
                if (j + 1) % 10 == 0:
                    el = time.time() - t1
                    log(f"[{name}] LR null {j+1}/{n_perm} ({el/(j+1-probe_n):.1f}s/"
                        f"perm, ETA {(n_perm-j-1)*el/(j+1-probe_n)/60:.0f}min)",
                        flush=True)
            lr_res["null_macro_f1_mean"] = float(null_l.mean())
            lr_res["null_macro_f1_sd"] = float(null_l.std())
            lr_res["null_macro_f1_p95"] = float(np.percentile(null_l, 95))
            lr_res["perm_p_value"] = float(
                (1 + int((null_l >= lr_res["macro_f1"]).sum())) / (1 + n_perm))
        else:
            lr_res["null_note"] = (
                f"measured LR permutation refit {s_per:.1f}s/fit × {n_perm} = "
                f"{s_per*n_perm/3600:.1f}h > budget {LR_TIME_BUDGET_S/3600:.0f}h → "
                "the significance test uses the closed-form ridge (obs+null same estimator); LR reports obs only")
            log(f"[{name}] LR null degraded to ridge ({s_per:.1f}s/perm over budget)", flush=True)
    res["logistic_regression"] = lr_res
    res["runtime_s"] = round(time.time() - t0, 1)
    log(f"[{name}] done {res['runtime_s']}s", flush=True)
    return res


def primary_stat(r: dict) -> tuple:
    """(obs, null_mean, null_sd, p, estimator_label)  —  uses the full LR set when the LR null is available,
    else ridge. Guarantees obs and null come from **the same estimator** (mixing is invalid)."""
    lr = r.get("logistic_regression", {})
    if lr.get("status") == "OK" and "null_macro_f1_mean" in lr:
        return (lr["macro_f1"], lr["null_macro_f1_mean"], lr["null_macro_f1_sd"],
                lr["perm_p_value"], "LogisticRegression lbfgs multinomial")
    return (r["ridge_macro_f1"], r["ridge_null_macro_f1_mean"],
            r["ridge_null_macro_f1_sd"], r["ridge_perm_p_value"],
            f"ridge α={RIDGE_ALPHA} (closed-form)")


# ══════════════════════════════ main pipeline ══════════════════════════════ #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prov", type=Path, default=DEFAULT_PROV)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--pca-dim", type=int, default=512)
    ap.add_argument("--max-samples", type=int, default=60_000)
    ap.add_argument("--min-group", type=int, default=5)
    ap.add_argument("--n-groups-bal", type=int, default=800)
    ap.add_argument("--cap-bal", type=int, default=20)
    args = ap.parse_args()
    t0 = time.time()

    # ── (a) genome mapping + per-group sample lists (one pass) ──
    host = first_host_genome(args.prov)
    by_group: dict[str, list[str]] = defaultdict(list)
    for sid, g in host.items():
        by_group[g].append(sid)
    for g in by_group:
        by_group[g].sort()                      # deterministic
    counts = {g: len(v) for g, v in by_group.items()}
    eligible = sorted(g for g, n in counts.items() if n >= args.min_group)
    gcode = {g: i for i, g in enumerate(eligible)}
    print(f"[prov] train samples {N_TRAIN_SAMPLES:,}; with a first HOST row {len(host):,} "
          f"(excluded for no single host: {N_TRAIN_SAMPLES - len(host):,}); unique genomes "
          f"{len(counts):,}; ≥{args.min_group}-sample groups {len(eligible):,} covering "
          f"{sum(counts[g] for g in eligible):,} samples", flush=True)

    # ── (d) phylum mapping ──
    phyl: dict[str, str] = {}
    man_meta = {"path": str(args.manifest), "exists": args.manifest.exists()}
    if args.manifest.exists():
        man = json.loads(args.manifest.read_text())["per_genome"]
        phyl = {k.replace("_genomic.fna", ""): v["phylum"] for k, v in man.items()}
        man_meta["n_genomes"] = len(phyl)
        man_meta["coverage_of_train_host_genomes"] = int(
            sum(1 for g in counts if g in phyl))
        man_meta["note"] = ("DeepFluHGT.8.0 local taxonomy manifest (FASTA header "
                            "organism → genus→phylum approximation); Unknown bucket (~74%) excluded, "
                            "phylum conclusions limited to the known subset")
    else:
        print(f"[WARN] taxonomy manifest does not exist: {args.manifest} → phylum probe "
              f"skipped; the genome-level result stands", flush=True)

    # ── (b) sampling design ──
    # G-full: equal per-group cap (largest c with Σ min(n_g,c) ≤ max_samples)
    cap = 1
    for c in range(1, 60):
        if sum(min(counts[g], c) for g in eligible) <= args.max_samples:
            cap = c
        else:
            break
    if cap < 3:
        raise SystemExit(
            f"[draw] G-full cap={cap} <3 (max_samples={args.max_samples} too small; "
            f"a stratified split needs ≥2 per class)  —  increase --max-samples or --min-group")
    rng_full = np.random.default_rng(42)
    ids_full, y_full = [], []
    for g in eligible:
        pool = by_group[g]
        for i in rng_full.choice(len(pool), size=min(len(pool), cap), replace=False):
            ids_full.append(pool[i]); y_full.append(gcode[g])
    y_full = np.array(y_full)
    print(f"[draw] G-full: {len(eligible):,} group × cap≤{cap} → n={len(ids_full):,}",
          flush=True)

    # G-bal: random 800 groups × cap 20 (class-balanced)
    rng_bal = np.random.default_rng(4242)
    sel = sorted(int(g) for g in rng_bal.choice(
        len(eligible), size=min(args.n_groups_bal, len(eligible)), replace=False))
    ids_bal, y_bal = [], []
    for di, gi in enumerate(sel):      # di = dense class code 0..K-1 (using the sparse
        pool = by_group[eligible[gi]]  # gi≤8987 directly would make macro-F1 divide by max+1≈8988,
        for i in rng_bal.choice(len(pool), size=min(len(pool), args.cap_bal),  # systematically
                                replace=False):                                # deflating it ~11×
            ids_bal.append(pool[i]); y_bal.append(di)
    y_bal = np.array(y_bal)
    print(f"[draw] G-bal: {len(sel)} group × cap≤{args.cap_bal} → n={len(ids_bal):,}",
          flush=True)

    # phylum: eligible samples with known phylum, proportionally stratified ≤ max_samples
    ids_ph: list[str] = []
    y_ph = np.array([], dtype=int)
    phyla: list[str] = []
    if phyl:
        rng_ph = np.random.default_rng(424242)
        per: dict[str, list[str]] = defaultdict(list)
        for g in eligible:
            p = phyl.get(g)
            if p and p not in UNKNOWN_PHYLA:
                per[p].extend(by_group[g])
        phyla = sorted(per)
        total_known = sum(len(v) for v in per.values())
        ys = []
        for pi, p in enumerate(phyla):
            pool = sorted(per[p])
            k = len(pool) if total_known <= args.max_samples else max(
                1, int(round(args.max_samples * len(pool) / total_known)))
            k = min(k, len(pool))
            for i in rng_ph.choice(len(pool), size=k, replace=False):
                ids_ph.append(pool[i]); ys.append(pi)
        y_ph = np.array(ys)
        # small-class guard: train_test_split(stratify) needs ≥2 per class; excluding <5-sample classes is consistent with the min-group caliber
        cnt_ph = Counter(ys)
        keep_cls = [c for c in range(len(phyla)) if cnt_ph[c] >= 5]
        if len(keep_cls) < len(phyla):
            print(f"[draw] phylum: after proportional stratification, "
                  f"{len(phyla) - len(keep_cls)} classes with <5 samples excluded (stratify guard)", flush=True)
        remap = {o: n for n, o in enumerate(keep_cls)}
        phyla = [phyla[c] for c in keep_cls]
        pairs = [(s, remap[c]) for s, c in zip(ids_ph, ys) if c in remap]
        ids_ph = [s for s, _ in pairs]
        y_ph = np.array([c for _, c in pairs])
        print(f"[draw] phylum: {len(phyla)} phyla, known-phylum eligible samples "
              f"{total_known:,} → n={len(ids_ph):,}", flush=True)

    # ── embedding extraction (cache loaded once; the three matrices total <2.5 GB) ──
    import torch
    print(f"[cache] loading {args.cache} ...", flush=True)
    t1 = time.time()
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    print(f"[cache] {len(cache):,} vectors ({time.time()-t1:.0f}s)", flush=True)

    def extract(ids, y):
        keep = [i for i, s in enumerate(ids) if s in cache]
        if len(keep) != len(ids):
            print(f"[WARN] {len(ids)-len(keep)} samples not in cache (excluded)", flush=True)
        M = np.stack([np.asarray(cache[ids[i]], dtype=np.float32) for i in keep])
        return M, y[keep], len(keep)

    X_full, y_full, n_full = extract(ids_full, y_full)
    X_bal, y_bal, n_bal = extract(ids_bal, y_bal)
    X_ph, n_ph = None, 0
    if phyla and len(ids_ph) > 100:
        X_ph, y_ph, n_ph = extract(ids_ph, y_ph)
    meta = {
        "prov": str(args.prov), "cache": str(args.cache),
        "cache_n": len(cache), "embedding": "Evo2 7B contig-mean 4096d (fp16→f32)",
        "min_group": args.min_group, "n_perm": args.n_perm, "seed": 42,
        "max_samples": args.max_samples,
        "n_samples_no_host_dropped": N_TRAIN_SAMPLES - len(host),
        "host_mapping_rule": "first seg_type=HOST ∧ source_type=microbe row "
                             "(multi_source_host samples take the first; documented choice)",
        "taxonomy_manifest": man_meta,
    }
    del cache
    gc.collect()

    runs: dict = {}

    print("[probe] G-full (closed-form ridge; LR/LinearSVC infeasible for K≈9000  —  noted)",
          flush=True)
    runs["genome_full_all_groups"] = run_probe(
        "G-full", X_full, y_full, args.n_perm, args.pca_dim, lr_primary=False)
    runs["genome_full_all_groups"]["design"] = (
        f"all ≥{args.min_group}-sample groups (K={len(eligible)}), equal per-group cap≤{cap} "
        f"(Σ≤{args.max_samples}); estimator closed-form ridge (α={RIDGE_ALPHA})  —  "
        "LogisticRegression (saga/lbfgs multinomial) and LinearSVC for K≈9000 classes "
        "(OvR, K binary classifications) at 6 CPU threads × 200 permutations is infeasible (dropped for speed, noted)")
    del X_full
    gc.collect()

    print("[probe] G-bal (LR lbfgs multinomial primary estimator)", flush=True)
    runs["genome_balanced_800"] = run_probe(
        "G-bal", X_bal, y_bal, args.n_perm, args.pca_dim, lr_primary=True)
    runs["genome_balanced_800"]["design"] = (
        f"random {args.n_groups_bal} groups (seed4242) × ≤{args.cap_bal} samples per group (class-balanced); "
        "primary estimator sklearn LogisticRegression lbfgs multinomial + per-permutation refit; "
        f"closed-form ridge (α={RIDGE_ALPHA}) obs+null full set as cross-check")
    del X_bal
    gc.collect()

    # ── (d) phylum level probe ──
    if X_ph is not None and len(np.unique(y_ph)) < 2:
        print(f"[draw] phylum: only {len(np.unique(y_ph))} classes after filtering → SKIPPED",
              flush=True)
        X_ph = None
    if X_ph is not None:
        runs["phylum_level"] = run_probe(
            "phylum", X_ph, y_ph, args.n_perm, args.pca_dim, lr_primary=True)
        runs["phylum_level"]["design"] = (
            f"{len(phyla)} known-phylum classes (Unknown bucket excluded), eligible samples proportionally stratified "
            f"≤{args.max_samples}; same method, LR (primary) + ridge (cross-check)")
        runs["phylum_level"]["phylum_counts"] = {
            p: int((y_ph == i).sum()) for i, p in enumerate(phyla)}
        del X_ph
        gc.collect()
    else:
        runs["phylum_level"] = {"status": "SKIPPED",
                                "reason": "manifest missing or too few known samples"}

    # ── RVM-6 interpretation (obs and null from the same estimator) ──
    verdict = {}
    for k, r in runs.items():
        if "n_classes" not in r:
            continue
        obs, null_mean, null_sd, p, est = primary_stat(r)
        delta = obs - null_mean
        verdict[k] = {
            "estimator": est,
            "obs_macro_f1": obs, "null_mean": null_mean, "null_sd": null_sd,
            "delta": round(float(delta), 4),
            "rvm6_level": "LOW" if delta <= 0.10 else
                          ("MODERATE" if delta <= 0.25 else "HIGH"),
            "perm_p": p,
            "significant_above_null": bool(p < 0.05),
        }

    out = {"meta": meta, "runs": runs, "rvm6_verdict": verdict,
           "interpretation": (
               "macro-F1 significantly above the permutation null → the features encode lineage (genome-identity) information. This information"
               "is highly correlated with GC/k-mer composition (composition-adjacent): elevated genome-level macro-F1 "
               "is the expected behavior of a genome-foundation-model embedding and does not constitute an HGT-label leakage channel (the probe "
               "targets host-genome ID, not the HGT label); the result is used for the RVM-6 feature-attribution row and provides "
               "interpretive background for the M3/shuffle control."),
           "runtime_s": round(time.time() - t0, 1)}
    EVID.mkdir(parents=True, exist_ok=True)
    (EVID / "TAXONOMY_LEAKAGE_FULL.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    _write_md(EVID / "TAXONOMY_LEAKAGE_FULL.md", out, eligible, counts, cap, args)
    (EVID / "TAXONOMY_LEAKAGE_FULL.EXIT").write_text(
        f"DONE {time.strftime('%Y-%m-%d %H:%M:%S')} runtime={out['runtime_s']}s\n")
    print(f"[tax] done -> {EVID/'TAXONOMY_LEAKAGE_FULL.md'} ({out['runtime_s']}s)",
          flush=True)


def _write_md(path: Path, d: dict, eligible, counts, cap, args) -> None:
    L = []
    A = L.append
    A("# Taxonomy leakage audit  -  formal full-cache version (closing the suspended RVM-6 item)\n")
    A(f"> generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  ·  DeepFluHGT  ·  CPU 6 threads "
      f" ·  nohup background  ·  runtime {d['runtime_s']}s")
    A(f"> cache: `{d['meta']['cache']}`  —  {d['meta']['cache_n']:,} × 4096 "
      f"({d['meta']['embedding']})")
    A(f"> host mapping: {d['meta']['host_mapping_rule']}; samples without a single host excluded: "
      f"{d['meta']['n_samples_no_host_dropped']:,}")
    A(f"> null: label permutation n={d['meta']['n_perm']} (same pipeline refit after permutation, "
      f"70/30 split and PCA fixed); interpretation RVM-6: Δ≤0.10 LOW / 0.10–0.25 MODERATE / "
      f">0.25 HIGH\n")

    A("## 0. Power upgrade (vs step0 shortcut_audit ③)\n")
    A("| | step0 (1,996 subset) | this audit (400k full cache) |")
    A("|---|---|---|")
    A("| coverage of ≥5-sample groups | 30 samples / 9 test (underpowered, suspended) | "
      f"{sum(counts[g] for g in eligible):,} eligible samples → sampling below |")
    A("| probe target | host GCF group | host GCF group (G-full/G-bal) + phylum level |\n")

    A("## 1. Genome-level probe (embedding → host-genome GCF group)\n")
    A("| run | design | n (tr/te) | K | primary estimator | macro-F1 | null mean±SD | Δ | perm p | RVM-6 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for k in ("genome_full_all_groups", "genome_balanced_800"):
        r = d["runs"][k]
        v = d["rvm6_verdict"][k]
        A(f"| {k} | equal cap≤{cap} / {args.n_groups_bal} groups×≤{args.cap_bal} | "
          f"{r['n_samples']:,} ({r['n_train']:,}/{r['n_test']:,}) | "
          f"{r['n_classes']:,} | {v['estimator']} | **{v['obs_macro_f1']:.4f}** | "
          f"{v['null_mean']:.4f}±{v['null_sd']:.4f} | "
          f"{v['delta']:+.3f} | {v['perm_p']:.4f} | **{v['rvm6_level']}** |")
    A("")
    for k in ("genome_full_all_groups", "genome_balanced_800"):
        r = d["runs"][k]
        A(f"- **{k}**: PCA-{r['pca_dim']} (variance explained "
          f"{r['pca_variance_explained']:.3f}); majority baseline macro-F1="
          f"{r['majority_baseline_macro_f1']:.4f}; ridge obs="
          f"{r['ridge_macro_f1']:.4f} (null {r['ridge_null_macro_f1_mean']:.4f}, "
          f"p={r['ridge_perm_p_value']:.4f})")
        lr = r.get("logistic_regression", {})
        if lr.get("status") == "OK":
            extra = (f"; LR obs={lr['macro_f1']:.4f} (fit {lr['fit_time_s']}s, "
                     f"n_iter={lr['n_iter']})")
            if "null_macro_f1_mean" in lr:
                extra += (f", LR null={lr['null_macro_f1_mean']:.4f}, "
                          f"p={lr['perm_p_value']:.4f}")
            else:
                extra += f"; {lr.get('null_note', '')}"
            A(extra)
        A("")
    A("")

    ph = d["runs"].get("phylum_level", {})
    A("## 2. Phylum-level probe (same method)\n")
    if ph.get("status") != "SKIPPED":
        v = d["rvm6_verdict"]["phylum_level"]
        A(f"- design: {ph['design']}")
        A(f"- n={ph['n_samples']:,} ({ph['n_train']:,}/{ph['n_test']:,}), "
          f"K={ph['n_classes']}, PCA-{ph['pca_dim']} (variance explained "
          f"{ph['pca_variance_explained']:.3f}); majority baseline "
          f"{ph['majority_baseline_macro_f1']:.4f}")
        A(f"- primary estimator {v['estimator']}: macro-F1 = **{v['obs_macro_f1']:.4f}** "
          f"(null {v['null_mean']:.4f}, p={v['perm_p']:.4f}) → Δ="
          f"{v['delta']:+.3f} → **{v['rvm6_level']}**")
        A(f"- ridge cross-check: obs={ph['ridge_macro_f1']:.4f} "
          f"(null {ph['ridge_null_macro_f1_mean']:.4f}, "
          f"p={ph['ridge_perm_p_value']:.4f})")
        A("- phylum sample counts: " + ", ".join(
            f"{k}={n}" for k, n in list(ph.get("phylum_counts", {}).items())))
        mm = d["meta"]["taxonomy_manifest"]
        A(f"- manifest: {mm.get('note', '')} (covers training host genomes "
          f"{mm.get('coverage_of_train_host_genomes')}/{len(counts)})\n")
    else:
        A(f"- SKIPPED: {ph.get('reason')}  —  —  **the genome-level result stands**.\n")

    A("## 3. interpretation\n")
    for k, v in d["rvm6_verdict"].items():
        A(f"- **{k}**: obs={v['obs_macro_f1']:.4f} vs null={v['null_mean']:.4f} "
          f"(Δ={v['delta']:+.3f}, p={v['perm_p']:.4f}, {v['estimator']}) → "
          f"**{v['rvm6_level']}**"
          + ("  —  significantly above null: the features encode lineage information" if v["significant_above_null"]
             else ""))
    A("")
    A(f"> framing: {d['interpretation']}")
    bal = d["rvm6_verdict"].get("genome_balanced_800", {})
    A(f"> **RVM-6 table row values** (primary row = G-bal, LR caliber): macro-F1="
      f"{bal.get('obs_macro_f1', float('nan')):.4f}, null="
      f"{bal.get('null_mean', float('nan')):.4f} (Δ="
      f"{bal.get('delta', float('nan')):+.3f}, p={bal.get('perm_p', float('nan')):.4f})"
      f" → {bal.get('rvm6_level', '?')}")
    A(">")
    A("> values: `TAXONOMY_LEAKAGE_FULL.json`; predecessor: step0 "
      "`evo_cache/step0_subset/rap/shortcut_audit.md` (suspended as underpowered; this file closes it out).")
    Path(path).write_text("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()
        print(err, file=sys.stderr, flush=True)
        EVID.mkdir(parents=True, exist_ok=True)
        (EVID / "TAXONOMY_LEAKAGE_FULL.EXIT").write_text(
            f"ERROR {time.strftime('%Y-%m-%d %H:%M:%S')}\n{err}\n")
        raise
