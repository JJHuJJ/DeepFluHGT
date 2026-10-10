#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Composition-baseline ceiling measurement on simulated data (Composition Ceiling)  -  DeepFluHGT [literature application plan #5 / R5]

Scientific question
--------
On the 400k samples of Contig_500k/train, train linear / ridge / MLP three-level probes on
**full-dimension kmer7 (4^7 = 16384, no RC folding) + log10(len) + GC** composition features over the
full volume, obtaining the "theoretical ceiling of composition features on simulated data" (composition ceiling).

Pre-registered interpretation rules (registered before running):
  - If the composition ceiling (best probe, same-fold AUROC) is **< 0.87** (M2 ep2 sim val AUPRC 0.8711):
    composition features are not separable while FM features are → proves FM features carry information beyond composition;
  - If the composition ceiling is **close to (≥) 0.87**: composition explanatory power must be taken seriously (the LAMBDA-warned failure mode),
    and the paper narrative is restated under the "composition+FM hybrid ceiling" caliber.
  Reference system (calibers noted; different splits are not directly comparable, used only for order-of-magnitude triangulation):
    M2 ep2 sim val AUPRC 0.8711 (official val split; [training in progress, terminal value pending])
    M0 terminal sim val AUPRC 0.8325 / test 0.827
    Same-fold (70/30 within train, seed42, fully consistent with RAP_FULL_400K) precedents:
      evo MLP AUROC 0.8862 / AUPRC 0.8865; evo linear 0.7775; kmer7_rc (RC-folded) linear 0.6277
  This script shares with rap_full.py the **same stratified_split(y, seed=42) and the same fasta order**
  → the test fold is fully consistent with RAP_FULL → run a **same-fold paired DeLong** against its existing predictions
  (composition vs evo / kmer7_rc / len+gc / evo+len+gc, per probe).

Implementation
----
  features  : full-dimension kmer7, 16384 relative frequencies (memmap written directly by workers) ⊕ log10(len) ⊕ GC = 16386 dims
  probes: Linear = LogisticRegression(C=1, lbfgs tol=1e-3)        [Readability]
          Ridge  = closed-form primal, **chunked float64 Gram** (16386² accumulation) [Stability]
          MLP    = 2×1024 (D→1024→1024→1, mini-batch Adam+early stopping)   [Capacity ceiling]
  protocol  : normalization (train statistics, in place) + bootstrap 1000 CI + label-permutation null
          (ridge=closed-form exact batched; linear/MLP=wall-clock-budget early stopping, actual n_perm recorded in the CSV)
  memory  : kmer memmap 26GB (disk) → X float32 copy 26GB (RAM, in-place normalization);
          ridge chunked float64 (chunk 2.6GB), never materializing the full float64;
          peak RSS ≈ 55GB (linear's X[tr]/X[te] copies), machine has 251GB.

output
----
  evo_cache/Contig_500k/composition_ceiling/composition_ceiling_results.csv
  evo_cache/Contig_500k/composition_ceiling/composition_ceiling_delong.csv
  docs/evidence/COMPOSITION_CEILING_SIM.md
  docs/evidence/COMPOSITION_CEILING_SIM.EXIT   (nohup completion marker)

usage (nohup, see scripts/run/composition_ceiling.sh):
  nohup python scripts/composition_ceiling.py \
      > logs/composition_ceiling.log 2>&1 &

Run discipline: CPU-only (CUDA_VISIBLE_DEVICES=""), ≤48 threads in any stage,
never touches GPUs (M2/M3 production training is running), read-only simulated_data (no writes).
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")          # CPU-only
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "16")

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import rap_probes as rap                    # already-validated fasta/statistics/metric implementation
import rap_full as rfp                      # stream_fasta_stats / mlp / linear / chol
from delong_test import delong_test_full

DEFAULT_FASTA = REPO / "simulated_data" / "Contig_500k" / "train" / "dataset.fasta"
DEFAULT_RAP_FULL = REPO / "evo_cache" / "Contig_500k" / "rap_full"
DEFAULT_OUT_DIR = REPO / "evo_cache" / "Contig_500k" / "composition_ceiling"
DEFAULT_MD = REPO / "docs" / "evidence" / "COMPOSITION_CEILING_SIM.md"
EXIT_MARK = REPO / "docs" / "evidence" / "COMPOSITION_CEILING_SIM.EXIT"

KMER_K = 7
N_BUCKET = 4 ** KMER_K                          # 16384 (full dimensions, no RC folding)
D_EXTRA = 2                                     # log10(len) + GC
GROUP_NAME = "kmer7full+len+gc"
RAP_GROUPS = ["evo", "len+gc", "kmer7_rc", "evo+len+gc"]     # DeLong controlgroup
NULL_WALLCLOCK = {"linear": 3600.0, "mlp": 1800.0}
RIDGE_CHUNK = 20_000                            # ridge chunk rows (~2.6GB float64)

# Reference numbers (for report interpretation; see "Pre-registered interpretation rules" in the file header)
REF = {"M2_ep2_val_auprc": 0.8711, "M0_val_auprc": 0.8325,
       "evo_mlp_same_fold_auroc": 0.8862, "evo_linear_same_fold_auroc": 0.7775,
       "kmer7rc_linear_same_fold_auroc": 0.6277}


# ═══════════════════════════ Full-dimension kmer (no RC) ═══════════════════════════ #

def _kmer_full_batch(task):
    """joblib worker: a batch of sequences → full-dim kmer7 (4^7=16384) relative frequencies written directly into a shared memmap."""
    from numpy.lib.stride_tricks import sliding_window_view
    path, n_rows, n_bucket, row0, seqs = task
    k = KMER_K
    lut = np.full(256, -1, dtype=np.int64)
    for b, i in rap._NUC2IDX.items():
        lut[ord(b)] = i
    pow4 = rap._POW4
    mm = np.memmap(path, dtype=np.float32, mode="r+", shape=(n_rows, n_bucket))
    for j, seq in enumerate(seqs):
        idx = lut[np.frombuffer(seq.encode(), dtype=np.uint8)]
        if len(idx) < k:
            continue
        win = sliding_window_view(idx, k)
        valid = (win >= 0).all(axis=1)
        wv = np.where(win[valid] >= 0, win[valid], 0)
        counts = np.bincount(wv @ pow4, minlength=n_bucket).astype(np.float32)
        tot = counts.sum()
        mm[row0 + j] = counts / tot if tot > 0 else 0.0
    mm.flush()
    return len(seqs)


def build_kmer_full_memmap(fasta: Path, ids: list[str], out_bin: Path, n_jobs: int):
    """Batches of 4000 compute full-dim kmer7 → workers write the raw-binary memmap directly (row number = fasta order)."""
    n = len(ids)
    row_of = {sid: i for i, sid in enumerate(ids)}
    batches, buf_ids, buf_seqs = [], [], []
    with open(fasta) as fh:
        name, chunks = None, []
        for line in fh:
            if line.startswith(">"):
                if name is not None and name in row_of:
                    if len(buf_seqs) >= 4000:
                        batches.append((buf_ids, buf_seqs)); buf_ids, buf_seqs = [], []
                    buf_ids.append(name); buf_seqs.append("".join(chunks))
                name = line[1:].strip().split()[0]; chunks = []
            else:
                chunks.append(line.strip().upper())
        if name is not None and name in row_of:
            buf_ids.append(name); buf_seqs.append("".join(chunks))
    if buf_seqs:
        batches.append((buf_ids, buf_seqs))

    tasks = []
    for bid, bseq in batches:
        row0 = row_of[bid[0]]
        assert all(row_of[s] == row0 + j for j, s in enumerate(bid))
        tasks.append((str(out_bin), n, N_BUCKET, row0, bseq))

    from joblib import Parallel, delayed
    saved = {v: os.environ.get(v) for v in
             ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"                       # pin to 1 thread before spawn (rap_full precedent)
    t0 = time.time()
    try:
        res = Parallel(n_jobs=n_jobs)(delayed(_kmer_full_batch)(t) for t in tasks)
    finally:
        for v, val in saved.items():
            if val is not None:
                os.environ[v] = val
    print(f"[kmer] {int(np.sum(res))} seqs → {out_bin.name} "
          f"({time.time()-t0:.0f}s, {n_jobs} workers, {N_BUCKET} buckets full-dim)",
          flush=True)


# ═══════════════════════════ Chunked ridge ═══════════════════════════ #

def _ridge_std_chunk(X, rows):
    """Row chunk → float64 (X is already globally normalized; only precision is converted here; chunking avoids materializing 400k×d float64)."""
    return np.asarray(X[rows], dtype=np.float64)


def ridge_chunked_fit(X, tr, te, ytr, alpha: float = 1.0):
    """Primal ridge, Gram accumulated chunk-wise in float64. Returns (test scores, solve function, Gram).

    solve(rhs) accepts an m-column rhs for batch solving → null permutations reuse the same factorization (rap_full protocol).
    """
    d = X.shape[1]
    G = np.zeros((d, d), dtype=np.float64)
    yc = ytr.astype(np.float64) - ytr.mean()
    rhs = np.zeros(d, dtype=np.float64)
    for i0 in range(0, len(tr), RIDGE_CHUNK):
        i1 = min(i0 + RIDGE_CHUNK, len(tr))
        a = _ridge_std_chunk(X, tr[i0:i1])
        G += a.T @ a
        rhs += a.T @ yc[i0:i1]
    G[np.diag_indices(d)] += alpha
    chol = rfp._chol_robust(G)                    # reuse rap_full's jitter fallback
    if chol is not None:
        def solve_(rhs_):
            return np.linalg.solve(chol.T, np.linalg.solve(chol, rhs_))
    else:
        print("[ridge] Cholesky failure (after jitter cap), falling back to LU", flush=True)

        def solve_(rhs_):
            return np.linalg.solve(G, rhs_)
    w = solve_(rhs)
    out = np.empty(len(te), dtype=np.float64)
    for i0 in range(0, len(te), RIDGE_CHUNK):
        i1 = min(i0 + RIDGE_CHUNK, len(te))
        out[i0:i1] = _ridge_std_chunk(X, te[i0:i1]) @ w
    return out + ytr.mean(), solve_, G


def ridge_chunked_null(solve_, X, tr, te, ytr_perm_2d):
    """Closed-form batched null: ytr_perm_2d [m, n_tr] → [n_te, m] (reuses the primary fit's factorization, exact)."""
    m = ytr_perm_2d.shape[0]
    yc = (ytr_perm_2d - ytr_perm_2d.mean(1, keepdims=True)).T.astype(np.float64)
    d = X.shape[1]
    rhs = np.zeros((d, m), dtype=np.float64)
    for i0 in range(0, len(tr), RIDGE_CHUNK):
        i1 = min(i0 + RIDGE_CHUNK, len(tr))
        rhs += _ridge_std_chunk(X, tr[i0:i1]).T @ yc[i0:i1]
    W = solve_(rhs)
    out = np.empty((len(te), m), dtype=np.float64)
    for i0 in range(0, len(te), RIDGE_CHUNK):
        i1 = min(i0 + RIDGE_CHUNK, len(te))
        out[i0:i1] = _ridge_std_chunk(X, te[i0:i1]) @ W
    return out + ytr_perm_2d.mean(1, keepdims=True).T


def _null_one(pname, xtr_path, xte_path, ytr_perm, yte_perm, seed):
    """Single permutation (linear/MLP): mmap-read features → fit(permuted train labels) → eval(permuted test)."""
    xtr = np.load(xtr_path, mmap_mode="r")
    xte = np.load(xte_path, mmap_mode="r")
    if pname == "linear":
        s, _ = rfp.probe_linear_fit(xtr, ytr_perm, xte, tol=1e-3)
    else:
        s, _ = rfp.mlp_2x1024(np.array(xtr), ytr_perm, np.array(xte), seed=seed,
                              epochs=10, patience=3,
                              threads=int(os.environ.get("_CC_NULL_THREADS", "8")))
    return rap.metrics(yte_perm, s)


# ═══════════════════════════ main pipeline ═══════════════════════════ #

def standardize_inplace(X: np.ndarray, tr_idx: np.ndarray) -> np.ndarray:
    mu = X[tr_idx].mean(0, keepdims=True)
    sd = X[tr_idx].std(0) + 1e-12
    step = 65536
    for i0 in range(0, X.shape[0], step):
        X[i0:i0 + step] = (X[i0:i0 + step] - mu) / sd
    return X


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    ap.add_argument("--rap-full-dir", type=Path, default=DEFAULT_RAP_FULL,
                    help="RAP_FULL_400K output directory (same-fold prediction .npy files, for paired DeLong)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--md", type=Path, default=DEFAULT_MD)
    ap.add_argument("--n-perm", type=int, default=100,
                    help="target number of linear null permutations (actual completions within the wall-clock budget recorded in the CSV)")
    ap.add_argument("--ridge-n-perm", type=int, default=200)
    ap.add_argument("--mlp-n-perm", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--null-jobs", type=int, default=4)
    ap.add_argument("--null-threads", type=int, default=8)
    ap.add_argument("--kmer-jobs", type=int, default=32)
    ap.add_argument("--mlp-threads", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0,
                    help=">0 stratified-subsample end-to-end smoke (keep 0 for the official run)")
    args = ap.parse_args()
    t_start = time.time()
    if args.limit:
        args.out_dir = args.out_dir.with_name(args.out_dir.name + "_smoke")
        args.md = args.md.with_suffix(".smoke.md")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    kmer_bin = args.out_dir / "kmer7full_400k.f32"
    kmer_ok = args.out_dir / "kmer7full_400k.OK"

    import torch
    torch.set_num_threads(min(args.mlp_threads, 32))
    os.environ["_CC_NULL_THREADS"] = str(args.null_threads)

    # ── ① fasta statistics (single streaming pass; read-only, sequences not retained) ──
    print("[load] stream fasta stats...", flush=True)
    t0 = time.time()
    ids, y, st = rfp.stream_fasta_stats(args.fasta)
    if args.limit:                                # stratified subsample (same as rap_full)
        _keep = sorted(
            [i for i, v in enumerate(y) if v][: args.limit // 2] +
            [i for i, v in enumerate(y) if not v][: args.limit // 2])
        ids = [ids[i] for i in _keep]; y = y[_keep]; st = st[_keep]
        print(f"[smoke] --limit {args.limit} stratified subsample "
              f"(pos={int(y.sum())} neg={int((1-y).sum())})", flush=True)
    n = len(ids)
    pos = int(y.sum())
    print(f"[load] n={n} pos={pos} neg={n-pos} "
          f"({time.time()-t0:.0f}s); len med={np.median(st[:,0]):.0f}", flush=True)

    # ── ② full-dim kmer7 memmap (written directly by workers; .OK guards against reusing partial files) ──
    if kmer_ok.is_file():
        print("[kmer] memmap already complete, reusing", flush=True)
    else:
        with open(kmer_bin, "wb") as fh:
            fh.truncate(n * N_BUCKET * 4)
        build_kmer_full_memmap(args.fasta, ids, kmer_bin, args.kmer_jobs)
        kmer_ok.write_text(f"n={n} buckets={N_BUCKET} "
                           f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ── ③ assemble X = [kmer7full | log10 len | GC] and normalize in place ──
    print("[asm] X = [kmer7full(16384) | log10 len | GC] ...", flush=True)
    t0 = time.time()
    X = np.empty((n, N_BUCKET + D_EXTRA), dtype=np.float32)
    X[:, :N_BUCKET] = np.memmap(kmer_bin, dtype=np.float32, mode="r",
                                shape=(n, N_BUCKET))
    X[:, N_BUCKET] = np.log10(st[:, 0])
    X[:, N_BUCKET + 1] = st[:, 1]
    len_gc = np.stack([np.log10(st[:, 0]), st[:, 1]], 1).astype(np.float32)

    # ── ④ split: fully consistent with rap_full (same fasta order + stratified_split seed42) ──
    tr, te = rap.stratified_split(y, seed=42)
    ytr, yte = y[tr].astype(int), y[te].astype(int)
    print(f"[split] train={len(tr)} test={len(te)} test_pos={int(yte.sum())} "
          f"(same fold as RAP_FULL_400K)", flush=True)
    standardize_inplace(X, tr)
    print(f"[asm] X {X.shape} {X.nbytes/1e9:.1f}GB, normalization done ({time.time()-t0:.0f}s)",
          flush=True)

    rng = np.random.default_rng(42)
    perm_ys = np.stack([rng.permutation(y) for _ in range(
        max(args.ridge_n_perm, args.n_perm))])     # [m, n]

    rows = []
    from joblib import Parallel, delayed
    import pandas as pd

    # ── ⑤ three-level probes ──
    preds = {}
    ridge_solve = None
    for pname in ["linear", "ridge", "mlp"]:
        t_cell = time.time()
        if pname == "linear":
            s, nonconv = rfp.probe_linear_fit(X[tr], ytr, X[te], tol=1e-3)
        elif pname == "ridge":
            s, ridge_solve, _G = ridge_chunked_fit(X, tr, te, ytr)
            nonconv = 0
        else:
            xtr = np.ascontiguousarray(X[tr]); xte = np.ascontiguousarray(X[te])
            s, ep_used = rfp.mlp_2x1024(xtr, ytr, xte, seed=42,
                                        threads=args.mlp_threads)
            del xtr, xte; gc.collect()
            nonconv = 0
        auroc, auprc = rap.metrics(yte, s)
        preds[pname] = np.asarray(s, dtype=np.float64)
        np.save(args.out_dir / f"_pred_comp_{pname}.npy", preds[pname])
        print(f"[fit] {GROUP_NAME}/{pname}: AUROC={auroc:.4f} AUPRC={auprc:.4f} "
              f"({time.time()-t_cell:.0f}s)", flush=True)

        # bootstrap CI (4 sub-chunks × 4 threads)
        chunks = Parallel(n_jobs=4, initializer=rfp._init_worker, initargs=(4,))(
            delayed(rfp._boot_chunk)(yte, s, args.n_boot // 4, 42 + k)
            for k in range(4))
        aucs = [v for c in chunks for v in c[0]]
        aps = [v for c in chunks for v in c[1]]
        al, ah = np.percentile(aucs, [2.5, 97.5])
        pl, ph = np.percentile(aps, [2.5, 97.5])

        # permutation null
        null_a, null_p = [], []
        if pname == "ridge":
            sc = ridge_chunked_null(ridge_solve, X, tr, te,
                                    perm_ys[: args.ridge_n_perm, tr])
            null_a = list(rap.auroc_cols(perm_ys[: args.ridge_n_perm, te].T, sc))
            from sklearn.metrics import average_precision_score
            for j in range(sc.shape[1]):
                null_p.append(average_precision_score(perm_ys[j, te], sc[:, j]))
            n_perm_actual = sc.shape[1]
        else:
            target = args.n_perm if pname == "linear" else args.mlp_n_perm
            xtr_p = args.out_dir / "_tmp_xtr.npy"
            xte_p = args.out_dir / "_tmp_xte.npy"
            if pname == "mlp":
                rng_sub = np.random.default_rng(42)
                sub_rows = np.sort(rng_sub.choice(
                    tr, min(50_000, len(tr)), replace=False))
                np.save(xtr_p, np.ascontiguousarray(X[sub_rows]))
                tr_rows_for_null = sub_rows
            else:
                np.save(xtr_p, X[tr])
                tr_rows_for_null = tr
            np.save(xte_p, X[te])
            t_null0 = time.time()
            budget = NULL_WALLCLOCK[pname]
            j0 = 0
            while j0 < target and time.time() - t_null0 < budget:
                j1 = min(j0 + args.null_jobs * 2, target)
                res = Parallel(n_jobs=args.null_jobs,
                               initializer=rfp._init_worker,
                               initargs=(args.null_threads,))(
                    delayed(_null_one)(pname, str(xtr_p), str(xte_p),
                                       perm_ys[j][tr_rows_for_null].astype(int),
                                       perm_ys[j][te].astype(int), 1000 + j)
                    for j in range(j0, j1))
                null_a += [r[0] for r in res]
                null_p += [r[1] for r in res]
                j0 = j1
            n_perm_actual = len(null_a)
            xtr_p.unlink(missing_ok=True); xte_p.unlink(missing_ok=True)
        p_au = (1 + int(np.sum(np.asarray(null_a) >= auroc))) / (1 + n_perm_actual)
        p_ap = (1 + int(np.sum(np.asarray(null_p) >= auprc))) / (1 + n_perm_actual)

        rows.append(dict(
            feature_group=GROUP_NAME, probe=pname,
            role={"linear": "Readability", "ridge": "Stability",
                  "mlp": "Capacity ceiling"}[pname],
            n_train=int(len(tr)), n_test=int(len(te)), prevalence=float(yte.mean()),
            auroc=auroc, auroc_ci95=f"[{al:.4f},{ah:.4f}]",
            auprc=auprc, auprc_ci95=f"[{pl:.4f},{ph:.4f}]",
            null_auroc_mean=float(np.mean(null_a)),
            null_auroc_sd=float(np.std(null_a)), null_auroc_p=p_au,
            null_auprc_mean=float(np.mean(null_p)),
            null_auprc_sd=float(np.std(null_p)), null_auprc_p=p_ap,
            n_perm=int(n_perm_actual), n_boot=args.n_boot,
            linear_nonconverged=int(nonconv) if pname == "linear" else 0,
            mlp_epochs_used=int(ep_used) if pname == "mlp" else 0,
            runtime_s=round(time.time() - t_cell, 1)))
        pd.DataFrame(rows).to_csv(
            args.out_dir / "composition_ceiling_results.csv", index=False)
        print(f"[ceiling] {pname}: AUROC={auroc:.4f} AUPRC={auprc:.4f} "
              f"(nullA {np.mean(null_a):.3f} p={p_au:.1e}; n_perm={n_perm_actual})",
              flush=True)

    # ── ⑥ same-fold paired DeLong (vs RAP_FULL's existing predictions; split consistency asserted first) ──
    drows = []
    rap_preds = {}
    for g in RAP_GROUPS:
        for pname in ["linear", "ridge", "mlp"]:
            p = args.rap_full_dir / f"_pred_{g}_{pname}.npy"
            if p.is_file():
                rap_preds[(g, pname)] = np.load(p)
    if rap_preds:
        ck = rap_preds.get(("evo", "linear"))
        same_fold = ck is not None and len(ck) == len(te)
        if same_fold:
            _a, _p = rap.metrics(yte, ck)
            print(f"[delong] fold consistency verify: evo/linear recomputed AUROC={_a:.4f} "
                  f"(published value 0.7775)", flush=True)
            for pname in ["linear", "ridge", "mlp"]:
                for g in RAP_GROUPS:
                    if (g, pname) not in rap_preds:
                        continue
                    d = delong_test_full(yte, preds[pname], rap_preds[(g, pname)])
                    drows.append(dict(probe=pname, group_a=GROUP_NAME, group_b=g,
                                      auc_a=d["auc_a"], auc_b=d["auc_b"],
                                      delta_auc=d["delta"], z=d["z"], p=d["p"],
                                      n=d["n"], n_pos=d["n_pos"]))
                    print(f"[delong] {pname} {GROUP_NAME} vs {g}: "
                          f"Δ={d['delta']:+.4f} p={d['p']:.2e}", flush=True)
            pd.DataFrame(drows).to_csv(
                args.out_dir / "composition_ceiling_delong.csv", index=False)
        else:
            print("[delong] RAP_FULL prediction shapes inconsistent, skipping paired DeLong", flush=True)

    total_s = time.time() - t_start
    _write_report(args, pd.DataFrame(rows), pd.DataFrame(drows), n, pos, total_s)
    if not args.limit:
        EXIT_MARK.write_text(
            f"completed={time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"total_s={total_s:.0f}\n")
    print(f"[composition-ceiling] done {total_s/60:.0f} min -> {args.md}", flush=True)


# ═══════════════════════════ report ═══════════════════════════ #

def _fmt(v, nd=3):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return " — "
        return f"{v:.{nd}f}"
    except Exception:
        return str(v)


def _write_report(args, df, ddf, n, pos, total_s) -> None:
    best_auprc = float(df.auprc.max()) if len(df) else float("nan")
    best_row = df.loc[df.auprc.idxmax()] if len(df) else None
    L = []
    A = L.append
    A("# COMPOSITION CEILING  -  composition-baseline ceiling measurement on simulated data (literature application plan #5 / R5)\n")
    A(f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  ·  DeepFluHGT  ·  CPU only")
    A(f"- data: `simulated_data/Contig_500k/train` n={n:,} (pos={pos:,}, "
      f"neg={n-pos:,}); label = id prefix pos_/neg_ (= BED ground truth)")
    A("- features: **full-dim kmer7 4^7=16384 relative frequencies** (no RC folding; RAP_FULL's kmer7_rc "
      "is the 8192 RC-folded variant; this measurement takes the broader full-dimension caliber) ⊕ log10(length) ⊕ GC = 16386 dims")
    A("- probes: Linear=LogisticRegression (Readability)  ·  Ridge=closed-form primal "
      "chunked float64 (Stability)  ·  MLP-2×1024 (Capacity ceiling)")
    A(f"- protocol: 70/30 stratified (seed42, **fully the same fold as RAP_FULL_400K**) + "
      f"bootstrap {args.n_boot} CI + label-permutation null "
      f"(ridge={args.ridge_n_perm} exact batched; linear={args.n_perm}/"
      f"MLP={args.mlp_n_perm} wall-clock budget, actual n_perm in the CSV)")
    A(f"- run: nohup CPU ≤48 threads, total runtime {total_s/60:.0f} min\n")

    A("## 1. Primary results (same fold; bootstrap CI)\n")
    A("| feature_group | probe (role) | AUROC [CI] | AUPRC [CI] | "
      "null AUROC (p) | null AUPRC (p) | n_perm |")
    A("|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        A(f"| {r.feature_group} | {r.probe} ({r.role}) | "
          f"{_fmt(r.auroc,4)} {r.auroc_ci95} | {_fmt(r.auprc,4)} {r.auprc_ci95} | "
          f"{_fmt(r.null_auroc_mean)} (p={r.null_auroc_p:.1e}) | "
          f"{_fmt(r.null_auprc_mean)} (p={r.null_auprc_p:.1e}) | {int(r.n_perm)} |")
    A(f"\nprevalence = {df.prevalence.iloc[0]:.4f} (class-balanced).\n")

    A("## 2. Interpretation (pre-registered rules)\n")
    A(f"- **composition ceiling (best probe, AUPRC) = {_fmt(best_auprc,4)} "
      f"({best_row.probe if best_row is not None else ' — '})**")
    A(f"- reference: M2 ep2 sim val AUPRC **{REF['M2_ep2_val_auprc']}** [training in progress, terminal value pending] "
      f"(official val split); M0 terminal {REF['M0_val_auprc']}")
    A(f"- same-fold reference (RAP_FULL_400K): evo MLP {REF['evo_mlp_same_fold_auroc']} / "
      f"evo linear {REF['evo_linear_same_fold_auroc']} / kmer7_rc linear "
      f"{REF['kmer7rc_linear_same_fold_auroc']}")
    gap = best_auprc - REF["M2_ep2_val_auprc"]
    verdict = ("**composition ceiling < 0.87 → FM features carry information beyond composition** "
               "(M2 exceeds what composition can reach)" if best_auprc < REF["M2_ep2_val_auprc"] else
               "**composition ceiling ≥ 0.87 → composition explanatory power must be taken seriously** (the LAMBDA-warned failure mode, "
               "the paper is restated under a composition+FM hybrid-ceiling caliber)")
    A(f"- Δ(composition ceiling − M2 ep2) = {gap:+.4f} → {verdict}\n")
    A("- caliber note: the probes are trained within the 70/30 (AUROC/AUPRC), M2 is the official val split "
      "AUPRC  —  different splits; interpretation is by order-of-magnitude triangulation (procedure-consistent with RAP_FULL vs evo); "
      "the strict same-fold statistical test is §3 DeLong.\n")

    if len(ddf):
        A("## 3. Same-fold paired DeLong (composition vs each RAP_FULL group)\n")
        for pname in ["linear", "ridge", "mlp"]:
            sub = ddf[ddf.probe == pname] if len(ddf) else ddf
            if not len(sub):
                continue
            A(f"### probe = {pname}\n")
            A("| comparison (a vs b) | AUROC_a | AUROC_b | Δ(a−b) | z | p | significant? |")
            A("|---|---|---|---|---|---|---|")
            for _, r in sub.sort_values("p").iterrows():
                A(f"| {r.group_a} vs {r.group_b} | {r.auc_a:.4f} | {r.auc_b:.4f} | "
                  f"{r.delta_auc:+.4f} | {r.z:+.2f} | {r.p:.2e} | "
                  f"{'**yes**' if r.p < 0.05 else 'no'} |")
            A("")

    A("## 4. Notes / calibers")
    A("- full-dim kmer7 written directly by workers via a 26GB raw-binary memmap; X normalized in place (train statistics); "
      "ridge Gram accumulated chunk-wise in float64 (400k×16386 float64 never materialized).")
    A("- difference from RAP_FULL_400K's kmer7_rc: this measurement does not RC-fold (16384 vs 8192 dims) "
      "and adds the log10(len)+GC dimensions  —  both are composition calibers; full-dim + covariates is the **broader** composition ceiling.")
    A("- full values: `evo_cache/Contig_500k/composition_ceiling/"
      "composition_ceiling_results.csv`  ·  `composition_ceiling_delong.csv`.\n")
    args.md.write_text("\n".join(L))


if __name__ == "__main__":
    main()
