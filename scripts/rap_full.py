#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RAP official full-volume version (400k train cache)  -  DeepFluHGT [P0-1 upgrade]

Full-volume re-run of the step0 version (`scripts/rap_probes.py`, n=1996):
  data  : evo_cache/Contig_500k/train.pt (400,000 contigs × 4096d, fp16
          Evo2-7B blocks.28.mlp.l3, window protocol 3000/2700 → equal-weight contig mean-pool)
          label = id prefix pos_/neg_ (= Contig_500k train BED ground truth, 100% consistent)
  feature groups: evo(4096) / len+GC(2) / RC-kmer7(8192) / evo+len+GC(4098)
  probes: Linear=LogisticRegression (Readability)
          Ridge=closed-form primal (Stability; the step0 dual form is infeasible at n=280k)
          MLP-2×1024 (D→1024→1024→1, mini-batch Adam+early stopping, Capacity ceiling)
           —  —  MLP was skipped at step0 for compute reasons; this version fills it in properly on 64 CPU threads.
  protocol  : 70/30 stratified (seed42) + bootstrap 2000 CI + label-permutation null
          (ridge=closed-form primal, batched exact 500; linear=lbfgs tol=1e-3, target 500 +
           wall-clock 5400s/cell early stopping; MLP=target 200, lightened to 10 epochs + a 50k train-row subsample,
           wall-clock 3600s/cell  —  actual n_perm recorded per cell in the CSV, following the
           crosstask_full_probe NULL_WALLCLOCK_S precedent)
          + paired DeLong across all feature-group pairs (6 pairs × 3 probes = 18, same test fold)
  memory  : evo fp16 dict (~3.3GB) → fp32 block-wise conversion (6.5GB);
          kmer7 via a raw-binary memmap (13GB) written directly by workers, no IPC;
          during the null stage xtr/xte are stored as .npy for worker mmap sharing (avoiding
          joblib's per-task re-serialization of 4.6GB). Peak RSS < 60GB (machine has 251GB).

output:
  evo_cache/Contig_500k/rap_full/rap_full_results.csv (incrementally written)
  evo_cache/Contig_500k/rap_full/rap_full_delong.csv
  docs/evidence/RAP_FULL_400K.md   (including step0 n=1996 controlcolumn)
  docs/evidence/RAP_FULL_400K.EXIT (nohup completion marker)

usage (nohup):
  nohup python scripts/rap_full.py \
      > logs/rap_full_400k.log 2>&1 &

Run discipline: CPU-only (CUDA_VISIBLE_DEVICES=""), ≤64 threads in any stage,
never touches GPUs (4090/A100 production workloads).

[FIX 2026-08-27 RAP400K_RETRY]
  1. Crash of the ridge null's float32 Gram Cholesky (not positive definite):
     the Gram switches to float64 accumulation + escalating relative jitter (1e-9→1e-2 · trace/d) +
     an LU fallback (_chol_robust/_ridge_core; primary fit and null share the same metric caliber).
  2. Checkpoint resume: cells already completed in the CSV (including _pred_*.npy) are skipped; a re-run after a crash only fills missing cells.
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
import rap_probes as rap                    # reuse the already-validated fasta/kmer/statistics implementation
from delong_test import delong_test_full

DEFAULT_CACHE = REPO / "evo_cache" / "Contig_500k"
DEFAULT_FASTA = REPO / "simulated_data" / "Contig_500k" / "train" / "dataset.fasta"
STEP0_CSV = REPO / "evo_cache" / "step0_subset" / "rap" / "rap_results.csv"
DEFAULT_OUT_DIR = DEFAULT_CACHE / "rap_full"
DEFAULT_MD = REPO / "docs" / "evidence" / "RAP_FULL_400K.md"
EXIT_MARK = REPO / "docs" / "evidence" / "RAP_FULL_400K.EXIT"

GROUPS = ["evo", "len+gc", "kmer7_rc", "evo+len+gc"]
KMER_K = rap.KMER_K
N_BUCKET = 4 ** KMER_K // 2                    # 8192
MLP_PERM_EPOCHS = 10                           # lightened epochs for null fits (noted in report)
MLP_PERM_SUBSAMPLE = 50_000                    # train-row subsample for the MLP null (noted in report)
NULL_WALLCLOCK = {"linear": 5400.0, "ridge": 0.0, "mlp": 3600.0}   # seconds/cell


# ═══════════════════════════ Data loading ═══════════════════════════ #

def stream_fasta_stats(path: Path):
    """Single streaming pass: returns (ids, y, [len, gc, n_pct]). Sequences are not retained."""
    ids, lens, gcs, ns = [], [], [], []
    name, chunks = None, []
    a = c = g = t = L = 0

    def _flush():
        if name is None:
            return
        acgt = a + c + g + t
        ids.append(name); lens.append(L)
        gcs.append((g + c) / acgt if acgt else np.nan)
        ns.append((L - acgt) / L if L else np.nan)

    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                _flush()
                name = line[1:].strip().split()[0]
                chunks = []; a = c = g = t = L = 0
            else:
                s = line.strip().upper()
                chunks.append(s); L += len(s)
                a += s.count("A"); c += s.count("C")
                g += s.count("G"); t += s.count("T")
    _flush()
    y = np.array([1 if i.startswith("pos_") else 0 for i in ids], dtype=np.int8)
    st = np.stack([np.array(lens), np.array(gcs), np.array(ns)], 1)
    return ids, y, st


def _kmer_batch(task):
    """joblib worker: a batch of sequences → RC-kmer7 relative frequencies written directly into a shared memmap row segment.

    task = (memmap_path, n_rows, n_bucket, row0, seqs)
    Row-number continuity is guaranteed by the dispatcher (fasta order = row-number order).
    """
    from numpy.lib.stride_tricks import sliding_window_view
    path, n_rows, n_bucket, row0, seqs = task
    k = KMER_K
    lut = np.full(256, -1, dtype=np.int64)
    for b, i in rap._NUC2IDX.items():
        lut[ord(b)] = i
    pow4 = rap._POW4
    canon = rap._CANON_TABLE
    mm = np.memmap(path, dtype=np.float32, mode="r+", shape=(n_rows, n_bucket))
    for j, seq in enumerate(seqs):
        idx = lut[np.frombuffer(seq.encode(), dtype=np.uint8)]
        if len(idx) < k:
            continue
        win = sliding_window_view(idx, k)
        valid = (win >= 0).all(axis=1)
        wv = np.where(win[valid] >= 0, win[valid], 0)
        bucket = canon[np.minimum(wv @ pow4, (3 - wv[:, ::-1]) @ pow4)]
        counts = np.bincount(bucket, minlength=n_bucket).astype(np.float32)
        tot = counts.sum()
        mm[row0 + j] = counts / tot if tot > 0 else 0.0
    mm.flush()
    return len(seqs)


def build_kmer_memmap(fasta: Path, ids: list[str], out_bin: Path, n_jobs: int):
    """Second streaming pass: batches of 4000 compute RC-kmer7 → workers write the raw-binary memmap directly."""
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
    if buf_ids:
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
        os.environ[v] = "1"                       # pin to 1 thread before spawn (oversubscription guard)
    t0 = time.time()
    try:
        res = Parallel(n_jobs=n_jobs)(delayed(_kmer_batch)(t) for t in tasks)
    finally:
        for v, val in saved.items():
            if val is not None:
                os.environ[v] = val
    print(f"[kmer] {int(np.sum(res))} seqs → {out_bin.name} "
          f"({time.time()-t0:.0f}s, {n_jobs} workers)", flush=True)


# ═══════════════════════════ probes ═══════════════════════════ #

def probe_linear_fit(xtr, ytr, xte, tol):
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m = LogisticRegression(C=1.0, max_iter=2000, tol=tol, solver="lbfgs")
        m.fit(xtr, ytr)
    n_iter = int(np.ravel(getattr(m, "n_iter_", [0]))[0])
    return m.predict_proba(xte)[:, 1], n_iter >= 2000


def _chol_robust(G: np.ndarray):
    """[FIX 2026-08-27 RAP400K_RETRY] Cholesky with escalating relative jitter; returns None on failure.

    Crash root cause: for evo (fp16→fp32) 4096d × n_tr=280k, float32 Gram accumulation has
    error ~√n · ε · diag(G) ≈ 18 ≫ α=1, drowning the small feature values along near-collinear
    directions → numerically indefinite, and np.linalg.cholesky raises "not positive definite".
    This function pairs with a float64 Gram (residual ~3e-8) and escalates the jitter as a fallback;
    on persistent failure it returns None, and the caller falls back to LU (np.linalg.solve, no PD required).
    """
    d = G.shape[0]
    base = max(1.0, float(np.trace(G)) / d)
    G[np.diag_indices(d)] += 1e-9 * base
    for j in (0.0, 1e-6, 1e-4, 1e-3, 1e-2):
        if j:
            G[np.diag_indices(d)] += j * base
        try:
            return np.linalg.cholesky(G)
        except np.linalg.LinAlgError:
            continue
    return None


def _ridge_core(xtr, xte, alpha: float):
    """Normalization + float64 Gram → (solver function rhs→W, a, b). Shared by primary fit and null."""
    mu = xtr.mean(0, keepdims=True); sd = xtr.std(0) + 1e-12
    a = ((xtr - mu) / sd).astype(np.float64)        # FIX: Gram uses float64 accumulation
    b = (xte - mu) / sd
    d = a.shape[1]
    G = a.T @ a
    G[np.diag_indices(d)] += alpha
    chol = _chol_robust(G)
    if chol is not None:
        def solve_(rhs):
            return np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))
    else:
        print("[ridge] Cholesky failure (after jitter cap), falling back to LU", flush=True)

        def solve_(rhs):
            return np.linalg.solve(G, rhs)
    return solve_, a, b


def ridge_primal(xtr, ytr, xte, alpha: float = 1.0):
    """Closed-form primal ridge (n>>d: O(d²n); the step0 dual with K=n×n is infeasible at n=280k)."""
    solve_, a, b = _ridge_core(xtr, xte, alpha)
    yc = ytr - ytr.mean()
    return b @ solve_(a.T @ yc) + ytr.mean()


def ridge_primal_null_batched(xtr, ytr_perm, xte, alpha: float = 1.0):
    """Batched closed-form primal null: one Cholesky, m permuted labels in one GEMM. Exact."""
    solve_, a, b = _ridge_core(xtr, xte, alpha)
    yc = ytr_perm - ytr_perm.mean(0, keepdims=True)      # [n_tr, m]
    return b @ solve_(a.T @ yc) + ytr_perm.mean(0, keepdims=True)  # [n_te, m]


def mlp_2x1024(xtr, ytr, xte, seed: int, epochs: int = 40, patience: int = 5,
               bs: int = 1024, lr: float = 1e-3, threads: int | None = None):
    """MLP-2×1024 capacity ceiling: mini-batch Adam + internal 10% val early stopping, CPU.

    Differences from step0's _torch_mlp (400k adaptation, noted in the report):
      hidden 256→1024×2 (per the task brief); full-batch→mini-batch 1024;
      epochs 200→40 (280k samples ≈ 11k steps). Returns (test probability, actual epochs).
    """
    import torch
    import torch.nn as nn
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(xtr)
    idx = rng.permutation(n)
    n_val = max(int(n * 0.10), 1000)
    val_i, fit_i = idx[:n_val], idx[n_val:]
    xt = torch.from_numpy(np.ascontiguousarray(xtr, dtype=np.float32))
    xte_t = torch.from_numpy(np.ascontiguousarray(xte, dtype=np.float32))
    yt = torch.from_numpy(ytr.astype(np.float32))
    model = nn.Sequential(
        nn.Linear(xtr.shape[1], 1024), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(1024, 1))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    best, bad, best_state, ep_used = np.inf, 0, None, 0
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        ep_used = ep + 1
        model.train()
        perm = torch.randperm(len(fit_i), generator=g)
        for b0 in range(0, len(fit_i), bs):
            bi = fit_i[perm[b0: b0 + bs]]
            opt.zero_grad()
            lossf(model(xt[bi]).squeeze(-1), yt[bi]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = 0.0
            for b0 in range(0, len(val_i), 8192):
                vi = val_i[b0: b0 + 8192]
                vloss += float(lossf(model(xt[vi]).squeeze(-1), yt[vi])) * len(vi)
            vloss /= len(val_i)
        if vloss < best - 1e-4:
            best, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    out = []
    with torch.no_grad():
        for b0 in range(0, len(xte_t), 8192):
            out.append(torch.sigmoid(
                model(xte_t[b0: b0 + 8192])).squeeze(-1).numpy())
    return np.concatenate(out).astype(np.float64), ep_used


# ═══════════════════════════ bootstrap / null workers ═══════════════════════════ #

def _init_worker(threads: int):
    import torch
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(threads)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(threads)
    except Exception:
        pass
    torch.set_num_threads(threads)


def _boot_chunk(y, s, n_boot, seed):
    """Bootstrap sub-chunk: returns (aucs, aps) lists (quantiles taken after merging)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs, aps = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        yy, ss = y[i], s[i]
        if yy.min() == yy.max():
            continue
        aucs.append(roc_auc_score(yy, ss))
        aps.append(average_precision_score(yy, ss))
    return aucs, aps


def _null_one(pname: str, xtr_path: str, xte_path: str,
              ytr_perm: np.ndarray, yte_perm: np.ndarray, seed: int):
    """Single permutation: mmap-read features → fit(permuted train labels) → eval(permuted test labels)."""
    xtr = np.load(xtr_path, mmap_mode="r")
    xte = np.load(xte_path, mmap_mode="r")
    if pname == "linear":
        s, _ = probe_linear_fit(xtr, ytr_perm, xte, tol=1e-3)
    else:
        s, _ = mlp_2x1024(np.array(xtr), ytr_perm, np.array(xte),
                          seed=seed, epochs=MLP_PERM_EPOCHS, patience=3,
                          threads=int(os.environ.get("_RAP_NULL_THREADS", "8")))
    return rap.metrics(yte_perm, s)


# ═══════════════════════════ main pipeline ═══════════════════════════ #

def standardize_inplace(X: np.ndarray, tr_idx: np.ndarray) -> np.ndarray:
    """In-place normalization with train statistics (chunked, avoiding a doubled 400k full copy)."""
    mu = X[tr_idx].mean(0, keepdims=True)
    sd = X[tr_idx].std(0) + 1e-12
    step = 65536
    for i0 in range(0, X.shape[0], step):
        X[i0:i0 + step] = (X[i0:i0 + step] - mu) / sd
    return X


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--md", type=Path, default=DEFAULT_MD)
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--mlp-n-perm", type=int, default=200,
                    help="target number of MLP null permutations (budget-limited, actual recorded; ridge/linear "
                         "target --n-perm)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--null-jobs", type=int, default=4,
                    help="parallel null workers (each worker holds a float64 feature copy, "
                         "4×16 threads = cap of 64; memory-safe)")
    ap.add_argument("--null-threads", type=int, default=16)
    ap.add_argument("--kmer-jobs", type=int, default=32)
    ap.add_argument("--mlp-threads", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0,
                    help=">0 uses only the first N fasta records (end-to-end smoke; keep 0 for the official run)")
    args = ap.parse_args()
    t_start = time.time()
    if args.limit:      # smoke: output redirected; no official report/EXIT/large memmap
        args.out_dir = args.out_dir.with_name(args.out_dir.name + "_smoke")
        args.md = args.md.with_suffix(".smoke.md")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    kmer_bin = args.out_dir / "kmer7_400k.f32"
    kmer_ok = args.out_dir / "kmer7_400k.OK"

    import torch
    torch.set_num_threads(min(args.mlp_threads, 32))
    os.environ["_RAP_NULL_THREADS"] = str(args.null_threads)

    # ── ① evo cache + fasta alignment ──
    print("[load] evo train.pt (4.9GB, fp16)...", flush=True)
    t0 = time.time()
    emb = torch.load(args.cache / "train.pt", map_location="cpu",
                     weights_only=False)
    print(f"[load] {len(emb)} contigs, {time.time()-t0:.0f}s", flush=True)

    ids, y, st = stream_fasta_stats(args.fasta)
    if args.limit:
        # the fasta is sorted by pos prefix → a naive prefix slice could yield pos=neg=0 (single class, sklearn rejects);
        # the smoke subsample uses a stratified prefix instead (first N/2 pos + first N/2 neg), affecting only the --limit smoke path
        _keep = sorted(
            [i for i, v in enumerate(y) if v][: args.limit // 2] +
            [i for i, v in enumerate(y) if not v][: args.limit // 2])
        ids = [ids[i] for i in _keep]; y = y[_keep]; st = st[_keep]
        print(f"[smoke] --limit {args.limit} stratified subsample in effect "
              f"(pos={int(y.sum())} neg={int((1-y).sum())})", flush=True)
    assert not [i for i in ids if i not in emb], "FATAL: fasta ids missing embeddings"
    n = len(ids)
    pos = int(y.sum())
    print(f"[load] n={n} pos={pos} neg={n-pos}; "
          f"len med={np.median(st[:,0]):.0f} max={st[:,0].max():.0f}", flush=True)

    # ── ② kmer7 memmap (written directly by workers; .OK marker prevents reusing partial files) ──
    if kmer_ok.is_file():
        print("[kmer] memmap already complete, reusing", flush=True)
    else:
        with open(kmer_bin, "wb") as fh:
            fh.truncate(n * N_BUCKET * 4)
        build_kmer_memmap(args.fasta, ids, kmer_bin, args.kmer_jobs)
        kmer_ok.write_text(f"n={n} buckets={N_BUCKET} "
                           f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ── ③ evo fp16 → fp32, block-wise ──
    print("[load] evo fp16→fp32 block-wise conversion...", flush=True)
    t0 = time.time()
    evo = np.empty((n, 4096), dtype=np.float32)
    for i0 in range(0, n, 50_000):
        i1 = min(i0 + 50_000, n)
        evo[i0:i1] = np.stack(
            [emb[ids[i]] for i in range(i0, i1)]).astype(np.float32)
    del emb; gc.collect()
    print(f"[load] evo X {evo.shape} {evo.nbytes/1e9:.1f}GB "
          f"({time.time()-t0:.0f}s)", flush=True)

    len_gc = np.stack([np.log10(st[:, 0]), st[:, 1]], 1).astype(np.float32)

    # ── ④ split (seed42, 70/30 stratified) ──
    tr, te = rap.stratified_split(y, seed=42)
    ytr, yte = y[tr].astype(int), y[te].astype(int)
    print(f"[split] train={len(tr)} test={len(te)} test_pos={int(yte.sum())}",
          flush=True)

    rng = np.random.default_rng(42)
    perm_ys = np.stack([rng.permutation(y) for _ in range(args.n_perm)])  # [m, n]

    rows, preds = [], {}
    from joblib import Parallel, delayed
    import pandas as pd

    # ── Checkpoint resume [FIX 2026-08-27 RAP400K_RETRY]: cells whose CSV rows + prediction .npy
    #    are complete are reused directly (fit/bootstrap/null all skipped); rows missing prediction
    #    files are recomputed (DeLong needs each cell's test predictions). Perms/splits are seed-deterministic → resumes are reproducible.──
    csv_path = args.out_dir / "rap_full_results.csv"
    if csv_path.is_file():
        prev = pd.read_csv(csv_path).to_dict("records")
        for r in prev:
            pfile = args.out_dir / f"_pred_{r['feature_group']}_{r['probe']}.npy"
            if pfile.is_file():
                preds[(r["feature_group"], r["probe"])] = np.load(pfile)
                rows.append(r)
        done = {(r["feature_group"], r["probe"]) for r in rows}
        if len(rows) < len(prev):
            print(f"[resume] {len(prev)-len(rows)} rows missing prediction files, will recompute", flush=True)
        print(f"[resume] reusing completed cells: {sorted(done)}", flush=True)
    else:
        done = set()

    for gname in GROUPS:
        if all((gname, p) in done for p in ("linear", "ridge", "mlp")):
            print(f"[resume] group {gname} fully complete, skipping assembly", flush=True)
            continue
        # ── assembly + normalization (in place, train statistics) ──
        if gname == "evo":
            X = np.array(evo)                    # writable copy (guards against in-place normalization contamination)
        elif gname == "evo+len+gc":
            X = np.empty((n, 4098), dtype=np.float32)
            for i0 in range(0, n, 65536):
                i1 = min(i0 + 65536, n)
                X[i0:i1, :4096] = evo[i0:i1]
                X[i0:i1, 4096:] = len_gc[i0:i1]
        elif gname == "kmer7_rc":
            X = np.array(np.memmap(              # memmap is read-only → force a writable copy
                kmer_bin, dtype=np.float32, mode="r", shape=(n, N_BUCKET)))
        else:
            X = np.array(len_gc)
        standardize_inplace(X, tr)

        xtr = xte = None
        for pname in ["linear", "ridge", "mlp"]:
            if (gname, pname) in done:
                print(f"[resume] {gname}/{pname} already complete, skipping", flush=True)
                continue
            t_cell = time.time()
            xtr = X[tr]; xte = X[te]

            # ① primary fit
            nonconv = 0; ep_used = 0
            if pname == "linear":
                tol = 1e-4 if X.shape[1] <= 8 else 1e-3
                s, nonconv = probe_linear_fit(xtr, ytr, xte, tol)
            elif pname == "ridge":
                s = ridge_primal(xtr, ytr, xte)
            else:
                s, ep_used = mlp_2x1024(xtr, ytr, xte, seed=42,
                                        threads=args.mlp_threads)
            auroc, auprc = rap.metrics(yte, s)
            preds[(gname, pname)] = np.asarray(s, dtype=np.float64)
            np.save(args.out_dir / f"_pred_{gname}_{pname}.npy",       # for resume/DeLong
                    preds[(gname, pname)])
            print(f"[fit] {gname}/{pname}: AUROC={auroc:.4f} "
                  f"AUPRC={auprc:.4f} ({time.time()-t_cell:.0f}s)", flush=True)

            # ② bootstrap 2000 (4 sub-chunks × parallel... 4 chunks → n_jobs=4)
            n_chunk = 4
            per = args.n_boot // n_chunk
            chunks = Parallel(n_jobs=n_chunk, initializer=_init_worker,
                              initargs=(4,))(
                delayed(_boot_chunk)(yte, s, per, 42 + k)
                for k in range(n_chunk))
            aucs = [v for c in chunks for v in c[0]]
            aps = [v for c in chunks for v in c[1]]
            q = lambda v: (float(np.percentile(v, 2.5)),
                           float(np.percentile(v, 97.5)))
            (al, ah), (pl, ph) = q(aucs), q(aps)

            # ③ permutation null (ridge exact batched; linear/MLP chunked dispatch + wall-clock-budget early stopping)
            null_a, null_p = [], []
            mlp_sub_rows = None
            if pname == "ridge":
                sc = ridge_primal_null_batched(xtr, perm_ys[:, tr].T, xte)
                null_a = list(rap.auroc_cols(perm_ys[:, te].T, sc))
                from sklearn.metrics import average_precision_score
                for j in range(sc.shape[1]):
                    null_p.append(average_precision_score(perm_ys[j, te],
                                                          sc[:, j]))
                n_perm_actual = sc.shape[1]
            else:
                target = args.n_perm if pname == "linear" else args.mlp_n_perm
                if pname == "mlp":
                    # lightened null: train-row subsample (fixed seed42; the null only needs the chance level)
                    rng_sub = np.random.default_rng(42)
                    mlp_sub_rows = np.sort(rng_sub.choice(
                        tr, min(MLP_PERM_SUBSAMPLE, len(tr)), replace=False))
                    xtr_p = args.out_dir / f"_tmp_xtr_{gname}_{pname}.npy"
                    np.save(xtr_p, np.ascontiguousarray(X[mlp_sub_rows]))
                else:
                    xtr_p = args.out_dir / f"_tmp_xtr_{gname}_{pname}.npy"
                    np.save(xtr_p, xtr)
                xte_p = args.out_dir / f"_tmp_xte_{gname}_{pname}.npy"
                np.save(xte_p, xte)

                t_null0 = time.time()
                budget = NULL_WALLCLOCK[pname]
                chunk = args.null_jobs * 2
                j0 = 0
                while j0 < target and (not budget or
                                       time.time() - t_null0 < budget):
                    j1 = min(j0 + chunk, target)
                    res = Parallel(
                        n_jobs=args.null_jobs, initializer=_init_worker,
                        initargs=(args.null_threads,))(
                        delayed(_null_one)(pname, str(xtr_p), str(xte_p),
                                           perm_ys[j][
                                               mlp_sub_rows if pname == "mlp"
                                               else tr].astype(int),
                                           perm_ys[j][te].astype(int),
                                           1000 + j)
                        for j in range(j0, j1))
                    null_a += [r[0] for r in res]
                    null_p += [r[1] for r in res]
                    j0 = j1
                n_perm_actual = len(null_a)
                el = time.time() - t_null0
                if n_perm_actual < target:
                    print(f"    [null] {gname}/{pname}: wall-clock budget "
                          f"{budget:.0f}s triggered, {n_perm_actual}/{target} "
                          f"({el:.0f}s)", flush=True)
                xtr_p.unlink(missing_ok=True); xte_p.unlink(missing_ok=True)

            p_au = (1 + int(np.sum(np.asarray(null_a) >= auroc))) / (1 + n_perm_actual)
            p_ap = (1 + int(np.sum(np.asarray(null_p) >= auprc))) / (1 + n_perm_actual)

            rows.append(dict(
                feature_group=gname, probe=pname,
                role={"linear": "Readability", "ridge": "Stability",
                      "mlp": "Capacity ceiling"}[pname],
                n_train=int(len(tr)), n_test=int(len(te)),
                prevalence=float(yte.mean()),
                auroc=auroc, auroc_ci95=f"[{al:.4f},{ah:.4f}]",
                auprc=auprc, auprc_ci95=f"[{pl:.4f},{ph:.4f}]",
                null_auroc_mean=float(np.mean(null_a)),
                null_auroc_sd=float(np.std(null_a)),
                null_auroc_p=p_au,
                null_auprc_mean=float(np.mean(null_p)),
                null_auprc_sd=float(np.std(null_p)),
                null_auprc_p=p_ap,
                n_perm=int(n_perm_actual), n_boot=args.n_boot,
                linear_nonconverged=int(nonconv) if pname == "linear" else 0,
                mlp_epochs_used=int(ep_used) if pname == "mlp" else 0,
                runtime_s=round(time.time() - t_cell, 1)))
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            r_ = rows[-1]
            print(f"[rap] {gname:12s} {pname:7s} AUROC={auroc:.4f} "
                  f"AUPRC={auprc:.4f} (nullA {np.mean(null_a):.3f} "
                  f"p={p_au:.1e}; n_perm={n_perm_actual}) "
                  f"[{r_['runtime_s']}s]", flush=True)
        del X
        xtr = xte = None
        gc.collect()

    # ── ⑤ all paired DeLong (6 pairs × 3 probes, same test fold) ──
    _gord = {g: i for i, g in enumerate(GROUPS)}
    _pord = {p: i for i, p in enumerate(["linear", "ridge", "mlp"])}
    rows.sort(key=lambda r: (_gord[r["feature_group"]], _pord[r["probe"]]))
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    drows = []
    for pname in ["linear", "ridge", "mlp"]:
        for i, ga in enumerate(GROUPS):
            for gb in GROUPS[i + 1:]:
                d = delong_test_full(yte, preds[(ga, pname)], preds[(gb, pname)])
                drows.append(dict(probe=pname, group_a=ga, group_b=gb,
                                  auc_a=d["auc_a"], auc_b=d["auc_b"],
                                  delta_auc=d["delta"], z=d["z"], p=d["p"],
                                  n=d["n"], n_pos=d["n_pos"]))
                print(f"[delong] {pname} {ga} vs {gb}: "
                      f"Δ={d['delta']:+.4f} p={d['p']:.2e}", flush=True)
    pd.DataFrame(drows).to_csv(args.out_dir / "rap_full_delong.csv", index=False)

    total_s = time.time() - t_start
    _write_report(args, pd.DataFrame(rows), pd.DataFrame(drows), n, pos, total_s)
    if not args.limit:
        EXIT_MARK.write_text(
            f"completed={time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"total_s={total_s:.0f}\n")
    print(f"[rap-full] done {total_s/60:.0f} min -> {args.md}", flush=True)


# ═══════════════════════════ report ═══════════════════════════ #

def _fmt(v, nd=3):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return " — "
        return f"{v:.{nd}f}"
    except Exception:
        return str(v)


def _write_report(args, df, ddf, n, pos, total_s) -> None:
    step0 = {}
    try:
        import pandas as pd
        s0 = pd.read_csv(STEP0_CSV)
        step0 = {(r.feature_group, r.probe): float(r.auroc) for _, r in s0.iterrows()}
    except Exception as e:                          # pragma: no cover
        print(f"[WARN] step0 comparison unavailable: {e}")

    L = []
    A = L.append
    A("# RAP FULL  -  official 400k full-volume version (Representation Accessibility Profiling)\n")
    A(f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  ·  DeepFluHGT  ·  CPU only")
    A(f"- data: `evo_cache/Contig_500k/train.pt` n={n:,} (pos={pos:,}, "
      f"neg={n-pos:,}); label = id prefix pos_/neg_ (= Contig_500k train BED ground truth)")
    A("- modality: Evo2-7B frozen, blocks.28.mlp.l3, window 3000/2700 → equal-weight contig "
      "mean-pool (4096d, fp16 cache → fp32 block-wise)")
    A(f"- protocol: 70/30 stratified (seed42) + bootstrap {args.n_boot} CI + label-permutation null "
      f"(target: ridge={args.n_perm} closed-form primal batched **exact**; linear={args.n_perm} "
      f"lbfgs tol=1e-3, wall-clock {NULL_WALLCLOCK['linear']:.0f}s/cell early stopping; "
      f"MLP={args.mlp_n_perm} lightened to 10 epochs + a 50k train-row subsample (seed42), "
      f"wall-clock {NULL_WALLCLOCK['mlp']:.0f}s/cell  —  actual n_perm recorded per cell in the CSV, "
      "p lower bound = 1/(n_perm+1) reported as-is)")
    A("- probes: Linear=LogisticRegression(C=1) Readability  ·  Ridge(α=1, "
      "closed-form primal) Stability  ·  MLP-2×1024 (D→1024→1024→1, mini-batch Adam, "
      "early stopping) Capacity ceiling **[skipped at step0 for compute; completed in this version]**")
    A("- features: evo(4096) / len+GC(log10 len, GC) / RC-kmer7(8192 canonical) / "
      "evo+len+GC(4098)")
    A(f"- run: nohup CPU ≤64 threads, total runtime {total_s/60:.0f} min\n")

    A("## 1. Primary results (seed42 70/30; bootstrap CI)\n")
    A("| feature_group | probe (role) | AUROC [CI] | AUPRC [CI] | "
      "null AUROC (p) | null AUPRC (p) | n_perm |")
    A("|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        A(f"| {r.feature_group} | {r.probe} ({r.role}) | "
          f"{_fmt(r.auroc,4)} {r.auroc_ci95} | {_fmt(r.auprc,4)} {r.auprc_ci95} | "
          f"{_fmt(r.null_auroc_mean)} (p={r.null_auroc_p:.1e}) | "
          f"{_fmt(r.null_auprc_mean)} (p={r.null_auprc_p:.1e}) | {int(r.n_perm)} |")
    A(f"\nprevalence = {df.prevalence.iloc[0]:.4f} (class-balanced, AUPRC null ≈ 0.5).\n")

    A("## 2. Comparison with step0 (n=1996)  —  AUROC and Δ\n")
    A("| feature_group | probe | 400k AUROC | step0 AUROC | Δ(400k−step0) |")
    A("|---|---|---|---|---|")
    for _, r in df.iterrows():
        v0 = step0.get((r.feature_group, r.probe), np.nan)
        ok = v0 == v0
        d = r.auroc - v0 if ok else np.nan
        A(f"| {r.feature_group} | {r.probe} | {_fmt(r.auroc,4)} | "
          f"{_fmt(v0,4) if ok else ' — '} | "
          f"{('+' if d >= 0 else '') + format(d, '.4f') if d == d else ' — '} |")
    A("\ninterpretation: at n=1996 the CI is wide (±0.04); 400k narrows it to the ±0.005 level; "
      "the Δ column shows the drift direction and magnitude of the small-sample step0 estimate.\n")

    A("## 3. Accessibility Gap summary (400k)\n")
    A("| feature_group | MLP−Linear | Ridge−Linear | evo+X − evo (Linear) |")
    A("|---|---|---|---|")
    piv = df.pivot(index="feature_group", columns="probe", values="auroc")
    for g in GROUPS:
        ml = piv.loc[g, "mlp"] - piv.loc[g, "linear"]
        rl = piv.loc[g, "ridge"] - piv.loc[g, "linear"]
        gain = " — " if g == "evo" else _fmt(
            piv.loc[g, "linear"] - piv.loc["evo", "linear"], 4)
        A(f"| {g} | {_fmt(ml,4)} | {_fmt(rl,4)} | {gain} |")
    A("")

    A("## 4. paired DeLong (same test fold, all 6 pairs × 3 probes)\n")
    for pname in ["linear", "ridge", "mlp"]:
        A(f"### probe = {pname}\n")
        A("| comparison (a vs b) | AUROC_a | AUROC_b | Δ(a−b) | z | p | significant? |")
        A("|---|---|---|---|---|---|---|")
        for _, r in ddf[ddf.probe == pname].sort_values("p").iterrows():
            A(f"| {r.group_a} vs {r.group_b} | {r.auc_a:.4f} | {r.auc_b:.4f} | "
              f"{r.delta_auc:+.4f} | {r.z:+.2f} | {r.p:.2e} | "
              f"{'**yes**' if r.p < 0.05 else 'no'} |")
        A("")

    A("## 5. Notes / calibers")
    A("- MLP-2×1024 is new in this version (skipped at step0 on CPU budget); primary fit = 40 epochs with early stopping "
      "(actual epochs in the CSV `mlp_epochs_used`); null fits lightened = 10 epochs + "
      "a 50k train-row subsample (the null only needs to characterize the chance level; the primary fit is unaffected).")
    A("- kmer7 via a 13GB raw-binary memmap written directly by workers; evo fp16→fp32 converted block-wise; "
      "null-stage features shared to 8 workers via .npy mmap (no per-task 4.6GB re-serialization).")
    A("- linear high-dim groups use tol=1e-3 (crosstask_full precedent), len+gc uses tol=1e-4; "
      "`linear_nonconverged` records lbfgs hitting the iteration cap.")
    A("- ridge switched to closed-form primal (the step0 dual with K=n×n is infeasible at n=280k); "
      "the null is the exact solution from one Cholesky + a batched GEMM.")
    A("- full values: `evo_cache/Contig_500k/rap_full/rap_full_results.csv`  ·  "
      "`rap_full_delong.csv`.\n")
    args.md.write_text("\n".join(L))


if __name__ == "__main__":
    main()
