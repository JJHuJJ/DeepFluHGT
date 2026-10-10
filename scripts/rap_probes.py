#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RAP (Representation Accessibility Profiling) — DeepFluHGT Stage B-0 [P0-1].

Three-level probe on the step0 subset (n=1996): accessibility profiling of frozen Evo2 7B embeddings:
  Linear (LogisticRegression) = Readability — is the signal linearly readable
  Ridge (closed-form ridge) = Stability — robust lower bound of a linear L2 probe
  MLP-2 (1 hidden layer) = Capacity — nonlinear capacity upper bound

featuregroup: evo(4096) / len+GC(2) / RC-kmer7(8192) / evo+len+GC / evo+kmer
protocol: 70/30 stratified split (seed42) + 5-seed duplicate (SD) + 1000 timestagpermutation null
        + 1000 times bootstrap CI (AUROC/AUPRC).CPU-only.

label: the sequence-id prefix pos_/neg_ (= simulator ground truth; 100% consistent with the M0 BED labels,
      the provenance.tsv label column has 23 flipped-noise cases, see the report note).

output (evo_cache/step0_subset/rap/): rap_results.csv + rap_report.md
the report includes the Representation Validity Matrix **skeleton** — only the prespecified decision rules,
result columns left blank; pre-filling PASS is not allowed.

usage:
  python scripts/rap_probes.py [--n-perm 1000] [--mlp-n-perm 200] \
      [--n-boot 1000] [--seeds 42,43,44,45,46] [--out-dir evo_cache/step0_subset/rap]
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU-only, never touches production GPUs
# parent-process BLAS thread cap (workers would get squeezed to 1 at spawn otherwise; see main)
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import argparse
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = REPO / "evo_cache" / "step0_subset"
DEFAULT_OUT = DEFAULT_CACHE / "rap"

KMER_K = 7
RC_TAB = str.maketrans("ACGT", "TGCA")


# ------------------------------─ dataload ------------------------------─ #

def load_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    name = None
    chunks: list[str] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(chunks)
                name = line[1:].strip().split()[0]
                chunks = []
            else:
                chunks.append(line.strip().upper())
    if name is not None:
        seqs[name] = "".join(chunks)
    return seqs


def load_embeddings(path: Path) -> dict[str, np.ndarray]:
    import torch
    raw = torch.load(path, map_location="cpu", weights_only=False)
    return {k: v.astype(np.float32) for k, v in raw.items()}


def seq_stats(seq: str) -> tuple[int, float, float]:
    """return (length, GC, N%). GC denominator counts ACGT only."""
    n = len(seq)
    a = seq.count("A"); c = seq.count("C"); g = seq.count("G"); t = seq.count("T")
    acgt = a + c + g + t
    gc = (g + c) / acgt if acgt else np.nan
    n_pct = (n - acgt) / n if n else np.nan
    return n, gc, n_pct


# --------------------─ RC-canonical k-mer counting (self-implemented) --------------------─ #

_NUC2IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
_POW4 = 4 ** np.arange(KMER_K - 1, -1, -1)


def _build_canonical_table(k: int) -> np.ndarray:
    """code → dense canonical bucket (min(code, rc(code)) renumbered to [0, 4^k/2))."""
    codes = np.arange(4 ** k)
    digits = (codes[:, None] // (4 ** np.arange(k - 1, -1, -1))) % 4 # [4^k, k]
    rc = ((3 - digits[:, ::-1]) * _POW4[:k]).sum(1)
    canon = np.minimum(codes, rc)
    uniq = np.unique(canon)  # 4^k/2 representatives (palindromes included)
    return np.searchsorted(uniq, canon).astype(np.int64)


_CANON_TABLE = _build_canonical_table(KMER_K)


def rc_kmer_counts(seq: str) -> tuple[np.ndarray, int]:
    """reverse-complement canonical k-mer frequencies (4^k/2 = 8192 buckets, vectorized).

    a k-mer and its RC map to the same bucket (the smaller integer code wins); windows containing
    windowskip.return (counts, total_valid_kmers).
    """
    from numpy.lib.stride_tricks import sliding_window_view
    k = KMER_K
    n_bucket = 4 ** k // 2
    lut = np.full(256, -1, dtype=np.int64)
    for b, i in _NUC2IDX.items():
        lut[ord(b)] = i
    idx = lut[np.frombuffer(seq.encode(), dtype=np.uint8)]
    if len(idx) < k:
        return np.zeros(n_bucket), 0
    win = sliding_window_view(idx, k) # [n-k+1, k]
    valid = (win >= 0).all(axis=1)
    wv = np.where(win[valid] >= 0, win[valid], 0)  # neutralize -1 so it cannot corrupt the encoding
    fwd = wv @ _POW4
    rev = (3 - wv[:, ::-1]) @ _POW4
    canon_code = np.minimum(fwd, rev)  # representative within [0, 4^k)
    bucket = _CANON_TABLE[canon_code] # dense [0, 8192)
    counts = np.bincount(bucket, minlength=n_bucket).astype(np.float64)
    return counts, int(valid.sum())


def kmer_feature(seqs: dict[str, str], ids: list[str]) -> np.ndarray:
    """count + normalize (relative frequency)."""
    mat = np.zeros((len(ids), 4 ** KMER_K // 2), dtype=np.float32)
    for i, sid in enumerate(ids):
        c, tot = rc_kmer_counts(seqs[sid])
        mat[i] = c / tot if tot else 0.0
    return mat


# ------------------------------─ probes ------------------------------─ #

def probe_linear(xtr, ytr, xte):
    """Readability: logistic regression (linear probe)."""
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m = LogisticRegression(C=1.0, max_iter=2000, tol=1e-4, solver="lbfgs")
        m.fit(xtr, ytr)
    probe_linear.n_nonconverged = int(np.ravel(getattr(m, "n_iter_", [0]))[0] >= 2000)
    return m.predict_proba(xte)[:, 1]


def _ridge_fit_predict(xtr, ytr, xte, alpha: float = 1.0):
    """Stability: ridge regression on the original labels y∈{0,1}; scores used for ranking metrics."""
    mu = xtr.mean(0, keepdims=True); sd = xtr.std(0) + 1e-12
    xtr_s = (xtr - mu) / sd; xte_s = (xte - mu) / sd
    yc = ytr - ytr.mean()
    k = xtr_s @ xtr_s.T + alpha * np.eye(len(xtr_s))
    coef = np.linalg.solve(k, yc)
    return xte_s @ xtr_s.T @ coef + ytr.mean()


def _torch_mlp(xtr, ytr, xte, seed: int, hidden=256, epochs=200, patience=20, lr=1e-3):
    """Capacity ceiling: 2 layer MLP (Linear→ReLU→Dropout→Linear), CPU, early stopping."""
    import torch
    import torch.nn as nn
    if os.environ.get("_RAP_WORKER") == "1":
        torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(xtr)
    idx = rng.permutation(n)
    n_val = max(int(n * 0.15), 20)
    val_i, fit_i = idx[:n_val], idx[n_val:]
    xt = torch.from_numpy(xtr)
    dev = "cpu"
    model = nn.Sequential(
        nn.Linear(xtr.shape[1], hidden), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(hidden, 1)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    best = np.inf; best_state = None; bad = 0
    for _ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = lossf(model(xt[fit_i]).squeeze(-1), ytr_t[fit_i])
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(lossf(model(xt[val_i]).squeeze(-1), ytr_t[val_i]))
        if vloss < best - 1e-4:
            best = vloss; bad = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        s = torch.sigmoid(model(torch.from_numpy(xte)).squeeze(-1)).numpy()
    return np.asarray(s, dtype=np.float64)


PROBES = {"linear": probe_linear, "ridge": _ridge_fit_predict, "mlp": _torch_mlp}


# ------------------------ metric / null / bootstrap ------------------------ #

def metrics(y, s) -> tuple[float, float]:
    from sklearn.metrics import roc_auc_score, average_precision_score
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def bootstrap_ci(y, s, n_boot=1000, seed=0):
    """test-set non-parameters bootstrap → AUROC/AUPRC 95% CI."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs, aps = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        yy, ss = y[i], s[i]
        if yy.min() == yy.max():
            continue
        aucs.append(roc_auc_score(yy, ss)); aps.append(average_precision_score(yy, ss))
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return q(aucs), q(aps)


def auroc_cols(Y: np.ndarray, S: np.ndarray) -> np.ndarray:
    """per-column (permutation) vectorized AUROC; S: [n, m] scores, Y: [n, m] permuted labels.

    the rank approximation ignores ties (continuous probe scores rarely tie); n_pos/n_neg>0 is guaranteed upfront by stratification.
    """
    n = S.shape[0]
    order = np.argsort(S, axis=0, kind="mergesort")
    r = np.empty_like(S, dtype=np.float64)
    np.put_along_axis(r, order, np.broadcast_to(
        np.arange(1, n + 1)[:, None], S.shape).astype(np.float64), axis=0)
    pos = Y == 1
    n_pos = pos.sum(0)
    n_neg = n - n_pos
    sum_ranks_pos = np.where(pos, r, 0.0).sum(0)
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / np.maximum(n_pos * n_neg, 1)


def ridge_null_batched(xtr, ytr_perm: np.ndarray, xte, alpha=1.0):
    """dual-form ridge: one decomposition, batch-solve the test scores of m permuted label sets."""
    mu = xtr.mean(0, keepdims=True); sd = xtr.std(0) + 1e-12
    a = (xtr - mu) / sd; b = (xte - mu) / sd
    yc = ytr_perm - ytr_perm.mean(0, keepdims=True) # [n, m]
    k = a @ a.T + alpha * np.eye(len(a))
    chol = np.linalg.cholesky(k)
    sol = np.linalg.solve(chol.T, np.linalg.solve(chol, yc))
    return b @ a.T @ sol + ytr_perm.mean(0, keepdims=True) # [n_te, m]


# -------------------------------- main pipeline -------------------------------- #

def stratified_split(y, seed, frac=0.30):
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=frac, stratify=y, random_state=seed)
    return np.sort(tr), np.sort(te)


def _null_one(pname: str, xtr, xte, ytr_perm, yte_perm, seed: int):
    """fit+eval for one permuted label set (module level so arrays pass as parameters → joblib auto-memmap)."""
    if pname == "mlp":
        s = _torch_mlp(xtr, ytr_perm, xte, seed=seed)
    else:
        s = PROBES[pname](xtr, ytr_perm, xte)
    return metrics(yte_perm, s)


def _init_worker() -> None:
    """joblib worker init: strict single-threading (BLAS spin-wait oversubscription once caused a 25x slowdown)."""
    import torch
    os.environ["_RAP_WORKER"] = "1"
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass
    torch.set_num_threads(1)


def build_features(emb, seqs, ids):
    st = np.array([seq_stats(seqs[sid]) for sid in ids], dtype=np.float64)
    feats = {
        "evo": np.stack([emb[sid] for sid in ids]).astype(np.float32),
        "len+gc": np.stack([np.log10(st[:, 0]), st[:, 1]], 1).astype(np.float32),
        "kmer7_rc": kmer_feature(seqs, ids),
        "evo+len+gc": None, "evo+kmer7_rc": None,
    }
    feats["evo+len+gc"] = np.concatenate([feats["evo"], feats["len+gc"]], 1)
    feats["evo+kmer7_rc"] = np.concatenate([feats["evo"], feats["kmer7_rc"]], 1)
    meta = {"length": st[:, 0], "gc": st[:, 1], "n_pct": st[:, 2]}
    return feats, meta


def standardize(xtr, xte):
    mu = xtr.mean(0, keepdims=True)
    sd = xtr.std(0) + 1e-12
    return (xtr - mu) / sd, (xte - mu) / sd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--mlp-n-perm", type=int, default=200,
                    help="MLP permutation count is capped by the CPU budget (linear/ridge still use --n-perm)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seeds", type=str, default="42,43,44,45,46")
    ap.add_argument("--n-jobs", type=int, default=48,
                    help="permutation-null parallel worker count (shared server, default 48)")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(8)  # be polite on the shared server; a single fit already takes <5s

    t_start = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    seqs = load_fasta(args.cache_dir / "step0_subset.fasta")
    emb = load_embeddings(args.cache_dir / "evo_subset.pt")
    ids = list(seqs.keys())
    missing = [i for i in ids if i not in emb]
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} ids missing embedding, such as {missing[:3]}")
    y = np.array([1 if i.startswith("pos_") else 0 for i in ids], dtype=int)

    feats, meta = build_features(emb, seqs, ids)
    groups = list(feats.keys())
    print(f"[rap] n={len(ids)} pos={int(y.sum())} neg={int((1 - y).sum())}; "
          f"groups={groups}")

    seeds = [int(s) for s in args.seeds.split(",")]
    rng = np.random.default_rng(42)
    perm_ys = np.stack([rng.permutation(y) for _ in range(args.n_perm)]) # [m, n]

    # force worker single-threading: set the env before the loky spawn (children inherit), preventing MKL thread pools
    # from spin-wait oversubscription (measured to cause ~25x slowdown)
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"

    rows = []
    for gname in groups:
        X = feats[gname]
        for pname, pfn in PROBES.items():
            t0 = time.time()
            tr, te = stratified_split(y, seed=42)
            xtr, xte = standardize(X[tr], X[te])
            n_perm = args.mlp_n_perm if pname == "mlp" else args.n_perm

            # ① primary split (seed42) + bootstrap CI
            if pname == "mlp":
                s = _torch_mlp(xtr, y[tr], xte, seed=42)
            else:
                s = pfn(xtr, y[tr], xte)
            auroc, auprc = metrics(y[te], s)
            (al, ah), (pl, ph) = bootstrap_ci(y[te], s, args.n_boot, seed=42)

            # ② 5-seed duplicate → SD
            rep_auroc, rep_auprc = [auroc], [auprc]
            for sd_ in seeds[1:]:
                tr2, te2 = stratified_split(y, seed=sd_)
                xtr2, xte2 = standardize(X[tr2], X[te2])
                if pname == "mlp":
                    s2 = _torch_mlp(xtr2, y[tr2], xte2, seed=sd_)
                else:
                    s2 = pfn(xtr2, y[tr2], xte2)
                a2, p2 = metrics(y[te2], s2)
                rep_auroc.append(a2); rep_auprc.append(p2)

            # permutation null (labels permuted wholesale, reusing the seed42 split)
            null_a, null_p = [], []
            if pname == "ridge":
                # closed-form batch solve: exact and iteration-free
                scores = ridge_null_batched(xtr, perm_ys[:, tr].T, xte) # [n_te, m]
                null_a = list(auroc_cols(perm_ys[:, te].T, scores))
                from sklearn.metrics import average_precision_score
                for j in range(scores.shape[1]):
                    null_p.append(average_precision_score(perm_ys[j, te], scores[:, j]))
            else:
                from joblib import Parallel, delayed

                res = Parallel(n_jobs=args.n_jobs, initializer=_init_worker)(
                    delayed(_null_one)(pname, xtr, xte,
                                       perm_ys[j][tr], perm_ys[j][te],
                                       1000 + j)
                    for j in range(n_perm))
                null_a = [r[0] for r in res]
                null_p = [r[1] for r in res]

            p_au = (1 + int(np.sum(np.asarray(null_a) >= auroc))) / (1 + len(null_a))
            p_ap = (1 + int(np.sum(np.asarray(null_p) >= auprc))) / (1 + len(null_p))

            rows.append(dict(
                feature_group=gname, probe=pname, role={"linear": "Readability",
                "ridge": "Stability", "mlp": "Capacity ceiling"}[pname],
                n_train=int(len(tr)), n_test=int(len(te)), prevalence=float(y[te].mean()),
                auroc=auroc, auroc_ci95=f"[{al:.4f},{ah:.4f}]",
                auprc=auprc, auprc_ci95=f"[{pl:.4f},{ph:.4f}]",
                auroc_seed_mean=float(np.mean(rep_auroc)), auroc_seed_sd=float(np.std(rep_auroc, ddof=1)),
                auprc_seed_mean=float(np.mean(rep_auprc)), auprc_seed_sd=float(np.std(rep_auprc, ddof=1)),
                null_auroc_mean=float(np.mean(null_a)), null_auroc_sd=float(np.std(null_a)),
                null_auroc_p=p_au, null_auprc_mean=float(np.mean(null_p)),
                null_auprc_sd=float(np.std(null_p)), null_auprc_p=p_ap,
                n_perm=len(null_a), n_boot=args.n_boot,
                linear_nonconverged=int(getattr(probe_linear, "n_nonconverged", 0))
                if pname == "linear" else 0,
                runtime_s=round(time.time() - t0, 1)))
            r = rows[-1]
            print(f"[rap] {gname:14s} {pname:7s} AUROC={auroc:.3f} "
                  f"(null {np.mean(null_a):.3f}, p={p_au:.4f}) AUPRC={auprc:.3f} "
                  f"seedSD={np.std(rep_auroc, ddof=1):.3f} [{r['runtime_s']}s]",
                  flush=True)

            # incremental write to disk (checkpoint insurance): write as soon as each group x probe finishes
            import pandas as pd
            pd.DataFrame(rows).to_csv(args.out_dir / "rap_results.csv", index=False)

    # -- written to disk --
    import pandas as pd
    df = pd.DataFrame(rows)
    csv_path = args.out_dir / "rap_results.csv"
    df.to_csv(csv_path, index=False)

    _write_report(args, df, ids, y, meta, time.time() - t_start)
    print(f"[rap] done -> {csv_path} ({time.time() - t_start:.0f}s total)")


def _fmt(v, nd=3):
    return " — " if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def _write_report(args, df, ids, y, meta, total_s) -> None:
    prevalence = float(y.mean())
    lines = []
    A = lines.append
    A("# RAP — Representation Accessibility Profiling (v7.5 Stage B-0)\n")
    A(f"- generatetimeinterval: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    A(f"- data: `evo_cache/step0_subset` (n={len(ids)}, pos={int(y.sum())}, "
      f"neg={int(len(y) - int(y.sum()))})")
    A(f"- modality: Evo2 7B frozen contig-mean embedding (4096d, fp16 cache → fp32)")
    A("- label: the sequence-id prefix pos_/neg_ (= simulator ground truth; 100% consistent with the Contig_500k train BED label column;"
      " the provenance.tsv label column has 23 additional flipped cases, see the dataset fingerprint)")
    A("- protocol: 70/30 stratified (seed42) + 5-seed duplicate (42–46) + tagpermutation null "
      f"(linear/ridge={args.n_perm}, mlp={args.mlp_n_perm}, capped by CPU budget) + "
      f"bootstrap {args.n_boot} CI")
    A("- probes: Linear=LogisticRegression(C=1) → *Readability*; "
      "Ridge(α=1, closed-form) → *Stability*; MLP-2(4096→256→ReLU→Dropout0.3→1, early stopping) → *Capacity ceiling*")
    A("- features: evo(4096) / len+GC(log10 len, GC) / RC-kmer7(8192 canonical, relative frequency), plus concatenations\n")

    A("## 1. primaryresult (seed42 split; 95% bootstrap CI)\n")
    A("| feature_group | probe (role) | AUROC [CI] | AUPRC [CI] | null AUROC (p) | null AUPRC (p) |")
    A("|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        A(f"| {r.feature_group} | {r.probe} ({r.role}) | "
          f"{_fmt(r.auroc)} {r.auroc_ci95} | {_fmt(r.auprc)} {r.auprc_ci95} | "
          f"{_fmt(r.null_auroc_mean)} (p={r.null_auroc_p:.1e}) | "
          f"{_fmt(r.null_auprc_mean)} (p={r.null_auprc_p:.1e}) |")
    A(f"\nprevalence ≈ {prevalence:.3f} (balanced classes, AUPRC null ≈ 0.5).\n")

    A("## 2. Stability (5-seed mean ± SD)\n")
    A("| feature_group | probe | AUROC mean±SD | AUPRC mean±SD |")
    A("|---|---|---|---|")
    for _, r in df.iterrows():
        A(f"| {r.feature_group} | {r.probe} | {_fmt(r.auroc_seed_mean)} ± {_fmt(r.auroc_seed_sd)} | "
          f"{_fmt(r.auprc_seed_mean)} ± {_fmt(r.auprc_seed_sd)} |")
    A("")

    A("## 3. Accessibility Gap summary (descriptive; decisions follow the §4 rules)\n")
    A("| feature_group | MLP−Linear (AUROC) | Ridge−Linear (AUROC) | evo concatenation gain (evo+X − evo) |")
    A("|---|---|---|---|")
    piv = df.pivot(index="feature_group", columns="probe", values="auroc")
    for g in piv.index:
        ml = piv.loc[g, "mlp"] - piv.loc[g, "linear"] if "mlp" in piv.columns else np.nan
        rl = piv.loc[g, "ridge"] - piv.loc[g, "linear"] if "ridge" in piv.columns else np.nan
        gain = " — " if g in ("evo",) else _fmt(piv.loc[g, "linear"] - piv.loc["evo", "linear"])
        A(f"| {g} | {_fmt(ml)} | {_fmt(rl)} | {gain} |")
    A("")

    A("## 4. Representation Validity Matrix (skeleton — prespecified rules, results left blank)\n")
    A("> result columns are filled in by the downstream evidence review; this script does **not** output PASS/FAIL verdicts.\n")
    A("| ID | check item | automaticdeterminationrulethen (prespecified) | result | determination |")
    A("|---|---|---|---|---|")
    A("| RVM-1 | Q1a linear readability: Linear AUROC on evo | AUROC ≥ 0.60 ∧ perm p < 0.001 → SIGNAL; 0.55–0.60 → WEAK; < 0.55 → NO-LINEAR-SIGNAL | — | — |")
    A("| RVM-2 | Q1b stability: 5-seed SD vs Ridge consistency | SD(AUROC) ≤ 0.05 ∧ Ridge AUROC ≥ Linear − 0.05 → STABLE; else UNSTABLE | — | — |")
    A("| RVM-3 | Q1c nonlinear gap: MLP − Linear | ΔAUROC ≥ 0.05 → NONLINEAR-GAP; < 0.02 → LINEAR-SATURATED; in between MARGINAL | — | — |")
    A("| RVM-4 | Q2 shortcut: evo→GC (shortcut_audit) | R² ≤ 0.30 → LOW; 0.30–0.60 → MODERATE; > 0.60 → HIGH | — | — |")
    A("| RVM-5 | Q2 shortcut: evo→log len | same RVM-4 threshold | — | — |")
    A("| RVM-6 | Q2 shortcut: evo→taxonomy proxy macro-F1 | (macro-F1 − null mean) ≤ 0.10 → LOW; 0.10–0.25 → MODERATE; > 0.25 → HIGH | — | — |")
    A("| RVM-7 | feature gain: evo+X − evo (Linear AUROC) | Δ ≥ 0.02 → X adds signal; < 0.02 → no gain | — | — |")
    A("| RVM-8 | composition baseline control: kmer7_rc vs evo | kmer AUROC ≥ 0.90 and evo < kmer − 0.02 → COMPOSITION-DOMINANT risk | — | — |")
    A("| RVM-9 | Q1xQ2 joint (overall validity verdict) | RVM-1 SIGNAL ∧ RVM-4/5/6 all ≤ MODERATE → REPRESENTATION-VALID; else SHORTCUT-CONFOUNDED | — | — |")
    A("")

    A("## 5. note\n")
    A(f"- runtime environment: CPU only (CUDA_VISIBLE_DEVICES=\"\"), total runtime {total_s:.0f}s.")
    A("- the MLP permutation-null count differs from linear/ridge (CPU budget); n_perm is annotated in table 1.")
    A("- full values in `rap_results.csv` (same directory).\n")
    (args.out_dir / "rap_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
