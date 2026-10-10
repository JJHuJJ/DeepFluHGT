#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B1 cross-task RAP  —  FULL probe (NTv3 + Evo2 dual FM; DeepFluHGT)

Run RAP probes over 5 feature groups on the 5 Feng 2025 datasets (test split):
  ntv3_1536   frozen NTv3-pre hidden mean-pool (the variant produced by embed_ntv3_cpu.py)
  evo_4096    frozen Evo2-7B blocks.28.mlp.l3 mean-pool
              (embed_crosstask_evo2_4090.py → {ds}_evo.npy)
  evo+ntv3    concatenation of both FM embeddings (5632d)  —  —  fusion accessibility
  len+gc      log10(len)+GC (2d composition-shortcut baseline)
  kmer7_rc    RC-canonical 7-mer relative frequency (8192d composition baseline)

Protocol fully consistent with crosstask_ntv3_probe.py (reuses rap_probes's
split/standardize/probe/bootstrap + delong_test paired test):
  70/30 stratified split within the test pool (seed42) + 5 seeds (42-46) +
  label-permutation null + bootstrap CI.
Sample alignment: the 644 all-N exclusions and within-pool duplicate statistics for
human_vs_worm follow the same logic; all feature groups exclude the same pool
(row_idx aligns the evo npy with the ntv3 npz).

High-dimensional null budget: linear null fits use tol=1e-3 (for n>5k, same precedent
as the NTv3 version); a per-cell null wall-clock cap is also enforced (default 1200s,
early stopping with the actual n_perm written to the CSV  —  —  the null sits at the
chance level under permuted labels and is insensitive to the count; when the count is
insufficient, the p-value floor = 1/(n_perm+1) is recorded as-is).

output (benchmark/crosstask/):
  crosstask_full_results.csv / crosstask_full_delong.csv / CROSTASK_FULL_RAP.md

usage: python scripts/crosstask_full_probe.py [--datasets ds1,ds2]
      python scripts/crosstask_full_probe.py --report-only
"""
from __future__ import annotations

import os
# Hard constraint: CPU-only + BLAS threads ≤8 (must be set before importing numpy/sklearn)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "8"

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rap_probes as rap
from delong_test import delong_test_full

CROSS = REPO / "benchmark" / "crosstask"
DATASETS = ["human_vs_worm", "promoter_b_amyloliquefaciens", "promoter_r_capsulatus",
            "ecoli_4mc", "celegans_4mc"]
GROUPS = ["ntv3_1536", "evo_4096", "evo+ntv3", "len+gc", "kmer7_rc"]
PROBES = ["linear", "ridge"]
DELONG_PAIRS = [("evo_4096", "ntv3_1536"),
                ("evo+ntv3", "ntv3_1536"), ("evo+ntv3", "evo_4096"),
                ("ntv3_1536", "len+gc"), ("evo_4096", "len+gc"),
                ("ntv3_1536", "kmer7_rc"), ("evo_4096", "kmer7_rc")]
TASK = {
    "human_vs_worm": ("species discrimination (200bp window)", "H. sapiens vs C. elegans", "AUROC"),
    "promoter_b_amyloliquefaciens": ("promoter recognition (40bp)", "B. amyloliquefaciens", "AUROC"),
    "promoter_r_capsulatus": ("promoter recognition (40bp)", "R. capsulatus", "AUROC"),
    "ecoli_4mc": ("4mC methylation (41bp, imbalanced 16.7%)", "E. coli K-12", "AUPRC (primary)"),
    "celegans_4mc": ("4mC methylation (41bp, balanced)", "C. elegans", "AUROC"),
}
NULL_WALLCLOCK_S = 1200.0


def n_perm_ladder(n_train: int, probe: str) -> int:
    if probe == "ridge":
        return 1000
    return 1000 if n_train <= 5000 else 100


def probe_linear_null(xtr, ytr, xte, tol=1e-3):
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m = LogisticRegression(C=1.0, max_iter=2000, tol=tol, solver="lbfgs")
        m.fit(xtr, ytr)
    return m.predict_proba(xte)[:, 1]


def ridge_null_batched(xtr, ytr_perm, xte, alpha=1.0):
    mu = xtr.mean(0, keepdims=True); sd = xtr.std(0) + 1e-12
    a = (xtr - mu) / sd; b = (xte - mu) / sd
    yc = ytr_perm - ytr_perm.mean(0, keepdims=True)
    d = a.shape[1]
    g = a.T @ a + alpha * np.eye(d)
    g[np.diag_indices(d)] += 1e-9 * max(1.0, float(np.trace(g)) / d)
    chol = np.linalg.cholesky(g)
    w = np.linalg.solve(chol.T, np.linalg.solve(chol, a.T @ yc))
    return b @ w + ytr_perm.mean(0, keepdims=True)


def _collect_meta(ds: str, extra: dict) -> tuple:
    """Load fasta + ntv3 npz + evo npy with same-pool alignment (all-N exclusion unified across all groups)."""
    ddir = CROSS / ds
    seqs = rap.load_fasta(ddir / "rap_input.fasta")
    z = np.load(ddir / f"{ds}_ntv3_pooled.npz", allow_pickle=False)
    ids = [str(s) for s in z["ids"]]
    assert ids == list(seqs.keys()), f"{ds}: npz ids do not match fasta order"
    y = z["labels"].astype(int)
    evo = np.load(ddir / f"{ds}_evo.npy").astype(np.float32)
    assert evo.shape[0] == len(ids), f"{ds}: evo npy rows {evo.shape[0]} != {len(ids)}"
    st = np.array([rap.seq_stats(seqs[i]) for i in ids])
    acgt_ok = ~np.isnan(st[:, 1])
    n_alln = int((~acgt_ok).sum())
    row_idx = np.where(acgt_ok)[0]
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
    p1536 = z["pooled_1536"].astype(np.float32)[row_idx]
    evo = evo[row_idx]
    feats = {
        "ntv3_1536": p1536,
        "evo_4096": evo,
        "evo+ntv3": np.concatenate([evo, p1536], axis=1),
        "len+gc": np.stack([np.log10(st[:, 0]), st[:, 1]], 1).astype(np.float32),
        "kmer7_rc": rap.kmer_feature(seqs, ids),
    }
    evo_meta = json.loads((ddir / f"{ds}_evo_manifest.json").read_text())
    extra[ds] = dict(n=len(ids), n_pool=int(len(z["ids"])), pos=int(y.sum()),
                     n_dup=int((~keep).sum()), n_allN_dropped=n_alln,
                     gc_mean=float(np.nanmean(st[:, 1])),
                     n_pct=float(np.nanmean(st[:, 2])),
                     task=TASK[ds][0], org=TASK[ds][1], prim=TASK[ds][2],
                     prev=float(y.mean()), evo_layer=evo_meta["layer"],
                     evo_elapsed_s=evo_meta.get("elapsed_s"))
    return seqs, ids, y, st, keep, feats


def run_dataset(ds: str, seeds, rows, drows, extra):
    seqs, ids, y, st, keep, feats = _collect_meta(ds, extra)
    print(f"\n[probe] {ds}: n={len(ids)} pos={int(y.sum())} "
          f"within-pool dups={extra[ds]['n_dup']} GC={extra[ds]['gc_mean']:.3f}", flush=True)

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

            # ── permutation null (with wall-clock-cap early stopping) ──
            n_perm = n_perm_ladder(len(tr), pname)
            null_a, null_p = [], []
            t_null0 = time.time()
            if pname == "ridge":
                perm_ys = np.stack([rng.permutation(y) for _ in range(n_perm)])
                sc = ridge_null_batched(xtr, perm_ys[:, tr].T, xte)
                from sklearn.metrics import average_precision_score
                null_a = list(rap.auroc_cols(perm_ys[:, te].T, sc))
                null_p = [average_precision_score(perm_ys[j, te], sc[:, j])
                          for j in range(n_perm)]
            else:
                fast = len(tr) > 5000
                done_perm = 0
                for j in range(n_perm):
                    if done_perm and done_perm % 10 == 0 and \
                            time.time() - t_null0 > NULL_WALLCLOCK_S:
                        print(f"    [null] {g}/{pname} wall-clock cap triggered: "
                              f"{done_perm}/{n_perm}", flush=True)
                        break
                    yperm = rng.permutation(y)
                    sj = (probe_linear_null(xtr, yperm[tr], xte, tol=1e-3) if fast
                          else pfn(xtr, yperm[tr], xte))
                    a_, p_ = rap.metrics(yperm[te], sj)
                    null_a.append(a_); null_p.append(p_)
                    done_perm += 1
                n_perm = done_perm
            p_au = (1 + int(np.sum(np.asarray(null_a) >= auroc))) / (1 + n_perm)
            p_ap = (1 + int(np.sum(np.asarray(null_p) >= auprc))) / (1 + n_perm)

            rows.append(dict(
                dataset=ds, feature_group=g, probe=pname,
                role={"linear": "Readability", "ridge": "Stability"}[pname],
                n_train=int(len(tr)), n_test=int(len(te)),
                prevalence=float(yte.mean()), primary_metric=TASK[ds][2],
                auroc=auroc, auroc_ci95=f"[{al:.4f},{ah:.4f}]",
                auprc=auprc, auprc_ci95=f"[{pl:.4f},{ph:.4f}]",
                auroc_seed_mean=float(np.mean(rep_a)),
                auroc_seed_sd=float(np.std(rep_a, ddof=1)),
                auprc_seed_mean=float(np.mean(rep_p)),
                auprc_seed_sd=float(np.std(rep_p, ddof=1)),
                null_auroc_mean=float(np.mean(null_a)),
                null_auprc_p=p_au, null_auprc_mean=float(np.mean(null_p)),
                null_auroc_p=p_ap,
                n_perm=n_perm, n_boot=n_boot,
                runtime_s=round(time.time() - t0, 1)))
            r = rows[-1]
            print(f"  {g:10s} {pname:7s} AUROC={auroc:.3f} AUPRC={auprc:.3f} "
                  f"(nullA p={p_au:.1e}, nullP p={p_ap:.1e}, n_perm={n_perm}) "
                  f"[{r['runtime_s']}s]", flush=True)

    for pname in PROBES:
        for ga, gb in DELONG_PAIRS:
            d_ = delong_test_full(yte, preds[(ga, pname)], preds[(gb, pname)])
            drows.append(dict(dataset=ds, probe=pname, group_a=ga, group_b=gb,
                              auc_a=d_["auc_a"], auc_b=d_["auc_b"],
                              delta_auc=d_["delta"], z=d_["z"], p=d_["p"],
                              n=d_["n"], n_pos=d_["n_pos"]))
    return extra[ds]


def write_report(df, ddf, extra, total_s):
    import pandas as pd
    def g3(ds, g, p, m):
        r = df[(df.dataset == ds) & (df.feature_group == g) & (df.probe == p)]
        return r.iloc[0][m] if len(r) else np.nan

    def pp(p):
        return "<1e-300" if p == 0 else (f"{p:.2e}" if p < 0.001 else f"{p:.3f}")

    L = []
    A = L.append
    A("# B1 cross-task RAP  —  FULL (NTv3 + Evo2 dual FM) results report (CROSTASK_FULL_RAP)")
    A("")
    A(f"> generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  ·  DeepFluHGT  ·  "
      "corresponds to V7_5_EVIDENCE_PLAN §B1 (completing the Evo2 half; predecessor: CROSTASK_NTV3_RAP.md)")
    A("> embedding: NTv3-pre 1536d (CPU, embed_ntv3_cpu.py) + Evo2-7B "
      "blocks.28.mlp.l3 4096d (4090, embed_crosstask_evo2_4090.py); "
      "probe: crosstask_full_probe.py (reuses rap_probes/delong_test)")
    A("")
    A("## 0. caliber")
    A("")
    A("- feature groups, 5 columns: **NTv3** / **Evo2** / **evo+ntv3** (concatenated 5632d) / len+GC / RC-kmer7;")
    A("- Protocol fully consistent with the NTv3 half: 70/30 stratified split within the "
      "test pool (seed42) + 5 seeds (42-46) + "
      "permutation null + bootstrap CI + paired DeLong (same test fold);")
    A("- human_vs_worm's 644 all-N items and within-pool duplicate statistics follow the "
      "NTv3 version's same-pool processing (uniform exclusion across all feature groups); "
      "ecoli_4mc primary metric AUPRC;")
    A("- linear null for high dimensions (4096/5632d): tol=1e-3 early stopping for n>5k "
      "(same precedent as the NTv3 version), "
      f"plus a per-cell {int(NULL_WALLCLOCK_S)}s wall-clock cap with early stopping; the "
      "actual n_perm is recorded in the CSV "
      "(the null sits at the chance level under permuted labels and is insensitive to the "
      "count; p-value floor = 1/(n_perm+1)).")
    A("")
    A("## 1. Five-column overview (linear probe; AUROC, ecoli_4mc additionally reported as AUPRC)")
    A("")
    A("| dataset | NTv3-1536 | Evo2-4096 | evo+ntv3 | len+GC | kmer7 | best |")
    A("|---|---|---|---|---|---|---|")
    for ds in DATASETS:
        if ds not in extra or not len(df[df.dataset == ds]):
            continue
        vals = [g3(ds, g, "linear", "auroc") for g in GROUPS]
        best = GROUPS[int(np.nanargmax(vals))]
        row = f"| {ds} | " + " | ".join(f"{v:.3f}" if np.isfinite(v) else " — "
                                        for v in vals)
        row += f" | {best} |"
        A(row)
    A("")
    A("ecoli_4mc, same table (AUPRC metric, primary metric):")
    A("")
    A("| dataset | NTv3-1536 | Evo2-4096 | evo+ntv3 | len+GC | kmer7 |")
    A("|---|---|---|---|---|---|")
    for ds in DATASETS:
        if ds not in extra or not len(df[df.dataset == ds]):
            continue
        vals = [g3(ds, g, "linear", "auprc") for g in GROUPS]
        A(f"| {ds} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    A("")
    A("## 2. Per-dataset details (linear; ridge in the CSV)")
    for ds in DATASETS:
        if ds not in extra:
            continue
        e = extra[ds]
        A("")
        A(f"### {ds}  —  {e['task']} ({e['org']})  ·  n={e['n']} "
          f"(pos={e['pos']}, prevalence={e['prev']:.3f})  ·  primary metric: {e['prim']}")
        A("")
        A("| feature_group | AUROC [CI] | AUPRC [CI] | seed SD (AUC) | null AUROC (p) |")
        A("|---|---|---|---|---|")
        for g in GROUPS:
            r = df[(df.dataset == ds) & (df.feature_group == g) & (df.probe == "linear")]
            if not len(r):
                continue
            r = r.iloc[0]
            A(f"| {g} | {r.auroc:.3f} {r.auroc_ci95} | {r.auprc:.3f} {r.auprc_ci95} "
              f"| {r.auroc_seed_sd:.3f} | {r.null_auroc_mean:.3f} ({pp(r.null_auroc_p)}) |")
        A("")
        dd = ddf[(ddf.dataset == ds) & (ddf.probe == "linear")]
        A("**DeLong (linear, same test fold)**: " + "; ".join(
            f"{r.group_a}−{r.group_b}={r.delta_auc:+.3f} (z={r.z:+.2f}, p={pp(r.p)})"
            for r in dd.itertuples()))
    A("")
    A("## 3. paired DeLong alltable (linear)")
    A("")
    A("| dataset | comparison (a vs b) | AUROC_a | AUROC_b | Δ(a−b) | z | p | significant? |")
    A("|---|---|---|---|---|---|---|---|")
    for r in ddf[ddf.probe == "linear"].itertuples():
        A(f"| {r.dataset} | {r.group_a} vs {r.group_b} | {r.auc_a:.3f} | "
          f"{r.auc_b:.3f} | {r.delta_auc:+.3f} | {r.z:+.2f} | {pp(r.p)} | "
          f"{'**yes**' if r.p < 0.05 else 'no'} |")
    A("")
    A("## 4. Interpretation notes")
    A("")
    A("- Evo2-4096 and NTv3-1536 are comparable columns under the **same pool and same "
      "protocol**; evo+ntv3 tests the linear complementarity of the two FMs' "
      "representations (concatenation gains ⇒ complementary information, not redundancy).")
    A("- embeddings are zero-training representations from frozen FMs; probes are "
      "linear/ridge only (accessibility, not a performance ceiling).")
    A("- On the 40-41bp tasks, kmer7 features are sparse (34-35 k-mers per sequence; "
      "see the NTv3 version §5 precedent).")
    A("- Runtime environment: probes CPU-only (BLAS ≤8 threads); Evo2 embeddings done "
      "on the 4090 (GPU lock).")
    A(f"- Full numeric values: crosstask_full_results.csv / crosstask_full_delong.csv "
      f"(this directory); probe total runtime {total_s/60:.1f} min.")
    (CROSS / "CROSTASK_FULL_RAP.md").write_text("\n".join(L) + "\n")


def _merged_save(frame, path, subset):
    """Concurrency-safe flush: read the existing CSV → merge → drop_duplicates(keep last) → write.

    When multiple processes run different datasets in parallel, each holds only its
    own rows; a direct to_csv would overwrite the whole file (clobber). The
    merge-then-write guarantees rows from datasets that finished first are not lost."""
    import pandas as pd
    if path.exists():
        old = pd.read_csv(path)
        frame = pd.concat([old, frame], ignore_index=True)
    frame.drop_duplicates(subset=subset, keep="last").to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    import pandas as pd
    seeds = [int(s) for s in args.seeds.split(",")]
    names = DATASETS if args.datasets == "all" else args.datasets.split(",")

    if args.report_only:
        df = pd.read_csv(CROSS / "crosstask_full_results.csv")
        ddf = pd.read_csv(CROSS / "crosstask_full_delong.csv")
        extra = {}
        for ds in DATASETS:
            try:
                _collect_meta(ds, extra)
            except FileNotFoundError:
                pass
        write_report(df, ddf, extra, 0.0)
        print("[report] CROSTASK_FULL_RAP.md alreadygenerate")
        return

    t0 = time.time()
    rows, drows, extra = [], [], {}
    if args.resume:
        for path, sink in ((CROSS / "crosstask_full_results.csv", rows),
                           (CROSS / "crosstask_full_delong.csv", drows)):
            if path.exists():
                old = pd.read_csv(path).to_dict("records")
                sink.extend(old)
                print(f"[probe] resume: loaded {len(old)} rows", flush=True)
    done = {r.get("dataset") for r in rows}

    for ds in names:
        if ds in done:
            print(f"[probe] {ds} already has results, skipped (resume)", flush=True)
            continue
        run_dataset(ds, seeds, rows, drows, extra)
        _merged_save(pd.DataFrame(rows), CROSS / "crosstask_full_results.csv",
                     ["dataset", "feature_group", "probe"])
        _merged_save(pd.DataFrame(drows), CROSS / "crosstask_full_delong.csv",
                     ["dataset", "probe", "group_a", "group_b"])

    for ds in names:
        if ds not in extra:
            _collect_meta(ds, extra)
    df = pd.DataFrame(rows).drop_duplicates(
        subset=["dataset", "feature_group", "probe"], keep="last")
    ddf = pd.DataFrame(drows).drop_duplicates(
        subset=["dataset", "probe", "group_a", "group_b"], keep="last")
    _merged_save(df, CROSS / "crosstask_full_results.csv",
                 ["dataset", "feature_group", "probe"])
    _merged_save(ddf, CROSS / "crosstask_full_delong.csv",
                 ["dataset", "probe", "group_a", "group_b"])
    write_report(df, ddf, extra, time.time() - t0)
    print(f"\n[probe] done: {len(df)} rows, {len(ddf)} delong pairs, "
          f"{(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
