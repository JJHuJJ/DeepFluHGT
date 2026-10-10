#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B1 cross-task RAP  —  train→test (Feng protocol), a protocol upgrade  —  DeepFluHGT

The existing CROSTASK_FULL_RAP.md covers the test-pool 70/30 stratified protocol
(split within the pool); this script adds the Feng 2025 original-protocol column:
**probes are trained on the full train.fasta and tested on the full test set**
(directly comparable with Feng's reported numbers); the two protocols coexist.

Feature groups (CPU-computable half; Evo train-side embeddings await a GPU
window, noted below):
  ntv3_1536  frozen NTv3-pre mean-pool (embed_ntv3_cpu.py --split train artifact
             {ds}_ntv3_pooled_train.npz; convention fully consistent with the test side)
  len+gc / kmer7_rc  form the baselines (computed from train.fasta)
probes: linear (primary) + ridge; protocol: fit on the full train set → evaluate
  on the full test set + bootstrap CI + permutation null (labels permuted on
  **both** sides, train+test, matching the rap/crosstask_full convention  —
  permuting only the train side leaves a 1-D monotone probe's |AUC-0.5| on the
  true test labels unchanged (only the direction flips at random) and degenerates
  the null, as empirically confirmed; ridge = primal closed-form batched, linear
  uses tol=1e-3 under wall-clock constraints) + within-group paired DeLong
  (same test set).

Sample processing (same pooling rules as the existing full version):
  train side: all-N sequences excluded; exact duplicates within the pool are
  **kept** (Feng trains on them as-is; counts are noted)
  test  side: all-N sequences excluded (same as _collect_meta); within-pool
  duplicates kept (same as the existing protocol)
  cross-split duplicates (human_vs_worm ~4 items / promoter_r 18 items) are counted and noted.

output:
  benchmark/crosstask/crosstask_train2test_results.csv
  benchmark/crosstask/crosstask_train2test_delong.csv
  benchmark/crosstask/CROSTASK_FULL_RAP.md  (appends §7, idempotent rewrite)
  docs/evidence/CROSSTASK_TRAIN2TEST.EXIT

Prerequisite: scripts/embed_ntv3_cpu.py --datasets all --split train --threads 48
usage: python scripts/crosstask_train2test.py [--datasets all] [--report-only]
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import rap_probes as rap
from delong_test import delong_test_full
import crosstask_full_probe as cfp      # reuse the null ladder / TASK metadata

CROSS = REPO / "benchmark" / "crosstask"
DATASETS = cfp.DATASETS
GROUPS = ["ntv3_1536", "len+gc", "kmer7_rc"]
PROBES = ["linear", "ridge"]
DELONG_PAIRS = [("ntv3_1536", "len+gc"), ("ntv3_1536", "kmer7_rc"),
                ("len+gc", "kmer7_rc")]
TASK = cfp.TASK
NULL_WALLCLOCK_S = 1200.0
RESULTS_CSV = CROSS / "crosstask_train2test_results.csv"
DELONG_CSV = CROSS / "crosstask_train2test_delong.csv"
RAP_MD = CROSS / "CROSTASK_FULL_RAP.md"
EXIT_MARK = REPO / "docs" / "evidence" / "CROSSTASK_TRAIN2TEST.EXIT"
SECTION_MARK = "## 7. train→test (Feng protocol)  —  protocol upgrade"


def load_side(ds: str, split: str):
    """split='train' → train.fasta + *_ntv3_pooled_train.npz;
    split='test' → rap_input.fasta + *_ntv3_pooled.npz. All-N sequences are excluded before returning."""
    ddir = CROSS / ds
    fasta = ddir / ("train.fasta" if split == "train" else "rap_input.fasta")
    npz = ddir / (f"{ds}_ntv3_pooled_train.npz" if split == "train"
                  else f"{ds}_ntv3_pooled.npz")
    seqs = rap.load_fasta(fasta)
    z = np.load(npz, allow_pickle=False)
    ids = [str(s) for s in z["ids"]]
    assert ids == list(seqs.keys()), f"{ds}/{split}: npz ids do not match fasta order"
    y = z["labels"].astype(int)
    p1536 = z["pooled_1536"].astype(np.float32)
    st = np.array([rap.seq_stats(seqs[i]) for i in ids])
    acgt_ok = ~np.isnan(st[:, 1])
    n_alln = int((~acgt_ok).sum())
    if n_alln:
        seqs = {i: seqs[i] for i, k in zip(ids, acgt_ok) if k}
        ids = [i for i, k in zip(ids, acgt_ok) if k]
        y = y[acgt_ok]; st = st[acgt_ok]; p1536 = p1536[acgt_ok]
    seen, keep = set(), np.ones(len(ids), bool)
    for i, s in enumerate(ids):
        if seqs[s] in seen:
            keep[i] = False
        else:
            seen.add(seqs[s])
    n_dup = int((~keep).sum())
    feats = {
        "ntv3_1536": p1536,
        "len+gc": np.stack([np.log10(st[:, 0]), st[:, 1]], 1).astype(np.float32),
        "kmer7_rc": rap.kmer_feature(seqs, ids),
    }
    meta = dict(n=len(ids), pos=int(y.sum()), n_allN=n_alln, n_dup=n_dup,
                seqs=seqs)
    return ids, y, feats, meta


def run_dataset(ds: str, rows: list, drows: list, extra: dict):
    ids_tr, y_tr, F_tr, m_tr = load_side(ds, "train")
    ids_te, y_te, F_te, m_te = load_side(ds, "test")
    cross_dup = len(set(m_tr["seqs"].values()) & set(m_te["seqs"].values()))
    extra[ds] = dict(n_train=m_tr["n"], n_test=m_te["n"],
                     train_pos=m_tr["pos"], test_pos=m_te["pos"],
                     train_allN=m_tr["n_allN"], test_allN=m_te["n_allN"],
                     train_dup=m_tr["n_dup"], test_dup=m_te["n_dup"],
                     cross_split_dup=cross_dup,
                     task=TASK[ds][0], org=TASK[ds][1], prim=TASK[ds][2],
                     prev_test=float(y_te.mean()))
    print(f"\n[train2test] {ds}: train={m_tr['n']} (pos={m_tr['pos']}) "
          f"test={m_te['n']} (pos={m_te['pos']}) "
          f"cross-split dups={cross_dup}", flush=True)

    n_tr = m_tr["n"]
    preds = {}
    for g in GROUPS:
        Xtr, Xte = F_tr[g], F_te[g]
        mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0) + 1e-12
        xtr = (Xtr - mu) / sd; xte = (Xte - mu) / sd
        for pname in PROBES:
            t0 = time.time()
            if pname == "linear":
                tol = 1e-3 if n_tr > 5000 else 1e-4
                s, nonconv = _linear(xtr, y_tr, xte, tol)
            else:
                s = _ridge(xtr, y_tr, xte); nonconv = 0
            auroc, auprc = rap.metrics(y_te, s)
            preds[(g, pname)] = np.asarray(s, dtype=np.float64)
            n_boot = 1000 if len(y_te) <= 10_000 else 500
            (al, ah), (pl, ph) = rap.bootstrap_ci(y_te, s, n_boot, seed=42)

            # permutation null: labels permuted on **both** sides, train+test (same metric convention as rap/crosstask_full).
            # If only the train side is permuted, a 1-D monotone probe's |AUC-0.5| on
            # the true test labels stays unchanged (only the direction flips at
            # random) and the null degenerates  —  empirically confirmed on promoter_b len+gc.
            rng = np.random.default_rng(42)
            n_perm_target = 200
            null_a, null_p = [], []
            t_null0 = time.time()
            if pname == "ridge":
                perm_tr = np.stack([rng.permutation(y_tr)
                                    for _ in range(n_perm_target)])
                perm_te = np.stack([rng.permutation(y_te)
                                    for _ in range(n_perm_target)])
                sc = cfp.ridge_null_batched(xtr, perm_tr.T, xte)
                from sklearn.metrics import average_precision_score
                null_a = list(rap.auroc_cols(perm_te.T, sc))
                for j in range(n_perm_target):
                    null_p.append(average_precision_score(perm_te[j], sc[:, j]))
                n_perm = n_perm_target
            else:
                n_perm = 0
                for j in range(n_perm_target):
                    if n_perm and n_perm % 10 == 0 and \
                            time.time() - t_null0 > NULL_WALLCLOCK_S:
                        print(f"    [null] {g} wall-clock limit hit: {n_perm}/"
                              f"{n_perm_target}", flush=True)
                        break
                    sj, _ = _linear(xtr, rng.permutation(y_tr), xte, tol=1e-3)
                    a_, p_ = rap.metrics(rng.permutation(y_te), sj)
                    null_a.append(a_); null_p.append(p_)
                    n_perm += 1
            p_au = (1 + int(np.sum(np.asarray(null_a) >= auroc))) / (1 + n_perm)
            p_ap = (1 + int(np.sum(np.asarray(null_p) >= auprc))) / (1 + n_perm)

            rows.append(dict(
                dataset=ds, feature_group=g, probe=pname,
                role={"linear": "Readability", "ridge": "Stability"}[pname],
                n_train=n_tr, n_test=int(len(y_te)),
                prevalence=float(y_te.mean()), primary_metric=TASK[ds][2],
                auroc=auroc, auroc_ci95=f"[{al:.4f},{ah:.4f}]",
                auprc=auprc, auprc_ci95=f"[{pl:.4f},{ph:.4f}]",
                null_auroc_mean=float(np.mean(null_a)),
                null_auprc_mean=float(np.mean(null_p)),
                null_auroc_p=p_au, null_auprc_p=p_ap,
                n_perm=int(n_perm), n_boot=n_boot,
                linear_nonconverged=int(nonconv) if pname == "linear" else 0,
                cross_split_dup=cross_dup, runtime_s=round(time.time() - t0, 1)))
            r_ = rows[-1]
            print(f"  {g:10s} {pname:7s} AUROC={auroc:.4f} AUPRC={auprc:.4f} "
                  f"(nullA p={p_au:.1e}, n_perm={n_perm}) "
                  f"[{r_['runtime_s']}s]", flush=True)

    for pname in PROBES:
        for ga, gb in DELONG_PAIRS:
            d = delong_test_full(y_te, preds[(ga, pname)], preds[(gb, pname)])
            drows.append(dict(dataset=ds, probe=pname, group_a=ga, group_b=gb,
                              auc_a=d["auc_a"], auc_b=d["auc_b"],
                              delta_auc=d["delta"], z=d["z"], p=d["p"],
                              n=d["n"], n_pos=d["n_pos"]))
    return extra[ds]


def _linear(xtr, ytr, xte, tol):
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m = LogisticRegression(C=1.0, max_iter=2000, tol=tol, solver="lbfgs")
        m.fit(xtr, ytr)
    return (m.predict_proba(xte)[:, 1],
            int(np.ravel(getattr(m, "n_iter_", [0]))[0]) >= 2000)


def _ridge(xtr, ytr, xte, alpha: float = 1.0):
    mu = xtr.mean(0, keepdims=True); sd = xtr.std(0) + 1e-12
    a = (xtr - mu) / sd; b = (xte - mu) / sd
    yc = ytr - ytr.mean()
    d = a.shape[1]
    G = a.T @ a + alpha * np.eye(d)
    G[np.diag_indices(d)] += 1e-9 * max(1.0, float(np.trace(G)) / d)
    return b @ np.linalg.solve(G, a.T @ yc) + ytr.mean()


# ══════════════════════ report (§7 idempotent append) ══════════════════════ #

def write_report(df, ddf, extra, total_s):
    import pandas as pd
    # existing 70/30 protocol numbers (split within the test pool)
    cur = {}
    try:
        c = pd.read_csv(CROSS / "crosstask_full_results.csv")
        for _, r in c.iterrows():
            if r.probe == "linear":
                cur[(r.dataset, r.feature_group)] = (
                    float(r.auroc), str(r.auroc_ci95), float(r.auprc))
    except Exception as e:
        print(f"[WARN] failed to read existing full results: {e}", flush=True)

    L = []
    A = L.append
    A(f"{SECTION_MARK}")
    A(f"*(auto-generated {time.strftime('%Y-%m-%d %H:%M:%S')}; re-running idempotently "
      "rewrites this section, §1–6 untouched)*\n")
    A("- protocol: probes trained on the full train.fasta → tested on the full test "
      "set (Feng 2025 original split protocol); coexists as a control with the §1–6 "
      "test-pool 70/30 protocol.")
    A("- feature groups: ntv3_1536 / len+gc / kmer7_rc (CPU-computable half; "
      "**evo_4096 train-side embeddings await a GPU window**; the column will be "
      "added under the same protocol once ready).")
    A("- probes: linear (primary) + ridge; permutation null = labels permuted on both "
      "train+test sides (same metric convention as rap/crosstask_full; permuting only "
      "the train side leaves a 1-D monotone probe's |AUC−0.5| on the true test labels "
      "unchanged and the null degenerate  —  empirically confirmed)  —  "
      f"ridge closed-form batched 200; linear tol=1e-3, wall-clock {NULL_WALLCLOCK_S:.0f}s/cell; "
      "bootstrap CI; DeLong on the same test set.")
    A("- sample: all-N sequences excluded on both sides; within-pool duplicates kept "
      "(train side trained as-is per Feng); cross-split exact-duplicate counts are "
      "noted in each table's footnote.\n")

    A("### 7.1 Overview (linear; AUROC, ecoli_4mc additionally reported as AUPRC)  —  "
      "train→test vs existing test-pool 70/30\n")
    A("| dataset | feature group | train→test AUROC [CI] | test-pool 70/30 AUROC [CI] | "
      "Δ(train→test − 70/30) |")
    A("|---|---|---|---|---|")
    for ds in [d for d in DATASETS if d in extra]:
        for g in GROUPS:
            sel = df[(df.dataset == ds) & (df.feature_group == g) &
                     (df.probe == "linear")]
            if not len(sel):
                continue
            r = sel.iloc[0]
            c0 = cur.get((ds, g))
            d = f"{r.auroc - c0[0]:+.4f}" if c0 else " — "
            A(f"| {ds} | {g} | {r.auroc:.4f} {r.auroc_ci95} | "
              f"{c0[0]:.4f} {c0[1]} | {d} |")
    A("")
    A("ecoli_4mc (primary metric AUPRC, imbalance 16.7%):\n")
    A("| feature group | train→test AUPRC [CI] | test-pool 70/30 AUPRC |")
    A("|---|---|---|")
    for g in GROUPS:
        sel = df[(df.dataset == "ecoli_4mc") & (df.feature_group == g) &
                 (df.probe == "linear")]
        if not len(sel):
            continue
        r = sel.iloc[0]
        c0 = cur.get(("ecoli_4mc", g))
        A(f"| {g} | {r.auprc:.4f} {r.auprc_ci95} | "
          f"{c0[2]:.4f} |" if c0 else f"| {g} | {r.auprc:.4f} |  —  |")
    A("")

    A("### 7.2 Per-dataset details (linear + ridge; null p and CI for all columns)\n")
    for ds in [d for d in DATASETS if d in extra]:
        e = extra[ds]
        A(f"**{ds}**  —  {e['task']} ({e['org']})  ·  train n={e['n_train']:,} "
          f"(pos={e['train_pos']:,}) → test n={e['n_test']:,} "
          f"(pos={e['test_pos']:,}, prev={e['prev_test']:.3f})  ·  "
          f"primary metric {e['prim']}  ·  all-N excluded train/test="
          f"{e['train_allN']}/{e['test_allN']}  ·  within-pool duplicates train/test="
          f"{e['train_dup']}/{e['test_dup']}  ·  **cross-split duplicates="
          f"{e['cross_split_dup']}**\n")
        A("| feature_group | probe | AUROC [CI] | AUPRC [CI] | "
          "null AUROC (p) | n_perm |")
        A("|---|---|---|---|---|---|")
        for _, r in df[df.dataset == ds].iterrows():
            A(f"| {r.feature_group} | {r.probe} | {r.auroc:.4f} "
              f"{r.auroc_ci95} | {r.auprc:.4f} {r.auprc_ci95} | "
              f"{r.null_auroc_mean:.3f} ({r.null_auroc_p:.1e}) | "
              f"{int(r.n_perm)} |")
        A("")

    A("### 7.3 paired DeLong (train→test, same test set)\n")
    A("| dataset | probe | comparison (a vs b) | AUROC_a | AUROC_b | Δ | z | p | "
      "significant? |")
    A("|---|---|---|---|---|---|---|---|---|")
    for _, r in ddf.sort_values("p").iterrows():
        A(f"| {r.dataset} | {r.probe} | {r.group_a} vs {r.group_b} | "
          f"{r.auc_a:.4f} | {r.auc_b:.4f} | {r.delta_auc:+.4f} | "
          f"{r.z:+.2f} | {r.p:.2e} | {'**yes**' if r.p < 0.05 else 'no'} |")
    A("")

    A("### 7.4 Interpretation notes")
    A("- The train→test vs 70/30 difference = the joint effect of training sample "
      "size (75k-85k vs ~70% of the pool) + distribution shift (Feng split); the "
      "mostly negative Δ column suggests the within-pool 70/30 split carries "
      "optimistic bias or a train/test distribution difference.")
    A("- Cross-split exact duplicates (human_vs_worm ≈4 / promoter_r ≈18) are a mild "
      "boundary leakage for train→test, accounting for <0.6%; note this when "
      "interpreting; the 70/30 protocol is unaffected.")
    A("- Runtime environment: probes on CPU (BLAS ≤16); NTv3-pre embeddings on CPU "
      "with 48 threads (`embed_ntv3_cpu.py --split train`).")
    A(f"- Full numeric values: `{RESULTS_CSV.name}` / `{DELONG_CSV.name}`  ·  "
      f"probe total runtime {total_s/60:.1f} min.")
    A("")

    # Idempotent append: strip the old §7 (including its content), then reattach
    old = RAP_MD.read_text()
    cut = old.find(f"\n{SECTION_MARK}")
    base = old if cut < 0 else old[:cut].rstrip() + "\n"
    RAP_MD.write_text(base + "\n" + "\n".join(L))


def main() -> None:
    global RESULTS_CSV, DELONG_CSV, RAP_MD, EXIT_MARK
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="smoke: redirect outputs to .smoke; do not write the official CSV/report/EXIT")
    args = ap.parse_args()
    if args.smoke:
        RESULTS_CSV = RESULTS_CSV.with_suffix(".smoke.csv")
        DELONG_CSV = DELONG_CSV.with_suffix(".smoke.csv")
        RAP_MD = CROSS / "CROSTASK_FULL_RAP.smoke.md"
        if not RAP_MD.is_file():          # copy the report header from the official version (idempotent)
            RAP_MD.write_text(
                CROSS.joinpath("CROSTASK_FULL_RAP.md").read_text())
        EXIT_MARK = EXIT_MARK.with_suffix(".smoke.EXIT")
    names = DATASETS if args.datasets == "all" else args.datasets.split(",")

    import pandas as pd
    t0 = time.time()
    rows, drows, extra = [], [], {}
    if not args.report_only:
        for ds in names:
            # precondition assert: the train embedding npz exists
            npz = CROSS / ds / f"{ds}_ntv3_pooled_train.npz"
            if not npz.is_file():
                raise SystemExit(
                    f"FATAL: {npz} does not exist  —  first run "
                    "scripts/embed_ntv3_cpu.py --datasets "
                    f"{ds} --split train")
            run_dataset(ds, rows, drows, extra)
        pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)
        pd.DataFrame(drows).to_csv(DELONG_CSV, index=False)
    else:
        df = pd.read_csv(RESULTS_CSV)
        ddf = pd.read_csv(DELONG_CSV)
        meta_p = REPO / "docs" / "evidence" / "crosstask_train2test_meta.json"
        extra = json.loads(meta_p.read_text())
        write_report(df, ddf, extra, 0.0)
        return
    total_s = time.time() - t0
    (REPO / "docs" / "evidence" /
     "crosstask_train2test_meta.json").write_text(json.dumps(extra, indent=1))
    write_report(pd.DataFrame(rows), pd.DataFrame(drows), extra, total_s)
    EXIT_MARK.write_text(
        f"completed={time.strftime('%Y-%m-%d %H:%M:%S')} "
        f"total_s={total_s:.0f}\n")
    print(f"[train2test] done {total_s/60:.1f} min -> {RAP_MD}", flush=True)


if __name__ == "__main__":
    main()
