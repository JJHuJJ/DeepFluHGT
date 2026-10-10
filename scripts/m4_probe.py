#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M4 final version — length+GC logistic probe (final row of the Attribution table) — DeepFluHGT.

contract (V7_CONTRACT.md "Representation Attribution Table" prespecifiedrow):
  | M4 | length + GC (logistic) | to fill (Step 0 already has: AUROC 0.673) | trivial covariate baseline |

this script = the official numbers for that row:
  1. window-level: the windowing protocol completely identical to M0's (`SlidingWindowContigDataset`,
     window=3000 / overlap=300 / max_eval_windows=10) rebuild
     - train 940,028 windows (is_training=True policy; random.seed(42) reproduces the training-log protocol)
     - test 199,707 window (is_training=False alloverride; assert = M0 traininglog test count)
     per window CPU-compute len / GC (BED coordinates + fasta), logistic regression (2 features)
     → test window-level AUPRC / AUROC + bootstrap 95% CI (n=1000, seed42).
  2. contig level (with M0 on-paper 0.827 same metric caliber):
     - window probabilities pass through `TrainingEngine._aggregate_contig_predictions` (≤3 windows max /
       4-10window top3mean / >10window max(top3mean,P90)) aggregation → 50,000 contig levelmetric;
     - directly contig level logistic (log10 contig_len + contig GC → contig label) control.
  3. DeLong: M0 test window-levelpredict**not yetwritten to disk** (checkpoints/ with docs/evidence/ allnone
     test_probs archived; only val has trainval_gap_epoch_*.npz) → per the fallback plan, compared against the M0 test
     on-paper value + DeLong omitted; additionally a **val window-level paired DeLong** (M0 ep2=best_model
     archive of 199,438 window predictions vs M4 on the same batch of windows, bitwise-alignment assert).

output: docs/evidence/M4_PROBE.md + docs/evidence/M4_PROBE.json + M4_PROBE.EXIT
runtime discipline: CPU-only (CUDA_VISIBLE_DEVICES=""), OMP/BLAS ≤6 threads, never touches production GPUs.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "") # CPU-only
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "6") # limit 6 thread

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "simulated_data" / "Contig_500k"
EVID = REPO / "docs" / "evidence"

# -- M0 caliberconstant (configs/contig_500k_ntv3only_4090.yaml + logs/M0_contig_ntv3only.log L19) --
WIN, OVERLAP, MAX_EVAL = 3000, 300, 10
EXPECTED = {"train": 940_028, "test": 199_707, "val": 199_438}
SEED = 42
N_BOOT = 1000
# M0 test on-papervalue (logs/M0_contig_ntv3only.log "Final Test Evaluation", best=ep2)
M0_TEST = {"contig_auprc": 0.8269, "contig_auroc": 0.8272,
           "win_hard_neg_auc": 0.9030, "win_easy_neg_auc": 0.8846}
M0_VAL_NPZ = EVID / "m0_trajectory" / "trainval_gap_epoch_2.npz" # ep2 = best_model


# ========================== data load (reuses the production dataset class) ========================== #

def build_split(split: str, is_training: bool):
    """BEDParser + SlidingWindowContigDataset — the same windowing protocol as M0 training/evaluation.

    train: is_training=True (training policy); reset random/np.random seed=42 **before** building
           (replicating the scripts/trainval_gap.py protocol, which empirically reproduces the 940,028 windows).
    test/val: is_training=False alloverride (deterministic, none RNG).
    """
    from data_pipeline.impl.parser import BEDParser
    from data_pipeline.impl.dataset import SlidingWindowContigDataset

    if is_training:
        random.seed(SEED)
        np.random.seed(SEED)
    samples = BEDParser(
        bed_path=str(DATA_DIR / split / "breakpoints.bed"),
        fasta_dir=str(DATA_DIR / split), mode="contig").parse()
    ds = SlidingWindowContigDataset(
        samples=samples, window_size=WIN, overlap=OVERLAP,
        is_training=is_training,
        max_neg_windows=2, max_eval_windows=MAX_EVAL,
        extract_bio=False, evo_cache_path=None, rc_aug_prob=0.0)
    exp = EXPECTED[split]
    if len(ds) != exp:
        print(f"[WARN] {split} window count {len(ds)} != M0 log protocol {exp} (protocol drift, noted in the report)",
              flush=True)
    else:
        print(f"[data] {split} windows = {len(ds)} ✓ (M0 logcaliber {exp:,})", flush=True)
    return ds


def window_features(ds) -> dict:
    """compute per-window len/GC on CPU from _window_index (fasta slicing, no tokenization)."""
    n = len(ds._window_index)
    ids = np.empty(n, dtype=object)
    y = np.empty(n, dtype=np.int8)
    types = np.empty(n, dtype=object)
    log_len = np.empty(n, dtype=np.float64)
    gc = np.empty(n, dtype=np.float64)
    seqs = ds.samples
    t0 = time.time()
    for i, (si, ws, we, lab, st) in enumerate(ds._window_index):
        seq = seqs[si].sequence[ws:we]
        a = seq.count("A"); c = seq.count("C"); g = seq.count("G"); t = seq.count("T")
        denom = max(1, a + c + g + t)
        ids[i] = seqs[si].seq_id
        y[i] = lab
        types[i] = st
        log_len[i] = np.log10(max(1, we - ws))
        gc[i] = (g + c) / denom
    print(f"[feat] {n:,} window len/GC compute {time.time()-t0:.0f}s "
          f"(pos={int(y.sum())}, neg={int((1-y).sum())})", flush=True)
    return {"ids": ids, "y": y, "types": types, "log_len": log_len, "gc": gc}


def contig_features_from_ds(ds) -> dict:
    """per-contig (whole contig) len/GC — used by the direct contig-level logistic."""
    m = len(ds.samples)
    ids = np.empty(m, dtype=object)
    y = np.empty(m, dtype=np.int8)
    log_len = np.empty(m, dtype=np.float64)
    gc = np.empty(m, dtype=np.float64)
    for i, s in enumerate(ds.samples):
        seq = s.sequence
        a = seq.count("A"); c = seq.count("C"); g = seq.count("G"); t = seq.count("T")
        ids[i] = s.seq_id
        y[i] = s.label
        log_len[i] = np.log10(max(1, len(seq)))
        gc[i] = (g + c) / max(1, a + c + g + t)
    return {"ids": ids, "y": y, "log_len": log_len, "gc": gc}


# ========================== statisticstool ========================== #

def bootstrap_ci(y, s, metric, n_boot=N_BOOT, seed=SEED):
    """percentile bootstrap 95% CI (replicating the scripts/bootstrap_ci.py protocol)."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64)
    N = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, N, N)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(metric(y[idx], s[idx]))
    v = np.asarray(vals)
    return {"n_boot": int(len(v)),
            "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
            "boot_mean": float(v.mean())}


def fit_logistic(xtr, ytr):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(xtr)
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    clf.fit(sc.transform(xtr), ytr)
    return sc, clf


def pred_logistic(sc, clf, x):
    return clf.predict_proba(sc.transform(x))[:, 1]


# ============================== main pipeline ============================== #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    t0 = time.time()
    from sklearn.metrics import average_precision_score, roc_auc_score

    out: dict = {"meta": {
        "probe": "logistic(length, GC) — trivial covariate baseline (Attribution table M4 row)",
        "features_window": "log10(window_len), window_GC (counted from fasta, CPU)",
        "features_contig": "log10(contig_len), contig_GC",
        "windowing": f"SlidingWindowContigDataset win={WIN} overlap={OVERLAP} "
                     f"max_eval={MAX_EVAL}; train=is_training(True,seed42) "
                     f"test=is_training(False)",
        "estimator": "sklearn LogisticRegression(C=1, lbfgs, standardized)",
        "n_boot": args.n_boot, "seed": SEED, "cpu_only": True,
    }}

    # -- 1. train window features (training policy, reproduced) --
    tr_ds = build_split("train", is_training=True)
    tr = window_features(tr_ds)
    tr_contig = contig_features_from_ds(tr_ds)
    del tr_ds
    gc.collect()

    # -- 2. test windowfeature --
    te_ds = build_split("test", is_training=False)
    te = window_features(te_ds)
    te_contig = contig_features_from_ds(te_ds)
    te_types = te["types"]
    del te_ds
    gc.collect()

    # -- 3. window-level logistic: fit on train → test metrics --
    Xtr = np.stack([tr["log_len"], tr["gc"]], 1)
    Xte = np.stack([te["log_len"], te["gc"]], 1)
    sc, clf = fit_logistic(Xtr, tr["y"])
    p_te = pred_logistic(sc, clf, Xte)
    y_te = te["y"].astype(int)

    w = {"n": int(len(y_te)), "pos": int(y_te.sum()),
         "auprc": float(average_precision_score(y_te, p_te)),
         "auroc": float(roc_auc_score(y_te, p_te))}
    w["auprc_ci"] = bootstrap_ci(y_te, p_te, average_precision_score, args.n_boot)
    w["auroc_ci"] = bootstrap_ci(y_te, p_te, roc_auc_score, args.n_boot)
    # single-feature references (interpretability)
    w["auroc_gc_only"] = float(roc_auc_score(y_te, te["gc"]))
    w["auroc_loglen_only"] = float(roc_auc_score(y_te, te["log_len"]))
    w["coef_loglen"], w["coef_gc"] = (float(clf.coef_[0][0]), float(clf.coef_[0][1]))
    w["window_len_note"] = (
        "window length is nearly degenerate: windows of contigs ≥3000bp are always 3000; only single windows of short contigs are <3000 "
        "(the feature is still retained by protocol; length's discriminative information lives mainly at the contig level, see contig-direct)")
    out["window_level_test"] = w
    print(f"[M4] test window-level: AUPRC={w['auprc']:.4f} {w['auprc_ci']['ci95']} "
          f"AUROC={w['auroc']:.4f} {w['auroc_ci']['ci95']} (n={w['n']:,})", flush=True)

    # -- 4. contig level: (a) windowprobabilityaggregation (same M0 caliber) --
    from training_engine.api import TrainingEngine
    cp, cy, cst = TrainingEngine._aggregate_contig_predictions(
        p_te, y_te, list(te["ids"]), list(te_types))
    agg = {"n": int(len(cy)), "pos": int(np.sum(cy)),
           "auprc": float(average_precision_score(cy, cp)),
           "auroc": float(roc_auc_score(cy, cp)),
           "aggregation": "≤3windowmax / 4-10windowtop3mean / >10windowmax(top3mean,P90) "
                          "(TrainingEngine._aggregate_contig_predictions)"}
    agg["auprc_ci"] = bootstrap_ci(cy, cp, average_precision_score, args.n_boot)
    agg["auroc_ci"] = bootstrap_ci(cy, cp, roc_auc_score, args.n_boot)
    out["contig_agg_test"] = agg
    print(f"[M4] test contig aggregation: AUPRC={agg['auprc']:.4f} {agg['auprc_ci']['ci95']} "
          f"AUROC={agg['auroc']:.4f} {agg['auroc_ci']['ci95']} (n={agg['n']:,})",
          flush=True)

    # -- 4b. contig level: (b) directly contig logistic --
    ctr = np.stack([tr_contig["log_len"], tr_contig["gc"]], 1)
    cte = np.stack([te_contig["log_len"], te_contig["gc"]], 1)
    csc, cclf = fit_logistic(ctr, tr_contig["y"])
    p_ct = pred_logistic(csc, cclf, cte)
    y_ct = te_contig["y"].astype(int)
    cdir = {"n": int(len(y_ct)), "pos": int(y_ct.sum()),
            "auprc": float(average_precision_score(y_ct, p_ct)),
            "auroc": float(roc_auc_score(y_ct, p_ct)),
            "coef_loglen": float(cclf.coef_[0][0]), "coef_gc": float(cclf.coef_[0][1])}
    cdir["auprc_ci"] = bootstrap_ci(y_ct, p_ct, average_precision_score, args.n_boot)
    cdir["auroc_ci"] = bootstrap_ci(y_ct, p_ct, roc_auc_score, args.n_boot)
    out["contig_direct_test"] = cdir
    print(f"[M4] test contig directly: AUPRC={cdir['auprc']:.4f} {cdir['auprc_ci']['ci95']} "
          f"AUROC={cdir['auroc']:.4f} {cdir['auroc_ci']['ci95']} (n={cdir['n']:,})",
          flush=True)
    del tr, tr_contig
    gc.collect()

    # -- 5. search for an M0 test prediction archive → DeLong feasibility --
    cand = [
        REPO / "checkpoints" / "Contig_500k_ntv3only" / "test_probs.npz",
        REPO / "eval_output" / "Contig_500k_ntv3only" / "test_probs.npz",
        EVID / "m0_trajectory" / "test_best_preds.npz",
        EVID / "m0_trajectory" / "test_preds.npz",
    ]
    found = [str(p) for p in cand if p.exists()]
    glob_hits = [str(p) for p in EVID.rglob("*test*pred*.npz")]
    out["m0_test_archive"] = {
        "searched": [str(p) for p in cand] + ["docs/evidence/**/*test*pred*.npz"],
        "found": found + glob_hits,
        "conclusion": "M0 test window-level/contig-level predictions were never archived → per the fallback plan: compare with the M0 test "
                      "on-paper value; DeLong (vs M0 test) omitted; a val paired DeLong is added instead (§6)",
    }
    out["m0_test_paper"] = M0_TEST

    # -- 6. val paired DeLong: M0 ep2 (best_model) archived vs M4 samebatchwindow --
    dlong = {"status": "NOT_RUN"}
    if M0_VAL_NPZ.exists():
        va_ds = build_split("val", is_training=False)
        va = window_features(va_ds)
        del va_ds
        gc.collect()
        m0 = np.load(M0_VAL_NPZ, allow_pickle=True)
        same_ids = bool((m0["val_ids"].astype(str) == va["ids"].astype(str)).all())
        same_lab = bool((m0["val_labels"].astype(int) == va["y"].astype(int)).all())
        Xva = np.stack([va["log_len"], va["gc"]], 1)
        p_va = pred_logistic(sc, clf, Xva)
        y_va = va["y"].astype(int)
        m0p = m0["val_probs"].astype(np.float64)
        dlong = {
            "status": "OK",
            "archive": str(M0_VAL_NPZ),
            "ckpt": "epoch_2 = best_model.pt (M0 finalized)",
            "n_windows": int(len(y_va)),
            "ids_aligned": same_ids, "labels_aligned": same_lab,
            "m0_ep2_val": {"auroc": float(roc_auc_score(y_va, m0p)),
                           "auprc": float(average_precision_score(y_va, m0p))},
            "m4_val": {"auroc": float(roc_auc_score(y_va, p_va)),
                       "auprc": float(average_precision_score(y_va, p_va))},
        }
        if same_ids and same_lab:
            sys.path.insert(0, str(REPO / "scripts"))
            from delong_test import delong_test
            auc_a, auc_b, z, p = delong_test(y_va, m0p, p_va)
            dlong["delong_m0_vs_m4"] = {
                "auc_m0_ep2": float(auc_a), "auc_m4": float(auc_b),
                "z": float(z), "p_value": float(p),
                "note": "paired DeLong on identical 199,438 val windows "
                        "(paired stand-in for the missing test archive; val is not test, label caveat)"}
        else:
            dlong["delong_m0_vs_m4"] = {"status": "SKIPPED",
                                        "reason": "ids/labels not yetbitwisealignment"}
        print(f"[M4] val DeLong M0(ep2) vs M4: "
              f"AUROC {dlong['m0_ep2_val']['auroc']:.4f} vs {dlong['m4_val']['auroc']:.4f}"
              + (f" z={z:+.2f} p={p:.3e}" if same_ids and same_lab else ""),
              flush=True)
    else:
        dlong = {"status": "NOT_RUN", "reason": f"val archivedmissing: {M0_VAL_NPZ}"}
    out["delong"] = dlong

    # -- 7. written to disk --
    out["meta"]["runtime_s"] = round(time.time() - t0, 1)
    EVID.mkdir(parents=True, exist_ok=True)
    (EVID / "M4_PROBE.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    _write_md(EVID / "M4_PROBE.md", out)
    (EVID / "M4_PROBE.EXIT").write_text(
        f"done {time.strftime('%Y-%m-%d %H:%M:%S')} runtime={out['meta']['runtime_s']}s\n")
    print(f"[M4] done -> {EVID/'M4_PROBE.md'} ({out['meta']['runtime_s']}s)", flush=True)


def _write_md(path: Path, d: dict) -> None:
    L = []
    A = L.append
    A("# M4 final version — length+GC logistic probe (final row of the Attribution table)\n")
    A(f"> generate: {time.strftime('%Y-%m-%d %H:%M:%S')} · DeepFluHGT · CPU only "
      f"(6 thread) · seed={d['meta']['seed']} · bootstrap n={d['meta']['n_boot']}")
    A(f"> protocol: {d['meta']['windowing']}")
    A(f"> estimator: {d['meta']['estimator']}\n")

    A("## 1. test window-level (199,707 windows, rebuilt under the same window protocol as M0)\n")
    w = d["window_level_test"]
    A("| metric | point estimate | bootstrap 95% CI |")
    A("|---|---|---|")
    A(f"| AUPRC | **{w['auprc']:.4f}** | [{w['auprc_ci']['ci95'][0]:.4f}, "
      f"{w['auprc_ci']['ci95'][1]:.4f}] |")
    A(f"| AUROC | **{w['auroc']:.4f}** | [{w['auroc_ci']['ci95'][0]:.4f}, "
      f"{w['auroc_ci']['ci95'][1]:.4f}] |")
    A(f"\n- n={w['n']:,} (pos={w['pos']:,}, positive rate {w['pos']/w['n']:.3f} = the AUPRC random baseline); "
      f"singlefeature: AUROC(GC)={w['auroc_gc_only']:.4f}, "
      f"AUROC(log10 len)={w['auroc_loglen_only']:.4f}; standardized coefficients: "
      f"loglen={w['coef_loglen']:+.3f}, GC={w['coef_gc']:+.3f}")
    A(f"- note: {w['window_len_note']}")
    A(f"- interpretation: len+GC has **no discriminative power** at the window level (AUROC≈0.5, "
      f"AUPRC below the positive-rate baseline; the train-fit coefficient direction flips slightly out-of-sample on test), the trivial covariates' "
      f"discriminative information lives mainly in contig-level length (see §2 direct contig protocol 0.61)."
      f"The old Step 0 value 0.673 was a 1,996 long-sequence subset protocol (length variation enriched), not directly comparable.\n")

    A("## 2. test contig level (with M0 on-paper 0.827 same metric caliber)\n")
    A("| caliber | n | AUPRC [95% CI] | AUROC [95% CI] |")
    A("|---|---|---|---|")
    agg, cdir = d["contig_agg_test"], d["contig_direct_test"]
    A(f"| M4 windowprobabilityaggregation (same M0 aggregation) | {agg['n']:,} | "
      f"**{agg['auprc']:.4f}** [{agg['auprc_ci']['ci95'][0]:.4f}, {agg['auprc_ci']['ci95'][1]:.4f}] | "
      f"**{agg['auroc']:.4f}** [{agg['auroc_ci']['ci95'][0]:.4f}, {agg['auroc_ci']['ci95'][1]:.4f}] |")
    A(f"| M4 directly contig logistic (log10 len + GC) | {cdir['n']:,} | "
      f"**{cdir['auprc']:.4f}** [{cdir['auprc_ci']['ci95'][0]:.4f}, {cdir['auprc_ci']['ci95'][1]:.4f}] | "
      f"**{cdir['auroc']:.4f}** [{cdir['auroc_ci']['ci95'][0]:.4f}, {cdir['auroc_ci']['ci95'][1]:.4f}] |")
    m0 = d["m0_test_paper"]
    A(f"| **M0 (NTv3-only, on-paper)** | 50,000 | **{m0['contig_auprc']:.4f}** | "
      f"**{m0['contig_auroc']:.4f}** |")
    A("")
    A(f"- M0 test prediction archive search: {d['m0_test_archive']['found'] or 'none'} → "
      f"{d['m0_test_archive']['conclusion']}")
    A(f"- differencevalue (M0 − M4, contig aggregationcaliber): ΔAUPRC = "
      f"{m0['contig_auprc']-agg['auprc']:+.4f}, ΔAUROC = "
      f"{m0['contig_auroc']-agg['auroc']:+.4f} — the share explainable by trivial covariates; "
      f"the remainder is the contribution interval of representation/contextual information.\n")

    dl = d["delong"]
    A("## 3. DeLong test\n")
    if dl.get("status") == "OK" and "delong_m0_vs_m4" in dl and \
            dl["delong_m0_vs_m4"].get("status") != "SKIPPED":
        dd = dl["delong_m0_vs_m4"]
        A(f"- **vs M0 test: omitted** (window-level predictions never archived, see the §2 search record).")
        pstr = ("<1e-300" if dd["p_value"] == 0.0 else f"{dd['p_value']:.3e}")
        A(f"- **val paired DeLong** ({dl['ckpt']}, n={dl['n_windows']:,}, bitwisealignment "
          f"ids={dl['ids_aligned']}/labels={dl['labels_aligned']}): "
          f"AUROC M0={dd['auc_m0_ep2']:.4f} vs M4={dd['auc_m4']:.4f}, "
          f"z={dd['z']:+.2f}, p={pstr} — — "
          f"{'M0 significantly above M4 (expected: representation information >> trivial covariates)' if dd['z'] > 0 else 'note: direction reversed'}")
        A(f"- val pointvalue: M0 AUROC={dl['m0_ep2_val']['auroc']:.4f}/"
          f"AUPRC={dl['m0_ep2_val']['auprc']:.4f}; "
          f"M4 AUROC={dl['m4_val']['auroc']:.4f}/AUPRC={dl['m4_val']['auprc']:.4f}")
        A(f"- {dd['note']}")
    else:
        A(f"- status: {dl.get('status')} ({dl.get('reason', '')})")
    A("")

    A("## 4. Attribution table M4 row, filled\n")
    A("| Model | Representation | AUPRC (test) | note |")
    A("|---|---|---|---|")
    A(f"| M4 | length + GC (logistic) | **{cdir['auprc']:.4f}** "
      f"(contig directly; aggregationcaliber {agg['auprc']:.4f}) | trivial covariate baseline; "
      f"window-level {w['auprc']:.4f}; the old Step 0 AUROC 0.673 was a subset protocol |")
    A("")
    A(f"- total runtime {d['meta']['runtime_s']}s; countvaluesee `M4_PROBE.json`.\n")
    Path(path).write_text("\n".join(L))


if __name__ == "__main__":
    main()
