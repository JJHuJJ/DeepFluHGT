#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shortcut / leakage audit  —  DeepFluHGT Stage B-0 [P0-2].

Test whether the frozen Evo2 7B embedding is merely a re-encoding of
composition/taxonomy shortcuts:

  ① embedding → GC        ridge regression (R² / MAE / Spearman + permuted-target null)
  ② embedding → log10(len) same as above
  ③ taxonomy proxy: embedding → host-genome source_file group (≥5 samples per group)
       linear probe macro-F1 + label-permutation null
  ④ prediction residual correlation  —  —  **placeholder**: requires the
     per-sample prediction file produced after M0 completes; not run for now
     since the input does not exist (implementation + docs provided only).

output: evo_cache/step0_subset/rap/shortcut_audit.json + shortcut_audit.md
CPU-only; read-only on production data, does not modify any production code.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU-only

import argparse
import csv
import json
import time
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = REPO / "evo_cache" / "step0_subset"
DEFAULT_OUT = DEFAULT_CACHE / "rap"

ALPHA = 1.0          # ridge regularization (on normalized features)
MIN_GROUP = 5        # minimum samples per group for the taxonomy proxy


# ─────────────────────────────── loading ─────────────────────────────── #

def load_fasta(path: Path) -> dict[str, str]:
    seqs, name, chunks = {}, None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name:
                    seqs[name] = "".join(chunks)
                name = line[1:].strip().split()[0]
                chunks = []
            else:
                chunks.append(line.strip().upper())
    if name:
        seqs[name] = "".join(chunks)
    return seqs


def load_provenance(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def taxonomy_proxy(prov: list[dict]) -> dict[str, "str | None"]:
    """Map each sample → the GCF accession of its host-genome source_file
    (taxonomy proxy).

    Only segments with seg_type=HOST ∧ source_type=microbe are accepted; for
    multi-host samples the first HOST record is used (a documented choice).
    Mixed/chimeric segments (HOST_AMALGAMATED, HOST_VIRUS_HOMOLOG, CONTAMINANT,
    VIRUS_GC_MATCHED, ...) have no unique taxonomy → None (excluded from the probe).
    """
    out: dict[str, "str | None"] = {}
    for r in prov:
        sid = r["seq_id"]
        if sid in out and out[sid] is not None:
            continue
        if r.get("seg_type") == "HOST" and r.get("source_type") == "microbe":
            base = os.path.basename(r.get("source_file", "") or "")
            out[sid] = base.replace("_genomic.fna", "") if base else None
        else:
            out.setdefault(sid, None)
    return out


# ──────────────────── ridge regression (closed-form, batchable permutations) ──────────────────── #

def _prep_dual(xtr):
    mu = xtr.mean(0, keepdims=True)
    sd = xtr.std(0) + 1e-12
    a = (xtr - mu) / sd
    k = a @ a.T + ALPHA * np.eye(len(a))
    chol = np.linalg.cholesky(k)
    return a, chol


def ridge_reg_fit(xtr, ytr, xte):
    """Return test predictions from the ridge fit; y is continuous (GC / log len)."""
    a, chol = _prep_dual(xtr)
    yc = ytr - ytr.mean()
    coef = np.linalg.solve(chol.T, np.linalg.solve(chol, yc))
    return (xte - xtr.mean(0, keepdims=True)) / (xtr.std(0) + 1e-12) @ a.T @ coef + ytr.mean()


def ridge_reg_null(xtr, ytr_perm: np.ndarray, xte) -> np.ndarray:
    """Test-prediction matrix [n_te, m] for the permuted-target null. One Cholesky, batched."""
    a, chol = _prep_dual(xtr)
    b = (xte - xtr.mean(0, keepdims=True)) / (xtr.std(0) + 1e-12)
    yc = ytr_perm - ytr_perm.mean(0, keepdims=True)          # [n_tr, m]
    sol = np.linalg.solve(chol.T, np.linalg.solve(chol, yc))
    return b @ a.T @ sol + ytr_perm.mean(0, keepdims=True)


def reg_metrics(y, p) -> dict:
    from scipy.stats import spearmanr
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "mae": float(np.mean(np.abs(y - p))),
        "spearman": float(spearmanr(y, p).statistic),
    }


# ─────────────────────── taxonomy linear probe ─────────────────────── #

def macro_f1(y_true, y_pred, labels) -> float:
    f1s = []
    for c in labels:
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(f1s))


def ridge_clf_fit(xtr, ytr, xte):
    """ridge → one-hot argmax classifier (identical estimator for the null)."""
    classes = np.unique(ytr)
    y_oh = np.zeros((len(ytr), len(classes)))
    for i, c in enumerate(classes):
        y_oh[ytr == c, i] = 1.0
    a, chol = _prep_dual(xtr)
    sol = np.linalg.solve(chol.T, np.linalg.solve(chol, y_oh))
    b = (xte - xtr.mean(0, keepdims=True)) / (xtr.std(0) + 1e-12)
    scores = b @ a.T @ sol + y_oh.mean(0)
    return classes[np.argmax(scores, 1)]


def ridge_clf_null(xtr, ytr_perm: np.ndarray, xte):
    """Return argmax predictions per permutation [n_te, m] (classes taken from each permuted train's unique values)."""
    a, chol = _prep_dual(xtr)
    b = (xte - xtr.mean(0, keepdims=True)) / (xtr.std(0) + 1e-12)
    m = ytr_perm.shape[1]
    preds = np.empty((len(xte), m), dtype=object)
    for j in range(m):
        classes = np.unique(ytr_perm[:, j])
        y_oh = np.zeros((len(xtr), len(classes)))
        for i, c in enumerate(classes):
            y_oh[ytr_perm[:, j] == c, i] = 1.0
        sol = np.linalg.solve(chol.T, np.linalg.solve(chol, y_oh))
        scores = b @ a.T @ sol + y_oh.mean(0)
        preds[:, j] = classes[np.argmax(scores, 1)]
    return preds


# ──────────── ④ placeholder: prediction residual correlation (not run) ──────────── #

def audit_prediction_residual_correlation(
    predictions_csv: Path | None = None,
    cache_dir: Path = DEFAULT_CACHE,
) -> dict:
    """corr(residual, GC / length / taxonomy-group)  —  —  placeholder implementation.

    Answers "is the classifier *exploiting* a composition shortcut" (embedding R²
    only shows that the features contain that information).
    Criterion: |Spearman(residual, covariate)| ≈ 0.

    Required input (exported after M0 completes):
      predictions_csv: at least the columns seq_id, prob (sigmoid probability), label
         —  —  and covering the step0 subset's 1996 seq_ids (the M0 training set is
           a superset, so they can be matched directly; if only contig-aggregated
           probabilities are available, use the aggregated value).
    compute:
      residual = label - prob (or |label - prob|)
      covariates = GC, log10(len), taxonomy group one-hot groupmean
      report Spearman + Pearson + between-group ANOVA η² after quantile binning.
    Current status: NOT_RUN (no input file).
    """
    if predictions_csv is None or not Path(predictions_csv).exists():
        return {
            "status": "NOT_RUN",
            "reason": "requires the per-sample prediction file produced after M0 completes (seq_id, prob, label); it does not exist yet",
            "required_input": "csv with columns [seq_id, prob, label] covering step0 ids",
            "planned_statistics": [
                "spearman(residual, GC)", "spearman(residual, log10_len)",
                "eta_squared(residual, taxonomy_group)",
            ],
        }
    raise NotImplementedError("to be implemented once the input is ready (Stage B-1)")


# ──────────────────────────────── main pipeline ──────────────────────────────── #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-perm", type=int, default=1000)
    args = ap.parse_args()
    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    emb = {k: v.astype(np.float32) for k, v in torch.load(
        args.cache_dir / "evo_subset.pt", map_location="cpu",
        weights_only=False).items()}
    seqs = load_fasta(args.cache_dir / "step0_subset.fasta")
    ids = list(seqs.keys())
    y = np.array([1 if i.startswith("pos_") else 0 for i in ids], dtype=int)

    gc = np.array([(seqs[i].count("G") + seqs[i].count("C")) /
                   max(1, sum(seqs[i].count(c) for c in "ACGT")) for i in ids])
    ln = np.log10(np.array([len(seqs[i]) for i in ids], dtype=np.float64))

    tax = taxonomy_proxy(load_provenance(args.cache_dir / "provenance.tsv"))
    # label integrity note (prefix vs provenance label)
    prov_lab: dict[str, str] = {}
    for r in load_provenance(args.cache_dir / "provenance.tsv"):
        prov_lab.setdefault(r["seq_id"], r["label"])
    mism = sum(int(prov_lab.get(i, "x")) != int(i.startswith("pos_")) for i in ids)

    from sklearn.model_selection import train_test_split
    idx = np.arange(len(ids))
    tr, te = train_test_split(idx, test_size=0.30, stratify=y, random_state=42)

    X = np.stack([emb[i] for i in ids])
    mu, sd = X[tr].mean(0, keepdims=True), X[tr].std(0) + 1e-12
    Xs = (X - mu) / sd
    xtr, xte = Xs[tr], Xs[te]

    rng = np.random.default_rng(42)
    out: dict = {"meta": {
        "n_samples": len(ids), "n_pos": int(y.sum()),
        "split": "70/30 stratified seed42", "ridge_alpha": ALPHA,
        "n_perm": args.n_perm, "embedding": "Evo2 7B contig-mean (4096d)",
        "label_source": "seq_id prefix", "runtime_note": "CPU only",
    }, "label_integrity": {
        "prefix_vs_provenance_label_mismatches": int(mism),
        "note": "prefix is 100% consistent with the M0 BED labels; the provenance.tsv label column contains flips (see fingerprint)",
    }}

    # ── ①② GC / len regression + permuted null ──
    for name, tgt in (("gc_regression", gc), ("loglen_regression", ln)):
        p = ridge_reg_fit(xtr, tgt[tr], xte)
        obs = reg_metrics(tgt[te], p)
        perm_t = np.stack([rng.permutation(tgt) for _ in range(args.n_perm)])
        preds = ridge_reg_null(xtr, perm_t[:, tr].T, xte)     # [n_te, m]
        null_r2 = 1.0 - ((tgt[te][:, None] - preds) ** 2).sum(0) / \
            ((tgt[te] - tgt[te].mean()) ** 2).sum()
        null_r2 = np.asarray(null_r2)
        out[name] = {
            "target": "GC_fraction" if name.startswith("gc") else "log10(length)",
            **obs,
            "null_r2_mean": float(null_r2.mean()),
            "null_r2_sd": float(null_r2.std()),
            "null_r2_p95": float(np.percentile(null_r2, 95)),
            "perm_p_value_r2": float((1 + int((null_r2 >= obs["r2"]).sum())) /
                                     (1 + len(null_r2))),
        }
        r = out[name]
        print(f"[shortcut] {r['target']:16s} R2={r['r2']:.3f} "
              f"(null {r['null_r2_mean']:.3f}±{r['null_r2_sd']:.3f}, "
              f"p={r['perm_p_value_r2']:.4f}) rho={r['spearman']:.3f}")

    # ── ③ taxonomy-proxy linear probe ──
    def run_tax_probe(min_group: int) -> dict:
        groups = np.array([tax[i] if tax[i] is not None else "" for i in ids])
        counts = {g: int((groups == g).sum()) for g in np.unique(groups) if g}
        keep_mask = np.array([bool(g) and counts[g] >= min_group for g in groups])
        keep_idx = idx[keep_mask]
        _vals, codes_kept = np.unique(groups[keep_mask], return_inverse=True)
        trc = np.isin(keep_idx, tr)
        tec = ~trc
        xtr_c, xte_c = Xs[keep_idx][trc], Xs[keep_idx][tec]
        yc_all = codes_kept
        ytr_c, yte_c = yc_all[trc], yc_all[tec]
        pred = ridge_clf_fit(xtr_c, ytr_c, xte_c)
        labels_eval = np.unique(np.concatenate([yte_c, pred]))
        mf1 = macro_f1(yte_c, pred, labels_eval)
        majority = np.bincount(ytr_c).argmax()
        mf1_base = macro_f1(yte_c, np.full_like(yte_c, majority), labels_eval)
        perm_codes = np.stack([rng.permutation(yc_all) for _ in range(args.n_perm)])
        null_pred = ridge_clf_null(xtr_c, perm_codes[:, trc].T, xte_c)
        null_mf1 = np.array([
            macro_f1(perm_codes[j][tec], null_pred[:, j], labels_eval)
            for j in range(args.n_perm)])
        return {
            "min_group_size": min_group,
            "n_groups": int(len(_vals)),
            "n_train": int(trc.sum()), "n_test": int(tec.sum()),
            "n_samples_kept": int(len(keep_idx)),
            "macro_f1": mf1,
            "majority_baseline_macro_f1": float(mf1_base),
            "null_macro_f1_mean": float(null_mf1.mean()),
            "null_macro_f1_sd": float(null_mf1.std()),
            "null_macro_f1_p95": float(np.percentile(null_mf1, 95)),
            "perm_p_value": float((1 + int((null_mf1 >= mf1).sum())) /
                                  (1 + len(null_mf1))),
        }

    tax_all = [tax[i] for i in ids]
    n_no_host = int(sum(t is None for t in tax_all))
    out["taxonomy_probe"] = {
        "proxy": "host microbe source_file (GCF accession, first HOST record)",
        "n_dropped_no_unambiguous_host": n_no_host,
        "granularity_note": (
            f"In the step0 subset, host genomes are roughly one genome per sample "
            f"(unique GCFs {len(set(t for t in tax_all if t))} / "
            f"{len(ids) - n_no_host} samples with a host); the ≥5-group cutoff covers only 30 samples → "
            "the primary test is underpowered, so ≥3 / ≥2 sensitivity tiers are attached; "
            "conclusions should be based on a re-run with the full cache"),
        "primary_min_group": MIN_GROUP,
        "runs": [run_tax_probe(m) for m in (MIN_GROUP, 3, 2)],
    }
    r = out["taxonomy_probe"]["runs"][0]
    print(f"[shortcut] taxonomy(>=5) groups={r['n_groups']} kept={r['n_samples_kept']} "
          f"macroF1={r['macro_f1']:.3f} (null {r['null_macro_f1_mean']:.3f}±"
          f"{r['null_macro_f1_sd']:.3f}, majority {r['majority_baseline_macro_f1']:.3f}, "
          f"p={r['perm_p_value']:.4f})")
    for r in out["taxonomy_probe"]["runs"][1:]:
        print(f"[shortcut]   sensitivity(>={r['min_group_size']}): groups={r['n_groups']} "
              f"kept={r['n_samples_kept']} macroF1={r['macro_f1']:.3f} "
              f"(null {r['null_macro_f1_mean']:.3f}, p={r['perm_p_value']:.4f})")

    # ── ④ placeholder ──
    out["prediction_residual_correlation"] = audit_prediction_residual_correlation()
    out["meta"]["runtime_s"] = round(time.time() - t0, 1)

    jpath = args.out_dir / "shortcut_audit.json"
    jpath.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    _write_md(args.out_dir / "shortcut_audit.md", out)
    print(f"[shortcut] done -> {jpath} ({out['meta']['runtime_s']}s)")


def _write_md(path: Path, d: dict) -> None:
    L = []
    A = L.append
    A("# Shortcut / Leakage Audit  —  5 Stage B-0\n")
    A(f"- generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}  (CPU only)")
    A(f"- sample: n={d['meta']['n_samples']} (pos={d['meta']['n_pos']}); "
      f"embedding = {d['meta']['embedding']}; ridge α={d['meta']['ridge_alpha']}; "
      f"{d['meta']['split']}; perm null m={d['meta']['n_perm']}\n")

    A("## ①② composition-shortcut regression (embedding → covariate)\n")
    A("| target | R² (test) | MAE | Spearman ρ | null R² (mean±SD, p95) | perm p |")
    A("|---|---|---|---|---|---|")
    for k in ("gc_regression", "loglen_regression"):
        r = d[k]
        A(f"| {r['target']} | {r['r2']:.3f} | {r['mae']:.4f} | {r['spearman']:.3f} | "
          f"{r['null_r2_mean']:.3f} ± {r['null_r2_sd']:.3f} ({r['null_r2_p95']:.3f}) | "
          f"{r['perm_p_value_r2']:.4f} |")
    A("\nInterpretation rule (RVM-4/5): R² ≤ 0.30 LOW / 0.30–0.60 MODERATE / > 0.60 HIGH.\n")

    t = d["taxonomy_probe"]
    A("## ③ taxonomy-proxy probe (embedding → host GCF group)\n")
    A(f"- proxy definition: {t['proxy']}; samples without a single unambiguous host "
      f"(chimeric/contaminant segments) excluded: {t['n_dropped_no_unambiguous_host']}")
    A(f"- **granularity limitation**: {t['granularity_note']}")
    A("")
    A("| min group | groups | kept | train/test | macro-F1 | majority baseline | null (mean±SD) | perm p |")
    A("|---|---|---|---|---|---|---|---|")
    for r in t["runs"]:
        A(f"| ≥{r['min_group_size']} | {r['n_groups']} | {r['n_samples_kept']} | "
          f"{r['n_train']}/{r['n_test']} | {r['macro_f1']:.3f} | "
          f"{r['majority_baseline_macro_f1']:.3f} | "
          f"{r['null_macro_f1_mean']:.3f} ± {r['null_macro_f1_sd']:.3f} | "
          f"{r['perm_p_value']:.4f} |")
    A("")
    A("- Interpretation rule (RVM-6): (macro-F1 − null mean) ≤ 0.10 LOW / 0.10–0.25 "
      "MODERATE / > 0.25 HIGH. Note: the ≥5 primary test covers only 30 samples / 9 "
      "test  —  —  **underpowered, no strong conclusions drawn**; the sensitivity "
      "tiers are for reference only; formal conclusions await a re-run with the full cache.\n")

    A("## ④ prediction residual correlation (placeholder, not yet run)\n")
    pr = d["prediction_residual_correlation"]
    A(f"- status: **{pr['status']}**  —  {pr['reason']}")
    A(f"- required input: `{pr['required_input']}`")
    A("- planned statistics: " + "; ".join(f"`{s}`" for s in pr["planned_statistics"]))
    A("- decision criterion: |corr| ≈ 0 → the classifier does not exploit the composition covariate.\n")

    A("## Notes\n")
    li = d["label_integrity"]
    A(f"- label integrity: prefix vs provenance.tsv label mismatches: "
      f"{li['prefix_vs_provenance_label_mismatches']} cases "
      f"(≈2%, consistent with the simulator's label_noise=0.02; {li['note']})")
    A(f"- total runtime {d['meta']['runtime_s']}s; numeric values in `shortcut_audit.json`.\n")
    Path(path).write_text("\n".join(L))


if __name__ == "__main__":
    main()
