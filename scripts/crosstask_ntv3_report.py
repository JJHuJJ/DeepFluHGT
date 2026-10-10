#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B1 cross-task RAP  —  report generation (CROSTASK_NTV3_RAP.md).
Called by scripts/crosstask_ntv3_probe.py; read-only on the results/delong DataFrames.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

DATASETS = ["human_vs_worm", "promoter_b_amyloliquefaciens", "promoter_r_capsulatus",
            "ecoli_4mc", "celegans_4mc"]
GROUPS = ["ntv3_1536", "len+gc", "kmer7_rc"]
PROBES = ["linear", "ridge"]


def _fmt(v, nd=3):
    return " — " if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def _fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return " — "
    if p == 0.0:
        return "<1e-300"
    return f"{p:.1e}" if p < 0.001 else f"{p:.3f}"


def write_report(cross: Path, df, ddf, extra: dict, total_s: float) -> None:
    L = []
    A = L.append
    A("# B1 cross-task RAP  —  NTv3-only half (CROSTASK_NTV3_RAP)\n")
    A(f"> generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  ·  DeepFluHGT  ·  "
      "corresponds to V7_5_EVIDENCE_PLAN §B1 (cross-task RAP, NTv3 half; the Evo2 half awaits a GPU window)")
    A("> embedding: `scripts/embed_ntv3_cpu.py` (frozen NTv3-pre, CPU, thread≤4)  ·  "
      "probe: `scripts/crosstask_ntv3_probe.py` (reuses the rap_probes linear/ridge protocol "
      "+ delong_test paired test)\n")

    A("## 0. Conventions (the two must be distinguished)")
    A("")
    A("| embedding variant | definition | purpose |")
    A("|---|---|---|")
    A("| **ntv3_1536** (primary) | NTv3-pre `hidden_states[-1]` (deconv full-resolution output, "
      "1536d) masked mean-pool over real tokens ([CLS]+seq+[SEP]) | RAP accessibility of the pure "
      "frozen-NTv3 representation |")
    A("| ntv3_768_proj (appendix) | the same hidden states passed through `GlobalEncoder._proj` "
      "(LayerNorm1536+Linear1536→768, **randomly initialized with seed42, not trained**) then "
      "mean-pooled — — step-by-step isomorphic to the forward pass during model training | "
      "pipeline-variant control only; the random projection is not evidence of representation quality |")
    A("")
    A("- The processing chain matches `model/impl/backbone.py GlobalEncoder._forward_nt` step by "
      "step: canonical tokenize → NTV3_REMAP → pad to a multiple of 128 (40bp→128, 200bp→256) → "
      "NTv3 forward.")
    A("- The embedding targets are each dataset's **test split** (`rap_input.fasta`, full set, not "
      "sampled); the probe uses a 70/30 stratified split within the pool (seed42) + 5 seeds (42–46) "
      "+ label-permutation null + bootstrap CI — — consistent with step0 `rap_probes.py`'s "
      "single-pool protocol. **train→test (Feng's reported protocol) is deferred to a GPU-window "
      "task** (to be filled in together with the Evo2 half); this report's numbers are not directly "
      "comparable with Feng's RF-on-embedding numbers and serve only as a cross-task accessibility "
      "profile.")
    A("- feature groups: ntv3_1536 vs **len+GC** (2d composition-shortcut baseline) vs **RC-kmer7** "
      "(8192d composition baseline); probes: linear=LogisticRegression(C=1) (Readability), "
      "ridge=α=1 closed-form (Stability).")
    A("- **ecoli_4mc is imbalanced (pos≈16.7%): primary metric AUPRC** (already noted in the "
      "README; AUROC is reported for reference only).\n")

    # ── 1. primary results ──
    A("## 1. Primary results (seed42 split; 95% bootstrap CI; permutation null p)\n")
    main = df[(df.probe.isin(PROBES)) & (~df.is_supplementary.astype(bool))]
    for ds in DATASETS:
        e = extra.get(ds, {})
        task, org, prim = (e.get("task"), e.get("org"), e.get("prim"))
        A(f"### {ds}  —  {task} ({org})  ·  n={e.get('n')} "
          f"(pos={e.get('pos')}, prevalence={e.get('prev', float('nan')):.3f})  ·  "
          f"primary metric: {prim}\n")
        A("| feature_group | probe | AUROC [CI] | AUPRC [CI] | null AUROC (p) | "
          "null AUPRC (p) | seed SD (AUC) |")
        A("|---|---|---|---|---|---|---|")
        sub = main[main.dataset == ds]
        for _, r in sub.iterrows():
            star = " ⚠" if ds == "ecoli_4mc" else ""
            A(f"| {r.feature_group} | {r.probe} | {_fmt(r.auroc)} {r.auroc_ci95} | "
              f"{_fmt(r.auprc)} {r.auprc_ci95}{star} | "
              f"{_fmt(r.null_auroc_mean)} ({_fmt_p(r.null_auroc_p)}) | "
              f"{_fmt(r.null_auprc_mean)} ({_fmt_p(r.null_auprc_p)}) | "
              f"{_fmt(r.auroc_seed_sd)} |")
        dd = df[(df.dataset == ds) & (df.probe == "linear_dedup")]
        if len(dd):
            for _, r in dd.iterrows():
                A(f"| {r.feature_group} | linear_dedup | {_fmt(r.auroc)} | "
                  f"{_fmt(r.auprc)} | (dedup sensitivity, n_test={r.n_test}) |  —  |  —  |")
        A("")
        d1 = ddf[(ddf.dataset == ds) & (ddf.probe == "linear")]
        def _d(ga, gb):
            m = d1[(d1.group_a == ga) & (d1.group_b == gb)]
            return m.iloc[0] if len(m) else None
        r1, r2 = _d("ntv3_1536", "len+gc"), _d("ntv3_1536", "kmer7_rc")
        if r1 is not None and r2 is not None:
            m = "AUPRC" if ds == "ecoli_4mc" else "AUROC"
            A(f"**DeLong (linear, same test fold)**: ntv3_1536 − len+GC = "
              f"{r1.delta_auc:+.3f} (z={r1.z:+.2f}, p={_fmt_p(r1.p)}); "
              f"ntv3_1536 − kmer7_rc = {r2.delta_auc:+.3f} (z={r2.z:+.2f}, "
              f"p={_fmt_p(r2.p)}).primary metriccaliber {m}.\n")

    # ── 2. cross-dataset overview ──
    A("## 2. Cross-dataset overview (linear probe; ridge in §1)\n")
    A("| dataset | NTv3-1536 | len+GC | kmer7 | NTv3−lenGC | NTv3−kmer7 | "
      "NTv3 > both? |")
    A("|---|---|---|---|---|---|---|")
    for ds in DATASETS:
        s = main[(main.dataset == ds) & (main.probe == "linear")]
        f = {r.feature_group: r for r in s.itertuples()}.get
        n, l, k = f("ntv3_1536"), f("len+gc"), f("kmer7_rc")
        if not (n and l and k):
            continue
        d1, d2 = n.auroc - l.auroc, n.auroc - k.auroc
        verdict = "yes" if (d1 > 0 and d2 > 0) else (
            "partial" if max(d1, d2) > 0 else "no")
        A(f"| {ds} | {_fmt(n.auroc)} | {_fmt(l.auroc)} | {_fmt(k.auroc)} | "
          f"{d1:+.3f} | {d2:+.3f} | {verdict} |")
    A("")
    A("ecoli_4mc, same table (AUPRC metric):\n")
    A("| dataset | NTv3-1536 | len+GC | kmer7 |")
    A("|---|---|---|---|")
    for ds in DATASETS:
        s = main[(main.dataset == ds) & (main.probe == "linear")]
        f = {r.feature_group: r for r in s.itertuples()}.get
        n, l, k = f("ntv3_1536"), f("len+gc"), f("kmer7_rc")
        if n:
            A(f"| {ds} | {_fmt(n.auprc)} | {_fmt(l.auprc)} | {_fmt(k.auprc)} |")
    A("")

    # ── 3. DeLong alltable ──
    A("## 3. paired DeLong alltable (seed42 test fold)\n")
    A("| dataset | probe | comparison (a vs b) | AUROC_a | AUROC_b | Δ(a−b) | z | p | significant? |")
    A("|---|---|---|---|---|---|---|---|---|")
    for _, r in ddf.iterrows():
        sig = "**yes**" if r.p < 0.05 else "no"
        A(f"| {r.dataset} | {r.probe} | {r.group_a} vs {r.group_b} | "
          f"{r.auc_a:.3f} | {r.auc_b:.3f} | {r.delta_auc:+.3f} | {r.z:+.2f} | "
          f"{_fmt_p(r.p)} | {sig} |")
    A("")

    # ── 4. appendix group 768 ──
    A("## 4. Appendix: ntv3_768_proj (random-projection variant)\n")
    A("| dataset | probe | 768_proj | 1536 | Δ(768−1536) | null p (768) |")
    A("|---|---|---|---|---|---|")
    supp = df[df.is_supplementary.astype(bool) & df.probe.isin(PROBES)]
    m1536 = {(r.dataset, r.probe): r for r in
             main.itertuples() if r.feature_group == "ntv3_1536"}
    for _, r in supp.iterrows():
        ref = m1536.get((r.dataset, r.probe))
        d_ = f"{r.auroc - ref.auroc:+.3f}" if ref is not None else " — "
        A(f"| {r.dataset} | {r.probe} | {_fmt(r.auroc)} | "
          f"{_fmt(ref.auroc) if ref is not None else ' — '} | {d_} | "
          f"{_fmt_p(r.null_auroc_p)} |")
    A("\n> The accessibility under the random projection (not trained) = the information retained "
      "by the 1536-d representation after a random linear mix; it only shows that the projected "
      "dimensionality loses no linear information; it does not explain model performance.\n")

    # ── 5. notes ──
    A("## 5. Data pitfalls and interpretation notes (continuing README §4)")
    for ds in DATASETS:
        e = extra.get(ds)
        if not e:
            continue
        notes = []
        if e.get("n_allN_dropped"):
            notes.append(f"excluded {e['n_allN_dropped']} all-N sequences (GC=NaN, hg38/ce11 "
                         f"gaps; i.e. the source of the within-pool duplicates recorded in the "
                         f"README; pool after exclusion n={e['n']})")
        if e.get("n_dup"):
            notes.append(f"remaining within-pool exact duplicates: {e['n_dup']} items "
                         "(linear_dedup sensitivity row attached)")
        if e.get("n_pct", 0) > 0.001:
            notes.append(f"mean N content {e['n_pct']*100:.2f}% (kmer automatically skips windows containing N)")
        if notes:
            A(f"- {ds}: " + "; ".join(notes))
    A("- For the 40-41bp tasks, kmer7 yields only 34-35 k-mers per sequence  —  —  the kmer "
      "baseline's features are sparse, and its relative weakness is partly due to feature "
      "density (part of the information-source ladder comparison; see README §4).")
    A("- permutation-null count tiered by n_train (linear: ≤5k→1000, >5k→100; ridge: 1000 "
      "batched closed-form); the actual counts are in the CSV `n_perm` column; n_boot likewise "
      "(≤10k→1000, else 500). For large datasets (n>5k), the linear **null fits** use tol=1e-3 "
      "early stopping (one lbfgs fit takes 35-45s at n=17.5k, so the full-protocol B=100 would "
      "need >10h); the **observed metrics** keep the full protocol (tol=1e-4) unchanged  —  —  "
      "the null sits at the chance level under permuted labels and is insensitive to the "
      "stopping tolerance.")
    A(f"- Runtime environment: CPU only (CUDA_VISIBLE_DEVICES=\"\"), BLAS/torch threads ≤4, "
      f"probe total runtime {total_s/60:.1f} min.")
    A("- embedding details (model path/variant/speed) are in the manifest field inside each "
      f"`{dataset}/{dataset}_ntv3_pooled.npz`.")
    A("- Full numeric values: `crosstask_ntv3_results.csv` / `crosstask_ntv3_delong.csv` (this directory).\n")

    out = cross / "CROSTASK_NTV3_RAP.md"
    out.write_text("\n".join(L))
    print(f"[report] -> {out}")
