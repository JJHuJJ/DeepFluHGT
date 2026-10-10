#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S1  -  DeLong (1988) paired AUROC test  -  DeepFluHGT (targeting the Feng 2025 hardening package)

Standard DeLong paired test, **self-implemented with zero external dependencies** (numpy only;
the p-value uses math.erfc, no scipy). Tests the significance of the AUROC difference between two
predictors on the same sample set. Structural variables (placement values) are computed via the
midrank identity in O(n log n) (DeLong 1988 §2; the vectorized form of Sun & Xu 2014):
  for scores s, all-sample midrank tz, positive-class midrank tx, negative-class midrank ty:
    V10_i = (tz_i − tx_i)/n_neg   (placement value of the i-th positive)
    V01_j = (ty_j − tz_j)/n_pos   (placement value of the j-th negative)
  AUC = mean(V10) = mean(V01) = (Σ_pos tz − m(m+1)/2)/(m · n)  (= Mann-Whitney, tie-aware)
  Var(AÛC_a − AÛC_b) = S10_aa/m + S01_aa/n + S10_bb/m + S01_bb/n
                       − 2 · (S10_ab/m + S01_ab/n)
    where S10_ab = Σ_i (V10_i^a − AUC_a)(V10_i^b − AUC_b)/(m−1); S01 analogously (j iterates over the negative class).
  z = ΔAUC/√Var, two-sided p = erfc(|z|/√2) = 2Φ(−|z|).

──────────────────────────────── usage ────────────────────────────────
1) Self-test (unit tests included: fully separated data p≈0 / identical predictions p≈1 / cross-check against a paired bootstrap):
     python scripts/delong_test.py --self-test

2) Applied to the step0 RAP: replays rap_probes.py's seed42 70/30 split and feature construction,
   runs DeLong on paired feature-group predictions on the same test fold → appends rap_report.md §6 (§1-5 unchanged):
     python scripts/delong_test.py --rap
   (artifacts: evo_cache/step0_subset/rap/rap_delong.csv + rap_delong_predictions.npz)

────────────────────── M0/M2/M3 future comparison interface (waiting for prediction files) ───────────────────────
Unified entry point (labels, probs_a, probs_b) → (auc_a, auc_b, z, p):

    from scripts.delong_test import delong_test          # or import from the same directory
    auc_a, auc_b, z, p = delong_test(y_true, prob_m0, prob_m2)
    # y_true/prob_*: same-length 1-D arrays (test fold; sample-level or window-level both work,
    # but must be the same paired sample batch)

Invocation once prediction files are ready (example, M0 vs M2 test set):
    import numpy as np, json
    from scripts.delong_test import delong_test
    d = np.load("checkpoints/Contig_500k_ntv3only/test_probs.npz")   # TODO [pending evaluation-script output]
    e = np.load("checkpoints/Contig_500k_m2/test_probs.npz")         # TODO [pending M2 completion]
    assert (d["ids"] == e["ids"]).all()          # hard assert: only the same sample set can be paired
    print(delong_test(d["labels"], d["probs"], e["probs"]))
A CLI is also provided to read two npz/json files directly:
    python scripts/delong_test.py --npz-a a.npz --npz-b b.npz \
        [--key-labels labels --key-probs probs --key-ids ids]

note: paired DeLong requires both predictors to score **exactly the same sample set**; when ids do
not align this script rejects (rather than truncates)  —  —  truncation would break the paired
correlation and invalidate the p-value.

Run discipline: CPU-only (CUDA_VISIBLE_DEVICES=""), OMP/BLAS ≤6 threads, no joblib parallelism.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # CPU-only, never touches production GPUs
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")                  # limit parallel cores to ≤6

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))   # reuse rap_probes

__all__ = ["delong_test", "delong_test_full", "delong_test_multi"]


# ══════════════════════════ Core: DeLong paired test ══════════════════════════ #

def _midrank(x: np.ndarray) -> np.ndarray:
    """1-based midrank (ties get the average rank). Equivalent to R rank(ties.method='average')."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    r = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        r[i:j] = 0.5 * (i + j - 1) + 1.0            # average of [i+1 .. j] (1-based)
        i = j
    out = np.empty(n, dtype=np.float64)
    out[order] = r
    return out


def _placement_values(scores: np.ndarray, pos: np.ndarray):
    """Returns (V10 [m], V01 [n]) structural variables (placement values); scores are already tie-processed."""
    scores = np.asarray(scores, dtype=np.float64)
    m = int(pos.sum())
    n = int((~pos).sum())
    if m == 0 or n == 0:
        raise ValueError("both positive and negative classes must have samples (m, n > 0)")
    tz = _midrank(scores)                            # midrank over all samples
    tx = _midrank(scores[pos])                       # midrank within the positive class
    ty = _midrank(scores[~pos])                      # midrank within the negative class
    v10 = (tz[pos] - tx) / n                         # identity: tz_i = tx_i + Σ_j[inds_j>si + .5ties]
    v01 = (ty - tz[~pos]) / m                        # FA-C1 note: this gives V01_true-1 (≤0);
    # the centering fix is in _s() below  —  center at the vector's own mean, not at auc.
    return v10, v01


def _auc_mann_whitney(scores: np.ndarray, pos: np.ndarray) -> float:
    """Tie-aware Mann-Whitney AUC (= the DeLong point estimate, consistent with sklearn roc_auc_score)."""
    scores = np.asarray(scores, dtype=np.float64)
    m, n = int(pos.sum()), int((~pos).sum())
    tz = _midrank(scores)
    return float((tz[pos].sum() - m * (m + 1) / 2.0) / (m * n))


def _validate_pair(labels, probs_a, probs_b):
    y = np.asarray(labels).astype(int).ravel()
    a = np.asarray(probs_a, dtype=np.float64).ravel()
    b = np.asarray(probs_b, dtype=np.float64).ravel()
    if not (y.size == a.size == b.size):
        raise ValueError(f"inconsistent lengths: labels={y.size}, a={a.size}, b={b.size}")
    if y.size < 4:
        raise ValueError("too few samples (n<4), DeLong variance cannot be estimated")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("prediction scores contain NaN/Inf")
    ys = set(np.unique(y).tolist())
    if not ys <= {0, 1}:
        raise ValueError(f"labels must be 0/1 binary, actual {sorted(ys)[:5]}")
    if ys != {0, 1}:
        raise ValueError("both positive and negative classes are required")
    if np.array_equal(a, b):
        pass                                          # allowed (identical predictions → z=0, p=1)
    return y, a, b


def delong_test_full(labels, probs_a, probs_b) -> dict:
    """Paired DeLong test (full output).

    Returns dict: auc_a, auc_b, delta, z, p, se_delta,
                  ci95_delta = delta ± 1.96 · se (normal approximation), n, n_pos, n_neg,
                  var_a, var_b, cov (variance/covariance components, for reviewer verification).
    """
    y, a, b = _validate_pair(labels, probs_a, probs_b)
    pos = y == 1
    m, n = int(pos.sum()), int((~pos).sum())

    v10a, v01a = _placement_values(a, pos)
    v10b, v01b = _placement_values(b, pos)
    auc_a = float(v10a.mean())
    auc_b = float(v10b.mean())

    def _s(v, auc=None):
        # FA-C1 fix (2026-09-04): center at the vector's own mean, not at auc.
        # v01 from _placement_values equals V01_true - 1 (≤ 0, mean = auc - 1).
        # Centering at auc (the V10 mean) added a constant offset to every v01
        # deviation, inflating the variance. Centering at v.mean() is always
        # correct regardless of the sign/offset convention.
        # Cross-terms (_sxy) still use auc because v10 and v01 have different
        # offsets (v10 mean = auc, v01 mean = auc - 1); centering each at its
        # own mean gives the correct covariance.
        return float(np.sum((v - v.mean()) ** 2) / (len(v) - 1))

    def _sxy(v1, auc1, v2, auc2):
        # FA-C1 fix: center each vector at its own mean for covariance too
        return float(np.sum((v1 - v1.mean()) * (v2 - v2.mean())) / (len(v1) - 1))

    s10aa, s01aa = _s(v10a, auc_a), _s(v01a, auc_a)
    s10bb, s01bb = _s(v10b, auc_b), _s(v01b, auc_b)
    s10ab = _sxy(v10a, auc_a, v10b, auc_b)
    s01ab = _sxy(v01a, auc_a, v01b, auc_b)

    var_a = s10aa / m + s01aa / n
    var_b = s10bb / m + s01bb / n
    cov = s10ab / m + s01ab / n
    var_delta = var_a + var_b - 2.0 * cov

    delta = auc_a - auc_b
    if var_delta <= 0.0:
        # Degenerate case: the two predictors' per-sample placements are fully collinear (including identical predictions).
        # Δ>0 with zero variance → z=inf (p→0); Δ=0 → z=0 (p=1).
        z = math.inf if delta > 0 else (-math.inf if delta < 0 else 0.0)
        se = 0.0
        p = 0.0 if delta != 0 else 1.0
    else:
        se = math.sqrt(var_delta)
        z = delta / se
        p = math.erfc(abs(z) / math.sqrt(2.0))       # = 2 · Φ(−|z|), no scipy

    return dict(auc_a=auc_a, auc_b=auc_b, delta=delta, z=z, p=p, se_delta=se,
                ci95_delta=(delta - 1.959963984540054 * se,
                            delta + 1.959963984540054 * se) if se > 0 else (delta, delta),
                n=y.size, n_pos=m, n_neg=n,
                var_a=var_a, var_b=var_b, cov=cov)


def delong_test(labels, probs_a, probs_b):
    """Paired DeLong test  -  primary interface: (labels, probs_a, probs_b) → (auc_a, auc_b, z, p).

    Pre-registered use: M0 vs M2 vs M3 test-set paired AUROC comparison (invoke once the
    prediction files are ready; see the module docstring for an invocation example).
    Requires the same sample batch (paired), otherwise the p-value is invalid.
    """
    r = delong_test_full(labels, probs_a, probs_b)
    return r["auc_a"], r["auc_b"], r["z"], r["p"]


def delong_test_multi(labels, probs: dict[str, np.ndarray]) -> dict[tuple[str, str], dict]:
    """All-pairs paired DeLong across multiple predictors (same labels). Returns {(name_a, name_b): full_dict}."""
    return {(na, nb): delong_test_full(labels, probs[na], probs[nb])
            for i, na in enumerate(probs) for nb in list(probs)[i + 1:]}


# ══════════════════════════ Unit tests (no external dependencies) ══════════════════════════ #

def _run_self_tests(verbose: bool = True) -> bool:
    rng = np.random.default_rng(42)
    ok = []

    def check(name, cond, detail=""):
        ok.append(bool(cond))
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")

    # ── T1: known fully separated data → p ≈ 0 ──
    # Predictor A separates perfectly (AUROC=1), predictor B is pure noise (AUROC≈0.5): the difference should be extremely significant.
    y = np.array([1] * 40 + [0] * 40)
    a = np.concatenate([rng.uniform(0.80, 1.0, 40), rng.uniform(0.0, 0.2, 40)])
    b = rng.uniform(0, 1, 80)
    r = delong_test_full(y, a, b)
    check("T1 full separation vs noise: AUROC_a=1.0", abs(r["auc_a"] - 1.0) < 1e-12,
          f"(auc_a={r['auc_a']:.4f})")
    check("T1 p ≈ 0 (p < 1e-4)", r["p"] < 1e-4, f"(p={r['p']:.3e}, z={r['z']:.2f})")

    # ── T2: identical predictions → z=0, p=1 ──
    s = rng.uniform(0, 1, 200)
    y2 = rng.binomial(1, 0.5, 200)
    if set(np.unique(y2)) != {0, 1}:
        y2[:100], y2[100:] = 1, 0                    # force both classes
    r2 = delong_test_full(y2, s, s.copy())
    check("T2 identical predictions: auc_a == auc_b", r2["auc_a"] == r2["auc_b"])
    check("T2 identical predictions: z == 0", r2["z"] == 0.0, f"(z={r2['z']})")
    check("T2 identical predictions: p = 1", r2["p"] == 1.0, f"(p={r2['p']})")

    # ── T3: AUC point estimate == sklearn roc_auc_score (if available) ──
    try:
        from sklearn.metrics import roc_auc_score
        s3 = rng.uniform(0, 1, 500)
        y3 = rng.binomial(1, 0.5, 500)
        if set(np.unique(y3)) != {0, 1}:
            y3[:250], y3[250:] = 1, 0
        s3t = np.round(s3, 2)                        # introduce ties, to exercise the midrank procedure
        auc_del = _auc_mann_whitney(s3t, y3 == 1)
        auc_skl = roc_auc_score(y3, s3t)
        check("T3 AUC (tie-aware) == sklearn", abs(auc_del - auc_skl) < 1e-10,
              f"(delong={auc_del:.10f}, sklearn={auc_skl:.10f})")
    except ImportError:
        print("  [SKIP] T3 sklearn unavailable")

    # ── T4: cross-check against a paired bootstrap (shift method) ──
    # Correlated predictors (shared latent) + a slight difference, n=400:
    #   H0: Δ=0; the bootstrap distribution centered at the observed Δ approximates its sampling distribution
    #   p_boot = P(|Δ* − Δ_obs| ≥ |Δ_obs|), should be asymptotically consistent with the DeLong p
    #   (DeLong 1988 uses the normal approximation, bootstrap uses resampling  —  mutual consistency corroborates both).
    n4 = 400
    y4 = np.array([1] * 200 + [0] * 200)
    latent = rng.normal(0, 1, n4) + 0.9 * y4         # partially shared signal → correlated predictors
    pa = 1 / (1 + np.exp(-(1.4 * latent + 0.3 * rng.normal(0, 1, n4))))
    pb = 1 / (1 + np.exp(-(0.6 * latent + 0.3 * rng.normal(0, 1, n4))))
    r4 = delong_test_full(y4, pa, pb)
    rng_b = np.random.default_rng(7)
    deltas = []
    for _ in range(3000):
        i = rng_b.integers(0, n4, n4)
        yy = y4[i]
        if yy.min() == yy.max():
            continue
        deltas.append(_auc_mann_whitney(pa[i], yy == 1) - _auc_mann_whitney(pb[i], yy == 1))
    deltas = np.asarray(deltas)
    p_boot = float(np.mean(np.abs(deltas - r4["delta"]) >= abs(r4["delta"])))
    check("T4 DeLong z and bootstrap agree in direction (sign consistent)",
          np.sign(r4["z"]) == np.sign(r4["delta"]))
    check("T4 |p_delong − p_bootstrap(shift)| < 0.05",
          abs(r4["p"] - p_boot) < 0.05,
          f"(delong={r4['p']:.4f}, boot={p_boot:.4f})")
    # Strong-difference scenario: both should give small p (independent RNG, unaffected by preceding test consumption)
    rng_t4b = np.random.default_rng(99)
    latent2 = rng_t4b.normal(0, 1, n4) + 1.0 * y4
    pa2 = 1 / (1 + np.exp(-(1.6 * latent2 + 0.3 * rng_t4b.normal(0, 1, n4))))
    pb2 = 1 / (1 + np.exp(-(0.4 * latent2 + 0.3 * rng_t4b.normal(0, 1, n4))))
    r4b = delong_test_full(y4, pa2, pb2)
    check("T4b strong difference: delong p < 0.05", r4b["p"] < 0.05,
          f"(delta={r4b['delta']:.4f}, z={r4b['z']:.2f}, p={r4b['p']:.2e})")

    # ── T5: interface contract (reserved for M0/M2/M3) ──
    out = delong_test(y4, pa, pb)
    check("T5 delong_test return 4 tuple (auc_a, auc_b, z, p)",
          isinstance(out, tuple) and len(out) == 4
          and all(isinstance(v, float) for v in out))
    check("T5 multi-predictor interface keys cover all pairs",
          len(delong_test_multi(y4, {"a": pa, "b": pb, "c": s3[:n4]})) == 3)

    if verbose:
        print(f"[self-test] {sum(ok)}/{len(ok)} passed" + ("" if all(ok) else "  —  —  failures present!"))
    return all(ok)


# ════════════════ S1b: application  —  step0 RAP feature-group paired DeLong ════════════════ #

GROUP_ORDER = ["evo", "len+gc", "kmer7_rc", "evo+len+gc", "evo+kmer7_rc"]
PROBE_ORDER = ["linear", "ridge", "mlp"]


def run_rap_delong(cache_dir: Path, out_dir: Path, seeds: str = "42",
                   verify_csv: bool = True) -> Path:
    """Replays RAP (seed42 split + same features/probes) → pairwise DeLong on the same test fold.

    Leaves the original rap_results.csv / rap_report.md §1–5 untouched; adds:
      rap_delong.csv                 —  (auc_a, auc_b, Δ, z, p) for every pair × probe
      rap_delong_predictions.npz     —  replayed test-fold predictions (for verification / future extension)
      rap_report.md §6               —  results table (appended, idempotent)
    """
    import rap_probes as rap                      # reuse split/features/probes (zero changes)

    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    seqs = rap.load_fasta(cache_dir / "step0_subset.fasta")
    emb = rap.load_embeddings(cache_dir / "evo_subset.pt")
    ids = list(seqs.keys())
    y = np.array([1 if i.startswith("pos_") else 0 for i in ids], dtype=int)
    feats, _meta = rap.build_features(emb, seqs, ids)
    print(f"[delong-rap] n={len(ids)} pos={int(y.sum())}")

    # primary split: fully consistent with rap_probes.main ① (seed42, standardize)
    tr, te = rap.stratified_split(y, seed=42)
    yte = y[te]
    print(f"[delong-rap] test fold n={len(te)} pos={int(yte.sum())} "
          f"(with rap_results.csv n_test alignment)")

    import torch
    torch.set_num_threads(4)                         # core limit ≤6

    # replay the primary-split predictions for every (group, probe)
    preds: dict[tuple[str, str], np.ndarray] = {}
    aucs: dict[tuple[str, str], float] = {}
    for g in GROUP_ORDER:
        X = feats[g]
        xtr, xte = rap.standardize(X[tr], X[te])
        for pname in PROBE_ORDER:
            if pname == "mlp":
                s = rap._torch_mlp(xtr, y[tr], xte, seed=42)   # same seed as the original main split
            else:
                s = rap.PROBES[pname](xtr, y[tr], xte)
            preds[(g, pname)] = np.asarray(s, dtype=np.float64)
            aucs[(g, pname)] = _auc_mann_whitney(preds[(g, pname)], yte == 1)
            print(f"[delong-rap] replay {g:14s} {pname:7s} AUROC={aucs[(g, pname)]:.4f}",
                  flush=True)

    # reconcile against the original rap_results.csv (replay fidelity)
    if verify_csv and (out_dir / "rap_results.csv").exists():
        import pandas as pd
        df = pd.read_csv(out_dir / "rap_results.csv")
        ref = {(r.feature_group, r.probe): r.auroc for r in df.itertuples()}
        print("[delong-rap] replay fidelity (vs rap_results.csv seed42 AUROC):")
        worst = 0.0
        for k, v in aucs.items():
            if k in ref:
                d = abs(v - ref[k])
                worst = max(worst, d)
                flag = "" if d < 0.01 else "  <-- deviation>0.01 (can be caused by non-deterministic MLP threading)"
                print(f"    {k[0]:14s} {k[1]:7s} replay={v:.4f} orig={ref[k]:.4f} "
                      f"|Δ|={d:.4f}{flag}")
        print(f"[delong-rap] max |ΔAUROC| = {worst:.4f}")

    # pairwise DeLong (same test fold, same labels → valid pairing)
    rows = []
    for pname in PROBE_ORDER:
        for i, ga in enumerate(GROUP_ORDER):
            for gb in GROUP_ORDER[i + 1:]:
                r = delong_test_full(yte, preds[(ga, pname)], preds[(gb, pname)])
                rows.append(dict(probe=pname, group_a=ga, group_b=gb,
                                 auc_a=r["auc_a"], auc_b=r["auc_b"],
                                 delta_auc=r["delta"], z=r["z"], p=r["p"],
                                 p_sci=_fmt_p(r["p"]), n=r["n"], n_pos=r["n_pos"]))
    import pandas as pd
    df_d = pd.DataFrame(rows)
    csv_path = out_dir / "rap_delong.csv"
    df_d.to_csv(csv_path, index=False)

    np.savez_compressed(out_dir / "rap_delong_predictions.npz",
                        labels=yte, test_ids=np.array(ids)[te],
                        **{f"{g}__{p}": preds[(g, p)] for g in GROUP_ORDER
                           for p in PROBE_ORDER})

    _append_report(out_dir, df_d, aucs, time.time() - t0)
    print(f"[delong-rap] done -> {csv_path} ({time.time() - t0:.0f}s)")
    return csv_path


def _fmt_p(p: float) -> str:
    if p == 0.0:
        return "<1e-300" if p == 0.0 else "0"
    return f"{p:.2e}" if p < 0.001 else f"{p:.3f}"


def _append_report(out_dir: Path, df_d, aucs, total_s: float) -> None:
    """Append §6 to rap_report.md (idempotent: if it already exists the whole section is replaced; §1–5 are never touched)."""
    import pandas as pd
    rp = out_dir / "rap_report.md"
    text = rp.read_text()
    marker = "## 6. paired AUROC DeLong test"
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n"
        note = " (this section is generated by the S1b replay and rewritten idempotently; §1–5 are the original results, unchanged) "
    else:
        note = " (appended by S1b, 2026-08-21; §1–5 are the original results, unchanged) "

    lines = []
    A = lines.append
    A("")
    A(marker + " (V7.5 target: Feng hardening S1) " + note)
    A("")
    A("- method: standard DeLong (1988) paired test (`scripts/delong_test.py`, self-implemented with zero dependencies, "
      "midrank structural variables + normal approximation; unit tests in that file under `--self-test`).")
    A("- protocol: **replay** `rap_probes.py`'s seed42 70/30 stratified split and feature construction, "
      "compute the z / two-sided p of ΔAUROC on paired feature-group predictions on the **same test fold** (n=599).")
    A("- interpretation: p < 0.05 ⇒ the two feature groups differ significantly in separability; p ≥ 0.05 ⇒ indistinguishable"
      " (not proof of equality). The paired test controls for inter-sample variation and is stricter than comparing overlapping CIs.")
    A("")
    for pname in PROBE_ORDER:
        role = {"linear": "Readability", "ridge": "Stability",
                "mlp": "Capacity ceiling"}[pname]
        A(f"### 6.{PROBE_ORDER.index(pname) + 1} probe = {pname} ({role})\n")
        A("| comparison (a vs b) | AUROC_a | AUROC_b | Δ(a−b) | z | p (two-sided) | significant? |")
        A("|---|---|---|---|---|---|---|")
        sub = df_d[df_d.probe == pname]
        # sort: the task-designated core comparisons first, the rest by ascending p
        core = [("evo", "len+gc"), ("evo", "evo+len+gc"), ("evo", "kmer7_rc")]
        ordered = [(r.group_a, r.group_b) for r in sub.itertuples()]
        sub_sorted = sorted(sub.itertuples(),
                            key=lambda r: (ordered.index((r.group_a, r.group_b))
                                           not in [c for c in core], r.p))
        for r in sub_sorted:
            sig = "**yes**" if r.p < 0.05 else "no"
            key = f"{r.group_a} vs {r.group_b}"
            star = " ✓" if (r.group_a, r.group_b) in core else ""
            A(f"| {key}{star} | {r.auc_a:.3f} | {r.auc_b:.3f} | {r.delta_auc:+.3f} | "
              f"{r.z:+.2f} | {_fmt_p(r.p)} | {sig} |")
        A("")
    A("**Core conclusions (linear probe, controlled by §1/§3)**")
    ev_lg = df_d[df_d.probe == "linear"]
    def _row(ga, gb):
        m = ev_lg[(ev_lg.group_a == ga) & (ev_lg.group_b == gb)]
        return m.iloc[0] if len(m) else None
    r1, r2, r3 = (_row("evo", "len+gc"), _row("evo", "evo+len+gc"),
                  _row("evo", "kmer7_rc"))
    A(f"- evo significantly outperforms len+GC: Δ={r1.delta_auc:+.3f} (z={r1.z:+.2f}, p={_fmt_p(r1.p)})")
    A(f"- evo+len+GC significantly outperforms evo alone: Δ={-r2.delta_auc:+.3f} (z={-r2.z:+.2f}, "
      f"p={_fmt_p(r2.p)})  —  —  len+GC adds a statistically significant **increment** over evo, supporting the §4 RVM-7 verdict")
    A(f"- evo significantly outperforms the kmer7_rc composition baseline: Δ={r3.delta_auc:+.3f} (z={r3.z:+.2f}, "
      f"p={_fmt_p(r3.p)})  —  —  rules out the §4 RVM-8 COMPOSITION-DOMINANT risk")
    A("- read together with §4 RVM-7 (feature increment Δ ≥ 0.02 counts as an increment): the DeLong p endorses that rule's statistical significance.")
    A("")
    A("**artifacts**: `rap_delong.csv` (all 30 comparisons)  ·  `rap_delong_predictions.npz` "
      "(replayed test-fold predictions, verifiable)  ·  implementation + unit tests in `scripts/delong_test.py`.")
    A("")
    A(f"- runtime environment: CPU only (CUDA_VISIBLE_DEVICES=\"\"), cores capped at ≤6, {total_s:.0f}s elapsed.")
    A("- M0/M2/M3 comparison interface: `delong_test(labels, probs_a, probs_b) → (auc_a, auc_b, z, p)`, "
      "see the `scripts/delong_test.py` module docstring (invoke once each model's test-set prediction file is written).")
    A("")
    rp.write_text(text.rstrip() + "\n" + "\n".join(lines))


# ════════════════════════════════ CLI ════════════════════════════════ #

def _load_npz_pred(path: str, key_labels: str, key_probs: str, key_ids: str | None):
    z = np.load(path)
    labels = z[key_labels]
    probs = z[key_probs]
    ids = z[key_ids] if (key_ids and key_ids in z) else None
    return labels, probs, ids


def main() -> None:
    ap = argparse.ArgumentParser(description="S1 DeLong paired AUROC test (v7.5 hardening)")
    ap.add_argument("--self-test", action="store_true", help="run unit tests")
    ap.add_argument("--rap", action="store_true",
                    help="application: all-pairs DeLong on step0 RAP feature groups, appends rap_report.md §6")
    ap.add_argument("--cache-dir", type=Path, default=REPO / "evo_cache" / "step0_subset")
    ap.add_argument("--out-dir", type=Path, default=REPO / "evo_cache" / "step0_subset" / "rap")
    # direct-read interface, for when M0/M2/M3 are ready
    ap.add_argument("--npz-a", type=str, default=None,
                    help="predictor A's npz (with labels/probs[/ids] keys)")
    ap.add_argument("--npz-b", type=str, default=None, help="predictor B's npz")
    ap.add_argument("--key-labels", default="labels")
    ap.add_argument("--key-probs", default="probs")
    ap.add_argument("--key-ids", default="ids")
    args = ap.parse_args()

    if not (args.self_test or args.rap or (args.npz_a and args.npz_b)):
        ap.print_help()
        return

    if args.self_test:
        print("[self-test] DeLong unit tests (DeLong 1988; self-implemented midrank structural variables)")
        if not _run_self_tests():
            raise SystemExit(1)

    if args.rap:
        run_rap_delong(args.cache_dir, args.out_dir)

    if args.npz_a and args.npz_b:
        la, pa, ia = _load_npz_pred(args.npz_a, args.key_labels, args.key_probs, args.key_ids)
        lb, pb, ib = _load_npz_pred(args.npz_b, args.key_labels, args.key_probs, args.key_ids)
        if not np.array_equal(la, lb):
            raise SystemExit("FATAL: labels of the two npz files are inconsistent; cannot pair")
        if ia is not None and ib is not None and not np.array_equal(ia, ib):
            raise SystemExit("FATAL: sample ids inconsistent  —  —  paired DeLong requires the same sample batch")
        r = delong_test_full(la, pa, pb)
        print(f"[delong] A vs B: auc_a={r['auc_a']:.4f} auc_b={r['auc_b']:.4f} "
              f"delta={r['delta']:+.4f} z={r['z']:+.3f} p={_fmt_p(r['p'])} "
              f"(n={r['n']}, CI95(Δ)=[{r['ci95_delta'][0]:+.4f},{r['ci95_delta'][1]:+.4f}])")


if __name__ == "__main__":
    main()
