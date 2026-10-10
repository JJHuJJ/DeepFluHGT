#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""residual_corr_terminal.py — M2(ep9)/M3(ep7) terminal decision-layer residual re-audit (val caliber).

background: SI ST1-C2 and SF7C promise "M2/M3 residual re-audit at terminal". The M0 version
(residual_corr.py) consumes the engine's val monitor npz (m0_trajectory/trainval_gap_epoch_2.npz);
the M2/M3 engines never saved window-level artifacts → scripts/val_window_probs_terminal.py supplies the val window
probabilities (BEDParser event-aware windows, isomorphic to the M0 npz), which this script consumes.

protocol: residual_corr.py (M0 val version) verbatim reuse — feature (per-window GC/log-len/kmer7-PCA10)
/ shortcut-only logistic 5 fold OOF / residual (prob primarycaliber + logit note) / inference caliber
(contig level cluster perm 2000 + bootstrap 2000 + phylum η²).alignment = with
rebuild_val_windows() ofwindow id/tagper-itemassert (rc0.main same type).

output (the file names the SF7 renderer expects by contract): docs/evidence/residual_corr_{m2,m3}final.json
     (+ residual_corr_terminal_report.md summary)

⚠️ protocol record (2026-09-10): an earlier version used the events=[] window set on the test split (inheriting the terminal
evaluation construction) — labels degenerated into contig-length indicators, the shortcut-model target became invalid, and those results were discarded.
this version uses val with event-aware labels, strictly the same protocol as M0.

usage: python scripts/residual_corr_terminal.py
      [--model m2] [--model m3] (default: run both; pure CPU, ~4 min/model)
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "24")
os.environ.setdefault("MKL_NUM_THREADS", "24")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "24")

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import residual_corr as rc0 # M0 version: feature/statisticsimplementallre-use
import rap_probes as rap

DATA_VAL = REPO / "simulated_data" / "Contig_500k" / "val"

MODELS = {
    "m2": dict(npz=REPO / "docs/evidence/terminal/M2/val_window_probs_final.npz",
               name="M2 (terminal ep9, augmented representation)"),
    "m3": dict(npz=REPO / "docs/evidence/terminal/M3/val_window_probs_final.npz",
               name="M3 (terminal ep7, capacity control)"),
}


def run_model(key: str, n_perm: int, n_boot: int, threads: int):
    m = MODELS[key]
    t0 = time.time()
    z = np.load(m["npz"], allow_pickle=True)
    probs = z["val_probs"].astype(np.float64)
    y = z["val_labels"].astype(int)
    ids = z["val_ids"].astype(str)
    n = len(probs)
    print(f"[in] {key}: n={n} pos={int(y.sum())} "
          f"(AUROC preflight: {rap.metrics(y, probs)[0]:.4f})", flush=True)

    # -- windowrebuild + alignmentassert (rc0.main same type) --
    ds = rc0.rebuild_val_windows()
    assert len(ds) == n, f"window count mismatch: {len(ds)} vs {n}"
    win_ids = np.array([ds.samples[w[0]].seq_id for w in ds._window_index])
    assert (win_ids == ids).all(), "windoworder/id with npz inconsistent"
    win_lab = np.array([w[3] for w in ds._window_index], dtype=int)
    assert (win_lab == y).all(), "windowtagwith npz inconsistent"
    print("[align] window id/tagconsistent item-by-item ✓", flush=True)

    gc, ln, kmer_bin = rc0.compute_window_features(
        ds, n_jobs=threads, tmpdir=REPO / "docs/evidence")
    pcs, evr = rc0.kmer_pca(kmer_bin, n)
    loglen = np.log10(ln)
    cid = rc0._contig_codes(ids)

    S = np.column_stack([gc, loglen, pcs]).astype(np.float64)
    ok_m = np.isfinite(S).all(1)
    S_use, y_use = S[ok_m], y[ok_m]
    oof, _pf, _m, sc_meta = rc0.shortcut_oof(S_use, y_use)
    p_short = np.full(n, np.nan)
    p_short[ok_m] = oof
    print(f"[shortcut] OOF AUROC={sc_meta['oof_auroc']:.4f}", flush=True)

    resid = probs - p_short
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) /
                             (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    resid_logit = logit(probs) - logit(p_short)

    rng = np.random.default_rng(42)
    res = {"model": m["name"], "split": "Contig_500k/val (same protocol as the M0 audit)",
           "n_windows": int(n), "n_valid": int(ok_m.sum()),
           "protocol": ("residual_corr.py (M0 val version) verbatim reuse; probabilitysource = "
                        "val_window_probs_terminal.py artifact (terminal checkpoint, "
                        "BEDParser event-aware windows, alignment assert passed)")}
    for tgt_name, tgt in [("residual", resid), ("probs_control", probs)]:
        for feat_name, feat in [("GC", gc), ("log10_len", loglen)]:
            mm = ok_m & np.isfinite(feat) & np.isfinite(tgt)
            r_win = rc0.pearson(feat[mm], tgt[mm])
            xc = rc0.contig_means(feat[mm], cid[mm])
            yc = rc0.contig_means(tgt[mm], cid[mm])
            r_c, p_perm = rc0.contig_perm_p_r(xc, yc, n_perm, rng)
            lo, hi = rc0.contig_bootstrap_r(xc, yc, n_boot, rng)
            r_s = rc0.spearman(xc, yc)
            res[f"{tgt_name}__{feat_name}"] = dict(
                pearson_r_window=float(r_win), pearson_r_contig=r_c,
                spearman_r_contig=float(r_s), perm_p=p_perm,
                boot_ci95=[lo, hi], n_contig=int(len(xc)))
            print(f" {tgt_name:13s} vs {feat_name:10s} r_win={r_win:+.4f} "
                  f"r_contig={r_c:+.4f} (ρ={r_s:+.4f}), perm p={p_perm:.4f}, "
                  f"CI[{lo:+.4f},{hi:+.4f}]", flush=True)

    try:
        gcf2ph = rc0.load_phylum_map()
        host = rc0.host_genome_map(DATA_VAL / "provenance.tsv")
        phyl_contig = {c: gcf2ph.get(g) for c, g in host.items()}
        phyl_win = np.array([phyl_contig.get(i, None) for i in ids],
                            dtype=object)
        known = np.array([p is not None and p not in rc0.UNKNOWN_PHYLA
                          for p in phyl_win])
        cnt = Counter(phyl_win[known])
        print(f"[phylum] alreadyknown phylum window {int(known.sum())}/{n} "
              f"({len(cnt)} phyla)", flush=True)
        for tgt_name, tgt in [("residual", resid), ("probs_control", probs)]:
            mk = known & np.isfinite(tgt)
            e_obs, p_perm = rc0.cluster_perm_p_eta(tgt[mk], phyl_win[mk],
                                                   cid[mk], n_perm, rng)
            res[f"{tgt_name}__phylum"] = dict(
                eta2=e_obs, perm_p=p_perm, n_known_windows=int(mk.sum()),
                n_phyla=int(len(cnt)))
            print(f" {tgt_name:13s} vs phylum η²={e_obs:.4f} "
                  f"(perm p={p_perm:.4f})", flush=True)
    except FileNotFoundError as e:
        res["phylum_missing"] = str(e)

    res["shortcut_model"] = sc_meta
    res["kmer_pca_evr"] = dict(top=[float(v) for v in evr[:3]],
                               cum10=float(evr.sum()))
    res["val_side"] = dict(auroc=rap.metrics(y, probs)[0],
                           auprc=rap.metrics(y, probs)[1])
    res["resid_logit_check"] = {
        f: dict(pearson_r=rc0.pearson(resid_logit[ok_m],
                                      {"GC": gc, "log10_len": loglen}[f][ok_m]))
        for f in ["GC", "log10_len"]}
    res["runtime_s"] = round(time.time() - t0, 1)

    out_json = REPO / "docs/evidence" / f"residual_corr_{key}final.json"
    out_json.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    os.remove(REPO / "docs/evidence" / "resid_kmer_win.f32")
    print(f"[done] {key}: {res['runtime_s']/60:.1f} min -> {out_json}",
          flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["m2", "m3"], action="append",
                    default=None, help="default: run both")
    ap.add_argument("--n-perm", type=int, default=rc0.N_PERM)
    ap.add_argument("--n-boot", type=int, default=rc0.N_BOOT)
    ap.add_argument("--threads", type=int, default=24)
    args = ap.parse_args()
    summary = {}
    for key in (args.model or ["m2", "m3"]):
        r = run_model(key, args.n_perm, args.n_boot, args.threads)
        summary[key] = {
            "r_gc_logit": round(r["resid_logit_check"]["GC"]["pearson_r"], 4),
            "r_len_logit": round(r["resid_logit_check"]["log10_len"]["pearson_r"], 4),
            "r_gc_prob_contig": round(r["residual__GC"]["pearson_r_contig"], 4),
            "r_len_prob_contig": round(r["residual__log10_len"]["pearson_r_contig"], 4),
            "eta2_phylum": round(r.get("residual__phylum", {}).get("eta2", float("nan")), 4),
            "shortcut_oof_auroc": round(r["shortcut_model"]["oof_auroc"], 4),
        }
    md = ["# M2/M3 terminal decision-layer residual re-audit (val caliber, with M0 isomorphic) — summary", "",
          "| model | r(GC) logit* | r(len) logit* | r(GC) prob-contig | r(len) prob-contig | η²(phylum) | shortcut OOF AUROC |",
          "|---|---|---|---|---|---|---|"]
    m0 = json.load(open(REPO / "docs/evidence/residual_corr_m0.json"))
    md.append(f"| M0 (reference, val ep2) | {m0['resid_logit_check']['GC']['pearson_r']:+.4f} | "
              f"{m0['resid_logit_check']['log10_len']['pearson_r']:+.4f} | "
              f"{m0['residual__GC']['pearson_r_contig']:+.4f} | "
              f"{m0['residual__log10_len']['pearson_r_contig']:+.4f} | "
              f"{m0['residual__phylum']['eta2']:.4f} | "
              f"{m0['shortcut_model']['oof_auroc']:.4f} |")
    for k, v in summary.items():
        md.append(f"| {v and k.upper()} terminal | {v['r_gc_logit']:+.4f} | {v['r_len_logit']:+.4f} | "
                  f"{v['r_gc_prob_contig']:+.4f} | {v['r_len_prob_contig']:+.4f} | "
                  f"{v['eta2_phylum']:.4f} | {v['shortcut_oof_auroc']:.4f} |")
    md += ["", "* logit = primary protocol (prespecified interpretation band |r| ≤ 0.10); prob-contig = disclosure protocol.",
           "generate: scripts/residual_corr_terminal.py · data: val_window_probs_final.npz"]
    (REPO / "docs/evidence/residual_corr_terminal_report.md").write_text(
        "\n".join(md) + "\n")
    print("[report] docs/evidence/residual_corr_terminal_report.md")


if __name__ == "__main__":
    main()
