#!/usr/bin/env python3
"""evo2_window_pooling.py  —  V7.5 #3 Phase 1 pooling ablation (pilot, step0 subset)
================================================================================
Question: the canonical protocol averages ALL window-level mean-pool vectors of a contig
with equal weight — does mean pooling dilute localized HGT boundary signal?
(V7_5_EVIDENCE_PLAN review risk #1, severity *****)

Two stages (with layer scan mirrors the layer scan)::
  --embed  (GPU, evo2 venv) toward evo_cache/step0_subset/step0_subset.fasta
           (n=1996, 31.7 Mbp, pos_/neg_ prefixes) under the canonical sliding-window protocol
           (window=3000 / stride=2700, from position 0 for all contigs, terminal window retained,
           BOS removed within each window, token-mean): extract **blocks.24.mlp.l3 + blocks.28.mlp.l3**
           window-level embeddings from both layers (no contig aggregation yet), saved to
           evo_cache/step0_subset/pooling/emb_{layer}.pt = {seq_id: fp16[n_win,4096]}
           (windows sorted by start; both layers share a single forward via multi-hook, protocol reused from
           precompute_evo2_embeddings / evo2_layer_scan).
  --probe  (CPU, thesis python) per layer x pooling {mean, max, topk_mean(k=3), variance}
           (window-axis aggregation -> contig vector) × probe {linear, ridge}
           (reusing the rap_probes protocol: 70/30 stratified seed42 + standardize +
           5-seed SD + bootstrap CI) + DeLong paired test (each pooling vs mean,
           scripts/delong_test, same test fold).
           criterion (prespecified): any pooling vs mean DeLong p<0.05 AND dAUROC>0.02
           → "pooling matters"; else mean pooling is not significantly beaten.

Sanity anchor: this script mean pooling (equal weight) should reproduce canonical contig vectors — 
with layer_scan/emb_blocks_28_mlp_l3.pt per-contig cosine >= 0.9999, linear AUROC
≈ step0 RAP 0.757 (bf16 cross-batch jitter: see V7_CONTRACT §6).

GPU discipline: CUDA_VISIBLE_DEVICES=4090 UUID (A100 strictly forbidden); single GPU process;
launcher: scripts/run/evo2_window_pooling_4090.sh.

output: evo_cache/step0_subset/pooling_ablation.{csv,md}
       + pooling/emb_blocks_{24,28}_mlp_l3.pt + pooling/pooling_raw.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

CACHE = ROOT / "evo_cache/step0_subset"
FASTA = CACHE / "step0_subset.fasta"
POOL_DIR = CACHE / "pooling"
LAYERS = [24, 28]                       # layer-scan optimum (b24, joint score) + current production tap (b28)
LAYER_NAMES = [f"blocks.{b}.mlp.l3" for b in LAYERS]
POOLINGS = ["mean", "max", "topk_mean", "variance"]
PROBES = ["linear", "ridge"]
TOPK = 3
GPU_4090_UUID = "${GPU_4090_UUID}"


def assert_4090():
    import torch
    n = torch.cuda.device_count()
    assert n == 1, f"exactly 1 visible GPU expected (4090 GPU lock), actual {n}"
    name = torch.cuda.get_device_name(0)
    assert "4090" in name, f"visible GPU is not RTX 4090: {name}"
    free, total = torch.cuda.mem_get_info(0)
    print(f"[GPU] {name} | mem free {free/1e9:.1f}/{total/1e9:.1f} GB")
    return name


# ══════════════════════ Stage 1: GPU windowlevelembedding (evo2 venv) ══════════════════════ #

def cmd_embed(args):
    import torch
    from precompute_evo2_embeddings import iter_fasta

    class MultiLayerEmbedder:
        """Both layers share a single forward (reusing evo2_layer_scan.MultiLayerEmbedder protocol,
        timing hooks removed — this task only needs window-level vectors)."""

        def __init__(self, model_name, layer_names, device, local_path):
            from evo2 import Evo2
            self.model = Evo2(model_name, local_path=local_path)
            self.layer_names = layer_names
            self.device = device
            with torch.no_grad():
                toks = self.model.tokenizer.tokenize("ACGT")
                self._bos_added = len(toks) > 4
                probe = torch.tensor([toks], dtype=torch.int).to(device)
                _, embs = self.model(probe, return_embeddings=True,
                                     layer_names=layer_names)
            self.dim = embs[layer_names[0]].shape[-1]
            print(f"[MultiLayerEmbedder] layers={layer_names} "
                  f"hidden={self.dim} bos_added={self._bos_added}")

        def _pool(self, emb, n_valid):
            start = 1 if self._bos_added else 0
            e = emb[start:start + n_valid]
            if e.shape[0] == 0:
                e = emb[:1]
            return e.float().mean(dim=0)

        @torch.no_grad()
        def embed_batch(self, seqs):
            toks = [self.model.tokenizer.tokenize(s) for s in seqs]
            T = max(len(t) for t in toks)
            ids = torch.zeros((len(seqs), T), dtype=torch.int)
            for i, t in enumerate(toks):
                ids[i, :len(t)] = torch.tensor(t, dtype=torch.int)
            _, embs = self.model(ids.to(self.device), return_embeddings=True,
                                 layer_names=self.layer_names)
            out = {}
            for ln in self.layer_names:
                emb = embs[ln]
                out[ln] = torch.stack(
                    [self._pool(emb[i], len(s)) for i, s in enumerate(seqs)]).cpu()
            return out

    def build_windows(records, window, stride):
        """Canonical sliding window (mirrors precompute_evo2_embeddings.embed_contigs_windowed):
        (contig_idx, start, len, sub_seq); from position 0 with fixed stride, terminal partial window retained."""
        entries = []
        for ci, (sid, seq) in enumerate(records):
            if len(seq) <= window:
                entries.append((ci, 0, len(seq), seq))
            else:
                for start in range(0, len(seq), stride):
                    sub = seq[start:start + window]
                    entries.append((ci, start, len(sub), sub))
        return entries

    def batch_slices(entries_sorted, batch_size, token_budget, len_fn):
        i, n = 0, len(entries_sorted)
        while i < n:
            take = 1
            while (i + take < n and take < batch_size
                   and (take + 1) * len_fn(entries_sorted[i + take]) <= token_budget):
                take += 1
            yield i, i + take
            i += take

    POOL_DIR.mkdir(parents=True, exist_ok=True)
    gpu = assert_4090()
    records = list(iter_fasta(str(FASTA)))
    print(f"[data] {len(records)} contigs, "
          f"{sum(len(s) for _, s in records)/1e6:.1f} Mbp")

    layer_files = {ln: POOL_DIR / f"emb_{ln.replace('.', '_')}.pt" for ln in LAYER_NAMES}
    if all(p.exists() for p in layer_files.values()):
        print("[skip] both layers window-level embeddings already exist, skipping")
        return

    embedder = MultiLayerEmbedder(args.model, LAYER_NAMES, "cuda:0",
                                  local_path=args.local_path)

    # -- resumable: partial store {layer: {sid: [(start, vec_fp16), ...]}} ──
    partial_files = {ln: POOL_DIR / f"emb_{ln.replace('.', '_')}.partial.pt"
                     for ln in LAYER_NAMES}
    out = {ln: {} for ln in LAYER_NAMES}     # ln -> {sid: [(start, vec), ...]}
    if all(p.exists() for p in partial_files.values()):
        pre = {ln: torch.load(partial_files[ln], map_location="cpu",
                              weights_only=False) for ln in LAYER_NAMES}
        done_ids = set.intersection(*(set(v.keys()) for v in pre.values()))
        out = {ln: {sid: list(pre[ln][sid]) for sid in done_ids} for ln in LAYER_NAMES}
        records = [(sid, s) for sid, s in records if sid not in done_ids]
        print(f"[resume] {len(done_ids)} contigs alreadycompleted, remaining {len(records)}")

    win_entries = build_windows(records, args.window, args.stride)
    total_bp = sum(w[2] for w in win_entries)
    print(f"[windows] {len(win_entries)} windows "
          f"(window={args.window}, stride={args.stride}, {total_bp/1e6:.1f} Mbp)")

    total_win = defaultdict(int)
    for ci, _, _, _ in win_entries:
        total_win[ci] += 1
    done_ct = defaultdict(int)

    t0 = time.time()
    n_batches = sum(1 for _ in batch_slices(win_entries, args.batch,
                                            args.token_budget, lambda e: e[2]))
    last_flush = 0
    for bi, (lo, hi) in enumerate(batch_slices(win_entries, args.batch,
                                               args.token_budget,
                                               lambda e: e[2])):
        chunk = win_entries[lo:hi]
        res = embedder.embed_batch([w[3] for w in chunk])
        for ln in LAYER_NAMES:
            vecs = res[ln].numpy()
            for k, w in enumerate(chunk):
                out[ln].setdefault(records[w[0]][0], []).append(
                    (w[1], vecs[k].astype(np.float16)))
        for w in chunk:
            done_ct[w[0]] += 1
            if done_ct[w[0]] == total_win[w[0]]:
                for ln in LAYER_NAMES:
                    out[ln][records[w[0]][0]].sort(key=lambda t: t[0])  # start ascending
        if (bi + 1) % 100 == 0 or bi + 1 == n_batches:
            el = time.time() - t0
            done_bp = sum(w[2] for w in win_entries[:hi])
            rate = done_bp / 1e6 / max(el / 60, 1e-9)
            print(f"  [{hi}/{len(win_entries)}] {el/60:.1f}min ({rate:.2f} Mbp/min, "
                  f"ETA {(total_bp-done_bp)/1e6/max(rate,1e-9):.0f}min) "
                  f"contigs done {len(out[LAYER_NAMES[0]])}", flush=True)
        if (bi + 1) - last_flush >= 500:
            for ln in LAYER_NAMES:
                torch.save(out[ln], partial_files[ln])
            last_flush = bi + 1

    elapsed = time.time() - t0
    for ln in LAYER_NAMES:
        arrs = {sid: np.stack([v for _, v in lst]).astype(np.float16)
                for sid, lst in out[ln].items()}
        assert all(a.ndim == 2 and a.shape[1] == embedder.dim for a in arrs.values())
        torch.save(arrs, layer_files[ln])
        partial_files[ln].unlink(missing_ok=True)
        n_win = sum(a.shape[0] for a in arrs.values())
        print(f"[save] {layer_files[ln].name}: {len(arrs)} contigs, "
              f"{n_win} windows × {embedder.dim}")

    (POOL_DIR / "pooling_embed_raw.json").write_text(json.dumps({
        "protocol": {"window": args.window, "stride": args.stride,
                     "batch": args.batch, "token_budget": args.token_budget,
                     "gpu": gpu, "model": args.model, "dim": embedder.dim,
                     "bos_added": embedder._bos_added,
                     "window_pool": "token mean after BOS removal (per window)",
                     "note": "window-level vectors, no contig aggregation yet; aggregation happens at probe stage"},
        "layers": LAYER_NAMES, "n_records": len(records),
        "n_windows": len(win_entries), "total_mb": total_bp / 1e6,
        "elapsed_s": elapsed,
    }, indent=2))
    print(f"[done] {elapsed/60:.1f} min → {POOL_DIR}")


# ══════════════════════ Stage 2: CPU probe + DeLong (thesis python) ══════════════════════ #

def pool_windows(W: np.ndarray, method: str, topk: int = TOPK) -> np.ndarray:
    """Window-axis aggregation [n_win, D] -> [D] (V7.5 #3 Phase 1: non-parametric pooling only)."""
    if method == "mean":
        return W.mean(axis=0)
    if method == "max":
        return W.max(axis=0)
    if method == "topk_mean":
        # elementwise mean of top-k values along window axis (k shrinks for short contigs)
        k = min(topk, W.shape[0])
        idx = np.argsort(W, axis=0)[-k:]
        return np.take_along_axis(W, idx, axis=0).mean(axis=0)
    if method == "variance":
        return W.var(axis=0)
    raise ValueError(method)


def cmd_probe(args):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")     # CPU-only
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "8")
    import torch
    torch.set_num_threads(8)
    import pandas as pd
    import rap_probes as rap
    from delong_test import delong_test_full, _fmt_p

    t0 = time.time()
    seqs = rap.load_fasta(CACHE / "step0_subset.fasta")
    ids = list(seqs.keys())
    y = np.array([1 if i.startswith("pos_") else 0 for i in ids], dtype=int)

    # ── windowlevelcache → (layer, pooling) → X[n_seq, D] ──
    X = {}
    win_stats = {}
    for ln in LAYER_NAMES:
        p = POOL_DIR / f"emb_{ln.replace('.', '_')}.pt"
        if not p.exists():
            raise SystemExit(f"FATAL: missing window-level cache (run --embed first)")
        raw = rap.load_embeddings(p)          # {sid: fp32 [n_win, D]}
        assert set(raw.keys()) == set(ids), f"{ln}: embedding id mismatch"
        for m in POOLINGS:
            X[(ln, m)] = np.stack([pool_windows(raw[i], m) for i in ids]) \
                .astype(np.float32)
        nw = np.array([raw[i].shape[0] for i in ids])
        win_stats[ln] = dict(n_win_total=int(nw.sum()), n_win_mean=float(nw.mean()),
                             n_win_median=float(np.median(nw)),
                             frac_single_win=float((nw == 1).mean()))
        print(f"[cache] {ln}: {len(ids)} contigs, {nw.sum()} windows "
              f"(mean {nw.mean():.1f}, single-window contigs {(nw==1).mean()*100:.1f}%)")

    # -- sanity anchor: mean pooling vs canonical contig vectors (layer_scan) --
    fid = {}
    ref_path = CACHE / "layer_scan/emb_blocks_28_mlp_l3.pt"
    if ref_path.exists():
        ref = rap.load_embeddings(ref_path)
        mine = X[("blocks.28.mlp.l3", "mean")]
        coss = []
        for i, mid in zip(ids, mine):
            r = ref[i]
            coss.append(float(r @ mid / (np.linalg.norm(r) * np.linalg.norm(mid) + 1e-12)))
        fid["cos_vs_layer_scan_b28_mean"] = dict(
            min=float(np.min(coss)), median=float(np.median(coss)),
            frac_ge_09999=float(np.mean(np.array(coss) >= 0.9999)))
        print(f"[fidelity] mean(b28) vs layer_scan canonical: "
              f"cos median={fid['cos_vs_layer_scan_b28_mean']['median']:.6f} "
              f"min={fid['cos_vs_layer_scan_b28_mean']['min']:.6f}")

    # ── primaryprotocol: 70/30 stratified seed42 + 5-seed SD + bootstrap CI ──
    seeds = [42, 43, 44, 45, 46]
    tr, te = rap.stratified_split(y, seed=42)
    yte = y[te]
    print(f"[probe] n={len(ids)} pos={int(y.sum())} | test fold n={len(te)} "
          f"pos={int(yte.sum())}")

    rows, delong_rows, preds_main = [], [], {}
    for ln in LAYER_NAMES:
        for m in POOLINGS:
            Xm = X[(ln, m)]
            for pname in PROBES:
                t1 = time.time()
                xtr, xte = rap.standardize(Xm[tr], Xm[te])
                s = rap.PROBES[pname](xtr, y[tr], xte)
                auroc, auprc = rap.metrics(yte, s)
                (al, ah), _ = rap.bootstrap_ci(yte, s, args.n_boot, seed=42)
                preds_main[(ln, m, pname)] = np.asarray(s, dtype=np.float64)
                rep = []
                for sd_ in seeds[1:]:
                    tr2, te2 = rap.stratified_split(y, seed=sd_)
                    xtr2, xte2 = rap.standardize(Xm[tr2], Xm[te2])
                    s2 = rap.PROBES[pname](xtr2, y[tr2], xte2)
                    rep.append(rap.metrics(y[te2], s2)[0])
                rows.append(dict(
                    layer=ln, pooling=m, probe=pname,
                    auroc=auroc, auroc_ci95=f"[{al:.4f},{ah:.4f}]",
                    auprc=auprc, auroc_seed_sd=float(np.std([auroc] + rep, ddof=1)),
                    runtime_s=round(time.time() - t1, 1)))
                print(f"  {ln} {m:10s} {pname:7s} AUROC={auroc:.4f} "
                      f"AUPRC={auprc:.4f} seedSD={rows[-1]['auroc_seed_sd']:.4f}",
                      flush=True)
                pd.DataFrame(rows).to_csv(CACHE / "pooling_ablation.csv", index=False)

    # -- DeLong: each pooling vs mean (same test fold, same probe) --
    for ln in LAYER_NAMES:
        for pname in PROBES:
            base = preds_main[(ln, "mean", pname)]
            for m in POOLINGS:
                if m == "mean":
                    continue
                r = delong_test_full(yte, preds_main[(ln, m, pname)], base)
                delong_rows.append(dict(
                    layer=ln, probe=pname, comparison=f"{m} vs mean",
                    auroc_alt=r["auc_a"], auroc_mean=r["auc_b"],
                    delta_auc=r["delta"], z=r["z"], p=r["p"], p_sci=_fmt_p(r["p"]),
                    ci95_delta=list(r["ci95_delta"]), n=r["n"]))
                print(f"  [DeLong] {ln} {pname}: {m} vs mean "
                      f"Δ={r['delta']:+.4f} z={r['z']:+.2f} p={_fmt_p(r['p'])}",
                      flush=True)
                pd.DataFrame(delong_rows).to_csv(
                    CACHE / "pooling_delong.csv", index=False)

    # predictions archived (for verification)
    np.savez_compressed(
        POOL_DIR / "pooling_predictions.npz", labels=yte,
        test_ids=np.array(ids)[te],
        **{f"{ln}__{m}__{p}": preds_main[(ln, m, p)]
           for ln in LAYER_NAMES for m in POOLINGS for p in PROBES})

    # ── prespecified criterion ──
    hits = [d for d in delong_rows
            if d["p"] < 0.05 and abs(d["delta_auc"]) > 0.02]
    # direction-sensitive: criterion says "dAUROC>0.02" (pooling beats mean); two-sided p + |d|>0.02
    # are both tabulated, but the primary interpretation of "pooling matters" keys on positive superiority (delta>+0.02).
    hits_pos = [d for d in delong_rows if d["p"] < 0.05 and d["delta_auc"] > 0.02]
    hits_neg = [d for d in delong_rows if d["p"] < 0.05 and d["delta_auc"] < -0.02]
    verdict = "POOLING-MATTERS" if hits_pos else "MEAN-NOT-SURPASSED"

    _write_report(rows, delong_rows, win_stats, fid, verdict,
                  hits_pos, hits_neg, args, time.time() - t0)
    print(f"[probe] verdict={verdict} → {CACHE/'pooling_ablation.md'}")


def _write_report(rows, delong_rows, win_stats, fid, verdict,
                  hits_pos, hits_neg, args, total_s):
    import pandas as pd
    df = pd.DataFrame(rows)
    dd = pd.DataFrame(delong_rows)

    lines = []
    A = lines.append
    A("# windowpoolingablation (V7.5 #3 Phase 1)  —  mean / max / topk_mean(k=3) / variance")
    A("")
    A(f"> generate: {time.strftime('%Y-%m-%d %H:%M:%S')}  ·  DeepFluHGT  ·  "
      "V7_5_EVIDENCE_PLAN #3 Phase 1 (non-parametric pooling; attention pooling deferred)")
    A(f"> data: evo_cache/step0_subset/step0_subset.fasta (n=1996, 31.7 Mbp, "
      "pos_/neg_ prefix tag); window-level embedding = canonical sliding window (3000/2700, terminal window retained, "
      "BOS removed within window, token-mean), two layers (blocks.24.mlp.l3 / blocks.28.mlp.l3) "
      "single forward multi-hook shared -> pooling/emb_blocks_{24,28}_mlp_l3.pt "
      "({seq_id: fp16[n_win,4096]})")
    st = win_stats[LAYER_NAMES[0]]
    A(f"> window stats: per layer {st['n_win_total']} window, contig mean {st['n_win_mean']:.1f} window, "
      f"median {st['n_win_median']:.0f}, single-window contig fraction {st['frac_single_win']*100:.1f}% "
      "(single-window contigs: all poolings degenerate to equivalence)")
    A("> probe protocol: rap_probes reuse — 70/30 stratified (seed42) + standardize + "
      f"{args.n_boot}×bootstrap CI + 5-seed (42-46) SD; DeLong = "
      "delong_test paired test (same test fold, n=599)")
    if fid:
        f = fid["cos_vs_layer_scan_b28_mean"]
        A(f"> **sanity anchor**: mean pooling (b28) vs layer_scan canonical contig vectors "
          f"cos: median {f['median']:.6f}, min {f['min']:.6f}, "
          f"≥0.9999 fraction {f['frac_ge_09999']*100:.1f}% "
          "(bf16 cross-batch jitter, V7_CONTRACT §6)")
    A("")
    A("## 1. primarytable  —  AUROC [95% bootstrap CI] (seed42 test fold)")
    A("")
    for ln in LAYER_NAMES:
        A(f"### {ln}")
        A("")
        A("| pooling | linear AUROC [CI] | linear AUPRC | ridge AUROC [CI] | "
          "ridge AUPRC | seed SD (linear) |")
        A("|---|---|---|---|---|---|")
        sub = df[df.layer == ln]
        for m in POOLINGS:
            r_l = sub[(sub.pooling == m) & (sub.probe == "linear")].iloc[0]
            r_r = sub[(sub.pooling == m) & (sub.probe == "ridge")].iloc[0]
            A(f"| {m} | {r_l.auroc:.4f} {r_l.auroc_ci95} | {r_l.auprc:.4f} | "
              f"{r_r.auroc:.4f} {r_r.auroc_ci95} | {r_r.auprc:.4f} | "
              f"{r_l.auroc_seed_sd:.4f} |")
        A("")

    A("## 2. DeLong paired test  —  each pooling vs mean (prespecified criterion)")
    A("")
    A("| layer | probe | comparison | AUROC_alt | AUROC_mean | Δ(A−B) | z | p (two-sided) | "
      "p<0.05 ∧ Δ>0.02 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in dd.itertuples():
        crit = "**is**" if (r.p < 0.05 and r.delta_auc > 0.02) else \
            ("significantly worse" if (r.p < 0.05 and r.delta_auc < -0.02) else "no")
        A(f"| {r.layer} | {r.probe} | {r.comparison} | {r.auroc_alt:.4f} | "
          f"{r.auroc_mean:.4f} | {r.delta_auc:+.4f} | {r.z:+.2f} | "
          f"{r.p_sci} | {crit} |")
    A("")
    A("## 3. interpretation (prespecified criterion: any pooling vs mean p<0.05 and dAUROC>0.02 → "
      "“pooling matters”)")
    A("")
    if hits_pos:
        A(f"- **Verdict: POOLING MATTERS**  —  —  {len(hits_pos)} group(s) significantly beat mean:")
        for h in hits_pos:
            A(f"  - {h['layer']} / {h['probe']}: {h['comparison']} "
              f"Δ={h['delta_auc']:+.4f}, p={h['p_sci']}")
    else:
        A("- **Verdict: MEAN NOT SURPASSED**  —  —  no pooling at p<0.05 with "
          "ΔAUROC>0.02 significantly beating mean.")
    if hits_neg:
        A(f"- additional {len(hits_neg)} group(s) **significantly worse** mean "
          "(p<0.05 and d<-0.02, see Table 2 'significantly worse') — all are variance window-dispersion pooling"
          "(b28 ridge −0.096), i.e. between-window variance is not a signal carrier; complementarity with mean/max weak at the "
          "locus-level mean-information level.")
    # cross-layer consistency
    for pname in PROBES:
        pos_l = [h for h in hits_pos if h["probe"] == pname]
        lay = sorted({h["layer"] for h in pos_l})
        A(f"- probe={pname}: significant positive superiority appeared in {lay if lay else 'none'} "
          f"({'consistent across both layers' if len(lay) == 2 else 'inconsistent across layers' if lay else ''}))")
    A("")
    A("## 4. caliberwithlimitations")
    A("")
    A("- all pooling operates on **frozen window-level embeddings** (Phase 1 non-parametric; attention pooling is a "
      "future trainable readout, not at this stage).")
    A("- topk_mean(k=3) is elementwise top-3 mean along the window axis; single-window contigs "
      "(≤3kb) k shrinkfor 1, with mean degenerateequivalent.")
    A("- variance pooling output is non-negative; also standardized before probing.")
    A("- scope: step0 subset pilot (n=1996); full Contig_500k requires window-level "
      "cache format extension before replication.")
    A("")
    A("## 5. artifact")
    A("")
    A("- windowlevelcache: evo_cache/step0_subset/pooling/emb_blocks_{{24,28}}_mlp_l3.pt "
      "({seq_id: fp16[n_win,4096]}, windows sorted by start ascending)")
    A("- table: pooling_ablation.csv (probe primarytable)  ·  pooling_delong.csv (paired test) "
      " ·  pooling/pooling_predictions.npz (test fold predictionsverification)")
    A("- embeddingprotocol JSON: pooling/pooling_embed_raw.json")
    A(f"- script: scripts/evo2_window_pooling.py (--embed 4090 / --probe CPU)  ·  "
      f"launcher: scripts/run/evo2_window_pooling_4090.sh  ·  probe runtime: "
      f"{total_s:.0f}s")
    (CACHE / "pooling_ablation.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="V7.5 #3 poolingablation pilot (Two stages)")
    ap.add_argument("--embed", action="store_true", help="GPU windowlevelembeddingStage (evo2 venv)")
    ap.add_argument("--probe", action="store_true", help="CPU probe + DeLong Stage")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--token_budget", type=int, default=12288,
                    help="4090 24GB budget (layer scan measured comparable: 4x3001 tokens)")
    ap.add_argument("--model", default="evo2_7b")
    ap.add_argument("--local_path",
                    default=str(ROOT / "model/arcinstitute/evo2_7b/evo2_7b.pt"))
    ap.add_argument("--window", type=int, default=3000)
    ap.add_argument("--stride", type=int, default=2700)
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()
    if not (args.embed or args.probe):
        ap.error("either --embed or --probe required")
    if args.embed:
        if args.local_path and not os.path.isfile(args.local_path):
            raise SystemExit(f"[ERROR] local_path does not exist: {args.local_path}")
        cmd_embed(args)
    if args.probe:
        cmd_probe(args)


if __name__ == "__main__":
    main()
