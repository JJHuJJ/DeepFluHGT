#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""F1 preview — NTv3-pre CPU parallel pooling (M2 preview) — DeepFluHGT

Goal: sample contigs from the full evo train cache (400k) and run frozen NTv3-pre CPU forward passes to
produce ntv3_pooled_1536 (**same window protocol as the evo cache**: 3000bp window / 2700bp
stride, terminal window retained, per-window mean-pool then contig-level unweighted average — mirrors
the canonical protocol of scripts/precompute_evo2_embeddings.py), then runs
all comparison arms of scripts/fusion_vs_ensemble.py (evo / ntv3 / concat+RF /
concat+LR / G1 gated / G2 stacking, bootstrap+DeLong).

parallel architecture (measured choice, 2026-08-27, EPYC 7713, NTv3 CPU fp32 forward):
  plan A: single process, torch 48 threads: L=3072 → 2.06 seq/s; L=256 → 19.0 seq/s
  plan B: 4 processes x 12 threads: L=3072 → 3.63 seq/s; L=256 → 38.4 seq/s ✓
  (plan C, 48 processes each loading the model = 48x2.5GB memory blowup, rejected — anticipated in the task brief)
  → adopt B: 4 workers each load NTv3-pre once (2.5GB/process, 10GB total), pooling forward passes;
  windows bucketed by token length → homogeneous-length batches → long tasks dispatched first (dynamic load balancing).

sampling and feasibility gate (task-brief protocol):
  target n=30,000 contig (seed42, with evo train.pt keyalignment);
  measured throughput below: if the total-window forward ETA for 30k exceeds 10h → **fall back to 10,000** and note it in the report.
  (400k contig length distribution med≈4.7kb / mean≈16kb → 30k ≈ 190k windows ≈ 15h, over budget;
   10k ≈ 61k windows ≈ 4-6h, feasible — the actual choice is decided automatically by this script's gate and recorded.)

output:
  evo_cache/Contig_500k/ntv3_pooled_train_sample.npz (ids/labels/
    pooled_1536/length/gc/manifest)
  evo_cache/Contig_500k/rap_full/fusion_preview_probs.npz (test fold probability)
  docs/evidence/F1_FUSION_PREVIEW.md
  docs/evidence/F1_FUSION_PREVIEW.EXIT

usage (nohup, ≤48 thread):
  nohup python \
      scripts/ntv3_pooled_parallel.py > logs/f1_ntv3_pool.log 2>&1 &
  # re-run fusion only (the embedding npz already exists):
  python scripts/ntv3_pooled_parallel.py --fusion-only

runtime discipline: CPU-only (CUDA_VISIBLE_DEVICES=""), never touches the GPU.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "") # CPU-only

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

EVO_TRAIN = REPO / "evo_cache" / "Contig_500k" / "train.pt"
TRAIN_FASTA = REPO / "simulated_data" / "Contig_500k" / "train" / "dataset.fasta"
NT_PATH = REPO / "model" / "InstaDeepAI" / "NTv3_650M_pre"
OUT_NPZ = REPO / "evo_cache" / "Contig_500k" / "ntv3_pooled_train_sample.npz"
OUT_MD = REPO / "docs" / "evidence" / "F1_FUSION_PREVIEW.md"
EXIT_MARK = REPO / "docs" / "evidence" / "F1_FUSION_PREVIEW.EXIT"
PROBS_NPZ = REPO / "evo_cache" / "Contig_500k" / "rap_full" / \
    "fusion_preview_probs.npz"

WIN, STRIDE = 3000, 2700 # evo cache canonical protocol
TARGET_N = 30_000
FALLBACK_N = 10_000
ETA_BUDGET_H = 10.0
MEAS_RATE_L3072 = 3.63  # seq/s @4x12thr (measured; drives the gate decision)
N_PROC, N_THREADS = 4, 12


# ====================== worker (spawn) ====================== #

_W_STATE: dict = {}


def _worker_init(nt_path: str, threads: int):
    import torch
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(threads)
    torch.set_num_threads(threads)
    from model.impl.backbone import GlobalEncoder
    from data_pipeline.impl.tokenizer import SingleNucleotideTokenizer
    enc = GlobalEncoder(pretrained_path=nt_path, use_flash_attention=False)
    enc.eval()
    assert not enc._use_fallback, "NTv3-pre loadfailure (fallback)"
    _W_STATE["enc"] = enc
    _W_STATE["tok"] = SingleNucleotideTokenizer()
    _W_STATE["pid"] = os.getpid()


def _batch_fwd(token_lists: list[list[int]]):
    """one batch of same-length-tier windows → pooled 1536 (masked mean-pool, mirrors
    embed_ntv3_cpu / GlobalEncoder._forward_nt step by step)."""
    import torch
    enc, tok = _W_STATE["enc"], _W_STATE["tok"]
    lb = max(len(t) for t in token_lists)
    ids_t = torch.zeros((len(token_lists), lb), dtype=torch.long) # PAD=0
    for r, t in enumerate(token_lists):
        ids_t[r, : len(t)] = torch.tensor(t, dtype=torch.long)
    mapped = enc._remap_ids(ids_t) # canonical → NTv3
    attn = (ids_t != 0).long()
    B, Lb = mapped.shape
    if Lb % 128 != 0:  # conv tower padding
        pad_len = 128 - (Lb % 128)
        mapped = torch.cat(
            [mapped, torch.full((B, pad_len), 1, dtype=mapped.dtype)], 1)
        attn = torch.cat(
            [attn, torch.zeros((B, pad_len), dtype=attn.dtype)], 1)
    with torch.no_grad():
        out = enc._nt_model(input_ids=mapped, attention_mask=attn,
                            output_hidden_states=True)
        h = out.hidden_states[-1][:, :Lb, :].float() # [B, L, 1536]
        mask = (attn[:, :Lb] != 0).unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
    return pooled.numpy().astype(np.float16)


def _embed_task(task):
    """task = (contig_ids, seqs) → [(contig_id, pooled_fp16[1536]), ...]"""
    enc_ok = "enc" in _W_STATE
    assert enc_ok
    tok = _W_STATE["tok"]
    out = []
    for cid, seq in zip(task[0], task[1]):
        # — evo cache canonical windowprotocol: starts = range(0, len, 2700),
        # sub = seq[s:s+3000] (terminal window retained); eachwindow mean-pool afteretc.weighted average — 
        subs = [seq[s: s + WIN] for s in range(0, len(seq), STRIDE)]
        if not subs:
            subs = [seq]
        toks = [tok.tokenize(s) for s in subs]
        # length bucketing (full windows vs tail window); same bucket batches together
        by_len = {}
        for t in toks:
            by_len.setdefault(len(t), []).append(t)
        vecs = []
        for _, group in sorted(by_len.items()):
            for b0 in range(0, len(group), 8):
                vecs.extend(_batch_fwd(group[b0: b0 + 8]))
        out.append((cid, np.mean(np.stack(vecs), axis=0).astype(np.float16)))
    return out


# ====================== main pipeline (phase A: poolingembedding) ====================== #

def load_sample(n_target: int):
    """seed42 sampling + fasta sequence load (only the sampled sequences are resident)."""
    import torch
    t0 = time.time()
    emb = torch.load(EVO_TRAIN, map_location="cpu", weights_only=False)
    keys = sorted(emb.keys())
    rng = np.random.default_rng(42)
    sel = np.sort(rng.choice(len(keys), min(n_target, len(keys)),
                             replace=False))
    ids = [keys[i] for i in sel]
    evo_fp16 = {i: emb[i] for i in ids}  # keep only the sample (82MB @10k)
    del emb
    want = set(ids)
    seqs = {}
    name, chunks = None, []
    with open(TRAIN_FASTA) as fh:
        for line in fh:
            if line.startswith(">"):
                if name in want:
                    seqs[name] = "".join(chunks)
                name = line[1:].strip().split()[0]; chunks = []
            else:
                chunks.append(line.strip().upper())
    if name in want:
        seqs[name] = "".join(chunks)
    assert len(seqs) == len(ids), f"fasta missingsampling id: {len(seqs)}/{len(ids)}"
    print(f"[sample] n={len(ids)} seed42 ({time.time()-t0:.0f}s)", flush=True)
    return ids, evo_fp16, seqs


def n_windows(L: int) -> int:
    return max(1, len(range(0, L, STRIDE)))


def stage_embed(args):
    import torch
    t_start = time.time()
    ids, evo_fp16, seqs = load_sample(args.n)
    lens = np.array([len(seqs[i]) for i in ids])
    tot_win = int(sum(n_windows(L) for L in lens))
    eta_h = tot_win / MEAS_RATE_L3072 / 3600
    n_target_used = args.n
    if args.n >= TARGET_N:
        note = (f"{args.n//1000}k window {tot_win:,} @ measured {MEAS_RATE_L3072} "
                f"seq/s → ETA {eta_h:.1f}h")
        if eta_h > ETA_BUDGET_H:
            print(f"[gate] {note} > {ETA_BUDGET_H}h budget → degrade {FALLBACK_N:,} "
                  f"(task-brief protocol, noted in the report)", flush=True)
            n_target_used = FALLBACK_N
            ids, evo_fp16, seqs = load_sample(FALLBACK_N)
            lens = np.array([len(seqs[i]) for i in ids])
            tot_win = int(sum(n_windows(L) for L in lens))
            note = (f"{TARGET_N//1000}k cannotrow (ETA {eta_h:.1f}h > "
                    f"{ETA_BUDGET_H}h) → degrade {FALLBACK_N:,}; actualwindow "
                    f"{tot_win:,}, conservative ETA {tot_win/MEAS_RATE_L3072/3600:.1f}h")
        else:
            print(f"[gate] {note} within budget, proceeding with {args.n}", flush=True)
    else:
        note = (f"smoke/custom n={args.n} (window {tot_win:,}, "
                f"ETA {eta_h:.1f}h) — gate skipped")
        print(f"[gate] {note}", flush=True)
    n = len(ids)

    # tasks sorted by total cost (Σ L²) and dispatched in descending order → long tasks run first (dynamic load balancing)
    tasks = sorted(
        ((ids[i], seqs[ids[i]]) for i in range(n)),
        key=lambda t: -(sum(min(WIN, len(t[1]) - s) ** 2
                            for s in range(0, len(t[1]), STRIDE))))
    task_args = [([t[0]], [t[1]]) for t in tasks]

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    results = {}
    t0 = time.time()
    done = 0
    with ctx.Pool(N_PROC, initializer=_worker_init,
                  initargs=(str(NT_PATH), N_THREADS)) as pool:
        for out in pool.imap_unordered(_embed_task, task_args, chunksize=1):
            for cid, vec in out:
                results[cid] = vec
            done += 1
            if done % 500 == 0 or done == n:
                el = time.time() - t0
                rate = done / el
                print(f" [embed] {done}/{n} contigs ({rate:.2f} contig/s, "
                      f"ETA {(n-done)/rate/60:.0f} min)", flush=True)

    el_embed = time.time() - t0
    pooled = np.stack([results[i] for i in ids]).astype(np.float16)
    y = np.array([1 if i.startswith("pos_") else 0 for i in ids], dtype=np.int8)
    gc = np.array([(seqs[i].count("G") + seqs[i].count("C")) /
                   max(1, sum(seqs[i].count(c) for c in "ACGT"))
                   for i in ids], dtype=np.float32)
    assert np.isfinite(pooled.astype(np.float32)).all()

    manifest = dict(
        n=n, target_n=n_target_used, fallback_n=FALLBACK_N, gate_note=note,
        seed=42, total_windows=tot_win,
        window_protocol=f"NTv3 windowmirror evo cache canonical: "
                        f"window={WIN}/stride={STRIDE}, terminal window retained, window mean-pool "
                        f"→ contig etc.weighted average",
        caliber="NTv3-pre hidden_states[-1] (full conv-tower resolution 1536d) over "
                "[CLS]+seq+[SEP] truereal token masked mean-pool (fp32 beforedirection, "
                "fp16 written to disk) — same embed_ntv3_cpu caliber",
        parallel=f"{N_PROC} proc x {N_THREADS} thr (measured 3.63 seq/s @L=3072 "
                 f"vs single process 48thr 2.06 — the faster one chosen)",
        achieved_contig_per_s=round(n / el_embed, 3),
        elapsed_s=round(el_embed, 1),
        script="scripts/ntv3_pooled_parallel.py",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
    np.savez_compressed(
        OUT_NPZ, ids=np.array(ids), labels=y, pooled_1536=pooled,
        length=lens.astype(np.float32), gc=gc,
        manifest=np.array(json.dumps(manifest)))
    print(f"[embed] done {n} contigs, {tot_win} windows, "
          f"{el_embed/60:.1f} min ({n/el_embed:.2f} contig/s) -> {OUT_NPZ.name}",
          flush=True)
    return manifest, el_embed


# ====================== phase B: fusion vs ensemble allarm ====================== #

def stage_fusion(args, manifest: dict, el_embed: float):
    import torch
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, str(args.fusion_threads))
    import fusion_vs_ensemble as fve
    fve.N_JOBS = args.fusion_jobs  # module constant defaults to 6 → raised to this task's quota
    from sklearn.model_selection import train_test_split

    t0 = time.time()
    z = np.load(OUT_NPZ, allow_pickle=False)
    ids = [str(s) for s in z["ids"]]
    nt = z["pooled_1536"].astype(np.float32)
    y = z["labels"].astype(int)

    emb = torch.load(EVO_TRAIN, map_location="cpu", weights_only=False)
    missing = [i for i in ids if i not in emb]
    assert not missing, f"evo cache missing {len(missing)} sampling id"
    evo = np.stack([emb[i] for i in ids]).astype(np.float32)
    del emb

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.30, stratify=y,
                              random_state=42)
    evo_tr, evo_te = evo[tr], evo[te]
    nt_tr, nt_te = nt[tr], nt[te]
    y_tr, y_te = y[tr], y[te]
    print(f"[fusion] n={len(y)} train={len(tr)} test={len(te)} "
          f"test_pos={int(y_te.sum())}", flush=True)

    probs, aux = fve.run_all_arms(evo_tr, evo_te, nt_tr, nt_te, y_tr, y_te,
                                  max_train=None, n_boot=args.n_boot)
    rows = fve.compare(probs, y_te, aux)
    el_f = time.time() - t0

    PROBS_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PROBS_NPZ, y_te=y_te, ids=np.array([ids[i] for i in te]),
                        **{k: v for k, v in probs.items()})
    _write_report(args, rows, aux, probs, y_te, manifest, len(y), len(tr),
                  len(te), el_embed, el_f)
    if args.n >= TARGET_N:
        EXIT_MARK.write_text(
            f"completed={time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"embed_s={el_embed:.0f} fusion_s={el_f:.0f}\n")
    print(f"[fusion] done -> {OUT_MD}", flush=True)


def _write_report(args, rows, aux, probs, y_te, manifest, n, n_tr, n_te,
                  el_embed, el_f):
    from sklearn.metrics import roc_auc_score
    L = []
    A = L.append
    A("# F1 — naive ensembling vs gated fusion (M2 preview, CPU NTv3 pooling)\n")
    A(f"- generate: {time.strftime('%Y-%m-%d %H:%M:%S')} · CPU only · "
      f"embedding {el_embed/60:.0f} min + fusion {el_f/60:.0f} min")
    A(f"- sample: evo train cache seed42 sampling n={n:,} "
      f"(gate: {manifest['gate_note']})")
    A(f"- ntv3_pooled_1536: {manifest['caliber']}")
    A(f"- evo: Evo2-7B frozen 4096d same cache; windowprotocolalignment "
      f"({manifest['window_protocol']})")
    A(f"- protocol: 70/30 stratified (seed42) train={n_tr:,}/test={n_te:,}; allarm = "
      "E1 evo+LR / E2 ntv3+LR / F1 concat+RF (Feng-style) / F2 concat+LR / "
      "G1 tanh(α) score gating / G2 score stacking; "
      f"bootstrap {args.n_boot} + paired DeLong (same test fold)")
    A(f"- parallel: {manifest['parallel']}; measured {manifest['achieved_contig_per_s']}"
      " contig/s; 4 proc x 12 thr (one NTv3-pre per process, 10GB total memory — "
      "the 48-process variant was rejected for memory)")
    A("- protocol note: the NTv3 features in this preview are **frozen NTv3-pre pooling** (not the M0 "
      "checkpoint's graph_pooled); G3 (M2 checkpoint full model) still awaits M2 training.\n")

    A("## 1. allarmresult (test fold)\n")
    A("| arm | AUROC | 95% CI | AUPRC |")
    A("|---|---|---|---|")
    for r in rows:
        if str(r["arm"]).startswith("DeLong"):
            continue
        lo, hi = r["auc_ci"]
        A(f"| {r['arm']} | {r['auc']:.4f} | [{lo:.4f},{hi:.4f}] | "
          f"{r['auprc']:.4f} |")
    g1 = aux.get("G1", {})
    if g1:
        A(f"\nG1 gating parameters: α={g1['alpha']:+.3f} (tanh={g1['tanh_alpha']:+.3f}; "
          f"train AUROC={g1['train_auc']:.4f}) — "
          f"{'evo injected with positive weight' if g1['tanh_alpha'] > 0 else 'evo suppressed'}\n")

    A("## 2. Key paired DeLong\n")
    A("| comparison | ΔAUROC | z | p | AUROC_a | AUROC_b |")
    A("|---|---|---|---|---|---|")
    for r in rows:
        if not str(r["arm"]).startswith("DeLong"):
            continue
        A(f"| {r['arm'].replace('DeLong: ', '')} | {r['delta']:+.4f} | "
          f"{r['z']:+.2f} | {r['p']:.3e} | {r['auc']:.4f} | "
          f"{r['auc_b']:.4f} |")

    aucs = {k: roc_auc_score(y_te, v) for k, v in probs.items()}
    A("\n## 3. Interpretation points (prespecified observations, not a final verdict)")
    A(f"- E2(ntv3)={aucs.get('E2_ntv3_logistic', float('nan')):.4f} vs "
      f"E1(evo)={aucs.get('E1_evo_logistic', float('nan')):.4f}: each FM's own "
      "accessibility lower bound.")
    A(f"- F1(concat+RF)={aucs.get('F1_concat_RF', float('nan')):.4f} − "
      f"max(E1,E2) = feature-level ensembling gain; G1={aucs.get('G1_gated_score', float('nan')):.4f} "
      "a 1-D simulated layer of the M2 gating mechanism.")
    A("- re-verified: `evo_cache/Contig_500k/rap_full/fusion_preview_probs.npz` "
      "(test fold allarmprobability) · embedding manifest see npz.\n")
    OUT_MD.write_text("\n".join(L))


def main() -> None:
    global OUT_NPZ, OUT_MD, PROBS_NPZ
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=TARGET_N,
                    help="sampled contig count (default 30000; <30000 treated as smoke, "
                         "gate skipped)")
    ap.add_argument("--fusion-only", action="store_true",
                    help="skip embedding (the npz already exists), run fusion only")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--fusion-threads", type=int, default=48)
    ap.add_argument("--fusion-jobs", type=int, default=48,
                    help="RF n_jobs (overrides fusion_vs_ensemble.N_JOBS)")
    args = ap.parse_args()
    if args.n < TARGET_N:  # smoke: redirect outputs, do not write official artifacts/EXIT
        OUT_NPZ = OUT_NPZ.with_name(OUT_NPZ.stem + "_smoke.npz")
        OUT_MD = OUT_MD.with_suffix(".smoke.md")
        PROBS_NPZ = PROBS_NPZ.with_name(PROBS_NPZ.stem + "_smoke.npz")
        print(f"[smoke] --n {args.n} in effect → {OUT_NPZ.name}", flush=True)
    if args.fusion_only:
        z = np.load(OUT_NPZ, allow_pickle=False)
        manifest = json.loads(str(z["manifest"]))
        stage_fusion(args, manifest, 0.0)
        return
    manifest, el_embed = stage_embed(args)
    stage_fusion(args, manifest, el_embed)


if __name__ == "__main__":
    main()
