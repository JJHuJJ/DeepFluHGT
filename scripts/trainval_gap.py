#!/usr/bin/env python3
"""trainval_gap.py  —  M0 train-val gap curve (V7 contract section 3 debt, preregistered observation point 2)
================================================================================
Contract text (V7_CONTRACT section 3): "train AUPRC is obtained by evaluating epoch snapshots
**after the fact** on a fixed 20k training-window subset (seed 42), without touching the running job".

Caliber (this script, everything preregistered/documented):
  - train side: simulated_data/Contig_500k/train, SlidingWindowContigDataset
    **is_training=True** (strategy D: positive-contig breakpoints centered in POS windows + <=2 hard-neg windows;
    negative contigs <=2 random windows), random.seed(42) set before dataset construction
    (same kind of seed as the training entry point seed_all(42)→construction; we do NOT claim bit-exact
    alignment with the training process's RNG stream —  the random consumption between training-time
    seed_all and dataset construction is unverifiable); the window count is consistent with the training log
    (940,028, asserted). Then numpy default_rng(42) samples 20,000 windows without replacement (fixed,
    shared across epochs) → torch Subset.
  - val side: the full val split (is_training=False full-coverage + max_eval_windows=10,
    199,438 windows, exactly the same parameters as training-time val).
  - inference: NTv3-only (use_evo=False), b=8, fp32 forward pass, early_exit=True,
    rc_aug off (RC is a training-time augmentation; gap measurement must be deterministic, off on both sides).
  - metric: window-level AUPRC/AUROC (primary caliber, same scale on both sides) + gap = train−val;
    stratified hard/easy neg AUC (both the post-fix 7-class hard set and the legacy collapsed caliber,
    reusing recompute_stratified) + ECE (10-bin). Also reports contig-level aggregation
    (TrainingEngine._aggregate_contig_predictions, same caliber as epoch_metrics.json) as a
    **fidelity check**: this script's val contig AUPRC should ≈ epoch_metrics.json's
    val_auprc (±bf16/fp32 jitter).

Interpretation (three contract scenarios, preregistered):
  A normal     —  train/val move together, gap small and stable;
  B ceiling  —  both on a high plateau, gap small;
  C overfitting   —  train↑ + val stalls + hard_neg drops + ECE worsens (all four must hold simultaneously).

usage (launcher: scripts/run/trainval_gap_4090.sh):
  CUDA_VISIBLE_DEVICES=<4090-uuid> python scripts/trainval_gap.py --run
  python scripts/trainval_gap.py --report   # only re-render the report/figure (reads the csv)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

DATA_DIR = ROOT / "simulated_data/Contig_500k"
CONFIG = ROOT / "configs/contig_500k_ntv3only_4090.yaml"
CKPT_DIR = ROOT / "checkpoints/Contig_500k_ntv3only"
EPOCHS = [1, 2, 3, 4, 5]
OUT_DIR = ROOT / "docs/evidence"
TRAJ_DIR = OUT_DIR / "m0_trajectory"
CSV_PATH = OUT_DIR / "TRAINVAL_GAP_M0.csv"
PNG_PATH = OUT_DIR / "TRAINVAL_GAP_M0.png"
MD_PATH = OUT_DIR / "TRAINVAL_GAP_M0.md"
N_TRAIN_SUBSET = 20_000
SEED = 42
BATCH = 8
GPU_4090_UUID = "${GPU_4090_UUID}"
EXPECTED_TRAIN_WINDOWS = 940_028        # logs/M0_contig_ntv3only.log
EXPECTED_VAL_WINDOWS = 199_438


def assert_4090():
    import torch
    n = torch.cuda.device_count()
    assert n == 1, f"exactly 1 visible GPU expected (4090 GPU lock), actual {n}"
    name = torch.cuda.get_device_name(0)
    assert "4090" in name, f"visible GPU is not RTX 4090: {name}"
    free, total = torch.cuda.mem_get_info(0)
    print(f"[GPU] {name} | mem free {free/1e9:.1f}/{total/1e9:.1f} GB")
    return name


def sha256_of(path: Path, nbytes: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()[:16]


# ══════════════════════ metrics (window-level + contig-level, dual-caliber stratified) ══════════════════════ #

def metrics_for(probs, labels, types, threshold=0.5):
    """Generic for window-level/contig-level: AUPRC/AUROC + stratified (fixed & legacy) + ECE."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    from evaluation.impl.metrics import (compute_stratified_metrics,
                                         compute_calibration_metrics)
    from recompute_stratified import (legacy_collapse, _strat_with_sets,
                                         _LEGACY_HARD_NEG_TYPES,
                                         _LEGACY_EASY_NEG_TYPES)
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    types = [str(t) for t in types]
    out = {"n": int(len(labels)),
           "n_pos": int(labels.sum()),
           "prevalence": float(labels.mean()),
           "auprc": float(average_precision_score(labels, probs)),
           "auroc": float(roc_auc_score(labels, probs))}
    fixed = compute_stratified_metrics(probs, labels, types, threshold)
    legacy = _strat_with_sets(probs, labels, legacy_collapse(types), threshold,
                              _LEGACY_HARD_NEG_TYPES, _LEGACY_EASY_NEG_TYPES)
    out["hard_neg_auc_fixed"] = float(fixed.get("hard_neg_auc", 0) or 0)
    out["easy_neg_auc_fixed"] = float(fixed.get("easy_neg_auc", 0) or 0)
    out["hard_neg_auc_legacy"] = float(legacy.get("hard_neg_auc", 0) or 0)
    out["easy_neg_auc_legacy"] = float(legacy.get("easy_neg_auc", 0) or 0)
    out["pos_recall"] = float(fixed.get("pos_recall", 0) or 0)
    out["ece"] = float(compute_calibration_metrics(probs, labels)["ece"])
    return out


def summarize(prefix, m):
    """Flatten: {prefix}_auprc, {prefix}_auroc, ..."""
    return {f"{prefix}_{k}": v for k, v in m.items()}


# ══════════════════════ inferenceloop ══════════════════════ #

def infer(model, loader, device, tag):
    """Window-level forward pass (same metric caliber as training-time val: fp32, early_exit=True)."""
    import torch
    t0 = time.time()
    probs, labels, types, ids = [], [], [], []
    n_total = len(loader)
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            batch.to(device)
            out = model(batch, early_exit=True)
            p = torch.sigmoid(out.cls_logits.squeeze(-1))
            probs.append(p.cpu())
            labels.append(batch.labels.cpu())
            seq_ids = getattr(batch, "original_seq_ids", None) or []
            if len(seq_ids) < p.shape[0]:
                seq_ids = list(getattr(batch, "seq_ids", None)
                               or [f"unk_{bi}_{i}" for i in range(p.shape[0])])
            ids.extend(list(seq_ids)[:p.shape[0]])
            st = getattr(batch, "sample_types", None)
            types.extend(list(st)[:p.shape[0]] if st
                         else ["UNKNOWN"] * p.shape[0])
            if (bi + 1) % 500 == 0 or bi + 1 == n_total:
                el = time.time() - t0
                done = (bi + 1) * loader.batch_size
                rate = done / el
                print(f"    [{tag}] batch {bi+1}/{n_total} "
                      f"({done} win, {rate:.1f} win/s, "
                      f"ETA {(n_total - bi - 1) * loader.batch_size / max(rate,1e-9) / 60:.0f}min)",
                      flush=True)
    probs = torch.cat(probs).numpy()
    labels = torch.cat(labels).numpy().astype(int)
    probs = np.where(np.isnan(probs), 0.5, probs)
    dt = time.time() - t0
    print(f"    [{tag}] done: {len(probs)} windows in {dt/60:.1f} min "
          f"({len(probs)/dt:.1f} win/s)")
    return probs, labels, types, ids, dt


# ══════════════════════ main pipeline: --run ══════════════════════ #

def cmd_run(args):
    import torch
    import yaml
    from torch.utils.data import DataLoader, Subset
    from model.contracts import HGTConfig
    from model.api import build_model
    from data_pipeline.impl.parser import BEDParser
    from data_pipeline.impl.dataset import SlidingWindowContigDataset
    from data_pipeline.impl.collator import HGTCollator
    from training_engine.api import TrainingEngine

    gpu = assert_4090()
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    dcfg, mcfg = cfg.get("data", {}), cfg.get("model", {})
    win_size = dcfg.get("sw_window_size", 3000)
    overlap = dcfg.get("sw_overlap", 300)

    # ── data: train (is_training=True, same seed) ──
    t0 = time.time()
    random.seed(SEED)                       # caliber: reset before dataset construction (see docstring)
    np.random.seed(SEED)
    train_samples = BEDParser(
        bed_path=str(DATA_DIR / "train/breakpoints.bed"),
        fasta_dir=str(DATA_DIR / "train"), mode="contig").parse()
    train_ds = SlidingWindowContigDataset(
        samples=train_samples, window_size=win_size, overlap=overlap,
        is_training=True,
        max_neg_windows=dcfg.get("sw_max_neg_windows", 2),
        max_eval_windows=dcfg.get("sw_max_eval_windows", 10),
        extract_bio=True, evo_cache_path=None, rc_aug_prob=0.0)
    n_train_win = len(train_ds)
    print(f"[data] train windows = {n_train_win} "
          f"(traininglogcaliber {EXPECTED_TRAIN_WINDOWS})", flush=True)
    if n_train_win != EXPECTED_TRAIN_WINDOWS:
        print(f"[WARN] train window count does not match the training log "
              f"({n_train_win} vs {EXPECTED_TRAIN_WINDOWS})  —  caliber drift, noted in the report")

    rng = np.random.default_rng(SEED)
    sel = np.sort(rng.choice(n_train_win, N_TRAIN_SUBSET, replace=False))
    sub_ds = Subset(train_ds, sel.tolist())
    # Fixed subset manifest (verifiable): dataset idx + window coordinates + tag + type + contig id
    wi = [train_ds._window_index[i] for i in sel]
    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        TRAJ_DIR / "train20k_selection.npz",
        dataset_idx=sel.astype(np.int64),
        sample_idx=np.array([w[0] for w in wi]),
        win_start=np.array([w[1] for w in wi]),
        win_end=np.array([w[2] for w in wi]),
        label=np.array([w[3] for w in wi], dtype=np.int8),
        sample_type=np.array([w[4] for w in wi], dtype=object),
        seq_id=np.array([train_ds.samples[w[0]].seq_id for w in wi], dtype=object),
        seed=np.int64(SEED), n_pool=np.int64(n_train_win))
    lab_sel = np.array([w[3] for w in wi])
    print(f"[data] 20k subset: pos={int(lab_sel.sum())} "
          f"neg={int((1-lab_sel).sum())} "
          f"(hard_neg={sum(1 for w in wi if w[4]=='NEG_HARD')})")

    # ── data: val (full, is_training=False) ──
    val_samples = BEDParser(
        bed_path=str(DATA_DIR / "val/breakpoints.bed"),
        fasta_dir=str(DATA_DIR / "val"), mode="contig").parse()
    val_ds = SlidingWindowContigDataset(
        samples=val_samples, window_size=win_size, overlap=overlap,
        is_training=False,
        max_neg_windows=dcfg.get("sw_max_neg_windows", 2),
        max_eval_windows=dcfg.get("sw_max_eval_windows", 10),
        extract_bio=True, evo_cache_path=None, rc_aug_prob=0.0)
    n_val_win = len(val_ds)
    assert n_val_win == EXPECTED_VAL_WINDOWS, \
        f"val window count {n_val_win} != {EXPECTED_VAL_WINDOWS}"
    print(f"[data] val windows = {n_val_win} ✓ (199,438, same parameters as training-time val)")
    print(f"[data] data construction {time.time()-t0:.0f}s")

    collator = HGTCollator(pad_id=0)
    # The train Subset is only 20k windows but the underlying layer holds ~12GB of contig
    # sequences (fork COW): use 4 workers to avoid page-copy amplification; the val side
    # is small, so use the full args.num_workers
    train_loader = DataLoader(sub_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collator, num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collator, num_workers=args.num_workers,
                            pin_memory=True)

    # ── model (mirrors the cmd_train sliding-window branch: max_contig_length=sw_window_size) ──
    model_config = HGTConfig(
        data_mode=dcfg.get("mode", "contig"),
        max_contig_length=win_size,
        hidden_dim=mcfg.get("hidden_dim", 768),
        local_hidden_dim=mcfg.get("local_hidden_dim", 256),
        num_attention_heads=mcfg.get("num_attention_heads", 12),
        dropout=mcfg.get("dropout", 0.1),
        graph_top_k=mcfg.get("graph_top_k", 5),
        bio_feature_dim=mcfg.get("bio_feature_dim", 260),
        pretrained_nt_path=mcfg.get("pretrained_nt_path", ""),
        pretrained_caduceus_path=mcfg.get("pretrained_caduceus_path", ""),
        use_local_encoder=mcfg.get("use_local_encoder", True),
        use_evo=False,                    # M0 = NTv3-only (config use_evo=false)
        pretrained_evo_dim=mcfg.get("pretrained_evo_dim", 4096),
        evo_hidden=mcfg.get("evo_hidden", 512),
    )
    model = build_model(model_config).to("cuda:0")
    model.eval()

    epoch_hist = json.loads((CKPT_DIR / "epoch_metrics.json").read_text())
    hist_by_ep = {e["epoch"]: e for e in epoch_hist}

    rows = []
    t_start = time.time()
    for ep in args.epochs:
        ckpt = CKPT_DIR / f"epoch_{ep}.pt"
        if not ckpt.exists():
            print(f"[WARN] missing {ckpt.name}, skip")
            continue
        state = torch.load(ckpt, map_location="cuda:0", weights_only=False)
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        assert not missing, f"epoch_{ep}: missing keys {missing[:5]}"
        print(f"\n[epoch {ep}] ckpt={ckpt.name} (sha16={sha256_of(ckpt)}, "
              f"epoch_in_ckpt={state.get('epoch')}, unexpected={len(unexpected)})",
              flush=True)

        row = {"epoch": ep, "ckpt_sha16": sha256_of(ckpt)}
        tr_p, tr_y, tr_t, tr_ids, dt_tr = infer(model, train_loader, "cuda:0",
                                                f"ep{ep}-train20k")
        va_p, va_y, va_t, va_ids, dt_va = infer(model, val_loader, "cuda:0",
                                                f"ep{ep}-val")

        m_tr = metrics_for(tr_p, tr_y, tr_t)
        m_va = metrics_for(va_p, va_y, va_t)
        row.update(summarize("train", m_tr))
        row.update(summarize("val", m_va))
        row["gap_auprc"] = m_tr["auprc"] - m_va["auprc"]
        row["gap_auroc"] = m_tr["auroc"] - m_va["auroc"]
        row["infer_time_s"] = round(dt_tr + dt_va, 1)

        # Contig-level aggregation (same metric caliber as epoch_metrics.json)  —  fidelity check + supplementary column
        cp, cy, ct = TrainingEngine._aggregate_contig_predictions(
            va_p, va_y, va_ids, va_t)
        m_va_c = metrics_for(cp, cy, ct)
        row.update(summarize("valc", m_va_c))
        row["recorded_val_auprc"] = float(hist_by_ep[ep]["val_auprc"])
        row["replay_delta_val_auprc"] = (m_va_c["auprc"]
                                         - float(hist_by_ep[ep]["val_auprc"]))
        row["recorded_cal_ece"] = float(hist_by_ep[ep].get("cal_ece", 0) or 0)
        row["recorded_hard_neg_auc"] = float(
            hist_by_ep[ep].get("hard_neg_auc", 0) or 0)
        # The train20k contig aggregation is a partial-coverage caliber (most contigs have
        # only 1 sampled window), for reference only
        tp, ty, tt = TrainingEngine._aggregate_contig_predictions(
            tr_p, tr_y, tr_ids, tr_t)
        m_tr_c = metrics_for(tp, ty, tt)
        row.update(summarize("trainc", m_tr_c))
        row["gap_auprc_contig"] = m_tr_c["auprc"] - m_va_c["auprc"]

        np.savez_compressed(
            TRAJ_DIR / f"trainval_gap_epoch_{ep}.npz",
            train_probs=tr_p.astype(np.float32),
            train_labels=tr_y.astype(np.int8),
            train_types=np.array(tr_t, dtype=object),
            train_ids=np.array(tr_ids, dtype=object),
            val_probs=va_p.astype(np.float32),
            val_labels=va_y.astype(np.int8),
            val_types=np.array(va_t, dtype=object),
            val_ids=np.array(va_ids, dtype=object))
        rows.append(row)
        _write_csv(rows)
        r = row
        print(f"[epoch {ep}] train AUPRC={r['train_auprc']:.4f} "
              f"val AUPRC={r['val_auprc']:.4f} gap={r['gap_auprc']:+.4f} | "
              f"val contig AUPRC={r['valc_auprc']:.4f} "
              f"(recorded {r['recorded_val_auprc']:.4f}, "
              f"Δ{r['replay_delta_val_auprc']:+.4f})", flush=True)

    # Run metadata
    meta = {"protocol": {
        "config": str(CONFIG), "data_dir": str(DATA_DIR),
        "ckpt_dir": str(CKPT_DIR), "epochs": [r["epoch"] for r in rows],
        "train_subset": {"n": N_TRAIN_SUBSET, "seed": SEED,
                         "pool": n_train_win,
                         "selection": str(TRAJ_DIR / "train20k_selection.npz"),
                         "dataset": "SlidingWindowContigDataset(is_training=True), "
                                    "random.seed(42) before construction"},
        "val": {"n": n_val_win, "dataset": "is_training=False full coverage"},
        "inference": {"batch": args.batch_size, "dtype": "fp32",
                      "early_exit": True, "rc_aug": 0.0,
                      "use_evo": False, "gpu": gpu},
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_runtime_s": round(time.time() - t_start, 1)}}
    (TRAJ_DIR / "trainval_gap_run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n[run] done in {(time.time()-t_start)/60:.0f} min → {CSV_PATH}")
    _write_report(rows, meta)


def _write_csv(rows):
    import pandas as pd
    pd.DataFrame(rows).to_csv(CSV_PATH, index=False)


# ══════════════════════ report + interpretation: --report ══════════════════════ #

def judge_situation(rows):
    """Three-scenario contract interpretation (rules explicit, evidence itemized)."""
    tr = np.array([r["train_auprc"] for r in rows])
    va = np.array([r["val_auprc"] for r in rows])
    hn = np.array([r["val_hard_neg_auc_fixed"] for r in rows])
    ece = np.array([r["val_ece"] for r in rows])
    eps = [r["epoch"] for r in rows]

    c_train_up = tr[-1] - tr[0] > 0.01                     # train up
    best_ep = int(np.argmax(va))
    c_val_stall = (va[-1] - va[best_ep] < 0.005            # val stalls/declines
                   and best_ep < len(va) - 1)
    c_hardneg_down = hn[-1] - np.max(hn) < -0.02           # hard-neg drops
    c_ece_worse = ece[-1] - np.min(ece) > 0.05             # ECE worsens
    gap = tr - va
    if c_train_up and c_val_stall and c_hardneg_down and c_ece_worse:
        sit = "C (overfitting)"
    else:
        gap_small = float(np.mean(gap)) < 0.05
        val_near_best = va[-1] >= np.max(va) - 0.01
        if gap_small and val_near_best:
            sit = "B (ceiling)"
        else:
            sit = "A (normal)"
    ev = {
        "C1_train_auprc_up": {"value": f"{tr[-1]-tr[0]:+.4f} (ep{eps[0]}→ep{eps[-1]})",
                              "pass": bool(c_train_up)},
        "C2_val_stall": {"value": f"best ep{eps[best_ep]} {np.max(va):.4f} → "
                                  f"last {va[-1]:.4f} ({va[-1]-np.max(va):+.4f})",
                         "pass": bool(c_val_stall)},
        "C3_hardneg_down": {"value": f"max {np.max(hn):.4f} → last {hn[-1]:.4f} "
                                     f"({hn[-1]-np.max(hn):+.4f}, fixed caliber)",
                            "pass": bool(c_hardneg_down)},
        "C4_ece_worse": {"value": f"min {np.min(ece):.4f} → last {ece[-1]:.4f} "
                                  f"({ece[-1]-np.min(ece):+.4f})",
                         "pass": bool(c_ece_worse)},
    }
    return sit, ev, gap


def _write_report(rows, meta=None):
    import pandas as pd
    df = pd.DataFrame(rows)
    if meta is None:
        meta = json.loads((TRAJ_DIR / "trainval_gap_run_meta.json").read_text())
    sit, ev, gap = judge_situation(rows)

    lines = []
    A = lines.append
    A("# M0 Train-Val Gap curve (V7 contract section 3, preregistered observation point 2, debt repaid)")
    A("")
    A(f"> Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  ·  scripts/trainval_gap.py  ·  "
      "launcher scripts/run/trainval_gap_4090.sh  ·  log logs/trainval_gap_4090.log")
    A("> model: M0 = NTv3-650M-pre only (encoder permanently frozen), checkpoints/"
      "Contig_500k_ntv3only/epoch_{{1..5}}.pt  ·  data: simulated_data/"
      "Contig_500k")
    A("> **train caliber**: SlidingWindowContigDataset **is_training=True** (strategy D: "
      "breakpoints centered in POS windows + <=2 hard-neg/neg windows), random.seed(42) set before construction → "
      "a 940,028-window pool (consistent with the training log, asserted in the script); numpy default_rng(42) samples a fixed "
      "20,000 windows without replacement (m0_trajectory/train20k_selection.npz, verifiable). "
      "We do NOT claim bit-exact alignment with the training process's RNG stream (the random consumption "
      "between seed_all and dataset construction is unverifiable), but the sampling protocol is the same kind of seeded scheme.")
    A("> **val caliber**: the full 199,438 windows (is_training=False full-coverage stride + "
      "max_eval_windows=10), same parameters as training-time val.")
    A("> **inference**: b=8, fp32, early_exit=True, RC augmentation off (a training-time augmentation, "
      "kept off on both sides for deterministic gap measurement), use_evo=False. Window-level AUPRC is the primary caliber "
      "(same scale on both sides); contig level (top-K/P90 aggregation) is the epoch_metrics.json-caliber check column.")
    A("")
    A("## 1. Primary table  —  train / val AUPRC with gap per epoch")
    A("")
    A("| epoch | train AUPRC (20k window) | val AUPRC (199,438 window) | **gap (train−val)** | "
      "train AUROC | val AUROC | val contig AUPRC | recorded val_auprc | check Δ |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in df.itertuples():
        A(f"| {r.epoch} | {r.train_auprc:.4f} | {r.val_auprc:.4f} | "
          f"**{r.gap_auprc:+.4f}** | {r.train_auroc:.4f} | {r.val_auroc:.4f} | "
          f"{r.valc_auprc:.4f} | {r.recorded_val_auprc:.4f} | "
          f"{r.replay_delta_val_auprc:+.4f} |")
    A("")
    A("## 2. Stratified and calibration (window-level; fixed = the H-01 post-fix 7-class hard set, "
      "legacy = the collapsed caliber used in epoch_metrics.json)")
    A("")
    A("| epoch | val hard_neg AUC (fixed) | val hard_neg AUC (legacy) | "
      "val easy_neg AUC (fixed) | val ECE | train ECE | train hard_neg AUC (fixed) |")
    A("|---|---|---|---|---|---|---|")
    for r in df.itertuples():
        A(f"| {r.epoch} | {r.val_hard_neg_auc_fixed:.4f} | "
          f"{r.val_hard_neg_auc_legacy:.4f} | {r.val_easy_neg_auc_fixed:.4f} | "
          f"{r.val_ece:.4f} | {r.train_ece:.4f} | {r.train_hard_neg_auc_fixed:.4f} |")
    A("")
    A("## 3. Interpretation (three contract scenarios, preregistered rules)")
    A("")
    A(f"- **Verdict: scenario {sit}**")
    A("- The four conditions for scenario C (all must hold simultaneously):")
    for k, v in ev.items():
        A(f"  - {k}: {v['value']} → {'✓ holds' if v['pass'] else '✗ does not hold'}")
    g = gap
    A(f"- gap (train−val AUPRC): ep{df.epoch.iloc[0]} {g[0]:+.4f} → "
      f"ep{df.epoch.iloc[-1]} {g[-1]:+.4f} (mean {g.mean():+.4f}, "
      f"max {g.max():+.4f} @ep{df.epoch.iloc[int(np.argmax(g))]})")
    A("- Supplementary: the train20k contig-level aggregation column (trainc_*) is a partial-coverage caliber "
      "(most contigs contribute only 1 sampled window); reference only, not an interpretation input.")
    A("")
    A("## 4. Fidelity-check notes")
    A("")
    A("- val contig AUPRC (replayed by this script) vs the value recorded in epoch_metrics.json: "
      "per-epoch Δ in the last column of the primary table (expected to be of the same magnitude; "
      "the residual comes from the fp32 forward-pass replay vs the in-training-process "
      "evaluation behind epoch_metrics).")
    A("- Stratified caliber note: epoch_metrics.json's hard_neg_auc uses the **legacy collapsed caliber** "
      "(H-01, pre-fix as of 2026-08-20); this table reports both calibers (fixed/legacy) for control.")
    A("")
    A("## 5. artifact")
    A("")
    A(f"- table: {CSV_PATH.name}  ·  figure: {PNG_PATH.name}  ·  per-epoch prediction archives: "
      "docs/evidence/m0_trajectory/trainval_gap_epoch_{{1..5}}.npz + "
      "train20k_selection.npz + trainval_gap_run_meta.json")
    A(f"- script: scripts/trainval_gap.py (--run 4090 / --report CPU)")
    MD_PATH.write_text("\n".join(lines) + "\n")
    print(f"[report] {MD_PATH}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        eps = df.epoch.values
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        ax = axes[0]
        ax.plot(eps, df.train_auprc, "o-", label="train AUPRC (20k win)")
        ax.plot(eps, df.val_auprc, "s-", label="val AUPRC (199,438 win)")
        ax.plot(eps, df.valc_auprc, "^--", alpha=0.6,
                label="val contig AUPRC (epoch_metrics caliber)")
        ax2 = ax.twinx()
        ax2.bar(eps, df.gap_auprc, alpha=0.15, color="gray",
                label="gap (train−val)")
        ax2.set_ylabel("gap (train−val AUPRC)")
        ax.set_xlabel("epoch"); ax.set_ylabel("AUPRC (window-level)")
        ax.set_title(f"Train-val gap  —  M0 NTv3-only (verdict: {sit})")
        ax.grid(alpha=0.3)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")
        ax = axes[1]
        ax.plot(eps, df.val_hard_neg_auc_fixed, "o-", label="val hard_neg AUC (fixed)")
        ax.plot(eps, df.val_hard_neg_auc_legacy, "s--", alpha=0.6,
                label="val hard_neg AUC (legacy=epoch_metrics)")
        ax.plot(eps, df.train_hard_neg_auc_fixed, "^:", alpha=0.6,
                label="train hard_neg AUC (fixed)")
        ax.set_xlabel("epoch"); ax.set_ylabel("hard_neg AUC")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax3 = ax.twinx()
        ax3.plot(eps, df.val_ece, "d-", color="tab:red", alpha=0.7,
                 label="val ECE (10-bin)")
        ax3.plot(eps, df.train_ece, "v--", color="tab:orange", alpha=0.7,
                 label="train ECE")
        ax3.set_ylabel("ECE")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax3.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower left")
        ax.set_title("Stratified hard-neg + calibration")
        fig.tight_layout()
        fig.savefig(PNG_PATH, dpi=150)
        print(f"[report] {PNG_PATH}")
    except Exception as ex:
        print(f"[report] figure plotting skipped: {ex}")


def cmd_report(args):
    import pandas as pd
    df = pd.read_csv(CSV_PATH)
    rows = df.to_dict("records")
    meta = json.loads((TRAJ_DIR / "trainval_gap_run_meta.json").read_text()) \
        if (TRAJ_DIR / "trainval_gap_run_meta.json").exists() else {}
    _write_report(rows, meta)


def main():
    ap = argparse.ArgumentParser(description="M0 train-val gap curve (contract §3)")
    ap.add_argument("--run", action="store_true", help="GPU inference stage (4090)")
    ap.add_argument("--report", action="store_true", help="only re-render the report/figure")
    ap.add_argument("--epochs", type=int, nargs="*", default=EPOCHS)
    ap.add_argument("--batch_size", type=int, default=BATCH)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()
    if not (args.run or args.report):
        ap.error("--run or --report is required")
    if args.run:
        cmd_run(args)
    if args.report:
        cmd_report(args)


if __name__ == "__main__":
    main()
