#!/usr/bin/env python
# scripts/recompute_stratified.py
"""
DeepFluHGT  —  H-01 (CODE_AUDIT_V7) offline stratified-metric recomputation
==========================================================
Background: the parser collapsed negative-sample types (it only recognized 4/10 classes,
falling back to "NEG" for the rest), which systematically distorted the
hard_neg_auc / easy_neg_auc / per_type_counts in epoch_metrics.json:
  - the hard pool actually contained only NEG_HARD (single-type subset AUC);
  - the easy pool mixed in 6 classes of hard negatives (55% of negative contigs during training) → easy_neg_auc systematically too low;
  - per_type_counts / neg_specificity_by_type merged six classes into NEG.
Fix (2026-08-20, approved): data_pipeline/impl/parser.py keeps the full negative-type set +
evaluation/impl/metrics.py expands the hard/easy sets. Pure evaluation-caliber change, weights unchanged.

** When to run: run after M0 training completes (replay mode by default detects a running
   training process and refuses to start; --force bypasses it). Historical epoch_metrics.json stratified
   numbers are NOT rewritten  —  —  this script outputs an independent
   JSON (including before/after-fix controls) for paper/analysis reference.**

Two modes
--------
1. replay (primary path): load the checkpoint + val/test split and replay the forward pass,
   producing an aligned (probs, labels, sample_types) archive and recomputing. The forward loop
   only collects cls probabilities; aggregation reuses TrainingEngine._aggregate_contig_predictions
   (same metric caliber as training-time val: contig-level top-K/P90 aggregation, early_exit defaults to True).

   CUDA_VISIBLE_DEVICES=<gpu-uuid> python \\
       scripts/recompute_stratified.py replay \\
       --config configs/contig_500k_ntv3only_4090.yaml \\
       --data_dir simulated_data/Contig_500k \\
       --checkpoint checkpoints/Contig_500k_ntv3only/best_model.pt \\
       --split val --batch_size 8 \\
       --dump-archive docs/evidence/m0_trajectory/val_best_preds.npz \\
       --output docs/evidence/m0_trajectory/stratified_recompute_val_best.json

2. archive (seconds-level): recompute directly from the npz written by replay --dump-archive
   (no forward pass; reuse the same archive when changing thresholds
   or comparing checkpoints).

   python scripts/recompute_stratified.py \\
       archive --preds docs/evidence/m0_trajectory/val_best_preds.npz \\
       --output docs/evidence/m0_trajectory/stratified_recompute_val_best.json

The "legacy" column: recomputed on **the same set of predictions** using the pre-fix collapsed
mapping (6 classes → "NEG") and the pre-fix hard/easy sets  —  —  it exactly reproduces the caliber
of the numbers the running M0 already wrote to disk and quantifies the caliber difference the fix
introduces (no training rerun). Model weights and training hyperparameters are untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Pre-fix caliber (use only for same-prediction controls; do not use for new reports) ──
# The pre-fix parser only passed through 4 classes; everything else fell back to "NEG".
_LEGACY_KNOWN_TYPES = {"NEG", "NEG_VIRUS", "NEG_HARD", "NEG_PCR"}
_LEGACY_HARD_NEG_TYPES = {"NEG_HARD", "NEG_HOMOLOGY", "NEG_CONSERVED",
                          "NEG_LOW_COMPLEXITY"}   # after the collapse, only NEG_HARD actually reaches this set
_LEGACY_EASY_NEG_TYPES = {"NEG", "NEG_VIRUS", "NEG_PCR"}

# Post-fix canonical sets (kept consistent with evaluation/impl/metrics.py; spelled out explicitly for report reference)
FIXED_HARD_NEG_TYPES = {
    "NEG_HARD", "NEG_HOMOLOGY", "NEG_CONSERVED", "NEG_LOW_COMPLEXITY",
    "NEG_GC_MATCHED_CHIMERIC", "NEG_AMALGAMATED_HGT", "NEG_PARTIAL_VIRUS_HIT",
}
FIXED_EASY_NEG_TYPES = {"NEG"}


def legacy_collapse(types):
    """Exactly emulate the pre-fix parser's type collapse: non-POS types outside the 4 known classes → 'NEG'."""
    return ["NEG" if (t not in _LEGACY_KNOWN_TYPES and t != "POS") else t
            for t in types]


def _strat_with_sets(probs, labels, types, threshold, hard_set, easy_set):
    """Compute stratified metrics with the specified hard/easy sets (replicates the bucketing
    logic of metrics.compute_stratified_metrics with parameterized sets  —  —  legacy-caliber control only)."""
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels).astype(int)
    types = list(types)
    n = len(probs)
    preds = (np.asarray(probs) >= threshold).astype(int)
    pos_mask = labels == 1
    neg_mask = labels == 0

    out = {}
    per_type_counts = {}
    for st in types:
        per_type_counts[st] = per_type_counts.get(st, 0) + 1
    out["per_type_counts"] = per_type_counts

    for name, sel in (("hard_neg_auc", hard_set), ("easy_neg_auc", easy_set)):
        neg_idx = np.array([i for i in range(n)
                            if neg_mask[i] and types[i] in sel])
        pos_idx = np.where(pos_mask)[0]
        if len(pos_idx) > 0 and len(neg_idx) > 0:
            idx = np.concatenate([pos_idx, neg_idx])
            out[name] = float(roc_auc_score(labels[idx], probs[idx]))
        else:
            out[name] = 0.0
    return out


def recompute_all(probs, labels, types, threshold=0.5):
    """Post-fix caliber (calls the single source of truth, evaluation.impl.metrics) + legacy control."""
    from evaluation.impl.metrics import compute_stratified_metrics

    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels).astype(int)

    fixed = compute_stratified_metrics(probs, labels, list(types), threshold)
    legacy_types = legacy_collapse(types)
    legacy = _strat_with_sets(probs, labels, legacy_types, threshold,
                              _LEGACY_HARD_NEG_TYPES, _LEGACY_EASY_NEG_TYPES)

    return {
        "n_samples": int(len(probs)),
        "threshold": float(threshold),
        "fixed": fixed,      # post-fix: hard=7 classes, easy=pure NEG, 10-class per_type_counts
        "legacy": legacy,    # pre-fix caliber (same-prediction control, = the caliber of the numbers the running M0 already wrote to disk)
        "delta": {
            "hard_neg_auc": fixed["hard_neg_auc"] - legacy["hard_neg_auc"],
            "easy_neg_auc": fixed["easy_neg_auc"] - legacy["easy_neg_auc"],
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════ #
#  M0 run guard (do not disturb running processes)
# ═══════════════════════════════════════════════════════════════════════════════ #

def guard_no_running_training(force: bool) -> None:
    """Replay-mode guard: detect a running DeepFluHGT training process and refuse to start.

    The replay itself only reads checkpoints/data, but it does occupy GPU/CPU/GPU memory;
    competing with M0 for the card would slow or even OOM the running job  —  —  discipline:
    run after M0 completes.
    """
    try:
        r = subprocess.run(["pgrep", "-af", "DeepFluHGT.py"],
                           capture_output=True, text=True, timeout=30)
        procs = [ln for ln in (r.stdout or "").strip().splitlines() if ln.strip()]
    except Exception as e:  # pgrep unavailable etc. → do not block, just print a notice
        print(f"[guard] pgrep unavailable ({e}), skipping running-process detection")
        return
    if procs:
        print("[guard] detected running DeepFluHGT process(es):")
        for ln in procs:
            print(f"        {ln}")
        if not force:
            print("[guard] H-01 discipline: wait for M0 to finish before running this script "
                  "(grabbing the GPU would disturb the running training). To run anyway, add --force.")
            sys.exit(1)
        print("[guard] --force: continuing (you accept the resource-contention risk)")


# ═══════════════════════════════════════════════════════════════════════════════ #
#  mode 1: replay (replay the forward pass)
# ═══════════════════════════════════════════════════════════════════════════════ #

def cmd_replay(args) -> None:
    guard_no_running_training(args.force)

    import torch
    import yaml
    from torch.utils.data import DataLoader
    from model.contracts import HGTConfig
    from model.api import build_model
    from data_pipeline.api import load_splits_sliding_window
    from data_pipeline.impl.collator import HGTCollator
    from training_engine.api import TrainingEngine

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_cfg, model_cfg = cfg.get("data", {}), cfg.get("model", {})
    dcfg = data_cfg  # shorthand

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[replay] device={device}, config={args.config}, "
          f"checkpoint={args.checkpoint}, split={args.split}")

    # ── data (same parameters as training-time val; eval splits get no RC augmentation / no random windows) ──
    splits = load_splits_sliding_window(
        data_dir=args.data_dir,
        window_size=dcfg.get("sw_window_size", 1500),
        overlap=dcfg.get("sw_overlap", 300),
        max_neg_windows=dcfg.get("sw_max_neg_windows", 2),
        max_eval_windows=dcfg.get("sw_max_eval_windows", 10),
        read_len=dcfg.get("read_length", 150),
        # evo_cache_path left empty: recomputing stratified metrics needs no evo input;
        # M0 is NTv3-only, and this avoids reading an evo_cache still being generated
        evo_cache_path="",
        rc_aug_prob=0.0,
    )
    ds = splits.get(args.split)
    if ds is None:
        print(f"[ERROR] split '{args.split}' does not exist (available: {list(splits)})")
        sys.exit(1)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=HGTCollator(pad_id=0),
                        num_workers=args.num_workers, pin_memory=(device == "cuda"))

    # ── model + checkpoint (mirrors the cmd_eval construction; strict=False tolerates the rotary buffer) ──
    model_config = HGTConfig(
        data_mode=dcfg.get("mode", "contig"),
        max_contig_length=dcfg.get("sw_window_size", 1500),
        hidden_dim=model_cfg.get("hidden_dim", 768),
        num_attention_heads=model_cfg.get("num_attention_heads", 12),
        dropout=model_cfg.get("dropout", 0.1),
        graph_top_k=model_cfg.get("graph_top_k", 5),
        bio_feature_dim=model_cfg.get("bio_feature_dim", 260),
        pretrained_nt_path=model_cfg.get("pretrained_nt_path", ""),
        pretrained_caduceus_path=model_cfg.get("pretrained_caduceus_path", ""),
        use_local_encoder=model_cfg.get("use_local_encoder", True),
        use_evo=False,   # recomputing stratified metrics needs no evo branch
        pretrained_evo_dim=model_cfg.get("pretrained_evo_dim", 4096),
        evo_hidden=model_cfg.get("evo_hidden", 512),
    )
    model = build_model(model_config).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    print(f"[replay] checkpoint loaded (missing={len(missing)}, "
          f"unexpected={len(unexpected)}, epoch={state.get('epoch', '?')})")
    threshold = float(state.get("optimal_threshold")
                      or (state.get("metrics") or {}).get("optimal_threshold", 0.5))
    model.eval()

    # ── forward pass: collect only cls probabilities + contig ids + types (aggregation logic reuses the engine's static method) ──
    all_probs, all_labels, all_ids, all_types = [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            batch.to(device)
            out = model(batch)   # early_exit defaults to True, consistent with training-time val
            probs = torch.sigmoid(out.cls_logits.squeeze(-1))
            all_probs.append(probs.cpu())
            all_labels.append(batch.labels.cpu())
            seq_ids = (getattr(batch, "original_seq_ids", None)
                       or batch.seq_ids or [])
            all_ids.extend(seq_ids[:probs.shape[0]]
                           if len(seq_ids) >= probs.shape[0]
                           else (batch.seq_ids or
                                 [f"unk_{bi}_{i}" for i in range(probs.shape[0])]))
            all_types.extend(list(batch.sample_types or ["UNKNOWN"] * probs.shape[0]))
            if (bi + 1) % 100 == 0:
                print(f"[replay] batches {bi + 1} ({time.time() - t0:.0f}s)")

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)

    # window-level → contig-level aggregation (same as training-time val: adaptive top-K/P90, first-window type as representative)
    probs_c, labels_c, types_c = TrainingEngine._aggregate_contig_predictions(
        probs, labels, all_ids, all_types)
    print(f"[replay] windows={len(probs)} -> contigs={len(probs_c)}")

    result = recompute_all(probs_c, labels_c, types_c, threshold)
    result["mode"] = "replay"
    result["config"] = args.config
    result["checkpoint"] = args.checkpoint
    result["split"] = args.split
    result["checkpoint_epoch"] = state.get("epoch")
    result["window_level"] = recompute_all(probs, labels, all_types, threshold)

    if args.dump_archive:
        os.makedirs(os.path.dirname(args.dump_archive) or ".", exist_ok=True)
        np.savez_compressed(
            args.dump_archive,
            probs_contig=probs_c.astype(np.float32),
            labels_contig=labels_c.astype(np.int8),
            types_contig=np.array(types_c, dtype=object),
            probs_window=probs.astype(np.float32),
            labels_window=labels.astype(np.int8),
            types_window=np.array(all_types, dtype=object),
            threshold=np.float64(threshold),
        )
        result["archive"] = args.dump_archive
        print(f"[replay] prediction archive → {args.dump_archive} (later reusable by archive mode for seconds-level recomputation)")

    _write_output(args.output, result)


# ═══════════════════════════════════════════════════════════════════════════════ #
#  mode 2: archive (recompute from a saved archive)
# ═══════════════════════════════════════════════════════════════════════════════ #

def cmd_archive(args) -> None:
    z = np.load(args.preds, allow_pickle=True)
    threshold = float(z["threshold"]) if "threshold" in z else args.threshold
    result = recompute_all(z["probs_contig"], z["labels_contig"],
                           [str(t) for t in z["types_contig"]], threshold)
    result["mode"] = "archive"
    result["preds"] = args.preds
    if "probs_window" in z:
        result["window_level"] = recompute_all(
            z["probs_window"], z["labels_window"],
            [str(t) for t in z["types_window"]], threshold)
    _write_output(args.output, result)


def _write_output(path, result):
    print("\n===== stratified metric recomputation (H-01: post-fix caliber vs pre-fix caliber, same set of predictions) =====")
    print(f"  n={result['n_samples']}, threshold={result['threshold']}")
    print(f"  hard_neg_auc: legacy={result['legacy']['hard_neg_auc']:.4f} -> "
          f"fixed={result['fixed']['hard_neg_auc']:.4f} "
          f"(delta={result['delta']['hard_neg_auc']:+.4f})")
    print(f"  easy_neg_auc: legacy={result['legacy']['easy_neg_auc']:.4f} -> "
          f"fixed={result['fixed']['easy_neg_auc']:.4f} "
          f"(delta={result['delta']['easy_neg_auc']:+.4f})")
    print(f"  per_type_counts (fixed): {result['fixed']['per_type_counts']}")
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n[result] JSON -> {path}")
    print("  (historical epoch_metrics.json is not rewritten; this JSON stands as independent control evidence)")


def main():
    ap = argparse.ArgumentParser(
        description="H-01 offline stratified-metric recomputation (run after M0 completes)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("replay", help="load a checkpoint and replay the val/test forward pass")
    p1.add_argument("--config", required=True)
    p1.add_argument("--data_dir", required=True)
    p1.add_argument("--checkpoint", required=True)
    p1.add_argument("--split", default="val", choices=["val", "test"])
    p1.add_argument("--batch_size", type=int, default=8)
    p1.add_argument("--num_workers", type=int, default=8)
    p1.add_argument("--dump-archive", default=None,
                    help="aligned prediction archive npz (probs/labels/types, contig+window level)")
    p1.add_argument("--output", default=None)
    p1.add_argument("--force", action="store_true",
                    help="bypass the running-training guard (not recommended)")
    p1.set_defaults(func=cmd_replay)

    p2 = sub.add_parser("archive", help="seconds-level recompute from a replay --dump-archive npz")
    p2.add_argument("--preds", required=True)
    p2.add_argument("--threshold", type=float, default=0.5)
    p2.add_argument("--output", default=None)
    p2.set_defaults(func=cmd_archive)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
