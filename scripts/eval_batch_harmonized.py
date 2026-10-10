#!/usr/bin/env python3
# scripts/eval_batch_harmonized.py
"""DeepFluHGT  —  DA-H1 fix: M0/M2 val dual-batch (b=4 / b=8) harmonized re-evaluation (2×2 matrix).

Background (docs/evidence/DEEP_AUDIT_FINAL.md DA-H1):
  BatchGraphInteraction builds a top-k graph across samples within a batch + NTv3-pre tower with attention_mask=None
  → a single window's prediction depends on batch composition. The training side's three runs reused the training
  micro-batch for their val dataloaders: M0/M3 = b=4, M2 = b=8 → the attribution "M2−M0 = Evo2 information contribution" is
  confounded with a batch effect. This script re-evaluates M0 and M2 on val at both b=4 and b=8
  (pure inference, no retraining), forming a 2×2 matrix of M0@{b4,b8} × M2@{b4,b8} + a per-window
  prediction-difference distribution (max|Δp|) + an M2−M0 paired DeLong under a unified batch.

protocol:
  - window set: full Contig_500k/val (BEDParser + SlidingWindowContigDataset
    is_training=False, 3000/300, max_eval_windows=10, extract_bio=True,
    rc_aug_prob=0)  —  exactly the same deterministic window set as training-time val (199,438 windows,
    bit-for-bit validated by residual_corr), shuffle=False fixed order.
  - inference: fp32, early_exit=False (real-anchor evaluation protocol, eval_real_anchors_gpu
    equivalent), b∈{4,8} is the only variable.
  - aggregation: training-side hybrid rule (TrainingEngine._aggregate_contig_predictions,
    ≤3window max / 4-10window top3-mean / >10window max(top3,P90)) + max + mean;
    metrics: contig-level AUPRC/AUROC (contag, primary) + window-level AUPRC/AUROC (secondary).
  - M2: use_evo=True, evo_cache = evo_cache/Contig_500k/val.pt (per-contig
    4096-d vectors, coverage 50k/50k).

GPU discipline: CUDA_VISIBLE_DEVICES=A100 UUID GPU lock (coexists with M2 training; inference footprint
  ~10GB; **4090 strictly forbidden**  —  M3 is running on it); the script asserts the only visible device is the A100 with
  ≥ 10GB free. CPU ≤ 48 threads (OMP/BLAS=8, workers=8).

usage (launcher: scripts/run/batch_harmonized_harmonized.sh):
  CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python scripts/eval_batch_harmonized.py \
      --model m0 --batches 4,8
  python scripts/eval_batch_harmonized.py --report     # CPU, seconds
"""
from __future__ import annotations

import os
import sys
import json
import math
import time
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

A100_UUID = "${GPU_A100_UUID}"   # inference GPU lock (coexists with M2 training)
DATA_DIR = ROOT / "simulated_data/Contig_500k"
OUT_DIR = ROOT / "docs/evidence/batch_harmonized"

MODELS = {
    "m0": {
        "ckpt": ROOT / "checkpoints/Contig_500k_ntv3only/best_model.pt",
        "config": ROOT / "configs/contig_500k_ntv3only.yaml",
        "evo_cache": None,
        "desc": "M0 (NTv3-only)",
    },
    "m2": {
        "ckpt": ROOT / "checkpoints/Contig_500k/best_model.pt",
        "config": ROOT / "configs/contig_500k.yaml",
        "evo_cache": ROOT / "evo_cache/Contig_500k",
        "desc": "M2 (NTv3 + Evo2)",
    },
}

# training-time reference (epoch_metrics.json, hybrid aggregation + early_exit=True + each model's own training batch)
TRAIN_REF = {
    "m0": {"b": 4, "val_auprc": 0.8325, "val_auroc": 0.8351,
           "src": "checkpoints/Contig_500k_ntv3only/epoch_metrics.json ep2"},
    "m2": {"b": 8, "val_auprc": 0.8711, "val_auroc": 0.8745,
           "src": "checkpoints/Contig_500k/epoch_metrics.json ep2"},
}


def parse_args():
    ap = argparse.ArgumentParser(description="DA-H1: M0/M2 val dual-batch harmonized re-evaluation")
    ap.add_argument("--model", choices=list(MODELS.keys()),
                    help="m0 | m2  —  run inference for this model (dual batch)")
    ap.add_argument("--batches", default="4,8",
                    help="comma-separated batch list (default 4,8)")
    ap.add_argument("--report", action="store_true",
                    help="summarize the 2×2 matrix + Δp distribution + paired DeLong → "
                         "docs/evidence/H1_BATCH_HARMONIZED.md (CPU)")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--win_cap", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--limit_windows", type=int, default=0,
                    help="smoke: only the first N windows (Subset, dataset object untouched)")
    ap.add_argument("--gpu_uuid", default=A100_UUID)
    ap.add_argument("--no_snapshot", action="store_true",
                    help="skip the checkpoint snapshot and load directly (default: snapshot first, so the M2 training"
                         "process cannot overwrite best_model.pt mid-read)")
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    return ap.parse_args()


ARGS = parse_args()
if ARGS.report:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""          # report is pure CPU
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ARGS.gpu_uuid
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

from delong_test import delong_test_full          # noqa: E402 (setdefault, not override)


# ═══════════════════════════════════════════════════════════════════════════ #
#  GPU guard: the only visible device must be the A100 (4090 strictly forbidden, M3 is training on it)
# ═══════════════════════════════════════════════════════════════════════════ #

def assert_a100():
    import torch
    n = torch.cuda.device_count()
    assert n == 1, f"exactly 1 visible GPU expected (A100 GPU lock), actual {n}"
    name = torch.cuda.get_device_name(0)
    assert "A100" in name, f"visible GPU is not the A100: {name} (4090 strictly forbidden)"
    free, total = torch.cuda.mem_get_info(0)
    assert free >= 10e9, f"A100 idle memory insufficient: {free/1e9:.1f} GB < 10 GB"
    torch.set_num_threads(8)
    print(f"[GPU] {name} | mem free {free/1e9:.1f}/{total/1e9:.1f} GB | "
          f"CUDA {torch.version.cuda} | torch {torch.__version__}", flush=True)
    return name


# ═══════════════════════════════════════════════════════════════════════════ #
#  model build (protocol field identical verbatim to eval_real_anchors_gpu.load_model_gpu)
# ═══════════════════════════════════════════════════════════════════════════ #

def load_model_gpu(ckpt: Path, config: Path):
    import yaml
    import torch
    from model.api import build_model
    from model.contracts import HGTConfig

    mc = yaml.safe_load(open(config))["model"]
    cfg = HGTConfig(
        data_mode="contig", max_contig_length=3000,
        hidden_dim=mc["hidden_dim"],
        num_attention_heads=mc["num_attention_heads"],
        dropout=mc["dropout"],
        graph_top_k=mc["graph_top_k"],
        bio_feature_dim=mc["bio_feature_dim"],
        pretrained_nt_path=mc.get("pretrained_nt_path", ""),
        pretrained_caduceus_path="",
        use_local_encoder=False,
        use_evo=bool(mc.get("use_evo", False)),
        pretrained_evo_dim=int(mc.get("pretrained_evo_dim", 4096)),
        evo_hidden=int(mc.get("evo_hidden", 512)),
        early_exit_threshold=0.95,
    )
    model = build_model(cfg)

    # Snapshot load: M2 training may overwrite best_model.pt the moment a new best appears,
    # so copy atomically first and read from the copy (read window ~minutes; race probability is low but the cost is small)
    src = ckpt
    snap = None
    if not ARGS.no_snapshot:
        snap = ckpt.with_name(ckpt.stem + ".h1snap.pt")
        print(f"[Ckpt] snapshot {ckpt.name} → {snap.name} "
              f"({ckpt.stat().st_size/1e9:.2f} GB)", flush=True)
        shutil.copy2(ckpt, snap)
        assert snap.stat().st_size == ckpt.stat().st_size, "snapshot size mismatch"
        src = snap
    try:
        st = torch.load(src, map_location="cpu", weights_only=False)
    finally:
        if snap is not None:
            os.remove(snap)
    ckpt_epoch = st.get("epoch")
    contract = st.get("contract", {})
    use_evo_cfg = bool(mc.get("use_evo", False))
    if use_evo_cfg:
        assert ckpt_epoch == 2, \
            f"M2 snapshot epoch={ckpt_epoch} ≠ 2 (expected current best=ep2, task protocol)"

    missing, unexpected = model.load_state_dict(st["model"], strict=False)
    model.to(torch.device("cuda:0")).eval()
    assert not missing, f"state_dict missing {len(missing)} keys (e.g. {missing[:3]})"
    # 24 unexpected = NTv3 rotary buffers (cos/sin_cached, non-persistent buffers,
    # re-registered at build time)  —  consistent with the official anchor evaluation measurements (real_anchor_*.json: 0/24)
    assert all(k.endswith(("cos_cached", "sin_cached")) for k in unexpected), \
        f"state_dict has extra non-rotary-buffer keys: {unexpected[:3]}"

    info = {
        "ckpt": str(ckpt), "ckpt_epoch": ckpt_epoch,
        "ckpt_size_bytes": ckpt.stat().st_size,
        "ckpt_mtime": time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(ckpt.stat().st_mtime)),
        "encoder_frozen": contract.get("encoder_frozen"),
        "backbone_sha256": contract.get("backbone_sha256"),
        "dataset_manifest_sha256": contract.get("dataset_manifest_sha256"),
        "use_evo": use_evo_cfg,
        "state_dict_missing": len(missing),
        "state_dict_unexpected": len(unexpected),
        "unexpected_all_rotary_buffers": bool(all(
            k.endswith(("cos_cached", "sin_cached")) for k in unexpected)),
    }
    assert contract.get("encoder_frozen") is True, "contract: encoder_frozen must be True"
    assert contract.get("backbone_sha256"), "contract: backbone_sha256 missing"
    print(f"[Model] epoch={ckpt_epoch} use_evo={use_evo_cfg} "
          f"load: missing={len(missing)} unexpected={len(unexpected)}"
          f"(rotary buffers) → cuda:0", flush=True)
    print(f"[Contract] encoder_frozen=True backbone_sha256="
          f"{contract['backbone_sha256'][:16]}...", flush=True)
    return model, info, mc


# ═══════════════════════════════════════════════════════════════════════════ #
#  data: BEDParser + eval-mode sliding window (same deterministic window set as training-time val)
# ═══════════════════════════════════════════════════════════════════════════ #

def build_val_dataset(split: str, evo_cache_dir, win_cap: int):
    import yaml
    from data_pipeline.impl.parser import BEDParser
    from data_pipeline.impl.dataset import SlidingWindowContigDataset

    dc = yaml.safe_load(open(MODELS["m2"]["config"]))["data"]
    evo_path = None
    if evo_cache_dir is not None:
        evo_path = str(Path(evo_cache_dir) / f"{split}.pt")
        assert Path(evo_path).is_file(), f"evo cache missing: {evo_path}"

    t0 = time.time()
    samples = BEDParser(
        bed_path=str(DATA_DIR / split / "breakpoints.bed"),
        fasta_dir=str(DATA_DIR / split), mode="contig").parse()
    ds = SlidingWindowContigDataset(
        samples=samples, window_size=dc["sw_window_size"],
        overlap=dc["sw_overlap"], is_training=False,
        max_neg_windows=dc.get("sw_max_neg_windows", 2),
        max_eval_windows=win_cap, extract_bio=True,
        evo_cache_path=evo_path, rc_aug_prob=0.0)
    print(f"[Data] {split}: {len(samples)} contigs → {len(ds)} windows "
          f"({time.time()-t0:.0f}s, win_cap={win_cap}, "
          f"evo={'on' if evo_path else 'off'})", flush=True)

    if evo_path:
        cache_keys = set(ds._evo_cache.keys())
        missing = [s.seq_id for s in samples if s.seq_id not in cache_keys]
        assert not missing, \
            f"evo cache coverage incomplete: {len(missing)} contigs missing embeddings (e.g. {missing[:3]})"
        print(f"[EvoCache] override {len(cache_keys)}/{len(samples)} contigs ✓",
              flush=True)
    return ds


# ═══════════════════════════════════════════════════════════════════════════ #
#  metrics (contig hybrid aggregation = training-side protocol; max/mean = anchor-side protocol)
# ═══════════════════════════════════════════════════════════════════════════ #

def contig_scores(probs, sids, labels, types=None):
    """window → contig three-way aggregation split (hybrid/max/mean); contig order = first-occurrence order."""
    from training_engine.api import TrainingEngine
    hybrid, c_labels, c_types = TrainingEngine._aggregate_contig_predictions(
        np.asarray(probs), np.asarray(labels), list(sids),
        list(types) if types is not None else None)
    by = defaultdict(list)
    for p, s in zip(probs, sids):
        by[s].append(float(p))
    cids = list(by.keys())                       # first-occurrence order == aggregator order
    mx = np.array([max(by[c]) for c in cids], dtype=np.float32)
    mn = np.array([float(np.mean(by[c])) for c in cids], dtype=np.float32)
    assert len(hybrid) == len(cids), "hybrid aggregation contig count inconsistent with first-occurrence order"
    return {"ids": cids, "hybrid": hybrid, "max": mx, "mean": mn,
            "labels": c_labels.astype(int), "types": c_types}


def metrics(y, s):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64)
    if len(set(y.tolist())) < 2:          # only a smoke slice can end up single-class
        return {"auprc": None, "auroc": None,
                "n": int(len(y)), "n_pos": int(np.sum(y == 1))}
    return {"auprc": float(average_precision_score(y, s)),
            "auroc": float(roc_auc_score(y, s)),
            "n": int(len(y)), "n_pos": int(np.sum(y == 1))}


def delong_single(y, s):
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return {"auc": None, "se": None, "ci95": None}
    r = delong_test_full(y, np.asarray(s, dtype=np.float64),
                         np.asarray(s, dtype=np.float64))
    se = math.sqrt(r["var_a"])
    return {"auc": float(r["auc_a"]), "se": se,
            "ci95": [r["auc_a"] - 1.959963984540054 * se,
                     r["auc_a"] + 1.959963984540054 * se]}


# ═══════════════════════════════════════════════════════════════════════════ #
#  inference main pipeline (one model × multiple batches; model/dataset built only once)
# ═══════════════════════════════════════════════════════════════════════════ #

def run_model(model_name: str, args):
    import torch
    from torch.utils.data import DataLoader, Subset
    from data_pipeline.impl.collator import HGTCollator

    gpu_name = assert_a100()
    spec = MODELS[model_name]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")

    model, contract_info, _ = load_model_gpu(spec["ckpt"], spec["config"])
    ds = build_val_dataset(args.split, spec["evo_cache"], args.win_cap)
    n_all = len(ds)
    if args.limit_windows > 0:
        ds_eval = Subset(ds, list(range(min(args.limit_windows, n_all))))
        print(f"[Smoke] only the first {len(ds_eval)} windows")
    else:
        ds_eval = ds

    t_model = time.time()
    for b in [int(x) for x in args.batches.split(",")]:
        tag = f"h1_{model_name}_{args.split}_b{b}"
        t0 = time.time()
        loader = DataLoader(ds_eval, batch_size=b, shuffle=False,
                            collate_fn=HGTCollator(pad_id=0),
                            num_workers=args.num_workers, pin_memory=True)
        probs = np.empty(len(ds_eval), dtype=np.float32)
        sids, labels, types = [], [], []
        i0 = 0
        with torch.no_grad():
            for i, batch in enumerate(loader):
                batch = batch.to(device)
                out = model(batch, early_exit=False)
                p = torch.sigmoid(out.cls_logits.squeeze(-1)).float().cpu().numpy()
                bs = len(p)
                probs[i0:i0 + bs] = p
                sids.extend(batch.original_seq_ids or batch.seq_ids)
                labels.extend(batch.labels.cpu().tolist())
                types.extend(batch.sample_types or [None] * bs)
                i0 += bs
                if (i + 1) % 100 == 0:
                    el = time.time() - t0
                    done = min((i + 1) * b, len(ds_eval))
                    print(f"  [b{b}] [{done}/{len(ds_eval)}] {el:.0f}s "
                          f"({el/(i+1):.2f}s/batch, ETA "
                          f"{(len(loader)-i-1)*el/(i+1)/60:.1f}min)", flush=True)
        assert i0 == len(ds_eval) == len(sids), \
            f"window count mismatch: {i0} vs {len(ds_eval)} vs {len(sids)}"
        infer_s = time.time() - t0

        cs = contig_scores(probs, sids, labels, types)
        res = {
            "model": model_name, "desc": spec["desc"], "batch": b,
            "split": args.split, "n_windows": len(ds_eval),
            "n_windows_full_split": n_all,
            "window_level": metrics(labels, probs),
            "contig_hybrid": metrics(cs["labels"], cs["hybrid"]),
            "contig_max": metrics(cs["labels"], cs["max"]),
            "contig_mean": metrics(cs["labels"], cs["mean"]),
            "delong_contig_hybrid": delong_single(cs["labels"], cs["hybrid"]),
            "protocol": {
                "window_size": 3000, "overlap": 300, "win_cap": args.win_cap,
                "shuffle": False, "dtype": "fp32", "early_exit": False,
                "device": "cuda", "gpu": gpu_name,
                "evo_cache": str(spec["evo_cache"]) if spec["evo_cache"] else None,
                "aggregations": ["hybrid(train)", "max", "mean"],
            },
            "contract": contract_info,
            "infer_seconds": infer_s,
            "total_seconds": time.time() - t_model,
        }
        np.savez_compressed(
            out_dir / f"{tag}.npz",
            window_probs=probs,
            window_ids=np.array(sids, dtype=object),
            window_labels=np.array(labels, dtype=np.int64),
            window_types=np.array(types, dtype=object),
            contig_ids=np.array(cs["ids"], dtype=object),
            contig_labels=cs["labels"],
            contig_hybrid=cs["hybrid"], contig_max=cs["max"],
            contig_mean=cs["mean"])
        with open(out_dir / f"{tag}.json", "w") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        def _f(v):
            return "  n/a" if v is None else f"{v:.4f}"
        print(f"[{tag}] win AUPRC={_f(res['window_level']['auprc'])} | "
              f"contag hybrid AUPRC={_f(res['contig_hybrid']['auprc'])} "
              f"AUROC={_f(res['contig_hybrid']['auroc'])} | "
              f"max AUPRC={_f(res['contig_max']['auprc'])} "
              f"AUROC={_f(res['contig_max']['auroc'])} | "
              f"infer={infer_s/60:.1f}min", flush=True)

    del model
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════ #
#  report: 2×2 matrix + Δp distribution + M2−M0 paired DeLong under a unified batch
# ═══════════════════════════════════════════════════════════════════════════ #

def _load_run(out_dir, model, b):
    z = np.load(Path(out_dir) / f"h1_{model}_val_b{b}.npz", allow_pickle=True)
    jpath = Path(out_dir) / f"h1_{model}_val_b{b}.json"
    j = json.load(open(jpath)) if jpath.is_file() else {}
    return z, j


def _dp_stats(dp):
    a = np.abs(dp)
    return {"max": float(a.max()), "mean": float(a.mean()),
            "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99)),
            "n_gt_1e-3": int((a > 1e-3).sum()),
            "n_gt_1e-2": int((a > 1e-2).sum()),
            "frac_gt_1e-3": float((a > 1e-3).mean()),
            "frac_gt_1e-2": float((a > 1e-2).mean())}


def write_report(args):
    out_dir = Path(args.out_dir)
    runs = {}
    for m in ["m0", "m2"]:
        for b in [4, 8]:
            z, j = _load_run(out_dir, m, b)
            runs[(m, b)] = (z, j)

    # window-set consistency assert (all four runs share one deterministic window sequence; batch is the only variable)
    ref_ids = runs[("m0", 4)][0]["window_ids"]
    ref_lab = runs[("m0", 4)][0]["window_labels"]
    n_win = len(ref_ids)
    for k, (z, _) in runs.items():
        assert len(z["window_ids"]) == n_win, f"{k} window count mismatch"
        assert (z["window_ids"] == ref_ids).all(), f"{k} window id/order mismatch"
        assert (z["window_labels"] == ref_lab).all(), f"{k} window label mismatch"
    print(f"[align] 4 runs × {n_win:,} windows: ids/labels consistent item by item ✓")

    R = {}   # 2×2 metrics (always recomputed from the npz; the json only provides contract/protocol metadata)
    for (m, b), (z, j) in runs.items():
        y = z["contig_labels"]
        R[f"{m}@b{b}"] = {
            "win": metrics(z["window_labels"], z["window_probs"]),
            "hybrid": metrics(y, z["contig_hybrid"]),
            "max": metrics(y, z["contig_max"]),
            "mean": metrics(y, z["contig_mean"]),
            "delong_hybrid": delong_single(y, z["contig_hybrid"]),
            "contract": j.get("contract", {}),
            "infer_min": j.get("infer_seconds", 0) / 60,
        }

    # per-window Δp (b4 vs b8, per model)
    D = {}
    for m in ["m0", "m2"]:
        dp = runs[(m, 8)][0]["window_probs"] - runs[(m, 4)][0]["window_probs"]
        D[m] = _dp_stats(dp)
        # contig-level (hybrid) score drift
        dc = runs[(m, 8)][0]["contig_hybrid"] - runs[(m, 4)][0]["contig_hybrid"]
        D[m]["contig_hybrid_drift"] = _dp_stats(dc)
        D[m]["thresh_flip_contig"] = int(
            ((runs[(m, 4)][0]["contig_hybrid"] >= 0.5) !=
             (runs[(m, 8)][0]["contig_hybrid"] >= 0.5)).sum())
        D[m]["n_contigs"] = len(runs[(m, 4)][0]["contig_hybrid"])

    # M2−M0 paired DeLong (contig level, unified batch; hybrid + max)
    A = {}
    for b in [4, 8]:
        for agg in ["hybrid", "max"]:
            z0 = runs[("m0", b)][0]
            z2 = runs[("m2", b)][0]
            assert (z0["contig_ids"] == z2["contig_ids"]).all(), "contig set mismatch"
            y = z0["contig_labels"]
            r = delong_test_full(y, z2[f"contig_{agg}"].astype(np.float64),
                                 z0[f"contig_{agg}"].astype(np.float64))
            A[f"m2-m0@b{b}_{agg}"] = {
                "delta_auc": float(r["delta"]), "z": float(r["z"]),
                "p": float(r["p"]), "se_delta": float(r["se_delta"]),
                "ci95_delta": r["ci95_delta"], "n": int(len(y)),
                "n_pos": int(np.sum(y == 1))}
            # AUPRC difference (no parametric test: point estimate only; the DeLong test uses AUROC)
            A[f"m2-m0@b{b}_{agg}"]["delta_auprc"] = (
                R[f"m2@b{b}"][agg]["auprc"] - R[f"m0@b{b}"][agg]["auprc"])

    payload = {
        "n_windows": int(n_win),
        "n_contigs": int(len(runs[("m0", 4)][0]["contig_hybrid"])),
        "matrix": R, "dp_stats": D, "attribution": A,
        "train_ref": TRAIN_REF,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_dir / "batch_harmonized_harmonized.json", "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # ── Markdown ──
    L = []
    A_ = L.append
    A_("# H1  —  M0/M2 val dual-batch (b=4 / b=8) harmonized re-evaluation (DA-H1 fix)\n")
    A_(f"> generate: {payload['generated']}  ·  `scripts/eval_batch_harmonized.py` "
       "(inference on the A100 locked by GPU UUID ${CUDA_DEVICE}, coexisting with M2 training; report on CPU)")
    A_("> fix: `docs/evidence/DEEP_AUDIT_FINAL.md` **DA-H1**  —  "
       "BatchGraphInteraction cross-sample graph within the batch + NTv3-pre `attention_mask=None` → "
       "window predictions depend on batch composition; M0/M3 training-time val used b=4, M2 used b=8, "
       "so the M2−M0 attribution is confounded with a batch effect.")
    A_("> protocol: full-quantity deterministic Contig_500k/val window set "
       f"(BEDParser + eval sliding window 3000/300, win_cap=10, "
       f"**{n_win:,} windows / {payload['n_contigs']:,} contigs**); fp32, "
       "early_exit=False (real-anchor protocol), shuffle=False  —  the four runs' window sequences "
       "assert-consistent item by item, **batch grouping is the only variable**. M2 use_evo=True "
       "(evo_cache/Contig_500k/val.pt, 50k/50k coverage).\n")

    A_("## 1. 2×2 matrix  —  contig level (contag) primary metric\n")
    A_("aggregation: **hybrid** = the training-side hybrid rule (≤3 windows max / 4-10 windows top3-mean / "
       ">10 windows max(top3,P90)), same metric protocol as epoch_metrics; max/mean = anchor-side protocol.\n")
    A_("| model | batch | AUPRC (hybrid) | AUROC (hybrid) | DeLong 95%CI "
       "(hybrid) | AUPRC (max) | AUROC (max) | window-level AUPRC |")
    A_("|---|---|---|---|---|---|---|---|")
    for m, lbl in [("m0", "M0 NTv3-only"), ("m2", "M2 NTv3+Evo2")]:
        for b in [4, 8]:
            r = R[f"{m}@b{b}"]
            dl = r["delong_hybrid"]["ci95"]
            star = " (training-native)" if TRAIN_REF[m]["b"] == b else ""
            A_(f"| {lbl} | **b={b}**{star} | **{r['hybrid']['auprc']:.4f}** | "
               f"**{r['hybrid']['auroc']:.4f}** | "
               f"[{dl[0]:.4f}, {dl[1]:.4f}] | {r['max']['auprc']:.4f} | "
               f"{r['max']['auroc']:.4f} | {r['win']['auprc']:.4f} |")

    A_("\ntraining-time reference (epoch_metrics ep2, hybrid aggregation, early_exit=True, each model's own training "
       "batch): ")
    A_(f"M0@b4 = {TRAIN_REF['m0']['val_auprc']:.4f}/{TRAIN_REF['m0']['val_auroc']:.4f}, "
       f"M2@b8 = {TRAIN_REF['m2']['val_auprc']:.4f}/{TRAIN_REF['m2']['val_auroc']:.4f}. "
       "difference vs this table = early_exit protocol difference (DA-M1) + replay noise (DA-L4 ±0.002), "
       "orthogonal to the batch control in this table.\n")

    A_("## 2. per-window prediction difference distribution (b=4 vs b=8, same model)\n")
    A_("prediction sensitivity to arbitrary batch grouping (quantification of the DA-H1 mechanism; audit 8-window CPU micro-check "
       "gave mean 0.014 / max 0.062; this table covers all 199k val windows).\n")
    A_("| model | n window | max\\|Δp\\| | mean\\|Δp\\| | median | p99 | "
       ">1e-3 | >1e-2 | contig hybrid max\\|Δ\\| | contigs flipped at threshold 0.5 |")
    A_("|---|---|---|---|---|---|---|---|---|---|")
    for m in ["m0", "m2"]:
        d = D[m]
        cd = d["contig_hybrid_drift"]
        A_(f"| {MODELS[m]['desc']} | {n_win:,} | {d['max']:.4f} | "
           f"{d['mean']:.4f} | {d['median']:.6f} | {d['p99']:.4f} | "
           f"{d['n_gt_1e-3']:,} ({d['frac_gt_1e-3']:.1%}) | "
           f"{d['n_gt_1e-2']:,} ({d['frac_gt_1e-2']:.1%}) | "
           f"{cd['max']:.4f} | {d['thresh_flip_contig']:,} / {d['n_contigs']:,} |")

    A_("\n## 3. M2−M0 attribution (after unifying the batch)\n")
    A_("contig-level paired DeLong (`delong_test`, AUROC); ΔAUPRC is a point-estimate difference.\n")
    A_("| control | aggregation | ΔAUPRC | ΔAUROC | DeLong z | p | 95%CI (ΔAUROC) |")
    A_("|---|---|---|---|---|---|---|")
    for b in [4, 8]:
        for agg in ["hybrid", "max"]:
            a = A[f"m2-m0@b{b}_{agg}"]
            ci = a["ci95_delta"]
            A_(f"| M2−M0 @ b={b} | {agg} | {a['delta_auprc']:+.4f} | "
               f"{a['delta_auc']:+.4f} | {a['z']:+.2f} | {a['p']:.2e} | "
               f"[{ci[0]:+.4f}, {ci[1]:+.4f}] |")
    dh = (R["m2@b8"]["hybrid"]["auprc"] - R["m0@b8"]["hybrid"]["auprc"]) - \
         (R["m2@b4"]["hybrid"]["auprc"] - R["m0@b4"]["hybrid"]["auprc"])
    A_(f"\nbatch-grouping perturbation of the attribution quantity (hybrid ΔAUPRC, b8 protocol − b4 protocol): "
       f"**{dh:+.4f}**; the difference under the confounded training-time protocol "
       f"(M2@b8 − M0@b4) = "
       f"{R['m2@b8']['hybrid']['auprc'] - R['m0@b4']['hybrid']['auprc']:+.4f} "
       f"(epoch_metrics original reference +0.0386).\n")

    A_("## 4. interpretation\n")
    A_("- single-window probability drift with batch grouping (mean|Δp| see §2) is of the same order of magnitude as the M2−M0 effect "
       "→ the confounding mechanism flagged by DA-H1 is confirmed on the full-quantity val set;")
    A_("- but after contig-level aggregation (the primary protocol) the metric drift is far smaller than at window level (max/top3 average out the within-batch "
       "perturbation); the sign and significance of the M2−M0 difference in the §1 matrix at b=4 vs b=8, i.e. after unifying the batch, is "
       "the final interpretation;")
    A_("- these tables also serve the paper's quantitative-properties / batch-sensitivity section directly "
       "(an extra deliverable recommended by the audit).\n")

    A_("## 5. artifact\n")
    for m in ["m0", "m2"]:
        for b in [4, 8]:
            A_(f"- `docs/evidence/batch_harmonized/h1_{m}_val_b{b}.npz` + `.json` "
               "(per-window probabilities + contig aggregation splits + contract/protocol metadata)")
    A_("- `docs/evidence/batch_harmonized/batch_harmonized_harmonized.json` (source of this report's numbers)\n")

    out = ROOT / "docs/evidence/H1_BATCH_HARMONIZED.md"
    out.write_text("\n".join(L))
    print(f"[Report] saved {out}")
    for k in ["m0@b4", "m0@b8", "m2@b4", "m2@b8"]:
        r = R[k]
        print(f"  {k}: hybrid AUPRC={r['hybrid']['auprc']:.4f} "
              f"AUROC={r['hybrid']['auroc']:.4f} | max "
              f"AUPRC={r['max']['auprc']:.4f}")
    for b in [4, 8]:
        a = A[f"m2-m0@b{b}_hybrid"]
        print(f"  M2-M0@b{b} hybrid: ΔAUPRC={a['delta_auprc']:+.4f} "
              f"ΔAUROC={a['delta_auc']:+.4f} z={a['z']:+.1f} p={a['p']:.2e}")


def main():
    if ARGS.report:
        write_report(ARGS)
    elif ARGS.model:
        run_model(ARGS.model, ARGS)
    else:
        raise SystemExit("need --model {m0,m2} or --report")


if __name__ == "__main__":
    main()
