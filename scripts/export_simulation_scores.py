#!/usr/bin/env python3
"""export_simulation_scores.py — re-run simulation eval, save per-contig scores + labels.

Uses the exact same checkpoint/config as P0_native_eval_v2, but captures the
per-contig mean-pool scores + labels so we can compute ROC/PR/calibration/bootstrap CI.

Does NOT modify DeepFluHGT.py. Monkeypatches evaluation.api.evaluate to save scores.

Reproducibility:
    CUDA_VISIBLE_DEVICES=1 python scripts/export_simulation_scores.py
"""
import sys, os, json, shutil
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "output" / "simulation_eval"
OUT.mkdir(parents=True, exist_ok=True)

# ═══ Monkeypatch evaluation.api.evaluate to capture scores ═══
import evaluation.api as eval_api
_original_evaluate = eval_api.evaluate

def patched_evaluate(model, dataloader, config=None, loss_fn=None):
    """Wrap original evaluate to capture all_probs + all_labels."""
    result = _original_evaluate(model, dataloader, config=config, loss_fn=loss_fn)

    # Capture from the internal lists built during evaluate
    # The original builds all_probs/all_labels inside, let's re-extract from result
    print("[export] EvalResult captured. Extracting scores...")

    # EvalResult should have metrics dict; let's check what's available
    if hasattr(result, 'metrics') and isinstance(result.metrics, dict):
        probs = result.metrics.get("_all_probs")
        labels = result.metrics.get("_all_labels")
    elif hasattr(result, '_all_probs'):
        probs = result._all_probs
        labels = result._all_labels
    else:
        # Try dict-like access
        probs = result.get("_all_probs") if hasattr(result, 'get') else None
        labels = result.get("_all_labels") if hasattr(result, 'get') else None

    if probs is not None and labels is not None:
        scores = np.array(probs)
        labs = np.array(labels)
        np.save(OUT / "scores.npy", scores)
        np.save(OUT / "labels.npy", labs)
        print(f"[export] Saved {len(labs)} scores ({int(labs.sum())} pos, {int((1-labs).sum())} neg)")
        print(f"[export] Mean-pool AUROC = {result.auroc:.4f}" if hasattr(result, 'auroc') else "")

    return result

eval_api.evaluate = patched_evaluate
print("[export] Monkeypatch installed on evaluation.api.evaluate")

# ═══ Now run the standard eval ═══
os.chdir(ROOT)

# Build the same command as P0
sys.argv = [
    "DeepFluHGT.py", "eval",
    "--config", "configs/contig_500k.yaml",
    "--data_dir", "simulated_data/Contig_500k",
    "--checkpoint", "checkpoints/Contig_500k_bf16/best_model.pt",
    "--mode", "contig",
    "--batch_size", "8",
    "--no_early_exit",
]

print(f"[export] Running eval: {' '.join(sys.argv[1:])}")
print(f"[export] Output dir: {OUT}")
print()

# Execute
exec(open(ROOT / "DeepFluHGT.py").read())
