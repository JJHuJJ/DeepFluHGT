# utils/api.py
"""
DeepFluHGT  —  Utils public interface
==================================
Unified interface:
  - CheckpointManager: checkpoint management
  - EarlyStopping: early-stopping control
  - Logger: log
  - mc_dropout_predict: uncertainty estimation
  - test(): self-test
"""

from __future__ import annotations

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.impl.checkpoint import CheckpointManager
from utils.impl.early_stopping import EarlyStopping
from utils.impl.logging import Logger
from utils.impl.uncertainty import mc_dropout_predict, predictive_entropy, mutual_information


def test() -> bool:
    """Utils self-test."""
    print("=" * 60)
    print("DeepFluHGT  —  Utils Self-Test")
    print("=" * 60)

    results = []
    PASS = "\033[92mPASS\033[0m"
    FAIL = "\033[91mFAIL\033[0m"

    def log(name, passed, detail=""):
        sym = PASS if passed else FAIL
        msg = f"  {sym} {name}"
        if detail:
            msg += f"  [{detail}]"
        print(msg)
        results.append(passed)

    import torch

    # ── 1. CheckpointManager ──
    print("\n[1] CheckpointManager")
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir, max_keep=3)

        # save
        path1 = mgr.save({"model": "v1"}, epoch=1, metric=1.0)
        log("save epoch=1", os.path.exists(path1))

        path2 = mgr.save({"model": "v2"}, epoch=2, metric=0.8, is_best=True)
        log("save epoch=2 (best)", os.path.exists(path2))

        path3 = mgr.save({"model": "v3"}, epoch=3, metric=0.9)
        log("save epoch=3", os.path.exists(path3))

        # Cleanup: max_keep=3, but best_model is not counted
        mgr.save({"model": "v4"}, epoch=4, metric=0.85)
        ckpts = mgr.list_checkpoints()
        log(f"cleanup: ≤3 checkpoints", len(ckpts) <= 3, f"got {len(ckpts)}")

        # Load the best
        best = mgr.load_best()
        log("load_best: metric=0.8", best is not None and best.get("_metric") == 0.8)

        # Load the latest
        latest = mgr.load_latest()
        log("load_latest: epoch=4", latest is not None and latest.get("_epoch") == 4)

    # ── 2. EarlyStopping ──
    print("\n[2] EarlyStopping")
    es = EarlyStopping(patience=3, mode="min")

    log("step 1.0 → not stop", not es.step(1.0))
    log("step 0.9 → not stop", not es.step(0.9))
    log("step 0.95 → not stop (patience=1)", not es.step(0.95))
    log("step 0.94 → not stop (patience=2)", not es.step(0.94))
    log("step 0.93 → stop (patience=3)", es.step(0.93))
    log("should_stop=True", es.should_stop)
    log("counter=3", es.counter == 3)

    # Reset
    es.reset()
    log("reset: counter=0", es.counter == 0)
    log("reset: should_stop=False", not es.should_stop)

    # Max mode
    es_max = EarlyStopping(patience=2, mode="max")
    es_max.step(0.5)
    es_max.step(0.6)  # improved
    log("max mode: counter=0 after improvement", es_max.counter == 0)
    es_max.step(0.55)
    es_max.step(0.54)
    log("max mode: stop after 2 non-improvement", es_max.should_stop)

    # ── 3. Logger ──
    print("\n[3] Logger")
    logger = Logger(backend="print")
    logger.log({"loss": 0.5, "acc": 0.9}, step=1)
    log("print logger: history len=1", len(logger.get_history()) == 1)
    logger.close()

    with tempfile.TemporaryDirectory() as tmpdir:
        flogger = Logger(backend="file", log_dir=tmpdir)
        flogger.log({"loss": 0.3}, step=1)
        flogger.log({"loss": 0.2}, step=2)
        flogger.close()

        log_path = os.path.join(tmpdir, "training.jsonl")
        log("file logger: file exists", os.path.exists(log_path))
        with open(log_path) as f:
            lines = f.readlines()
        log("file logger: 2 lines", len(lines) == 2, f"got {len(lines)}")

    # ── 4. Uncertainty ──
    print("\n[4] Uncertainty")
    import numpy as np

    # MC Dropout (using a simple model)
    from model.contracts import HGTConfig
    from model.api import build_model
    from data_pipeline.contracts import HGTBatch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model_config = HGTConfig(
        data_mode="paired_end",
        _smoke_global_layers=2,
        _smoke_local_layers=2,
        pretrained_nt_path="",
        pretrained_caduceus_path="",
    )
    model = build_model(model_config).to(DEVICE)
    batch = HGTBatch(
        input_ids=torch.randint(0, 10, (2, 303), device=DEVICE),
        attention_mask=torch.ones(2, 303, dtype=torch.long, device=DEVICE),
        bio_features=torch.randn(2, 260, device=DEVICE),
        labels=torch.tensor([1, 0], device=DEVICE),
        breakpoint_targets=torch.zeros(2, 303, device=DEVICE),
        region_targets=torch.zeros(2, 303, device=DEVICE),
        pe_case=torch.zeros(2, dtype=torch.long, device=DEVICE),
        hgt_mode=torch.tensor([0, 3], device=DEVICE),
        n_events=torch.tensor([1, 0], device=DEVICE),
    )

    result = mc_dropout_predict(model, batch, n_forward=3, device=DEVICE)
    log("mc_dropout: mean_probs shape", result["mean_probs"].shape == (2,))
    log("mc_dropout: std_probs shape", result["std_probs"].shape == (2,))
    log("mc_dropout: entropy shape", result["entropy"].shape == (2,))
    log("mc_dropout: all finite", np.all(np.isfinite(result["mean_probs"])))

    # Entropy functions
    mean_p = np.array([0.9, 0.5, 0.1])
    ent = predictive_entropy(mean_p)
    log("predictive_entropy: max at 0.5", ent[1] > ent[0] and ent[1] > ent[2])

    all_p = np.random.rand(5, 3)
    mi = mutual_information(all_p)
    log("mutual_information: shape (3,)", mi.shape == (3,))

    # ── Summary ──
    total_n = len(results)
    passed = sum(results)
    failed = total_n - passed

    print("\n" + "=" * 60)
    print(f"TOTAL: {total_n}  |  PASSED: {passed}  |  FAILED: {failed}")
    print("=" * 60)

    return failed == 0
