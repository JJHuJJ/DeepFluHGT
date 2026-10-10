#!/usr/bin/env python
"""Archive M0 epoch-6 metrics (lost from epoch_metrics.json by an early-stop writer bug).

Audit M18 (docs/evidence/CONSISTENCY_AUDIT_2026-09-02.md): the training engine's
epoch-metrics writer does not persist the final epoch when early stopping fires.
M0 ran epochs 1-6; epoch_metrics.json holds only ep1-5 (last write 08-24 04:02),
while the log carries the complete ep6 record (training summary "Epochs: 6",
early stop at 08-25 ~04:55). Best epoch (2) and all published values unaffected.

This script extracts the ep6 block from the training log and archives it as
docs/evidence/M0_EPOCH6_RECOVERY.json with provenance. Read-only w.r.t. logs and
checkpoints. Run: python scripts/m0_epoch6_recovery.py [--log LOG]
"""
import argparse
import json
import os
import re
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "logs", "M0_contig_ntv3only.log")
OUT = os.path.join(ROOT, "docs", "evidence", "M0_EPOCH6_RECOVERY.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=LOG)
    args = ap.parse_args()

    txt = open(args.log, encoding="utf-8", errors="replace").read()

    # epoch 6 summary line: "[Epoch 6/12] train_loss=... time=..."
    m6 = re.search(r"\[Epoch 6/12\] train_loss=([\d.]+)\s+cls=([\d.]+)\s+bp=([\d.]+)\s+lr=([\de.-]+)\s+time=([\d.]+)s\+([\d.]+)s", txt)
    # val block follows the ep6 summary
    vblock = txt[m6.end(): m6.end() + 1500] if m6 else ""
    mv = re.search(r"auprc=([\d.]+)", vblock)
    ma = re.search(r"auc=([\d.]+)", vblock)
    mf1 = re.search(r"F1@optimal=([\d.]+)", vblock)
    mece = re.search(r"calibration: ECE=([\d.]+)", vblock)
    mstop = re.search(r"\[Early Stop\] (\d+) consecutive epochs without validation AUPRC improvement \(best=([\d.]+)\)", txt)
    mepochs = re.search(r"Epochs:\s*(\d+)", txt)

    assert m6 and mv and mstop, "ep6 block not found  —  log format changed?"

    rec = {
        "purpose": "M0 epoch-6 archival (audit M18): epoch_metrics.json writer drops the final epoch when early stop fires; values below are transcribed from the training log, which is the surviving primary record for ep6.",
        "run": "Contig_500k_ntv3only (M0)",
        "epoch": 6,
        "train_loss": float(m6.group(1)),
        "lr": float(m6.group(4)),
        "train_time_s": float(m6.group(5)),
        "val": {
            "auprc": float(mv.group(1)),
            "auroc": float(ma.group(1)),
            "f1_at_opt": float(mf1.group(1)),
            "ece": float(mece.group(1)),
        },
        "early_stop": {
            "patience_epochs_without_improvement": int(mstop.group(1)),
            "best_val_auprc": float(mstop.group(2)),
        },
        "training_summary_epochs": int(mepochs.group(1)) if mepochs else None,
        "impact": "None on published values: best epoch = 2 (val AUPRC 0.8325); ep6 (0.7436) is below best and below all archived rows' relevance to early-stop corroboration (ECE degrades on non-best epochs, consistent).",
        "provenance": {
            "log": os.path.relpath(args.log, ROOT),
            "log_line_hint": "'[Epoch 6/12] train_loss=1.6980 ... time=81957.2s+7571.5s' followed by the Val block, then '[Early Stop] 4 consecutive epochs'",
            "epoch_metrics_json": "checkpoints/Contig_500k_ntv3only/epoch_metrics.json (ep1-5 only; ep6 absent by writer bug)",
            "last_model_pt_mtime": "2026-08-25 04:55 (postdates epoch_metrics.json 2026-08-24 04:02  —  on-disk footprint of the lost row)",
            "extracted": datetime.now().isoformat(timespec="seconds"),
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(rec, fh, indent=2)
    print(json.dumps(rec, indent=2))
    print(f"\narchived -> {OUT}")


if __name__ == "__main__":
    main()
