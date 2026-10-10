#!/bin/bash
set -euo pipefail
cd .
PY=python
$PY scripts/val_window_probs_terminal.py --model m2 \
  --gpu_uuid ${GPU_A100_UUID} --batch 8 \
  > logs/val_window_probs_m2.log 2>&1
