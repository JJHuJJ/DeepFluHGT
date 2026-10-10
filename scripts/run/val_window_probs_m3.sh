#!/bin/bash
set -euo pipefail
cd .
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=python
$PY scripts/val_window_probs_terminal.py --model m3 \
  --gpu_uuid ${GPU_4090_UUID} --batch 4 --min_free_gb 20 \
  > logs/val_window_probs_m3.log 2>&1
