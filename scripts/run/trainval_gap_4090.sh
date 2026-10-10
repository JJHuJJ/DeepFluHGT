#!/bin/bash
# scripts/run/trainval_gap_4090.sh
#   → docs/evidence/TRAINVAL_GAP_M0.{md,csv,png}
set -u
cd .
G4090=${GPU_4090_UUID}
PY=python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs
echo "[launcher] pre-flight $(date '+%F %T')"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv

CUDA_VISIBLE_DEVICES=$G4090 $PY -u scripts/trainval_gap.py --run \
    --batch_size 8 --num_workers 8 \
    > logs/trainval_gap_4090.log 2>&1
echo "TRAINVAL_GAP_EXIT=$?" >> logs/trainval_gap_4090.log

$PY -u scripts/trainval_gap.py --report >> logs/trainval_gap_4090.log 2>&1
echo "TRAINVAL_GAP_REPORT_EXIT=$?" >> logs/trainval_gap_4090.log
echo "[launcher] done $(date '+%F %T')"
