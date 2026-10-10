#!/bin/bash
# scripts/run/evo2_window_pooling_4090.sh
#          → evo_cache/step0_subset/pooling/emb_blocks_{24,28}_mlp_l3.pt
#          → evo_cache/step0_subset/pooling_ablation.{md,csv}
set -u
cd .
G4090=${GPU_4090_UUID}
PYE=python
PY=python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

echo "[launcher] pre-flight $(date '+%F %T')"; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv

CUDA_VISIBLE_DEVICES=$G4090 $PYE -u scripts/evo2_window_pooling.py --embed \
    --batch 4 --token_budget 12288 \
    > logs/evo2_window_pooling_embed.log 2>&1
echo "POOLING_EMBED_EXIT=$?" >> logs/evo2_window_pooling_embed.log
grep -q "POOLING_EMBED_EXIT=0" logs/evo2_window_pooling_embed.log || { echo "[launcher] embed FAILED"; exit 1; }

# 2) CPU probe + DeLong
$PY -u scripts/evo2_window_pooling.py --probe \
    > logs/evo2_window_pooling_probe.log 2>&1
echo "POOLING_PROBE_EXIT=$?" >> logs/evo2_window_pooling_probe.log
grep -q "POOLING_PROBE_EXIT=0" logs/evo2_window_pooling_probe.log || { echo "[launcher] probe FAILED"; exit 1; }

echo "[launcher] done $(date '+%F %T')"
