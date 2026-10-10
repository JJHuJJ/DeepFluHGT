#!/usr/bin/env bash
# ============================================================================
# ----------------------------------------------------------------------------
#
#
# ============================================================================
set -u
cd .
PY=python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M="${1:?usage: terminal_eval_chain_inner.sh <m2|m3>}"
GPU_UUID="${GPU_UUID:-${GPU_A100_UUID}}"
LOG=logs/terminal_eval_chain.log

echo "[inner] ══ model=$M gpu=${GPU_UUID:0:16}… $(date '+%F %T') ══" | tee -a "$LOG"

for anchor in benbow iv4 ice philympics; do
    env CUDA_VISIBLE_DEVICES=$GPU_UUID $PY -u scripts/interim_eval.py --model "$M" --terminal \
        --gpu_uuid "$GPU_UUID" --task anchors --anchor "$anchor" --win_cap 10 --batch 8 \
        2>&1 | tee -a "logs/terminal_${M}_anchor_${anchor}.log"
done
env CUDA_VISIBLE_DEVICES=$GPU_UUID $PY -u scripts/interim_eval.py --model "$M" --terminal \
    --gpu_uuid "$GPU_UUID" --task test --win_cap 10 --batch 8 2>&1 | tee -a "logs/terminal_${M}_test.log"
env CUDA_VISIBLE_DEVICES=$GPU_UUID $PY -u scripts/interim_eval.py --model "$M" --terminal \
    --gpu_uuid "$GPU_UUID" --task ood 2>&1 | tee -a "logs/terminal_${M}_ood.log"
for tier in D1 D2 D3; do
    env CUDA_VISIBLE_DEVICES=$GPU_UUID $PY -u scripts/interim_eval.py --model "$M" --terminal \
        --gpu_uuid "$GPU_UUID" --task difficulty --tier "$tier" --win_cap 10 --batch 8 \
        2>&1 | tee -a "logs/terminal_${M}_diff_${tier}.log"
done
env CUDA_VISIBLE_DEVICES="" $PY -u scripts/interim_eval.py --model "$M" \
    --terminal --task report --which anchors 2>&1 | tee -a "logs/terminal_${M}_report.log"
env CUDA_VISIBLE_DEVICES="" $PY -u scripts/interim_eval.py --model "$M" \
    --terminal --task report --which difficulty 2>&1 | tee -a "logs/terminal_${M}_report.log"

echo "[inner] ══ model=$M DONE $(date '+%F %T') ══" | tee -a "$LOG"
