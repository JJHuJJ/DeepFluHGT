#!/bin/bash
#       docs/evidence/COMPOSITION_CEILING_SIM.md + .EXIT
set -u
cd .

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

PY=python
LOG=logs/composition_ceiling.log

echo "[launcher] $(date '+%F %T') composition_ceiling start (CPU, ≤48 thread)" >> "$LOG"
nohup "$PY" -u scripts/composition_ceiling.py \
    --n-perm 100 --ridge-n-perm 200 --mlp-n-perm 20 \
    --n-boot 1000 --kmer-jobs 32 --mlp-threads 32 \
    --null-jobs 4 --null-threads 8 \
    >> "$LOG" 2>&1 &
PID=$!
echo "[launcher] PID=$PID, log=$LOG (EXIT marker: docs/evidence/COMPOSITION_CEILING_SIM.EXIT)"
echo "$PID" > logs/composition_ceiling.pid
