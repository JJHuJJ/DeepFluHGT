#!/bin/bash
set -u
cd .
PY=python
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=44 MKL_NUM_THREADS=44 OPENBLAS_NUM_THREADS=44

echo "=== [1/4] attention (40 POS window) $(date) ==="
$PY scripts/interp_m0.py --part attn --n 40 || echo "ATTN_FAIL"

echo "=== [2/4] tsne (500 val window) $(date) ==="
$PY scripts/interp_m0.py --part tsne --n 500 || echo "TSNE_FAIL"

echo "=== [3/4] integrated gradients (30 POS window, 32 steps) $(date) ==="
$PY scripts/interp_m0.py --part ig --n 30 --steps 32 || echo "IG_FAIL"

echo "=== [4/4] uncertainty-artifact corr (all val 199,438 window) $(date) ==="
$PY scripts/interp_m0.py --part unc || echo "UNC_FAIL"

echo "completed=$(date '+%Y-%m-%d %H:%M:%S')" > docs/evidence/INTERP_M0.EXIT
echo "=== chain done $(date) ==="
