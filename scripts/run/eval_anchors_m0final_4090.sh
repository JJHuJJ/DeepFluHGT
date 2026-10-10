#!/bin/bash
# scripts/run/eval_anchors_m0final_4090.sh
set -u
cd .
G4090=${GPU_4090_UUID}   # 4090 UUID (index 1)
PY=python
SMODE=""
[ "${1:-}" = "smoke" ] && SMODE="--limit 5 --tag m0final_smoke"

for A in benbow iv4 ice; do
  echo "[launcher] anchor=$A $(date '+%F %T')"
  CUDA_VISIBLE_DEVICES=$G4090 $PY -u scripts/eval_real_anchors_gpu.py \
      --anchor $A --win_cap 10 --batch 8 $SMODE \
      > logs/eval_anchor_${A}_m0final_gpu.log 2>&1
  echo "  exit=$? $(date '+%F %T')"
done
