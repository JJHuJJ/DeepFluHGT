#!/bin/bash
# scripts/run/eval_philympics_m0final_4090.sh
set -u
cd .
G4090=${GPU_4090_UUID}
CUDA_VISIBLE_DEVICES=$G4090 python -u \
    scripts/eval_real_anchors_gpu.py \
    --anchor philympics --win_cap 10 --batch 8 \
    > logs/eval_anchor_philympics_m0final_gpu.log 2>&1
echo "PHIL_EVAL_EXIT=$?" >> logs/eval_anchor_philympics_m0final_gpu.log
