#!/bin/bash
# scripts/run/train_m0_4090.sh
set -u
M0_CKPT=checkpoints/Contig_500k_ntv3only/last_model.pt
M0_RESUME=""
[ -f "$M0_CKPT" ] && M0_RESUME="--resume $M0_CKPT" && echo "[launcher] resume from $M0_CKPT"
cd .
G4090=${GPU_4090_UUID}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=$G4090 python -u \
    DeepFluHGT.py train \
    --config configs/contig_500k_ntv3only_4090.yaml \
    --data_dir simulated_data/Contig_500k ${M0_RESUME} \
    > logs/M0_contig_ntv3only.log 2>&1
echo "M0_EXIT=$?" >> logs/M0_contig_ntv3only.log
