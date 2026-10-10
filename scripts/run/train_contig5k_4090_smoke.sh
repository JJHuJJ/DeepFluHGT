#!/bin/bash
# scripts/run/train_contig5k_4090_smoke.sh
set -u
cd .
G4090=${GPU_4090_UUID}
CUDA_VISIBLE_DEVICES=$G4090 python -u \
    DeepFluHGT.py train \
    --config configs/contig5k_4090_ntv3only_smoke.yaml \
    --data_dir simulated_data/Contig_5k \
    > logs/contig5k_4090_smoke.log 2>&1
echo "EXIT=$?" >> logs/contig5k_4090_smoke.log
