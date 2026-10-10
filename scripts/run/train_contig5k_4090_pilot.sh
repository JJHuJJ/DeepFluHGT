#!/bin/bash
# scripts/run/train_contig5k_4090_pilot.sh
set -u
cd .
G4090=${GPU_4090_UUID}
CUDA_VISIBLE_DEVICES=$G4090 python -u \
    DeepFluHGT.py train \
    --config configs/contig5k_4090_ntv3only_pilot.yaml \
    --data_dir simulated_data/Contig_5k \
    > logs/contig5k_4090_pilot.log 2>&1
echo "EXIT=$?" >> logs/contig5k_4090_pilot.log
