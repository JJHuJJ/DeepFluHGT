#!/bin/bash
# scripts/run/evo_full_cache_contig500k.sh
set -u
cd .
A100=${GPU_A100_UUID}
CUDA_VISIBLE_DEVICES=$A100 python -u \
    scripts/precompute_evo2_embeddings.py \
    --data_dir simulated_data/Contig_500k \
    --output_dir evo_cache/Contig_500k \
    >> logs/evo_contig_full_cache.log 2>&1
echo "FULL_CACHE_EXIT=$?" >> logs/evo_contig_full_cache.log
