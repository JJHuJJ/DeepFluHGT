#!/bin/bash
# scripts/run/evo2_layer_scan_4090.sh
set -u
cd .
G4090=${GPU_4090_UUID}
PYE=python
PY=python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=$G4090 $PYE -u scripts/evo2_layer_scan.py --embed \
    --batch 4 --token_budget 12288 --tap mlp_l3 \
    > logs/evo2_layer_scan_embed.log 2>&1
echo "LAYER_SCAN_EMBED_EXIT=$?" >> logs/evo2_layer_scan_embed.log
grep -q "LAYER_SCAN_EMBED_EXIT=0" logs/evo2_layer_scan_embed.log || exit 1

CUDA_VISIBLE_DEVICES=$G4090 $PYE -u scripts/evo2_layer_scan.py --embed \
    --batch 4 --token_budget 12288 --tap block \
    > logs/evo2_layer_scan_embed_blocktap.log 2>&1
echo "LAYER_SCAN_BLOCKTAP_EXIT=$?" >> logs/evo2_layer_scan_embed_blocktap.log
grep -q "LAYER_SCAN_BLOCKTAP_EXIT=0" logs/evo2_layer_scan_embed_blocktap.log || exit 1

$PY -u scripts/evo2_layer_scan_probe.py \
    > logs/evo2_layer_scan_probe.log 2>&1
echo "LAYER_SCAN_PROBE_EXIT=$?" >> logs/evo2_layer_scan_probe.log
