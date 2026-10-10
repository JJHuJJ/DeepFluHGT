#!/bin/bash
set -u
cd .
PY=python
SRC=evo_cache/Contig_500k; DST=evo_cache/Contig_500k_shuffle
G4090=${GPU_4090_UUID}
echo "[watch] start $(date)" >> logs/M3_watch.log
while true; do
  if [ -f $SRC/val.pt ] && [ ! -f $DST/val.pt ]; then
    $PY scripts/build_shuffle_cache.py $SRC/val.pt $DST/val.pt >> logs/M3_watch.log 2>&1
  fi
  if [ -f $SRC/test.pt ] && [ ! -f $DST/test.pt ]; then
    $PY scripts/build_shuffle_cache.py $SRC/test.pt $DST/test.pt >> logs/M3_watch.log 2>&1
  fi
  if [ -f $DST/val.pt ] && [ -f $DST/test.pt ]; then
    N=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G4090)
    if [ "$N" -gt 1000 ]; then echo "[watch] 4090 busy (${N}MiB), waiting for the chain task to yield $(date)" >> logs/M3_watch.log; sleep 600; continue; fi
    echo "[watch] all three ready, starting M3 $(date)" >> logs/M3_watch.log
    CUDA_VISIBLE_DEVICES=$G4090 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup $PY -u DeepFluHGT.py train \
      --config configs/contig_500k_m3.yaml \
      --data_dir simulated_data/Contig_500k \
      > logs/M3_contig_shuffle.log 2>&1
    echo "M3_EXIT=$?" >> logs/M3_contig_shuffle.log
    break
  fi
  sleep 600
done
