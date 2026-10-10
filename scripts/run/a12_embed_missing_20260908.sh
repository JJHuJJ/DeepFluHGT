#!/usr/bin/env bash
set -u
cd .
EVOPY=python
GA100=${GPU_A100_UUID}
LOG=logs/a12_embed_missing_20260908.log
MD=evo_cache/anchors/_missing_20260908

echo "[embed-missing] start $(date '+%F %T')" | tee -a "$LOG"
for i in $(seq 1 24); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GA100" | head -1)
  [ "$FREE" -ge 20000 ] && break
  echo "[embed-missing] A100 free=${FREE}MB <20GB, waiting 5min ($i/24)" | tee -a "$LOG"; sleep 300
done
[ "$FREE" -ge 20000 ] || { echo "[embed-missing] ABORT: A100 has no spare capacity" | tee -a "$LOG"; exit 1; }

for A in iv4 ice; do
  if [ -f "$MD/${A}_missing.pt" ]; then
    echo "[embed-missing] $A already has a backfilled embedding artifact, skip" | tee -a "$LOG"; continue
  fi
  echo "[embed-missing] embedding $A ($(date '+%F %T'))" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$GA100 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $EVOPY scripts/precompute_evo2_embeddings.py --batch_size 8 \
      --fasta "$MD/${A}_missing.fasta" --output "$MD/${A}_missing.pt" 2>&1 | tail -3 | tee -a "$LOG"
done
echo "[embed-missing] done $(date '+%F %T')" | tee -a "$LOG"
