#!/usr/bin/env bash
set -u
cd .
PY=python
EVOPY=python
GA100=${GPU_A100_UUID}
pick_gpu() {
  for i in $(seq 1 48); do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GA100" | head -1)
    [ "$FREE" -ge 20000 ] && return 0
    echo "[A12] A100 free=${FREE}MB <20GB, waiting 5min ($i/48)" | tee -a "$LOG"; sleep 300
  done
  return 1
}
LOG=logs/a12_anchor_completion.log
echo "[A12] start $(date '+%F %T')" | tee -a "$LOG"

cat > /tmp/a12_fastas.txt << 'LIST'
benbow	benchmark/literature_gi/repo/test_data/benbow_test_data.fasta
iv4	benchmark/islandviewer4/ALL_MERGED_GENOMIC_ISLANDS.fasta
ice	benchmark/iceberg2/raw_downloads/ICE_seq_experimental.fas
philympics	benchmark/philympics/v7_anchor/v7_paired.fasta
LIST
cat /tmp/a12_fastas.txt | tee -a "$LOG"

mkdir -p evo_cache/anchors
pick_gpu || { echo "[A12] ABORT: A100 has no spare capacity" | tee -a "$LOG"; exit 1; }
while IFS=$'\t' read -r name path; do
  [ "$name" = "ERR" ] && continue
  if [ -f "evo_cache/anchors/${name}.pt" ]; then
    echo "[A12] $name already has a cache, skip" | tee -a "$LOG"
  elif [ -f "$path" ]; then
    echo "[A12] embedding $name: $path" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$GA100 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $EVOPY scripts/precompute_evo2_embeddings.py --batch_size 8 \
      --fasta "$path" --output "evo_cache/anchors/${name}.pt" 2>&1 | tail -2 | tee -a "$LOG"
  elif [ -d "$path" ]; then
    for f in "$path"/*.fasta; do
      echo "[A12] embedding $name: $f" | tee -a "$LOG"
      CUDA_VISIBLE_DEVICES=$GA100 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $EVOPY scripts/precompute_evo2_embeddings.py --batch_size 8 \
        --fasta "$f" --output "evo_cache/anchors/${name}_$(basename $f .fasta).pt" 2>&1 | tail -2 | tee -a "$LOG"
    done
  fi
done < /tmp/a12_fastas.txt

$PY - << 'PYEOF' | tee -a "$LOG"
import torch, glob
merged={}
for p in sorted(glob.glob('evo_cache/anchors/*.pt')):
    d=torch.load(p,map_location='cpu',weights_only=False)
    merged.update(d); print(f"  {p}: {len(d)}")
torch.save(merged,'evo_cache/anchors/merged.pt')
print(f"[A12] merged: {len(merged)} seqs → evo_cache/anchors/merged.pt")
PYEOF
echo "[A12] embedding done $(date '+%F %T')" | tee -a "$LOG"
