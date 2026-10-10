#!/bin/bash
# scripts/run/train_m2_a100.sh
# ============================================================================
#
set -u
cd .

PY=python
EVO_DIR=evo_cache/Contig_500k
BED_TRAIN=simulated_data/Contig_500k/train/breakpoints.bed
LOG=logs/M2_contig_fusion.log

echo "[M2-preflight] start $(date '+%F %T')"

FAIL=0
for split in train val test; do
    if [ ! -f "$EVO_DIR/$split.pt" ]; then
        echo "[M2-preflight][FAIL] (a) missing $EVO_DIR/$split.pt (H-02: risk of silently disabling Evo2, refuse to start)"
        FAIL=1
    else
        echo "[M2-preflight][OK]   (a) $EVO_DIR/$split.pt ($(du -h "$EVO_DIR/$split.pt" | cut -f1))"
    fi
done
if ls "$EVO_DIR"/*.partial.pt >/dev/null 2>&1; then
    echo "[M2-preflight][FAIL] (a) leftover partial file (cache still generating):"
    ls -lh "$EVO_DIR"/*.partial.pt
    FAIL=1
else
    echo "[M2-preflight][OK]   (a) no *.partial.pt leftovers"
fi

if [ ! -f "$EVO_DIR/cache_manifest.json" ]; then
    echo "[M2-preflight][FAIL] (b) $EVO_DIR/cache_manifest.json does not exist (cache job not finished or monitor not yet run)"
    FAIL=1
else
    COMPLETED=$($PY -c "import json;print(json.load(open('$EVO_DIR/cache_manifest.json')).get('completed'))" 2>/dev/null || echo ERR)
    if [ "$COMPLETED" != "True" ]; then
        echo "[M2-preflight][FAIL] (b) cache_manifest.json completed=$COMPLETED (expected True; check FULL_CACHE_EXIT for all three splits)"
        FAIL=1
    else
        echo "[M2-preflight][OK]   (b) cache_manifest.json completed=True"
    fi
fi

if [ $FAIL -eq 0 ]; then
    $PY - "$EVO_DIR/train.pt" "$BED_TRAIN" <<'PYEOF'
import random, sys, torch
cache_path, bed_path = sys.argv[1], sys.argv[2]
ids = set()
with open(bed_path) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.add(line.split("\t")[0])
sample = sorted(random.Random(42).sample(sorted(ids), min(1000, len(ids))))
cache = torch.load(cache_path, map_location="cpu", weights_only=False)
keys = set(cache.keys())
missing = [s for s in sample if s not in keys]
print(f"[M2-preflight] (c) BED unique seq_id={len(ids)}, sampled={len(sample)}, "
      f"train.pt vector count={len(keys)}")
if missing:
    print(f"[M2-preflight][FAIL] (c) coverage {len(sample)-len(missing)}/{len(sample)} "
          f"({100.0*(len(sample)-len(missing))/len(sample):.2f}%), missing examples "
          f"(first 10): {missing[:10]}  (M-05: missing samples would silently fall back to zero vectors)")
    sys.exit(1)
print("[M2-preflight][OK]   (c) sampled coverage 100% (1000/1000)")
PYEOF
    if [ $? -ne 0 ]; then
        echo "[M2-preflight][FAIL] (c) train.pt key coverage sampling check did not pass"
        FAIL=1
    fi
else
    echo "[M2-preflight][SKIP] (c) a prior assert already failed, skip the coverage check"
fi

if [ -f scripts/batch_sweep.py ]; then
    echo "[M2-preflight][OK]   (d) scripts/batch_sweep.py exists (its throughput calibration can be used once M2 starts)"
else
    echo "[M2-preflight][FAIL] (d) scripts/batch_sweep.py does not exist"
    FAIL=1
fi

if [ $FAIL -ne 0 ]; then
    echo "[M2-preflight] FAIL  —  —  hard guardrails not passed, refusing to start M2 (H-02)"
    exit 1
fi
echo "[M2-preflight] PASS  —  —  all asserts passed $(date '+%F %T'), starting M2 training"

# ════════════════════════════════════════════════════════════════════════════ #
# ════════════════════════════════════════════════════════════════════════════ #
M2_CKPT=checkpoints/Contig_500k/last_model.pt   # = configs/contig_500k.yaml checkpoint_dir
M2_RESUME=""
[ -f "$M2_CKPT" ] && M2_RESUME="--resume $M2_CKPT" && echo "[launcher] resume from $M2_CKPT"

A100=${GPU_A100_UUID}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=$A100 $PY -u \
    DeepFluHGT.py train \
    --config configs/contig_500k.yaml \
    --data_dir simulated_data/Contig_500k ${M2_RESUME} \
    > $LOG 2>&1
echo "M2_EXIT=$?" >> $LOG
