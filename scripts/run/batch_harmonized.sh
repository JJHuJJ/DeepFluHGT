#!/bin/bash
# scripts/run/batch_harmonized_harmonized.sh
# ============================================================================
#   STEP3  --report (CPU): docs/evidence/H1_BATCH_HARMONIZED.md + json
#
#
#         > logs/batch_harmonized_harmonized.log 2>&1 &
# ============================================================================
set -u
cd .

GA100=${GPU_A100_UUID}
PY=python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs docs/evidence/batch_harmonized

echo "[h1] ════ batch-harmonized chain START $(date '+%F %T') pid=$$ ════"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv || true

gpu_gate() {
    local free_mb
    free_mb=$(nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader,nounits \
                2>/dev/null | grep "^$GA100" | head -1 | cut -d, -f2 | tr -d ' ')
    [ -n "$free_mb" ] && [ "$free_mb" -ge 10000 ] 2>/dev/null && return 0
    echo "[h1] $(date '+%F %T') A100 idle ${free_mb:-?}MB < 10GB, retry in 60s"
    return 1
}

step() {
    local name="$1" guard="$2"; shift 2
    if [ "$guard" != "-" ] && [ -e "$guard" ]; then
        echo "[h1] $(date '+%F %T') SKIP ${name} (artifact already exists: $guard)"
        return 0
    fi
    echo "[h1] $(date '+%F %T') START ${name}"
    until gpu_gate; do sleep 60; done
    "$@"
    local rc=$?
    echo "${name}_EXIT=${rc}" >> logs/batch_harmonized_harmonized.log.inner
    echo "[h1] $(date '+%F %T') EXIT  ${name} rc=${rc}"
    return 0
}

step STEP1_M0 docs/evidence/batch_harmonized/bh_m0_val_b8.json \
    env CUDA_VISIBLE_DEVICES=$GA100 $PY -u scripts/eval_batch_harmonized.py \
        --model m0 --batches 4,8 --win_cap 10

step STEP2_M2 docs/evidence/batch_harmonized/bh_m2_val_b8.json \
    env CUDA_VISIBLE_DEVICES=$GA100 $PY -u scripts/eval_batch_harmonized.py \
        --model m2 --batches 4,8 --win_cap 10

step STEP3_REPORT docs/evidence/H1_BATCH_HARMONIZED.md \
    env CUDA_VISIBLE_DEVICES="" $PY -u scripts/eval_batch_harmonized.py --report

echo "[h1] ════ batch-harmonized chain DONE $(date '+%F %T') ════"
echo "H1_BATCH_HARMONIZED_EXIT=0"
