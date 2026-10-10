#!/bin/bash
# scripts/run/difficulty_curve_chain.sh
# ============================================================================
#     → docs/evidence/ABLATION_M0.md
#
#
#         > logs/difficulty_curve_chain.log 2>&1 &
# ============================================================================
set -u
cd .

G4090=${GPU_4090_UUID}
PY=python
PYE=python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs docs/evidence evo_cache/difficulty

echo "[chain] ════ difficulty_curve_chain START $(date '+%F %T') pid=$$ ════"
echo "[chain] pre-flight GPU snapshot:"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv || true

wait_gpu_free() {
    local max_wait="${1:-7200}" waited=0 n
    while :; do
        n=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
                --format=csv,noheader 2>/dev/null | grep -c "$G4090" || true)
        [ "$n" -eq 0 ] && return 0
        if [ "$waited" -ge "$max_wait" ]; then
            echo "[chain] wait_gpu_free timeout (${max_wait}s), 4090 still occupied:"
            nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
                --format=csv,noheader | grep "$G4090" || true
            return 1
        fi
        [ "$waited" -eq 0 ] && echo "[chain] 4090 occupied, waiting for release (≤$((max_wait/60))min):" && \
            nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
                --format=csv,noheader | grep "$G4090" || true
        sleep 60; waited=$((waited+60))
    done
}

step() {
    local name="$1" log="$2" tmo="$3"; shift 3
    echo "[chain] $(date '+%F %T') START ${name}"
    if [ "$tmo" -gt 0 ]; then
        timeout "$tmo" "$@" >> "$log" 2>&1
    else
        "$@" >> "$log" 2>&1
    fi
    local rc=$?
    echo "${name}_EXIT=${rc}" >> "$log"
    echo "[chain] $(date '+%F %T') EXIT  ${name} rc=${rc}"
    return 0
}

gstep() {
    local name="$1" log="$2" tmo="$3" guard="$4"; shift 4
    if [ "$guard" != "-" ] && [ -f "$guard" ]; then
        echo "${name}_EXIT=0 (SKIP: $guard already exists)" >> "$log"
        echo "[chain] $(date '+%F %T') SKIP  ${name} (artifact already exists: $guard)"
        return 0
    fi
    wait_gpu_free 7200 || {
        echo "${name}_EXIT=99 (GPU occupied, wait timed out)" >> "$log"
        echo "[chain] $(date '+%F %T') GPUWAIT-FAIL ${name}"
        return 0
    }
    step "$name" "$log" "$tmo" "$@"
}

step STEP_A0_GEN logs/gen_difficulty_chain.log 14400 \
    env RUN=1 bash scripts/run/gen_difficulty_variants.sh all

gstep STEP_A1_EVO_D1 logs/evo_difficulty_D1_gi_length.log 39600 \
    evo_cache/difficulty/D1_gi_length/test.pt \
    env CUDA_VISIBLE_DEVICES=$G4090 $PYE -u scripts/precompute_evo2_embeddings.py \
        --data_dir simulated_data/difficulty_ladder/D1_gi_length \
        --output_dir evo_cache/difficulty/D1_gi_length \
        --splits test --batch_size 4 --token_budget 12288
gstep STEP_A1_EVO_D2 logs/evo_difficulty_D2_donor_sim.log 39600 \
    evo_cache/difficulty/D2_donor_sim/test.pt \
    env CUDA_VISIBLE_DEVICES=$G4090 $PYE -u scripts/precompute_evo2_embeddings.py \
        --data_dir simulated_data/difficulty_ladder/D2_donor_sim \
        --output_dir evo_cache/difficulty/D2_donor_sim \
        --splits test --batch_size 4 --token_budget 12288
gstep STEP_A1_EVO_D3 logs/evo_difficulty_D3_fragmentation.log 18000 \
    evo_cache/difficulty/D3_fragmentation/test.pt \
    env CUDA_VISIBLE_DEVICES=$G4090 $PYE -u scripts/precompute_evo2_embeddings.py \
        --data_dir simulated_data/difficulty_ladder/D3_fragmentation \
        --output_dir evo_cache/difficulty/D3_fragmentation \
        --splits test --batch_size 4 --token_budget 12288

gstep STEP_A2_EVAL_D1 logs/difficulty_eval_D1.log 10800 \
    docs/evidence/difficulty_D1_difficulty_m0.json \
    env CUDA_VISIBLE_DEVICES=$G4090 $PY -u scripts/difficulty_eval.py \
        --tier D1 --win_cap 10 --batch 8
gstep STEP_A2_EVAL_D2 logs/difficulty_eval_D2.log 10800 \
    docs/evidence/difficulty_D2_difficulty_m0.json \
    env CUDA_VISIBLE_DEVICES=$G4090 $PY -u scripts/difficulty_eval.py \
        --tier D2 --win_cap 10 --batch 8
gstep STEP_A2_EVAL_D3 logs/difficulty_eval_D3.log 10800 \
    docs/evidence/difficulty_D3_difficulty_m0.json \
    env CUDA_VISIBLE_DEVICES=$G4090 $PY -u scripts/difficulty_eval.py \
        --tier D3 --win_cap 10 --batch 8

step STEP_A3_REPORT logs/difficulty_curve_report.log 900 \
    env CUDA_VISIBLE_DEVICES="" $PY -u scripts/difficulty_eval.py --report

gstep STEP_B_CROSSTAXA logs/ood_crosstaxa_m0.log 32400 \
    docs/evidence/OOD_CROSSTAXA_M0.md \
    env CUDA_VISIBLE_DEVICES=$G4090 $PY -u scripts/crosstaxa_eval.py \
        --win_cap 10 --batch 8

gstep STEP_C_ABLATION logs/ablation_m0.log 18000 \
    docs/evidence/ABLATION_M0.md \
    env CUDA_VISIBLE_DEVICES=$G4090 $PY -u scripts/ablation_occlusion.py \
        --n_windows 20000 --batch 8

echo "[chain] ════ difficulty_curve_chain DONE $(date '+%F %T') ════"
echo "[chain] artifact: docs/evidence/DIFFICULTY_CURVE_M0.md | OOD_CROSSTAXA_M0.md | ABLATION_M0.md"
echo "DIFFICULTY_CURVE_CHAIN_EXIT=0"
