#!/usr/bin/env bash
# ============================================================================
# ----------------------------------------------------------------------------
# ============================================================================
set -u
cd .
PY=python
GA100=${GPU_A100_UUID}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FC=V7_FINAL_CANDIDATE_CURRENT_BEST
LOG=logs/final_candidate_eval.log

echo "[FC-eval] START $(date '+%F %T')" | tee -a "$LOG"

run_one() {  # run_one <m2|m3> <M2|M3> <task-args...>
    local key="$1" lab="$2"; shift 2
    local name="$1"; shift
    echo "[FC-eval] ── $lab/$name ── $(date '+%T')" | tee -a "$LOG"
    env CUDA_VISIBLE_DEVICES=$GA100 $PY -u scripts/interim_eval.py \
        --model "$key" --best_now --gpu_uuid "$GA100" \
        --ckpt_override "$FC/08_METRICS/${lab}_ASSUMED_TERMINAL/checkpoint.pt" \
        --expect_epoch_override 7 \
        --out_dir "$FC/09_EVAL/$lab" \
        "$@" 2>&1 | tee -a "logs/fc_eval_${lab}_${name}.log" | tail -2
    echo "[FC-eval]   exit=${PIPESTATUS[0]}" | tee -a "$LOG"
}

for KEYLAB in "m2 M2" "m3 M3"; do
    set -- $KEYLAB; KEY=$1; LAB=$2
    for A in benbow iv4 ice philympics; do
        run_one "$KEY" "$LAB" "anchor_${A}" --task anchors --anchor "$A" --win_cap 10 --batch 8
    done
    run_one "$KEY" "$LAB" "test"   --task test --win_cap 10 --batch 8
    run_one "$KEY" "$LAB" "ood"    --task ood
    for T in D1 D2 D3; do
        run_one "$KEY" "$LAB" "diff_${T}" --task difficulty --tier "$T" --win_cap 10 --batch 8
    done
    env CUDA_VISIBLE_DEVICES="" $PY -u scripts/interim_eval.py --model "$KEY" --best_now \
        --ckpt_override "$FC/08_METRICS/$LAB_ASSUMED_TERMINAL/checkpoint.pt" \
        --expect_epoch_override 7 --out_dir "$FC/09_EVAL/$LAB" \
        --task report --which anchors 2>&1 | tee -a logs/fc_eval_${LAB}_report.log | tail -1
    env CUDA_VISIBLE_DEVICES="" $PY -u scripts/interim_eval.py --model "$KEY" --best_now \
        --ckpt_override "$FC/08_METRICS/${LAB}_ASSUMED_TERMINAL/checkpoint.pt" \
        --expect_epoch_override 7 --out_dir "$FC/09_EVAL/$LAB" \
        --task report --which difficulty 2>&1 | tee -a logs/fc_eval_${LAB}_report.log | tail -1
    echo "[FC-eval] ══ $LAB DONE $(date '+%F %T') ══" | tee -a "$LOG"
done
echo "[FC-eval] ALL DONE $(date '+%F %T')  —  next step: production chain builder (figures/tables/SI/RS/cover/audits)" | tee -a "$LOG"
