#!/usr/bin/env bash
# ============================================================================
# ----------------------------------------------------------------------------
# ============================================================================
set -u
cd .
PY=python
G4090=${GPU_4090_UUID}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=logs/pi_review_m2best_eval.log

FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$G4090" 2>/dev/null | head -1 | tr -d ' ')
echo "[m2best] 4090 free=${FREE_MB:-?}MB $(date '+%F %T')" | tee -a "$LOG"
if [ -n "${FREE_MB:-}" ] && [ "$FREE_MB" -lt 18000 ]; then
    echo "[m2best] ABORT: 4090 GPU memory insufficient (<18GB)  —  check for leftover processes" | tee -a "$LOG"
    exit 1
fi

run_ev() {  # run_ev <logname> <args...>
    local ln="$1"; shift
    echo "[m2best] ── $ln ── $(date '+%T')" | tee -a "$LOG"
    env CUDA_VISIBLE_DEVICES=$G4090 $PY -u scripts/interim_eval.py --model m2 \
        --best_now --gpu_uuid "$G4090" "$@" \
        2>&1 | tee -a "logs/pi_review_m2best_${ln}.log" | tail -2
    echo "[m2best]   $ln exit=${PIPESTATUS[0]}" | tee -a "$LOG"
}

for anchor in benbow iv4 ice philympics; do
    run_ev "anchor_${anchor}" --task anchors --anchor "$anchor" --win_cap 10 --batch 8
done
run_ev "test" --task test --win_cap 10 --batch 8
run_ev "ood"  --task ood
for tier in D1 D2 D3; do
    run_ev "diff_${tier}" --task difficulty --tier "$tier" --win_cap 10 --batch 8
done
env CUDA_VISIBLE_DEVICES="" $PY -u scripts/interim_eval.py --model m2 --best_now \
    --task report --which anchors 2>&1 | tee -a logs/pi_review_m2best_report.log | tail -1
env CUDA_VISIBLE_DEVICES="" $PY -u scripts/interim_eval.py --model m2 --best_now \
    --task report --which difficulty 2>&1 | tee -a logs/pi_review_m2best_report.log | tail -1

echo "[m2best] ══ DONE $(date '+%F %T')  —  next step: python scripts/build_pi_review.py ══" | tee -a "$LOG"
