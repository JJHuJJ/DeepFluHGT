#!/usr/bin/env bash
# ============================================================================
# ----------------------------------------------------------------------------
#
#
#
# ============================================================================
set -u
cd .
GA100=${GPU_A100_UUID}
PY=python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODELS="${1:-m2 m3}"
LOG=logs/terminal_eval_chain.log

echo "[chain] ════ terminal_eval_chain PREFLIGHT $(date '+%F %T') ════" | tee -a "$LOG"
for m in $MODELS; do
    if ! $PY scripts/terminal_eval_preflight.py --model "$m" >> "$LOG" 2>&1; then
        echo "TERMINAL_EVAL_BLOCKED (model=$m) — see $LOG" | tee -a "$LOG"
        exit 1
    fi
done
echo "[chain] all hard gates PASSED — terminal evaluation may proceed" | tee -a "$LOG"

GPU_UUID="${GPU_UUID:-$GA100}"
export GPU_UUID
for m in $MODELS; do
    bash scripts/run/terminal_eval_chain_inner.sh "$m"
done

$PY scripts/verdict_calculator.py 2>&1 | tee -a "$LOG" | tail -6
$PY scripts/final_result_provenance_gate.py 2>&1 | tee -a "$LOG" | tail -3
echo "[chain] ════ DONE $(date '+%F %T') — if ALLOW: run fill_placeholders.py ════" | tee -a "$LOG"
