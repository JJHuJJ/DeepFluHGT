#!/usr/bin/env bash
# ============================================================================
# ----------------------------------------------------------------------------
#   3. GPU_UUID=<4090> bash scripts/run/terminal_eval_chain.sh m3
#
# ============================================================================
set -u
cd .
PY=python
G4090=${GPU_4090_UUID}
M3_PATTERN="DeepFluHGT.py train --config configs/contig_500k_m3.yaml"
LOG=logs/pi_review_autopilot.log
STATUS=PI_REVIEW_M2_BEST/AUTOPILOT_STATUS.md
PIDFILE=logs/pi_review_autopilot.pid

echo $$ > "$PIDFILE"
mkdir -p PI_REVIEW_M2_BEST

stamp() { date '+%F %T'; }
log()   { echo "[autopilot $(stamp)] $*" | tee -a "$LOG"; }
stage() {
    { echo "## $(stamp)  —  $1"; echo "${2:-}"; echo; } >> "$STATUS"
    log "$1 ${2:-}"
}

echo "
echo "start: $(stamp)  ·  pid=$$  ·  polling M3 (pattern=$M3_PATTERN)" >> "$STATUS"
log "started, pid=$$"

stage "STAGE-1 WAIT FOR M3 EXIT" "M3 decisive round: ep7 eval expected ~05:30; if a new best appears, training continues and this script keeps waiting"
WAITED=0
while pgrep -f "$M3_PATTERN" > /dev/null 2>&1; do
    sleep 300
    WAITED=$((WAITED + 1))
    if [ $((WAITED % 12)) -eq 0 ]; then log "still waiting for M3 exit (${WAITED}x5min)"; fi
    if [ "$WAITED" -ge 576 ]; then
        stage "TIMEOUT" "M3 did not exit within 48h (still hitting new highs)  —  this script exits; manual re-evaluation required"
        exit 0
    fi
done
stage "STAGE-1 M3 PROCESS GONE" "settle 3 minutes: wait for last_model/epoch_metrics to be written to disk and workers cleaned up"

sleep 180
FREE_OK=0
for i in $(seq 1 12); do
    FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
              --id="$G4090" 2>/dev/null | head -1 | tr -d ' ')
    if [ -n "${FREE_MB:-}" ] && [ "$FREE_MB" -ge 18000 ]; then FREE_OK=1; break; fi
    log "4090 free=${FREE_MB:-?}MB (<18000)  —  waiting 5min ($i/12)"
    sleep 300
done
[ "$FREE_OK" -eq 1 ] || { stage "ABORT" "4090 GPU memory not released within 60min (free=${FREE_MB:-?}MB)  —  manual check needed"; exit 1; }
stage "STAGE-2 4090 READY" "free=${FREE_MB}MB"

stage "STAGE-3 M3 terminal_eval_chain (4090)" "preregistered chain: preflight G1-G3 → anchors×4 → test → ood → D1-D3 → report×2 → verdict/gate"
GPU_UUID="$G4090" bash scripts/run/terminal_eval_chain.sh m3 >> "$LOG" 2>&1
CH=$?
if [ "$CH" -ne 0 ]; then
    stage "STAGE-3 BLOCKED/FAIL (exit=$CH)" "see logs/terminal_eval_chain.log  —  the three-hard-gate check chain was interrupted; no partial artifacts mislabeled as terminal"
    exit 1
fi
stage "STAGE-3 M3 terminal chain complete" "artifact: docs/evidence/terminal/M3/ (*_m3final.json)"

stage "STAGE-4 M2 --best_now isolated evaluation (4090)" "output → PI_REVIEW_M2_BEST/eval_interim/M2/ (tag=m2best_interim, never matches the terminal glob)"
EV="env CUDA_VISIBLE_DEVICES=$G4090"
run_ev() {  # run_ev <logname> <args...>
    local ln="$1"; shift
    $EV $PY -u scripts/interim_eval.py --model m2 --best_now --gpu_uuid "$G4090" "$@" \
        2>&1 | tee -a "logs/pi_review_m2best_${ln}.log" | tail -2
    local rc=${PIPESTATUS[0]}
    log "  $ln exit=$rc"
    return $rc
}
for anchor in benbow iv4 ice philympics; do
    run_ev "anchor_${anchor}" --task anchors --anchor "$anchor" --win_cap 10 --batch 8
    stage "STAGE-4 anchor=$anchor done" ""
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
stage "STAGE-4 M2 best-now evaluation complete" "PI_REVIEW_M2_BEST/eval_interim/M2/  —  ready to run scripts/build_pi_review.py"

rm -f "$PIDFILE"
stage "ALL-DONE" "M3 terminal + M2 interim all ready; next step: assemble the PI draft (build_pi_review.py) "
log "ALL-DONE"
