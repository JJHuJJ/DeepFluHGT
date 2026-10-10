#!/bin/bash
# scripts/run/interim_eval_chain.sh
# ============================================================================
#
#
#
#
#
#         > logs/interim_eval_chain.log 2>&1 &
# ============================================================================
set -u
cd .

GA100=${GPU_A100_UUID}
PY=python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

M2DIR=docs/evidence/interim/M2_ep4
M3DIR=docs/evidence/interim/M3_ep3
mkdir -p logs "$M2DIR" "$M3DIR"

echo "[chain] ════ interim_eval_chain START $(date '+%F %T') pid=$$ ════"
echo "[chain] pre-flight GPU/RAM snapshot:"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv || true
free -g | head -2

a100_free_mib() {
    nvidia-smi --id="$GA100" --query-gpu=memory.free \
        --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

wait_a100_mem() {
    local min_mib="$1" max_wait="$2" waited=0 f
    while :; do
        f=$(a100_free_mib)
        if [ -n "$f" ] && [ "$f" -ge "$min_mib" ] 2>/dev/null; then
            echo "[chain] A100 free ${f}MiB ≥ ${min_mib}MiB → proceeding"
            return 0
        fi
        if [ "$waited" -ge "$max_wait" ]; then
            echo "[chain] wait_a100_mem timeout (${max_wait}s): free=${f}MiB < ${min_mib}MiB"
            return 1
        fi
        if [ "$waited" -eq 0 ]; then
            echo "[chain] A100 free ${f}MiB < ${min_mib}MiB, polling every 2min (≤$((max_wait/60))min):"
            nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
                --format=csv,noheader | grep "$GA100" || true
        fi
        sleep 120; waited=$((waited+120))
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

# gstep <NAME> <logfile> <timeout_s> <guardfile|-> <min_free_mib|0> <cmd...>
gstep() {
    local name="$1" log="$2" tmo="$3" guard="$4" minmib="$5"; shift 5
    if [ "$guard" != "-" ] && [ -f "$guard" ]; then
        echo "${name}_EXIT=0 (SKIP: $guard already exists)" >> "$log"
        echo "[chain] $(date '+%F %T') SKIP  ${name} (artifact already exists: $guard)"
        return 0
    fi
    if [ "$minmib" -gt 0 ]; then
        wait_a100_mem "$minmib" 14400 || {
            echo "${name}_EXIT=98 (A100 GPU memory insufficient, wait timed out)" >> "$log"
            echo "[chain] $(date '+%F %T') MEMWAIT-FAIL ${name}"
            return 0
        }
    fi
    step "$name" "$log" "$tmo" "$@"
}

EV="env CUDA_VISIBLE_DEVICES=$GA100"

# ════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════
gstep M2_ANCHOR_BENBOW logs/interim_m2_anchor_benbow.log 5400 \
    "$M2DIR/real_anchor_benbow_m2ep4.json" 12288 \
    $EV $PY -u scripts/interim_eval.py --model m2 --task anchors --anchor benbow --win_cap 10 --batch 8
gstep M2_ANCHOR_IV4 logs/interim_m2_anchor_iv4.log 5400 \
    "$M2DIR/real_anchor_iv4_m2ep4.json" 12288 \
    $EV $PY -u scripts/interim_eval.py --model m2 --task anchors --anchor iv4 --win_cap 10 --batch 8
gstep M2_ANCHOR_ICE logs/interim_m2_anchor_ice.log 5400 \
    "$M2DIR/real_anchor_ice_m2ep4.json" 12288 \
    $EV $PY -u scripts/interim_eval.py --model m2 --task anchors --anchor ice --win_cap 10 --batch 8
gstep M2_ANCHOR_PHILYMPICS logs/interim_m2_anchor_philympics.log 5400 \
    "$M2DIR/real_anchor_philympics_m2ep4.json" 12288 \
    $EV $PY -u scripts/interim_eval.py --model m2 --task anchors --anchor philympics --win_cap 10 --batch 8
gstep M2_ANCHOR_REPORT logs/interim_m2_anchor_report.log 900 \
    "$M2DIR/real_anchors.md" 0 \
    env CUDA_VISIBLE_DEVICES="" $PY -u scripts/interim_eval.py --model m2 --task report --which anchors

gstep M2_TEST logs/interim_m2_test.log 28800 \
    "$M2DIR/test_eval.md" 12288 \
    $EV $PY -u scripts/interim_eval.py --model m2 --task test --win_cap 10 --batch 8
gstep M2_OOD logs/interim_m2_ood.log 3600 \
    "$M2DIR/ood.md" 0 \
    env CUDA_VISIBLE_DEVICES="" $PY -u scripts/interim_eval.py --model m2 --task ood

gstep M2_DIFF_D1 logs/interim_m2_diff_D1.log 14400 \
    "$M2DIR/difficulty_D1_m2ep4.json" 12288 \
    $EV $PY -u scripts/interim_eval.py --model m2 --task difficulty --tier D1 --win_cap 10 --batch 8
gstep M2_DIFF_D2 logs/interim_m2_diff_D2.log 14400 \
    "$M2DIR/difficulty_D2_m2ep4.json" 12288 \
    $EV $PY -u scripts/interim_eval.py --model m2 --task difficulty --tier D2 --win_cap 10 --batch 8
gstep M2_DIFF_D3 logs/interim_m2_diff_D3.log 10800 \
    "$M2DIR/difficulty_D3_m2ep4.json" 12288 \
    $EV $PY -u scripts/interim_eval.py --model m2 --task difficulty --tier D3 --win_cap 10 --batch 8
gstep M2_DIFF_REPORT logs/interim_m2_diff_report.log 900 \
    "$M2DIR/difficulty.md" 0 \
    env CUDA_VISIBLE_DEVICES="" $PY -u scripts/interim_eval.py --model m2 --task report --which difficulty

# ════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════
gstep M3_SHUF_D1 logs/interim_m3_shuffle_D1.log 1800 \
    evo_cache/difficulty/D1_gi_length/test_shuffled.pt 0 \
    env CUDA_VISIBLE_DEVICES="" $PY -u scripts/build_shuffle_cache.py \
        evo_cache/difficulty/D1_gi_length/test.pt \
        evo_cache/difficulty/D1_gi_length/test_shuffled.pt
gstep M3_SHUF_D2 logs/interim_m3_shuffle_D2.log 1800 \
    evo_cache/difficulty/D2_donor_sim/test_shuffled.pt 0 \
    env CUDA_VISIBLE_DEVICES="" $PY -u scripts/build_shuffle_cache.py \
        evo_cache/difficulty/D2_donor_sim/test.pt \
        evo_cache/difficulty/D2_donor_sim/test_shuffled.pt
gstep M3_SHUF_D3 logs/interim_m3_shuffle_D3.log 1800 \
    evo_cache/difficulty/D3_fragmentation/test_shuffled.pt 0 \
    env CUDA_VISIBLE_DEVICES="" $PY -u scripts/build_shuffle_cache.py \
        evo_cache/difficulty/D3_fragmentation/test.pt \
        evo_cache/difficulty/D3_fragmentation/test_shuffled.pt

gstep M3_ANCHOR_BENBOW logs/interim_m3_anchor_benbow.log 5400 \
    "$M3DIR/real_anchor_benbow_m3ep3.json" 12288 \
    $EV $PY -u scripts/interim_eval.py --model m3 --task anchors --anchor benbow --win_cap 10 --batch 8
gstep M3_ANCHOR_IV4 logs/interim_m3_anchor_iv4.log 5400 \
    "$M3DIR/real_anchor_iv4_m3ep3.json" 12288 \
    $EV $PY -u scripts/interim_eval.py --model m3 --task anchors --anchor iv4 --win_cap 10 --batch 8
gstep M3_ANCHOR_ICE logs/interim_m3_anchor_ice.log 5400 \
    "$M3DIR/real_anchor_ice_m3ep3.json" 12288 \
    $EV $PY -u scripts/interim_eval.py --model m3 --task anchors --anchor ice --win_cap 10 --batch 8
gstep M3_ANCHOR_PHILYMPICS logs/interim_m3_anchor_philympics.log 5400 \
    "$M3DIR/real_anchor_philympics_m3ep3.json" 12288 \
    $EV $PY -u scripts/interim_eval.py --model m3 --task anchors --anchor philympics --win_cap 10 --batch 8
gstep M3_ANCHOR_REPORT logs/interim_m3_anchor_report.log 900 \
    "$M3DIR/real_anchors.md" 0 \
    env CUDA_VISIBLE_DEVICES="" $PY -u scripts/interim_eval.py --model m3 --task report --which anchors

gstep M3_TEST logs/interim_m3_test.log 28800 \
    "$M3DIR/test_eval.md" 12288 \
    $EV $PY -u scripts/interim_eval.py --model m3 --task test --win_cap 10 --batch 8
gstep M3_OOD logs/interim_m3_ood.log 3600 \
    "$M3DIR/ood.md" 0 \
    env CUDA_VISIBLE_DEVICES="" $PY -u scripts/interim_eval.py --model m3 --task ood

gstep M3_DIFF_D1 logs/interim_m3_diff_D1.log 14400 \
    "$M3DIR/difficulty_D1_m3ep3.json" 12288 \
    $EV $PY -u scripts/interim_eval.py --model m3 --task difficulty --tier D1 --win_cap 10 --batch 8
gstep M3_DIFF_D2 logs/interim_m3_diff_D2.log 14400 \
    "$M3DIR/difficulty_D2_m3ep3.json" 12288 \
    $EV $PY -u scripts/interim_eval.py --model m3 --task difficulty --tier D2 --win_cap 10 --batch 8
gstep M3_DIFF_D3 logs/interim_m3_diff_D3.log 10800 \
    "$M3DIR/difficulty_D3_m3ep3.json" 12288 \
    $EV $PY -u scripts/interim_eval.py --model m3 --task difficulty --tier D3 --win_cap 10 --batch 8
gstep M3_DIFF_REPORT logs/interim_m3_diff_report.log 900 \
    "$M3DIR/difficulty.md" 0 \
    env CUDA_VISIBLE_DEVICES="" $PY -u scripts/interim_eval.py --model m3 --task report --which difficulty

# ════════════════════════════════════════════════════════════════════════════
echo "[chain] ════ interim_eval_chain DONE $(date '+%F %T') ════"
echo "[chain] artifact inventory (expected: 8 files each × 2 models):"
for d in "$M2DIR" "$M3DIR"; do
    echo "--- $d ---"; ls -la "$d" 2>/dev/null || true
done
echo "[chain] EXIT status of each step:"
grep -h "_EXIT=" logs/interim_m*_*.log 2>/dev/null | sort || true
echo "INTERIM_EVAL_CHAIN_EXIT=0"
