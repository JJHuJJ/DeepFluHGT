#!/bin/bash
# scripts/run/crosstask_full_probe.sh
set -u
cd .
PY=python
mkdir -p logs

$PY -u scripts/crosstask_full_probe.py --datasets celegans_4mc --resume \
    > logs/crosstask_full_probe_celegans.log 2>&1 &
P1=$!
$PY -u scripts/crosstask_full_probe.py --datasets human_vs_worm --resume \
    > logs/crosstask_full_probe_htw.log 2>&1 &
P2=$!
$PY -u scripts/crosstask_full_probe.py \
    --datasets promoter_b_amyloliquefaciens,promoter_r_capsulatus,ecoli_4mc --resume \
    > logs/crosstask_full_probe_small.log 2>&1 &
P3=$!
echo "[launcher] probe pids: $P1 $P2 $P3"
wait $P1 $P2 $P3
echo "[launcher] all probe processes exited"

$PY -u scripts/crosstask_full_probe.py --report-only \
    > logs/crosstask_full_probe_report.log 2>&1
echo "CROSSTASK_PROBE_EXIT=$?" >> logs/crosstask_full_probe_report.log
