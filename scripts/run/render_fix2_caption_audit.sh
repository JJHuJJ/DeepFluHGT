#!/usr/bin/env bash
# =============================================================================
# render_fix2_caption_audit.sh — caption/visualization audit fix batch (2026-09-02)
# Re-renders ALL figures after the renderer-only fixes (ghosts off value axis,
# dot+CI pooling/difficulty, panel-letter dedup, categorical ramp, CI-family
# tags, humanized FDR strip, <=183mm schematics, comma counts, no printed
# INSERT tokens). Data values + provenance unchanged; forensic gate must stay
# FAIL=0.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=python

echo "=== [1/3]  publication renderer (all 6 slots) ==="
"$PY" scripts/render_figures.py 2>&1 | tee logs/render_figures_fix2.log
R76=${PIPESTATUS[0]}

echo "=== [2/3] 5 ghost renderer (Figs 1-5) ==="
"$PY" scripts/render_figures.py 2>&1 | tee logs/render_v75_ghost_fix2.log
R75=${PIPESTATUS[0]}

echo "=== [3/3] forensic gate (append) ==="
"$PY" scripts/figure_forensic.py 2>&1 | tee -a logs/figure_forensic.log
FG=${PIPESTATUS[0]}

echo "exit codes: v76=$R76 v75=$R75 forensic=$FG (forensic 0 == FAIL=0)"
exit $(( R76 | R75 | FG ))
