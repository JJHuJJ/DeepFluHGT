#!/bin/bash
set -uo pipefail
cd .
PY=python
LOG=logs/residual_audit_chain.log
exec >> $LOG 2>&1
echo "=== chain start $(date) ==="

for i in $(seq 1 180); do
  a=docs/evidence/terminal/M2/val_window_probs_final.npz
  b=docs/evidence/terminal/M3/val_window_probs_final.npz
  [ -f "$a" ] && [ -f "$b" ] && break
  sleep 60
done
if [ ! -f "$a" ] || [ ! -f "$b" ]; then echo "FATAL: npz did not appear within 3h"; exit 1; fi
echo "--- npz ready $(date) ---"
ls -la $a $b

$PY scripts/residual_corr_terminal.py || { echo "FATAL: residual audit"; exit 1; }

python3 scripts/render_supp_figs.py --only 7 || echo "WARN: SF7 render rc=$?"

FC=V7_FINAL_CANDIDATE_CURRENT_BEST
M=_full_version/03_FIGURES_MASTER/rendered
S=_final_submission
for f in SF1A SF1B SF1C SF1D SF7A SF7B SF7C SF7D SF8A SF8B SF8C SF8D; do
  for ext in pdf png; do
    [ -f $M/$f.$ext ] || continue
    [ -d $FC/02_FIGURES ] && cp $M/$f.$ext $FC/02_FIGURES/$f.$ext
    for pkg in $FC/11_SUBMISSION_PACKAGE/NCS $FC/11_SUBMISSION_PACKAGE/Nature_Computational_Science $FC/11_SUBMISSION_PACKAGE/Nature_Methods $FC/11_SUBMISSION_PACKAGE/Genome_Biology $FC/11_SUBMISSION_PACKAGE/Nature_Communications; do
      [ -f "$pkg/Figures/$f.$ext" ] && cp $M/$f.$ext "$pkg/Figures/$f.$ext"
    done
    for d in Figures 03_FIGURES_SUPPLEMENTARY; do
      [ -f $S/$d/$f.$ext ] && cp $M/$f.$ext $S/$d/$f.$ext
    done
  done
done
echo "=== chain done $(date) ==="
