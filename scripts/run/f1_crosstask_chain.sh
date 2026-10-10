#!/usr/bin/env bash
#                  -> docs/evidence/F1_FUSION_PREVIEW.{md,EXIT}
#                  -> docs/evidence/CROSSTASK_TRAIN2TEST.EXIT
set -euo pipefail
cd .
PY=python
export CUDA_VISIBLE_DEVICES=""

echo "[chain $(date '+%F %T')] step1/3: ntv3_pooled_parallel (F1 preview)"
"$PY" scripts/ntv3_pooled_parallel.py
echo "[chain $(date '+%F %T')] step1 done (F1_FUSION_PREVIEW.EXIT should exist)"

echo "[chain $(date '+%F %T')] step2/3: crosstask train split NTv3 embedding (5 ds)"
"$PY" scripts/embed_ntv3_cpu.py --datasets all --split train --threads 48
echo "[chain $(date '+%F %T')] step2 done (each {ds}_ntv3_pooled_train.npz)"

echo "[chain $(date '+%F %T')] step3/3: train→test probe (Feng protocol)"
"$PY" scripts/crosstask_train2test.py
echo "[chain $(date '+%F %T')] ALL DONE"
