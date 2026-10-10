#!/bin/bash
# scripts/run/crosstask_evo2_4090.sh
set -u
cd .
G4090=${GPU_4090_UUID}
CUDA_VISIBLE_DEVICES=$G4090 python -u \
    scripts/embed_crosstask_evo2_4090.py --datasets all \
    > logs/crosstask_evo2_4090.log 2>&1
echo "CROSSTASK_EVO2_EXIT=$?" >> logs/crosstask_evo2_4090.log
