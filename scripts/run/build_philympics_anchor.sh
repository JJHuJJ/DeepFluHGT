#!/bin/bash
# scripts/run/build_philympics_anchor.sh
set -u
cd .
python -u scripts/build_philympics_anchor.py \
    --threads 32 \
    > logs/build_philympics_anchor.log 2>&1
echo "PHIL_BUILD_EXIT=$?" >> logs/build_philympics_anchor.log
