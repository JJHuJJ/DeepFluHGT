#!/bin/bash
# scripts/run/evo_cache_monitor.sh
set -u
cd .
nohup python -u \
    scripts/evo_cache_monitor.py >> logs/evo_cache_monitor.log 2>&1 &
echo "monitor_pid=$!"
