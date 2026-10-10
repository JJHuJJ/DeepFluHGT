#!/bin/bash
cd .
mkdir -p logs
echo "gpu_health_monitor start $(date '+%F %T'), interval=6h" >> logs/gpu_health.log
while true; do
    TS=$(date '+%F %T')
    nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks_event_reasons.active \
        --format=csv,noheader 2>/dev/null | while IFS= read -r row; do
        echo "$TS,$row" >> logs/gpu_health.log
    done
    sleep 21600
done
