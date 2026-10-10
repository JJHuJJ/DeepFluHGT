#!/usr/bin/env bash
set -u
cd .
PY=python
GA100=${GPU_A100_UUID}
FC=V7_FINAL_CANDIDATE_CURRENT_BEST
LOG=logs/a12_reeval_20260908.log
echo "[reeval] start $(date '+%F %T')" | tee -a "$LOG"
for KEYLAB in "m2 M2" "m3 M3"; do
    set -- $KEYLAB; KEY=$1; LAB=$2
    for A in benbow iv4 ice philympics; do
        TLOG=logs/a12_reeval_${LAB}_${A}_20260908.log
        echo "[reeval] $LAB/$A $(date '+%F %T')" | tee -a "$LOG"
        env CUDA_VISIBLE_DEVICES=$GA100 $PY -u scripts/interim_eval.py --model "$KEY" --best_now \
            --gpu_uuid $GA100 --ckpt_override "$FC/08_METRICS/${LAB}_ASSUMED_TERMINAL/checkpoint.pt" \
            --expect_epoch_override 7 --out_dir "$FC/09_EVAL/$LAB" \
            --task anchors --anchor "$A" --win_cap 10 --batch 8 2>&1 | tee "$TLOG" | tail -8
        if ! $PY - "$LAB" "$A" <<'PYCHK'
import json, sys
lab, a = sys.argv[1], sys.argv[2]
p = f"V7_FINAL_CANDIDATE_CURRENT_BEST/09_EVAL/{lab}/real_anchor_{a}_{lab.lower()}best_interim.json"
j = json.load(open(p)); e = j.get("evo", {})
assert str(e.get("evo_cache_path","")).endswith("anchors/merged.pt"), \
    f"{lab}/{a}: not yetmakeuse merged.pt: {e.get('evo_cache_path')}"
hit, total = e.get("contigs_cache_hit",0), e.get("contigs_total",0)
assert total > 0 and hit == total, f"{lab}/{a}: embedding hits {hit}/{total}, incomplete"
print(f"[chk] {lab}/{a}: merged.pt hits {hit}/{total} OK")
PYCHK
        then echo "[reeval] FAIL at $LAB/$A  —  halting the chain" | tee -a "$LOG"; exit 1; fi
    done
done
echo "[reeval] all done $(date '+%F %T')" | tee -a "$LOG"
