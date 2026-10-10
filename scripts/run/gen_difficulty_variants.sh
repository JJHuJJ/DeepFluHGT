#!/bin/bash
# scripts/run/gen_difficulty_variants.sh
# ============================================================================
#
#
#
# ============================================================================
set -euo pipefail
cd .

PY=python
RUN="${RUN:-0}"
TIER="${1:-all}"                      # D1 | D2 | D3 | all
PROC=8
N=5000

gen_one () {
    local name="$1"                                   # D1_gi_length | D2_donor_sim | D3_fragmentation
    local cfg="configs/difficulty_ladder/${name}.yaml"
    local out="simulated_data/difficulty_ladder/${name}"
    local log="logs/gen_${name}.log"
    [ -f "$cfg" ] || { echo "[ERR] missing $cfg"; exit 1; }
    if [ -f "${out}/test/dataset.fasta" ]; then
        echo "[skip] ${out}/test/dataset.fasta already exists"
        return 0
    fi
    echo "[gen] ${name}: n_pos=n_neg=${N} processes=${PROC} -> ${out}"
    if [ "$RUN" = "1" ]; then
        mkdir -p logs
        $PY -u DeepFluHGT.py simulate \
            --mode contig \
            --virus_fasta <PATH_TO_IAV_FASTA> \
            --microbe_dir   ./microbe_genomes \
            --output_dir    "${out}" \
            --config        "${cfg}" \
            --n_pos ${N} --n_neg ${N} \
            --processes ${PROC} \
            2>&1 | tee "${log}"
    fi
}

case "$TIER" in
    D1) gen_one D1_gi_length ;;
    D2) gen_one D2_donor_sim ;;
    D3) gen_one D3_fragmentation ;;
    all) gen_one D1_gi_length; gen_one D2_donor_sim; gen_one D3_fragmentation ;;
    *) echo "usage: RUN=1 bash $0 [D1|D2|D3|all]"; exit 1 ;;
esac

stats () {
    $PY - "$@" <<'EOF'
import json, os, sys
import numpy as np
for split_dir in sys.argv[1:]:
    if not os.path.isfile(f"{split_dir}/test/metadata.json"):
        print(f"{split_dir}: [not generated yet, skipped]")
        continue
    meta = json.load(open(f"{split_dir}/test/metadata.json"))
    pos = meta["positive"] if isinstance(meta, dict) else [m for m in meta if m["label"]==1]
    spans, frag = [], []
    for m in pos:
        ev = m.get("events", [])
        if ev:
            spans.append(max(e["hgt_end"] for e in ev) - min(e["hgt_start"] for e in ev))
            frag += [e.get("virus_len", 0) for e in ev]
    a, f = np.array(spans), np.array(frag)
    print(f"{split_dir}: n_pos={len(pos)} island-span[min/p50/p90/max]="
          f"{a.min()}/{np.median(a):.0f}/{np.percentile(a,90):.0f}/{a.max()} "
          f"per-frag med={np.median(f):.0f} (IV4 reference: p50=8019 p90=28155)")
EOF
}
echo
echo "── post-generation acceptance statistics (run automatically when RUN=1; tiers not yet generated are skipped) ──"
if [ "$RUN" = "1" ]; then
    stats simulated_data/difficulty_ladder/D1_gi_length simulated_data/difficulty_ladder/D2_donor_sim simulated_data/difficulty_ladder/D3_fragmentation || true
fi
echo
echo "⚠ Next steps (run on GPU nodes, out of scope for this script):"
echo "  1) evo/ntv3 re-embedding: scripts/precompute_evo2_embeddings.py toward simulated_data/difficulty_ladder/*/test/"
echo "  2) checkpoint inference (no retraining): reuse the eval pipeline; run D0 in the same batch using simulated_data/Contig_500k/test/"
echo "  3) Produce the difficulty-AUPRC curve: AUPRC for each tier D0→D1→D2→D3 + DeLong/bootstrap CI"
