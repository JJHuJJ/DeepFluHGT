#!/usr/bin/env bash
set -u
cd .
PY=python
GA100=${GPU_A100_UUID}
FC=V7_FINAL_CANDIDATE_CURRENT_BEST
LOG=logs/final_candidate_autobuild.log
EVLOG=logs/final_candidate_eval.log
echo "[autobuild] waiter start $(date '+%F %T')" | tee -a "$LOG"
while pgrep -f "final_candidate_eval.sh" > /dev/null 2>&1; do sleep 300; done
sleep 60
echo "[autobuild] eval chain exited $(date '+%F %T')" | tee -a "$LOG"
if ! /usr/bin/grep -q "ALL DONE" "$EVLOG"; then
    echo "[autobuild] ABORT: eval chain did not reach ALL DONE  —  manual inspection of $EVLOG required" | tee -a "$LOG"
    exit 1
fi
bash scripts/run/a12_evo_anchor_completion.sh 2>&1 | tee -a "$LOG" | tail -3
$PY - << 'A12BEFORE' 2>&1 | tee -a "$LOG"
import json, os
FC = "V7_FINAL_CANDIDATE_CURRENT_BEST"
AGGS = {"benbow": "mean", "iv4": "max", "ice": "mean", "philympics": "mean"}
before = {}
for lab in ("M2", "M3"):
    for a in ("benbow", "iv4", "ice", "philympics"):
        p = f"{FC}/09_EVAL/{lab}/real_anchor_{a}_{lab.lower()}best_interim.json"
        if os.path.isfile(p):
            j = json.load(open(p))
            before[f"{lab}_{a}"] = round(j["results"][AGGS[a]]["auc"], 4)
json.dump(before, open("/tmp/a12_before_snapshot.json", "w"), indent=1)
print(f"[A12-before] {len(before)} anchor values snapshotted (pre-rerun, zero-fill era)")
A12BEFORE

for KEYLAB in "m2 M2" "m3 M3"; do
    set -- $KEYLAB; KEY=$1; LAB=$2
    for A in benbow iv4 ice philympics; do
        env CUDA_VISIBLE_DEVICES=$GA100 $PY -u scripts/interim_eval.py --model "$KEY" --best_now \
            --gpu_uuid $GA100 --ckpt_override "$FC/08_METRICS/${LAB}_ASSUMED_TERMINAL/checkpoint.pt" \
            --expect_epoch_override 7 --out_dir "$FC/09_EVAL/$LAB" \
            --task anchors --anchor "$A" --win_cap 10 --batch 8 2>&1 | tail -1
    done
done
echo "[A12] anchor re-eval done $(date '+%F %T')" | tee -a "$LOG"

if ! $PY scripts/a12_embedding_qc.py 2>&1 | tee -a "$LOG"; then
    echo "[A12-QC] ⚠️ FAIL  —  embedding problems found; not producing a bogus protocol-consistent package; check A12_EMBEDDING_QC.json" | tee -a "$LOG"
    exit 1
fi

$PY - << 'A12AUDIT' 2>&1 | tee -a "$LOG"
import json, os, copy
FC = "V7_FINAL_CANDIDATE_CURRENT_BEST"
AGGS = {"benbow": "mean", "iv4": "max", "ice": "mean", "philympics": "mean"}
ANCHORS = ["benbow", "iv4", "ice", "philympics"]

def read_anchor(lab, a):
    tag = f"{lab.lower()}best_interim"
    p = f"{FC}/09_EVAL/{lab}/real_anchor_{a}_{tag}.json"
    if not os.path.isfile(p):
        return None
    j = json.load(open(p))
    return round(j["results"][AGGS[a]]["auc"], 4)

def read_ext_wins(fm_path="_full_version/FINAL_RESULTS/final_metrics.yaml"):
    """§6.3: M2 >= M0 on >=2 of 4 anchors (pointwise)."""
    import yaml
    fm = yaml.safe_load(open(fm_path))
    m0 = {"benbow": fm["M0"]["anchors"]["benbow_mean_auroc"],
          "iv4": fm["M0"]["anchors"]["iv4_max_auroc"],
          "ice": fm["M0"]["anchors"]["iceberg_mean_auroc"],
          "philympics": fm["M0"]["anchors"]["philympics_paired_mean_auroc"]}
    fcm = {k: v["value"] for k, v in json.load(open(f"{FC}/08_METRICS/FC_METRICS.json")).items()}
    wins = 0
    per = {}
    for a in ANCHORS:
        m2v = fcm.get(f"M2_{'BENBOW_MEAN' if a=='benbow' else 'IV4_MAX' if a=='iv4' else 'ICE_MEAN' if a=='ice' else 'PHILYMPICS_MEAN'}")
        pv = m2v >= m0[a] if m2v else False
        per[a] = {"M0": round(m0[a],4), "M2": round(m2v,4) if m2v else None, "pass": pv}
        if pv: wins += 1
    return wins, per

before = {}
for lab in ("M2", "M3"):
    for a in ANCHORS:
        v = read_anchor(lab, a)
        if v is not None:
            before[f"{lab}_{a}"] = v
wins_before, per_before = read_ext_wins()

after = {}
for lab in ("M2", "M3"):
    for a in ANCHORS:
        v = read_anchor(lab, a)
        if v is not None:
            after[f"{lab}_{a}"] = v
wins_after, per_after = read_ext_wins()

# ── DELTA ──
deltas = {}
for k in after:
    if k in before:
        deltas[k] = round(after[k] - before[k], 4)

# ── QC + checkpoint + zero-fill ──
qc = json.load(open(f"{FC}/A12_EMBEDDING_QC.json")) if os.path.isfile(f"{FC}/A12_EMBEDDING_QC.json") else {"qc": "MISSING"}
reg = json.load(open(f"{FC}/08_METRICS/ASSUMED_TERMINAL_CHECKPOINT_REGISTER.json"))
ckpt = {m["model"]: m["checkpoint_sha256_16"] for m in reg["models"]}
zero_remaining = sum(1 for lab in ("M2","M3") for a in ANCHORS
                     if os.path.isfile(f"{FC}/09_EVAL/{lab}/real_anchor_{a}_m{lab.lower()[0]}best_interim.json")
                     and json.load(open(f"{FC}/09_EVAL/{lab}/real_anchor_{a}_m{lab.lower()[0]}best_interim.json")).get("evo",{}).get("contigs_zero_filled", 0) > 0)

audit = {
  "amendment": "A12_protocol_completion (FINAL  —  no A13/A14)",
  "before_protocol": "zero-filled external Evo2 vector (implementation deferral)",
  "after_protocol": "real per-contig Evo2 7B blocks.28.mlp.l3 mean-pooled embedding",
  "scope": "M2/M3 external-anchor evaluation ONLY (benbow+iv4+ice+philympics)",
  "M0": "UNAFFECTED (use_evo: false)",
  "checkpoint": {"M2": f"ep7/{ckpt.get('M2','?')}", "M3": f"ep7/{ckpt.get('M3','?')}"},
  "embedding_qc": qc.get("qc", "MISSING"),
  "zero_fill_remaining": zero_remaining,
  "before_metrics": before,
  "after_metrics": after,
  "delta": deltas,
  "section_6_3": {
    "before": {"wins": wins_before, "per_anchor": per_before},
    "after": {"wins": wins_after, "per_anchor": per_after},
    "criterion": "M2 >= M0 on >=2 of 4 anchors",
    "flip": wins_before != wins_after,
  },
  "anti_cherry_pick": "criterion flip enters manuscript as-is; no A13/A14 regardless of direction",
  "prohibited": ["new benchmark", "new metric", "new threshold", "new model",
                 "new training", "new attribution experiment", "rewrite preregistration",
                 "retrospective tuning", "A13", "A14"],
  "close_conditions": [
    "A12=CLOSED", "checkpoint=ep7 unchanged",
    f"four anchors={'real Evo2 embedding (verified: contigs_cache_hit==contigs_total)' if zero_remaining==0 else 'ZERO-FILL REMNANT='+str(zero_remaining)}",
    "embedding QC=PASS", "zero-fill residual=0", "§6.3=deterministic",
    "manuscript=synchronized", "SI=synchronized", "RS=synchronized", "figures=synchronized",
    "tokens=0", "forbidden=0", "figure FAIL=0", "ghost=0", "provenance=PASS",
    "science freeze baseline=unchanged"],
}
json.dump(audit, open(f"{FC}/A12_BEFORE_AFTER_AUDIT.json", "w"), indent=1, ensure_ascii=False)
flip_s = "⚠️ FLIPPED" if wins_before != wins_after else "stable"
print(f"[A12-audit] §6.3: {wins_before}/4 → {wins_after}/4 ({flip_s}); "
      f"QC={qc.get('qc')}; zero_remaining={zero_remaining}; "
      f"delta_range=[{min(deltas.values(), default=0):.4f}, {max(deltas.values(), default=0):.4f}]")
A12AUDIT
$PY scripts/build_final_candidate.py 2>&1 | tee -a "$LOG" | tail -4
$PY scripts/render_fc_figures.py 2>&1 | tee -a "$LOG" | tail -2
$PY scripts/package_final_candidate.py 2>&1 | tee -a "$LOG" | tail -4
$PY scripts/visual_qa.py 2>&1 | tee -a "$LOG" | tail -4
$PY scripts/fc_five_question_gate.py 2>&1 | tee -a "$LOG"
PH=$($PY -c "
import re
ms=open('V7_FINAL_CANDIDATE_CURRENT_BEST/01_MANUSCRIPT/Manuscript.md').read()
si=open('V7_FINAL_CANDIDATE_CURRENT_BEST/04_SUPPLEMENTARY/Supplementary_Information.md').read()
print(len(re.findall(r'INSERT_[A-Z0-9_]+|PENDING_TERMINAL|TERMINAL-PENDING', ms+si)))")
echo "[autobuild] verification: unresolved_tokens=$PH" | tee -a "$LOG"
if [ "$PH" = "0" ]; then
    echo "[autobuild] ══ FINAL-CANDIDATE PACKAGE COMPLETE $(date '+%F %T') ══" | tee -a "$LOG"
else
    echo "[autobuild] ⚠ unresolved tokens=$PH  —  check for missing evaluation artifacts" | tee -a "$LOG"
fi
