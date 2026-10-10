#!/usr/bin/env bash
# ============================================================================
# ----------------------------------------------------------------------------
# ============================================================================
set -euo pipefail
cd .
PY=python
MODELS="${1:-}"
LOG="logs/finalize_$(date +%Y%m%d_%H%M%S).log"
echo "[finalize] ════ START $(date '+%F %T') models=$MODELS ════" | tee -a "$LOG"

step() {  # step <name> <cmd...>
    local name="$1"; shift
    echo "[finalize] ── $name ── $(date '+%T')" | tee -a "$LOG"
    if "$@" >> "$LOG" 2>&1; then
        echo "[finalize]   PASS" | tee -a "$LOG"
    else
        echo "[finalize]   FAIL — STOPPED at $name (see $LOG)" | tee -a "$LOG"
        exit 1
    fi
}

info() { echo "[finalize] $*" | tee -a "$LOG"; }

# ── 1. Preflight (EXIT rule + contract SHA + immutability) ──────────────
for m in $MODELS; do
    step "preflight:$m" $PY scripts/terminal_eval_preflight.py --model "$m"
done

# ── 2. Terminal evaluation (anchors + test + OOD + difficulty) ─────────
for m in $MODELS; do
    step "terminal_eval:$m" bash scripts/run/terminal_eval_chain_inner.sh "$m"
done

# ── 3. Verdict calculator (machine branch selection) ─────────────────────
step "verdict" $PY scripts/verdict_calculator.py

# ── 4. Provenance gate (ALLOW = manuscript write permission) ────────────
step "provenance_gate" $PY scripts/final_result_provenance_gate.py

# ── 5. Placeholder fill (mechanical tokens only) ──────────────────────────
step "fill_placeholders" $PY scripts/fill_placeholders.py

# ── 6. Numeric consistency check (all artifacts, same numbers) ───────────
step "numeric_consistency" $PY scripts/final_numeric_consistency_check.py

# ── 7. Five-engine validation ─────────────────────────────────────────────
step "engine:forensic" $PY scripts/figure_forensic.py
step "engine:placeholder" $PY scripts/validate_placeholders.py
step "engine:forbidden" $PY scripts/forbidden_scan.py
step "engine:crossref" $PY scripts/check_reference_links.py
step "engine:audit_summary" $PY scripts/audit_summary.py

# ── 8. Generate FINAL_STATUS.md ───────────────────────────────────────────
$PY - <<'PYEOF' >> "$LOG" 2>&1
import json, datetime, os
status = {
    "generated": datetime.datetime.now().isoformat(timespec="seconds"),
    "models_finalized": os.environ.get("MODELS", ""),
    "verdict": "see docs/evidence/VERDICT_COMPUTED.json",
    "provenance_gate": "see docs/evidence/FINAL_RESULT_PROVENANCE_GATE.json",
    "unresolved_tokens": "see 5_submission_final/07_REPORTING/PLACEHOLDER_VALIDATION.md",
    "human_sign_off_required": True,
    "next_step": "fill narrative tokens → FINAL_HUMAN_SIGNOFF.md → freeze → build_submission_package.sh",
}
with open("FINAL_STATUS.md", "w") as f:
    f.write("# FINAL STATUS\n\n")
    f.write(f"Generated: {status['generated']}\n\n")
    for k, v in status.items():
        if k != "generated":
            f.write(f"- **{k}**: {v}\n")
    f.write("\n## Chain executed\n\n")
    f.write("1. preflight ✅\n2. terminal_eval ✅\n3. verdict ✅\n4. provenance_gate ✅\n")
    f.write("5. fill_placeholders ✅\n6. numeric_consistency ✅\n7. five engines ✅\n")
    f.write("\n**Awaiting: human narrative sign-off**\n")
print("FINAL_STATUS.md written")
PYEOF
info "FINAL_STATUS.md generated"

# ── 9. Freeze candidate ───────────────────────────────────────────────────
step "freeze" bash scripts/freeze_submission.sh

echo "[finalize] ════ ALL STEPS COMPLETE ════" | tee -a "$LOG"
echo "[finalize] → next: human narrative sign-off → FINAL_HUMAN_SIGNOFF.md" | tee -a "$LOG"
echo "[finalize] → then: bash scripts/build_submission_package.sh" | tee -a "$LOG"
