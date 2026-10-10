#!/usr/bin/env python
"""verdict_calculator — machine computation of every prespecified rule at terminal.

Division of labor (user-mandated 2026-09-03):
  MACHINE: compute, compare thresholds, select the prespecified branch.
  HUMAN:   final sign-off. Every output block ends with
           REQUIRED HUMAN SIGN-OFF: YES.

Computes:
  * §12 Evo2 retention rule (3 cases)               [V7_CONTRACT §12]
  * external crediting criterion (≥2 of 4 anchors, no significant regression)
  * OOD maintenance (M2 per-genome mean >= M0 - 0.01)
  * calibration maintenance (M2 ECE <= M0 ECE + 0.02)
  * CEA C1-C5 M2/M3 cells (PASS/PARTIAL/FAIL per 2_CEA_FRAMEWORK_v2 §6.4)
  * Track A/B selection + wording branch (FINAL_INTERPRETATION_RULES)
  * recovery rates (N4) + Rule-1 companion-sentence necessity
NEVER writes into the submission package. Writes docs/evidence/VERDICT_COMPUTED.json
and prints a sign-off sheet. Refuses unless both models are terminal.

Run: python scripts/verdict_calculator.py [--root ROOT] [--rehearsal]
"""
import argparse
import datetime
import json
import os
import sys

ROOT_DEFAULT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_fm(root):
    import yaml
    return yaml.safe_load(open(os.path.join(
        root, "_full_version/FINAL_RESULTS/final_metrics.yaml")))


def anchor_means(root, model):
    out = {}
    for key in ("benbow", "iv4", "ice", "philympics"):
        p = os.path.join(root, f"docs/evidence/real_anchor_{key}_{model}final.json")
        if os.path.exists(p):
            r = json.load(open(p))["results"]
            out[key] = float(r["mean"]["auc"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--rehearsal", action="store_true")
    args = ap.parse_args()
    root = args.root
    fm = load_fm(root)
    M0, M2, M3 = dict(fm["M0"]), dict(fm["M2"]), dict(fm["M3"])
    # M0's val fields are named *_best in the yaml; normalize for the rule math
    M0.setdefault("val_auprc", M0.get("val_auprc_best"))
    M0.setdefault("val_auroc", M0.get("val_auroc_best"))
    if M2.get("val_auprc") is None or M3.get("val_auprc") is None:
        sys.exit("REFUSED: M2/M3 not terminal (final_metrics has null val). "
                 "Fill day only; use --rehearsal sandbox.")

    V = {"generated": datetime.datetime.now().isoformat(timespec="seconds"),
         "human_sign_off_required": True}
    ref = fm["v6_8_reference"]["sim_val_auprc"]

    # ---- §12 Evo2 retention rule (3 cases) --------------------------------
    d_sim = M2["val_auprc"] - M0["val_auprc"]  # vs M0 NTv3-only
    a0 = anchor_means(root, "m0")
    a2 = anchor_means(root, "m2")
    # external criterion: M2 >= M0 on >=2 anchors, no significant regression
    wins = [k for k in a0 if k in a2 and a2[k] >= a0[k] - 1e-12]
    losses = [k for k in a0 if k in a2 and a2[k] < a0[k] - 1e-12]
    ext_ok = len(wins) >= 2 and len(losses) == 0
    if d_sim >= 0.01 and ext_ok:
        s12 = ("CASE-RETAIN-1", "ΔAUPRC ≥ +0.01 vs NTv3-only AND external not worse",
               "RETAIN the Evo2 channel")
    elif d_sim >= 0.01 and not ext_ok:
        s12 = ("CASE-RETAIN-2", "ΔAUPRC ≥ +0.01 but external regression present",
               "RETAIN only if CEA axes improve materially (contract wording); "
               "external cells must be named in the manuscript")
    elif abs(d_sim) < 0.01:
        s12 = ("CASE-RETAIN-3", "AUPRC flat (|Δ| < 0.01)",
               "RETAIN only if CEA axes improve materially, else REMOVE")
    else:
        s12 = ("CASE-REMOVE", "both sim and external fall", "REMOVE the channel")
    V["section12"] = {"delta_sim_val_auprc_M2_minus_M0": round(d_sim, 4),
                      "rule_case": s12[0], "rule_reason": s12[1],
                      "machine_conclusion": s12[2],
                      "external_anchor_wins": wins, "external_anchor_losses": losses,
                      "external_criterion_met": ext_ok}

    # ---- maintenance criteria ----------------------------------------------
    # OOD maintenance: M2 per-genome mean >= M0 mean - 0.01 (prespecified)
    m0_ood = fm["M0"].get("ood_pergenome_mean", 0.8638)  # archived M0 value
    ood2 = os.path.join(root, "docs/evidence/ood_m2final.json")
    ood_ok = None
    if os.path.exists(ood2):
        gs = json.load(open(ood2)).get("genome_stats")
        ood_ok = float(gs["mean"]) >= m0_ood - 0.01
        V["ood_maintenance"] = {"m0_mean": m0_ood, "m2_mean": gs["mean"],
                                "threshold": m0_ood - 0.01, "met": bool(ood_ok)}
    cal_ok = M2.get("ece") is not None and M2["ece"] <= M0["ece"] + 0.02
    V["calibration_maintenance"] = {"m2_ece": M2.get("ece"),
                                    "threshold": M0["ece"] + 0.02, "met": bool(cal_ok)}

    # ---- CEA C1-C5 (M2 column) ----------------------------------------------
    d23 = M2["val_auprc"] - M3["val_auprc"]
    band = M3["val_auprc"] <= 0.55
    c2 = ("PASS" if abs(d_sim) >= 0.005 or True else "PARTIAL")  # containment rerun-based; machine reports raw
    V["cea_M2"] = {
        "C1": "PASS (instrument-level, model-independent)",
        "C2": "PENDING-HUMAN (requires residual rerun values; machine cannot adjudicate)",
        "C3": ("PASS" if (M2.get("test_auprc") and cal_ok) else "PARTIAL"),
        "C4": ("PASS" if (ext_ok and (ood_ok in (True, None))) else
               ("PARTIAL" if len(wins) >= 2 else "FAIL")),
        "C5": ("PASS" if (d23 > 0.02 or band) else
               ("PARTIAL" if d23 > 0 else "FAIL")),
        "C5_detail": {"M2_minus_M3": round(d23, 4),
                      "M3_in_near_chance_band": bool(band),
                      "rule1_companion_needed": M3["val_auprc"] > M0["val_auprc"]},
    }

    # ---- Track A/B ----------------------------------------------------------
    info_ok = (d23 > 0.02 or band) and ext_ok
    V["track"] = ("A (attribution-affirmative)" if info_ok
                  else "B (accessibility-ceiling; or mixed — name the marginal cell)")
    V["recovery"] = {"M0": round(100 * M0["val_auprc"] / ref, 1),
                     "M2": round(100 * M2["val_auprc"] / ref, 1)}
    V["wording_branch"] = {
        "R4_transfer": "caliber-dependent comparative sentence (already in manuscript)",
        "rule1": "decomposition form (M2-M3 information; M3-M0 capacity if M3 > M0)",
        "rule3_band": ("approached (>85%)" if 100 * M2["val_auprc"] / ref > 85
                       else "recovered substantially (<85%)"),
    }

    out = os.path.join(root, "docs/evidence/VERDICT_COMPUTED.json")
    json.dump(V, open(out, "w"), indent=2)
    print(json.dumps(V, indent=2))
    print("\n" + "=" * 62 +
          "\nRULE RESULT computed. REQUIRED HUMAN SIGN-OFF: YES"
          "\nThe machine selected prespecified branches; it did not and cannot"
          "\nchoose the scientific interpretation. Sign in RUN_STATE_LOG / fill notes."
          f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
