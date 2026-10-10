#!/usr/bin/env python
"""P2-② FINAL_RESULT_BINDING_MAP — canonical metric → every location it appears.

Answers: "this M2 AUPRC value — where exactly in all artifacts must it land?"
Generated once now; used by fill_placeholders + final_numeric_consistency_check.
"""
import json, os, re, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "v7.5_submission_final")
OUT_MD = os.path.join(ROOT, "docs/evidence/FINAL_RESULT_BINDING_MAP.md")
OUT_JSON = os.path.join(ROOT, "docs/evidence/FINAL_RESULT_BINDING_MAP.json")

# canonical key → definition
CANONICAL = {
    # sim metrics
    "M2_VAL_AUPRC":     ("final_metrics.yaml M2.val_auprc",  "AUPRC", 4),
    "M2_VAL_AUROC":     ("final_metrics.yaml M2.val_auroc",  "AUROC", 4),
    "M2_TEST_AUPRC":    ("final_metrics.yaml M2.test_auprc", "AUPRC", 4),
    "M2_TEST_AUROC":    ("final_metrics.yaml M2.test_auroc", "AUROC", 4),
    "M3_VAL_AUPRC":     ("final_metrics.yaml M3.val_auprc",  "AUPRC", 4),
    "M3_TEST_AUPRC":    ("final_metrics.yaml M3.test_auprc", "AUPRC", 4),
    "M3_TEST_AUROC":    ("final_metrics.yaml M3.test_auroc", "AUROC", 4),
    # anchors (mean caliber unless noted)
    "M2_BENBOW_MEAN":   ("real_anchor_benbow_m2final.json results.mean.auc", "AUROC", 4),
    "M2_BENBOW_MAX":    ("real_anchor_benbow_m2final.json results.max.auc",  "AUROC", 4),
    "M2_IV4_MAX":       ("real_anchor_iv4_m2final.json results.max.auc",     "AUROC", 4),
    "M2_IV4_MEAN":      ("real_anchor_iv4_m2final.json results.mean.auc",    "AUROC", 4),
    "M2_ICE_MEAN":      ("real_anchor_ice_m2final.json results.mean.auc",    "AUROC", 4),
    "M2_PHILYMPICS_MEAN": ("real_anchor_philympics_m2final.json results.mean.auc", "AUROC", 4),
    "M3_BENBOW_MEAN":   ("real_anchor_benbow_m3final.json results.mean.auc", "AUROC", 4),
    "M3_IV4_MAX":       ("real_anchor_iv4_m3final.json results.max.auc",     "AUROC", 4),
    "M3_ICE_MEAN":      ("real_anchor_ice_m3final.json results.mean.auc",    "AUROC", 4),
    "M3_PHILYMPICS_MEAN": ("real_anchor_philympics_m3final.json results.mean.auc", "AUROC", 4),
    # OOD
    "M2_OOD_MEAN":      ("ood_m2final.json genome_stats.mean",  "AUPRC", 4),
    "M2_OOD_GE07":      ("ood_m2final.json genome_stats.frac_ge_07", "%", 1),
    "M3_OOD_MEAN":      ("ood_m3final.json genome_stats.mean",  "AUPRC", 4),
    # difficulty
    "M2_DIFF_D1":       ("difficulty_D1_m2final.json results.max.auprc", "AUPRC", 4),
    "M2_DIFF_D2":       ("difficulty_D2_m2final.json results.max.auprc", "AUPRC", 4),
    "M2_DIFF_D3":       ("difficulty_D3_m2final.json results.max.auprc", "AUPRC", 4),
    "M3_DIFF_D1":       ("difficulty_D1_m3final.json results.max.auprc", "AUPRC", 4),
    "M3_DIFF_D3":       ("difficulty_D3_m3final.json results.max.auprc", "AUPRC", 4),
    # derived
    "M2_RECOVERY_PCT":  ("100 * M2_VAL_AUPRC / 0.951 (N4)",  "%", 1),
    "M2_MINUS_M3":      ("M2_VAL_AUPRC - M3_VAL_AUPRC",      "AUPRC", 4),
    "M2_MINUS_M0":      ("M2_VAL_AUPRC - 0.8325",            "AUPRC", 4),
    "M3_MINUS_M0":      ("M3_VAL_AUPRC - 0.8325",            "AUPRC", 4),
    # ECE / calibration
    "M2_ECE":           ("final_metrics.yaml M2.ece", "ECE", 3),
    # adapter RAP
    "M2_ADAPTER_RAP":   ("rap_adapter_m2final.json auroc",   "AUROC", 4),
}

# token aliases: canonical → [all placeholder token names that should receive this value]
ALIASES = {
    "M2_VAL_AUPRC":     ["INSERT_M2_VAL_AUPRC", "INSERT_M2_SIM_VAL_AUPRC"],
    "M2_VAL_AUROC":     ["INSERT_M2_VAL_AUROC"],
    "M2_TEST_AUPRC":    ["INSERT_M2_TEST_AUPRC", "INSERT_M2_SIM_TEST_AUPRC"],
    "M2_TEST_AUROC":    ["INSERT_M2_TEST_AUROC", "INSERT_M2_SIM_TEST_AUROC"],
    "M3_VAL_AUPRC":     ["INSERT_M3_VAL_AUPRC"],
    "M3_TEST_AUPRC":    ["INSERT_M3_TEST_AUPRC", "INSERT_M3_SIM_TEST_AUPRC"],
    "M3_TEST_AUROC":    ["INSERT_M3_TEST_AUROC", "INSERT_M3_SIM_TEST_AUROC"],
    "M2_BENBOW_MEAN":   ["INSERT_M2_BENBOW_MEAN"],
    "M2_BENBOW_MAX":    ["INSERT_M2_BENBOW_MAX"],
    "M2_IV4_MAX":       ["INSERT_M2_IV4_AUROC"],
    "M2_ICE_MEAN":      ["INSERT_M2_ICE_MEAN"],
    "M2_PHILYMPICS_MEAN": ["INSERT_M2_PHILYMPICS_MEAN"],
    "M2_OOD_MEAN":      ["INSERT_M2_OOD_PERGENOME_MEAN", "INSERT_M2_OOD_BLOCK"],
    "M2_OOD_GE07":      ["INSERT_M2_OOD_SHARE_GE07"],
    "M3_OOD_MEAN":      ["INSERT_M3_OOD_PERGENOME_MEAN"],
    "M2_ECE":           ["INSERT_M2_ECE_CHECK"],
    "M2_ADAPTER_RAP":   ["INSERT_M2_ADAPTER_RAP_CELL"],
    "M2_RECOVERY_PCT":  ["INSERT_RECOVERY_RATE_M2"],
    "M2_MINUS_M3":      ["INSERT_M2_MINUS_M3_SIM_DELTA", "INSERT_M2_MINUS_M3_CONTRAST",
                         "INSERT_M2_MINUS_M3_SIM_DELTA_CI"],
    "M2_DIFF_D1":       ["INSERT_M2_DIFFICULTY_ROW"],
    "M3_DIFF_D1":       ["INSERT_M3_DIFFICULTY_ROW"],
}

# files to scan for token occurrences (relative to PKG)
SCAN_FILES = []
for sub in ("01_MANUSCRIPT", "04_SUPPLEMENTARY", "editorial"):
    d = os.path.join(PKG, sub)
    if os.path.isdir(d):
        for base, _dirs, files in os.walk(d):
            for fn in sorted(files):
                if fn.endswith((".md", ".yaml")):
                    SCAN_FILES.append(os.path.relpath(os.path.join(base, fn), PKG))

TOKEN_RE = re.compile(r"\[(INSERT_[A-Z0-9_]+|FROM_CROSSTASK_CSV)[^\]]*\]")


def main():
    binding = {}
    for ckey, (src, unit, dp) in CANONICAL.items():
        tokens = ALIASES.get(ckey, [f"INSERT_{ckey}"])
        locations = defaultdict(list)  # file → [line numbers]
        for rel in SCAN_FILES:
            path = os.path.join(PKG, rel)
            try:
                for i, line in enumerate(open(path, encoding="utf-8"), 1):
                    for tok in tokens:
                        if tok in line:
                            locations[rel].append(i)
            except OSError:
                continue
        binding[ckey] = {
            "canonical_source": src,
            "unit": unit, "decimal_places": dp,
            "tokens": tokens,
            "occurrences": {f: ls for f, ls in locations.items() if ls},
            "total_occurrences": sum(len(ls) for ls in locations.values()),
        }

    json.dump(binding, open(OUT_JSON, "w"), indent=2)
    with open(OUT_MD, "w") as fh:
        fh.write("# FINAL RESULT BINDING MAP (P2-②)\n\n")
        fh.write("Every canonical terminal metric → every file/line it must appear in.\n")
        fh.write(f"Generated: 2026-09-04 · {len(binding)} canonical keys\n\n")
        fh.write("| canonical | source | unit | dp | tokens | occurrences (files) |\n|---|---|---|---|---|---|\n")
        for ck, b in sorted(binding.items()):
            occ_summary = "; ".join(f"{f}×{len(ls)}" for f, ls in b["occurrences"].items())
            fh.write(f"| {ck} | {b['canonical_source'][:50]} | {b['unit']} | {b['decimal_places']} | "
                     f"{len(b['tokens'])} | {b['total_occurrences']} in {len(b['occurrences'])} files |\n")
        # detail
        fh.write("\n## Detail (per canonical key)\n\n")
        for ck, b in sorted(binding.items()):
            fh.write(f"### {ck}\n")
            fh.write(f"- Source: `{b['canonical_source']}`\n")
            fh.write(f"- Unit: {b['unit']} | Decimal places: {b['decimal_places']}\n")
            fh.write(f"- Tokens: {', '.join(b['tokens'])}\n")
            if b["occurrences"]:
                fh.write(f"- Appears in:\n")
                for f, ls in b["occurrences"].items():
                    fh.write(f"  - `{f}` lines {ls}\n")
            else:
                fh.write(f"- No live token occurrences (may be filled or structural)\n")
            fh.write("\n")

    print(f"canonical keys: {len(binding)} | total token occurrences: "
          f"{sum(b['total_occurrences'] for b in binding.values())}")
    zero = [ck for ck, b in binding.items() if b["total_occurrences"] == 0]
    if zero:
        print(f"keys with zero live tokens (expected for derived/structural): {len(zero)}")
    print(f"-> {OUT_MD}\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
