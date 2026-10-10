# H-01 unit test (CODE_AUDIT_V7, HIGH): negative-type preservation + hard/easy stratified sets
# Pure CPU (CUDA_VISIBLE_DEVICES=""), synthetic data written to /tmp, touches no repo data or running processes.
#
# assert:
#   1. parser: all 10 negative types in Contig mode + PE mode (pe_case column) are kept
#      with their true identity in HGTSample.sample_type; unknown NEG_* pass through; positives stay "POS".
#   2. metrics: hard pool = {NEG_HARD, HOMOLOGY, CONSERVED, LOW_COMPLEXITY,
#      GC_MATCHED_CHIMERIC, AMALGAMATED_HGT, PARTIAL_VIRUS_HIT}, easy pool = {NEG};
#      NEG_VIRUS/NEG_PCR enter neither pool (their probability shifts do not affect hard/easy AUC);
#      per_type_counts / neg_specificity_by_type cover all 10 types.
#   3. legacy control: the pre-fix caliber (6 types collapsed into NEG + the old set),
#      recomputed on the same predictions, has easy_neg_auc contaminated (< post-fix), quantifying the H-01 impact.
import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_pipeline.impl.parser import BEDParser, NEG_TYPE_MAP
from evaluation.impl.metrics import (
    compute_stratified_metrics, _HARD_NEG_TYPES, _EASY_NEG_TYPES)

results = {}
SEQ = "ACGT" * 12   # 48 bp

CONTIG_TYPES = [
    "NEG", "NEG_VIRUS", "NEG_HARD", "NEG_PCR", "NEG_CONSERVED",
    "NEG_HOMOLOGY", "NEG_LOW_COMPLEXITY", "NEG_GC_MATCHED_CHIMERIC",
    "NEG_AMALGAMATED_HGT", "NEG_PARTIAL_VIRUS_HIT",
]

# ────────────────────────────────────────────────────────────────────────────── #
# 1. parser: Contig mode
# ────────────────────────────────────────────────────────────────────────────── #
with tempfile.TemporaryDirectory() as td:
    bed = os.path.join(td, "breakpoints.bed")
    fa = os.path.join(td, "dataset.fasta")
    rows = ["#seq_id\tevent_idx\tlabel\tseq_len\thgt_start\thgt_end\tvirus_len\thgt_mode"]
    rows.append(f"pos_probe\t0\t1\t{len(SEQ)}\t5\t30\t25\tinsertion")
    for t in CONTIG_TYPES + ["NEG_FUTURE_PROBE"]:
        rows.append(f"{t}\t0\t0\t{len(SEQ)}\t-1\t-1\t0\t{t}")
    with open(bed, "w") as f:
        f.write("\n".join(rows) + "\n")
    with open(fa, "w") as f:
        f.write(f">pos_probe\n{SEQ}\n")
        for t in CONTIG_TYPES + ["NEG_FUTURE_PROBE"]:
            f.write(f">{t}\n{SEQ}\n")

    samples = BEDParser(bed, td).parse()
    got = {s.seq_id: s.sample_type for s in samples}
    assert got["pos_probe"] == "POS", got
    for t in CONTIG_TYPES:
        assert got[t] == t, f"contig-mode type collapsed: {t} -> {got[t]}"
    assert got["NEG_FUTURE_PROBE"] == "NEG_FUTURE_PROBE", \
        f"unknown NEG_* should pass through, not collapse: {got['NEG_FUTURE_PROBE']}"
    results["parser_contig"] = {"n": len(samples),
                                "types_preserved": CONTIG_TYPES,
                                "unknown_passthrough": True}

# ────────────────────────────────────────────────────────────────────────────── #
# 2. parser: PE mode (pe_case column carries neg_type)
# ────────────────────────────────────────────────────────────────────────────── #
with tempfile.TemporaryDirectory() as td:
    bed = os.path.join(td, "breakpoints.bed")
    r1 = os.path.join(td, "R1.fasta")
    r2 = os.path.join(td, "R2.fasta")
    pe_types = ["NEG", "NEG_HARD", "NEG_GC_MATCHED_CHIMERIC",
                "NEG_PE_HOST_DIVERSITY", "NEG_FUTURE_PROBE"]
    rows = ["#" + "\t".join(
        ["seq_id", "event_idx", "label", "insert_len", "hgt_start", "hgt_end",
         "virus_len", "hgt_mode", "pe_case", "r1_vs", "r1_ve", "r2_vs", "r2_ve"])]
    rows.append("\t".join(["pos_probe", "0", "1", "300", "5", "30", "25",
                           "insertion", "both_overlap", "10", "30", "10", "30"]))
    for t in pe_types:
        rows.append("\t".join([t, "0", "0", "300", "-1", "-1", "0",
                               "none", t, "-1", "-1", "-1", "-1"]))
    with open(bed, "w") as f:
        f.write("\n".join(rows) + "\n")
    with open(r1, "w") as f:
        f.write(">pos_probe\n" + SEQ + "\n")
        for t in pe_types:
            f.write(f">{t}\n{SEQ}\n")
    with open(r2, "w") as f:
        f.write(">pos_probe\n" + SEQ + "\n")
        for t in pe_types:
            f.write(f">{t}\n{SEQ}\n")

    samples = BEDParser(bed, td).parse()
    got = {s.seq_id: s.sample_type for s in samples}
    assert got["pos_probe"] == "POS", got
    for t in pe_types:
        assert got[t] == t, f"PE-mode type collapsed: {t} -> {got[t]}"
    results["parser_pe"] = {"n": len(samples), "types_preserved": pe_types}

# ────────────────────────────────────────────────────────────────────────────── #
# 3. metrics: hard/easy bucketing spec
# ────────────────────────────────────────────────────────────────────────────── #
from sklearn.metrics import roc_auc_score

assert _HARD_NEG_TYPES == {
    "NEG_HARD", "NEG_HOMOLOGY", "NEG_CONSERVED", "NEG_LOW_COMPLEXITY",
    "NEG_GC_MATCHED_CHIMERIC", "NEG_AMALGAMATED_HGT", "NEG_PARTIAL_VIRUS_HIT"}, \
    _HARD_NEG_TYPES
assert _EASY_NEG_TYPES == {"NEG"}, _EASY_NEG_TYPES

types, labels, probs = [], [], []
def add(t, lab, p, n=2):
    for _ in range(n):
        types.append(t); labels.append(lab); probs.append(p)

add("POS", 1, 0.80, 4)
add("NEG", 0, 0.20, 3)
for t in ["NEG_HARD", "NEG_HOMOLOGY", "NEG_CONSERVED", "NEG_LOW_COMPLEXITY",
          "NEG_GC_MATCHED_CHIMERIC", "NEG_AMALGAMATED_HGT", "NEG_PARTIAL_VIRUS_HIT"]:
    add(t, 0, 0.60)
add("NEG_VIRUS", 0, 0.99)   # not in the hard/easy pools
add("NEG_PCR", 0, 0.99)     # not in the hard/easy pools
labels = np.array(labels); probs = np.array(probs)

strat = compute_stratified_metrics(probs, labels, types, threshold=0.5)

# Expected values: buckets selected by hand per the approved spec (single-source-of-truth control)
idx_pos = np.where(labels == 1)[0]
idx_hard = np.array([i for i in range(len(types))
                     if labels[i] == 0 and types[i] in _HARD_NEG_TYPES])
idx_easy = np.array([i for i in range(len(types))
                     if labels[i] == 0 and types[i] in _EASY_NEG_TYPES])
exp_hard = roc_auc_score(labels[np.r_[idx_pos, idx_hard]], probs[np.r_[idx_pos, idx_hard]])
exp_easy = roc_auc_score(labels[np.r_[idx_pos, idx_easy]], probs[np.r_[idx_pos, idx_easy]])
assert abs(strat["hard_neg_auc"] - exp_hard) < 1e-12, (strat["hard_neg_auc"], exp_hard)
assert abs(strat["easy_neg_auc"] - exp_easy) < 1e-12, (strat["easy_neg_auc"], exp_easy)
assert exp_hard == 1.0 and exp_easy == 1.0   # under this construction the two pools should be perfectly separable

# per-type completeness
assert set(strat["per_type_counts"]) == set(types), strat["per_type_counts"]
assert all(t in strat["neg_specificity_by_type"] for t in CONTIG_TYPES), \
    strat["neg_specificity_by_type"]

# Invariant: flipping NEG_VIRUS/NEG_PCR probabilities must not affect hard/easy AUC (not in the pools)
probs2 = [0.01 if types[i] in ("NEG_VIRUS", "NEG_PCR") else probs[i]
          for i in range(len(types))]
strat2 = compute_stratified_metrics(np.array(probs2), labels, types, 0.5)
assert strat2["hard_neg_auc"] == strat["hard_neg_auc"]
assert strat2["easy_neg_auc"] == strat["easy_neg_auc"]

# legacy control: pre-fix caliber (parser collapses 6 types + the old set) recomputed on the same predictions → easy pool contaminated
legacy_known = {"NEG", "NEG_VIRUS", "NEG_HARD", "NEG_PCR"}
legacy_types = ["NEG" if (t not in legacy_known and t != "POS") else t for t in types]
legacy_hard_set = {"NEG_HARD", "NEG_HOMOLOGY", "NEG_CONSERVED", "NEG_LOW_COMPLEXITY"}
legacy_easy_set = {"NEG", "NEG_VIRUS", "NEG_PCR"}
idx_lhard = np.array([i for i in range(len(legacy_types))
                      if labels[i] == 0 and legacy_types[i] in legacy_hard_set])
idx_leasy = np.array([i for i in range(len(legacy_types))
                      if labels[i] == 0 and legacy_types[i] in legacy_easy_set])
legacy_easy_auc = roc_auc_score(labels[np.r_[idx_pos, idx_leasy]],
                                probs[np.r_[idx_pos, idx_leasy]])
assert legacy_easy_auc < strat["easy_neg_auc"], \
    f"the legacy easy pool should be contaminated (collapsed types/virus/pcr mixed in): {legacy_easy_auc}"

results["metrics"] = {
    "hard_neg_auc": strat["hard_neg_auc"],
    "easy_neg_auc": strat["easy_neg_auc"],
    "hard_types": sorted(_HARD_NEG_TYPES),
    "easy_types": sorted(_EASY_NEG_TYPES),
    "virus_pcr_invariance": True,
    "per_type_counts": strat["per_type_counts"],
    "legacy_easy_neg_auc_same_predictions": float(legacy_easy_auc),
}
results["neg_type_map_size"] = len(NEG_TYPE_MAP)
results["status"] = "PASS"

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs", "evidence", "bugfix", "negtype_strat_after.json")
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"PASS: parser keeps {len(CONTIG_TYPES)} contig types (+PE/unknown passthrough); "
      f"hard=7 types / easy=pure NEG; virus/pcr invariant; "
      f"legacy easy_auc={legacy_easy_auc:.4f} < fixed={strat['easy_neg_auc']:.4f}")
print(f"evidence -> {out}")
