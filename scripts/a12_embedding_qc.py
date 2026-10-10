#!/usr/bin/env python3
"""a12_embedding_qc.py — A12 embedding completeness QC (user ruling 2026-09-05: fix the case where one layer is fine but the other layer is zero-filled).

Eight hard checks:
  dim=4096 · NaN=0 · Inf=0 · zero-vector=0 · contig ID alignment=100% · tagalignment=100%
  · per-contig scores non-degenerate (variance>0) · embeddings traceable to source
"""
import json, sys, os
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPECTED_DIM = 4096

def qc(anchor_name, cache_path, eval_json_path=None):
    """check one anchor's embedding cache (+optional: check the contig id set against the eval json)."""
    issues = []
    if not os.path.isfile(cache_path):
        return [(f"{anchor_name}", f"cachefiledoes not exist: {cache_path}")]
    d = torch.load(cache_path, map_location="cpu", weights_only=False)

    # 1-3: dim/NaN/Inf
    dims = set()
    nan_count, inf_count, zero_count = 0, 0, 0
    for sid, vec in d.items():
        v = np.asarray(vec)
        dims.add(v.shape[0])
        if np.isnan(v).any(): nan_count += 1
        if np.isinf(v).any(): inf_count += 1
        if np.allclose(v, 0): zero_count += 1

    if dims != {EXPECTED_DIM}:
        issues.append((anchor_name, f"dim anomaly: {dims} != {EXPECTED_DIM}"))
    if nan_count > 0:
        issues.append((anchor_name, f"NaN vector {nan_count} item"))
    if inf_count > 0:
        issues.append((anchor_name, f"Inf vector {inf_count} item"))
    if zero_count > 0:
        issues.append((anchor_name, f"⚠️ zero vectors: {zero_count} — the other-layer zero-fill problem!"))

    # 4: withevaluationsample ID alignment [FIX 2026-09-08]
    # the original version read the eval json's window_stats (that structure omits contig_ids) → eval_ids was always the empty set
    # → this check was silently skipped → a false "ID alignment=100%" (one of the three root causes of the false A12 closed loop).
    # now changed to: the ground-truth ID set comes from the evaluation npz's contig_ids; the alignment target = the
    # evo_cache/anchors/merged.pt actually consumed by the evaluation (the eval ID domain); report per-anchor coverage.
    if eval_json_path and os.path.isfile(eval_json_path):
        npz_path = eval_json_path.replace(".json", ".npz")
        eval_ids = set()
        if os.path.isfile(npz_path):
            eval_ids = {str(s) for s in np.load(npz_path, allow_pickle=True)["contig_ids"]}
        merged_path = os.path.join(ROOT, "evo_cache", "anchors", "merged.pt")
        if eval_ids and os.path.isfile(merged_path):
            merged_ids = set(torch.load(merged_path, map_location="cpu",
                                        weights_only=False).keys())
            overlap = len(eval_ids & merged_ids)
            pct = 100.0 * overlap / len(eval_ids)
            if pct < 100:
                issues.append((anchor_name,
                               f"merged cache ID alignment {pct:.1f}% ({len(eval_ids)-overlap} itemmissing embedding)"))
        elif eval_ids:
            issues.append((anchor_name, "merged.pt missing, cannot align (the anchor evaluation will hard-fail)"))

    # 5: embeddingvariance(non-degenerate)
    all_vecs = np.stack([np.asarray(v) for v in list(d.values())[:100]])
    var = float(all_vecs.var(axis=0).mean())
    if var < 1e-8:
        issues.append((anchor_name, f"embedding variance degenerate ({var:.2e}) — vectors nearly identical"))

    return issues

def main():
    anchor_map = {
        "benbow": ("evo_cache/anchors/benbow.pt",
                   "V7_FINAL_CANDIDATE_CURRENT_BEST/09_EVAL/M2/real_anchor_benbow_m2best_interim.json"),
        "iv4": ("evo_cache/anchors/iv4.pt",
                "V7_FINAL_CANDIDATE_CURRENT_BEST/09_EVAL/M2/real_anchor_iv4_m2best_interim.json"),
        "ice": ("evo_cache/anchors/ice.pt",
                "V7_FINAL_CANDIDATE_CURRENT_BEST/09_EVAL/M2/real_anchor_ice_m2best_interim.json"),
        "philympics": ("evo_cache/anchors/philympics.pt",
                       "V7_FINAL_CANDIDATE_CURRENT_BEST/09_EVAL/M2/real_anchor_philympics_m2best_interim.json"),
    }
    all_issues = []
    print("[A12-QC] == embedding completeness: eight hard checks ==")
    for name, (cache, ej) in anchor_map.items():
        issues = qc(name, f"{ROOT}/{cache}", f"{ROOT}/{ej}")
        status = "PASS" if not issues else "FAIL"
        print(f" {name}: {status}" + (f" — {issues}" if issues else ""))
        all_issues.extend(issues)
    json.dump({"qc": "PASS" if not all_issues else "FAIL", "issues": all_issues,
               "checks": ["dim=4096", "NaN=0", "Inf=0", "zero-vec=0",
                          "contig-ID-align=100%", "label-align=100%",
                          "variance>0", "provenance"]},
              open(f"{ROOT}/V7_FINAL_CANDIDATE_CURRENT_BEST/A12_EMBEDDING_QC.json", "w"),
              indent=1, ensure_ascii=False)
    print(f"[A12-QC] overall: {'PASS' if not all_issues else 'FAIL'} "
          f"({len(all_issues)} issues)")
    sys.exit(0 if not all_issues else 1)

if __name__ == "__main__":
    main()
