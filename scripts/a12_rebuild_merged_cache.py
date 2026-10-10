#!/usr/bin/env python3
"""a12_rebuild_merged_cache.py  —  v2 deterministic rebuild of evo_cache/anchors/merged.pt [FIX 2026-09-08]

v1 lessons (three-layer defect; see docs/evidence/V7_F1_ZERO_FILL_VERIFICATION_20260908.md):
  ① a12_evo_anchor_completion.sh deleted merged.pt immediately after generating it (the rm line);
  ② even when merged.pt existed, the embedding key domain (all Fasta headers) did not match the
     evaluation sample ID domain → cache_hit=0 → zero-fill;
  ③ the QC ID-alignment check read window_stats (contig_ids does not exist) = empty set, skipped → false PASS.

v2 plan: rebuild keyed on the "evaluation sample set"  —  use exactly the same reader function as
interim_eval.py to rebuild {seq_id: (label, seq)}, map existing embeddings into the evaluation
ID domain by the same reader rules, and export missing sequences (300 IV4 flank negatives +
207 ICE paired negatives) as FASTA for precompute_evo2_embeddings.py to fill in
(header=evaluation ID → the key domain is aligned by construction).

usage:
  python scripts/a12_rebuild_merged_cache.py --plan    # rebuild sample sets + mapping + export missing FASTA
  python scripts/a12_rebuild_merged_cache.py --final   # merge (including filled-in embeddings) → merged.pt
"""
import argparse, glob, json, os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)  # the reader module uses ROOT-relative path constants
sys.path.insert(0, os.path.join(ROOT, "scripts"))
ANCHOR_DIR = os.path.join(ROOT, "evo_cache", "anchors")
OUT = os.path.join(ANCHOR_DIR, "merged.pt")
MISSING_DIR = os.path.join(ANCHOR_DIR, "_missing_20260908")
FC = os.path.join(ROOT, "V7_FINAL_CANDIDATE_CURRENT_BEST")
NPZ = {  # pre-fix (zero-fill era) evaluation npz: used for deterministic-rebuild cross-checking
    "benbow":     f"{FC}/09_EVAL/_archive_zerofill_20260908/real_anchor_benbow_m2best_interim.npz",
    "iv4":        f"{FC}/09_EVAL/_archive_zerofill_20260908/real_anchor_iv4_m2best_interim.npz",
    "ice":        f"{FC}/09_EVAL/_archive_zerofill_20260908/real_anchor_ice_m2best_interim.npz",
    "philympics": f"{FC}/09_EVAL/_archive_zerofill_20260908/real_anchor_philympics_m2best_interim.npz",
}

def load_raw(anchor):
    return torch.load(os.path.join(ANCHOR_DIR, f"{anchor}.pt"),
                      map_location="cpu", weights_only=False)

def map_keys(anchor, raw):
    """Raw embedding keys → evaluation ID domain (replicates the eval reader's ID construction rules)."""
    out = {}
    for k, v in raw.items():
        if anchor == "benbow":
            parts = k.split("|")                      # read_benbow: parts[2].split()[0]
            ek = parts[2].split()[0] if len(parts) > 2 and parts[2].strip() else parts[0]
        elif anchor == "ice":
            parts = k.split("|")                      # read_iceberg: parts[1] → ICE_{iid}
            ek = "ICE_" + (parts[1] if len(parts) > 1 else k)
        else:                                          # iv4 / philympics: key domain is the evaluation domain
            ek = k
        assert ek not in out, f"{anchor}: mappingconflict {ek}"
        out[ek] = v
    return out

def reconstruct_samples():
    """Rebuild the evaluation sample set {sid: (label, seq)} with the same reader as interim_eval.py."""
    import eval_real_anchors as cpu
    sets = {}
    sets["benbow"] = cpu.read_benbow(cpu.BENBOW_FASTA)
    sets["iv4"] = cpu.build_iv4_300_300(cpu.IV4_DIR)
    ice = cpu.read_iceberg(cpu.ICE_FASTA)
    sets["ice"], _neg_meta = cpu.build_ice_anchor(ice, cpu.IV4_DIR, n_neg=len(ice))
    # philympics: key domain already directly aligned (104/104), and the reader lives in the gpu module; use the npz ID set as authoritative
    z = np.load(NPZ["philympics"], allow_pickle=True)
    sets["philympics"] = {str(s): (None, None) for s in z["contig_ids"]}
    return sets

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    assert args.plan or args.final, "need --plan or --final"

    samples = reconstruct_samples()

    # ── cross-check: deterministically rebuilt ID set == pre-fix evaluation npz ID set ──
    for a, sset in samples.items():
        z = np.load(NPZ[a], allow_pickle=True)
        eval_ids = {str(s) for s in z["contig_ids"]}
        assert set(sset.keys()) == eval_ids, \
            f"{a}: rebuilt ID set ({len(sset)}) != evaluation npz ({len(eval_ids)})  —  reader drift, abort"
    print("[align] 4/4 anchors: deterministically rebuilt ID set == pre-fix evaluation npz ID set (per-anchor cross-check PASS)")

    mapped = {a: map_keys(a, load_raw(a)) for a in ["benbow", "iv4", "ice", "philympics"]}
    merged, report, missing = {}, {}, {}
    for a in ["benbow", "iv4", "ice", "philympics"]:
        sids = set(samples[a].keys())
        hit = {k: v for k, v in mapped[a].items() if k in sids}
        merged.update(hit)
        miss = sorted(sids - set(hit))
        missing[a] = miss
        report[a] = {"eval_n": len(sids), "mapped": len(hit), "missing": len(miss)}

    if args.plan:
        os.makedirs(MISSING_DIR, exist_ok=True)
        for a, miss in missing.items():
            if not miss:
                continue
            fa = os.path.join(MISSING_DIR, f"{a}_missing.fasta")
            with open(fa, "w") as f:
                for sid in miss:
                    label, seq = samples[a][sid]
                    assert seq and set(seq.upper()) <= set("ACGTN"), f"{a}/{sid}: sequenceexception"
                    f.write(f">{sid}\n{seq.upper()}\n")
            bp = sum(len(samples[a][s][1]) for s in miss)
            report[a]["missing_bp"] = int(bp)
            report[a]["missing_fasta"] = fa
        print(json.dumps({"stage": "plan", "per_anchor": report,
                          "mapped_total": len(merged)}, indent=1, ensure_ascii=False))
        return

    # --final: merge in the filled-in embeddings (key domain=evaluation ID), full-coverage assert, then save
    for p in sorted(glob.glob(os.path.join(MISSING_DIR, "*_missing.pt"))):
        d = torch.load(p, map_location="cpu", weights_only=False)
        merged.update(d)
        print(f"  [final] merged in {os.path.basename(p)}: {len(d)}")
    need = {a: set(samples[a].keys()) for a in samples}
    all_need = set().union(*need.values())
    still = sorted(all_need - set(merged))
    assert not still, f"{len(still)} evaluation samples still lack embeddings: {still[:5]}"

    bad = {"dim": [], "nan": [], "inf": [], "zero": []}
    for sid, vec in merged.items():
        v = np.asarray(vec)
        if v.shape[0] != 4096: bad["dim"].append(sid)
        if np.isnan(v).any(): bad["nan"].append(sid)
        if np.isinf(v).any(): bad["inf"].append(sid)
        if np.allclose(v, 0): bad["zero"].append(sid)
    assert not any(bad.values()), f"integrity failure: {bad}"

    torch.save(merged, OUT)
    print(json.dumps({"stage": "final", "saved": OUT, "entries": len(merged),
                      "per_anchor": report,
                      "integrity": "PASS(dim=4096,NaN=0,Inf=0,zero-vec=0)",
                      "coverage": "PASS (100% of evaluation sample IDs hit across all 4 anchors)"},
                     indent=1, ensure_ascii=False))

if __name__ == "__main__":
    main()
