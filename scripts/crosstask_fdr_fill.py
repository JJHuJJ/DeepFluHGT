#!/usr/bin/env python
"""Cross-task DeLong family BH-FDR fill for [INSERT_CROSSTASK_FDR_COLUMN].

RED-ACTION token (fillable NOW, not M2-dependent):
  source  = benchmark/crosstask/crosstask_full_delong.csv (70 paired contrasts)
  target  = 5_submission_final/01_MANUSCRIPT/4_PLACEHOLDER_TABLES.md Table ST2
            (one FDR cell per dataset row, 5 rows)
  method  = Benjamini-Hochberg FDR at alpha=0.05 over ALL 70 contrasts
            (one family; Methods section 10 / manuscript L332 already state
             the family summary -- this script VERIFIES those pre-stated
             numbers before writing anything).

Verification gates (script exits 1 on any mismatch, writes nothing):
  V1  family size == 70
  V2  nominally significant (p<=0.05) count == 55
  V3  survivors at q<0.05 == 53
  V4  promoter_r ridge  concat-vs-ntv3  p=0.041 -> q=0.053 (non-survivor #1)
  V5  ecoli_4mc linear  concat-vs-evo   p=0.050 -> q=0.063 (non-survivor #2)
  V6  promoter_b linear concat-vs-evo   p=0.035 -> q=0.047 (marginal survivor)
  V7  human_vs_worm linear evo-vs-ntv3  p=0.027 -> q=0.037 (marginal survivor)

Cell format (compact, per dataset row):
  "<surv>/<nominally> nom. sig. at p<=0.05; max q=<q> (<worst contrast>)"
  where worst contrast = highest-q nominally-significant contrast in that
  dataset; datasets with zero non-survivors report "all survive".

Run:  python scripts/crosstask_fdr_fill.py [--apply]
      default = verify-only (no files touched)
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "benchmark", "crosstask", "crosstask_full_delong.csv")
OUT_JSON = os.path.join(ROOT, "docs", "evidence", "CROSSTASK_FDR_FILL.json")

DATASET_LABELS = {
    "human_vs_worm": "human_vs_worm",
    "promoter_b_amyloliquefaciens": "promoter_b",
    "promoter_r_capsulatus": "promoter_r",
    "celegans_4mc": "celegans_4mC",
    "ecoli_4mc": "ecoli_4mC",
}


def bh_fdr(pvals):
    """Benjamini-Hochberg q-values (step-up, monotone enforced)."""
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    q = [0.0] * n
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        pos = n - rank + 1          # 1-based rank of this p (ascending)
        val = min(prev, pvals[idx] * n / pos)
        q[idx] = val
        prev = val
    return q


def fmt_q(x):
    return f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write cells into 4_PLACEHOLDER_TABLES.md (default: verify only)")
    args = ap.parse_args()

    rows = []
    with open(CSV) as fh:
        for r in csv.DictReader(fh):
            r["p"] = float(r["p"])
            r["delta_auc"] = float(r["delta_auc"])
            rows.append(r)

    errs = []

    # V1 family size
    if len(rows) != 70:
        errs.append(f"V1 FAIL: family size {len(rows)} != 70")

    q = bh_fdr([r["p"] for r in rows])
    for r, qi in zip(rows, q):
        r["q"] = qi

    # V2/V3 counts
    nom = [r for r in rows if r["p"] <= 0.05]
    surv = [r for r in nom if r["q"] < 0.05]
    if len(nom) != 55:
        errs.append(f"V2 FAIL: nominally significant {len(nom)} != 55")
    if len(surv) != 53:
        errs.append(f"V3 FAIL: survivors {len(surv)} != 53")

    def find(ds, probe, ga, gb):
        for r in rows:
            if (r["dataset"] == ds and r["probe"] == probe
                    and r["group_a"] == ga and r["group_b"] == gb):
                return r
        return None

    # V4-V7 named contrasts (manuscript L332 pre-stated values)
    named = [
        ("V4", "promoter_r_capsulatus", "ridge", "evo+ntv3", "ntv3_1536", 0.041, 0.053, True),
        ("V5", "ecoli_4mc", "linear", "evo+ntv3", "evo_4096", 0.050, 0.063, True),
        ("V6", "promoter_b_amyloliquefaciens", "linear", "evo+ntv3", "evo_4096", 0.035, 0.047, False),
        ("V7", "human_vs_worm", "linear", "evo_4096", "ntv3_1536", 0.027, 0.037, False),
    ]
    for vid, ds, probe, ga, gb, p_exp, q_exp, is_dead in named:
        r = find(ds, probe, ga, gb)
        if r is None:
            errs.append(f"{vid} FAIL: contrast {ds}/{probe}/{ga}-vs-{gb} not in CSV")
            continue
        ok_p = abs(r["p"] - p_exp) <= 5e-4 + 1e-6   # CSV stores full precision
        ok_q = abs(r["q"] - q_exp) <= 5.5e-4
        ok_dead = (r["q"] >= 0.05) == is_dead
        if not (ok_p and ok_q and ok_dead):
            errs.append(f"{vid} FAIL: {ds}/{probe}/{ga}-vs-{gb} "
                        f"p={r['p']:.4f}(exp {p_exp}) q={r['q']:.4f}(exp {q_exp}) dead={r['q']>=0.05}")

    report = {
        "family_size": len(rows),
        "nominally_significant": len(nom),
        "survivors_q05": len(surv),
        "non_survivors": [
            {"dataset": r["dataset"], "probe": r["probe"],
             "contrast": f"{r['group_a']} vs {r['group_b']}",
             "p": round(r["p"], 6), "q": round(r["q"], 4)}
            for r in nom if r["q"] >= 0.05],
        "per_dataset": {},
        "checks_passed": not errs,
        "errors": errs,
    }

    # per-dataset cells
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)
    cells = {}
    for ds, rs in by_ds.items():
        ds_nom = [r for r in rs if r["p"] <= 0.05]
        ds_surv = [r for r in ds_nom if r["q"] < 0.05]
        ds_dead = [r for r in ds_nom if r["q"] >= 0.05]
        if not ds_nom:
            cells[ds] = "no nominally significant contrast (all q n/a)"
        elif not ds_dead:
            worst = max(ds_nom, key=lambda r: r["q"])
            cells[ds] = f"all {len(ds_surv)}/{len(ds_nom)} survive; max q = {fmt_q(worst['q'])}"
        else:
            worst_dead = max(ds_dead, key=lambda r: r["q"])
            cells[ds] = (f"{len(ds_surv)}/{len(ds_nom)} survive; "
                         f"max q = {fmt_q(worst_dead['q'])} "
                         f"({worst_dead['probe']}, {worst_dead['group_a']} vs {worst_dead['group_b']})")
        report["per_dataset"][DATASET_LABELS.get(ds, ds)] = {
            "n_contrasts": len(rs),
            "n_nominal": len(ds_nom),
            "n_survive": len(ds_surv),
            "cell": cells[ds],
        }

    print(f"family={len(rows)}  nominal={len(nom)}  survivors={len(surv)}")
    for r in report["non_survivors"]:
        print(f"  NON-SURVIVOR: {r['dataset']} {r['probe']} {r['contrast']} p={r['p']} q={r['q']}")
    for ds, cell in cells.items():
        print(f"  {DATASET_LABELS.get(ds, ds):12s} -> {cell}")

    if errs:
        print("\n".join(errs))
        print("VERIFY FAILED -- nothing written")
        sys.exit(1)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nALL CHECKS PASSED; evidence -> {OUT_JSON}")

    # labeled per-contrast CSV for the figure pipeline (replaces the
    # label-less crosstask_fdr_computed.csv committed 2026-09-01)
    fig_csv = os.path.join(ROOT, "v7.5_submission_final", "05_DATA_FOR_FIGURES",
                           "crosstask_fdr_computed.csv")
    with open(fig_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "probe", "group_a", "group_b", "delta_auc", "p", "bh_q"])
        for r in sorted(rows, key=lambda r: (r["dataset"], r["probe"], r["group_a"], r["group_b"])):
            w.writerow([r["dataset"], r["probe"], r["group_a"], r["group_b"],
                        f"{r['delta_auc']:.6f}", f"{r['p']:.6g}", f"{r['q']:.4f}"])
    print(f"labeled FDR CSV (70 rows) -> {fig_csv}")

    if args.apply:
        pt = os.path.join(ROOT, "v7.5_submission_final", "01_MANUSCRIPT",
                          "4_PLACEHOLDER_TABLES.md")
        with open(pt) as fh:
            txt = fh.read()
        n_repl = 0
        for ds, cell in cells.items():
            key = DATASET_LABELS.get(ds, ds)
            # match the row by its leading label, then replace the token in it
            lines = txt.split("\n")
            for i, ln in enumerate(lines):
                if ln.startswith(f"| {key}") or (key == "ecoli_4mC" and ln.startswith(f"| {key} ")):
                    if "[INSERT_CROSSTASK_FDR_COLUMN]" in ln:
                        lines[i] = ln.replace("[INSERT_CROSSTASK_FDR_COLUMN]",
                                              f"{cell} (BH q, family of 70)")
                        n_repl += 1
            txt = "\n".join(lines)
        with open(pt, "w") as fh:
            fh.write(txt)
        print(f"APPLIED: {n_repl}/5 cells written to 4_PLACEHOLDER_TABLES.md")
        if n_repl == 0 and "[INSERT_CROSSTASK_FDR_COLUMN]" not in txt:
            print("cells already applied (idempotent re-run)")
        elif n_repl != 5:
            print("WARN: expected 5 replacements")
            sys.exit(1)
    else:
        print("verify-only mode; rerun with --apply to write cells")


if __name__ == "__main__":
    main()
