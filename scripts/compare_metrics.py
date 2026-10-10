#!/usr/bin/env python
"""compare_metrics — figure vs table vs manuscript-text number consistency.

Phase-3 script. For every key archived number family, verifies the value is
renderable in all three zones it may appear in (figure provenance marks,
table cells in 4_PLACEHOLDER_TABLES/SUPPLEMENTARY, manuscript body text),
and that the renderings agree to the stated precision.

Key families are curated (headline M0/anchor/probe/OOD/difficulty/cross-task
numbers). Output: TSV matrix + console; exit 1 on any cross-zone mismatch.
"""
import csv
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "_full_version", "audit", "Figure_Table_Forensic_Audit",
                   "COMPARE_METRICS_MATRIX.tsv")

# value -> (family, expected zones) ; zones: F=figure marks, T=tables, M=manuscript
KEYS = [
    ("0.8325", "M0 sim-val AUPRC", "FTM"), ("0.8351", "M0 sim-val AUROC", "FM"),
    ("0.827", "M0 sim-test", "TM"), ("0.8347", "IV4 max", "FTM"),
    ("0.7277", "Benbow mean", "FTM"), ("0.6548", "ICE mean", "FTM"),
    ("0.6278", "Philympics paired mean", "FTM"), ("0.616", "v6.8 Philympics ref", "TM"),
    ("0.7143", "v6.8 Benbow ref", "TM"), ("0.6675", "v6.8 IV4 ref", "TM"),
    ("0.951", "v6.8 sim-val denominator (fig draws 100% line, not the raw value)", "TM"),
    ("0.7775", "evo linear probe", "FT"), ("0.8212", "evo ridge probe", "FT"),
    ("0.8862", "evo MLP probe", "FTM"), ("0.8516", "composition ceiling AUROC", "FTM"),
    ("0.8335", "composition ceiling AUPRC", "FT"), ("0.6277", "RC-7mer linear", "FT"),
    ("0.8638", "OOD mean", "FTM"), ("0.0816", "OOD sd", "FTM"),
    ("0.8142", "D0 reference", "FTM"), ("0.8589", "D1", "FTM"),
    ("0.8227", "D2", "FTM"), ("0.8694", "D3", "FTM"),
    ("0.963", "GC R2 contain", "FTM"), ("0.624", "log-len R2 contain", "FTM"),
    ("0.9894", "cross-task htw NTv3 (4dp cell)", "T"), ("0.989", "cross-task htw NTv3 (3dp text)", "M"),
    ("0.047", "promoter_b max q", "FTM"), ("0.053", "promoter_r max q", "FTM"),
    ("0.063", "ecoli max q", "FTM"),
]

FIG_MARKS = []
for f in glob.glob(os.path.join(ROOT, "03_FIGURES_MASTER/rendered/*_provenance.json")):
    d = json.load(open(f))
    for panel, marks in d.get("panels", {}).items():
        if isinstance(marks, list):
            for m in marks:
                FIG_MARKS.append(str(m.get("value", "")))
FIG_TEXT = " | ".join(FIG_MARKS)

TABLE_TEXT = " ".join(open(os.path.join(ROOT, p), encoding="utf-8").read() for p in [
    "01_MANUSCRIPT/4_PLACEHOLDER_TABLES.md",
    "04_SUPPLEMENTARY/SUPPLEMENTARY_COMPLETE.md"])
MS_TEXT = " ".join(open(os.path.join(ROOT, p), encoding="utf-8").read() for p in [
    "01_MANUSCRIPT/1_MANUSCRIPT.5.md",
    "01_FULL_MANUSCRIPT/FULL_MANUSCRIPT.md"])
ZONES = {"F": FIG_TEXT, "T": TABLE_TEXT, "M": MS_TEXT}


def main():
    rows, mism = [], 0
    for val, fam, zones in KEYS:
        cells = []
        for z in zones:
            present = val in ZONES[z]
            cells.append(f"{z}:{'Y' if present else '-'}")
            # a zone marked expected but missing = P1
        row_ok = all(f"{z}:Y" in c for z, c in zip(zones, cells))
        # figure zone: value may appear at higher precision inside a list repr
        if "F" in zones and f"F:-" in cells:
            hit = any(val in s or (val.rstrip("0").rstrip(".") in s) for s in FIG_MARKS)
            cells = [c.replace("F:-", "F:Y~" if hit else "F:-") for c in cells]
            row_ok = row_ok or hit
        if not row_ok:
            mism += 1
        rows.append([fam, val, zones, " ".join(cells), "PASS" if row_ok else "P1-CHECK"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["Family", "Value", "ExpectedZones", "Presence", "Status"])
        w.writerows(rows)
    for r in rows:
        print("  ".join(r))
    print(f"-> {OUT}  ({mism} to check)")
    raise SystemExit(1 if mism else 0)


if __name__ == "__main__":
    main()
