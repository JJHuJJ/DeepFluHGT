#!/usr/bin/env python3
"""IslandViewer4 baseline — uses precomputed predictions to score the independent Benbow benchmark (P/R/F1/specificity).

IslandViewer4 outputs binary GI calls (no probabilities), so we report precision/recall/F1/specificity/accuracy
(not AUC), compared on Benbow (experimentally validated, independent of the IV4 library); IslandViewer's own positive entries would
leak, so they are not used. CPU only; no DeepFluHGT model/training involved.

Decision rule: a Benbow interval [bs,be] counts as predicted if it overlaps an IV4-predicted GI [is,ie].
  lenient = any overlap; strict = bidirectional overlap ratio >0.5 (IoU-like).

usage:
  python scripts/baseline_islandviewer4.py \
      --iv4 benchmark/islandviewer4/all_gis_islandviewer_iv4.txt \
      --benbow benchmark/literature_gi/repo/test_data/benbow_test_data.fasta \
      --out audit_output/14_baseline_islandviewer4.json
"""
from __future__ import annotations
import json, argparse, re
from collections import defaultdict

ROOT = "./DeepFluHGT.8.0"


def load_iv4(path):
    """{accession: [(start, end), ...]}"""
    gis = defaultdict(list)
    with open(path) as f:
        next(f) # header
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) >= 3:
                try:
                    gis[p[0]].append((int(p[1]), int(p[2])))
                except ValueError:
                    continue
    return gis


def read_benbow(path):
    """[(accession, start, end, label)] — header: >acc|label|acc:start-end org"""
    out, sid, label, buf = [], None, None, []
    hdr_re = re.compile(r"^(\S+)\|(\d+)\|(\S+?):(\d+)-(\d+)")
    for ln in open(path):
        ln = ln.strip()
        if ln.startswith(">"):
            m = hdr_re.match(ln[1:])
            if m and sid is not None:
                pass  # previous record already handled (coords live in the header)
            if m:
                acc, lbl, _, s, e = m.group(1), int(m.group(2)), m.group(3), int(m.group(4)), int(m.group(5))
                out.append((acc, s, e, lbl))
            sid = ln
        # sequence body ignored (only header coordinates are needed)
    return out


def overlaps(bs, be, gis):
    """return the list of IV4 GIs overlapping [bs,be]."""
    return [(s, e) for (s, e) in gis if s <= be and e >= bs]


def metrics(records, iv4, strict=False):
    tp = fp = fn = tn = 0
    for acc, bs, be, lbl in records:
        gis = iv4.get(acc, [])
        ov = overlaps(bs, be, gis)
        if strict:
            # bidirectional overlap ratio > 0.5
            def ror(s, e):
                inter = max(0, min(be, e) - max(bs, s))
                return inter / max(1, min(be - bs, e - s))
            ov = [(s, e) for (s, e) in ov if ror(s, e) > 0.5]
        call = 1 if ov else 0
        if lbl == 1 and call == 1: tp += 1
        elif lbl == 1 and call == 0: fn += 1
        elif lbl == 0 and call == 1: fp += 1
        else: tn += 1
    n = tp + fp + fn + tn
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    spec = tn / max(1, tn + fp)
    acc = (tp + tn) / max(1, n)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, n=n,
                precision=prec, recall=rec, f1=f1, specificity=spec, accuracy=acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iv4", required=True)
    ap.add_argument("--benbow", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    iv4 = load_iv4(args.iv4)
    print(f"[IV4] {len(iv4)} genomes, {sum(len(v) for v in iv4.values())} predicted GIs", flush=True)
    records = read_benbow(args.benbow)
    n_pos = sum(1 for r in records if r[3] == 1)
    n_neg = len(records) - n_pos
    print(f"[Benbow] {len(records)} regions: {n_pos} GI+ / {n_neg} GI-")
    covered = sum(1 for r in records if r[0] in iv4)
    print(f" Benbow accessions present in the IV4 library: {covered}/{len(records)}")

    res = {
        "benbow_n": len(records), "n_pos": n_pos, "n_neg": n_neg,
        "accession_in_iv4": covered,
        "lenient_any_overlap": metrics(records, iv4, strict=False),
        "strict_ror_gt_0.5": metrics(records, iv4, strict=True),
        "note": "IslandViewer4 emits binary GI calls → P/R/F1/specificity reported (no AUC). "
                "Benbow = an experimentally validated independent benchmark (avoids leakage from IslandViewer's own positives). "
                "DeepFluHGT comparisons must be made at matched specificity (or matched F1).",
    }
    print("\n" + "=" * 60)
    print("IslandViewer4 baseline (independent Benbow benchmark)")
    print("=" * 60)
    for k in ["lenient_any_overlap", "strict_ror_gt_0.5"]:
        m = res[k]
        print(f"\n[{k}]")
        print(f" TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")
        print(f" Precision={m['precision']:.3f} Recall={m['recall']:.3f} "
              f"F1={m['f1']:.3f} Specificity={m['specificity']:.3f} Accuracy={m['accuracy']:.3f}")
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
