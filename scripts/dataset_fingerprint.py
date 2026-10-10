#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dataset fingerprint  —  DeepFluHGT Stage B-0 [P1-4].

step0 subset (the evidence base for the RAP/shortcut audit) vs the M0 training set
(train/val of simulated_data/Contig_500k) distribution comparison:

  - two-sample KS tests on length / GC (step0 vs train, step0 vs val)
  - positive/negative ratio, hard-negative ratio (same _HARD_NEG_TYPES as evaluation/impl/metrics.py)
  - M0-split membership of step0 (double-matched by seq_id + seq_len), cross-checked against BED labels
  - GC/N% computed by streaming over the fasta (chunked binary parsing, no full in-memory residency)

output: docs/evidence/V7_5_DATASET_FINGERPRINT_STEP0.json
       docs/evidence/V7_5_DATASET_FINGERPRINT_M0.json
       docs/evidence/V7_5_DATASET_FINGERPRINT.md   (human-readable summary)
CPU-only.read-only.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_M0 = REPO / "simulated_data" / "Contig_500k"
DEFAULT_STEP0 = REPO / "evo_cache" / "step0_subset"
DEFAULT_OUT = REPO / "docs" / "evidence"

# fully consistent with evaluation/impl/metrics.py::_HARD_NEG_TYPES (synced from the production source, read-only)
HARD_NEG_TYPES = {"NEG_HARD", "NEG_HOMOLOGY", "NEG_CONSERVED", "NEG_LOW_COMPLEXITY"}


# ───────────────────────── fasta streaming statistics (chunked) ───────────────────────── #

def stream_fasta_stats(path: Path, max_seq: int = 0) -> dict[str, tuple]:
    """return {seq_id: (length, gc_frac, n_frac)}. 64MB-chunk binary parsing.

    Record delimiter = b"\\n>"; record bodies include the '>' header line;
    case-sensitive counting (datasets are uppercase).
    """
    out: dict[str, tuple] = {}
    buf = b""

    def _handle(rec: bytes) -> None:
        if not rec or rec[:1] != b">":
            return
        nl = rec.find(b"\n")
        if nl == -1:
            return
        sid = rec[1:nl].decode(errors="replace").strip().split()[0]
        seq = rec[nl + 1:].replace(b"\n", b"").replace(b"\r", b"")
        n = len(seq)
        a = seq.count(b"A"); c = seq.count(b"C")
        g = seq.count(b"G"); t = seq.count(b"T")
        acgt = a + c + g + t
        out[sid] = (n,
                    (g + c) / acgt if acgt else float("nan"),
                    (n - acgt) / n if n else float("nan"))

    t0 = time.time()
    start = 0        # start offset of the current record in buf
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 26)
            if not chunk:
                break
            buf = buf + chunk if buf else chunk
            pos = start
            while True:
                i = buf.find(b"\n>", pos)
                if i == -1:
                    break
                _handle(buf[start:i])
                start = pos = i + 1
                if max_seq and len(out) >= max_seq:
                    print(f"[fingerprint]   fasta {path.name}: {len(out)} seqs "
                          f"(capped, {time.time() - t0:.0f}s)")
                    return out
            buf = buf[start:]
            start = 0
    _handle(buf)
    print(f"[fingerprint]   fasta {path.name}: {len(out)} seqs "
          f"({time.time() - t0:.0f}s)")
    return out


def load_fasta_small(path: Path) -> dict[str, tuple]:
    """Parse a small fasta (step0) directly."""
    out: dict[str, tuple] = {}
    name, seq = None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name:
                    s = "".join(seq)
                    acgt = sum(s.count(c) for c in "ACGT")
                    out[name] = (len(s),
                                 (s.count("G") + s.count("C")) / acgt if acgt else float("nan"),
                                 (len(s) - acgt) / len(s) if len(s) else float("nan"))
                name = line[1:].strip().split()[0]
                seq = []
            else:
                seq.append(line.strip().upper())
    if name:
        s = "".join(seq)
        acgt = sum(s.count(c) for c in "ACGT")
        out[name] = (len(s),
                     (s.count("G") + s.count("C")) / acgt if acgt else float("nan"),
                     (len(s) - acgt) / len(s) if len(s) else float("nan"))
    return out


# ───────────────────────────────── BED parse ───────────────────────────────── #

def parse_bed(path: Path) -> dict[str, dict]:
    """{seq_id: {label, seq_len, mode}}  —  —  first line per sequence (event-level line)."""
    out: dict[str, dict] = {}
    with open(path) as fh:
        rd = csv.reader(fh, delimiter="\t")
        for row in rd:
            if not row or row[0].startswith("#"):
                continue
            sid, _ev, lab, slen, _hs, _he, _vl, mode = row[:8]
            if sid in out:
                continue
            out[sid] = {"label": int(lab), "seq_len": int(slen), "mode": mode}
    return out


# ────────────────────────────────── statistics ────────────────────────────────── #

def dist_stats(v: np.ndarray) -> dict:
    v = v[np.isfinite(v)]
    q = np.percentile(v, [5, 25, 50, 75, 95]) if len(v) else [np.nan] * 5
    return {"n": int(len(v)), "mean": float(v.mean()) if len(v) else None,
            "std": float(v.std()) if len(v) else None,
            "p5": float(q[0]), "p25": float(q[1]), "median": float(q[2]),
            "p75": float(q[3]), "p95": float(q[4])}


def ks(a: np.ndarray, b: np.ndarray) -> dict:
    from scipy.stats import ks_2samp
    r = ks_2samp(a[np.isfinite(a)], b[np.isfinite(b)])
    return {"statistic": float(r.statistic), "pvalue": float(r.pvalue)}


# ────────────────────────────────── main pipeline ────────────────────────────────── #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m0-dir", type=Path, default=DEFAULT_M0)
    ap.add_argument("--step0-dir", type=Path, default=DEFAULT_STEP0)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--splits", type=str, default="train,val")
    ap.add_argument("--max-seq", type=int, default=0,
                    help="max sequences per split fasta (0=all; for quick trial runs)")
    args = ap.parse_args()
    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── step0 ──
    print("[fingerprint] step0 ...")
    st0 = load_fasta_small(args.step0_dir / "step0_subset.fasta")
    ids0 = list(st0)
    y0 = np.array([1 if i.startswith("pos_") else 0 for i in ids0])
    len0 = np.array([st0[i][0] for i in ids0], float)
    gc0 = np.array([st0[i][1] for i in ids0], float)
    n0 = np.array([st0[i][2] for i in ids0], float)

    # BED membership + negative type + label cross-validation
    beds = {s: parse_bed(args.m0_dir / s / "breakpoints.bed")
            for s in args.splits.split(",")}
    y0d = dict(zip(ids0, y0.tolist()))
    split_of: dict[str, str] = {}
    neg_types0: dict[str, str] = {}
    label_mismatch = 0
    for sid, L in zip(ids0, len0):
        for sp, bed in beds.items():
            r = bed.get(sid)
            if r and r["seq_len"] == int(L):
                split_of[sid] = sp
                if y0d[sid] == 0:
                    neg_types0[sid] = r["mode"]
                if r["label"] != y0d[sid]:
                    label_mismatch += 1
                break
    from collections import Counter
    neg_type_counts = dict(Counter(neg_types0.values()).most_common())
    hard0 = sum(t in HARD_NEG_TYPES for t in neg_types0.values())

    # provenance label flip count (prefix vs provenance.tsv)
    prov_lab = {}
    with open(args.step0_dir / "provenance.tsv") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            prov_lab.setdefault(r["seq_id"], r["label"])
    prov_mismatch = sum(int(prov_lab.get(i, -1)) != int(v) for i, v in zip(ids0, y0))

    step0_json = {
        "meta": {"source": "evo_cache/step0_subset", "generated":
                 time.strftime("%Y-%m-%d %H:%M:%S"),
                 "label_source": "seq_id prefix (pos_/neg_)"},
        "n_total": len(ids0), "n_pos": int(y0.sum()), "n_neg": int(len(y0) - y0.sum()),
        "pos_ratio": float(y0.mean()),
        "length": dist_stats(len0), "gc": dist_stats(gc0), "n_base_frac": dist_stats(n0),
        "m0_split_membership": dict(Counter(split_of.values())),
        "m0_bed_label_vs_prefix_mismatches": label_mismatch,
        "provenance_label_vs_prefix_mismatches": prov_mismatch,
        "provenance_mismatch_note":
            "the provenance.tsv label column has a one-sided flip (neg→1) at a rate ≈ label_noise=0.02; "
            "BED/prefix are consistent  —  M0 training labels and RAP labels share the same source",
        "neg_type_counts": neg_type_counts,
        "hard_neg_ratio_among_neg": hard0 / max(1, len(neg_types0)),
        "hard_neg_types": sorted(HARD_NEG_TYPES),
    }
    (args.out_dir / "V7_5_DATASET_FINGERPRINT_STEP0.json").write_text(
        json.dumps(step0_json, indent=2, ensure_ascii=False))

    # ── M0 splits ──
    m0_json = {"meta": {
        "source": f"simulated_data/{args.m0_dir.name}",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "max_seq_per_split": args.max_seq or "all",
        "ks_reference": "step0_subset (n=1996)",
    }, "splits": {}}
    for sp in args.splits.split(","):
        print(f"[fingerprint] M0/{sp} ...")
        bed = beds[sp]
        fsta = stream_fasta_stats(args.m0_dir / sp / "dataset.fasta", args.max_seq)
        ids = [i for i in bed if i in fsta]
        lab = np.array([bed[i]["label"] for i in ids])
        L = np.array([bed[i]["seq_len"] for i in ids], float)
        G = np.array([fsta[i][1] for i in ids], float)
        Nfrac = np.array([fsta[i][2] for i in ids], float)
        modes = [bed[i]["mode"] for i in ids]
        neg_types = Counter(m for m, l in zip(modes, lab) if l == 0)
        hard = sum(c for t, c in neg_types.items() if t in HARD_NEG_TYPES)
        m0_json["splits"][sp] = {
            "n_sequences": len(bed),
            "n_sequences_with_fasta_stats": len(ids),
            "n_pos": int(lab.sum()), "n_neg": int(len(lab) - lab.sum()),
            "pos_ratio": float(lab.mean()),
            "length": dist_stats(L), "gc": dist_stats(G), "n_base_frac": dist_stats(Nfrac),
            "neg_type_counts": dict(neg_types.most_common()),
            "hard_neg_ratio_among_neg": hard / max(1, int(len(lab) - lab.sum())),
            "pos_mode_counts": dict(Counter(
                m for m, l in zip(modes, lab) if l == 1).most_common()),
            "ks_vs_step0": {
                "length": ks(len0, L),
                "gc": ks(gc0, G),
                "n_base_frac": ks(n0, Nfrac),
            },
        }
        print(f"[fingerprint]   {sp}: n={len(ids)} KS(len) p="
              f"{m0_json['splits'][sp]['ks_vs_step0']['length']['pvalue']:.3e}, "
              f"KS(GC) p={m0_json['splits'][sp]['ks_vs_step0']['gc']['pvalue']:.3e}")
    (args.out_dir / "V7_5_DATASET_FINGERPRINT_M0.json").write_text(
        json.dumps(m0_json, indent=2, ensure_ascii=False))

    _write_md(args.out_dir / "V7_5_DATASET_FINGERPRINT.md", step0_json, m0_json)
    print(f"[fingerprint] done ({time.time() - t0:.0f}s) -> {args.out_dir}")


def _write_md(path: Path, s0: dict, m0: dict) -> None:
    L = []
    A = L.append
    A("# Dataset Fingerprint  —  step0 vs M0 (v7.5 Stage B-0)\n")
    A(f"generate: {s0['meta']['generated']}  (CPU, read-only)\n")
    A("## step0 subset\n")
    A(f"- n={s0['n_total']} (pos={s0['n_pos']}, neg={s0['n_neg']}, "
      f"pos_ratio={s0['pos_ratio']:.3f})")
    A(f"- length: median={s0['length']['median']:.0f} "
      f"[IQR {s0['length']['p25']:.0f}–{s0['length']['p75']:.0f}]; "
      f"GC: median={s0['gc']['median']:.3f}")
    A(f"- M0 split membership (id+seq_len matched): {s0['m0_split_membership']}")
    A(f"- BED label vs prefix mismatches: {s0['m0_bed_label_vs_prefix_mismatches']}; "
      f"provenance label vs prefix mismatches: "
      f"{s0['provenance_label_vs_prefix_mismatches']}  —  {s0['provenance_mismatch_note']}")
    A(f"- hard-negative ratio (among negatives): {s0['hard_neg_ratio_among_neg']:.3f} "
      f"(type set synced with the production evaluation/_HARD_NEG_TYPES)\n")
    A("## M0 splits\n")
    A("| split | n | pos_ratio | len median [IQR] | GC median | hard-neg ratio | "
      "KS(len) D/p | KS(GC) D/p |")
    A("|---|---|---|---|---|---|---|---|")
    for sp, r in m0["splits"].items():
        kl, kg = r["ks_vs_step0"]["length"], r["ks_vs_step0"]["gc"]
        A(f"| {sp} | {r['n_sequences']} | {r['pos_ratio']:.3f} | "
          f"{r['length']['median']:.0f} [{r['length']['p25']:.0f}–{r['length']['p75']:.0f}] | "
          f"{r['gc']['median']:.3f} | {r['hard_neg_ratio_among_neg']:.3f} | "
          f"{kl['statistic']:.3f} / {kl['pvalue']:.2e} | "
          f"{kg['statistic']:.3f} / {kg['pvalue']:.2e} |")
    A("\nNote: the KS test is extremely sensitive to tiny differences in large samples; "
      "report D (effect size) in preference to p. Numeric details are in the two JSONs in the same directory.\n")
    Path(path).write_text("\n".join(L))


if __name__ == "__main__":
    main()
