#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
microhomology_enrich.py — B2: microhomology-junction enrichment statistics
==============================================================================
DeepFluHGT.5 submission enhancement (Task B2). CPU-only, read-only inputs.

Background
----------
The simulator config (configs/contig_500k.yaml: `microhomology_len: 3`) embeds a
3-bp microhomology at HGT junctions, mimicking the short host-donor sequence sharing
produced by MMEJ/NHEJ-type repair. Two signatures are distinguishable, and both are
tested here:

  T1  Junction-local 3-bp direct repeat (classical microhomology join):
      left  junction: seq[s-3:s] == seq[s:s+3]
      right junction: seq[e-3:e] == seq[e:e+3]

  T2  The signature the simulator ACTUALLY grafts (code inspection,
      simulator/impl/sequence_generator.py:apply_microhomology, verified on data):
      the insert's first 3 bp are overwritten with the host background's first 3 bp
      and the insert's last 3 bp with the host background's last 3 bp, i.e.
      left  edge: seq[s:s+3] == contig[0:3]
      right edge: seq[e-3:e] == contig[-3:]
      (the docstring says "junction bases" but the implementation copies
      bg_seq[:3] / bg_seq[-3:]; empirically confirmed on pos_00000000/1).

Controls (same counts): random interior positions on negative contigs of the same
test split, measuring (a) the chance rate of a 3-bp direct repeat across a random
position and (b) the chance rate that a random 3-mer equals the contig's head/tail
3-mer.

Statistics: Fisher exact (one-sided, greater), chi-square, odds ratio with 95% Woolf
CI. Contig-level robustness rows (first event per contig only) are reported because
events within one contig are not independent.

Outputs (new files)
-------------------
docs/evidence/BIOLOGICAL_GROUNDING/microhomology_enrich.md   (written separately)
docs/evidence/BIOLOGICAL_GROUNDING/microhomology_enrich.csv
docs/evidence/BIOLOGICAL_GROUNDING/microhomology_enrich.json

Usage
-----
python3 scripts/microhomology_enrich.py
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_bed(path):
    pos_events, neg_contigs = [], []
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().lstrip("#").rstrip("\n").split("\t")
        idx = {c: i for i, c in enumerate(header)}
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            sid = parts[idx["seq_id"]]
            slen = int(parts[idx["seq_len"]])
            if int(parts[idx["label"]]) == 1:
                pos_events.append({
                    "seq_id": sid,
                    "event_idx": int(parts[idx["event_idx"]]),
                    "seq_len": slen,
                    "s": int(parts[idx["hgt_start"]]),
                    "e": int(parts[idx["hgt_end"]]),
                    "virus_len": int(parts[idx["virus_len"]]),
                    "mode": parts[idx["hgt_mode"]],
                })
            else:
                neg_contigs.append((sid, slen))
    return pos_events, neg_contigs


def load_fasta(path, want=None):
    seqs, name, buf = {}, None, []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None and (want is None or name in want):
                    seqs[name] = "".join(buf)
                name = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if name is not None and (want is None or name in want):
        seqs[name] = "".join(buf)
    return seqs


# --------------------------------------------------------------------------- #
# statistics helpers
# --------------------------------------------------------------------------- #

def woolf_or_ci(a, b, c, d):
    """Odds ratio (a/b)/(c/d) with 95% Woolf CI; returns (or, lo, hi)."""
    def _f(x):
        return x if x > 0 else 0.5  # Haldane-Anscombe-ish continuity guard
    a_, b_, c_, d_ = _f(a), _f(b), _f(c), _f(d)
    odds = (a_ / b_) / (c_ / d_)
    se = math.sqrt(1.0 / a_ + 1.0 / b_ + 1.0 / c_ + 1.0 / d_)
    z = stats.norm.ppf(0.975)
    return odds, odds * math.exp(-z * se), odds * math.exp(z * se)


def fisher_table(k_pos, n_pos, k_ctl, n_ctl, label):
    """2x2 [[pos_hit, pos_miss],[ctl_hit, ctl_miss]] -> result dict."""
    table = [[k_pos, n_pos - k_pos], [k_ctl, n_ctl - k_ctl]]
    fisher = stats.fisher_exact(table, alternative="greater")
    fisher_two = stats.fisher_exact(table, alternative="two-sided")
    try:
        chi2 = stats.chi2_contingency(table, correction=True)
        chi2_stat, chi2_p = float(chi2[0]), float(chi2[1])
    except ValueError:
        chi2_stat, chi2_p = float("nan"), float("nan")
    orr, lo, hi = woolf_or_ci(k_pos, n_pos - k_pos, k_ctl, n_ctl - k_ctl)
    # exact one-sided p in log space so extreme tables do not underflow silently
    N = n_pos + n_ctl
    K = k_pos + k_ctl
    n = n_pos
    hi_k = min(K, n)
    ks = np.arange(k_pos, hi_k + 1)
    log10_p = float(np.sum(stats.hypergeom.logpmf(ks, N, K, n)) / math.log(10))
    p_display = ("< 1e-300" if log10_p < -300 else
                 f"{math.pow(10, log10_p):.3e}")
    return {
        "test": label,
        "positive_hits": int(k_pos), "positive_n": int(n_pos),
        "positive_rate": k_pos / n_pos,
        "control_hits": int(k_ctl), "control_n": int(n_ctl),
        "control_rate": k_ctl / n_ctl,
        "fisher_p_one_sided": float(fisher.pvalue),
        "fisher_p_one_sided_display": p_display,
        "fisher_log10_p_one_sided": log10_p,
        "fisher_p_two_sided": float(fisher_two.pvalue),
        "fisher_OR": float(fisher.statistic),
        "chi2": chi2_stat, "chi2_p": float(chi2_p),
        "odds_ratio": orr, "or_ci95_lo": lo, "or_ci95_hi": hi,
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bed", default=os.path.join(
        ROOT, "simulated_data/Contig_500k/test/breakpoints.bed"))
    ap.add_argument("--fasta", default=os.path.join(
        ROOT, "simulated_data/Contig_500k/test/dataset.fasta"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=os.path.join(
        ROOT, "docs/evidence/BIOLOGICAL_GROUNDING"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    pos_events, neg_contigs = load_bed(args.bed)
    print(f"[1] positive events: {len(pos_events):,}  negative contigs: {len(neg_contigs):,}")

    # one random position per negative contig (25k controls)
    neg_pos = {}
    for nid, L in neg_contigs:
        if L >= 12:
            neg_pos[nid] = rng.randint(4, L - 7)

    want = {ev["seq_id"] for ev in pos_events} | set(neg_pos)
    seqs = load_fasta(args.fasta, want=want)
    print(f"[2] contigs loaded: {len(seqs):,}")

    # ---- positive-event junction measurements -------------------------------
    per_event = []
    events_by_seq = defaultdict(list)
    for ev in pos_events:
        seq = seqs.get(ev["seq_id"])
        if seq is None or len(seq) != ev["seq_len"]:
            continue
        s, e = ev["s"], ev["e"]
        if s < 3 or e + 3 > len(seq) or e - s < 6:
            continue
        row = {
            "seq_id": ev["seq_id"], "event_idx": ev["event_idx"],
            "mode": ev["mode"], "virus_len": e - s,
            "t1_left_repeat": int(seq[s - 3:s] == seq[s:s + 3]),
            "t1_right_repeat": int(seq[e - 3:e] == seq[e:e + 3]),
            "t2_left_graft": int(seq[s:s + 3] == seq[0:3]),
            "t2_right_graft": int(seq[e - 3:e] == seq[-3:]),
        }
        per_event.append(row)
        events_by_seq[ev["seq_id"]].append(row)

    n_ev = len(per_event)
    print(f"[3] measurable positive events: {n_ev:,}")

    t1_l = sum(r["t1_left_repeat"] for r in per_event)
    t1_r = sum(r["t1_right_repeat"] for r in per_event)
    t2_l = sum(r["t2_left_graft"] for r in per_event)
    t2_r = sum(r["t2_right_graft"] for r in per_event)

    # ---- negative-control measurements --------------------------------------
    c_t1 = c_t2h = c_t2t = 0
    n_ctl = 0
    for nid, p in neg_pos.items():
        seq = seqs.get(nid)
        if seq is None:
            continue
        if p < 3 or p + 3 > len(seq):
            continue
        n_ctl += 1
        c_t1 += int(seq[p - 3:p] == seq[p:p + 3])
        c_t2h += int(seq[p:p + 3] == seq[0:3])
        c_t2t += int(seq[p - 3:p] == seq[-3:])
    print(f"[4] measurable control positions: {n_ctl:,}")

    # ---- tests ---------------------------------------------------------------
    tests = {
        "T1_left_junction_direct_repeat":
            fisher_table(t1_l, n_ev, c_t1, n_ctl,
                         "T1 left junction 3-bp direct repeat vs negative random position"),
        "T1_right_junction_direct_repeat":
            fisher_table(t1_r, n_ev, c_t1, n_ctl,
                         "T1 right junction 3-bp direct repeat vs negative random position"),
        "T2_left_insert_edge_vs_contig_head":
            fisher_table(t2_l, n_ev, c_t2h, n_ctl,
                         "T2 insert left edge == contig head 3-mer vs negative random position"),
        "T2_right_insert_edge_vs_contig_tail":
            fisher_table(t2_r, n_ev, c_t2t, n_ctl,
                         "T2 insert right edge == contig tail 3-mer vs negative random position"),
    }

    # contig-level robustness (first event per contig only -> independent units)
    first_rows = [v[0] for v in events_by_seq.values() if v]
    n1 = len(first_rows)
    t1l1 = sum(r["t1_left_repeat"] for r in first_rows)
    t1r1 = sum(r["t1_right_repeat"] for r in first_rows)
    t2l1 = sum(r["t2_left_graft"] for r in first_rows)
    t2r1 = sum(r["t2_right_graft"] for r in first_rows)
    tests_contig = {
        "T1_left_contiglevel": fisher_table(t1l1, n1, c_t1, n_ctl,
                                            "T1 left, first-event-per-contig"),
        "T1_right_contiglevel": fisher_table(t1r1, n1, c_t1, n_ctl,
                                             "T1 right, first-event-per-contig"),
        "T2_left_contiglevel": fisher_table(t2l1, n1, c_t2h, n_ctl,
                                            "T2 left, first-event-per-contig"),
        "T2_right_contiglevel": fisher_table(t2r1, n1, c_t2t, n_ctl,
                                             "T2 right, first-event-per-contig"),
    }

    # per-mode breakdown (event level, descriptive)
    per_mode = {}
    for mode in sorted({r["mode"] for r in per_event}):
        rows = [r for r in per_event if r["mode"] == mode]
        per_mode[mode] = {
            "n_events": len(rows),
            "t1_left_rate": np.mean([r["t1_left_repeat"] for r in rows]),
            "t1_right_rate": np.mean([r["t1_right_repeat"] for r in rows]),
            "t2_left_rate": np.mean([r["t2_left_graft"] for r in rows]),
            "t2_right_rate": np.mean([r["t2_right_graft"] for r in rows]),
        }

    # insert-length stratification for T2 (short inserts are the hard cases)
    len_bins = [(0, 100), (100, 300), (300, 800), (800, 1500), (1500, 10 ** 9)]
    per_lenbin = {}
    for lo, hi in len_bins:
        rows = [r for r in per_event if lo <= r["virus_len"] < hi]
        if not rows:
            continue
        per_lenbin[f"{lo}-{min(hi, 2341)}"] = {
            "n_events": len(rows),
            "t2_left_rate": float(np.mean([r["t2_left_graft"] for r in rows])),
            "t2_right_rate": float(np.mean([r["t2_right_graft"] for r in rows])),
        }

    # ---- outputs -------------------------------------------------------------
    result = {
        "task": "B2 microhomology-junction enrichment",
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "inputs": {"bed": args.bed, "fasta": args.fasta},
        "n_events": n_ev, "n_control_positions": n_ctl,
        "counts": {
            "t1_left": int(t1_l), "t1_right": int(t1_r),
            "t2_left": int(t2_l), "t2_right": int(t2_r),
            "control_t1": int(c_t1), "control_t2_head": int(c_t2h),
            "control_t2_tail": int(c_t2t),
        },
        "event_level_tests": tests,
        "contig_level_tests": tests_contig,
        "per_mode": per_mode,
        "per_insert_length_bin": per_lenbin,
        "note": ("T1 = classical junction-local 3-bp direct repeat; "
                 "T2 = the graft the simulator actually applies "
                 "(apply_microhomology copies bg[:3]/bg[-3:] onto insert edges)."),
    }
    out_json = os.path.join(args.outdir, "microhomology_enrich.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    out_csv = os.path.join(args.outdir, "microhomology_enrich.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["seq_id", "event_idx", "mode", "virus_len",
                    "t1_left_repeat", "t1_right_repeat",
                    "t2_left_graft", "t2_right_graft"])
        for r in per_event:
            w.writerow([r["seq_id"], r["event_idx"], r["mode"], r["virus_len"],
                        r["t1_left_repeat"], r["t1_right_repeat"],
                        r["t2_left_graft"], r["t2_right_graft"]])
        w.writerow([])
        w.writerow(["# controls: one random position per negative contig"])
        w.writerow(["control_n", "t1_direct_repeat", "t2_head_match", "t2_tail_match"])
        w.writerow([n_ctl, c_t1, c_t2h, c_t2t])

    print(f"[5] wrote {out_json}")
    print(f"    wrote {out_csv}")
    print()
    print("== headline numbers ==")
    print(f"T1 junction direct repeat : pos {t1_l}/{n_ev} ({t1_l/n_ev:.4f}) left, "
          f"{t1_r}/{n_ev} ({t1_r/n_ev:.4f}) right | ctrl {c_t1}/{n_ctl} ({c_t1/n_ctl:.4f})")
    for k, t in tests.items():
        if k.startswith("T1"):
            print(f"  {k}: OR={t['odds_ratio']:.2f} "
                  f"[{t['or_ci95_lo']:.2f}-{t['or_ci95_hi']:.2f}], "
                  f"Fisher two-sided p={t['fisher_p_two_sided']:.3e} (depletion)")
    print(f"T2 terminus graft        : pos {t2_l}/{n_ev} ({t2_l/n_ev:.4f}) left, "
          f"{t2_r}/{n_ev} ({t2_r/n_ev:.4f}) right | ctrl head {c_t2h}/{n_ctl} "
          f"({c_t2h/n_ctl:.4f}), tail {c_t2t}/{n_ctl} ({c_t2t/n_ctl:.4f})")
    for k, t in tests.items():
        if k.startswith("T2"):
            print(f"  {k}: OR={t['odds_ratio']:.2f} "
                  f"[{t['or_ci95_lo']:.2f}-{t['or_ci95_hi']:.2f}], "
                  f"Fisher p={t['fisher_p_one_sided_display']} "
                  f"(log10 p = {t['fisher_log10_p_one_sided']:.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
