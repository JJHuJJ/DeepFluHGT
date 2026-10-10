#!/usr/bin/env python3
# scripts/crosstaxa_eval.py
"""DeepFluHGT  —  task B: OOD cross-taxa (M0, genome-level proxy) inference evaluation (4090).

Logic reference (read-only): ../DeepFluHGT.8.0/scripts/prepare_cross_taxa.py:
  genome → FASTA header organism → genus→phylum approximate mapping (PHYLUM_MAP;
  this script imports the very same scripts/prepare_cross_taxa.py and reuses it verbatim).

Design:
  from simulated_data/Contig_500k/test/provenance.tsv, group contigs by source_file
  (microbe genome GCF_x.y; each contig takes the genome of its first HOST/microbe
  segment = recipient background) → M0 (use_evo=False) sliding-window inference over
  all 50,000 test contigs (protocol identical to task A: window=3000/overlap=300/
  win_cap=10/batch=8; reuses the windowing/aggregation/metric of
  scripts/eval_real_anchors.py) → group-level (per-genome) AUPRC/AUROC +
  between-group variance + phylum-level summary →
  docs/evidence/OOD_CROSSTAXA_M0.md.

Convention note (also stated inside the report):
  - the test genome pool is fully isolated from train (genome_split_manifest.json,
    microbe train 9,242 / test 1,155, overlap=0) → each genome's AUPRC
    = generalization to an unseen genome; **genome-level as a taxa proxy** (GCF
    accessions cannot be queried directly against NCBI taxonomy; phylum is
    approximated from the header organism).

GPU discipline: CUDA_VISIBLE_DEVICES=4090 UUID GPU lock (A100 strictly forbidden);
  only this one GPU process at a time (guaranteed by the serial chain script).

usage (launcher: scripts/run/difficulty_curve_chain.sh):
  CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python scripts/crosstaxa_eval.py          # full set
  python scripts/crosstaxa_eval.py --plan                                  # CPU dry run
  CUDA_VISIBLE_DEVICES=... python scripts/crosstaxa_eval.py --limit 60       # smoke
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

GPU_4090_UUID = "${GPU_4090_UUID}"
DEFAULT_CKPT = ROOT / "checkpoints/Contig_500k_ntv3only/best_model.pt"
DEFAULT_CFG = ROOT / "configs/contig_500k_ntv3only.yaml"
NT_FALLBACK = ROOT / "model/InstaDeepAI/NTv3_650M_pre"
TEST_DIR = ROOT / "simulated_data/Contig_500k/test"
MICROBE_DIR = Path("./microbe_genomes")
SPLIT_MANIFEST = ROOT / "simulated_data/Contig_500k/genome_split_manifest.json"

# D0 reference (same as task A): M0 final sim test (docs/evidence/REAL_ANCHORS_M0final.md)
D0_REF_AUPRC = 0.827


def parse_args():
    ap = argparse.ArgumentParser(description="OOD cross-taxa (genome-level) M0 evaluation (task B)")
    ap.add_argument("--plan", action="store_true",
                    help="CPU dry run: parse provenance + phylum annotation + grouping statistics, no model load")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--win_cap", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N contigs per class")
    ap.add_argument("--min_group", type=int, default=20,
                    help="minimum contigs per group to enter the primary table (both classes present)")
    ap.add_argument("--gpu_uuid", default=GPU_4090_UUID)
    ap.add_argument("--tag", default="ood_crosstaxa_m0")
    ap.add_argument("--out_dir", default=str(ROOT / "docs/evidence"))
    return ap.parse_args()


ARGS = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = ARGS.gpu_uuid

import eval_real_anchors as cpu  # noqa: E402  (windowing/aggregation/metric reused verbatim)
os.environ["CUDA_VISIBLE_DEVICES"] = ARGS.gpu_uuid  # restore immediately (the cpu module had blanked it)

from delong_test import delong_test_full  # noqa: E402  (setdefault does not override)
import prepare_cross_taxa as pctx  # noqa: E402  (v6.8 genus→phylum mapping reused verbatim)

import torch  # noqa: E402


def assert_4090():
    n = torch.cuda.device_count()
    assert n == 1, f"exactly 1 visible GPU expected (4090 GPU lock), actual {n}"
    name = torch.cuda.get_device_name(0)
    assert "4090" in name, f"visible GPU is not RTX 4090: {name}"
    free, total = torch.cuda.mem_get_info(0)
    print(f"[GPU] {name} | mem free {free/1e9:.1f}/{total/1e9:.1f} GB | "
          f"CUDA {torch.version.cuda} | torch {torch.__version__}")
    return name


# ═══════════════════════════════════════════════════════════════════════════ #
#  provenance → contig grouping (source_file = recipient genome GCF)
# ═══════════════════════════════════════════════════════════════════════════ #

def parse_provenance_groups(prov_path: Path):
    """→ {seq_id: (label, genome_basename)}; genome = first HOST/microbe segment."""
    labels, genomes = {}, {}
    n_rows = 0
    with open(prov_path) as f:
        header = f.readline().rstrip("\n").split("\t")
        i_sid, i_lab = header.index("seq_id"), header.index("label")
        i_seg, i_src, i_file = (header.index("seg_type"), header.index("source_type"),
                                header.index("source_file"))
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            n_rows += 1
            if len(p) <= i_file:
                continue
            sid = p[i_sid]
            if sid not in labels:
                labels[sid] = int(p[i_lab])
            if sid not in genomes and (p[i_src] == "microbe" or p[i_seg] == "HOST"):
                genomes[sid] = os.path.basename(p[i_file])
    unmapped = sum(1 for s in labels if s not in genomes)
    print(f"[Prov] {len(labels)} contigs ({n_rows} segment rows) | "
          f"pos={sum(labels.values())} neg={len(labels)-sum(labels.values())} | "
          f"no-host-segment={unmapped}")
    return {s: (labels[s], genomes.get(s, "NO_HOST_SEGMENT")) for s in labels}


def annotate_phylum(genomes: set) -> dict:
    """genome basename → {organism, phylum} (logic verbatim from v6.8 prepare_cross_taxa)."""
    out, missing = {}, 0
    for g in sorted(genomes):
        if g == "NO_HOST_SEGMENT":
            continue
        fpath = MICROBE_DIR / g
        organism = pctx.extract_organism_from_fasta(str(fpath)) if fpath.is_file() else ""
        if not organism:
            missing += 1
        out[g] = {"organism": organism or "(header not found)",
                  "phylum": pctx.get_phylum_from_name(organism) if organism else "Unknown"}
    print(f"[Taxonomy] {len(out)} genomes annotated | header missing {missing}")
    return out


def split_isolation():
    """train/test genome isolation (genome_split_manifest.json) → report convention note."""
    if not SPLIT_MANIFEST.is_file():
        return None
    m = json.load(open(SPLIT_MANIFEST))
    mb = m.get("microbe", {})
    def _files(sp):
        v = mb.get(sp, {})
        return set(v.get("files", [])) if isinstance(v, dict) else set(v)
    tr, te = _files("train"), _files("test")
    return {"train_genomes": len(tr), "test_genomes": len(te),
            "overlap": len(tr & te), "seed": m.get("seed"),
            "ratios": m.get("ratios", {}).get("microbe")}


# ═══════════════════════════════════════════════════════════════════════════ #
#  FASTA read (keep only contigs labeled in provenance; same type as task A)
# ═══════════════════════════════════════════════════════════════════════════ #

def read_test_samples(fasta: Path, groups: dict, limit: int = 0) -> dict:
    samples, header, buf = {}, None, []
    with open(fasta) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None and header in groups:
                    samples[header] = (groups[header][0], "".join(buf))
                header = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.upper())
    if header is not None and header in groups:
        samples[header] = (groups[header][0], "".join(buf))

    if limit > 0:
        pos = {k: v for k, v in samples.items() if v[0] == 1}
        neg = {k: v for k, v in samples.items() if v[0] == 0}
        samples = dict(list(pos.items())[:limit])
        samples.update(dict(list(neg.items())[:limit]))
        print(f"[Smoke] limited to {len(samples)} contigs")
    n_pos = sum(1 for v in samples.values() if v[0] == 1)
    print(f"[Data] {len(samples)} contigs (pos={n_pos}, neg={len(samples)-n_pos}) "
          f"from {fasta.name} ({fasta.stat().st_size/1e6:.0f} MB)")
    return samples


# ═══════════════════════════════════════════════════════════════════════════ #
#  model + windowinference (verbatim replication eval_real_anchors_gpu / difficulty_eval)
# ═══════════════════════════════════════════════════════════════════════════ #

def load_model_gpu(ckpt: Path, config: Path):
    import yaml
    from model.api import build_model
    from model.contracts import HGTConfig

    device = torch.device("cuda:0")
    mc = yaml.safe_load(open(config))["model"]
    cfg = HGTConfig(
        data_mode="contig", max_contig_length=3000,
        hidden_dim=mc["hidden_dim"],
        num_attention_heads=mc["num_attention_heads"],
        dropout=mc["dropout"],
        graph_top_k=mc["graph_top_k"],
        bio_feature_dim=mc["bio_feature_dim"],
        pretrained_nt_path=mc.get("pretrained_nt_path", str(NT_FALLBACK)),
        pretrained_caduceus_path="",
        use_local_encoder=False,
        use_evo=bool(mc.get("use_evo", False)),
        pretrained_evo_dim=int(mc.get("pretrained_evo_dim", 4096)),
        early_exit_threshold=0.95,
    )
    model = build_model(cfg)

    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    ckpt_epoch = st.get("epoch")
    contract = st.get("contract", {})
    if ckpt_epoch != 2:  # same rule as the real-anchor evaluation: best_model must be ep2
        alt = ckpt.parent / "epoch_2.pt"
        print(f"[WARN] {ckpt.name} epoch={ckpt_epoch} ≠ 2 → fallback {alt.name}")
        st = torch.load(alt, map_location="cpu", weights_only=False)
        ckpt_epoch = st.get("epoch")
        contract = st.get("contract", {})
    missing, unexpected = model.load_state_dict(st["model"], strict=False)
    model.to(device).eval()

    contract_info = {
        "ckpt": str(ckpt), "ckpt_epoch": ckpt_epoch,
        "encoder_frozen": contract.get("encoder_frozen"),
        "backbone_sha256": contract.get("backbone_sha256"),
        "use_evo": bool(mc.get("use_evo", False)),
        "state_dict_missing": len(missing),
        "state_dict_unexpected": len(unexpected),
    }
    assert contract.get("encoder_frozen") is True, "contract: encoder_frozen must be True"
    print(f"[Model] epoch={ckpt_epoch} use_evo={contract_info['use_evo']} "
          f"load: missing={len(missing)} unexpected={len(unexpected)} → {device}")
    return model, contract_info, device


def predict_windows_gpu(model, samples: dict, window: int, overlap: int,
                        win_cap: int, batch_size: int, device):
    from torch.utils.data import DataLoader
    from data_pipeline.impl.dataset import SlidingWindowContigDataset
    from data_pipeline.impl.collator import HGTCollator
    from data_pipeline.contracts import HGTSample

    hgt_samples, labels_map = [], {}
    for sid, (label, seq) in samples.items():
        hgt_samples.append(HGTSample(
            seq_id=sid, label=label, sequence=seq, r2_sequence=None,
            events=[], sample_type="POS" if label == 1 else "NEG",
            seq_len=len(seq)))
        labels_map[sid] = label

    ds = SlidingWindowContigDataset(
        samples=hgt_samples, window_size=window, overlap=overlap,
        is_training=False, max_neg_windows=9999, max_eval_windows=win_cap,
        extract_bio=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=HGTCollator(pad_id=0), num_workers=4)
    print(f"[Data] {len(hgt_samples)} contigs, {len(ds)} windows "
          f"(win_cap={win_cap}, batch={batch_size})")

    contig_windows = defaultdict(list)
    t0 = time.time()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            output = model(batch, early_exit=False)
            probs = torch.sigmoid(output.cls_logits.squeeze(-1)).float().cpu().numpy()
            sids = batch.original_seq_ids or batch.seq_ids
            for sid, p in zip(sids, probs):
                contig_windows[sid].append(float(p))
            if (i + 1) % 100 == 0:
                el = time.time() - t0
                done = (i + 1) * batch_size
                print(f"  [{min(done,len(ds))}/{len(ds)}] {el:.0f}s "
                      f"({el/(i+1):.2f}s/batch, "
                      f"ETA {(len(loader)-i-1)*el/(i+1)/60:.1f}min)", flush=True)
    infer_s = time.time() - t0
    print(f"[Predict] {len(ds)} windows in {infer_s:.0f}s "
          f"({infer_s/max(1,len(ds)):.3f}s/win)")
    return contig_windows, labels_map, len(ds), infer_s


# ═══════════════════════════════════════════════════════════════════════════ #
#  grouplevelevaluation
# ═══════════════════════════════════════════════════════════════════════════ #

def _agg(kind, w):
    if kind == "max":
        return max(w)
    if kind == "mean":
        return float(np.mean(w))
    return float(np.mean(sorted(w, reverse=True)[:3]))


def group_eval(contig_windows, labels_map, groups, phylum, min_group):
    from sklearn.metrics import roc_auc_score, average_precision_score

    by_group = defaultdict(list)
    for sid in contig_windows:
        by_group[groups[sid][1]].append(sid)

    rows = []
    for g, sids in by_group.items():
        y = np.array([labels_map[s] for s in sids])
        if y.sum() == 0 or y.sum() == len(y):
            continue  # single-class group: AUPRC undefined
        s_max = np.array([_agg("max", contig_windows[s]) for s in sids])
        s_mean = np.array([_agg("mean", contig_windows[s]) for s in sids])
        r = {"genome": g,
             "phylum": phylum.get(g, {}).get("phylum", "Unknown"),
             "organism": phylum.get(g, {}).get("organism", ""),
             "n_contigs": len(sids), "n_pos": int(y.sum()),
             "n_neg": int(len(y) - y.sum())}
        for tag_, s in [("max", s_max), ("mean", s_mean)]:
            r[f"auprc_{tag_}"] = float(average_precision_score(y, s))
            r[f"auroc_{tag_}"] = float(roc_auc_score(y, s))
        rows.append(r)

    rows.sort(key=lambda r: r["auprc_max"])
    evaluable = [r for r in rows if r["n_contigs"] >= min_group]
    a = np.array([r["auprc_max"] for r in evaluable]) if evaluable else np.array([])

    # phylum-level summary (pooling within-group contig scores)
    phyl_rows = []
    by_phyl = defaultdict(list)
    for sid in contig_windows:
        by_phyl[phylum.get(groups[sid][1], {}).get("phylum", "Unknown")].append(sid)
    for ph, sids in by_phyl.items():
        y = np.array([labels_map[s] for s in sids])
        if y.sum() == 0 or y.sum() == len(y):
            continue
        s_max = np.array([_agg("max", contig_windows[s]) for s in sids])
        phyl_rows.append({
            "phylum": ph, "n_genomes": len({groups[s][1] for s in sids}),
            "n_contigs": len(sids), "n_pos": int(y.sum()),
            "auprc_max": float(average_precision_score(y, s_max)),
            "auroc_max": float(roc_auc_score(y, s_max))})
    phyl_rows.sort(key=lambda r: -r["n_contigs"])

    stats = {
        "n_genomes_total": len(by_group),
        "n_genomes_evaluable": len(rows),
        "n_genomes_main_table": len(evaluable),
        "min_group": min_group,
        "group_contigs_median": float(np.median([r["n_contigs"] for r in rows])) if rows else None,
        "auprc_max_mean": float(a.mean()) if len(a) else None,
        "auprc_max_median": float(np.median(a)) if len(a) else None,
        "auprc_max_std": float(a.std(ddof=1)) if len(a) > 1 else None,
        "auprc_max_var": float(a.var(ddof=1)) if len(a) > 1 else None,
        "auprc_max_min": float(a.min()) if len(a) else None,
        "auprc_max_max": float(a.max()) if len(a) else None,
        "frac_genomes_auprc_ge_0.7": float((a >= 0.7).mean()) if len(a) else None,
        "frac_genomes_auprc_lt_0.5": float((a < 0.5).mean()) if len(a) else None,
    }
    if len(a) > 2 and len(rows) > 2:
        ns = np.array([r["n_contigs"] for r in evaluable], dtype=float)
        stats["corr_groupsize_auprc"] = float(np.corrcoef(ns, a)[0, 1])
    return rows, evaluable, phyl_rows, stats


def write_md(out_dir: Path, tag, payload, rows, evaluable, phyl_rows, stats,
             isolation, infer_s, total_s):
    md = []
    md.append("# OOD cross-taxa  —  M0 (NTv3-only, ep2) grouplevel AUPRC (genome-level proxy)\n\n")
    md.append(f"> generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  ·  task B  ·  "
              "`scripts/crosstaxa_eval.py`\n\n")
    md.append("> model: `checkpoints/Contig_500k_ntv3only/best_model.pt` (ep2, "
              "use_evo=False).\n\n")
    md.append("> protocol: Contig_500k/test full 50,000 contigs; sliding window 3000/300, "
              "win_cap=10, batch=8 (same type as task A / the real-anchor evaluation); "
              "contig aggregation max/mean. Grouping: the genome (GCF accession) of the "
              "first HOST segment's source_file in provenance.tsv.\n\n")
    if isolation:
        md.append(f"> **OOD convention**: the test genome pool is fully isolated from train "
                  f"(microbe train {isolation['train_genomes']:,} / "
                  f"test {isolation['test_genomes']}, "
                  f"overlap={isolation['overlap']}, "
                  f"ratios={isolation['ratios']}, seed={isolation['seed']}) → "
                  f"each genome's AUPRC is generalization to that **unseen recipient genome**.\n\n")
    md.append("> **Proxy statement**: genome-level (GCF) as a proxy layer for taxa  —  —  "
              "GCF accessions cannot be queried directly against NCBI taxonomy; phylum is "
              "approximated from the genome FASTA header organism's genus name "
              "(v6.8 `prepare_cross_taxa.py` PHYLUM_MAP, reused read-only).\n\n")

    md.append("## 1. Between-group distribution (per-genome AUPRC, max aggregation)\n\n")
    md.append("| metric | value |\n|---|---|")
    md.append(f"| evaluable genomes (both classes present) | {stats['n_genomes_evaluable']} / "
              f"{stats['n_genomes_total']} |")
    md.append(f"| genomes in the primary table (n≥{stats['min_group']}) | {stats['n_genomes_main_table']} |")
    md.append(f"| median contigs per group | {stats['group_contigs_median']:.0f} |")
    if stats["auprc_max_mean"] is not None:
        md.append(f"| AUPRC mean ± std | **{stats['auprc_max_mean']:.4f} ± "
                  f"{stats['auprc_max_std']:.4f}** |")
        md.append(f"| AUPRC median | {stats['auprc_max_median']:.4f} |")
        md.append(f"| AUPRC min / max | {stats['auprc_max_min']:.4f} / "
                  f"{stats['auprc_max_max']:.4f} |")
        md.append(f"| **between-group variance** σ² | {stats['auprc_max_var']:.5f} "
                  f"(σ={stats['auprc_max_std']:.4f}) |")
        md.append(f"| P(AUPRC≥0.7) | {stats['frac_genomes_auprc_ge_0.7']:.1%} |")
        md.append(f"| P(AUPRC<0.5) | {stats['frac_genomes_auprc_lt_0.5']:.1%} |")
    if "corr_groupsize_auprc" in stats and stats.get("corr_groupsize_auprc") is not None:
        md.append(f"| corr(group size, AUPRC) | {stats['corr_groupsize_auprc']:+.3f} "
                  f"(small-sample noise check) |")
    pooled = payload.get("pooled")
    if pooled:
        md.append(f"| pooled reference (this protocol, 50k) | AUPRC {pooled['auprc_max']:.4f} / "
                  f"AUROC {pooled['auroc_max']:.4f} "
                  f"(M0 final convention 0.827, see REAL_ANCHORS_M0final.md) |")
    md.append("")

    md.append("## 2. Phylum-level summary (within-group pooling)\n\n")
    md.append("| Phylum | genomes | contigs | pos | AUPRC(max) | AUROC(max) |")
    md.append("|---|---|---|---|---|---|")
    for r in phyl_rows:
        md.append(f"| {r['phylum']} | {r['n_genomes']} | {r['n_contigs']:,} | "
                  f"{r['n_pos']:,} | {r['auprc_max']:.4f} | {r['auroc_max']:.4f} |")
    md.append("")

    md.append("## 3. Hardest/easiest genomes (n≥min_group, 5 each)\n\n")
    md.append("| genome | phylum | organism | n | AUPRC(max) | AUROC(max) |")
    md.append("|---|---|---|---|---|---|")

    def _row(r):
        org = (r["organism"][:44] + "...") if len(r["organism"]) > 45 else r["organism"]
        return (f"| {r['genome']} | {r['phylum']} | {org} | {r['n_contigs']} | "
                f"{r['auprc_max']:.4f} | {r['auroc_max']:.4f} |")
    for r in evaluable[:5]:
        md.append(_row(r))
    for r in reversed(evaluable[-5:]):
        md.append(_row(r))
    md.append("")

    md.append("## 4. All evaluable genomes table\n\n")
    md.append(f"The full per-genome table is in `{tag}_groups.csv` (groups with both classes, sorted ascending by AUPRC).\n\n")
    md.append("## 5. artifact\n\n")
    md.append(f"- `{tag}.json` / `{tag}.npz` (per-contig window probabilities + genome attribution, verifiable)\n")
    md.append(f"- `{tag}_groups.csv` (per genome)\n\n")
    md.append(f"infer={infer_s/60:.1f}min total={total_s/60:.1f}min\n")

    out = out_dir / "OOD_CROSSTAXA_M0.md"
    with open(out, "w") as f:
        f.write("\n".join(md))
    print(f"[Report] saved {out}")


# ═══════════════════════════════════════════════════════════════════════════ #

def main():
    t_start = time.time()
    out_dir = Path(ARGS.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = ARGS.tag

    groups = parse_provenance_groups(TEST_DIR / "provenance.tsv")
    phylum = annotate_phylum({g for _, g in groups.values()})
    isolation = split_isolation()
    print(f"[Isolation] {isolation}")

    by_g = defaultdict(list)
    for s, (_, g) in groups.items():
        by_g[g].append(_)
    print(f"[Groups] {len(by_g)} genomes | median contigs/genome="
          f"{np.median([len(v) for v in by_g.values()]):.0f} | "
          f"phyla={len({phylum.get(g, {}).get('phylum', 'Unknown') for g in by_g})}")

    if ARGS.plan:
        print("[Plan] CPU dry run finished (no model loaded)")
        return

    gpu_name = assert_4090()
    samples = read_test_samples(TEST_DIR / "dataset.fasta", groups, ARGS.limit)
    model, contract_info, device = load_model_gpu(Path(ARGS.ckpt), Path(ARGS.config))

    import yaml as _yaml
    _dc = _yaml.safe_load(open(ARGS.config))["data"]
    contig_windows, labels_map, n_windows, infer_s = predict_windows_gpu(
        model, samples, window=_dc["sw_window_size"], overlap=_dc["sw_overlap"],
        win_cap=ARGS.win_cap, batch_size=ARGS.batch, device=device)
    del model
    torch.cuda.empty_cache()

    # pooled reference (this protocol's convention)
    results = cpu.aggregate_and_eval(contig_windows, labels_map)
    pooled = {"auprc_max": results["max"]["auprc"], "auroc_max": results["max"]["auc"],
              "auprc_mean": results["mean"]["auprc"], "auroc_mean": results["mean"]["auc"]}

    rows, evaluable, phyl_rows, stats = group_eval(
        contig_windows, labels_map, groups, phylum, ARGS.min_group)

    sid_list = sorted(contig_windows.keys())
    np.savez(out_dir / f"{tag}.npz",
             contig_ids=np.array(sid_list, dtype=object),
             labels=np.array([labels_map[s] for s in sid_list], dtype=np.int64),
             genome=np.array([groups[s][1] for s in sid_list], dtype=object),
             n_windows=np.array([len(contig_windows[s]) for s in sid_list],
                                dtype=np.int64),
             window_probs=np.array(
                 [np.asarray(contig_windows[s], dtype=np.float32) for s in sid_list],
                 dtype=object),
             agg_max=np.array([_agg("max", contig_windows[s]) for s in sid_list],
                              dtype=np.float32),
             agg_mean=np.array([_agg("mean", contig_windows[s]) for s in sid_list],
                               dtype=np.float32))

    import csv
    with open(out_dir / f"{tag}_groups.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                             ["genome", "phylum", "organism", "n_contigs"])
        wtr.writeheader()
        wtr.writerows(rows)

    payload = {
        "task": "B_OOD_cross_taxa",
        "protocol": {
            "source": "task B: eval_real_anchors windowing/aggregation reused (GPU); "
                      "grouping logic references v8.0 prepare_cross_taxa.py (read-only)",
            "window_size": _dc["sw_window_size"], "overlap": _dc["sw_overlap"],
            "batch_size": ARGS.batch, "max_eval_windows": ARGS.win_cap,
            "aggregations": ["max", "mean"],
            "device": "cuda", "gpu": gpu_name, "dtype": "fp32",
            "grouping": "first HOST segment's source_file basename in provenance.tsv (GCF)",
            "proxy_note": "genome-level taxa proxy; phylum approximated from the "
                          "header organism's genus name (v6.8 PHYLUM_MAP)",
        },
        "contract": contract_info,
        "split_isolation": isolation,
        "group_stats": stats,
        "phylum_rows": phyl_rows,
        "pooled": pooled,
        "d0_ref_auprc": D0_REF_AUPRC,
        "infer_seconds": infer_s,
        "total_seconds": time.time() - t_start,
    }
    with open(out_dir / f"{tag}.json", "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False,
                  default=lambda o: float(o) if isinstance(o, np.floating) else o)

    write_md(out_dir, tag, payload, rows, evaluable, phyl_rows, stats,
             isolation, infer_s, time.time() - t_start)

    print(f"\n{'='*70}")
    print(f"[OOD cross-taxa] genomes evaluable={stats['n_genomes_evaluable']} | "
          f"AUPRC {stats['auprc_max_mean']:.4f} ± {stats['auprc_max_std']:.4f} | "
          f"σ²={stats['auprc_max_var']:.5f}")
    print(f"  pooled (50k, this protocol): AUPRC {pooled['auprc_max']:.4f} "
          f"/ AUROC {pooled['auroc_max']:.4f}")
    print(f"  infer={infer_s/60:.1f}min total={(time.time()-t_start)/60:.1f}min")
    print(f"Saved: {out_dir}/{tag}.json + .npz + _groups.csv + OOD_CROSSTAXA_M0.md")


if __name__ == "__main__":
    main()
