#!/usr/bin/env python3
# scripts/cross_taxa_eval.py
"""DeepFluHGT — task B: OOD cross-taxa grouped evaluation (M0, 4090).

Logic reference (read-only): ../DeepFluHGT.8.0/scripts/prepare_cross_taxa.py
  (= scripts/prepare_cross_taxa.py, verbatim identical) — a genus→phylum static map
  PHYLUM_MAP approximates phylum-level classification from the organism name in genome FASTA headers.

this script adds the following on top of it:
  1. Contig_500k/test/provenance.tsv by source_file (seg_type=HOST of
     microbe genome GCF file) — multi-source-host contigs take the first HOST row as the primary genome,
     (and the multi-source fraction is counted).
  2. GCF → local benchmark genome library header (/data/.../HGT/microbe_genomes) →
     organism → phylum. Genera outside PHYLUM_MAP use the curated extension table in this script,
     GENUS_PHYLUM_EXT (only common genera with undisputed phylum-level assignment are included; the rest → Unknown, kept as
     separate groups and reported as such). No NCBI metadata is available → phylum-level grouping is a "header approximation",
     and species-level and genome (per-GCF) level groupings are also given as proxies (as stated in the task brief).
  3. M0 inference (re-usetask A inferenceprotocol: window=3000/overlap=300/win_cap=10/batch=8,
     aggregation max/mean/top3_mean, verbatim reuse eval_real_anchors ofaggregation/metric) →
     eachgroup AUPRC/AUROC + DeLong → docs/evidence/OOD_CROSSTAXA_M0.md.
  4. --plan: only print the grouping table (CPU, seconds; no inference, never touches the GPU).

GPU discipline: CUDA_VISIBLE_DEVICES=4090 UUID GPU lock (A100 strictly forbidden); assert the only visible device
  is the 4090; only this one GPU process may run at any moment (guaranteed by chain serialization).

usage (launcher: scripts/run/difficulty_curve_chain.sh):
  python scripts/cross_taxa_eval.py --plan  # grouping health check (CPU)
  CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python scripts/cross_taxa_eval.py
"""
from __future__ import annotations

import os
import sys
import json
import math
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
GENOME_DIR = Path("./microbe_genomes")

D0_TEST_REF = {"auprc": 0.827, "auroc": 0.827,
               "source": "docs/evidence/REAL_ANCHORS_M0final.md (M0 finalized test)"}

# -- curated genus→phylum extension (common genera not yet covered by PHYLUM_MAP; undisputed phylum-level assignments) --------
# nomenclature consistent with PHYLUM_MAP (classic NCBI phylum names: Proteobacteria / Tenericutes etc.).
# genera in neither this table nor PHYLUM_MAP → "Unknown" (kept as a separate group and reported as such, no guessing).
GENUS_PHYLUM_EXT = {
    # Proteobacteria (Alpha/Beta/Gamma/Delta — merged at phylum level, same convention as PHYLUM_MAP)
    "wolbachia": "Proteobacteria", "xanthomonas": "Proteobacteria",
    "paraburkholderia": "Proteobacteria", "bradyrhizobium": "Proteobacteria",
    "bartonella": "Proteobacteria", "sphingomonas": "Proteobacteria",
    "comamonas": "Proteobacteria", "brucella": "Proteobacteria",
    "thauera": "Proteobacteria", "neisseria": "Proteobacteria",
    "marinobacter": "Proteobacteria", "caminibacter": "Proteobacteria",
    "devosia": "Proteobacteria", "brevundimonas": "Proteobacteria",
    "pseudoalteromonas": "Proteobacteria", "aeromonas": "Proteobacteria",
    "alteromonas": "Proteobacteria", "idiomarina": "Proteobacteria",
    "colwellia": "Proteobacteria", "glaciecola": "Proteobacteria",
    "halomonas": "Proteobacteria", "marinomonas": "Proteobacteria",
    "ralstonia": "Proteobacteria", "methylobacterium": "Proteobacteria",
    "achromobacter": "Proteobacteria", "alcaligenes": "Proteobacteria",
    "bordetella": "Proteobacteria", "citrobacter": "Proteobacteria",
    "enterobacter": "Proteobacteria", "pantoea": "Proteobacteria",
    "proteus": "Proteobacteria", "aliivibrio": "Proteobacteria",
    "arcobacter": "Proteobacteria", "agrobacterium": "Proteobacteria",
    "mesorhizobium": "Proteobacteria", "sinorhizobium": "Proteobacteria",
    "azospirillum": "Proteobacteria", "stenotrophomonas": "Proteobacteria",
    "rhodobacter": "Proteobacteria", "erythrobacter": "Proteobacteria",
    "novosphingobium": "Proteobacteria", "sphingopyxis": "Proteobacteria",
    "sphingobium": "Proteobacteria", "xylella": "Proteobacteria",
    "lysobacter": "Proteobacteria", "alcanivorax": "Proteobacteria",
    "pelagibacter": "Proteobacteria", "methylomicrobium": "Proteobacteria",
    "methylophaga": "Proteobacteria", "kangiella": "Proteobacteria",
    "oceanospirillum": "Proteobacteria", "sulfurovum": "Proteobacteria",
    "sulfurimonas": "Proteobacteria", "desulfovibrio": "Proteobacteria",
    "geobacter": "Proteobacteria", "bdellovibrio": "Proteobacteria",
    "myxococcus": "Proteobacteria", "stigmatella": "Proteobacteria",
    # Actinobacteria
    "gordonia": "Actinobacteria", "nocardioides": "Actinobacteria",
    "microlunatus": "Actinobacteria", "kocuria": "Actinobacteria",
    "micrococcus": "Actinobacteria", "rothia": "Actinobacteria",
    "brevibacterium": "Actinobacteria", "dietzia": "Actinobacteria",
    "janibacter": "Actinobacteria", "leifsonia": "Actinobacteria",
    "nocardiopsis": "Actinobacteria", "williamsia": "Actinobacteria",
    "cellulomonas": "Actinobacteria", "agrococcus": "Actinobacteria",
    "kytococcus": "Actinobacteria", "streptosporangium": "Actinobacteria",
    "saccharopolyspora": "Actinobacteria", "clavibacter": "Actinobacteria",
    "curtobacterium": "Actinobacteria", "rathayibacter": "Actinobacteria",
    # Bacteroidetes
    "mucilaginibacter": "Bacteroidetes", "pedobacter": "Bacteroidetes",
    "hymenobacter": "Bacteroidetes", "kaistella": "Bacteroidetes",
    "chryseobacterium": "Bacteroidetes", "tenacibaculum": "Bacteroidetes",
    "gramella": "Bacteroidetes", "maribacter": "Bacteroidetes",
    "zobellia": "Bacteroidetes", "winogradskyella": "Bacteroidetes",
    "pontibacter": "Bacteroidetes", "spirosoma": "Bacteroidetes",
    "flectobacillus": "Bacteroidetes", "runella": "Bacteroidetes",
    "arcicella": "Bacteroidetes", "olivibacter": "Bacteroidetes",
    "terrimonas": "Bacteroidetes", "niastella": "Bacteroidetes",
    "sphingobacterium": "Bacteroidetes",
    # Firmicutes
    "brochothrix": "Firmicutes", "carnobacterium": "Firmicutes",
    "vagococcus": "Firmicutes", "jeotgalicoccus": "Firmicutes",
    "macrococcus": "Firmicutes", "lactiplantibacillus": "Firmicutes",
    "limosilactobacillus": "Firmicutes", "lacticaseibacillus": "Firmicutes",
    "latilactobacillus": "Firmicutes", "lendilactobacillus": "Firmicutes",
    "schleiferilactobacillus": "Firmicutes", "fructilactobacillus": "Firmicutes",
    # Tenericutes (Mollicutes; same convention as PHYLUM_MAP's mycoplasma→Tenericutes)
    "spiroplasma": "Tenericutes", "mycoplasmopsis": "Tenericutes",
    "entomoplasma": "Tenericutes", "mesoplasma": "Tenericutes",
    "acholeplasma": "Tenericutes",
    # Chlamydiae
    "chlamydia": "Chlamydiae",
    # Euryarchaeota
    "natrinema": "Euryarchaeota", "haloarcula": "Euryarchaeota",
    "halorubrum": "Euryarchaeota", "natrialba": "Euryarchaeota",
    "haloterrigena": "Euryarchaeota", "halomicrobium": "Euryarchaeota",
    "haloferax": "Euryarchaeota", "thermococcus": "Euryarchaeota",
    "pyrococcus": "Euryarchaeota", "archaeoglobus": "Euryarchaeota",
    "methanolobus": "Euryarchaeota", "natronomonas": "Euryarchaeota",
    # Thaumarchaeota
    "nitrosopumilus": "Thaumarchaeota", "nitrososphaera": "Thaumarchaeota",
    # Cyanobacteria
    "cyanothece": "Cyanobacteria", "acaryochloris": "Cyanobacteria",
    "thermosynechococcus": "Cyanobacteria", "trichodesmium": "Cyanobacteria",
    "crocosphaera": "Cyanobacteria", "nodularia": "Cyanobacteria",
    "aphanizomenon": "Cyanobacteria", "dolichospermum": "Cyanobacteria",
    "raphidiopsis": "Cyanobacteria", "leptolyngbya": "Cyanobacteria",
    "pseudanabaena": "Cyanobacteria", "synechocystis": "Cyanobacteria",
    "gloeobacter": "Cyanobacteria", "calothrix": "Cyanobacteria",
    "fischerella": "Cyanobacteria", "rivularia": "Cyanobacteria",
    # Deinococcus-Thermus
    "deinococcus": "Deinococcus-Thermus", "thermus": "Deinococcus-Thermus",
    "meiothermus": "Deinococcus-Thermus",
    # Planctomycetes
    "planctomyces": "Planctomycetes", "rhodopirellula": "Planctomycetes",
    "blastopirellula": "Planctomycetes", "pirellula": "Planctomycetes",
    "gemmata": "Planctomycetes", "singulisphaera": "Planctomycetes",
    "gimesia": "Planctomycetes",
    # Verrucomicrobia
    "verrucomicrobium": "Verrucomicrobia", "akkermansia": "Verrucomicrobia",
    "opitutus": "Verrucomicrobia", "chthoniobacter": "Verrucomicrobia",
    "luteolibacter": "Verrucomicrobia", "haloferula": "Verrucomicrobia",
    # Fusobacteria
    "cetobacterium": "Fusobacteria", "ilyobacter": "Fusobacteria",
    # Aquificae / Thermotogae / Nitrospirae / Acidobacteria / Chloroflexi
    "aquifex": "Aquificae", "sulfurihydrogenibium": "Aquificae",
    "thermocrinis": "Aquificae", "persephonella": "Aquificae",
    "thermotoga": "Thermotogae", "fervidobacterium": "Thermotogae",
    "petrotoga": "Thermotogae", "thermosipho": "Thermotogae",
    "nitrospira": "Nitrospirae", "leptospirillum": "Nitrospirae",
    "acidobacterium": "Acidobacteria", "granulicella": "Acidobacteria",
    "terriglobus": "Acidobacteria",
    "chloroflexus": "Chloroflexi", "roseiflexus": "Chloroflexi",
    "herpetosiphon": "Chloroflexi", "ktedonobacter": "Chloroflexi",
}


def parse_args():
    ap = argparse.ArgumentParser(description="OOD cross-taxa grouped evaluation (task B)")
    ap.add_argument("--plan", action="store_true",
                    help="only print the grouping table (CPU, no inference)")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--win_cap", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="smoke: eachclassbefore N item")
    ap.add_argument("--min_phylum_n", type=int, default=100)
    ap.add_argument("--min_species_n", type=int, default=40)
    ap.add_argument("--min_genome_n", type=int, default=20)
    ap.add_argument("--gpu_uuid", default=GPU_4090_UUID)
    ap.add_argument("--tag", default="m0")
    ap.add_argument("--out_dir", default=str(ROOT / "docs/evidence"))
    return ap.parse_args()


ARGS = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = ARGS.gpu_uuid

import eval_real_anchors as cpu # noqa: E402 (aggregation/metricverbatim reuse)
os.environ["CUDA_VISIBLE_DEVICES"] = ARGS.gpu_uuid # establishnamelyresume

from delong_test import delong_test_full # noqa: E402
from prepare_cross_taxa import ( # noqa: E402 (v6.8 read-onlyreplicated, samefile)
    get_phylum_from_name, extract_organism_from_fasta)

if not ARGS.plan:
    import torch # noqa: E402


# =========================================================================== #
# splitgroupbuild (CPU): provenance → primaryhost primarygenome → phylum/species/genome
# =========================================================================== #

def phylum_of_organism(organism: str) -> str:
    """organism → phylum: PHYLUM_MAP first, curated extension table as fallback, else Unknown."""
    if not organism:
        return "Unknown"
    words = organism.lower().split()
    if words[0] == "candidatus" and len(words) > 1: # "Candidatus Pelagibacter ..."
        words = words[1:]
    genus = words[0]
    ph = get_phylum_from_name(" ".join(words)) # PHYLUM_MAP (including fallback)
    if ph in ("Unknown", "Bacteria_Unknown") and genus in GENUS_PHYLUM_EXT:
        return GENUS_PHYLUM_EXT[genus]
    if ph == "Unknown" and genus in GENUS_PHYLUM_EXT:
        return GENUS_PHYLUM_EXT[genus]
    return ph


def build_groups(test_dir: Path):
    """→ label_map, host_primary (seq_id→basename), genome→{organism,phylum}."""
    label, primary = {}, {}
    n_multi = 0
    with open(test_dir / "provenance.tsv") as f:
        header = f.readline().rstrip("\n").split("\t")
        i_sid, i_lab = header.index("seq_id"), header.index("label")
        i_seg, i_src = header.index("seg_type"), header.index("source_file")
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) <= i_src:
                continue
            sid = p[i_sid]
            label[sid] = int(p[i_lab])
            if p[i_seg] == "HOST" and p[i_src]:
                b = Path(p[i_src]).name
                if sid in primary:
                    if primary[sid] != b:
                        n_multi += 1
                else:
                    primary[sid] = b
    print(f"[Provenance] contigs={len(label)} "
          f"pos={sum(label.values())} multi-host(different-source) contig rows={n_multi}")

    gmeta, n_missing = {}, 0
    for b in sorted(set(primary.values())):
        fp = GENOME_DIR / b
        if fp.is_file():
            org = extract_organism_from_fasta(str(fp))
            gmeta[b] = {"organism": org, "phylum": phylum_of_organism(org),
                        "file_found": True}
        else:
            n_missing += 1
            gmeta[b] = {"organism": "", "phylum": "Unknown",
                        "file_found": False}
    print(f"[Genomes] distinct={len(gmeta)} headermissing(localnonefile)={n_missing}")
    return label, primary, gmeta


def group_counts(label, primary, gmeta, key):
    """key: 'phylum'|'species'|'genome' → {group: [n_pos, n_neg]}."""
    def g_of(b):
        m = gmeta[b]
        if key == "phylum":
            return m["phylum"]
        if key == "species":
            return m["organism"] or "Unknown_organism"
        return b
    cnt = defaultdict(lambda: [0, 0])
    for sid, lab in label.items():
        if sid in primary:
            cnt[g_of(primary[sid])][lab] += 1
    return cnt


# =========================================================================== #
# inference (protocolwithtask A / eval_real_anchors_gpu verbatimconsistent)
# =========================================================================== #

def assert_4090():
    n = torch.cuda.device_count()
    assert n == 1, f"exactly 1 visible GPU expected (4090 GPU lock), actual {n}"
    name = torch.cuda.get_device_name(0)
    assert "4090" in name, f"visible GPU is not RTX 4090: {name}"
    free, total = torch.cuda.mem_get_info(0)
    print(f"[GPU] {name} | mem free {free/1e9:.1f}/{total/1e9:.1f} GB")
    return name


def load_model_gpu(ckpt: Path, config: Path):
    import yaml
    from model.api import build_model
    from model.contracts import HGTConfig
    device = torch.device("cuda:0")
    mc = yaml.safe_load(open(config))["model"]
    cfg = HGTConfig(
        data_mode="contig", max_contig_length=3000,
        hidden_dim=mc["hidden_dim"], num_attention_heads=mc["num_attention_heads"],
        dropout=mc["dropout"], graph_top_k=mc["graph_top_k"],
        bio_feature_dim=mc["bio_feature_dim"],
        pretrained_nt_path=mc.get("pretrained_nt_path", str(NT_FALLBACK)),
        pretrained_caduceus_path="", use_local_encoder=False,
        use_evo=bool(mc.get("use_evo", False)),
        pretrained_evo_dim=int(mc.get("pretrained_evo_dim", 4096)),
        early_exit_threshold=0.95)
    model = build_model(cfg)
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    ckpt_epoch = st.get("epoch")
    if ckpt_epoch != 2:
        alt = ckpt.parent / "epoch_2.pt"
        print(f"[WARN] {ckpt.name} epoch={ckpt_epoch} ≠ 2 → fallback {alt.name}")
        st = torch.load(alt, map_location="cpu", weights_only=False)
        ckpt_epoch = st.get("epoch")
    contract = st.get("contract", {})
    missing, unexpected = model.load_state_dict(st["model"], strict=False)
    model.to(device).eval()
    assert contract.get("encoder_frozen") is True
    print(f"[Model] epoch={ckpt_epoch} use_evo={bool(mc.get('use_evo', False))} "
          f"missing={len(missing)} unexpected={len(unexpected)} → {device}")
    return model, device


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
                print(f" [{min(done,len(ds))}/{len(ds)}] {el:.0f}s "
                      f"({el/(i+1):.2f}s/batch, "
                      f"ETA {(len(loader)-i-1)*el/(i+1)/60:.1f}min)", flush=True)
    infer_s = time.time() - t0
    print(f"[Predict] {len(ds)} windows in {infer_s:.0f}s "
          f"({infer_s/max(1,len(ds)):.3f}s/win)")
    return contig_windows, labels_map, len(ds), infer_s


def delong_single(y, scores):
    r = delong_test_full(y, scores, scores)
    se = math.sqrt(r["var_a"])
    auc = r["auc_a"]
    z = (auc - 0.5) / se if se > 0 else math.inf
    return {"auc": auc, "se": se,
            "ci95": [auc - 1.959963984540054 * se,
                     auc + 1.959963984540054 * se],
            "z_vs_half": z, "p_vs_half": math.erfc(abs(z) / math.sqrt(2.0)),
            "n": int(len(y)), "n_pos": int(np.sum(y == 1))}


def read_test_samples(test_dir: Path, limit: int):
    """dataset.fasta + provenance tag → {seq_id: (label, seq)}."""
    label, primary, gmeta = build_groups(test_dir)
    samples, header, buf = {}, None, []
    with open(test_dir / "dataset.fasta") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None and header in label:
                    samples[header] = (label[header], "".join(buf))
                header = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.upper())
    if header is not None and header in label:
        samples[header] = (label[header], "".join(buf))
    if limit > 0:
        pos = {k: v for k, v in samples.items() if v[0] == 1}
        neg = {k: v for k, v in samples.items() if v[0] == 0}
        samples = dict(list(pos.items())[:limit])
        samples.update(dict(list(neg.items())[:limit]))
        print(f"[Smoke] limited to {len(samples)} contigs")
    return samples, label, primary, gmeta


# =========================================================================== #
# metric: groupwithin AUPRC/AUROC (max aggregationprimarycaliber)
# =========================================================================== #

def group_metrics(contig_windows, labels_map, group_of, min_n):
    from sklearn.metrics import roc_auc_score, average_precision_score
    members = defaultdict(list)
    for sid in contig_windows:
        g = group_of(sid)
        if g is not None:
            members[g].append(sid)
    out = {}
    for g, sids in members.items():
        y = np.array([labels_map[s] for s in sids])
        if len(sids) < min_n or y.sum() == 0 or y.sum() == len(y):
            continue
        s_max = np.array([max(contig_windows[s]) for s in sids])
        s_mean = np.array([float(np.mean(contig_windows[s])) for s in sids])
        out[g] = {
            "n": len(sids), "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum()),
            "auprc_max": float(average_precision_score(y, s_max)),
            "auroc_max": float(roc_auc_score(y, s_max)),
            "auprc_mean": float(average_precision_score(y, s_mean)),
            "auroc_mean": float(roc_auc_score(y, s_mean)),
            "delong": delong_single(y, s_max),
        }
    return out


# =========================================================================== #

def main():
    out_dir = Path(ARGS.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label, primary, gmeta = build_groups(TEST_DIR)

    phyl_cnt = group_counts(label, primary, gmeta, "phylum")
    if ARGS.plan:
        print(f"\n{'phylum':24s} {'contigs':>8s} {'pos':>7s} {'neg':>7s} {'genomes':>8s}")
        genomes_per_phylum = defaultdict(set)
        for b, m in gmeta.items():
            genomes_per_phylum[m["phylum"]].add(b)
        for ph, (p, n) in sorted(phyl_cnt.items(), key=lambda x: -sum(x[1])):
            print(f"{ph:24s} {p+n:>8d} {p:>7d} {n:>7d} "
                  f"{len(genomes_per_phylum[ph]):>8d}")
        sp_cnt = group_counts(label, primary, gmeta, "species")
        print(f"\nspecies-level groups: {len(sp_cnt)} "
              f"(≥{ARGS.min_species_n} contigs: "
              f"{sum(1 for p,n in sp_cnt.values() if p+n >= ARGS.min_species_n)})")
        print("[plan] completed (not yetinference)")
        return

    gpu_name = assert_4090()
    t_start = time.time()
    samples, label, primary, gmeta = read_test_samples(TEST_DIR, ARGS.limit)
    model, device = load_model_gpu(Path(ARGS.ckpt), Path(ARGS.config))

    import yaml as _yaml
    _dc = _yaml.safe_load(open(ARGS.config))["data"]
    contig_windows, labels_map, n_windows, infer_s = predict_windows_gpu(
        model, samples, window=_dc["sw_window_size"], overlap=_dc["sw_overlap"],
        win_cap=ARGS.win_cap, batch_size=ARGS.batch, device=device)
    del model
    torch.cuda.empty_cache()

    # overall consistency check (against the M0 finalized test 0.827 reference; protocol differences noted)
    overall = cpu.aggregate_and_eval(contig_windows, labels_map)
    wstats = cpu.window_stats(contig_windows, labels_map)

    sid_list = sorted(contig_windows.keys())
    y_all = np.array([labels_map[s] for s in sid_list], dtype=np.int64)
    s_max = np.array([max(contig_windows[s]) for s in sid_list], dtype=np.float32)
    delong_all = delong_single(y_all, s_max)

    def g_phylum(sid):
        return gmeta[primary[sid]]["phylum"] if sid in primary else None

    def g_species(sid):
        return (gmeta[primary[sid]]["organism"] or "Unknown_organism") \
            if sid in primary else None

    def g_genome(sid):
        return primary.get(sid)

    phyl_m = group_metrics(contig_windows, labels_map, g_phylum,
                           ARGS.min_phylum_n)
    sp_m = group_metrics(contig_windows, labels_map, g_species,
                         ARGS.min_species_n)
    gn_m = group_metrics(contig_windows, labels_map, g_genome,
                         ARGS.min_genome_n)

    # genome-level AUPRC distribution (OOD proxy: one AUPRC per genome)
    gn_auprc = np.array([v["auprc_max"] for v in gn_m.values()])
    gn_stats = {
        "n_genomes": int(len(gn_auprc)),
        "mean": float(gn_auprc.mean()), "median": float(np.median(gn_auprc)),
        "p10": float(np.percentile(gn_auprc, 10)),
        "p25": float(np.percentile(gn_auprc, 25)),
        "p75": float(np.percentile(gn_auprc, 75)),
        "p90": float(np.percentile(gn_auprc, 90)),
        "min": float(gn_auprc.min()), "frac_below_0.5": float((gn_auprc < 0.5).mean()),
    }
    worst = sorted(gn_m.items(), key=lambda kv: kv[1]["auprc_max"])[:15]
    best = sorted(gn_m.items(), key=lambda kv: -kv[1]["auprc_max"])[:5]

    # archived
    np.savez(out_dir / f"ood_crosstaxa_{ARGS.tag}.npz",
             contig_ids=np.array(sid_list, dtype=object), labels=y_all,
             n_windows=np.array([len(contig_windows[s]) for s in sid_list],
                                dtype=np.int64),
             window_probs=np.array(
                 [np.asarray(contig_windows[s], dtype=np.float32) for s in sid_list],
                 dtype=object),
             agg_max=s_max)

    payload = {
        "task": "B — OOD cross-taxa (M0)",
        "protocol": {
            "source": "v6.8 prepare_cross_taxa.py logic replicated + curated genus→phylum extension",
            "multi_host_policy": "multi-source-host contigs take the first provenance HOST row as the primary genome",
            "window_size": _dc["sw_window_size"], "overlap": _dc["sw_overlap"],
            "max_eval_windows": ARGS.win_cap, "batch_size": ARGS.batch,
            "gpu": gpu_name,
            "min_phylum_n": ARGS.min_phylum_n,
            "min_species_n": ARGS.min_species_n,
            "min_genome_n": ARGS.min_genome_n,
        },
        "overall": {"results_max": overall["max"], "results_mean": overall["mean"],
                    "window_stats": wstats, "delong_max": delong_all,
                    "d0_reference": D0_TEST_REF},
        "phylum_groups": phyl_m,
        "species_groups": sp_m,
        "genome_stats": gn_stats,
        "genome_worst15": {k: v["auprc_max"] for k, v in worst},
        "genome_best5": {k: v["auprc_max"] for k, v in best},
        "infer_seconds": infer_s, "total_seconds": time.time() - t_start,
    }
    with open(out_dir / f"ood_crosstaxa_{ARGS.tag}.json", "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False,
                  default=lambda o: float(o) if isinstance(o, np.floating) else o)

    # -- report --
    md = []
    md.append("# OOD cross-taxa — M0 (NTv3-only, ep2) grouped evaluation\n")
    md.append("> generate: %s · task B · `scripts/cross_taxa_eval.py`\n"
              % time.strftime("%Y-%m-%d %H:%M:%S"))
    md.append("> grouping basis: the HOST source_file (GCF genome) column of Contig_500k/test provenance.tsv "
              "source_file (GCF genome) → localgenome header organism → phylumlevel "
              "(v6.8 `prepare_cross_taxa.py` PHYLUM_MAP + this script's curated genus→phylum"
              "extension table; **no NCBI metadata available → the phylum level is a header approximation**). species-level and "
              "per-genome-level groupings are also given side by side as proxies (as stated in the task brief).\n")
    md.append("> multi-source-host contigs take the first HOST row as the primary genome; the inference protocol matches task A / the real anchors "
              "consistent (3000/300, win_cap=10, batch=8, fp32, max/mean aggregation).\n")

    md.append("\n## 1. Overall (re-measured under the same protocol vs the M0 finalized run)\n")
    om = overall["max"]
    md.append(f"- this run: n_contigs={wstats['n_contigs']:,} / "
              f"windows={wstats['n_windows']:,} | "
              f"AUPRC(max)={om.get('auprc', float('nan')):.4f} · "
              f"AUROC(max)={om.get('auc', float('nan')):.4f} "
              f"(mean caliber AUPRC={overall['mean'].get('auprc', float('nan')):.4f})\n"
              f"- M0 finalized test reference: AUPRC 0.827 / AUROC 0.827 "
              f"({D0_TEST_REF['source']}; training-pipeline aggregation protocol, the value difference stems from the protocol difference)\n")

    md.append("\n## 2. phylumlevel (phylum) splitgroup AUPRC\n")
    md.append(f"(within-group n ≥ {ARGS.min_phylum_n} and both classes non-zero)\n")
    md.append("| phylum | n | pos | neg | AUPRC(max) | AUROC(max) | "
              "AUPRC(mean) | DeLong 95% CI |")
    md.append("|---|---|---|---|---|---|---|---|")
    for g, v in sorted(phyl_m.items(), key=lambda kv: kv[1]["auprc_max"]):
        d = v["delong"]
        md.append(f"| {g} | {v['n']:,} | {v['n_pos']:,} | {v['n_neg']:,} | "
                  f"{v['auprc_max']:.4f} | {v['auroc_max']:.4f} | "
                  f"{v['auprc_mean']:.4f} | "
                  f"[{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}] |")

    md.append("\n## 3. Species-level groups (first 15 by ascending AUPRC)\n")
    md.append(f"(groupwithin n ≥ {ARGS.min_species_n}; completetablesee json)\n")
    md.append("| species | n | AUPRC(max) | AUROC(max) |")
    md.append("|---|---|---|---|")
    for g, v in sorted(sp_m.items(), key=lambda kv: kv[1]["auprc_max"])[:15]:
        md.append(f"| {g} | {v['n']:,} | {v['auprc_max']:.4f} | "
                  f"{v['auroc_max']:.4f} |")

    md.append("\n## 4. Genome level (per-GCF) AUPRC distribution — OOD proxy\n")
    md.append(f"(eachgenomegroupwithin n ≥ {ARGS.min_genome_n}; n_genomes={gn_stats['n_genomes']})\n")
    md.append(f"- mean={gn_stats['mean']:.4f} · median={gn_stats['median']:.4f} · "
              f"IQR=[{gn_stats['p25']:.4f}, {gn_stats['p75']:.4f}] · "
              f"p10={gn_stats['p10']:.4f} · p90={gn_stats['p90']:.4f} · "
              f"min={gn_stats['min']:.4f}\n"
              f"- **AUPRC<0.5 ofgenomefraction: {gn_stats['frac_below_0.5']:.1%}** "
              "(OOD failure groups below the 0.5 flip line)\n")
    md.append("\nLargest 15 differences (OOD candidates):\n")
    for g, a in payload["genome_worst15"].items():
        org = gmeta.get(g, {}).get("organism", "")
        md.append(f"- `{g}` ({org}): AUPRC={a:.4f}\n")

    md.append("\n## 5. limitations (reported as-is)\n")
    md.append("- phylum-level classification = a genus→phylum static-map approximation of the header organism "
              "(PHYLUM_MAP + curated extension), not authoritative NCBI/GTDB metadata; "
              "unmapped genera go into a separate Unknown group.\n"
              "- multi-source-host contigs (multi_source_host) are grouped by the first HOST row, "
              "introducing a small amount of cross-group contamination.\n"
              "- within-group AUPRC is affected by the within-group positive/negative ratio (prevalence ≠ the global 0.51), "
              "cross-group comparison should consider n together with the CI.\n")

    out_md = out_dir / "OOD_CROSSTAXA_M0.md"
    with open(out_md, "w") as f:
        f.write("".join(md))
    print(f"[Report] saved {out_md}")
    print(f"[Overall] AUPRC(max)={om.get('auprc', float('nan')):.4f} "
          f"AUROC(max)={om.get('auc', float('nan')):.4f} "
          f"(ref 0.827) | phylum groups={len(phyl_m)} "
          f"genomes={gn_stats['n_genomes']} "
          f"P(AUPRC<0.5)={gn_stats['frac_below_0.5']:.1%}")
    print(f"infer={infer_s/60:.1f}min total={(time.time()-t_start)/60:.1f}min")


if __name__ == "__main__":
    main()
