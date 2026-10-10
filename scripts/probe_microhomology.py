#!/usr/bin/env python3
"""Frozen-Model Probe: R2-C3 microhomology artifact dependency quantification (retraining forbidden).

Causal perturbation experiments on the current Contig E2 checkpoint over simulated test data
(analogous to the audit_02 GC ablation): if the model relies on the microhomology shortcut
"virus first/last 3bp == bg prefix", randomizing those positions should sharply reduce AUC;
the control group (3bp randomized in the middle of the virus segment) isolates "generic sequence sensitivity".

Design:
  positives: 3kb window taken at the virus insertion point ±1.5kb (contains the microhomology artifact)
  three conditions (only positives perturbed; negatives untouched):
    baseline  : original window
    treat     : virus first 3bp + last 3bp (microhomology positions) randomized → breaks the artifact match
    control   : 3bp in the middle of the virus segment randomized → equal perturbation but not at artifact positions
  AUC_base / AUC_treat / AUC_ctrl; Δ_treat=AUC_base-AUC_treat, Δ_ctrl=...
  Dependency: Δ_treat >> Δ_ctrl → HIGH; Δ_treat ≈ Δ_ctrl → generic sensitivity (not artifact-specific) → LOW/NONE

usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/probe_microhomology.py \
      --ckpt checkpoints/Contig_500k/best_model.pt \
      --config configs/contig_500k.yaml \
      --data-dir simulated_data/Contig_500k/test \
      --n 150 --out audit_output/13_probe_microhomology.json
"""
from __future__ import annotations
import os, sys, json, argparse, random
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # A100 (reversemapping)
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from DeepFluHGT import load_config
from model.contracts import HGTConfig
from model.api import build_model
from data_pipeline.impl.dataset import SlidingWindowContigDataset
from data_pipeline.impl.collator import HGTCollator
from data_pipeline.contracts import HGTSample

WIN = 3000


def read_fasta(p):
    d, sid, buf = {}, None, []
    for ln in open(p):
        ln = ln.strip()
        if ln.startswith(">"):
            if sid: d[sid] = "".join(buf)
            sid, buf = ln[1:].split()[0], []
        else: buf.append(ln.upper())
    if sid: d[sid] = "".join(buf)
    return d


def read_virus_coords(prov):
    """{pos_sid: (read_start, read_end)} taking the VIRUS segment (the first one)."""
    coords = {}
    rows = [r.split("\t") for r in open(prov).read().splitlines()[1:] if r.strip()]
    for r in rows:
        if len(r) > 14 and r[6] == "VIRUS":  # seg_type=idx6
            sid = r[0]
            if sid not in coords:
                try: coords[sid] = (int(r[13]), int(r[14]))  # read_start=idx13, read_end=idx14
                except ValueError: pass
    return coords


def perturb(seq, positions, rng):
    """Replace positions (index list) with random bases (different from the original)."""
    bases = list(seq)
    comp = {"A":"CGT","C":"AGT","G":"ACT","T":"ACG","N":"ACGT"}
    for i in positions:
        if 0 <= i < len(bases):
            bases[i] = rng.choice(comp.get(bases[i], "ACGT"))
    return "".join(bases)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", required=True); ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config); mc, dc = cfg["model"], cfg["data"]; sw = dc["sw_window_size"]
    model = build_model(HGTConfig(data_mode="contig", max_contig_length=sw,
        hidden_dim=mc["hidden_dim"], local_hidden_dim=mc["local_hidden_dim"],
        num_attention_heads=mc["num_attention_heads"], dropout=mc["dropout"],
        graph_top_k=mc["graph_top_k"], bio_feature_dim=mc["bio_feature_dim"],
        pretrained_nt_path=mc["pretrained_nt_path"],
        pretrained_caduceus_path=mc["pretrained_caduceus_path"])).to("cuda")
    state = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    model.load_state_dict(state["model"], strict=False); model.eval()
    print(f"[model] epoch={state.get('epoch')}", flush=True)

    seqs = read_fasta(Path(args.data_dir) / "dataset.fasta")
    vcoords = read_virus_coords(Path(args.data_dir) / "provenance.tsv")
    rng = random.Random(42)
    pos_ids = [s for s in seqs if s.startswith("pos_") and s in vcoords and len(seqs[s]) >= WIN]
    neg_ids = [s for s in seqs if s.startswith("neg_") and len(seqs[s]) >= WIN]
    rng.shuffle(pos_ids); rng.shuffle(neg_ids)
    pos_ids, neg_ids = pos_ids[:args.n], neg_ids[:args.n]
    print(f"[data] {len(pos_ids)} pos (with virus coordinates) + {len(neg_ids)} neg", flush=True)

    def window_3kb(seq, center):
        half = WIN // 2
        s = max(0, min(center - half, len(seq) - WIN))
        return seq[s:s + WIN], s  # window, global_start

    def make_samples(versions):  # versions: {sid: {cond: seq}}
        samples = []
        sids = []
        for sid in pos_ids:
            for cond in ["base", "treat", "ctrl"]:
                seq = versions[sid][cond]
                samples.append(HGTSample(seq_id=f"{sid}#{cond}", label=1, sequence=seq,
                    r2_sequence=None, events=[], sample_type="POS", seq_len=len(seq)))
                sids.append((sid, cond, 1))
        for sid in neg_ids:
            seq = versions[sid]["base"]
            samples.append(HGTSample(seq_id=f"{sid}#base", label=0, sequence=seq,
                r2_sequence=None, events=[], sample_type="NEG", seq_len=len(seq)))
            sids.append((sid, "base", 0))
        return samples, sids

    # build the three condition windows
    versions = {}
    for sid in pos_ids:
        seq = seqs[sid]; vs, ve = vcoords[sid]
        win, gstart = window_3kb(seq, vs)
        vl = vs - gstart  # virus local start
        ve_l = ve - gstart
        mh_positions = list(range(vl, vl + 3)) + list(range(max(0, ve_l - 3), ve_l))  # first+last 3bp
        ctrl_center = vl + (ve_l - vl) // 2
        ctrl_positions = list(range(ctrl_center, ctrl_center + 3))
        versions[sid] = {
            "base": win,
            "treat": perturb(win, mh_positions, rng),
            "ctrl": perturb(win, ctrl_positions, rng),
        }
    for sid in neg_ids:
        win, _ = window_3kb(seqs[sid], rng.randint(WIN // 2, len(seqs[sid]) - WIN // 2))
        versions[sid] = {"base": win, "treat": win, "ctrl": win}

    samples, sids = make_samples(versions)
    ds = SlidingWindowContigDataset(samples=samples, window_size=sw, overlap=dc["sw_overlap"],
        is_training=False, max_neg_windows=9999, max_eval_windows=dc["sw_max_eval_windows"], extract_bio=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=HGTCollator(pad_id=0), num_workers=2)
    print(f"[infer] {len(ds)} windows", flush=True)

    probs = {}  # (sid,cond) -> prob
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.to("cuda")
            out = model(batch, early_exit=False)
            p = torch.sigmoid(out.cls_logits).item()
            sids_b = getattr(batch, "original_seq_ids", None) or batch.seq_ids
            key = sids_b[0]
            sid, cond = key.split("#")
            probs[(sid, cond)] = p
            if (i + 1) % 100 == 0:
                torch.cuda.empty_cache()
                print(f"  {i+1}/{len(ds)}", flush=True)

    # AUC per condition (pos under that condition vs neg)
    def auc(cond):
        y, s = [], []
        for sid in pos_ids: y.append(1); s.append(probs[(sid, cond)])
        for sid in neg_ids: y.append(0); s.append(probs[(sid, "base")])
        return roc_auc_score(y, s)
    a_base, a_treat, a_ctrl = auc("base"), auc("treat"), auc("ctrl")
    pos_mean = {c: np.mean([probs[(s, c)] for s in pos_ids]) for c in ["base", "treat", "ctrl"]}
    d_treat, d_ctrl = a_base - a_treat, a_base - a_ctrl

    # dependency score
    if d_treat < 0.005: dep = "NONE"
    elif d_treat - d_ctrl < 0.01: dep = "LOW"   # perturbation effect = control → not artifact-specific
    elif d_treat - d_ctrl < 0.03: dep = "MEDIUM"
    else: dep = "HIGH"

    res = {"model_epoch": state.get("epoch"), "n_pos": len(pos_ids), "n_neg": len(neg_ids),
           "auc_base": a_base, "auc_treat": a_treat, "auc_ctrl": a_ctrl,
           "delta_treat": d_treat, "delta_ctrl": d_ctrl, "delta_excess": d_treat - d_ctrl,
           "pos_mean_prob": pos_mean, "artifact_dependency": dep}
    print("\n" + "=" * 60)
    print("R2-C3 Microhomology Artifact Dependency Probe (Frozen Contig E2)")
    print("=" * 60)
    print(f"AUC base={a_base:.4f}  treat={a_treat:.4f}  ctrl={a_ctrl:.4f}")
    print(f"Δ_treat={d_treat:+.4f}  Δ_ctrl={d_ctrl:+.4f}  excess over control={d_treat-d_ctrl:+.4f}")
    print(f"pos mean prob: base={pos_mean['base']:.3f} treat={pos_mean['treat']:.3f} ctrl={pos_mean['ctrl']:.3f}")
    print(f">>> Artifact Dependency Score: {dep}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
