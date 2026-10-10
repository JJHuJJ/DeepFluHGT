#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M0 interpretability four-pack  —  ported to the API from 8.0 phase1_interpretability 01-04.

Tasks (paper supplementary figures, all CPU):
  (1) attention   attention heatmaps for 40 breakpoint-bearing POS windows: NTv3 frozen layer (output_attentions,
                conv tower 128x downsampling, 24 tokens/window) + adapter layers (splitter
                donor/recipient self-attn, cross_attn local->donor/recipient,
                full-token resolution 3002) + split_gates (donor gate per token).
  (2) tsne        t-SNE of 500 val windows' donor_pool/recipient_pool (HGTOutput field,
                splitter CLS pooling, L2-normalized), colored by pos/neg.
  (3) ig          per-base Integrated Gradients on 30 POS windows (adapter entry:
                interpolate at global_hidden, frozen NTv3 as a fixed feature tower; 32 steps;
                aligned with bp_heatmaps + breakpoint-neighborhood enrichment).
  (4) unc         uncertainty-artifact correlation: no evidential head (use_evidential=False)
                → cls_logit |margin| as the uncertainty proxy (u=-|logit|); correlate with
                GC extremeness / N content / window length (all 199,438 val windows, reusing
                the GPU inference probabilities from trainval_gap_epoch_2.npz, no re-inference).

Run discipline: CUDA_VISIBLE_DEVICES="" CPU-only; <=48 threads; do not touch files of running jobs.
protocol: window 3000/300/max_eval_windows=10 (same style as the real-anchor/difficulty-ladder evaluations),
      model checkpoints/Contig_500k_ntv3only/best_model.pt (ep2),
      load protocol verbatim-replicates scripts/difficulty_eval.py::load_model_gpu (device=cpu only).

usage:
  python scripts/interp_m0.py --part attn  [--n 40]
  python scripts/interp_m0.py --part tsne  [--n 500]
  python scripts/interp_m0.py --part ig    [--n 30 --steps 32]
  python scripts/interp_m0.py --part unc
Outputs: docs/evidence/interp/*.png + *.npz/*.json (verification data)
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")           # CPU-only, strictly forbidden GPU
os.environ.setdefault("OMP_NUM_THREADS", "44")
os.environ.setdefault("MKL_NUM_THREADS", "44")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "44")

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DEFAULT_CKPT = REPO / "checkpoints/Contig_500k_ntv3only/best_model.pt"
DEFAULT_CFG = REPO / "configs/contig_500k_ntv3only.yaml"
NT_FALLBACK = REPO / "model/InstaDeepAI/NTv3_650M_pre"
DATA_DIR = REPO / "simulated_data/Contig_500k"
VAL_NPZ = REPO / "docs/evidence/m0_trajectory/trainval_gap_epoch_2.npz"
OUT_DIR = REPO / "docs/evidence/interp"

WIN, OVERLAP, MAX_EVAL = 3000, 300, 10
SEED = 42
NT_PAD_MULT = 128          # the NTv3 conv tower requires input length to be a multiple of 128
NT_DOWNSAMPLE = 128        # the transformer runs at 1/128 resolution (num_downsamples=7)

# ── Palette (first slots that passed the dataviz check: all CVD-safe with deltaE>=9) ──
C_POS = "#2a78d6"      # blue   (slot1)  —  positive
C_NEG = "#eb6834"      # orange (slot2)  —  negative
C_HARD = "#1baf7a"     # aqua   (slot3)  —  auxiliary hard-neg marker
C_GATE = "#2a78d6"     # donor gate curve
C_IG = "#2a78d6"       # IG curve
C_BP = "#e34948"       # breakpoint vertical line (red, status-color semantics: marks event positions)
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b",
    "text.color": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--part", required=True,
                   choices=["attn", "tsne", "ig", "unc"])
    p.add_argument("--n", type=int, default=0, help="number of windows (0=default: attn40/tsne500/ig30)")
    p.add_argument("--steps", type=int, default=32, help="IG steps")
    p.add_argument("--threads", type=int, default=44)
    p.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    p.add_argument("--config", default=str(DEFAULT_CFG))
    p.add_argument("--smoke", action="store_true", help="quick self-check (few windows)")
    return p.parse_args()


# ══════════════════════ Shared: model / data ══════════════════════ #

def load_model_cpu(ckpt: Path, config: Path, threads: int):
    """Verbatim replication of difficulty_eval.load_model_gpu, device=cpu only."""
    import yaml
    from model.api import build_model
    from model.contracts import HGTConfig

    torch.set_num_threads(threads)
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
    assert bool(mc.get("use_evidential", False)) is False, \
        "this script assumes no evidential head (see the margin proxy in part unc)"
    model = build_model(cfg)

    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    ckpt_epoch = st.get("epoch")
    contract = st.get("contract", {})
    if ckpt_epoch != 2:
        alt = ckpt.parent / "epoch_2.pt"
        print(f"[WARN] {ckpt.name} epoch={ckpt_epoch} ≠ 2 → fallback {alt.name}")
        st = torch.load(alt, map_location="cpu", weights_only=False)
        ckpt_epoch = st.get("epoch")
        contract = st.get("contract", {})
    missing, unexpected = model.load_state_dict(st["model"], strict=False)
    model.eval()
    assert contract.get("encoder_frozen") is True, "contract: encoder_frozen must be True"
    info = {
        "ckpt": str(ckpt), "ckpt_epoch": ckpt_epoch,
        "encoder_frozen": contract.get("encoder_frozen"),
        "backbone_sha256": contract.get("backbone_sha256"),
        "state_dict_missing": len(missing),
        "state_dict_unexpected": len(unexpected),
        "device": "cpu", "threads": threads,
    }
    print(f"[Model] epoch={ckpt_epoch} missing={len(missing)} "
          f"unexpected={len(unexpected)} → cpu({threads} thr)", flush=True)
    return model, info


def load_val_dataset(extract_bio: bool = True):
    """val split → SlidingWindowContigDataset (deterministic eval-protocol windows)."""
    from data_pipeline.impl.parser import BEDParser
    from data_pipeline.impl.dataset import SlidingWindowContigDataset
    t0 = time.time()
    samples = BEDParser(
        bed_path=str(DATA_DIR / "val" / "breakpoints.bed"),
        fasta_dir=str(DATA_DIR / "val"), mode="contig").parse()
    ds = SlidingWindowContigDataset(
        samples=samples, window_size=WIN, overlap=OVERLAP,
        is_training=False, max_neg_windows=2, max_eval_windows=MAX_EVAL,
        extract_bio=extract_bio, evo_cache_path=None, rc_aug_prob=0.0)
    print(f"[Data] val {len(samples)} contigs → {len(ds)} windows "
          f"({time.time()-t0:.0f}s)", flush=True)
    return ds


def window_events(ds, idx: int):
    """Remap the events of window idx into the window's token space (CLS offset +1)."""
    si, s, e, _, _ = ds._window_index[idx]
    evs = []
    for ev in ds.samples[si].events:
        ns, ne = ev.hgt_start - s, ev.hgt_end - s
        if ne > 0 and ns < (e - s):
            evs.append({
                "start": max(0, ns), "end": min(ne, e - s),
                "virus_len": ev.virus_len, "hgt_mode": ev.hgt_mode,
                "tok_start": max(0, ns) + 1, "tok_end": min(ne, e - s) + 1,
            })
    return evs


def select_pos_windows(ds, n: int, margin: int = 200):
    """Deterministically pick n POS windows: breakpoints strictly contained (>=margin bp from the window edges), single-event preferred."""
    rng = np.random.default_rng(SEED)
    cand = []
    for idx, (si, s, e, lb, st) in enumerate(ds._window_index):
        if lb != 1:
            continue
        evs = window_events(ds, idx)
        if not evs:
            continue
        wl = e - s
        ok = all(ev["start"] >= margin and ev["end"] <= wl - margin for ev in evs)
        if not ok:
            continue
        # multi-event windows keep only events separated by >=300bp (avoids visual aliasing)
        evs_sorted = sorted(evs, key=lambda x: x["start"])
        sep_ok = all(b["start"] - a["end"] >= 300
                     for a, b in zip(evs_sorted, evs_sorted[1:]))
        cand.append({"idx": idx, "n_ev": len(evs), "evs": evs_sorted,
                     "sep_ok": sep_ok})
    clean = [c for c in cand if c["n_ev"] == 1 and c["sep_ok"]]
    multi = [c for c in cand if c["n_ev"] > 1 and c["sep_ok"]]
    print(f"[Select] breakpoint-bearing POS window candidates {len(cand)} "
          f"(single-event {len(clean)}, multi-event {len(multi)})")
    take = clean if len(clean) >= n else clean + multi
    order = rng.permutation(len(take))
    picked, used_contig = [], {}
    for j in order:                       # at most 2 windows per contig, for diversity
        c = take[j]
        si = ds._window_index[c["idx"]][0]
        if used_contig.get(si, 0) >= 2:
            continue
        used_contig[si] = used_contig.get(si, 0) + 1
        picked.append(c)
        if len(picked) == n:
            break
    print(f"[Select] picked {len(picked)} POS windows (margin>={margin}, seed={SEED})")
    return picked


def select_stratified_windows(ds, n: int):
    """For t-SNE: pos/neg stratified (neg proportional to type), at most 2 windows per contig."""
    rng = np.random.default_rng(SEED)
    pos_idx, neg_by_type = [], {}
    for idx, (si, s, e, lb, st) in enumerate(ds._window_index):
        if lb == 1:
            pos_idx.append(idx)
        else:
            neg_by_type.setdefault(st, []).append(idx)

    def cap2(idxs):
        out, used = [], {}
        for j in rng.permutation(len(idxs)):
            si = ds._window_index[idxs[j]][0]
            if used.get(si, 0) >= 2:
                continue
            used[si] = used.get(si, 0) + 1
            out.append(idxs[j])
        return out

    n_pos = n // 2
    pos = cap2(pos_idx)[:n_pos]
    n_neg = n - len(pos)
    # allocate neg types by their global val proportions
    total_neg = sum(len(v) for v in neg_by_type.values())
    neg = []
    quota = {k: int(round(n_neg * len(v) / total_neg)) for k, v in neg_by_type.items()}
    for k, q in sorted(quota.items(), key=lambda kv: -len(neg_by_type[kv[0]])):
        neg.extend(cap2(neg_by_type[k])[:q])
    rng.shuffle(neg)
    neg = neg[:n_neg]
    sel = sorted(pos + neg)
    types = [ds._window_index[i][4] for i in sel]
    labels = [ds._window_index[i][3] for i in sel]
    print(f"[Select] t-SNE windows {len(sel)} (pos={sum(labels)}, "
          f"neg type distribution={ {t: types.count(t) for t in sorted(set(types))} })")
    return sel


def make_loader(ds, indices, batch_size):
    from torch.utils.data import DataLoader
    from data_pipeline.impl.collator import HGTCollator
    from data_pipeline.impl.dataset import SlidingWindowContigDataset as _SW

    class _Subset(torch.utils.data.Dataset):
        def __init__(self, ds, indices):
            self.ds, self.indices = ds, indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            return self.ds[self.indices[i]]

    sub = _Subset(ds, indices)
    return DataLoader(sub, batch_size=batch_size, shuffle=False,
                      collate_fn=HGTCollator(pad_id=0), num_workers=0)


# ══════════════════════ ① attention ══════════════════════ #

ATT_MODULES = [
    ("splitter.donor_self_attn", "donor self-attn (splitter)"),
    ("splitter.recipient_self_attn", "recipient self-attn (splitter)"),
    ("cross_attn.local_a_to_donor", "cross-attn: local→donor"),
    ("cross_attn.local_a_to_recipient", "cross-attn: local→recipient"),
]


def _bin2d(x: torch.Tensor, n_bin: int) -> np.ndarray:
    """[B, H, L, L] → [B, H, n_bin, n_bin] (mean binning, adaptive pooling)."""
    import torch.nn.functional as TF
    B, H, L, _ = x.shape
    pooled = TF.adaptive_avg_pool2d(x.reshape(B * H, 1, L, L), (n_bin, n_bin))
    return pooled.reshape(B, H, n_bin, n_bin).numpy()


def forward_with_attention(model, batch):
    """Step-by-step replay of the 15-step forward pass (model/api.py), capturing per-head attention of each MHA.

    return dict: cls_logit, split_gates [B,L], bp_heatmap [B,L],
    key_profiles {module: [B,H,L]} (key-side mass after query-mean),
    qk_binned {module: [B,H,nb,nb]}, donor_pool/recipient_pool,
    ntv3_attn [B, layers, heads, Lt, Lt] (128x downsampling).
    """
    B, L = batch.input_ids.shape
    kpm = (batch.attention_mask == 0)

    # ── Step 1: backbone (NTv3, frozen) ──
    bo = model.backbone(batch.input_ids, batch.attention_mask)
    gh, gp = bo["global_hidden"], bo["global_pooled"]

    # ── NTv3 frozen-layer attention (invoked separately, output_attentions=True) ──
    ge = model.backbone.global_encoder
    ntv3_attn = None
    if not getattr(ge, "_use_fallback", True):
        mapped = ge._remap_ids(batch.input_ids)
        pad = (NT_PAD_MULT - L % NT_PAD_MULT) % NT_PAD_MULT
        if pad:
            mapped = torch.cat([mapped, torch.full((B, pad), 1, dtype=mapped.dtype)], 1)
            m_in = torch.cat([batch.attention_mask,
                              torch.zeros(B, pad, dtype=batch.attention_mask.dtype)], 1)
        else:
            m_in = batch.attention_mask
        with torch.no_grad():
            nt_out = ge._nt_model(input_ids=mapped, attention_mask=m_in,
                                  output_hidden_states=True, output_attentions=True)
        # tuple(layers) × [B, heads, Lt, Lt] @ 1/128 resolution
        ntv3_attn = torch.stack([a.float() for a in nt_out.attentions], dim=1)

    with torch.no_grad():
        # ── Step 3: splitter replay (per-head attention) ──
        gates = torch.sigmoid(model.splitter.gate_mlp(gh))          # [B,L,1]
        donor, recipient = gates * gh, (1.0 - gates) * gh
        d_out, d_w = model.splitter.donor_self_attn(
            donor, donor, donor, key_padding_mask=kpm,
            need_weights=True, average_attn_weights=False)
        donor_hidden = model.splitter.donor_norm(donor + d_out)
        r_out, r_w = model.splitter.recipient_self_attn(
            recipient, recipient, recipient, key_padding_mask=kpm,
            need_weights=True, average_attn_weights=False)
        recipient_hidden = model.splitter.recipient_norm(recipient + r_out)
        donor_pool = torch.nn.functional.normalize(donor_hidden[:, 0, :], dim=-1)
        recipient_pool = torch.nn.functional.normalize(recipient_hidden[:, 0, :], dim=-1)

        # ── Step 4/5: role + cross_attn replay ──
        role_logit = model.role_head.forward_pre_graph(gp)
        role_w = torch.sigmoid(role_logit)
        dv, dv_w = model.cross_attn.local_a_to_donor(
            gh, donor_hidden, donor_hidden, key_padding_mask=kpm,
            need_weights=True, average_attn_weights=False)
        rv, rv_w = model.cross_attn.local_a_to_recipient(
            gh, recipient_hidden, recipient_hidden, key_padding_mask=kpm,
            need_weights=True, average_attn_weights=False)
        fused_a = model.cross_attn.norm_a(
            role_w.unsqueeze(1) * dv + (1.0 - role_w.unsqueeze(1)) * rv + gh)

        # ── Steps 6-11: fpn → graph → cls ──
        fpn_out = model.fpn(fused_a)
        graph_pooled = model.graph(fpn_out[:, 0, :])
        cls_logit = model.cls_head(graph_pooled)["cls_logit"]

        # ── Step 14: bp heatmap ──
        bp_heat = model.bp_refiner(
            fpn_out, batch.attention_mask, batch.n_events)["bp_heatmaps"]

        weights = {
            "splitter.donor_self_attn": d_w,
            "splitter.recipient_self_attn": r_w,
            "cross_attn.local_a_to_donor": dv_w,
            "cross_attn.local_a_to_recipient": rv_w,
        }
        nb = 100
        key_profiles = {k: w.mean(dim=2).numpy() for k, w in weights.items()}  # [B,H,L]
        qk_binned = {k: _bin2d(w, nb) for k, w in weights.items()}
        del weights, d_w, r_w, dv_w, rv_w, d_out, r_out, dv, rv

    return {
        "cls_logit": cls_logit.numpy().ravel(),
        "split_gates": gates.squeeze(-1).numpy(),
        "bp_heatmap": bp_heat.numpy(),
        "key_profiles": key_profiles,
        "qk_binned": qk_binned,
        "donor_pool": donor_pool.numpy(),
        "recipient_pool": recipient_pool.numpy(),
        "ntv3_attn": (ntv3_attn.numpy() if ntv3_attn is not None else None),
    }


def run_attn(args):
    from scipy.stats import mannwhitneyu, wilcoxon

    def paired_p(x):
        """Paired enrichment test: one-sample Wilcoxon (log-ratio != 0 / delta != 0)."""
        try:
            return float(wilcoxon(x).pvalue)
        except ValueError:
            return float("nan")
    n = args.n or (4 if args.smoke else 40)
    model, info = load_model_cpu(Path(args.ckpt), Path(args.config), args.threads)
    ds = load_val_dataset(extract_bio=True)
    picked = select_pos_windows(ds, n)
    loader = make_loader(ds, [p["idx"] for p in picked], batch_size=1)

    rows, store = [], []
    t0 = time.time()
    for i, batch in enumerate(loader):
        with torch.no_grad():
            ref = model(batch, early_exit=False)
            ref_cls = ref.cls_logits.numpy().ravel()
        out = forward_with_attention(model, batch)
        replay_err = float(np.max(np.abs(out["cls_logit"] - ref_cls)))
        assert replay_err < 5e-3, f"replayed cls_logit deviation {replay_err:.2e} too large"

        evs = picked[i]["evs"]
        L = out["split_gates"].shape[1]
        seg = np.zeros(L, dtype=bool)          # HGT insertion segment (token space, including the CLS offset)
        bp_toks = []
        for ev in evs:
            seg[ev["tok_start"]:ev["tok_end"]] = True
            bp_toks += [ev["tok_start"], ev["tok_end"]]
        flk = np.zeros(L, dtype=bool)          # breakpoint ±100bp neighborhood
        for t in bp_toks:
            flk[max(0, t - 100):min(L, t + 100)] = True

        row = {"idx": picked[i]["idx"], "n_ev": len(evs),
               "replay_err": replay_err,
               "cls_logit": float(out["cls_logit"][0]),
               "virus_len": evs[0]["virus_len"], "hgt_mode": evs[0]["hgt_mode"],
               "bp_toks": bp_toks, "evs": evs}
        g = out["split_gates"][0]
        row["gate_in_seg"], row["gate_out_seg"] = float(g[seg].mean()), float(g[~seg].mean())
        for k, _ in ATT_MODULES:
            prof = out["key_profiles"][k][0].mean(0)             # head mean [L]
            row[f"{k}.flank_mean"] = float(prof[flk].mean())
            row[f"{k}.bg_mean"] = float(prof[~flk].mean())
        # NTv3 frozen layer: key-side mass upsampled to base resolution (repeat 128)
        if out["ntv3_attn"] is not None:
            nt = out["ntv3_attn"][0].mean(axis=(0, 1))            # [Lt, Lt]
            kp = torch.from_numpy(nt).mean(dim=0).repeat_interleave(NT_DOWNSAMPLE)[:L].numpy()
            kp = kp / max(kp.sum(), 1e-9) * L                     # normalize to mean 1
            row["ntv3.flank_mean"], row["ntv3.bg_mean"] = float(kp[flk].mean()), float(kp[~flk].mean())
            row["_ntv3_key_profile"] = kp
        # bp_heatmap alignment (Spearman)
        from scipy.stats import spearmanr
        bp_h = out["bp_heatmap"][0]
        row["spearman_bpheat_vs_flankmask"] = float(
            spearmanr(bp_h, flk.astype(float)).statistic)

        rows.append({k: v for k, v in row.items() if not k.startswith("_")})
        store.append({
            "idx": picked[i]["idx"], "cls_logit": out["cls_logit"][0],
            "split_gates": out["split_gates"][0], "bp_heatmap": bp_h,
            "key_profiles": {k: v[0] for k, v in out["key_profiles"].items()},
            "bp_toks": bp_toks, "evs": evs,
            "ntv3_attn": (out["ntv3_attn"][0] if out["ntv3_attn"] is not None else None),
            "ntv3_key_profile": row.get("_ntv3_key_profile"),
            "qk_binned": {k: v[0].mean(0) for k, v in out["qk_binned"].items()},
        })
        if (i + 1) % 5 == 0 or i == len(picked) - 1:
            el = time.time() - t0
            print(f"  [attn {i+1}/{len(picked)}] {el:.0f}s "
                  f"({el/(i+1):.1f}s/win)", flush=True)

    # ── statistics summary ──
    gate_in = np.array([r["gate_in_seg"] for r in rows])
    gate_out = np.array([r["gate_out_seg"] for r in rows])
    stats = {
        "n_windows": len(rows),
        "max_replay_cls_err": max(r["replay_err"] for r in rows),
        "gate_in_seg_mean": float(gate_in.mean()),
        "gate_out_seg_mean": float(gate_out.mean()),
        "gate_in_out_delta": float((gate_in - gate_out).mean()),
        "gate_in_gt_out_frac": float((gate_in > gate_out).mean()),
        "gate_paired_p": paired_p(gate_in - gate_out),
        "cls_logit_mean": float(np.mean([r["cls_logit"] for r in rows])),
    }
    for k, _ in ATT_MODULES:
        fl = np.array([r[f"{k}.flank_mean"] for r in rows]) + 1e-12
        bg = np.array([r[f"{k}.bg_mean"] for r in rows]) + 1e-12
        stats[k] = {
            "flank_vs_bg_ratio": float((fl / bg).mean()),
            "flank_gt_bg_frac": float((fl > bg).mean()),
            "p": paired_p(np.log(fl / bg)),
        }
    if "ntv3.flank_mean" in rows[0]:
        fl = np.array([r["ntv3.flank_mean"] for r in rows]) + 1e-12
        bg = np.array([r["ntv3.bg_mean"] for r in rows]) + 1e-12
        stats["ntv3_frozen"] = {
            "flank_vs_bg_ratio": float((fl / bg).mean()),
            "flank_gt_bg_frac": float((fl > bg).mean()),
            "p": paired_p(np.log(fl / bg)),
        }
    stats["bp_heatmap_spearman_flank_mean"] = float(
        np.mean([r["spearman_bpheat_vs_flankmask"] for r in rows]))

    # ── figure ──
    _plot_attention(store, stats)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / "attention_windows.npz",
        model_info=json.dumps(info), stats=json.dumps(stats),
        rows=json.dumps(rows),
        **{f"w{i}_gates": s["split_gates"] for i, s in enumerate(store)},
        **{f"w{i}_bpheat": s["bp_heatmap"] for i, s in enumerate(store)},
        **{f"w{i}_xattn_donor": s["key_profiles"]["cross_attn.local_a_to_donor"].mean(0)
           for i, s in enumerate(store)},
        **{f"w{i}_xattn_recip": s["key_profiles"]["cross_attn.local_a_to_recipient"].mean(0)
           for i, s in enumerate(store)},
        **({f"w{i}_ntv3": s["ntv3_key_profile"] for i, s in enumerate(store)
            if s["ntv3_key_profile"] is not None} or {}),
        **{f"w{i}_qk_{mod.split('.')[-1]}": s["qk_binned"][mod]
           for i, s in enumerate(store) for mod in s["qk_binned"]},
    )
    with open(OUT_DIR / "attention_stats.json", "w") as f:
        json.dump({"model": info, "stats": stats, "rows": rows}, f,
                  indent=1, ensure_ascii=False)
    print(f"[attn] done: {json.dumps({k: v for k, v in stats.items() if not isinstance(v, dict)})}")


def _smooth(x, k=15):
    if len(x) < k:
        return x
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="same")


def _plot_attention(store, stats):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_show = min(8, len(store))
    show = store[:n_show]

    # ── Panel A+B: 8 example windows × (gate curve | cross-attn key mass) ──
    fig, axes = plt.subplots(n_show, 2, figsize=(11, 1.55 * n_show),
                             sharex="col")
    if n_show == 1:
        axes = axes.reshape(1, 2)
    for i, s in enumerate(show):
        L = s["split_gates"].shape[0]
        xs = np.arange(L) - 1                # bp coordinates (minus CLS)
        ax = axes[i, 0]
        ax.plot(xs, _smooth(s["split_gates"], 15), color=C_GATE, lw=0.9)
        for ev in s["evs"]:
            ax.axvspan(ev["start"], ev["end"], color=C_POS, alpha=0.13, lw=0)
        for t in s["bp_toks"]:
            ax.axvline(t - 1, color=C_BP, lw=0.7, alpha=0.65)
        ax.set_xlim(0, L - 2)
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"w{i}\ngate", fontsize=7)
        if i == 0:
            ax.set_title("splitter donor gate σ(g_mlp)  "
                         "(blue band = HGT insert, red lines = breakpoints)",
                         fontsize=8)
        ax2 = axes[i, 1]
        kd = _smooth(s["key_profiles"]["cross_attn.local_a_to_donor"].mean(0), 15) * L
        kr = _smooth(s["key_profiles"]["cross_attn.local_a_to_recipient"].mean(0), 15) * L
        ax2.plot(xs, kd, color=C_POS, lw=0.9, label="→donor")
        ax2.plot(xs, kr, color=C_NEG, lw=0.9, label="→recipient")
        for ev in s["evs"]:
            ax2.axvspan(ev["start"], ev["end"], color=C_POS, alpha=0.13, lw=0)
        for t in s["bp_toks"]:
            ax2.axvline(t - 1, color=C_BP, lw=0.7, alpha=0.65)
        ax2.set_xlim(0, L - 2)
        if i == 0:
            ax2.set_title("cross-attention key-side mass (head mean, ×L renorm)",
                          fontsize=8)
            ax2.legend(fontsize=6, loc="upper right", frameon=False)
        if i == n_show - 1:
            ax.set_xlabel("window position (bp)")
            ax2.set_xlabel("window position (bp)")
    fig.suptitle(f"M0 (ep2) attention on breakpoint-bearing val windows  —  "
                 f"n={len(store)}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT_DIR / "attention_heatmaps.png", bbox_inches="tight")
    plt.close(fig)

    # ── Panel C: QK binned heatmaps (4 modules) for 2 example windows + NTv3 frozen layer ──
    n_h = min(2, len(store))
    fig, axes = plt.subplots(n_h, 5, figsize=(13.5, 2.9 * n_h))
    axes = np.atleast_2d(axes)
    for i in range(n_h):
        s = store[i]
        mods = list(s["qk_binned"].keys())
        for j, mod in enumerate(mods):
            ax = axes[i, j]
            M = s["qk_binned"][mod]
            # z-score boosts contrast: softmax attention is near-uniform (1/L), raw magnitudes show no visible structure
            Z = (M - M.mean()) / (M.std() + 1e-12)
            ax.imshow(np.clip(Z, 0, 4), origin="lower", cmap="Blues",
                      vmin=0, vmax=4)
            ax.set_title(mod.split(".")[-1], fontsize=7)
            ax.grid(False)
            if j == 0:
                ax.set_ylabel(f"w{i} (query)", fontsize=7)
            ax.set_xlabel("key", fontsize=6)
        ax = axes[i, 4]
        if s["ntv3_attn"] is not None:
            nt = s["ntv3_attn"].mean(axis=(0, 1))
            Z = (nt - nt.mean()) / (nt.std() + 1e-12)
            ax.imshow(np.clip(Z, 0, 4), origin="lower", cmap="Blues", vmin=0, vmax=4)
            ax.set_title(f"NTv3 frozen (12L×24H mean, {nt.shape[0]} tok = 1/128 res)",
                         fontsize=6.5)
            ax.grid(False)
            ax.set_xlabel("key (128bp/token)", fontsize=6)
    fig.suptitle("Query–key attention maps, adapter layers (100×100 binned) + frozen NTv3 layer",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_DIR / "attention_heatmaps_qk.png", bbox_inches="tight")
    plt.close(fig)

    # ── Panel D: summary bars (breakpoint neighborhood vs background enrichment) ──
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    labels, vals, errs = [], [], []
    gate_ratio = stats["gate_in_seg_mean"] / max(stats["gate_out_seg_mean"], 1e-9)
    labels.append("splitter donor gate\n(seg vs flank)")
    vals.append(gate_ratio); errs.append(0)
    for k, name in ATT_MODULES:
        labels.append(name.replace(" (splitter)", "").replace("cross-attn: ", ""))
        vals.append(stats[k]["flank_vs_bg_ratio"]); errs.append(0)
    if "ntv3_frozen" in stats:
        labels.append("NTv3 frozen attn\n(12L mean)")
        vals.append(stats["ntv3_frozen"]["flank_vs_bg_ratio"])
    ypos = np.arange(len(labels))[::-1]
    ax.barh(ypos, vals, height=0.62, color=C_POS, alpha=0.9)
    ax.axvline(1.0, color="#52514e", lw=1, ls="--")
    for y, v in zip(ypos, vals):
        ax.text(v + 0.02, y, f"{v:.2f}×", va="center", fontsize=8, color="#0b0b0b")
    ax.set_yticks(ypos, labels, fontsize=7)
    ax.set_xlabel("enrichment ratio (breakpoint ±100 bp  vs  window background)")
    ax.set_title(f"Attention / gate enrichment at breakpoints  —  n={stats['n_windows']} POS windows")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "attention_enrichment.png", bbox_inches="tight")
    plt.close(fig)


# ══════════════════════ ② t-SNE ══════════════════════ #

def run_tsne(args):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import roc_auc_score
    from scipy.stats import mannwhitneyu

    n = args.n or (24 if args.smoke else 500)
    model, info = load_model_cpu(Path(args.ckpt), Path(args.config), args.threads)
    ds = load_val_dataset(extract_bio=True)
    sel = select_stratified_windows(ds, n)
    loader = make_loader(ds, sel, batch_size=8)

    Dp, Rp, probs, labels, types, sids = [], [], [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            out = model(batch, early_exit=False)
            Dp.append(out.donor_pool.numpy())
            Rp.append(out.recipient_pool.numpy())
            probs.append(torch.sigmoid(out.cls_logits.squeeze(-1)).numpy())
            labels.extend(batch.labels.numpy().tolist())
            types.extend(batch.sample_types)
            sids.extend(batch.original_seq_ids or batch.seq_ids)
            if (i + 1) % 8 == 0 or i == len(loader) - 1:
                el = time.time() - t0
                print(f"  [tsne {((i+1)*8)}/{len(sel)}] {el:.0f}s "
                      f"({el/((i+1)*8):.2f}s/win)", flush=True)
    Dp, Rp = np.concatenate(Dp), np.concatenate(Rp)
    probs, labels = np.concatenate(probs), np.array(labels)
    types = np.array(types)

    def embed(X):
        Z = PCA(n_components=min(50, X.shape[0] - 1), random_state=SEED).fit_transform(X)
        perp = min(30.0, (X.shape[0] - 1) / 3.0)
        return TSNE(n_components=2, perplexity=perp, init="pca",
                    random_state=SEED).fit_transform(Z)

    print("[tsne] fitting t-SNE (donor / recipient)", flush=True)
    E_d, E_r = embed(Dp), embed(Rp)

    def sep_auc(X):
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
        return float(cross_val_score(
            LogisticRegression(max_iter=2000), X, labels, cv=cv,
            scoring="roc_auc").mean())

    stats = {
        "n": int(len(labels)), "n_pos": int(labels.sum()),
        "auc_donor_pool": sep_auc(Dp), "auc_recipient_pool": sep_auc(Rp),
        "auc_concat": sep_auc(np.concatenate([Dp, Rp], axis=1)),
        "model_auc_on_windows": float(roc_auc_score(labels, probs)),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / "tsne_donor_recipient.npz",
        donor_pool=Dp, recipient_pool=Rp, emb_donor=E_d, emb_recipient=E_r,
        labels=labels, probs=probs, types=types, sids=sids, sel_idx=np.array(sel),
        model_info=json.dumps(info), stats=json.dumps(stats))

    hard = np.isin(types, ["NEG_HARD"])
    neg_easy = (labels == 0) & ~hard
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    for ax, E, name in zip(axes, [E_d, E_r],
                           ["donor_pool (splitter CLS, L2-norm)",
                            "recipient_pool (splitter CLS, L2-norm)"]):
        ax.scatter(E[neg_easy, 0], E[neg_easy, 1], s=9, c=C_NEG, alpha=0.65,
                   linewidths=0, label="neg")
        ax.scatter(E[hard, 0], E[hard, 1], s=16, c=C_HARD, marker="^",
                   alpha=0.85, linewidths=0, label="neg (hard)")
        ax.scatter(E[labels == 1, 0], E[labels == 1, 1], s=9, c=C_POS,
                   alpha=0.65, linewidths=0, label="pos")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        ax.legend(fontsize=7, frameon=False, loc="best")
    fig.suptitle(f"M0 (ep2) donor/recipient pooled representations  —  "
                 f"{len(labels)} val windows (PCA50 → t-SNE, seed 42)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_DIR / "tsne_donor_recipient.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[tsne] stats: {json.dumps(stats)}")


# ══════════════════════ ③ Integrated Gradients ══════════════════════ #

def _adapter_forward(model, gh, role_logit, mask):
    """splitter→cross→fpn→graph→cls (the IG target path; replicates model/api.py Steps 3-11).

    Note: bio_encoder only feeds role_head(post_graph), not the cls path, so it is not part of the IG target.
    """
    split_out = model.splitter(gh, mask)
    donor_hidden, recipient_hidden = split_out["donor_hidden"], split_out["recipient_hidden"]
    cross_out = model.cross_attn(gh, donor_hidden, recipient_hidden, role_logit,
                                 attention_mask=mask)
    fpn_out = model.fpn(cross_out["fused_a"])
    graph_pooled = model.graph(fpn_out[:, 0, :])
    return model.cls_head(graph_pooled)["cls_logit"], fpn_out


def run_ig(args):
    from scipy.stats import spearmanr, wilcoxon
    n = args.n or (4 if args.smoke else 30)
    steps = 8 if args.smoke else args.steps
    model, info = load_model_cpu(Path(args.ckpt), Path(args.config), args.threads)
    ds = load_val_dataset(extract_bio=True)
    picked = select_pos_windows(ds, n)
    n = len(picked)
    loader = make_loader(ds, [p["idx"] for p in picked], batch_size=2)

    results = []
    t0 = time.time()
    for bi, batch in enumerate(loader):
        B, L = batch.input_ids.shape
        ids = batch.input_ids
        mask = batch.attention_mask
        with torch.no_grad():
            bo = model.backbone(ids, mask)
            gh0, gp0 = bo["global_hidden"], bo["global_pooled"]
            role_logit = model.role_head.forward_pre_graph(gp0)
            f_x, fpn_ref = _adapter_forward(model, gh0, role_logit, mask)
            f_x = f_x.numpy().ravel()
            bp_heat = model.bp_refiner(
                fpn_ref, mask, batch.n_events)["bp_heatmaps"].numpy()

        h = gh0.detach().clone().requires_grad_(True)
        baseline = torch.zeros_like(h)
        total_grad = torch.zeros_like(h)
        for step in range(steps):
            alpha = (step + 1) / steps
            x_in = baseline + alpha * (h - baseline)
            score, _ = _adapter_forward(model, x_in, role_logit, mask)
            score.sum().backward()
            total_grad += h.grad.detach().clone()
            h.grad = None
        ig = (total_grad / steps) * (h - baseline)          # [B, L, D]
        ig_tok = ig.detach().norm(dim=-1).numpy()           # [B, L] per-base attribution magnitude
        f0, _ = _adapter_forward(model, baseline, role_logit, mask)
        f0 = f0.detach().numpy().ravel()
        comp_err = float(np.mean(np.abs(
            ig.detach().sum(dim=(1, 2)).numpy() - (f_x - f0)) /
            (np.abs(f_x - f0) + 1e-6)))

        for b in range(B):
            j = bi * 2 + b
            if j >= n:
                break
            evs = picked[j]["evs"]
            Lw = mask[b].sum().item()
            prof = ig_tok[b, :Lw]
            bp_toks = []
            for ev in evs:
                bp_toks += [ev["tok_start"], min(ev["tok_end"], Lw - 1)]
            flk = np.zeros(Lw, dtype=bool)
            for t in bp_toks:
                flk[max(0, t - 100):min(Lw, t + 100)] = True
            seg = np.zeros(Lw, dtype=bool)
            for ev in evs:
                seg[ev["tok_start"]:min(ev["tok_end"], Lw)] = True
            results.append({
                "idx": picked[j]["idx"], "cls_logit": float(f_x[b]),
                "completeness_rel_err": comp_err,
                "ig_flank_mean": float(prof[flk].mean()),
                "ig_bg_mean": float(prof[~flk].mean()),
                "ig_in_seg_mean": float(prof[seg].mean()) if seg.any() else np.nan,
                "spearman_ig_vs_bpheat": float(spearmanr(prof, bp_heat[b, :Lw]).statistic),
                "virus_len": evs[0]["virus_len"], "n_ev": len(evs),
                "_prof": prof, "_bpheat": bp_heat[b, :Lw], "_bp_toks": bp_toks,
                "_evs": evs, "_L": Lw,
            })
        print(f"  [ig {min((bi+1)*2, n)}/{n}] {time.time()-t0:.0f}s "
              f"({(time.time()-t0)/((bi+1)*2):.1f}s/win, steps={steps})", flush=True)

    fl = np.array([r["ig_flank_mean"] for r in results]) + 1e-12
    bg = np.array([r["ig_bg_mean"] for r in results]) + 1e-12
    try:
        ig_p = float(wilcoxon(np.log(fl / bg)).pvalue)
    except ValueError:
        ig_p = float("nan")
    stats = {
        "n_windows": len(results), "ig_steps": steps,
        "completeness_rel_err_mean": float(np.mean(
            [r["completeness_rel_err"] for r in results])),
        "ig_flank_vs_bg_ratio_mean": float((fl / bg).mean()),
        "ig_flank_gt_bg_frac": float((fl > bg).mean()),
        "ig_flank_p": ig_p,
        "spearman_ig_vs_bpheat_mean": float(np.mean(
            [r["spearman_ig_vs_bpheat"] for r in results])),
    }

    # Breakpoint-aligned metagene curve (±500bp, 25bp bins, normalized within window)
    half, bw = 500, 25
    nb = 2 * half // bw
    curves = np.zeros((len(results), nb))
    for i, r in enumerate(results):
        prof = r["_prof"] / max(r["_prof"].mean(), 1e-9)
        for t in r["_bp_toks"]:
            lo = max(0, t - half); hi = min(r["_L"], t + half)
            seg_c = prof[lo:hi]
            pos = np.arange(lo, hi) - t
            c = np.zeros(nb)
            for p, v in zip(pos, seg_c):
                c[(p + half) // bw] += v
            cnt = np.zeros(nb)
            for p in pos:
                cnt[(p + half) // bw] += 1
            curves[i] += np.where(cnt > 0, c / np.maximum(cnt, 1), 0)
    curve_mean = curves.mean(0)
    stats["metagene_peak_to_bg"] = float(curve_mean.max() / max(curve_mean.min(), 1e-9))

    # ── figure ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_show = min(6, len(results))
    fig = plt.figure(figsize=(11.5, 1.45 * n_show + 3.0))
    gs = fig.add_gridspec(n_show + 2, 1, height_ratios=[1] * n_show + [1.4, 1.4])
    for i in range(n_show):
        r = results[i]
        ax = fig.add_subplot(gs[i, 0])
        xs = np.arange(r["_L"]) - 1
        ax.plot(xs, _smooth(r["_prof"], 21) / max(r["_prof"].mean(), 1e-9),
                color=C_IG, lw=0.8, label="IG (smoothed, ×window mean)")
        ax.plot(xs, _smooth(r["_bpheat"], 21) / max(np.abs(r["_bpheat"]).mean(), 1e-9),
                color=C_NEG, lw=0.8, alpha=0.75, label="bp_heatmap (model head)")
        for ev in r["_evs"]:
            ax.axvspan(ev["start"], min(ev["end"], r["_L"] - 2), color=C_POS,
                       alpha=0.12, lw=0)
        for t in r["_bp_toks"]:
            ax.axvline(t - 1, color=C_BP, lw=0.7, alpha=0.6)
        ax.set_xlim(0, r["_L"] - 2)
        ax.set_ylabel(f"w{i}", fontsize=7)
        if i == 0:
            ax.legend(fontsize=6, loc="upper right", frameon=False)
    ax = fig.add_subplot(gs[n_show, 0])
    xs_b = (np.arange(nb) - nb // 2) * bw
    ax.plot(xs_b, curve_mean, color=C_IG, lw=1.4)
    ax.fill_between(xs_b, curve_mean, curve_mean.min(), color=C_POS, alpha=0.12)
    ax.axvline(0, color=C_BP, lw=0.8, ls="--")
    ax.set_xlabel("distance to breakpoint (bp)")
    ax.set_ylabel("IG (×window mean)")
    ax.set_title(f"metagene: per-base IG around breakpoints  —  "
                 f"n={len(results)} windows × {stats['ig_flank_p']:.1e}", fontsize=8)
    ax = fig.add_subplot(gs[n_show + 1, 0])
    flr = np.array([r["ig_flank_mean"] for r in results])
    bgr = np.array([r["ig_bg_mean"] for r in results])
    ax.boxplot([bgr, flr], widths=0.5, patch_artist=True,
               boxprops=dict(facecolor="#cde2fb", edgecolor="#2a78d6"),
               medianprops=dict(color="#0d366b"))
    ax.set_xticks([1, 2], ["background", "breakpoint ±100bp"])
    ax.set_ylabel("IG magnitude")
    stats_text = (f"ratio {stats['ig_flank_vs_bg_ratio_mean']:.2f}× | "
                  f"win-share {stats['ig_flank_gt_bg_frac']*100:.0f}% | "
                  f"p={stats['ig_flank_p']:.1e}\n"
                  f"Spearman(IG, bp_heat)={stats['spearman_ig_vs_bpheat_mean']:.3f} | "
                  f"completeness err={stats['completeness_rel_err_mean']:.1%}")
    ax.set_title(stats_text, fontsize=7.5)
    fig.suptitle("Integrated Gradients (adapter entry: global_hidden → cls_logit) "
                 "on 30 breakpoint-bearing POS windows", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(OUT_DIR / "ig_attribution.png", bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(
        OUT_DIR / "ig_attribution.npz",
        model_info=json.dumps(info), stats=json.dumps(stats),
        rows=json.dumps([{k: v for k, v in r.items() if not k.startswith("_")}
                         for r in results]),
        metagene=curve_mean,
        **{f"w{i}_ig": r["_prof"] for i, r in enumerate(results)},
        **{f"w{i}_bpheat": r["_bpheat"] for i, r in enumerate(results)})
    with open(OUT_DIR / "ig_stats.json", "w") as f:
        json.dump({"model": info, "stats": stats,
                   "rows": [{k: v for k, v in r.items() if not k.startswith("_")}
                            for r in results]}, f, indent=1, ensure_ascii=False)
    print(f"[ig] stats: {json.dumps(stats)}")


# ══════════════════════ (4) uncertainty–artifact correlation ══════════════════════ #

def _win_stats(task):
    """joblib worker: (rows0, [(seq, s, e)]) → gc / n_frac / len."""
    rows0, specs = task
    gc = np.empty(len(specs)); nf = np.empty(len(specs)); ln = np.empty(len(specs))
    for j, (seq, s, e) in enumerate(specs):
        w = seq[s:e].upper()
        a, c, g, t = w.count("A"), w.count("C"), w.count("G"), w.count("T")
        acgt = a + c + g + t
        gc[j] = (g + c) / acgt if acgt else np.nan
        nf[j] = 1.0 - acgt / len(w) if len(w) else np.nan
        ln[j] = len(w)
    return rows0, gc, nf, ln


def run_unc(args):
    from scipy.stats import pearsonr, spearmanr, mannwhitneyu
    from sklearn.metrics import roc_auc_score
    from joblib import Parallel, delayed

    z = np.load(VAL_NPZ, allow_pickle=True)
    probs = z["val_probs"].astype(np.float64)
    labels = z["val_labels"].astype(int)
    ids = z["val_ids"]
    ds = load_val_dataset(extract_bio=False)
    n = len(ds)
    assert n == len(probs), f"window count mismatch: ds={n} vs npz={len(probs)}"
    # Order reconciliation (deterministic window reconstruction; residual_corr already asserts this on the full set, here spot-checked on a sample)
    chk = np.random.default_rng(SEED).choice(n, 200, replace=False)
    for i in chk:
        si, s, e, lb, _ = ds._window_index[i]
        assert ids[i] == ds.samples[si].seq_id and labels[i] == lb, f"window {i} reconciliation failure"

    t0 = time.time()
    B = 8000
    tasks = []
    for b0 in range(0, n, B):
        specs = [(ds.samples[ds._window_index[i][0]].sequence,
                  ds._window_index[i][1], ds._window_index[i][2])
                 for i in range(b0, min(b0 + B, n))]
        tasks.append((b0, specs))
    gc = np.empty(n); nf = np.empty(n); ln = np.empty(n)
    for rows0, g_, f_, l_ in Parallel(n_jobs=args.threads // 2, batch_size="auto")(
            delayed(_win_stats)(t) for t in tasks):
        gc[rows0:rows0 + len(g_)] = g_
        nf[rows0:rows0 + len(f_)] = f_
        ln[rows0:rows0 + len(l_)] = l_
    print(f"[unc] window features {n} done ({time.time()-t0:.0f}s)", flush=True)

    # group by contig (contig-level i.i.d. caliber, same as residual_corr)
    contig = np.array([ds._window_index[i][0] for i in range(n)])

    p = np.clip(probs, 1e-7, 1 - 1e-7)
    logit = np.log(p / (1 - p))
    margin = np.abs(logit)
    unc = -margin                                   # uncertainty proxy (larger = more uncertain)

    gc_ext = np.abs(gc - 0.5)                       # GC extremeness (distance from 50%)
    loglen = np.log10(np.maximum(ln, 1))
    feats = {"GC_extremeness(|GC-0.5|)": gc_ext, "N_fraction": nf,
             "log10_window_len": loglen, "GC_raw": gc}

    def corr_block(x, y):
        """Per-contig mean → Pearson/Spearman + permutation p."""
        rng = np.random.default_rng(SEED)
        order = np.argsort(contig, kind="stable")
        cx = np.array([x[order][s].mean() for s in _group_slices(contig[order])])
        cy = np.array([y[order][s].mean() for s in _group_slices(contig[order])])
        m = np.isfinite(cx) & np.isfinite(cy)
        cx, cy = cx[m], cy[m]
        pr = pearsonr(cx, cy).statistic
        sr = spearmanr(cx, cy).statistic
        perm = []
        for _ in range(2000):
            perm.append(pearsonr(rng.permutation(cx), cy).statistic)
        pval = (1 + np.sum(np.array(perm) >= abs(pr))) / 2001 if pr >= 0 else \
               (1 + np.sum(np.array(perm) <= -abs(pr))) / 2001
        return float(pr), float(sr), float(pval)

    rows = []
    for name, f in feats.items():
        pr, sr, pv = corr_block(unc, f)
        pr_w = pearsonr(unc[np.isfinite(f)], f[np.isfinite(f)]).statistic
        rows.append({"feature": name, "pearson_contig": pr, "spearman_contig": sr,
                     "perm_p_contig": pv, "pearson_window_desc": float(pr_w)})

    # uncertainty → prediction error (validity check of the uncertainty)
    err = ((probs >= 0.5).astype(int) != labels).astype(int)
    auc_err = float(roc_auc_score(err, unc))
    mw = float(mannwhitneyu(unc[err == 1], unc[err == 0], alternative="greater").pvalue)

    # Binning: |margin| deciles vs error rate (a calibration-view reliability curve)
    qs = np.quantile(margin, np.linspace(0, 1, 11))
    qs[0] -= 1e-9
    bin_id = np.digitize(margin, qs) - 1
    bin_id = np.clip(bin_id, 0, 9)
    err_rate = [float(err[bin_id == b].mean()) for b in range(10)]
    bin_acc = [float((labels[bin_id == b] == (probs[bin_id == b] >= 0.5)).mean())
               for b in range(10)]

    # Uncertainty control on the artifact-window subset (N>0 or extreme GC)
    artifact = (nf > 0) | (gc_ext > 0.15)
    stats = {
        "n_windows": int(n), "n_contigs": int(contig.max() + 1),
        "proxy": "u = -|logit(p)| (cls_logit margin; use_evidential=False)",
        "corr": rows,
        "auc_unc_predict_error": auc_err, "mw_p_unc_err": mw,
        "err_rate_by_margin_decile": err_rate,
        "acc_by_margin_decile": bin_acc,
        "mean_unc_artifact_windows": float(unc[artifact].mean()),
        "mean_unc_clean_windows": float(unc[~artifact].mean()),
        "mw_p_artifact_vs_clean": float(mannwhitneyu(
            unc[artifact], unc[~artifact], alternative="greater").pvalue),
        "frac_artifact_windows": float(artifact.mean()),
        "mean_margin": float(margin.mean()), "median_margin": float(np.median(margin)),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "uncertainty_corr.json", "w") as f:
        json.dump(stats, f, indent=1, ensure_ascii=False)
    np.savez_compressed(
        OUT_DIR / "uncertainty_windows.npz", probs=probs[:], labels=labels[:],
        gc=gc, n_frac=nf, len=ln, unc=unc, err=err, bin_id=bin_id,
        artifact=artifact, contig=contig)

    # ── figure (supplementary) ──
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    ax = axes[0]
    sub = np.random.default_rng(SEED).choice(n, 20000, replace=False)
    for lab, c, name in [(1, C_POS, "pos"), (0, C_NEG, "neg")]:
        m = labels[sub] == lab
        ax.scatter(gc_ext[sub][m], unc[sub][m], s=2.5, c=c, alpha=0.18,
                   linewidths=0, label=name)
    ax.set_xlabel("|GC − 0.5| (GC extremeness)")
    ax.set_ylabel("uncertainty −|logit p|")
    ax.set_title("uncertainty vs GC extremeness (20k subsample)", fontsize=8)
    ax.legend(fontsize=7, markerscale=3, frameon=False)
    ax = axes[1]
    sub2 = sub[nf[sub] > 0] if (nf[sub] > 0).any() else sub[:1]
    bins = np.linspace(0, max(0.05, np.quantile(nf, 0.999)), 20)
    ax.hist2d(nf[sub2], unc[sub2], bins=[bins, 30], cmap="Blues", cmin=1)
    ax.set_xlabel("N fraction"); ax.set_ylabel("uncertainty")
    ax.set_title(f"uncertainty vs N content (windows with N>0: "
                 f"{stats['frac_artifact_windows']*100:.1f}% artifact)", fontsize=8)
    ax.grid(False)
    ax = axes[2]
    xm = np.arange(1, 11)
    ax.plot(xm, bin_acc, color=C_POS, lw=1.6, marker="o", ms=4)
    ax.axhline(np.mean(labels == (probs >= 0.5)), color="#52514e", ls="--", lw=1,
               label=f"overall acc {np.mean(labels == (probs >= 0.5)):.3f}")
    ax.set_xlabel("|logit p| decile (confidence)")
    ax.set_ylabel("accuracy")
    ax.set_title(f"confidence ↔ accuracy (AUC(unc→error)={auc_err:.3f})", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "uncertainty_corr.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[unc] stats: {json.dumps({k: v for k, v in stats.items() if k != 'corr'})}")


def _group_slices(sorted_keys):
    """List of slices for runs of equal values (sorted)."""
    out, start = [], 0
    for i in range(1, len(sorted_keys) + 1):
        if i == len(sorted_keys) or sorted_keys[i] != sorted_keys[start]:
            out.append(slice(start, i))
            start = i
    return out


def main():
    args = parse_args()
    t0 = time.time()
    if args.part == "attn":
        run_attn(args)
    elif args.part == "tsne":
        run_tsne(args)
    elif args.part == "ig":
        run_ig(args)
    elif args.part == "unc":
        run_unc(args)
    print(f"[{args.part}] total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
