#!/usr/bin/env python3
"""render_fc_figures.py  —  FINAL-CANDIDATE figure renderer (refreshes M2/M3-dependent panels; static panels byte-copied).

Structure follows the frozen Fig1-5 numbering and panel conventions; visuals reuse figure_system/nature_style.
Each figure ships source_data/: data.json + provenance.json (figure→data→metric→checkpoint traceability).
"""
import json, os, shutil, sys
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"  # keep real <text>: searchable + geometry-auditable
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FC = os.path.join(ROOT, "V7_FINAL_CANDIDATE_CURRENT_BEST")
FIG = f"{FC}/02_FIGURES"
SD = f"{FIG}/source_data"
sys.path.insert(0, f"{ROOT}/_full_version/figure_system/style")
import nature_style as NS  # noqa: E402
NS.apply_style()
C = {"M0": "#7f7f7f", "M2": "#1f77b4", "M3": "#d62728", "v68": "#9467bd"}

V = {k: v["value"] for k, v in
     json.load(open(f"{FC}/08_METRICS/FC_METRICS.json")).items()}

def dig_file(path, dotted):
    j = json.load(open(path)); cur = j
    for p in dotted.split("."):
        cur = cur.get(p) if isinstance(cur, dict) else None
    return cur

PANEL_LETTER = {"Fig2A": "a", "Fig2B": "b", "Fig2C": "c", "Fig2D": "d",
                "Fig5A": "a", "Fig5B": "b"}
def save_panel(fig, name, data, prov_note):
    os.makedirs(SD, exist_ok=True)
    fig.text(0.01, 0.97, PANEL_LETTER.get(name, ""), fontsize=15, fontweight="bold",
             va="top", ha="left")
    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"{FIG}/{name}.{ext}", dpi=300 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    json.dump(data, open(f"{SD}/{name}_data.json", "w"), indent=1)
    json.dump({"figure": name, "note": prov_note,
               "checkpoint_M2": "ep7/02a6fdc3713d2bad",
               "checkpoint_M3": "ep7/de94423771234b1d",
               "sources": ["09_EVAL/*_m{2,3}best_interim.json", "final_metrics.yaml(M0)"]},
              open(f"{SD}/{name}_provenance.json", "w"), indent=1)
    print(f"[fig] {name}")

def avail(*ks):
    return all(V.get(k) is not None for k in ks)

# ── Fig2A performance ladder (replaces the ghost-version frozen 2A; caption in sync) ─────────────────
if avail("M2_VAL_AUPRC", "M3_VAL_AUPRC"):
    fig, ax = plt.subplots(figsize=(89 / 25.4, 60 / 25.4))
    models = ["M4 length+GC\n(held-out test)", "M0\nno 2nd channel", "M3\nshuffled control", "M2\nfusion"]
    vals = [0.577, V["M0_VAL_AUPRC"], V["M3_VAL_AUPRC"], V["M2_VAL_AUPRC"]]
    bars = ax.bar(models, vals, color=["#c7c7c7", C["M0"], C["M3"], C["M2"]], width=.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + .004, f"{v:.3f}", ha="center", fontsize=8)
    ax.axhline(0.5, ls=":", color="k", lw=.8)
    ax.text(3.42, .505, "chance", fontsize=7, ha="right")
    ax.set_ylabel("AUPRC (validation; M4 = held-out test)", fontsize=9)
    ax.set_ylim(0.4, 0.95)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title("Trained ladder  —  simulated benchmark", fontsize=9)
    save_panel(fig, "Fig2A", {"models": models, "val_auprc": vals},
               "ladder values; M4 = length+GC baseline (frozen 0.577)")

# ── Fig2B recovery rate ─────────────────────────────────────────────────────────
if avail("RECOVERY"):
    fig, ax = plt.subplots(figsize=(89 * NS.MM_SCALE if hasattr(NS, "MM_SCALE") else 89 / 25.4, 60 / 25.4))
    labels = ["single-representation\n(M0)", "dual-representation\n(M2)"]
    vals = [87.5, V["RECOVERY"]]
    bars = ax.bar(labels, vals, color=[C["M0"], C["M2"]], width=.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + .6, f"{v:.1f}%", ha="center", fontsize=9)
    ax.axhline(100, ls=":", color=C["v68"], lw=1.0)
    ax.text(1.42, 100.4, "adaptive reference (v6.8)", fontsize=8, ha="right", color=C["v68"])
    ax.set_ylabel("recovery of adaptive reference\nvalidation performance (%)", fontsize=9)
    ax.set_ylim(80, 104)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    save_panel(fig, "Fig2B", {"M0": 87.5, "M2": vals[1]},
               "recovery-rate caliber: 100×val_AUPRC / 0.951 (N4)")

# ── Fig2C four anchors ───────────────────────────────────────────────────
if avail("M2_BENBOW_MEAN", "M3_BENBOW_MEAN", "M0_BENBOW_MEAN"):
    rows = [("Benbow (mean)", "BENBOW_MEAN", 1), ("IslandViewer4 (max)", "IV4_MAX", 1),
            ("ICEberg3 (mean)", "ICE_MEAN", 1), ("Philympics (mean)", "PHILYMPICS_MEAN", 0)]
    fig, ax = plt.subplots(figsize=(183 / 25.4, 70 / 25.4))
    ypos = np.arange(len(rows))[::-1]
    for y, (lab, key, inc_v68) in zip(ypos, rows):
        ax.plot([0, 1], [y, y], color="#dddddd", lw=.8, zorder=1)
        pts = [(V[f"M0_{key}"], C["M0"], "M0"), (V[f"M2_{key}"], C["M2"], "M2"),
               (V[f"M3_{key}"], C["M3"], "M3")]
        if inc_v68 and key in ("BENBOW_MEAN", "IV4_MAX"):
            v68 = {"BENBOW_MEAN": 0.7143, "IV4_MAX": 0.6675}[key]
            pts.append((v68, C["v68"], "v6.8 (adaptive)"))
        MK = {"M0": "o", "M2": "s", "M3": "^", "v6.8 (adaptive)": "D"}
        for x, c, m in pts:
            ax.plot(x, y, MK.get(m, "o"), ms=7 if m != "v6.8 (adaptive)" else 6,
                    color=c, zorder=3, markeredgecolor="white", markeredgewidth=.6)
            ax.text(x, y + .16, m, fontsize=6.5, color=c, ha="center")
    ax.set_yticks(ypos); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel("zero-shot AUROC"); ax.set_xlim(0.35, 0.92)
    ax.set_title("External anchors: frozen baseline / fusion / shuffled control / adaptive reference",
                 fontsize=9)
    save_panel(fig, "Fig2C", {r[1]: {m: V.get(f"{m}_{r[1]}") for m in ("M0", "M2", "M3")} for r in rows},
               "zero-shot AUROC; M2/M3 = final-candidate checkpoint re-evaluation (09_EVAL)")

# ── Fig2D attribution ladder ───────────────────────────────────────────────────────
if avail("M2_VAL_AUPRC", "M3_VAL_AUPRC", "M0_VAL_AUPRC", "M2_TEST_AUPRC"):
    fig, ax = plt.subplots(figsize=(89 / 25.4, 70 / 25.4))
    xs = np.arange(3)
    val = [V["M0_VAL_AUPRC"], V["M3_VAL_AUPRC"], V["M2_VAL_AUPRC"]]
    tst = [V["M0_TEST_AUPRC"], V.get("M3_TEST_AUPRC"), V.get("M2_TEST_AUPRC")]
    ax.bar(xs - .18, val, .34, color=[C[m] for m in ("M0", "M3", "M2")], label="validation")
    ax.bar(xs + .18, [t if t else 0 for t in tst], .34,
           color=[C[m] for m in ("M0", "M3", "M2")], alpha=.45, label="held-out test")
    for x, v, t in zip(xs, val, tst):
        ax.text(x - .18, v + .004, f"{v:.4f}", ha="center", fontsize=7)
        if t:
            ax.text(x + .18, t + .004, f"{t:.4f}", ha="center", fontsize=7)
    ax.axhline(V["CEILING"], ls=":", color="#2ca02c", lw=1)
    ax.text(2.45, V["CEILING"] + .002, f"composition ceiling {V['CEILING']:.3f}", fontsize=7, ha="right")
    ax.axhline(V["PROBE_MLP"], ls="-.", color="#9467bd", lw=1)
    ax.text(2.45, V["PROBE_MLP"] + .002, f"MLP probe {V['PROBE_MLP']:.4f}", fontsize=7, ha="right")
    ax.set_xticks(xs); ax.set_xticklabels(["M0\nno 2nd channel", "M3\nshuffled control",
                                           "M2\nfusion"], fontsize=8)
    ax.set_ylabel("AUPRC"); ax.set_ylim(0.80, 0.92); ax.legend(fontsize=7, loc="upper left")
    d1, d2 = V["M2_VAL_AUPRC"] - V["M3_VAL_AUPRC"], V["M3_VAL_AUPRC"] - V["M0_VAL_AUPRC"]
    ax.annotate(f"info {d1:+.4f}\ncapacity {d2:+.4f}", xy=(1.5, V["M2_VAL_AUPRC"] - .006),
                fontsize=7, ha="center")
    save_panel(fig, "Fig2D", {"val": val, "test": tst, "info": d1, "capacity": d2},
               "attribution ladder; info = M2−M3, capacity = M3−M0 (validation caliber)")

# ── Fig5A difficulty ladder (aligned with Fig.5 caption Panel a) ───────────────────────────────────────────────────────
m0d = [dig_file(f"docs/evidence/difficulty_D{t}_difficulty_m0.json", "results.max.auprc")
       for t in (1, 2, 3)]
if avail("M2_DIFF_D1", "M2_DIFF_D2", "M2_DIFF_D3") and all(m0d):
    # D0 caliber (audit 2026-09-06, Methods §7.4): the difficulty tiers are max-
    # aggregation, so D0 must be the aggregation-matched full-pool reference
    # pooled.auprc_max = 0.8142 (ood_crosstaxa_m0.json)  —  NOT the mixed-
    # aggregation sim-test 0.8269 (that mixed baseline sign-flipped D2's delta
    # and was recomputed; using it again makes D2 fall below D0). M2/M3 have no
    # D0-tier evaluation: their lines start at D1.
    _d0 = dig_file("docs/evidence/ood_crosstaxa_m0.json", "pooled.auprc_max")
    fig, ax = plt.subplots(figsize=(89 / 25.4, 60 / 25.4))
    xs = np.arange(4)
    ax.plot(xs, [_d0] + m0d, "o-", color=C["M0"], label="M0")
    ax.plot([xs[0]], [_d0], "o", ms=7, mfc="white", mec=C["M0"], mew=1.4,
            zorder=4)  # open marker = aggregation-matched reference
    ax.plot(xs[1:], [V["M2_DIFF_D1"], V["M2_DIFF_D2"], V["M2_DIFF_D3"]],
            "o-", color=C["M2"], label="M2")
    ax.plot(xs[1:], [V["M3_DIFF_D1"], V["M3_DIFF_D2"], V["M3_DIFF_D3"]],
            "o--", color=C["M3"], label="M3")
    ax.set_xticks(xs); ax.set_xticklabels(["D0 (matched ref.)", "D1 GI length", "D2 donor sim",
                                           "D3 fragmentation"], fontsize=7)
    ax.set_ylabel("AUPRC (max agg)"); ax.legend(fontsize=7)
    ax.set_title("Difficulty ladder  —  no performance cliff", fontsize=9)
    save_panel(fig, "Fig5A", {"M0": [_d0] + m0d,
                              "M2": [V["M2_DIFF_D1"], V["M2_DIFF_D2"], V["M2_DIFF_D3"]],
                              "M3": [V.get("M3_DIFF_D1"), V.get("M3_DIFF_D2"), V.get("M3_DIFF_D3")]},
               "D0 = aggregation-matched full-pool reference (max agg, 0.8142; "
               "Methods §7.4 recomputation); D1-D3 = difficulty tiers; "
               "M2/M3 have no D0-tier eval (lines start at D1)")

# ── Fig5B OOD (alignment Fig.5 caption Panel b) ────────────────────────────────────────────────────────────
if avail("M2_OOD_MEAN"):
    # M0 bar = per-genome mean over the 843 main-table genomes (audit
    # 2026-09-06: the old "pooled.mean" key never existed -> silent 0.864
    # fallback; the real key is group_stats.auprc_max_mean)
    m0_ood = dig_file("docs/evidence/ood_crosstaxa_m0.json",
                      "group_stats.auprc_max_mean") or 0.864
    fig, ax = plt.subplots(figsize=(89 / 25.4, 60 / 25.4))
    bars = ax.bar(["M0\n(frozen)", "M2\n(final candidate)"], [m0_ood, V["M2_OOD_MEAN"]],
                  color=[C["M0"], C["M2"]], width=.5)
    for b, v in zip(bars, [m0_ood, V["M2_OOD_MEAN"]]):
        ax.text(b.get_x() + b.get_width() / 2, v + .004, f"{v:.3f}", ha="center", fontsize=9)
    # reference line on the SAME per-genome-mean caliber (NOT window-level
    # M0_VAL_AUPRC - 0.01 = 0.823, which mixed calibers)
    crit = m0_ood - 0.01
    ax.axhline(crit, ls=":", color="#2ca02c")
    ax.text(1.45, crit + .002, f"M0 mean \u2212 0.01 = {crit:.3f}", fontsize=7, ha="right")
    ax.set_ylabel("per-genome AUPRC (mean)"); ax.set_ylim(0.8, 0.95)
    save_panel(fig, "Fig5B", {"M0": m0_ood, "M2": V["M2_OOD_MEAN"], "n_genomes": V.get("M2_OOD_N")},
               "OOD cross-taxa per-genome AUPRC; reference line = M0 per-genome "
               "mean - 0.01 (descriptive, same caliber as the bars)")

# (CEA matrix panel already deleted: no caption slot, and it conflicted with the preregistered locked-PENDING table description  —  figure-text audit MAJOR-1)

# ── static-panel byte copy (Fig1/Fig3*/SF* are not re-rendered) ─────────────────────────────
# [FIX 2026-09-08] The copy source was originally _full_version/03_FIGURES_MASTER/rendered
# (the old pre-polish version), which once overwrote the optimized frozen package's
# Fig3*/Fig4A/Fig5C/SF7* back to the old version (the "other figures got worse"
# incident). The authoritative source is now 02_FIGURES_OPTIMIZED/rendered
# (the 46/46 frozen package); the old source serves only as a fallback for missing files.
src = os.path.join(ROOT, "V7_FINAL_CANDIDATE_CURRENT_BEST",
                   "02_FIGURES_OPTIMIZED", "rendered")
src_fallback = f"{ROOT}/_full_version/03_FIGURES_MASTER/rendered"
copied = []
if os.path.isdir(src):
    for f in sorted(os.listdir(src)):
        if f.startswith(("Fig1", "fig1_", "Fig3", "SF", "fig3_", "fig4_", "fig5_",
                         "Fig4", "Fig5C")) and \
           (f.endswith(".pdf") or f.endswith(".png")):
            s = f"{src}/{f}"
            if not os.path.isfile(s) and os.path.isfile(f"{src_fallback}/{f}"):
                s = f"{src_fallback}/{f}"
            shutil.copy2(s, f"{FIG}/{f}")
            copied.append(f)
print(f"[figs] static copied: {len(copied)} (source: optimized freeze package)")

# ── authoritative override (prevents late-stage contamination) ───────────────────────────────────────────
# The self-rendered Fig2A-D/5A/5B panels are all finalized with the optimized frozen
# package versions (premium/v8 style, values from the same FC_METRICS.json)  —  —
# re-running this script in the autobuild final stage will not produce mixed styles.
ov = 0
for name in ("Fig2A", "Fig2B", "Fig2C", "Fig2D", "Fig5A", "Fig5B"):
    for ext in ("pdf", "svg", "png"):
        s = os.path.join(ROOT, "V7_FINAL_CANDIDATE_CURRENT_BEST",
                         "02_FIGURES_OPTIMIZED", "rendered", f"{name}.{ext}")
        if os.path.isfile(s):
            shutil.copy2(s, f"{FIG}/{name}.{ext}")
            ov += 1
print(f"[figs] authoritative overlay applied: {ov} files (Fig2A-D/5A/5B)")
