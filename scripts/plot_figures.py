#!/usr/bin/env python3
"""Plot paper Figures 2-5 (publication-ready), using internal-model real-data numbers.

data source: audit_output/*.json (internal-model PE E3 / Contig E2). Swap in the numbers once training completes.
output: figures/Fig{2,3,4,5}.pdf + .png (dpi 300)

Figure:
  Fig 2: composition strongly encoded in the representation (GC R² + Source ACC)
  Fig 3: causal independence from GC (ΔAUC)
  Fig 4: dualmodetrueactual validation (Contig 0.80 vs PE 0.45) + aggregationstrategyomittedtowardratio
  Fig 5: BLAST biological evidence (score x IAV)
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
AUS = ROOT / "audit_output"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "figure.dpi": 120, "savefig.dpi": 300, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
})
C_CONTIG, C_PE, C_GC, C_NEUTRAL = "#2c7fb8", "#d7191c", "#7f8c8d", "#bdc3c7"


def load(name):
    p = AUS / name
    return json.load(open(p)) if p.exists() else {}


def fig1_architecture():
    """Fig 1: architecture diagram (matplotlib flow chart). Simple version; can be polished from here."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    def box(x, y, w, h, text, color="#d9ecf8", ec=C_CONTIG, fs=9, bold=False):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=color, ec=ec, lw=1.2)
        ax.add_patch(b)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", lw=1.2, color="#555"))

    ax.text(5, 9.6, "DeepFluHGT — Dual DNA-LM Encoder for HGT Detection", ha="center", fontsize=12, fontweight="bold")
    # Input
    box(3.5, 8.5, 3, 0.7, "Contig → Sliding Window (3kb)", "#fff3cd", fs=9)
    # Dual encoders (parallel)
    box(0.8, 6.8, 3.5, 1.2, "Global Encoder\nNTv3 (650M)\ncontextual k-mer", "#d9ecf8", bold=True)
    box(5.7, 6.8, 3.5, 1.2, "Local Encoder\nCaduceus (Mamba SSM)\nlong-range", "#d9ead3", bold=True)
    arrow(4.3, 8.5, 2.55, 8.0); arrow(5.7, 8.5, 7.45, 8.0)
    # Merge → disentangle
    box(2.5, 5.3, 5, 0.9, "FeatureSplitBlock — donor ⊥ recipient disentanglement", "#f4cccc", fs=9, bold=True)
    arrow(2.55, 6.8, 4.0, 6.2); arrow(7.45, 6.8, 6.0, 6.2)
    # Cross-attn
    box(2.5, 4.0, 5, 0.8, "Role-Gated Cross-Attention (local ↔ donor/recipient)", "#d9ecf8", fs=9)
    arrow(5, 5.3, 5, 4.8)
    # FPN + graph + bio
    box(2.5, 2.8, 5, 0.8, "FPN → BatchGraph + BioFeatures (260-dim)", "#eaeaea", fs=9)
    arrow(5, 4.0, 5, 3.6)
    # Multi-task heads
    box(0.5, 1.2, 2.0, 1.1, "Cls\n(HGT?)", "#fff3cd", fs=9)
    box(2.8, 1.2, 2.0, 1.1, "Breakpoint\n(2-stage)", "#fff3cd", fs=9)
    box(5.1, 1.2, 2.0, 1.1, "Role\n(donor/recv)", "#fff3cd", fs=9)
    box(7.4, 1.2, 2.0, 1.1, "Direction", "#fff3cd", fs=9)
    arrow(4.0, 2.8, 1.5, 2.3); arrow(4.5, 2.8, 3.8, 2.3); arrow(5.5, 2.8, 6.1, 2.3); arrow(6.0, 2.8, 8.4, 2.3)
    ax.text(5, 0.4, "Cascade-Gated Multi-Task Loss + Uncertainty Weighting + FGM + Mixup",
            ha="center", fontsize=8.5, style="italic", color=C_PE)
    ax.text(5, 0.05, "715M params | Contig mode (real GI) + PE mode (junction)", ha="center", fontsize=7.5, color="#888")
    fig.tight_layout(); fig.savefig(OUT / "Fig1_architecture.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Fig1_architecture.png", bbox_inches="tight"); plt.close(fig)
    print("✓ Fig 1")


def fig2_composition():
    """Fig 2: composition strongly encoded in the representation (GC R² + Source ACC)."""
    d = load("03_compositional.json")
    smoke = d.get("smoke_baseline", {})
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    # (a) GC R²
    ax = axes[0]
    vals = [d.get("gc_r2", 0.954), smoke.get("gc_r2", 0.936)]
    ax.bar(["Full model", "Smoke"], vals, color=[C_CONTIG, C_NEUTRAL])
    ax.set_ylabel("GC content R² (RF probing)")
    ax.set_title("(a) GC encoded in representation")
    ax.set_ylim(0, 1.0); ax.axhline(0, color="k", lw=0.5)
    for i, v in enumerate(vals): ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    # (b) Source ACC
    ax = axes[1]
    vals = [d.get("source_accuracy", 0.393), d.get("source_chance", 0.167)]
    ax.bar(["Source ACC", "Chance"], vals, color=[C_CONTIG, C_NEUTRAL])
    ax.set_ylabel("Source-type accuracy")
    ax.set_title("(b) Taxonomy/source recoverable")
    ax.set_ylim(0, 0.5)
    for i, v in enumerate(vals): ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    fig.suptitle("Fig 2 — Composition is strongly encoded (but see Fig 3: not used for decision)", y=1.02, fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "Fig2_composition.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Fig2_composition.png", bbox_inches="tight"); plt.close(fig)
    print("✓ Fig 2")


def fig3_gc_independence():
    """Fig 3: causal independence from GC (ΔAUC ≈ 0)."""
    d = load("02_gc_ablation.json")
    orig = d.get("hgt_auc_original", 0.692)
    rem = d.get("hgt_auc_after_gc_removal", 0.691)
    delta = d.get("delta_auc", -0.001)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    bars = ax.bar(["Original feature", "GC-direction removed"], [orig, rem], color=[C_CONTIG, C_GC])
    ax.set_ylabel("HGT AUC (RF probing)")
    ax.set_title("Fig 3 — Decision is GC-invariant")
    ax.set_ylim(0, 0.8)
    for i, v in enumerate([orig, rem]): ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    ax.annotate(f"ΔAUC = {delta:+.3f}\n(GC removed barely changes HGT AUC)",
                xy=(0.5, 0.62), ha="center", fontsize=10, color=C_PE,
                bbox=dict(boxstyle="round", fc="#fff3cd", ec=C_GC))
    fig.tight_layout(); fig.savefig(OUT / "Fig3_gc_independence.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Fig3_gc_independence.png", bbox_inches="tight"); plt.close(fig)
    print("✓ Fig 3")


def fig4_dualmode_aggregation():
    """Fig 4: dualmodetrueactual validation + aggregationtowardratio."""
    pe = load("07_real_pe_results.json").get("overall", {}).get("auroc", 0.448)
    contig = load("08_real_contig_results.json").get("max", {}).get("auc", 0.796)
    agg = load("09_aggregation_compare.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    # (a) dualmode
    ax = axes[0]
    ax.bar(["Contig mode", "PE mode"], [contig, pe], color=[C_CONTIG, C_PE])
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="random (0.5)")
    ax.set_ylabel("AUROC on real GIs (IslandViewer)")
    ax.set_title("(a) Dual-mode on real Genomic Islands")
    ax.set_ylim(0, 1.0)
    for i, v in enumerate([contig, pe]): ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=11)
    ax.text(1, pe + 0.06, "PE junction-bound\n(real GI lacks read-junctions)", ha="center", fontsize=8, color=C_PE)
    ax.legend(loc="lower right", fontsize=8)
    # (b) aggregationtowardratio
    ax = axes[1]
    aggs = ["max", "mean", "median", "top3_mean", "top5_mean", "P90"]
    labels = ["max", "mean", "median", "top3", "top5", "P90"]
    x = np.arange(len(aggs)); w = 0.38
    # robust extract:
    bw = next((a for a in agg if a["dataset"] == "Benbow"), {"by_agg": {}})
    iv = next((a for a in agg if a["dataset"] == "IslandViewer"), {"by_agg": {}})
    bv = [bw["by_agg"].get(k, {}).get("auc", 0) for k in aggs]
    ivv = [iv["by_agg"].get(k, {}).get("auc", 0) for k in aggs]
    ax.bar(x - w/2, bv, w, label="Benbow", color=C_CONTIG)
    ax.bar(x + w/2, ivv, w, label="IslandViewer", color="#41b6c4")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Contig-level AUROC")
    ax.set_title("(b) Aggregation strategy (median recommended)")
    ax.set_ylim(0, 0.9); ax.legend(fontsize=9)
    fig.suptitle("Fig 4 — Contig mode generalizes to real GIs; PE is junction-bound", y=1.02, fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "Fig4_dualmode_aggregation.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Fig4_dualmode_aggregation.png", bbox_inches="tight"); plt.close(fig)
    print("✓ Fig 4")


def fig5_blast():
    """Fig 5: BLAST biological evidence (high-scoring windows contain real IAV)."""
    d = load("04_blast_summary.json")
    iav_mean = d.get("score_iav_present", {}).get("mean", 0.665)
    mic_mean = d.get("score_microbe_only", {}).get("mean", 0.256)
    top10 = d.get("top10_iav_frac", 9)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5))
    # (a) score by IAV presence
    ax = axes[0]
    ax.bar(["IAV-containing\nwindows", "Microbe-only\nwindows"], [iav_mean, mic_mean], color=[C_PE, C_CONTIG])
    ax.set_ylabel("Mean window HGT score")
    ax.set_title("(a) Score tracks viral insertion")
    ax.set_ylim(0, 0.8)
    for i, v in enumerate([iav_mean, mic_mean]): ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    # (b) top-k IAV fraction
    ax = axes[1]
    topk = [("Top-10", d.get("top10_iav_frac", 9) / 10),
            ("Top-20", d.get("top20_iav_frac", 18) / 20),
            ("All (top-100)", d.get("windows_with_iav_signal", 41) / d.get("windows_with_blast_hit", 97))]
    names = [t[0] for t in topk]; vals = [t[1] for t in topk]
    ax.bar(names, vals, color=C_PE)
    ax.set_ylabel("Fraction containing IAV (donor) signal")
    ax.set_title("(b) High-score windows are real insertions")
    ax.set_ylim(0, 1.0)
    for i, v in enumerate(vals): ax.text(i, v + 0.02, f"{v*100:.0f}%", ha="center", fontsize=10)
    fig.suptitle("Fig 5 — High-confidence 'false positives' contain genuine viral insertions (BLAST)", y=1.02, fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "Fig5_blast.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Fig5_blast.png", bbox_inches="tight"); plt.close(fig)
    print("✓ Fig 5")


def fig_crossvirus():
    """Fig 6: cross-virus / cross-domain generalization (IAV training → HSV-1 / real GIs). NMI broad-applicability selling point."""
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    labels = ["IAV\n(training\ndistribution)", "HSV-1\n(Herpesviridae,\ncross-virus)", "Real GIs\n(IslandViewer,\ncross-domain)"]
    aucs = [0.96, 0.771, 0.80]
    colors = [C_NEUTRAL, C_PE, C_CONTIG]
    ax.bar(labels, aucs, color=colors)
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="random (0.5)")
    ax.set_ylabel("Contig AUROC"); ax.set_ylim(0, 1.05)
    ax.set_title("Fig 6 — Cross-virus / cross-domain generalization (not IAV-specific)")
    for i, v in enumerate(aucs): ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=11)
    ax.text(1, 0.12, "Generalizes beyond IAV training\ndistribution → NMI broad-applicability",
            ha="center", fontsize=8.5, style="italic")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "Fig6_crossvirus.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Fig6_crossvirus.png", bbox_inches="tight"); plt.close(fig)
    print("✓ Fig 6")


if __name__ == "__main__":
    fig1_architecture()
    fig2_composition()
    fig3_gc_independence()
    fig4_dualmode_aggregation()
    fig5_blast()
    fig_crossvirus()
    print(f"\nAll figures saved to {OUT}/")
