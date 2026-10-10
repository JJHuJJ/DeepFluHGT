#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M0 trajectory analysis — DeepFluHGT Stage B-0 [P1-5].

reads `checkpoints/<run>/epoch_metrics.json` (read-only), emits the four-piece output + auxiliary trajectory table and figure:
  ① m0_trajectory.csv — wide per-epoch table
  ② m0_trajectory_summary.json — — each metric of first/last/best/Δ with epoch count
  ③ m0_trajectory.png — training dynamics figure (loss/lr, main metrics, operating point, stratification)
  ④ m0_trajectory_report.md — human-readable report

**no recommendation of any kind** — descriptive only (Stage A discipline).
M0 currently has only ep1, so figures/tables render as a single point + best-value annotations; re-run after later epochs are appended.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

METRICS_NUMERIC = [
    "train_loss", "lr", "train_time_s", "val_auprc", "val_auroc",
    "optimal_threshold", "f1_at_optimal", "recall_at_p90", "recall_at_p95",
    "cal_ece", "hard_neg_auc", "easy_neg_auc", "pos_recall",
    "direction_accuracy", "best_val_auprc",
]
HIGHER_BETTER = {"val_auprc", "val_auroc", "f1_at_optimal", "recall_at_p90",
                 "recall_at_p95", "hard_neg_auc", "easy_neg_auc", "pos_recall",
                 "direction_accuracy", "best_val_auprc"}
LOWER_BETTER = {"train_loss", "cal_ece"}
# known degenerate-by-implementation metric (see metric_sanity_check): direction_accuracy is 0-filled by default
KNOWN_DEGENERATE = {"direction_accuracy"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="Contig_500k_ntv3only")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "docs" / "evidence" / "m0_trajectory")
    args = ap.parse_args()
    t0 = time.time()

    src = REPO / "checkpoints" / args.run / "epoch_metrics.json"
    hist = json.loads(src.read_text())
    hist.sort(key=lambda e: e.get("epoch", 0))
    eps = [e["epoch"] for e in hist]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # -- ① CSV wide table --
    import pandas as pd
    df = pd.DataFrame(hist)[["epoch"] + [m for m in METRICS_NUMERIC
                                         if m in hist[0]]]
    df.to_csv(args.out_dir / "m0_trajectory.csv", index=False)

    # -- ② summary JSON --
    summary = {
        "meta": {
            "run": args.run, "source": str(src.relative_to(REPO)),
            "n_epochs": len(hist), "epochs": eps,
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "policy": "descriptive only, no recommendation",
        },
        "metrics": {},
    }
    for m in METRICS_NUMERIC:
        if m not in hist[0]:
            continue
        v = np.array([e.get(m, np.nan) for e in hist], dtype=float)
        if m in HIGHER_BETTER:
            best_i = int(np.nanargmax(v))
        elif m in LOWER_BETTER:
            best_i = int(np.nanargmin(v))
        else:
            best_i = None
        summary["metrics"][m] = {
            "first": float(v[0]), "last": float(v[-1]),
            "delta": float(v[-1] - v[0]),
            "best": float(v[best_i]) if best_i is not None else None,
            "best_epoch": eps[best_i] if best_i is not None else None,
            "direction": ("higher_better" if m in HIGHER_BETTER else
                          "lower_better" if m in LOWER_BETTER else "monitor"),
            "degenerate_flag": (m in KNOWN_DEGENERATE and
                                "0 filled by default, not interpretable (see M0_METRIC_SANITY)"),
            "single_epoch_note": ("only 1 epoch; Δ/best are meaningless" if len(hist) == 1
                                  else None),
        }
    # derived: hard-easy gap
    if "hard_neg_auc" in hist[0] and "easy_neg_auc" in hist[0]:
        gap = [e["hard_neg_auc"] - e["easy_neg_auc"] for e in hist]
        summary["derived"] = {
            "hard_minus_easy_neg_auc": {
                "values": gap, "last": gap[-1],
                "note": ">0 means ranking is better on hard negatives than on simple negatives"},
        }
    (args.out_dir / "m0_trajectory_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    # -- ③ figure --
    _plot(args.out_dir / "m0_trajectory.png", hist, eps)

    # -- ④ report MD --
    _report(args.out_dir / "m0_trajectory_report.md", summary, df, hist)
    print(f"[traj] {len(hist)} epochs -> {args.out_dir} "
          f"({time.time() - t0:.1f}s)")


def _plot(path: Path, hist, eps) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    single = len(eps) == 1
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("M0 trajectory (descriptive only — no recommendation)")

    def _series(ax_, key, label, color=None):
        v = [e.get(key) for e in hist]
        ax_.plot(eps, v, "o-" if not single else "o", label=label, color=color)
        if not single:
            ax_.set_xticks(eps)

    a = ax[0, 0]
    _series(a, "train_loss", "train_loss")
    a2 = a.twinx()
    a2.plot(eps, [e.get("lr", np.nan) for e in hist], "s--", color="tab:orange",
            alpha=.7, label="lr")
    a.set_ylabel("train_loss"); a2.set_ylabel("lr")
    a.set_title("loss / lr"); a.legend(loc="upper left", fontsize=8)
    a2.legend(loc="upper right", fontsize=8)

    b = ax[0, 1]
    _series(b, "val_auprc", "val_auprc", "tab:blue")
    _series(b, "val_auroc", "val_auroc", "tab:cyan")
    _series(b, "f1_at_optimal", "f1@opt", "tab:green")
    b.axhline(0.5, ls=":", c="gray", lw=.8)
    b.set_ylim(0, 1); b.set_title("main metrics"); b.legend(fontsize=8)

    c = ax[1, 0]
    _series(c, "recall_at_p90", "recall@P90", "tab:red")
    _series(c, "recall_at_p95", "recall@P95", "tab:pink")
    c.set_title("operating-point recall\n(4e-5 ~= 1/n_pos: single-TP point, see sanity audit)")
    c.legend(fontsize=8)
    if not single:
        c.set_yscale("symlog", linthresh=1e-5)

    d = ax[1, 1]
    _series(d, "hard_neg_auc", "hard_neg_auc", "tab:purple")
    _series(d, "easy_neg_auc", "easy_neg_auc", "violet")
    _series(d, "pos_recall", "pos_recall", "tab:olive")
    _series(d, "cal_ece", "cal_ece", "tab:brown")
    d.set_ylim(0, 1); d.set_title("stratified / calibration"); d.legend(fontsize=8)

    for a_ in ax.flat:
        a_.set_xlabel("epoch")
        a_.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _df_markdown(df, floatfmt=".4g") -> str:
    """render a markdown table without the tabulate dependency."""
    def _fmt(v):
        if isinstance(v, float):
            return format(v, floatfmt)
        return str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines)


def _report(path: Path, summary, df, hist) -> None:
    L = []
    A = L.append
    m = summary["meta"]
    A("# M0 Trajectory Report (descriptive)\n")
    A(f"- run: `{m['run']}`; source: `{m['source']}`; epochs: {m['epochs']} "
      f"(n={m['n_epochs']})")
    A(f"- generate: {m['generated']}; discipline: {m['policy']}\n")
    A("## per- epoch metric\n")
    A(_df_markdown(df))
    A("\n## per-metric summary\n")
    A("| metric | first | last | Δ | best (epoch) | direction | note |")
    A("|---|---|---|---|---|---|---|")
    for k, v in summary["metrics"].items():
        note = v.get("degenerate_flag") or v.get("single_epoch_note") or ""
        best = " — " if v["best"] is None else f"{v['best']:.4g}"
        if v["best_epoch"] is not None:
            best += f" (ep{v['best_epoch']})"
        A(f"| {k} | {v['first']:.4g} | {v['last']:.4g} | {v['delta']:+.4g} | "
          f"{best} | {v['direction']} | {note} |")
    if "derived" in summary:
        d = summary["derived"]["hard_minus_easy_neg_auc"]
        A(f"\n## Derived quantity\n\n- hard − easy neg AUC (last) = {d['last']:+.4f} — {d['note']}")
    A("\n## metric semantics links\n")
    A("- semantics audit of recall@P90/P95 and direction_accuracy: "
      "`docs/evidence/M0_METRIC_SANITY_*.md`")
    A("- this report contains no optimizer recommendations (Stage A/B discipline).\n")
    Path(path).write_text("\n".join(L))


if __name__ == "__main__":
    main()
