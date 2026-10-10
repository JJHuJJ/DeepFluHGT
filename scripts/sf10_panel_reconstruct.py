#!/usr/bin/env python3
"""P1.5 — per-panel standalone reconstruction for SF10 (A-D).

Same method as scripts/sf_panel_reconstruct.py (visibility switching on the
original composite figure, native bbox_inches='tight' crop; no artist is
moved or re-parented, artist counts asserted). Reused via import so the
SF2/SF4/SF5/SF6/SF9 pipeline file stays untouched.

Why this exists: the earlier SF10A-D per-panel files were produced by a
fixed-cell crop of the composite, which clipped content at cell boundaries —
the source of the edge-ink flags (e.g. SF10B right edge ~357 dark px,
SF10A/D top edge ~120-170). The tight crop here cannot clip by construction.

No data, statistics, thresholds, aggregation or interpretation is touched.
Output: _full_version/03_FIGURES_MASTER/rendered/SF10A-D.{pdf,png}
"""
import os
import sys

sys.path.insert(0, "scripts")
import matplotlib
matplotlib.use("Agg")
import render_supp_figs as R
import sf_panel_reconstruct as PR

OUT_DIR = R.OUT_DIR


def main():
    orig = R.save_panels

    def spy(fig, stem, panels, out_dir):
        if stem == "SF10":
            PR.reconstruct(fig, stem, panels, OUT_DIR)
            R.done.add(stem)
        raise RuntimeError("captured")

    R.save_panels = spy
    R.done = set()
    try:
        R.sf10()
    except RuntimeError:
        pass
    R.save_panels = orig
    missing = [t for t in ("SF10A", "SF10B", "SF10C", "SF10D")
               if not os.path.exists(os.path.join(OUT_DIR, t + ".png"))]
    if missing:
        raise SystemExit(f"missing panels: {missing}")
    print("SF10 per-panel reconstruction done:", OUT_DIR)


if __name__ == "__main__":
    main()
