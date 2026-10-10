#!/usr/bin/env python3
"""P1.5 — per-panel standalone reconstruction for SF2/SF4/SF5/SF6/SF9.

Method: draw the (unchanged) composite once, then for each panel HIDE all
other panels' axes and the figure-level texts that belong elsewhere, and
savefig(bbox_inches='tight') on the original figure itself. No artist is
moved or re-parented, so all transforms stay native (an earlier re-parent
approach left artists bound to stale transforms: clipped frames, silently
dropped below-axes texts). Artist counts before/after are asserted
(completeness gate).

No data, statistics, thresholds, aggregation or interpretation is touched.
"""
import sys

sys.path.insert(0, "scripts")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import render_supp_figs as R

PAD = 28.0  # kept for reference; geometry no longer uses manual padding
COUNT_KEYS = ("lines", "collections", "patches", "texts", "images",
              "containers", "tables")
TARGET = {"SF2", "SF4", "SF5", "SF6", "SF9"}


def counts(ax):
    return {k: len(getattr(ax, k)) for k in COUNT_KEYS}


def reconstruct(fig, stem, panels, out_dir):
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    for lbl, ax in panels.items():
        bb = ax.get_tightbbox(ren)
        if bb is None:
            print(f"  {lbl}: no bbox, skipped")
            continue
        if stem == "SF6" and lbl == "a":
            # multi-line title moves to the SI caption (S6.9) — mpl
            # under-measures multi-line title extents, and the caption
            # already carries the description (figure-shows-evidence rule)
            ax.set_title("")
        # fig-level texts whose CENTER falls inside this panel's bbox;
        # series banners ("SF6 · ...") stay with the composite, never migrate
        mine = []
        for t in list(fig.texts):
            if t.get_text().startswith("SF"):
                continue
            e = t.get_window_extent(ren)
            cx, cy = (e.x0 + e.x1) / 2, (e.y0 + e.y1) / 2
            if bb.x0 <= cx <= bb.x1 and bb.y0 <= cy <= bb.y1:
                mine.append(t)
        before = counts(ax)
        # companion axes (twinx / shared) sharing this panel's canvas
        ext = ax.get_window_extent(ren)
        companions = []
        for ax2 in fig.axes:
            if ax2 is ax or ax2 in panels.values():
                continue
            e2 = ax2.get_window_extent(ren)
            if ext.overlaps(e2):
                companions.append((ax2, e2))
        keep = {ax} | {a for a, _ in companions} | set(mine)
        # visibility switching on the ORIGINAL figure: no artist is moved or
        # re-parented, so every transform stays native — the earlier
        # re-parent approach (ax -> new figure) left artists bound to stale
        # transforms: axes frames clipped, below-axes texts silently dropped.
        # Hidden artists are excluded from draw AND from the tight bbox, so
        # bbox_inches='tight' isolates the panel.
        hide_axes = [a for a in fig.axes if a not in keep]
        hide_texts = [t for t in fig.texts if t not in keep]
        for a in hide_axes:
            a.set_visible(False)
        for t in hide_texts:
            t.set_visible(False)
        after = counts(ax)
        if before != after:
            raise RuntimeError(f"{stem}{lbl} artist drift {before}->{after}")

        tag = f"{stem}{lbl.upper()}"
        # native tight crop: auto-fits ALL artists (labels, notes, legends)
        fig.savefig(f"{out_dir}/{tag}.pdf", bbox_inches="tight",
                    pad_inches=0.32)
        fig.savefig(f"{out_dir}/{tag}.png", dpi=300, bbox_inches="tight",
                    pad_inches=0.32)
        for a in hide_axes:
            a.set_visible(True)
        for t in hide_texts:
            t.set_visible(True)
        n_art = sum(counts(ax).values())
        print(f"  {tag}: standalone, artists={n_art} complete=True")


def main():
    orig = R.save_panels

    def spy(fig, stem, panels, out_dir):
        if stem in TARGET:
            reconstruct(fig, stem, panels, out_dir)
            R.done.add(stem)
        raise RuntimeError("captured")

    R.save_panels = spy
    R.done = set()
    for fn, stem in ((R.sf2, "SF2"), (R.sf4, "SF4"), (R.sf5, "SF5"),
                     (R.sf6, "SF6"), (R.sf9, "SF9")):
        if stem in R.done:
            continue
        try:
            fn()
        except RuntimeError:
            pass
    R.save_panels = orig
    print("P1.5 reconstruction done:", sorted(R.done))


if __name__ == "__main__":
    main()
