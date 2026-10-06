#!/usr/bin/env python3
"""Per-mode bar charts: what each strategy costs in time and in machine code.

Two panels rather than one chart with two y-axes: the quantities have different
units and a shared axis would invite a false comparison. Modes are ordered by
the code they emit, so the left panel reads as "what does that code buy".

    make_bars.py --scratch <dir> --out docs/figures/modes
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("/home/soonbkwon/projects/walrus")

OURS = {"-O1": "#86b6ef", "-O2": "#2a78d6", "-O3": "#104281"}
# Lazy and Tier-up did not exist in Walrus; we implemented them as baselines. They are
# kept apart from the interpreter and all-JIT so it is clear what is ours and what is not.
MINE = {"Lazy": "#d9822b", "Tier-up": "#d9822b"}
BASE = "#bfbeb9"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": ["DejaVu Sans"],
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.unicode_minus": False, "text.color": INK,
})


def load(path: Path):
    """Build per-mode absolute time and machine code size from one run_modes.py JSON."""
    d = json.loads(path.read_text())
    tags = sorted(d)
    ti = sum(d[t]["interp"]["t"] for t in tags)
    rows = []
    for mode, name in (("interp", "Interpreter"), ("O1", "-O1"), ("O2", "-O2"),
                       ("O3", "-O3"), ("tierup", "Tier-up"), ("lazy", "Lazy"),
                       ("alljit", "All-JIT")):
        if not all(d[t].get(mode) for t in tags):
            continue
        rows.append((name,
                     sum(d[t][mode]["t"] for t in tags),
                     sum(d[t][mode]["code"] for t in tags)))
    return rows, ti, len(tags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="run_modes.py results JSON")
    ap.add_argument("--out", default=str(R / "docs/figures/modes"))
    args = ap.parse_args()
    rows, ti, n = load(Path(args.data))

    rows = sorted(rows, key=lambda r: r[2])             # ascending code size
    order = [r[0] for r in rows]
    times = [r[1] for r in rows]
    codes = [r[2] / 2**20 for r in rows]
    norm = [t / ti for t in times]
    colors = [OURS.get(k) or MINE.get(k) or BASE for k in order]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.5))
    x = range(len(order))

    # all-JIT is already a bar, so no separate baseline line is drawn.
    a1.bar(x, norm, color=colors, edgecolor=SURFACE, lw=1.0, width=0.68, zorder=3)
    for i, (v, t) in enumerate(zip(norm, times)):
        a1.text(i, v + 0.025, f"{t:.0f}s", ha="center", fontsize=7.8, color=INK2)
    a1.set_ylabel("Execution time\n(fraction of interpreter)", fontsize=9.5, color=INK2)
    a1.set_ylim(0, 1.16)
    a1.set_title("Lower is faster", fontsize=9.5, color=INK2, loc="left", pad=8)

    a2.bar(x, codes, color=colors, edgecolor=SURFACE, lw=1.0, width=0.68, zorder=3)
    for i, v in enumerate(codes):
        a2.text(i, v + max(codes) * 0.022, f"{v:.2f}", ha="center", fontsize=7.8, color=INK2)
    a2.set_ylabel("Machine code emitted (MB)", fontsize=9.5, color=INK2)
    a2.set_ylim(0, max(codes) * 1.16)
    a2.set_title("Lower is smaller", fontsize=9.5, color=INK2, loc="left", pad=8)

    for ax in (a1, a2):
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(order, fontsize=8.8, rotation=28, ha="right")
        ax.tick_params(axis="y", labelsize=8.5, colors=INK2, length=0)
        ax.tick_params(axis="x", length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200)
    print(f"wrote {args.out}.png / .pdf\n")
    print(f"  {'mode':<12}{'time (s)':>10}{'/interp':>9}{'code (MB)':>11}")
    for k, t, c in rows:
        print(f"  {k:<12}{t:>10.1f}{t/ti:>9.3f}{c/2**20:>11.2f}")


if __name__ == "__main__":
    main()
