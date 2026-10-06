#!/usr/bin/env python3
"""Draw the code-size / speed trade-off plane used in the paper.

Every mode is one point: x is the machine code it emits as a fraction of what
compiling everything emits, y is how much of the achievable speed-up it captures.
Compiling everything sits at (1, 1) and the interpreter at (0, 0), so the useful
region is the upper-left corner: fast with little code.

    make_pareto.py --out docs/figures/pareto
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

# dataviz reference palette: blue ramp for our own staged model, neutral ink for
# the baselines, so the eye separates "ours" from "theirs" before reading labels.
OURS = {"O1": "#86b6ef", "O2": "#2a78d6", "O3": "#104281"}
BASE = "#52514e"
# 우리가 기준선으로 구현한 것(Lazy, Tier-up)은 Walrus 가 원래 가진 것과 구분한다.
MINE = "#d9822b"
ORACLE = "#8a8880"
SURFACE = "#fcfcfb"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": ["DejaVu Sans"],
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.unicode_minus": False, "text.color": INK,
})


def load(path: Path):
    """run_modes.py 가 쓴 JSON 하나에서 모든 모드의 좌표를 만든다."""
    d = json.loads(path.read_text())
    tags = sorted(d)
    ti = sum(d[t]["interp"]["t"] for t in tags)
    ta = sum(d[t]["alljit"]["t"] for t in tags)
    ca = sum(d[t]["alljit"]["code"] for t in tags)
    pts = {}
    for mode, name in (("interp", "Interpreter"), ("alljit", "All-JIT"),
                       ("lazy", "Lazy"), ("tierup", "Tier-up"),
                       ("O1", "O1"), ("O2", "O2"), ("O3", "O3"),
                       ("oracle", "Oracle")):
        if not all(d[t].get(mode) for t in tags):
            continue
        tsum = sum(d[t][mode]["t"] for t in tags)
        csum = sum(d[t][mode]["code"] for t in tags)
        pts[name] = ((ti - tsum) / (ti - ta), csum / ca)
    return pts, len(tags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="run_modes.py 결과 JSON")
    ap.add_argument("--frontier", help="make_theta_frontier.py 결과 JSON")
    ap.add_argument("--out", default=str(R / "docs/figures/pareto"))
    args = ap.parse_args()
    pts, n = load(Path(args.data))

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, lw=0.8)

    # Our staged model, joined so the three levels read as one dial the user turns.
    xs = [pts[l][1] for l in ("O1", "O2", "O3")]
    ys = [pts[l][0] for l in ("O1", "O2", "O3")]
    ax.plot(xs, ys, "-", color=OURS["O2"], lw=1.6, zorder=3, alpha=0.85)
    # -O1 과 -O2 는 가까이 붙는 일이 많아 이름표를 곡선의 아래와 위로 갈라 놓는다.
    label_at = {"O1": ((6, -17), "left"), "O2": ((-8, 8), "right"), "O3": ((9, -12), "left")}
    for l in ("O1", "O2", "O3"):
        y, x = pts[l]
        ax.scatter([x], [y], s=110, color=OURS[l], edgecolors=SURFACE, lw=1.6, zorder=5)
        off, ha = label_at[l]
        ax.annotate(l, (x, y), textcoords="offset points", xytext=off, ha=ha,
                    fontsize=11, fontweight="bold", color=OURS["O3"])

    for name, mk, off in (("Tier-up", "s", (10, -3)),
                          ("Lazy", "^", (-14, -20)),
                          ("All-JIT", "D", (-52, -4)), ("Interpreter", "o", (10, 2))):
        if name not in pts:
            continue
        y, x = pts[name]
        c = MINE if name in ("Tier-up", "Lazy") else BASE
        ax.scatter([x], [y], s=70, marker=mk, facecolors="none",
                   edgecolors=c, lw=1.6, zorder=4)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=off,
                    fontsize=9.5, color=c if c == MINE else INK2,
                    ha="right" if off[0] < 0 else "left")
    if "Oracle" in pts:
        y, x = pts["Oracle"]
        ax.scatter([x], [y], s=70, marker="*", color=ORACLE, zorder=4)
        ax.annotate("Oracle labels", (x, y), textcoords="offset points",
                    xytext=(8, -4), fontsize=9.5, color=INK2, style="italic")

    ax.set_xlabel("Machine code emitted  (fraction of all-JIT)", fontsize=10.5, color=INK2)
    ax.set_ylabel("Speed-up captured  (fraction of all-JIT)", fontsize=10.5, color=INK2)
    ax.set_xlim(-0.04, 1.06)
    ax.set_ylim(-0.06, 1.10)
    ax.tick_params(labelsize=9, colors=INK2, length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200)
    print(f"{args.out}.png / .pdf 작성")
    for k, (y, x) in sorted(pts.items(), key=lambda kv: kv[1][1]):
        print(f"  {k:<12} code {x:.3f}  speed {y:.3f}")


if __name__ == "__main__":
    main()
