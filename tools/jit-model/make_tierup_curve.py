#!/usr/bin/env python3
"""Speed against code: tier-up over a range of K against the three levels.

The speed axis is the geometric-mean attainment of the paper; the code axis is
the summed code ratio. Tier-up runs from the interpreter through K = 16384 .. 1
to lazy compilation; the levels run from the interpreter through -O1 .. -O3.
The oracle point comes from a separate run on the same benchmarks (--oracle).

    make_tierup_curve.py --sweep results/ksweep_test_arm.json \
        --oracle results/test60_arm_v6.json --out figures/pareto
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OURS = {"O1": "#86b6ef", "O2": "#2a78d6", "O3": "#104281"}
MINE, LAZY, BASE, ORACLE = "#d9822b", "#1baf7a", "#52514e", "#8a8880"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
KS = [16384, 4096, 1024, 256, 64, 16, 4, 1]
LABEL_K = (16384, 1024, 256, 64, 1)

plt.rcParams.update({
    "font.family": ["DejaVu Sans"],
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.unicode_minus": False, "text.color": INK,
})


SPEEDUP = False


def scorer(d):
    tags = sorted(t for t, v in d.items() if v)
    ca = sum(d[t]["alljit"]["code"] for t in tags)
    lj = sum(math.log(d[t]["interp"]["t"] / d[t]["alljit"]["t"]) for t in tags)

    def pt(m):
        c = sum(d[t][m]["code"] for t in tags) / ca
        lm = sum(math.log(d[t]["interp"]["t"] / d[t][m]["t"]) for t in tags)
        # SPEEDUP: geometric-mean speed-up over the interpreter; else log-normalised attainment
        a = math.exp(lm / len(tags)) if SPEEDUP else lm / lj
        return c, a
    return pt, tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True, help="k_sweep.py result JSON")
    ap.add_argument("--oracle", help="run_modes.py result JSON with an oracle mode (same benchmarks)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--speedup", action="store_true",
                    help="y axis is the geometric-mean speed-up over the interpreter (log scale)")
    args = ap.parse_args()
    global SPEEDUP
    SPEEDUP = args.speedup

    d = json.loads(Path(args.sweep).read_text())
    pt, tags = scorer(d)
    base = 1.0 if SPEEDUP else 0.0     # the interpreter
    tu = [(0.0, base)] + [pt(f"tierup_K{k}") for k in KS] + [pt("lazy")]
    curve = tu[:-1]   # tier-up proper: interpreter, then K = 16384 .. 1
    ol = [(0.0, base)] + [pt(m) for m in ("O1", "O2", "O3")]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, lw=0.8)
    ax.plot([p[0] for p in curve], [p[1] for p in curve], "-s", color=MINE, lw=1.8, ms=5.5,
            mec=SURFACE, mew=1.2, zorder=3, label="Tier-up, K = 16384 to 1")
    # Lazy compilation compiles on first call without counters, so it is a separate point.
    ax.plot([p[0] for p in ol], [p[1] for p in ol], "-", color=OURS["O2"], lw=1.8,
            alpha=0.85, zorder=4, label="Our levels")
    for m, (c, a) in zip(("O1", "O2", "O3"), ol[1:]):
        ax.scatter([c], [a], s=100, color=OURS[m], edgecolors=SURFACE, lw=1.5, zorder=5)
        off = {"O1": (7, -14), "O2": (-8, 8), "O3": (-4, 9)}[m]
        ax.annotate(m, (c, a), textcoords="offset points", xytext=off,
                    ha="left" if off[0] > 0 else "right", fontsize=10.5,
                    fontweight="bold", color=OURS["O3"])
    for k, (c, a) in zip(KS, tu[1:-1]):
        if k in LABEL_K:
            off = (9, -2) if k == KS[0] else (6, -12)
            ax.annotate(f"K={k}", (c, a), textcoords="offset points", xytext=off,
                        fontsize=8, color=INK2)
    c, a = tu[-1]
    ax.scatter([c], [a], s=70, marker="^", color=LAZY, edgecolors=SURFACE, lw=1.2, zorder=5,
               label="Lazy compilation")
    ax.annotate("Lazy", (c, a), textcoords="offset points", xytext=(-6, -14), ha="right",
                fontsize=9, color=LAZY)
    aj = pt("alljit")[1]
    ax.scatter([1], [aj], s=60, marker="D", facecolors="none", edgecolors=BASE, lw=1.5, zorder=4)
    ax.annotate("All-JIT", (1, aj), textcoords="offset points", xytext=(-8, -14), ha="right",
                fontsize=9, color=INK2)
    ax.scatter([0], [base], s=60, marker="o", facecolors="none", edgecolors=BASE, lw=1.5, zorder=6)
    ax.annotate("Interpreter", (0, base), textcoords="offset points", xytext=(8, -12),
                fontsize=9, color=INK2)
    if args.oracle:
        od = json.loads(Path(args.oracle).read_text())
        opt, otags = scorer(od)
        assert otags == tags, "oracle run covers different benchmarks"
        c, a = opt("oracle")
        ax.scatter([c], [a], s=90, marker="*", color=ORACLE, zorder=5)
        ax.annotate("Oracle", (c, a), textcoords="offset points", xytext=(8, -4),
                    fontsize=9, color=INK2, style="italic")
    ax.set_xlabel("Machine code emitted  (fraction of all-JIT)", fontsize=10, color=INK2)
    ax.set_xlim(-0.04, 1.06)
    if SPEEDUP:
        from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
        ax.set_yscale("log")
        ticks = [t for t in (1, 1.5, 2, 3, 5, 8, 12, 20) if t <= aj * 1.25]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}\u00d7"))
        ax.set_ylim(0.93, aj * 1.25)
        ax.set_ylabel("Speed-up over the interpreter  (geometric mean)", fontsize=10, color=INK2)
    else:
        ax.set_ylim(-0.06, 1.10)
        ax.set_ylabel("Speed-up captured  (geometric mean)", fontsize=10, color=INK2)
    ax.tick_params(labelsize=8.5, colors=INK2, length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200)
    print(f"{args.out}.png / .pdf ({len(tags)} benchmarks)")
    for k, p in zip(KS, tu[1:-1]):
        print(f"  tierup_K{k:<6} code {p[0]:.3f}  speed {p[1]:.3f}")
    for m, p in zip(("O1", "O2", "O3"), ol[1:]):
        print(f"  {m:<12} code {p[0]:.3f}  speed {p[1]:.3f}")
    print(f"  lazy         code {tu[-1][0]:.3f}  speed {tu[-1][1]:.3f}")
    if args.oracle:
        print(f"  oracle       code {c:.3f}  speed {a:.3f}")


if __name__ == "__main__":
    main()
