#!/usr/bin/env python3
"""Plot the results of actually removing each feature and retraining.

Each feature-removed model was built separately and run alternately per benchmark. All three
levels are plotted because importance differs by level. Features useful at -O1, which compiles
little, are not the same as those useful at -O3, which compiles generously, and reading only one
level leads to the opposite conclusion.

    make_ablation_fig.py --measured ablation_measured.json --out docs/figures/ablation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("/home/soonbkwon/projects/walrus")

# A single sequential blue ramp: darker means the feature is harder to do without.
DARK, MID, LIGHT = "#104281", "#2a78d6", "#86b6ef"
NEUTRAL = "#bfbeb9"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": ["DejaVu Sans"],
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.unicode_minus": False, "text.color": INK,
})


def load_measured(path):
    """Build per-feature, per-level loss from run_modes.py results.

    The loss is the time attainment lost by the model without that feature. Positive means the feature is missed.
    """
    d = json.loads(Path(path).read_text())
    tags = sorted(d)
    ti = sum(d[t]["interp"]["t"] for t in tags)
    ta = sum(d[t]["alljit"]["t"] for t in tags)
    ca = sum(d[t]["alljit"]["code"] for t in tags)

    def score(mode):
        if not all(d[t].get(mode) for t in tags):
            return None
        return ((ti - sum(d[t][mode]["t"] for t in tags)) / (ti - ta),
                sum(d[t][mode]["code"] for t in tags) / ca)

    base = {lv: score(lv) for lv in ("O1", "O2", "O3")}
    names = sorted({m.split("@", 1)[1] for t in tags for m in d[t] if "@" in m})
    rows = []
    for n in names:
        drop = {}
        for lv in ("O1", "O2", "O3"):
            s = score(f"{lv}@{n}")
            if s is None:
                break
            drop[lv] = (base[lv][0] - s[0], s[1] - base[lv][1])
        if len(drop) == 3:
            rows.append((n, drop))
    return base, rows, len(tags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measured", required=True, help="per-feature measurements from run_modes.py")
    ap.add_argument("--cumulative", help="cumulative removal measurements")
    ap.add_argument("--order", help="JSON with the cumulative removal order")
    ap.add_argument("--out", default=str(R / "docs/figures/ablation"))
    args = ap.parse_args()
    base, rows, nbench = load_measured(args.measured)

    # Sort by the mean over the three levels. Sorting by one level pushes features that
    # matter only at that level to the top and skews the figure.
    rows.sort(key=lambda r: sum(v[0] for v in r[1].values()) / 3)
    names = [n for n, _ in rows]
    levels = ("O1", "O2", "O3")
    colors = {"O1": LIGHT, "O2": MID, "O3": DARK}

    if args.cumulative:
        fig, (ax, a2) = plt.subplots(1, 2, figsize=(11.6, 6.2),
                                     gridspec_kw={"width_ratios": [1.25, 1]})
    else:
        fig, ax = plt.subplots(figsize=(7.4, 6.4))
        a2 = None
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    h = 0.26
    for k, lv in enumerate(levels):
        ys = [i + (k - 1) * h for i in range(len(rows))]
        vs = [r[1][lv][0] for r in rows]
        ax.barh(ys, vs, height=h * 0.9, color=colors[lv], edgecolor=SURFACE,
                lw=0.6, zorder=3, label=f"-{lv}")
    ax.axvline(0, color=INK2, lw=0.9, zorder=2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(names, fontsize=8.6)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=8.4, colors=INK2, length=0)
    ax.set_xlabel("Speed-up lost when the feature is withheld and the model retrained",
                  fontsize=9.5, color=INK2, labelpad=9)
    ax.legend(frameon=False, fontsize=8.6, loc="lower right", labelcolor=INK,
              title="level", title_fontsize=8.6)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    if a2 is not None:
        import json as _json
        cd = _json.loads(Path(args.cumulative).read_text())
        names_cum = _json.loads(Path(args.order).read_text())
        tags = sorted(cd)
        cti = sum(cd[x]["interp"]["t"] for x in tags)
        cta = sum(cd[x]["alljit"]["t"] for x in tags)
        cca = sum(cd[x]["alljit"]["code"] for x in tags)

        def csc(mode):
            return ((cti - sum(cd[x][mode]["t"] for x in tags)) / (cti - cta),
                    sum(cd[x][mode]["code"] for x in tags) / cca)

        ks = [0]
        while f"O3@c{len(ks)}" in cd[tags[0]]:
            ks.append(len(ks))
        a2.set_axisbelow(True)
        a2.yaxis.grid(True, color=GRID, lw=0.8)
        for lv, col in (("O1", LIGHT), ("O2", MID), ("O3", DARK)):
            ys = [csc(lv if k == 0 else f"{lv}@c{k}")[0] for k in ks]
            a2.plot(ks, ys, "-o", color=col, lw=1.7, ms=5, zorder=4,
                    markeredgecolor=SURFACE, markeredgewidth=1.1, label=f"-{lv}")
        cs = [csc("O3" if k == 0 else f"O3@c{k}")[1] for k in ks]
        a2.plot(ks, cs, "--s", color=NEUTRAL, lw=1.3, ms=4, zorder=3,
                markeredgecolor=SURFACE, label="code, -O3")
        a2.set_xticks(ks)
        a2.set_xlabel("Features withheld together, least costly first",
                      fontsize=9.5, color=INK2, labelpad=9)
        a2.set_ylim(0, 1.0)
        a2.tick_params(labelsize=8.4, colors=INK2, length=0)
        a2.set_title("Together", fontsize=9.5, color=INK2, loc="left", pad=8)
        a2.legend(frameon=False, fontsize=8.4, loc="lower left", labelcolor=INK)
        for sp in ("top", "right"):
            a2.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a2.spines[sp].set_color(GRID)
        ax.set_title("One at a time", fontsize=9.5, color=INK2, loc="left", pad=8)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200)
    print(f"{args.out}.png / .pdf   ({nbench} benches)")
    print(f"  {'feature':<34}{'O1':>9}{'O2':>9}{'O3':>9}{'mean':>9}")
    for n, d in reversed(rows):
        v = [d[lv][0] for lv in levels]
        print(f"  {n:<34}" + "".join(f"{x:>+9.3f}" for x in v)
              + f"{sum(v)/3:>+9.3f}")


if __name__ == "__main__":
    main()
