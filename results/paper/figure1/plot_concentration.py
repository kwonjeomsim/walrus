"""Concentration curve for the motivation (Section 2.4).

Input : train_functions.csv (training benchmarks only; self time from the interpreter)
Output: concentration.pdf / concentration.png, and the numbers quoted in the text.

For each module, functions are ordered by time share / code share, highest first.
The x-axis stops at 0.5 because every quartile has reached 1.0 by then.
The curve plots the cumulative share of all-JIT machine code (x) against the
cumulative share of interpreted run time (y). We show the median over modules
and the interquartile band. Vertical lines give the median share of machine code
that the label of Eq. (1) selects at theta = 4, 2, 1.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "train_functions.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "concentration"
NOISE = 0.002  # same noise floor as the label

d = pd.read_csv(src)
g = d.groupby("benchmark")
d["ts"] = d.self_time_ms / g.self_time_ms.transform("sum")
d["cs"] = d.alljit_code_bytes / g.alljit_code_bytes.transform("sum")
d["r"] = d.ts / d.cs
mods = g.size().index

grid = np.linspace(0, 1, 1001)
curves = []
for _, x in d.groupby("benchmark"):
    x = x.sort_values(["r", "cs"], ascending=[False, True])
    cx = np.r_[0, x.cs.cumsum().to_numpy()]
    cy = np.r_[0, x.ts.cumsum().to_numpy()]
    curves.append(np.interp(grid, cx, cy))
curves = np.array(curves)
med = np.median(curves, 0)
q1, q3 = np.quantile(curves, [0.25, 0.75], axis=0)

pts = {}
for th in (4, 2, 1):
    hot = (d.ts >= th * d.cs) & (d.ts >= NOISE)
    h = d[hot].groupby("benchmark")[["cs", "ts"]].sum().reindex(mods, fill_value=0)
    pts[th] = (h.cs.median(), h.ts.median(), hot.sum())

# ---- numbers for the text -------------------------------------------------
print(f"modules {len(mods)}, functions {len(d)}")
ne = d.executed == 0
ne_cs = d[ne].groupby("benchmark").cs.sum().reindex(mods, fill_value=0)
print(f"never executed: {ne.sum()} functions ({100*ne.mean():.1f}%), "
      f"median code share per module {ne_cs.median():.3f}")
for th, (cx, cy, n) in pts.items():
    print(f"theta={th}: {n} functions ({100*n/len(d):.1f}%), "
          f"median module code {cx:.3f}, time {cy:.3f}")
for q in (0.05, 0.10, 0.20):
    i = int(round(q * 1000))
    print(f"first {q:.0%} of code -> median {med[i]:.3f} of run time "
          f"(IQR {q1[i]:.3f}-{q3[i]:.3f})")

# ---- figure ---------------------------------------------------------------
INK, INK2, GRID, BG = "#0B0B0B", "#52514E", "#E4E3DF", "#FCFCFB"
LEVEL = {4: "#86B6EF", 2: "#2A78D6", 1: "#104281"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42,          # no Type 3 fonts
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
})
fig, ax = plt.subplots(figsize=(3.4, 2.5), facecolor=BG)
ax.set_facecolor(BG)
ax.grid(True, color=GRID, lw=0.6)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

XMAX = 0.5  # every quartile is at 1.0 well before this point
ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#A9A8A3", zorder=1)
ax.fill_between(grid, q1, q3, color="#C9C8C4", alpha=0.55, lw=0, zorder=2,
                label="Interquartile range")
ax.plot(grid, med, color=INK, lw=1.4, zorder=3, label="Median across modules")
for th, (cx, cy, _) in pts.items():
    ax.axvline(cx, color=LEVEL[th], lw=1.3, ls=(0, (3, 1.5)), zorder=4,
               label=fr"Label at $\theta = {th}$ (median)")

ax.set_xlim(0, XMAX)
ax.set_ylim(0, 1.03)
ax.set_xticks(np.arange(0, XMAX + 1e-9, 0.1))
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_xlabel("Cumulative share of machine code (all-JIT)")
ax.set_ylabel("Cumulative share of run time")
fig.canvas.draw()
p0 = ax.transData.transform((0, 0)); p1 = ax.transData.transform((0.4, 0.4))
ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
ax.text(0.30, 0.315, "uniform spread", color=INK2, fontsize=6.5,
        rotation=ang, rotation_mode="anchor", ha="left", va="bottom")
h, l = ax.get_legend_handles_labels()
order = [1, 0, 2, 3, 4]
ax.legend([h[i] for i in order], [l[i] for i in order], loc="center right", bbox_to_anchor=(1.0, 0.70),
          frameon=True, facecolor=BG, edgecolor="none", framealpha=1,
          fontsize=6.5, labelcolor=INK2, handlelength=2.2)
fig.tight_layout(pad=0.3)
fig.savefig(out + ".pdf", facecolor=BG)
fig.savefig(out + ".png", dpi=300, facecolor=BG)
