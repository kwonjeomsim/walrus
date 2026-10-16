#!/usr/bin/env python3
"""Summarize the tier-up K sweep from k_sweep.py results.

1. Attainment, code ratio, and RSS ratio per mode with BCa 95% intervals (benchmark resampling)
2. Equal-code comparison: linearly interpolate the tier-up curve (interp, large K -> K=1, lazy)
   along the code-ratio axis to get its attainment at the code ratios of -O1/-O2/-O3, and
   recompute the difference from each level in every resample.
3. Dominance: (attainment, code, RSS) differences between each K and each level. Marked as
   "dominates" when the intervals show tier-up has higher attainment with equal or less code.
4. Timing loss: pre_K - tierup_K. The extra attainment from compiling the same function set
   at load time, i.e. the share lost to warm-up and the lack of OSR.
5. Attainment/code ratio per group (embedded, LLVM, applications)
6. With --train, pick per level the K that maximizes U = attainment - theta * code ratio
   on the training set, and report that K's held-out values.

    k_analyze.py results/ksweep_test_x86.json --train results/ksweep_train_x86.json \\
        --md results/ksweep_x86.md --fig results/ksweep_x86
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from bootstrap_ci import bca, jackknife

LEVELS = [("O1", 4.0), ("O2", 2.0), ("O3", 1.0)]
GROUPS = [("Embedded suites", ("beebs_", "tacle_", "embench_")),
          ("LLVM test-suite", ("llvm_",)),
          ("Applications", ("app_",))]


def kval(m):
    r = re.fullmatch(r"(?:tierup|pre)_K(\d+)", m)
    return int(r.group(1)) if r else None


class Data:
    def __init__(self, path):
        d = json.loads(Path(path).read_text())
        self.benches = sorted(t for t, v in d.items() if v)
        self.failed = sorted(t for t, v in d.items() if not v)
        modes = set.intersection(*(set(d[t]) for t in self.benches))
        self.modes = sorted(modes, key=self._order)
        self.ix = {m: i for i, m in enumerate(self.modes)}
        g = lambda f: np.array([[float(d[t][m][f]) for m in self.modes] for t in self.benches])
        self.t, self.code, self.rss = g("t"), g("code"), g("rss")
        self.ks = sorted({kval(m) for m in self.modes if m.startswith("tierup_K")})

    @staticmethod
    def _order(m):
        fixed = ["interp", "alljit", "O1", "O2", "O3", "lazy"]
        if m in fixed:
            return (0, fixed.index(m), 0)
        k = kval(m)
        if k is not None:
            return (1 if m.startswith("tierup") else 2, 0, k)
        return (3, 0, int(re.sub(r"\D", "", m) or 0))

    def stats(self, idx):
        """(attainment, code ratio, RSS ratio) per mode for resample idx."""
        i, a = self.ix["interp"], self.ix["alljit"]
        st, sc, sr = self.t[idx].sum(0), self.code[idx].sum(0), self.rss[idx].sum(0)
        att = (st[i] - st) / (st[i] - st[a])
        cr = sc / sc[a]
        rr = (sr - sr[i]) / (sr[a] - sr[i]) if sr[a] != sr[i] else np.full_like(sr, np.nan)
        return att, cr, rr

    def curve(self, att, cr, prefix):
        """Curve from large K (less code) to small K, with interp and lazy at the two ends."""
        ms = ["interp"] + [f"{prefix}_K{k}" for k in reversed(self.ks)]
        if "lazy" in self.ix:
            ms.append("lazy")
        xs = np.array([cr[self.ix[m]] for m in ms])
        ys = np.array([att[self.ix[m]] for m in ms])
        o = np.argsort(xs, kind="stable")
        return xs[o], ys[o]


def boot(data: Data, fn, n_boot, seed):
    """fn(idx) -> 1-D array. Returns the point estimate, the BCa interval (lo, hi), and the resample distribution."""
    n = len(data.benches)
    full = np.arange(n)
    est = fn(full)
    rng = np.random.default_rng(seed)
    bs = np.array([fn(rng.integers(0, n, n)) for _ in range(n_boot)])
    jk = np.array([fn(np.delete(full, j)) for j in range(n)])
    ci = [bca(est[k], bs[:, k], jk[:, k]) for k in range(est.size)]
    return est, ci, bs


def fmt(v, ci=None, sign=False):
    f = "{:+.3f}" if sign else "{:.3f}"
    s = f.format(v)
    if ci is not None:
        s += " [" + ", ".join(f.format(c) for c in ci) + "]"
    return s


def iso_code(data, prefix):
    """Per level: (level attainment - curve attainment at the same code ratio)."""
    lv = [m for m, _ in LEVELS if m in data.ix]

    def fn(idx):
        att, cr, _ = data.stats(idx)
        xs, ys = data.curve(att, cr, prefix)
        return np.array([att[data.ix[m]] - np.interp(cr[data.ix[m]], xs, ys) for m in lv])
    return lv, fn


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("test")
    ap.add_argument("--train", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--fig", default=None, help="figure file name (without extension)")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261015)
    args = ap.parse_args()

    D = Data(args.test)
    out = []
    p = out.append
    p(f"# Tier-up K sweep: {Path(args.test).name}\n")
    p(f"- benchmarks: {len(D.benches)} (failed: {', '.join(D.failed) or 'none'})")
    p(f"- K: {D.ks}; bootstrap {args.boot} resamples, BCa 95%\n")

    # 1. Per-mode table
    nm = len(D.modes)
    est, ci, _ = boot(D, lambda idx: np.concatenate(D.stats(idx)), args.boot, args.seed)
    p("## 1. Modes\n")
    p("| Mode | Attainment | Code ratio | RSS ratio |")
    p("|---|---|---|---|")
    for j, m in enumerate(D.modes):
        if m in ("interp", "alljit"):
            continue
        p(f"| {m} | {fmt(est[j], ci[j])} | {fmt(est[nm + j], ci[nm + j])} | "
          f"{fmt(est[2 * nm + j], ci[2 * nm + j])} |")
    p("")

    # 2. Equal-code comparison
    p("## 2. Levels against the tier-up curve at equal code\n")
    p("Positive = the level attains more than tier-up would at the same code ratio "
      "(curve: interp, K large to small, lazy; linear in code ratio).\n")
    p("| Level | vs tier-up curve | vs pre-compiled curve |")
    p("|---|---|---|")
    lv, fn_t = iso_code(D, "tierup")
    has_pre = all(f"pre_K{k}" in D.ix for k in D.ks)
    e1, c1, _ = boot(D, fn_t, args.boot, args.seed + 1)
    if has_pre:
        _, fn_p = iso_code(D, "pre")
        e2, c2, _ = boot(D, fn_p, args.boot, args.seed + 2)
    for j, m in enumerate(lv):
        right = fmt(e2[j], c2[j], True) if has_pre else "-"
        p(f"| {m} | {fmt(e1[j], c1[j], True)} | {right} |")
    p("")

    # 3. Dominance
    p("## 3. Each K against each level (tier-up minus level)\n")
    p("`DOMINATES` = attainment interval above 0 and code interval at or below 0.\n")
    p("| K | Level | Δ attainment | Δ code ratio | Δ RSS ratio | |")
    p("|---|---|---|---|---|---|")
    pairs = [(k, m) for k in D.ks for m, _ in LEVELS if m in D.ix]

    def fn_dom(idx):
        att, cr, rr = D.stats(idx)
        v = []
        for k, m in pairs:
            a, b = D.ix[f"tierup_K{k}"], D.ix[m]
            v += [att[a] - att[b], cr[a] - cr[b], rr[a] - rr[b]]
        return np.array(v)
    e3, c3, _ = boot(D, fn_dom, args.boot, args.seed + 3)
    n_dom = 0
    for j, (k, m) in enumerate(pairs):
        a, c, r = 3 * j, 3 * j + 1, 3 * j + 2
        dom = c3[a][0] > 0 and c3[c][1] <= 0
        n_dom += dom
        p(f"| {k} | {m} | {fmt(e3[a], c3[a], True)} | {fmt(e3[c], c3[c], True)} | "
          f"{fmt(e3[r], c3[r], True)} | {'DOMINATES' if dom else ''} |")
    p(f"\nPairs where tier-up dominates a level: {n_dom} of {len(pairs)}.\n")

    # 4. Timing loss
    if has_pre:
        p("## 4. Timing loss: pre-compiled set minus tier-up, same functions\n")
        p("| K | tier-up | pre-compiled | timing loss | code ratio |")
        p("|---|---|---|---|---|")

        def fn_tl(idx):
            att, cr, _ = D.stats(idx)
            return np.array([att[D.ix[f"pre_K{k}"]] - att[D.ix[f"tierup_K{k}"]] for k in D.ks])
        e4, c4, _ = boot(D, fn_tl, args.boot, args.seed + 4)
        att, cr, _ = D.stats(np.arange(len(D.benches)))
        for j, k in enumerate(D.ks):
            p(f"| {k} | {att[D.ix[f'tierup_K{k}']]:.3f} | {att[D.ix[f'pre_K{k}']]:.3f} | "
              f"{fmt(e4[j], c4[j], True)} | {cr[D.ix[f'tierup_K{k}']]:.3f} |")
        p("")

    # 5. Per group
    p("## 5. By group (attainment / code ratio within the group)\n")
    show = [m for m in ("O1", "O2", "O3", "lazy") if m in D.ix] + \
           [f"tierup_K{k}" for k in D.ks] + ([f"pre_K{D.ks[0]}"] if has_pre else [])
    p("| Group | n | " + " | ".join(show) + " |")
    p("|---|---|" + "---|" * len(show))
    for g, pre in GROUPS:
        idx = np.array([i for i, t in enumerate(D.benches) if t.startswith(pre)])
        if idx.size == 0:
            continue
        att, cr, _ = D.stats(idx)
        p(f"| {g} | {idx.size} | " + " | ".join(
            f"{att[D.ix[m]]:.3f} / {cr[D.ix[m]]:.3f}" for m in show) + " |")
    p("")

    # 6. Pick K on the training set
    chosen = {}
    if args.train:
        T = Data(args.train)
        att, cr, _ = T.stats(np.arange(len(T.benches)))
        p(f"## 6. K chosen on the training set ({len(T.benches)} benchmarks, "
          f"failed: {', '.join(T.failed) or 'none'})\n")
        p("U = attainment - theta * code ratio, maximised over K on the training set, "
          "then read on the held-out set.\n")
        p("| Level | theta | chosen K | train U | held-out tier-up | held-out level | Δ att | Δ code |")
        p("|---|---|---|---|---|---|---|---|")
        att_h, cr_h, _ = D.stats(np.arange(len(D.benches)))
        for m, th in LEVELS:
            u = {k: att[T.ix[f"tierup_K{k}"]] - th * cr[T.ix[f"tierup_K{k}"]] for k in T.ks}
            k = max(u, key=u.get)
            chosen[m] = k
            if f"tierup_K{k}" not in D.ix or m not in D.ix:
                continue
            a, b = D.ix[f"tierup_K{k}"], D.ix[m]
            p(f"| {m} | {th:g} | {k} | {u[k]:.3f} | {att_h[a]:.3f} / {cr_h[a]:.3f} | "
              f"{att_h[b]:.3f} / {cr_h[b]:.3f} | {att_h[a] - att_h[b]:+.3f} | {cr_h[a] - cr_h[b]:+.3f} |")
        p("")

    text = "\n".join(out)
    print(text)
    if args.md:
        Path(args.md).write_text(text + "\n")
    if args.fig:
        figure(D, args.fig, chosen)


def figure(D: Data, stem, chosen):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    att, cr, _ = D.stats(np.arange(len(D.benches)))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    xs, ys = D.curve(att, cr, "tierup")
    ax.plot(xs, ys, "-o", color="#D9822B", ms=4, label="Tier-up, K swept")
    for k in D.ks:
        j = D.ix[f"tierup_K{k}"]
        ax.annotate(f"K={k}", (cr[j], att[j]), fontsize=7, color="#9A5A1C",
                    xytext=(4, -9), textcoords="offset points")
    if all(f"pre_K{k}" in D.ix for k in D.ks):
        xs, ys = D.curve(att, cr, "pre")
        ax.plot(xs, ys, "--s", color="#7A7A7A", ms=3,
                label="Same functions, compiled at load")
    us = [m for m in D.modes if m.startswith("uniform_T")]
    if us:
        ax.scatter([cr[D.ix[m]] for m in us], [att[D.ix[m]] for m in us], marker="x",
                   color="#B03A2E", label="Tier-up, uniform threshold", zorder=3)
    lv = [m for m, _ in LEVELS if m in D.ix]
    ax.plot([cr[D.ix[m]] for m in lv], [att[D.ix[m]] for m in lv], "-o", color="#2A78D6",
            ms=6, label="-O1 / -O2 / -O3", zorder=4)
    for m in lv:
        ax.annotate("-" + m, (cr[D.ix[m]], att[D.ix[m]]), fontsize=8, color="#104281",
                    xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("Machine code emitted (fraction of all-JIT)")
    ax.set_ylabel("Speed-up captured (fraction of all-JIT)")
    ax.set_xlim(-0.02, 0.7)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=200)


if __name__ == "__main__":
    main()
