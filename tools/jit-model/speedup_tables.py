#!/usr/bin/env python3
"""Paper numbers with speed reported as a geometric-mean speed-up (x).

    speed-up of m      S_m = exp(mean_b ln(T_interp,b / T_m,b))
    m against n        S_m / S_n = exp(mean_b ln(T_n,b / T_m,b))
    code ratio         sum_b C_m,b / sum_b C_all-JIT,b

Intervals are BCa 95% over resampled benchmarks for the main results and
percentile 95% for the ablation, as in the paper.

    speedup_tables.py --arm A.json --x86 X.json --ablation ab.json --cumulative cum.json \
        --ksweep ks.json --ksweep-x86 ksx.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bootstrap_ci import bca, jackknife

MODES = ["lazy", "tierup", "O1", "O2", "O3", "oracle"]
GROUPS = [("Embedded suites", ("beebs_", "tacle_", "embench_")),
          ("LLVM test-suite", ("llvm_",)), ("Applications", ("app_",))]
KS = [16384, 4096, 1024, 256, 64, 16, 4, 1]
B, SEED = 10000, 20260928


class Data:
    def __init__(self, path, modes, tags=None):
        d = json.loads(Path(path).read_text())
        self.tags = tags or sorted(t for t, v in d.items() if v)
        self.modes = ["interp", "alljit"] + [m for m in modes if all(m in d[t] for t in self.tags)]
        self.t = np.array([[d[b][m]["t"] for m in self.modes] for b in self.tags])
        self.c = np.array([[float(d[b][m].get("code", 0)) for m in self.modes] for b in self.tags])
        self.lg = np.log(self.t[:, :1] / self.t)       # ln speed-up per benchmark and mode

    def k(self, m):
        return self.modes.index(m)

    def S(self, idx):
        return np.exp(self.lg[idx].mean(0))

    def C(self, idx):
        s = self.c[idx].sum(0)
        return s / s[1]


def draws(n):
    return np.random.default_rng(SEED).integers(0, n, size=(B, n))


def bci(f, n, dr):
    pt = f(np.arange(n))
    b = np.array([f(x) for x in dr])
    lo, hi = bca(pt, b, jackknife(f, n))
    return pt, lo, hi


def main_results(D, name):
    n = len(D.tags)
    dr = draws(n)
    print(f"\n#### {name} ({n} benchmarks)")
    print(f"all-JIT speed-up {D.S(np.arange(n))[1]:.2f}x")
    for m in MODES:
        i = D.k(m)
        s = bci(lambda x: D.S(x)[i], n, dr)
        c = bci(lambda x: D.C(x)[i], n, dr)
        print(f"  {m:7} speed-up {s[0]:.2f}x [{s[1]:.2f}, {s[2]:.2f}]   code {c[0]:.3f} [{c[1]:.3f}, {c[2]:.3f}]")
    print("  pairs: speed-up ratio a/b [CI], ahead%, d code [CI]")
    for a, b_ in [("O1", "tierup"), ("O2", "tierup"), ("O3", "tierup"), ("O2", "O1"),
                  ("O3", "O2"), ("oracle", "O3"), ("oracle", "O2")]:
        i, j = D.k(a), D.k(b_)
        r = bci(lambda x: D.S(x)[i] / D.S(x)[j], n, dr)
        c = bci(lambda x: D.C(x)[i] - D.C(x)[j], n, dr)
        q = bci(lambda x: D.C(x)[i] / D.C(x)[j], n, dr)
        tw = bci(lambda x: D.t[x, j].sum() / D.t[x, i].sum(), n, dr)
        ahead = np.mean([D.S(x)[i] > D.S(x)[j] for x in dr[:2000]]) * 100
        print(f"  {a:6}/{b_:7} {r[0]:.2f}x [{r[1]:.2f}, {r[2]:.2f}]  ahead {ahead:.0f}%   "
              f"d code {c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]   code ratio {q[0]:.2f} [{q[1]:.2f}, {q[2]:.2f}]   "
              f"summed-time ratio {tw[0]:.2f} [{tw[1]:.2f}, {tw[2]:.2f}]")
    print("  groups: speed-up / code ratio")
    for g, pre in GROUPS:
        idx = np.array([i for i, t in enumerate(D.tags) if t.startswith(pre)])
        s, c = D.S(idx), D.C(idx)
        print(f"  {g:16} n={len(idx):2d} all-JIT {s[1]:.2f}x  " + "  ".join(
            f"{m} {s[D.k(m)]:.2f}x/{c[D.k(m)]:.3f}" for m in ["O1", "O2", "O3", "tierup", "oracle"]))
    lin = (D.t[:, :1] - D.t) / (D.t[:, :1] - D.t[:, 1:2])
    sub = np.where(lin[:, D.k("tierup")] > 0.20)[0]
    s, c = D.S(sub), D.C(sub)
    print(f"  tier-up subset (linear att > 0.20) n={len(sub)}: all-JIT {s[1]:.2f}x  " + "  ".join(
        f"{m} {s[D.k(m)]:.2f}x/{c[D.k(m)]:.3f}" for m in ["tierup", "O2", "O3"]))
    tu = D.t[:, D.k("tierup")] / D.t[:, 0]
    print(f"  benchmarks where tier-up gains <5%: {(tu > 1 / 1.05).sum()}  (time > 0.95 interp: {(tu > 0.95).sum()})")
    return D


def transfer(A, X):
    n = len(A.tags)
    dr = draws(n)
    print("\n#### transfer: speed-up AArch64 / x86-64, and ratios against tier-up and all-JIT")
    for m in ["O1", "O2", "O3", "tierup", "alljit"]:
        print(f"  {m:7} {A.S(np.arange(n))[A.k(m)]:.2f}x  {X.S(np.arange(n))[X.k(m)]:.2f}x")
    for m in ["O1", "O2", "O3"]:
        f = lambda x, D: D.S(x)[D.k(m)] / D.S(x)[D.k("tierup")]
        ra, rx = f(np.arange(n), A), f(np.arange(n), X)
        d = bci(lambda x: f(x, A) / f(x, X), n, dr)
        print(f"  {m}/tierup  arm {ra:.2f}x  x86 {rx:.2f}x  arm/x86 {d[0]:.3f} [{d[1]:.3f}, {d[2]:.3f}]")
    for m in ["O1", "O2", "O3", "tierup"]:
        f = lambda x, D: D.S(x)[D.k(m)] / D.S(x)[1]
        print(f"  {m}/alljit  arm {f(np.arange(n), A):.3f}  x86 {f(np.arange(n), X):.3f}")


def ablation(path, cumulative=False):
    d = json.loads(Path(path).read_text())
    extra = sorted(m for m in d[next(iter(d))] if "@" in m)
    D = Data(path, ["O1", "O2", "O3"] + extra)
    n = len(D.tags)
    dr = draws(n)
    S0 = D.S(np.arange(n))
    print(f"\n#### {'cumulative' if cumulative else 'single'} ablation: base " +
          "  ".join(f"{L} {S0[D.k(L)]:.2f}x/{D.C(np.arange(n))[D.k(L)]:.3f}" for L in ("O1", "O2", "O3")))
    feats = sorted({m.split("@")[1] for m in extra}, key=(lambda s: int(s[1:])) if cumulative else None)
    for f in feats:
        cells = []
        for L in ("O1", "O2", "O3"):
            i, j = D.k(f"{L}@{f}"), D.k(L)
            r = lambda x: D.S(x)[i] / D.S(x)[j] - 1
            c = lambda x: D.C(x)[i] - D.C(x)[j]
            rb = np.array([r(x) for x in dr]); cb = np.array([c(x) for x in dr])
            rl, rh = np.percentile(rb, [2.5, 97.5]); cl, ch = np.percentile(cb, [2.5, 97.5])
            full = np.arange(n)
            cells.append(f"{L} {S0[i]:.2f}x speed {100*r(full):+.1f}% [{100*rl:+.1f},{100*rh:+.1f}] "
                         f"code {c(full):+.3f} [{cl:+.3f},{ch:+.3f}]")
        print(f"  {f:22} " + " | ".join(cells))


def ksweep(path, label):
    D = Data(path, ["O1", "O2", "O3", "lazy"] + [f"tierup_K{k}" for k in KS])
    n = len(D.tags)
    dr = draws(n)
    full = np.arange(n)
    S, C = D.S(full), D.C(full)
    print(f"\n#### K sweep {label}: " + "  ".join(f"K{k} {S[D.k(f'tierup_K{k}')]:.2f}x/{C[D.k(f'tierup_K{k}')]:.3f}" for k in KS))
    print("  levels " + "  ".join(f"{m} {S[D.k(m)]:.2f}x/{C[D.k(m)]:.3f}" for m in ["O1", "O2", "O3", "lazy"]))
    tu = ["interp"] + [f"tierup_K{k}" for k in KS]
    ti = [D.k(m) for m in tu]

    def eq(x, L):
        s, c = np.log(D.S(x)), D.C(x)
        o = np.argsort(c[ti])
        return np.exp(s[D.k(L)] - np.interp(c[D.k(L)], c[ti][o], s[ti][o]))
    for L in ["O1", "O2", "O3"]:
        r = bci(lambda x: eq(x, L), n, dr)
        print(f"  {L} vs tier-up curve at equal code: {r[0]:.2f}x [{r[1]:.2f}, {r[2]:.2f}]")
    dom = 0
    for k in KS:
        for L in ["O1", "O2", "O3"]:
            i, j = D.k(f"tierup_K{k}"), D.k(L)
            r = bci(lambda x: D.S(x)[i] / D.S(x)[j], n, dr)
            c = bci(lambda x: D.C(x)[i] - D.C(x)[j], n, dr)
            if r[1] > 1 and c[2] <= 0:
                dom += 1
    print(f"  pairs where tier-up dominates a level: {dom} of 24")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--x86", required=True)
    ap.add_argument("--ablation")
    ap.add_argument("--cumulative")
    ap.add_argument("--ksweep")
    ap.add_argument("--ksweep-x86")
    a = ap.parse_args()
    A = main_results(Data(a.arm, MODES), "AArch64")
    X = main_results(Data(a.x86, MODES), "x86-64")
    transfer(A, X)
    if a.ablation:
        ablation(a.ablation)
    if a.cumulative:
        ablation(a.cumulative, cumulative=True)
    if a.ksweep:
        ksweep(a.ksweep, "AArch64")
    if a.ksweep_x86:
        ksweep(a.ksweep_x86, "x86-64")


if __name__ == "__main__":
    main()
