#!/usr/bin/env python3
"""Summarize k_sweep.py results with the paper metrics (geomean attainment, sum-based code ratio).

1. Attainment and code ratio per mode (BCa 95%, resampling benchmarks)
2. On the training set, pick for each level the K maximizing U = attainment - theta * code ratio,
   then compare that K's tier-up paired with the level on the held-out set.
3. Equal-code comparison: linearly interpolate the tier-up curve (interp, large K -> 1, lazy)
   along the code ratio axis, take the attainment at each level's code ratio and subtract it from the level.
4. Which K dominate a level (attainment interval > 0 and code interval <= 0).

    ksweep_geomean.py results/ksweep_test_arm.json --train results/ksweep_train_x86.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bootstrap_ci import bca, jackknife

KS = [16384, 4096, 1024, 256, 64, 16, 4, 1]
LEVELS = {"O1": 4.0, "O2": 2.0, "O3": 1.0}


class Sweep:
    def __init__(self, path, modes):
        d = json.loads(Path(path).read_text())
        self.tags = sorted(t for t, v in d.items() if v)
        self.modes = ["interp", "alljit"] + [m for m in modes if all(m in d[t] for t in self.tags)]
        self.t = np.array([[d[b][m]["t"] for m in self.modes] for b in self.tags])
        self.c = np.array([[float(d[b][m]["code"]) for m in self.modes] for b in self.tags])
        self.lg = np.log(self.t[:, :1] / self.t)

    def k(self, m):
        return self.modes.index(m)

    def stat(self, idx):
        s, c = self.lg[idx].sum(0), self.c[idx].sum(0)
        return s / s[1], c / c[1]


def curve_at(codes, atts, x):
    """Attainment at code x on the tier-up curve (sorted by code)."""
    o = np.argsort(codes)
    return float(np.interp(x, np.asarray(codes)[o], np.asarray(atts)[o]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test")
    ap.add_argument("--train")
    ap.add_argument("-B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261015)
    a = ap.parse_args()

    modes = list(LEVELS) + ["lazy"] + [f"tierup_K{k}" for k in KS]
    S = Sweep(a.test, modes)
    n = len(S.tags)
    allidx = np.arange(n)
    draws = np.random.default_rng(a.seed).integers(0, n, size=(a.B, n))
    pa, pc = S.stat(allidx)
    boot = [S.stat(x) for x in draws]
    ba, bc = np.array([b[0] for b in boot]), np.array([b[1] for b in boot])
    jk = [S.stat(i) for i in [np.delete(allidx, j) for j in range(n)]]
    ja, jc = np.array([j[0] for j in jk]), np.array([j[1] for j in jk])

    def ci(pt, b, j):
        lo, hi = bca(pt, b, j)
        return f"[{lo:+.3f}, {hi:+.3f}]"

    print(f"# {a.test}: {n} benchmarks, geometric-mean attainment / summed code ratio")
    for m in modes:
        if m not in S.modes:
            continue
        i = S.k(m)
        print(f"  {m:14} {pa[i]:.3f} {ci(pa[i], ba[:, i], ja[:, i])}   code {pa[i] * 0 + pc[i]:.3f}")

    # equal code
    tu = ["interp"] + [f"tierup_K{k}" for k in KS] + ["lazy"]
    ti = [S.k(m) for m in tu]
    print("\n# levels against the tier-up curve at equal code (level - curve)")
    for L in LEVELS:
        i = S.k(L)
        f = lambda A, C: A[i] - curve_at(C[ti], A[ti], C[i])
        pt = f(pa, pc)
        b = np.array([f(x, y) for x, y in zip(ba, bc)])
        j = np.array([f(x, y) for x, y in zip(ja, jc)])
        print(f"  {L}: {pt:+.3f} {ci(pt, b, j)}")

    # dominance
    dom = 0
    for k in KS:
        for L in LEVELS:
            i, j_ = S.k(f"tierup_K{k}"), S.k(L)
            da = bca(pa[i] - pa[j_], ba[:, i] - ba[:, j_], ja[:, i] - ja[:, j_])
            dc = bca(pc[i] - pc[j_], bc[:, i] - bc[:, j_], jc[:, i] - jc[:, j_])
            if da[0] > 0 and dc[1] <= 0:
                dom += 1
                print(f"  K={k} dominates {L}")
    print(f"\n# pairs where tier-up dominates a level: {dom} of {len(KS) * len(LEVELS)}")

    if a.train:
        T = Sweep(a.train, [f"tierup_K{k}" for k in KS])
        ta, tc = T.stat(np.arange(len(T.tags)))
        print(f"\n# K chosen on the training set ({len(T.tags)} benchmarks), read on held-out")
        for L, th in LEVELS.items():
            u = {k: ta[T.k(f"tierup_K{k}")] - th * tc[T.k(f"tierup_K{k}")] for k in KS}
            kb = max(u, key=u.get)
            i, j_ = S.k(L), S.k(f"tierup_K{kb}")
            da, dc = pa[i] - pa[j_], pc[i] - pc[j_]
            print(f"  {L} theta={th:g}: K={kb} (train U {u[kb]:.3f})  tier-up {pa[j_]:.3f}/{pc[j_]:.3f}"
                  f"  level {pa[i]:.3f}/{pc[i]:.3f}  level-tierup att {da:+.3f}"
                  f" {ci(da, ba[:, i] - ba[:, j_], ja[:, i] - ja[:, j_])}"
                  f" code {dc:+.3f} {ci(dc, bc[:, i] - bc[:, j_], jc[:, i] - jc[:, j_])}")


if __name__ == "__main__":
    main()
