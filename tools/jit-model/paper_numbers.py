#!/usr/bin/env python3
"""Paper numbers that speedup_tables.py does not print: summed times, code in MiB,
group shares, run-to-run variation and peak RSS (Table rss, median of the repetitions).

    paper_numbers.py --arm A.json --x86 X.json [--ksweep ks.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bootstrap_ci import bca, jackknife

MODES = ["interp", "alljit", "lazy", "tierup", "O1", "O2", "O3", "oracle"]
GROUPS = [("Embedded suites", ("beebs_", "tacle_", "embench_")),
          ("LLVM test-suite", ("llvm_",)), ("Applications", ("app_",))]
B, SEED = 10000, 20260928
MiB = 1 << 20


def load(path):
    d = json.loads(Path(path).read_text())
    tags = sorted(t for t, v in d.items() if v)
    t = np.array([[d[b][m]["t"] for m in MODES] for b in tags])
    c = np.array([[float(d[b][m].get("code", 0)) for m in MODES] for b in tags])
    r = np.array([[float(np.median(d[b][m]["rss_reps"])) / MiB for m in MODES] for b in tags])
    return d, tags, t, c, r


def k(m):
    return MODES.index(m)


def draws(n):
    return np.random.default_rng(SEED).integers(0, n, size=(B, n))


def bci(f, n, dr):
    pt = f(np.arange(n))
    b = np.array([f(x) for x in dr])
    lo, hi = bca(pt, b, jackknife(f, n))
    return pt, lo, hi, b


def sums(name, d, tags, t, c, r):
    print(f"\n#### {name}")
    print("  summed time (s): " + "  ".join(f"{m} {t[:, k(m)].sum():.1f}" for m in MODES))
    print("  summed code (MiB): " + "  ".join(f"{m} {c[:, k(m)].sum() / MiB:.2f}" for m in MODES))
    for g, pre in GROUPS:
        idx = [i for i, x in enumerate(tags) if x.startswith(pre)]
        lg = np.log(t[idx][:, :1] / t[idx])
        print(f"  {g:16} n={len(idx):2d} time share {100 * t[idx, 0].sum() / t[:, 0].sum():.1f}%  "
              f"code share {100 * c[idx, 1].sum() / c[:, 1].sum():.1f}%  "
              f"oracle {np.exp(lg[:, k('oracle')].mean()):.2f}x/{c[idx, k('oracle')].sum() / c[idx, 1].sum():.3f}")
    cvs = []
    for b in tags:
        for m in MODES:
            reps = d[b][m].get("reps")
            if isinstance(reps, list) and len(reps) > 1:
                a = np.array([x["t"] if isinstance(x, dict) else x for x in reps], float)
                cvs.append(a.std() / a.mean())
    if cvs:
        print(f"  run-to-run CV: median {100 * np.median(cvs):.2f}% over {len(cvs)} benchmark-mode pairs")


def rss(name, tags, r, paired=True):
    n = len(tags)
    dr = draws(n)
    add = r - r[:, :1]
    ratio = lambda x, m: add[x, k(m)].sum() / add[x, 1].sum()
    print(f"\n#### peak RSS {name}: interp {r[:, 0].sum():.0f} MiB, all-JIT adds {add[:, 1].sum():.1f} MiB")
    for m in ["lazy", "tierup", "O1", "O2", "O3", "oracle"]:
        p = bci(lambda x: ratio(x, m), n, dr)
        print(f"  {m:7} ratio {p[0]:.3f} [{p[1]:.3f}, {p[2]:.3f}]  added {add[:, k(m)].sum():.1f} MiB  "
              f"+{100 * add[:, k(m)].sum() / r[:, 0].sum():.1f}% of interp")
    if paired:
        for a in ["O1", "O2", "O3"]:
            p = bci(lambda x: ratio(x, a) - ratio(x, "tierup"), n, dr)
            print(f"  {a} - tierup  {p[0]:+.3f} [{p[1]:+.3f}, {p[2]:+.3f}]  resamples below 0: {100 * (p[3] < 0).mean():.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--x86", required=True)
    ap.add_argument("--ksweep")
    a = ap.parse_args()
    A = load(a.arm)
    X = load(a.x86)
    sums("AArch64", *A)
    sums("x86-64", *X)
    ta, tx = A[2][:, :2].sum(0), X[2][:, :2].sum(0)
    print(f"\n  interp AArch64/x86 {ta[0] / tx[0]:.2f}  all-JIT {ta[1] / tx[1]:.2f}  "
          f"all-JIT code AArch64/x86 {A[3][:, 1].sum() / X[3][:, 1].sum():.2f}")
    rss("AArch64", A[1], A[4])
    rss("x86-64", X[1], X[4])
    if a.ksweep:
        d = json.loads(Path(a.ksweep).read_text())
        tags = A[1]
        lg = lambda src, m: np.exp(np.mean([np.log(src[b]["interp"]["t"] / src[b][m]["t"]) for b in tags]))
        dev = [abs(lg(d, m) / lg(A[0], m) - 1) for m in ["O1", "O2", "O3", "lazy", "alljit"]]
        dev.append(abs(lg(d, "tierup_K1") / lg(A[0], "tierup") - 1))
        print(f"\n  K-sweep run vs main run, largest speed-up difference {100 * max(dev):.2f}%")


if __name__ == "__main__":
    main()
