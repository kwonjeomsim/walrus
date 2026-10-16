#!/usr/bin/env python3
"""Bootstrap 95% confidence intervals for attainment and code ratio.

There are two sources of variation, and they are resampled over different units.

    Benchmark resampling  Redraw the 55 held-out benchmarks with replacement and recompute.
                          Answers "how much would this number change with a different 55?"
                          Attainment and code ratio are both sums across benchmarks
                          (a ratio of sums, not a ratio of ratios), so the bootstrap fits.

    Repeat resampling     Redraw the 10 runs per benchmark and mode, recompute the median.
                          Answers "what if the same 55 were rerun?" Measurement noise only.

Differences between modes are computed pairwise **within the same resample**, so the
benchmark mix affects both modes equally and the difference interval is narrower than either.

    tools/jit-model/bootstrap_ci.py ~/measure-arm/final.json
    tools/jit-model/bootstrap_ci.py ~/measure-arm/final.json --x86 ~/measure-x86/final.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MODES = ["lazy", "tierup", "O1", "O2", "O3", "oracle"]
LABEL = {"lazy": "Lazy", "tierup": "Tier-up", "O1": "-O1", "O2": "-O2",
         "O3": "-O3", "oracle": "Oracle labels"}


def load(path: Path):
    """Read as a benchmark x mode matrix. t holds medians, reps the 10 raw runs."""
    d = json.loads(Path(path).expanduser().read_text())
    benches = sorted(d)
    modes = ["interp", "alljit"] + MODES
    t = np.array([[d[b][m]["t"] for m in modes] for b in benches])
    code = np.array([[float(d[b][m]["code"]) for m in modes] for b in benches])
    reps = [[np.asarray(d[b][m]["reps"], dtype=float) for m in modes] for b in benches]
    return benches, modes, t, code, reps


def attain(t: np.ndarray, idx) -> np.ndarray:
    """Time attainment per mode. t is the (bench, mode) matrix, idx the drawn bench indices."""
    s = t[idx].sum(axis=0)
    return (s[0] - s) / (s[0] - s[1])


def ratio(code: np.ndarray, idx) -> np.ndarray:
    s = code[idx].sum(axis=0)
    return s / s[1]


def bca(theta_hat: float, boot: np.ndarray, jack: np.ndarray, alpha=0.05):
    """Bias-corrected and accelerated interval. A ratio of sums is asymmetric, so it needs correction."""
    from scipy.stats import norm
    b = boot[np.isfinite(boot)]
    if b.size == 0:
        return (float("nan"), float("nan"))
    p0 = (b < theta_hat).mean()
    p0 = min(max(p0, 1.0 / b.size), 1 - 1.0 / b.size)
    z0 = norm.ppf(p0)
    jm = jack.mean()
    num = ((jm - jack) ** 3).sum()
    den = 6.0 * (((jm - jack) ** 2).sum() ** 1.5)
    a = num / den if den != 0 else 0.0
    out = []
    for z in (norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)):
        adj = z0 + (z0 + z) / (1 - a * (z0 + z))
        out.append(float(np.percentile(b, 100 * norm.cdf(adj))))
    return tuple(out)


def jackknife(stat, n):
    """Jackknife: recompute the statistic leaving out one benchmark at a time. Used for BCa's a."""
    allidx = np.arange(n)
    return np.array([stat(np.delete(allidx, i)) for i in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", type=Path)
    ap.add_argument("--x86", type=Path)
    ap.add_argument("-B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    benches, modes, t, code, reps = load(args.arm)
    n = len(benches)
    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, n, size=(args.B, n))

    ba = np.array([attain(t, d) for d in draws])
    bc = np.array([ratio(code, d) for d in draws])
    pa, pc = attain(t, np.arange(n)), ratio(code, np.arange(n))
    ja = jackknife(lambda i: attain(t, i), n)
    jc = jackknife(lambda i: ratio(code, i), n)

    rows = []
    print(f"# Benchmark resampling ({n} benches, B={args.B}, BCa 95%)\n")
    print(f"{'mode':<14}{'attainment':>26}{'code ratio':>26}")
    for k, m in enumerate(modes):
        if m not in MODES:
            continue
        lo, hi = bca(pa[k], ba[:, k], ja[:, k])
        clo, chi = bca(pc[k], bc[:, k], jc[:, k])
        print(f"{LABEL[m]:<14}{pa[k]:>10.3f} [{lo:.3f}, {hi:.3f}]{pc[k]:>10.3f} [{clo:.3f}, {chi:.3f}]")
        rows.append({"mode": m, "attainment": pa[k], "att_ci": [lo, hi],
                     "code_ratio": pc[k], "code_ci": [clo, chi]})

    # Differences between modes: subtract within the same resample.
    print("\n# Paired differences (same resample)")
    pairs = [("O2", "tierup"), ("O3", "tierup"), ("O2", "O1"), ("O3", "O2"),
             ("oracle", "O3")]
    diffs = {}
    for x, y in pairs:
        i, j = modes.index(x), modes.index(y)
        dpt = pa[i] - pa[j]
        dboot = ba[:, i] - ba[:, j]
        djack = ja[:, i] - ja[:, j]
        lo, hi = bca(dpt, dboot, djack)
        cpt = pc[i] - pc[j]
        clo, chi = bca(cpt, bc[:, i] - bc[:, j], jc[:, i] - jc[:, j])
        print(f"{LABEL[x]} - {LABEL[y]:<10}  att {dpt:+.3f} [{lo:+.3f}, {hi:+.3f}]"
              f"   code {cpt:+.3f} [{clo:+.3f}, {chi:+.3f}]")
        diffs[f"{x}-{y}"] = {"att": dpt, "att_ci": [lo, hi],
                             "code": cpt, "code_ci": [clo, chi]}

    # Measurement noise only: redraw the 10 repeats and recompute the median.
    print("\n# Repeat resampling (measurement noise only, same 55)")
    rb = np.empty((min(args.B, 2000), len(modes)))
    allidx = np.arange(n)
    for s in range(rb.shape[0]):
        tt = np.array([[np.median(rng.choice(reps[b][m], size=reps[b][m].size))
                        for m in range(len(modes))] for b in range(n)])
        rb[s] = attain(tt, allidx)
    for k, m in enumerate(modes):
        if m not in MODES:
            continue
        lo, hi = np.percentile(rb[:, k], [2.5, 97.5])
        print(f"{LABEL[m]:<14}{pa[k]:>10.3f} [{lo:.3f}, {hi:.3f}]  (width {hi-lo:.4f})")

    out = {"n": n, "B": args.B, "seed": args.seed, "arm": rows, "diffs": diffs,
           "rep_noise": {modes[k]: list(np.percentile(rb[:, k], [2.5, 97.5]))
                         for k in range(len(modes)) if modes[k] in MODES}}

    # Differences between machines are also paired within the same resample.
    if args.x86:
        b2, m2, t2, c2, _ = load(args.x86)
        if b2 != benches:
            raise SystemExit("the two machines have different benchmark lists")
        pa2 = attain(t2, allidx)
        ba2 = np.array([attain(t2, d) for d in draws])
        ja2 = jackknife(lambda i: attain(t2, i), n)
        print("\n# Between machines (AArch64 - x86-64, paired by benchmark)")
        tr = {}
        for k, m in enumerate(modes):
            if m not in ("O1", "O2", "O3", "tierup"):
                continue
            d = pa[k] - pa2[k]
            lo, hi = bca(d, ba[:, k] - ba2[:, k], ja[:, k] - ja2[:, k])
            print(f"{LABEL[m]:<14}{pa[k]:>8.3f}{pa2[k]:>9.3f}{d:>+9.3f} [{lo:+.3f}, {hi:+.3f}]")
            tr[m] = {"arm": pa[k], "x86": pa2[k], "diff": d, "ci": [lo, hi]}
        out["transfer"] = tr

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
