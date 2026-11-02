#!/usr/bin/env python3
"""The three levels one and two doublings lower (Section 3.4).

Cross-validated inside the 232 training benchmarks with the folds and the
configuration of the deployed model. Each ladder trains one tree per theta and
takes -On as the union of levels 1..n, exactly as deployed; the ladders differ
only in their thresholds. Scored offline (eval_metrics.Pool, the score of
Section 3.3) and with the geometric-mean estimate of single_tree_cv.py.

    WALRUS_BIN=<tier-up-v6 build> theta_lower_cv.py --baselines B.json \
        --labels 0.5=/tmp/theta/train6_t0.5 --labels 0.25=/tmp/theta/train6_t0.25
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_metrics as M  # noqa: E402
from jit_decision_tree import FEATURE_NAMES  # noqa: E402
from single_tree_cv import CFG, FEATS, HOT_WEIGHT, gm_attain  # noqa: E402
from tune_cv import REPO, bench_weights, folds  # noqa: E402

LADDERS = [(4.0, 2.0, 1.0), (2.0, 1.0, 0.5), (1.0, 0.5, 0.25)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--codesize", default=str(REPO / "profile/codesize6"))
    ap.add_argument("--labels", action="append", default=[], metavar="THETA=DIR",
                    help="label directories beyond profile/train6_t{4,2,1}")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()

    dirs = {th: REPO / f"profile/train6_t{th}" for th in (4.0, 2.0, 1.0)}
    for spec in a.labels:
        th, _, d = spec.partition("=")
        dirs[float(th)] = Path(d)
    w = M.Weighting(Path(a.baselines), Path(a.codesize))
    X = pool = None
    Y = {}
    for th, d in dirs.items():
        Xi, yi, pi = M.load_pool(d, w)
        if X is None:
            X, pool = Xi, pi
        elif not np.array_equal(X, Xi):
            raise RuntimeError(f"feature rows differ for theta {th}")
        Y[th] = yi.astype(bool)
    Xs = X[:, [FEATURE_NAMES.index(n) for n in FEATS]]
    n = len(Xs)
    groups = pool.groups

    res = {lad: [] for lad in LADDERS}
    for seed in range(a.repeats):
        fs = folds(groups, a.k, seed)
        oof = {th: np.zeros(n, bool) for th in Y}
        for val in fs:
            tr = np.setdiff1d(np.arange(n), val)
            sw = bench_weights(groups[tr], "equal")
            for th, y in Y.items():
                m = DecisionTreeClassifier(**CFG, class_weight={0: 1, 1: HOT_WEIGHT},
                                           random_state=0).fit(Xs[tr], y[tr], sample_weight=sw)
                oof[th][val] = m.predict(Xs[val]).astype(bool)
        for lad in LADDERS:
            union = np.zeros(n, bool)
            row = []
            for th in lad:
                union |= oof[th]
                s = pool.score(union)
                row.append((s["code"], s["time"], gm_attain(pool, union)))
            res[lad].append(row)

    print(f"cross-validated, {a.k}-fold x {a.repeats}, 232 training benchmarks")
    print(f"{'thetas':18}" + "".join(f"{lv:>28}" for lv in ("-O1", "-O2", "-O3")))
    print(f"{'':18}" + "".join(f"{'code / offline / gm':>28}" for _ in range(3)))
    for lad, rows in res.items():
        r = np.array(rows).mean(axis=0)
        print(f"{str(lad):18}" + "".join(f"{c:>10.3f}{t:>9.3f}{g:>9.3f}" for c, t, g in r))
    base = np.array(res[LADDERS[0]])
    print("\ntop level against the deployed ladder: d code, d offline, d gm (min..max of d offline)")
    for lad in LADDERS[1:]:
        d = np.array(res[lad])[:, 2, :] - base[:, 2, :]
        print(f"{str(lad):18} {d[:, 0].mean():+.3f} {d[:, 1].mean():+.3f} {d[:, 2].mean():+.3f}"
              f"   ({d[:, 1].min():+.3f}..{d[:, 1].max():+.3f})")


if __name__ == "__main__":
    main()
