#!/usr/bin/env python3
"""Separately trained levels against one tree cut at several probabilities (3.4 footnote).

Cross-validated inside the 232 training benchmarks with the folds and the
configuration of the deployed model (tune_cv.py, direct features, depth 6,
20 samples per leaf, alpha 3e-4, hot weight 2, equal benchmark weights).

    separate  the deployed construction: one tree per theta, -On = union of 1..n
    single    one tree trained at one theta; its out-of-fold leaf probabilities
              are cut so that the selection emits the code ratio of each level

Both are scored offline with eval_metrics.Pool (the cross-validation score of
Section 3.3) and, for reference, with a geometric-mean estimate in which a
benchmark's run time falls by the self time its compiled functions cover.

    WALRUS_BIN=<tier-up-v6 build> single_tree_cv.py --baselines B.json
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
from tune_cv import LEVELS, REPO, bench_weights, folds  # noqa: E402

# Training column order of the deployed model (retrain_v6.sh).
FEATS = ["local_count", "branch_count", "call_indirect_count", "is_leaf_function",
         "max_own_loop_depth", "call_site_count", "caller_in_loop_count",
         "max_caller_loop_depth", "call_graph_depth", "call_site_work"]
CFG = dict(max_depth=6, min_samples_leaf=20, ccp_alpha=3e-4)
HOT_WEIGHT = 2


def gm_attain(pool, pred):
    """Geometric-mean attainment, estimating each benchmark's time from coverage."""
    num = den = 0.0
    for b in pool.benches:
        m = pool._mask[b]
        ti, tj = pool.w.base[b]["interp_s"], pool.w.base[b]["alljit_s"]
        if tj >= ti:
            continue
        ts = pool._tsum[b]
        cov = float(pool.tfrac[m][pred[m]].sum() / ts) if ts > 0 else 0.0
        tm = ti - cov * (ti - tj)
        num += np.log(ti / tm)
        den += np.log(ti / tj)
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--codesize", default=str(REPO / "profile/codesize6"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()

    w = M.Weighting(Path(a.baselines), Path(a.codesize))
    X = pool = None
    Y = []
    for _, _, d in LEVELS:
        Xi, yi, pi = M.load_pool(REPO / d, w)
        X, pool = (Xi, pi) if X is None else (X, pool)
        Y.append(yi.astype(bool))
    Xs = X[:, [FEATURE_NAMES.index(n) for n in FEATS]]
    n = len(Xs)
    groups = pool.groups

    res = {"separate": [], **{f"single_t{th:g}": [] for _, th, _ in LEVELS}}
    for seed in range(a.repeats):
        fs = folds(groups, a.k, seed)
        oof = [np.zeros(n, bool) for _ in LEVELS]
        prob = [np.zeros(n) for _ in LEVELS]
        for val in fs:
            tr = np.setdiff1d(np.arange(n), val)
            sw = bench_weights(groups[tr], "equal")
            for li, y in enumerate(Y):
                m = DecisionTreeClassifier(**CFG, class_weight={0: 1, 1: HOT_WEIGHT},
                                           random_state=0).fit(Xs[tr], y[tr], sample_weight=sw)
                oof[li][val] = m.predict(Xs[val]).astype(bool)
                prob[li][val] = m.predict_proba(Xs[val])[:, 1]
        union = np.zeros(n, bool)
        sep = []
        for li in range(len(LEVELS)):
            union |= oof[li]
            s = pool.score(union)
            sep.append((s["code"], s["time"], gm_attain(pool, union)))
        res["separate"].append(sep)
        for li, (_, th, _) in enumerate(LEVELS):
            p = prob[li]
            cuts = np.unique(p)[::-1]
            pts = []
            for c in cuts:
                sel = p >= c
                s = pool.score(sel)
                pts.append((s["code"], s["time"], sel))
            row = []
            for code, _, _ in sep:
                # largest selection whose code does not exceed the level's code
                ok = [q for q in pts if q[0] <= code + 1e-12]
                q = max(ok, key=lambda q: q[0]) if ok else (0.0, 0.0, np.zeros(n, bool))
                row.append((q[0], q[1], gm_attain(pool, q[2])))
            res[f"single_t{th:g}"].append(row)

    print(f"cross-validated, {a.k}-fold x {a.repeats}, 232 training benchmarks")
    print(f"{'construction':16}" + "".join(f"{lv:>30}" for lv, _, _ in LEVELS))
    print(f"{'':16}" + "".join(f"{'code / offline / gm':>30}" for _ in LEVELS))
    for name, rows in res.items():
        r = np.array(rows).mean(axis=0)
        print(f"{name:16}" + "".join(f"{c:>12.3f}{t:>9.3f}{g:>9.3f}" for c, t, g in r))
    base = np.array(res["separate"])
    print("\nsingle minus separate, offline attainment: mean (min, max over repeats)")
    for name, rows in res.items():
        if name == "separate":
            continue
        d = np.array(rows)[:, :, 1] - base[:, :, 1]
        print(f"{name:16}" + "".join(f"   {d[:, i].mean():+.3f} ({d[:, i].min():+.3f}, {d[:, i].max():+.3f})"
                                    for i in range(d.shape[1])))


if __name__ == "__main__":
    main()
