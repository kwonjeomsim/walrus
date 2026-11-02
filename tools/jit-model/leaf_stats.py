#!/usr/bin/env python3
"""Training statistics of the leaf each given function reaches, per level.

    leaf_stats.py --features feats.tsv --funcs 58,60 [--names map.txt]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_metrics as M  # noqa: E402
from jit_decision_tree import FEATURE_NAMES  # noqa: E402
from trace_tree import arr  # noqa: E402
from tune_cv import LEVELS, REPO  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path(__file__).resolve().parents[2] / "src/jit/JITModelData.h"))
    ap.add_argument("--features", required=True)
    ap.add_argument("--funcs", required=True)
    ap.add_argument("--names")
    ap.add_argument("--baselines", default=str(Path(__file__).resolve().parents[2] / "profile/baselines_train232.json"))
    a = ap.parse_args()
    src = Path(a.model).read_text()
    fn = arr(src, "kFeatureNames", str)
    feat, thr, left, right, roots = (arr(src, k) for k in ("kNodeFeature", "kNodeThreshold", "kNodeLeft", "kNodeRight", "kLevelRoot"))

    def leaf(get, L):
        n = roots[L]
        while feat[n] >= 0:
            n = left[n] if get(fn[feat[n]]) <= thr[n] else right[n]
        return n
    w = M.Weighting(Path(a.baselines), REPO / "profile/codesize6")
    pools = []
    for _, _, d in LEVELS:
        X, y, pool = M.load_pool(REPO / d, w)
        pools.append((X, y.astype(bool), pool))
    X, _, pool = pools[0]
    col = {n: FEATURE_NAMES.index(n) for n in fn if n in FEATURE_NAMES}
    leaves = [np.array([leaf(lambda f, r=r: r[col[f]], L) for r in X]) for L in range(3)]
    y1 = pools[2][1]
    rows = Path(a.features).read_text().splitlines()
    cols = rows[0].split("\t")
    F = {int(r.split("\t")[0]): dict(zip(cols, map(int, r.split("\t")))) for r in rows[1:] if r.strip()}
    names = {}
    if a.names:
        names = dict((int(i), n) for i, n in (l.split(":", 1) for l in Path(a.names).read_text().splitlines() if ":" in l))
    print(f"training functions {len(X)}, profitable at theta=1 {y1.sum()}")
    for i in map(int, a.funcs.split(",")):
        cells = []
        for L in range(3):
            n = leaf(lambda f: F[i][f], L)
            m = leaves[L] == n
            yL = pools[L][1]
            cells.append(f"O{L + 1} leaf {n:3d} {'HOT ' if left[n] == 1 else 'cold'} n={m.sum():5d} "
                         f"prof@lvl {yL[m].sum():3d} prof@1 {y1[m].sum():3d} bench {len(set(pool.groups[m])):3d}")
        print(f"{i:4d} {names.get(i, '')[:24]:24} " + " | ".join(cells))


if __name__ == "__main__":
    main()
