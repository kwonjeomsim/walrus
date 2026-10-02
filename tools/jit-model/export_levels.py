#!/usr/bin/env python3
"""Train one decision tree per optimization level (three in total) and export them to JITModelData.h.

Each level corresponds to one training set labeled with a theta (see labeling.py).
-O1 saves memory most aggressively, so its theta is the largest, and -O3's is the
smallest. The runtime classifies hot/cold with the selected level's tree and compiles right away.

    export_levels.py --level-dir O1=profile/train_t4.0 --level-dir O2=... --level-dir O3=...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset as D
from jit_decision_tree import FEATURE_NAMES

HEADER_PATH = Path(__file__).resolve().parents[2] / "src/jit/JITModelData.h"


def fmt(name: str, ctype: str, arr) -> str:
    body = ",\n    ".join(str(int(v)) for v in arr)
    return f"constexpr {ctype} {name}[kNodeCount] = {{\n    {body},\n}};"


def bench_weights(groups, mode):
    if mode == "none":
        return None
    g = np.asarray(groups)
    _, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    return w * len(w) / w.sum()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level-dir", action="append", required=True,
                    metavar="NAME=DIR", help="e.g. O1=profile/train_t4.0")
    ap.add_argument("--test-dir", default=None,
                    help="test directory prefix for the same level names (e.g. profile/test_t)")
    ap.add_argument("--max-depth", type=int, default=12)
    ap.add_argument("--min-samples-leaf", type=int, default=2)
    ap.add_argument("--ccp-alpha", type=float, default=0.0005)
    ap.add_argument("--hot-weight", type=float, default=4.0)
    ap.add_argument("--bench-weight", choices=["none", "equal"], default="none",
                    help="equal: give every benchmark the same total function weight (so modules with many functions do not dominate)")
    ap.add_argument("--features", default=None,
                    help="comma-separated feature names. All features if omitted")
    ap.add_argument("--wide-thresholds", action="store_true",
                    help="export thresholds as int32_t (for feature-removed experimental models)")
    ap.add_argument("--out", default=str(HEADER_PATH))
    args = ap.parse_args()

    levels = []
    for spec in args.level_dir:
        name, _, d = spec.partition("=")
        levels.append((name, Path(d)))

    use = args.features.split(",") if args.features else list(FEATURE_NAMES)
    cols = [FEATURE_NAMES.index(n) for n in use]

    feat, thr, left, right, roots = [], [], [], [], []
    for name, d in levels:
        X, y, tfrac, bsize, groups, kept, dropped = D.load_dataset(
            d, glob_pat="*.txt", verbose=False, tag=name)
        m = DecisionTreeClassifier(max_depth=args.max_depth,
                                   min_samples_leaf=args.min_samples_leaf,
                                   ccp_alpha=args.ccp_alpha,
                                   class_weight={0: 1, 1: args.hot_weight},
                                   random_state=0).fit(X[:, cols], y, sample_weight=bench_weights(groups, args.bench_weight))
        t = m.tree_
        base = len(feat)
        roots.append(base)
        n = int(t.node_count)
        val = t.value.reshape(n, -1)
        for i in range(n):
            f = int(t.feature[i])
            leaf = f < 0
            feat.append(-1 if leaf else cols[f])
            thr.append(0 if leaf else int(np.floor(t.threshold[i])))
            # At a leaf, the left slot holds the predicted class. At an internal node it is
            # a child index, and since the three trees are concatenated, base is added.
            hot = val[i][1] / max(val[i].sum(), 1e-12) > 0.5
            left.append(int(hot) if leaf else int(t.children_left[i]) + base)
            right.append(0 if leaf else int(t.children_right[i]) + base)
        print(f"  {name}: trained on {len(kept)} benches, hot {int(y.sum())} of {len(y)} functions, "
              f"{n} nodes")

    total = len(feat)
    # Thresholds of the shipped model fit in int16. Experimental models trained without some
    # features may split on large values of the remaining ones and overflow, so
    # --wide-thresholds widens them. For a model whose values fit in int16 both widths give the same tree.
    thr_type = "int32_t" if args.wide_thresholds else "int16_t"
    if not args.wide_thresholds:
        bad = [v for v in thr if not (-32768 <= v <= 32767)]
        if bad:
            raise OverflowError(f"values exceed the int16 range: {bad[:3]}")
    for arr in (left, right):
        bad = [v for v in arr if not (-32768 <= v <= 32767)]
        if bad:
            raise OverflowError(f"child indices exceed the int16 range: {bad[:3]}")

    names = ",\n    ".join(f'"{k}"' for k in FEATURE_NAMES)
    roots_s = ", ".join(str(r) for r in roots)
    level_names = ", ".join(f'"{n}"' for n, _ in levels)
    content = f"""// Generated by tools/jit-model/export_levels.py. Do not edit.
#pragma once

#include <cstdint>

namespace Walrus {{
namespace JITPredictorModel {{

constexpr int kFeatureCount = {len(FEATURE_NAMES)};
// Number of optimization levels. Each tree handles one level, and all trees are
// concatenated in the arrays below. kLevelRoot gives the root node of each tree.
constexpr int kLevelCount = {len(levels)};
constexpr int kNodeCount = {total};
constexpr const char* kLevelNames[kLevelCount] = {{ {level_names} }};
constexpr int16_t kLevelRoot[kLevelCount] = {{ {roots_s} }};

// Feature order used in training.
constexpr const char* kFeatureNames[kFeatureCount] = {{
    {names},
}};

// Feature index tested by the node. -1 for a leaf.
{fmt("kNodeFeature", "int8_t", feat)}
// Floor of the threshold. All features are integers, so integer comparison gives the same result.
{fmt("kNodeThreshold", thr_type, thr)}
// Internal node: index of the left child (feature <= threshold). Leaf: the prediction (0 or 1).
{fmt("kNodeLeft", "int16_t", left)}
{fmt("kNodeRight", "int16_t", right)}
}} // namespace JITPredictorModel
}} // namespace Walrus
"""
    Path(args.out).write_text(content)
    print(f"wrote {args.out} ({total} nodes, {len(levels)} levels)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
