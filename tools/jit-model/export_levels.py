#!/usr/bin/env python3
"""최적화 단계마다 하나씩, 결정 트리 세 개를 학습해 JITModelData.h 로 내보낸다.

각 단계는 theta 기준으로 라벨링된 학습 집합 하나에 대응한다(labeling.py 참고).
-O1 은 가장 공격적으로 메모리를 절약하는 단계이므로 theta 가 가장 크고, -O3 이
가장 작다. 런타임은 선택된 단계의 트리로 hot/cold 만 판정해서 바로 컴파일한다.

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
                    metavar="NAME=DIR", help="예: O1=profile/train_t4.0")
    ap.add_argument("--test-dir", default=None,
                    help="같은 단계 이름의 테스트 디렉터리 접두사 (예: profile/test_t)")
    ap.add_argument("--max-depth", type=int, default=12)
    ap.add_argument("--min-samples-leaf", type=int, default=2)
    ap.add_argument("--ccp-alpha", type=float, default=0.0005)
    ap.add_argument("--hot-weight", type=float, default=4.0)
    ap.add_argument("--bench-weight", choices=["none", "equal"], default="none",
                    help="equal: 벤치마크마다 함수 무게의 합을 같게 한다 (함수가 많은 모듈이 학습을 지배하지 않게)")
    ap.add_argument("--features", default=None,
                    help="쉼표로 구분한 특징 이름. 주지 않으면 전부 쓴다")
    ap.add_argument("--wide-thresholds", action="store_true",
                    help="임계값을 int32_t 로 내보낸다 (특징을 뺀 실험용 모델)")
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
            # 잎에서는 왼쪽 자리에 예측 클래스를 넣는다. 내부 노드에서는
            # 자식 인덱스인데, 트리 세 개를 이어 붙였으므로 base 를 더한다.
            hot = val[i][1] / max(val[i].sum(), 1e-12) > 0.5
            left.append(int(hot) if leaf else int(t.children_left[i]) + base)
            right.append(0 if leaf else int(t.children_right[i]) + base)
        print(f"  {name}: 학습 {len(kept)}벤치, 함수 {len(y)}개 중 hot {int(y.sum())}, "
              f"노드 {n}")

    total = len(feat)
    # 배포하는 모델의 임계값은 int16 에 들어간다. 특징을 빼고 학습한 실험용 모델은
    # 남은 특징의 큰 값으로 쪼개려 해서 넘칠 수 있으므로, --wide-thresholds 로
    # 넓힌다. 값이 int16 에 들어가는 모델이라면 두 폭은 같은 트리를 만든다.
    thr_type = "int32_t" if args.wide_thresholds else "int16_t"
    if not args.wide_thresholds:
        bad = [v for v in thr if not (-32768 <= v <= 32767)]
        if bad:
            raise OverflowError(f"int16 범위를 넘는 값이 있다: {bad[:3]}")
    for arr in (left, right):
        bad = [v for v in arr if not (-32768 <= v <= 32767)]
        if bad:
            raise OverflowError(f"자식 색인이 int16 범위를 넘는다: {bad[:3]}")

    names = ",\n    ".join(f'"{k}"' for k in FEATURE_NAMES)
    roots_s = ", ".join(str(r) for r in roots)
    level_names = ", ".join(f'"{n}"' for n, _ in levels)
    content = f"""// tools/jit-model/export_levels.py 가 생성한다. 직접 고치지 말 것.
#pragma once

#include <cstdint>

namespace Walrus {{
namespace JITPredictorModel {{

constexpr int kFeatureCount = {len(FEATURE_NAMES)};
// 최적화 단계 수. 트리 하나가 단계 하나를 담당하며, 모든 트리가 아래 배열에
// 이어 붙어 있다. kLevelRoot 가 각 트리의 뿌리 노드 위치다.
constexpr int kLevelCount = {len(levels)};
constexpr int kNodeCount = {total};
constexpr const char* kLevelNames[kLevelCount] = {{ {level_names} }};
constexpr int16_t kLevelRoot[kLevelCount] = {{ {roots_s} }};

// 학습에 쓴 특징 순서.
constexpr const char* kFeatureNames[kFeatureCount] = {{
    {names},
}};

// 노드가 검사하는 특징 번호. 잎이면 -1.
{fmt("kNodeFeature", "int8_t", feat)}
// 임계값의 내림값. 특징이 전부 정수라 정수 비교로 같은 결과가 나온다.
{fmt("kNodeThreshold", thr_type, thr)}
// 내부 노드에서는 왼쪽 자식의 위치(특징값 <= 임계값). 잎에서는 예측 결과(0 또는 1).
{fmt("kNodeLeft", "int16_t", left)}
{fmt("kNodeRight", "int16_t", right)}
}} // namespace JITPredictorModel
}} // namespace Walrus
"""
    Path(args.out).write_text(content)
    print(f"{args.out} 작성 ({total} 노드, 단계 {len(levels)}개)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
