#!/usr/bin/env python3
"""학습셋 안에서만 특징 집합과 트리 설정을 고른다 (모듈 단위 group k-fold).

held-out 테스트셋은 읽지 않는다. 같은 프로그램이 여러 스위트에 있으면
(beebs_crc32 / embench_crc32, Misc_flops-1 / flops-3 …) 한 묶음으로 같은 fold 에 넣는다.

점수는 정확도가 아니라 논문의 두 축이다. 각 단계 n (θ = 4, 2, 1)의 -On 예측
(1..n 단계 트리의 합집합)을 실측과 같은 저울(eval_metrics.Pool)로 채점해

    효용 U_n = 시간 달성도 − θ_n · 코드 비율

을 구한다. θ 라벨(시간 몫 ≥ θ·기계어 몫)이 바로 이 효용을 함수마다 최대로 만드는
선택이므로, 같은 fold 에서 라벨 자체의 효용(오라클)으로 나눈 값을 쓴다.
설정 하나의 점수는 반복마다 모든 fold 의 out-of-fold 예측을 모아 한 번 채점한
U_n / U_n(오라클)의 세 단계 평균이고, 반복들의 평균을 보고한다.

    tune_cv.py run  --baselines B.json --out cv.json [--repeats 3 --jobs 12]
    tune_cv.py pick cv.json        1-SE 규칙으로 고른 설정을 보여 준다
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_metrics as M  # noqa: E402
from jit_decision_tree import FEATURE_NAMES  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
LEVELS = [("O1", 4.0, "profile/train5_t4.0"), ("O2", 2.0, "profile/train5_t2.0"),
          ("O3", 1.0, "profile/train5_t1.0")]

BODY = ["local_count", "branch_count", "call_indirect_count", "is_leaf_function",
        "max_own_loop_depth"]
DIRECT = ["call_site_count", "caller_count", "caller_in_loop_count", "max_caller_loop_depth",
          "call_graph_depth", "call_site_work"]
EST = ["exec_count_sweep_log2", "exec_count_rounds_log2", "exec_work_rounds_log2"]
INDIRECT = ["indirect_caller_count", "max_indirect_caller_loop_depth"]
FEATURE_SETS = {
    "body5": BODY,
    "direct11": BODY + DIRECT,
    "est14": BODY + DIRECT + EST,
    "all16": BODY + DIRECT + EST + INDIRECT,
    "prev14": [n for n in FEATURE_NAMES if n not in ("call_site_count", "exec_count_rounds_log2")],
}
GRID = {
    "max_depth": [4, 6, 8, 10, 12, 16, None],
    "min_samples_leaf": [1, 2, 5, 10, 20],
    "ccp_alpha": [0.0, 1e-4, 3e-4, 1e-3],
    "hot_weight": [1, 2, 4, 8, 16, 32],
    "bench_weight": ["none", "equal"],
}


def program_key(bench: str) -> str:
    base = bench.split("_", 1)[1] if "_" in bench else bench
    base = re.sub(r"-\d+$", "", base)
    return re.sub(r"[^a-z0-9]", "", base.lower())


def bench_weights(groups: np.ndarray, mode: str):
    if mode == "none":
        return None
    _, inv, cnt = np.unique(groups, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    return w * len(w) / w.sum()


def folds(groups: np.ndarray, k: int, seed: int):
    keys = np.array([program_key(g) for g in groups])
    uniq = np.unique(keys)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    fold_of = {uniq[j]: i % k for i, j in enumerate(order)}
    f = np.array([fold_of[x] for x in keys])
    return [np.where(f == i)[0] for i in range(k)]


def load(baselines: Path, codesize: Path):
    w = M.Weighting(baselines, codesize)
    X = pool = None
    Y = []
    for _, _, d in LEVELS:
        Xi, yi, pi = M.load_pool(REPO / d, w)
        if X is None:
            X, pool = Xi, pi
        elif not np.array_equal(X, Xi):
            raise RuntimeError("단계마다 특징 행이 다르다")
        Y.append(yi)
    return X, Y, pool


def evaluate(cfg, X, Y, pool, fold_sets):
    cols = [FEATURE_NAMES.index(n) for n in FEATURE_SETS[cfg["features"]]]
    Xs = X[:, cols]
    groups = pool.groups
    per_rep, nodes = [], []
    detail = None
    for fs in fold_sets:
        oof = [np.zeros(len(Xs), dtype=bool) for _ in LEVELS]
        for val in fs:
            tr = np.setdiff1d(np.arange(len(Xs)), val)
            sw = bench_weights(groups[tr], cfg["bench_weight"])
            n_tot = 0
            for li, y in enumerate(Y):
                m = DecisionTreeClassifier(
                    max_depth=cfg["max_depth"], min_samples_leaf=cfg["min_samples_leaf"],
                    ccp_alpha=cfg["ccp_alpha"], class_weight={0: 1, 1: cfg["hot_weight"]},
                    random_state=0).fit(Xs[tr], y[tr], sample_weight=sw)
                oof[li][val] = m.predict(Xs[val]).astype(bool)
                n_tot += m.tree_.node_count
            nodes.append(n_tot)
        union = np.zeros(len(Xs), dtype=bool)
        lab = np.zeros(len(Xs), dtype=bool)
        ratios, rep_detail = [], []
        for li, (name, theta, _) in enumerate(LEVELS):
            union |= oof[li]
            lab |= Y[li].astype(bool)
            s = pool.score(union, theta)
            o = pool.score(lab, theta)
            ratios.append(s["util"] / o["util"] if o["util"] > 0 else 0.0)
            rep_detail.append({"level": name, "time": s["time"], "code": s["code"],
                               "oracle_time": o["time"], "oracle_code": o["code"]})
        per_rep.append(float(np.mean(ratios)))
        detail = detail or rep_detail
    return {**cfg, "score": float(np.mean(per_rep)), "score_reps": per_rep,
            "nodes": float(np.mean(nodes)), "levels_rep0": detail}


def cmd_run(args) -> int:
    X, Y, pool = load(Path(args.baselines), Path(args.codesize))
    missing = pool.unknown()
    if missing:
        print(f"기준선이나 코드 크기가 없어 채점에서 빠지는 벤치마크 {len(missing)}개: {missing}")
    keys = {program_key(g) for g in pool.benches}
    print(f"학습 함수 {len(X)}개, 벤치마크 {len(pool.benches)}개, 프로그램 묶음 {len(keys)}개")
    fold_sets = [folds(pool.groups, args.k, seed) for seed in range(args.repeats)]
    cfgs = [dict(zip(["features", *GRID], v))
            for v in itertools.product(args.feature_sets or list(FEATURE_SETS), *GRID.values())]
    if args.limit:
        cfgs = cfgs[: args.limit]
    print(f"설정 {len(cfgs)}개 × {args.repeats}회 × {args.k}-fold × 단계 3")
    res = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(evaluate)(c, X, Y, pool, fold_sets) for c in cfgs)
    Path(args.out).write_text(json.dumps({"k": args.k, "repeats": args.repeats,
                                          "results": res}, indent=1))
    print(f"{args.out} 작성")
    return 0


def cmd_pick(args) -> int:
    d = json.loads(Path(args.cv).read_text())
    res = d["results"]
    for r in res:
        r["se"] = float(np.std(r["score_reps"], ddof=1) / np.sqrt(len(r["score_reps"]))) \
            if len(r["score_reps"]) > 1 else 0.0
    best = max(res, key=lambda r: r["score"])
    within = [r for r in res if r["score"] >= best["score"] - best["se"]]
    simple = min(within, key=lambda r: (len(FEATURE_SETS[r["features"]]), r["nodes"]))
    fmt = lambda r: (f"{r['features']:<9} depth={str(r['max_depth']):<4} leaf={r['min_samples_leaf']:<2} "
                     f"α={r['ccp_alpha']:<6} hw={r['hot_weight']:<2} bw={r['bench_weight']:<5} "
                     f"점수 {r['score']:.4f}±{r['se']:.4f} 노드 {r['nodes']:.0f}")
    print("최고점:  " + fmt(best))
    print(f"1-SE 안의 설정 {len(within)}개 중 가장 단순한 것:")
    print("  선택:  " + fmt(simple))
    for lv in simple["levels_rep0"]:
        print(f"    {lv['level']}: 시간 {lv['time']:.3f} 코드 {lv['code']:.3f} "
              f"(오라클 {lv['oracle_time']:.3f}/{lv['oracle_code']:.3f})")
    print("\n특징 집합별 최고점:")
    for fs in FEATURE_SETS:
        rs = [r for r in res if r["features"] == fs]
        if rs:
            print("  " + fmt(max(rs, key=lambda r: r["score"])))
    print("\n상위 10개:")
    for r in sorted(res, key=lambda r: -r["score"])[:10]:
        print("  " + fmt(r))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--baselines", required=True)
    r.add_argument("--codesize", default=str(REPO / "profile/codesize5"))
    r.add_argument("--out", required=True)
    r.add_argument("--k", type=int, default=5)
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--jobs", type=int, default=12)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--feature-sets", nargs="*")
    p = sub.add_parser("pick")
    p.add_argument("cv")
    args = ap.parse_args()
    return cmd_run(args) if args.cmd == "run" else cmd_pick(args)


if __name__ == "__main__":
    sys.exit(main())
