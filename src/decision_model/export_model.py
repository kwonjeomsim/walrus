#!/usr/bin/env python3
"""Train the Walrus JIT-priority decision tree on labeled profiles
and emit the C++ header that the runtime compiles into the JIT predictor.

Fixed data layout (per project policy):
    Train: walrus/tmp/profile_log/*.txt              (CHStone loop/ +
           CppBenchmark + mibench perf/ + PolyBenchC LARGE; labeled by
           walrus/tmp/profile.py's combined call/time/RSS rule)
    Test:  ~/projects/wabench/walrus_profile/*.txt   (wabench held-out,
           labeled by the same rule)

Pipeline:
  1. Read every labeled profile log under PROFILE_DIR. Filter by run quality:
     keep rc==0 with wall >= MIN_WALL_S, plus rc==124 timeouts (the partial
     dump still carries useful counter data). Drop fast / trapped / errored
     runs.
  2. Extract static features (jit_decision_tree.extract_features_from_wasm)
     from the wasm path stored in each log's `# source` header line.
  3. Read the per-function `label` column (set upstream by tmp/profile.py's
     call-count rule); align to features by wasm function index.
  4. 5-fold StratifiedGroupKFold CV (one fold per benchmark group); fit the
     final model on the full pool; emit JITModelData.h plus figures.

Policy knobs:
  --hot-weight W       class_weight for the hot class. High W (e.g. 22) is
                       recall-first / aggressive; low W (1) or
                       "balanced"/"none" is precision-first / conservative.
  --min-samples-leaf,  tree-complexity knobs; larger values = simpler trees.
  --ccp-alpha
  --emit / --no-emit   write JITModelData.h, or just report CV (sweeps).

Run:
    python src/decision_model/export_model.py [--hot-weight 22] [--no-emit]
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import (
    DecisionTreeClassifier,
    export_graphviz,
    export_text,
    plot_tree,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jit_decision_tree import (  # noqa: E402
    FEATURE_NAMES,
    extract_features_from_wasm,
)

WALRUS_ROOT = Path(__file__).resolve().parents[2]
HEADER_PATH = WALRUS_ROOT / "src/jit/JITModelData.h"
FIG_DIR = WALRUS_ROOT / "src/decision_model/figures"

# Fixed training source. Each <stem>.txt is a per-benchmark labeled profile
# written by walrus/tmp/profile.py (combined call/time/RSS rule). The pool
# combines CHStone loop/ + CppBenchmark + mibench perf/ + PolyBenchC LARGE,
# covering compute-heavy kernels, microbenchmark-style modules, and steady-
# state hot loops.
PROFILE_DIR = WALRUS_ROOT / "tmp/profile_log"
# Fixed held-out test source. Same labeled file format, different wasm suite
# (wabench: JetStream2 + MiBench + PolyBench + Whole_Applications) so the
# overlap with the training pool is benchmark-level zero — a real
# generalization check.
TEST_PROFILE_DIR = Path("/home/soonbkwon/projects/wabench/walrus_profile")

# Wall-time floor for the "completed cleanly" filter. Runs shorter than this
# leave the JIT no room to amortize compile cost, so their labels are noise.
MIN_WALL_S = 0.5


# ---------------------------------------------------------------------------
# Profile-log parsing
# ---------------------------------------------------------------------------

# Each labeled row carries:
#   function module index call_count loop_count bytecode_sz \
#       cum_time_ns peak_rss_b time_frac rss_frac label
# time_frac / rss_frac are emitted as 0.0 under the call-count labeling rule;
# they're kept in the dataclass only for log-format compatibility.
@dataclass
class ProfileEntry:
    call_count: int
    loop_count: int
    bytecode_sz: int
    cum_time_ns: int
    peak_rss_b: int
    time_frac: float
    rss_frac: float
    label: int


def parse_profile(path: Path) -> Dict[int, ProfileEntry]:
    """Return {walrus_func_index: ProfileEntry} from one labeled log."""
    out: Dict[int, ProfileEntry] = {}
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 11:
                continue
            name = parts[0]
            if name != "func" and not name.startswith("$"):
                continue
            try:
                idx = int(parts[2])
                entry = ProfileEntry(
                    call_count=int(parts[3]),
                    loop_count=int(parts[4]),
                    bytecode_sz=int(parts[5]),
                    cum_time_ns=int(parts[6]),
                    peak_rss_b=int(parts[7]),
                    time_frac=float(parts[8]),
                    rss_frac=float(parts[9]),
                    label=int(parts[10]),
                )
            except ValueError:
                continue
            out[idx] = entry
    return out


def read_header(txt: Path) -> Tuple[Optional[str], Optional[float], Optional[int]]:
    """Pull `# source`, `# wall_s = ... rc=...` from a labeled log."""
    source: Optional[str] = None
    wall_s: Optional[float] = None
    rc: Optional[int] = None
    with open(txt) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("# source"):
                source = s.split("=", 1)[1].strip()
            elif s.startswith("# wall_s"):
                bits = s.split("=", 1)[1].strip().split()
                if bits:
                    try:
                        wall_s = float(bits[0])
                    except ValueError:
                        pass
                for b in bits[1:]:
                    if b.startswith("rc="):
                        try:
                            rc = int(b[3:])
                        except ValueError:
                            pass
            elif not s.startswith("#") and not s.startswith("===="):
                break
    return source, wall_s, rc


def keep_bench(rc: Optional[int], wall_s: Optional[float]) -> bool:
    """Keep clean long-enough runs and SIGTERM-timeouts.

    rc==124 is `timeout` killing the process; the partial dump from walrus's
    atexit hook still has meaningful counters, so the labels are usable.
    """
    if rc is None:
        return False
    if rc == 124:
        return True
    return rc == 0 and (wall_s or 0.0) >= MIN_WALL_S


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def load_dataset(
    profile_dir: Path,
    glob_pat: str = "[0-9]*.txt",
    verbose: bool = True,
    tag: str = "Dataset",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str], List[Tuple[str, Optional[float], Optional[int]]]]:
    """Return (X, y, tfrac, bsize, groups, kept_benches, dropped) from a dir.

    groups[i] = benchmark tag (file stem) so StratifiedGroupKFold keeps
    each wasm in a single fold. `glob_pat` defaults to wabench's numeric-
    prefixed layout; the CppBenchmark test set uses `*.txt` (no prefix).

    tfrac[i] = the function's time_frac (inclusive cum_time / program total).
    bsize[i] = the function's bytecode size (compile-cost proxy). Both feed
    the cost-model sample weights and the time-weighted recall report.
    """
    X_rows: List[List[int]] = []
    y_rows: List[int] = []
    tf_rows: List[float] = []
    bs_rows: List[int] = []
    groups: List[str] = []
    kept_benches: List[str] = []
    dropped: List[Tuple[str, Optional[float], Optional[int]]] = []

    if verbose:
        print(f"[{tag}] reading logs from {profile_dir}\n")
    for txt in sorted(profile_dir.glob(glob_pat)):
        source, wall_s, rc = read_header(txt)
        if not keep_bench(rc, wall_s) or not source or not os.path.exists(source):
            dropped.append((txt.stem, wall_s, rc))
            continue
        profile = parse_profile(txt)
        feats = extract_features_from_wasm(source)
        hot = kept = 0
        for f in feats:
            entry = profile.get(f.index)
            if entry is None:
                continue
            X_rows.append(f.to_vector())
            y_rows.append(entry.label)
            tf_rows.append(entry.time_frac)
            bs_rows.append(entry.bytecode_sz)
            groups.append(txt.stem)
            hot += entry.label
            kept += 1
        kept_benches.append(txt.stem)
        if verbose:
            print(f"  {txt.stem:<46} wall={wall_s if wall_s is not None else 0.0:>7.2f}s "
                  f"rc={rc:<4} funcs={len(feats):>4} kept={kept:>4} hot={hot:>3}")
    X = np.asarray(X_rows, dtype=np.int64)
    y = np.asarray(y_rows, dtype=np.int64)
    tfrac = np.asarray(tf_rows, dtype=np.float64)
    bsize = np.asarray(bs_rows, dtype=np.float64)
    return X, y, tfrac, bsize, groups, kept_benches, dropped


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def parse_weight(w: str):
    if w.lower() in ("none", "null"):
        return None
    if w.lower() == "balanced":
        return "balanced"
    return {0: 1, 1: float(w)}


def build_sample_weight(y: np.ndarray, tfrac: np.ndarray, bsize: np.ndarray,
                        floor: float, beta: float) -> np.ndarray:
    """Cost-model per-sample weight = misclassification cost of each function.

    Each sample is weighted by what it costs to get it wrong, so the tree
    spends its capacity where runtime actually cares:

      hot  (y=1): floor + time_frac        — an FN here loses that function's
                  runtime share (missed JIT speedup).
      cold (y=0): floor + beta * size_norm — an FP here wastes compile time +
                  memory, modelled ∝ bytecode size (size_norm = bsize/median).

    `beta` is the precision↔coverage knob: beta=0 reproduces the pure
    time-weight (cold ~ floor only, model floods hot); larger beta makes
    over-compiling large cold functions expensive, lifting precision while
    keeping the time-heavy hot funcs covered. The asymmetry is fully encoded
    here, so use `--hot-weight 1` (no extra class_weight) with this.
    """
    med = float(np.median(bsize[bsize > 0])) if np.any(bsize > 0) else 1.0
    size_norm = bsize / max(med, 1.0)
    # time_frac is a fraction in [0,1], but inclusive cum_time double-counts
    # nested recursion (e.g. shootout-fib2 reports ~12.0), which would hand one
    # function a runaway weight. Clamp to 1.0 — a function can own at most all
    # of the runtime.
    hot_w = floor + np.clip(tfrac, 0.0, 1.0)
    cold_w = floor + beta * size_norm
    return np.where(y == 1, hot_w, cold_w)


def time_weighted_recall(y: np.ndarray, pred: np.ndarray, tfrac: np.ndarray) -> float:
    """Share of hot runtime actually covered = sum(tfrac of correctly
    predicted-hot) / sum(tfrac of all hot). Maps directly to JIT speedup
    potential captured, unlike plain recall which counts every hot func
    equally regardless of how much time it owns.
    """
    tf = np.clip(tfrac, 0.0, 1.0)  # cap recursion-inflated inclusive time
    hot = y == 1
    denom = float(tf[hot].sum())
    if denom <= 0:
        return 0.0
    num = float(tf[hot & (pred == 1)].sum())
    return num / denom


def make_model(
    class_weight,
    min_samples_leaf: int = 2,
    ccp_alpha: float = 0.001936,
    max_depth: int = 5,
) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        ccp_alpha=ccp_alpha,
        random_state=0,
    )


def cross_validate_groups(
    X: np.ndarray,
    y: np.ndarray,
    groups: List[str],
    build_model_fn: Callable[[], DecisionTreeClassifier],
    n_splits: int = 10,
    verbose: bool = True,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """K-fold StratifiedGroupKFold by benchmark tag (default K=10).

    Each wasm's functions stay together in one fold so feature distributions
    don't leak across train/valid. Returns (oof_pred, per-fold metrics).
    `verbose=False` silences the per-fold print rows — used by the grid
    search in `cv_select_hyperparams`, which evaluates dozens of CV runs.

    `sample_weight`, if given, is passed to each fold's `fit` so the tree
    focuses its splits on time-important functions (option A).
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    oof = np.full_like(y, fill_value=-1)
    fold_stats: List[Dict[str, float]] = []
    group_arr = np.asarray(groups)
    if verbose:
        print(f"\n[CV] StratifiedGroupKFold(n_splits={n_splits}) "
              f"groups={len(set(groups))}")
    for fi, (tr, va) in enumerate(cv.split(X, y, group_arr)):
        model = build_model_fn()
        if sample_weight is not None:
            model.fit(X[tr], y[tr], sample_weight=sample_weight[tr])
        else:
            model.fit(X[tr], y[tr])
        pred = model.predict(X[va])
        oof[va] = pred
        tp = int(((pred == 1) & (y[va] == 1)).sum())
        fp = int(((pred == 1) & (y[va] == 0)).sum())
        fn = int(((pred == 0) & (y[va] == 1)).sum())
        tn = int(((pred == 0) & (y[va] == 0)).sum())
        pos = int((y[va] == 1).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        acc = float((pred == y[va]).mean())
        held = sorted(set(group_arr[va]))
        fold_stats.append(
            dict(fold=fi, n=int(len(va)), pos=pos, acc=acc, prec=prec,
                 rec=rec, tp=tp, fp=fp, fn=fn, tn=tn,
                 modules=len(held))
        )
        if verbose:
            print(f"  fold {fi}: n={len(va):>4} pos={pos:>3} "
                  f"modules={len(held):>2} acc={acc:.3f} "
                  f"prec={prec:.3f} rec={rec:.3f} "
                  f"tp={tp} fp={fp} fn={fn} tn={tn}")
    return oof, fold_stats


# ---------------------------------------------------------------------------
# K-fold hyperparameter selection
# ---------------------------------------------------------------------------

def _score_oof(y: np.ndarray, pred: np.ndarray, metric: str,
               weight: Optional[np.ndarray] = None) -> float:
    """Compute the requested scalar metric on OOF predictions.

    If `weight` is given, tp/fp/fn are summed in weight units so the metric
    rewards getting time-important functions right (option A).
    """
    if weight is None:
        tp = float(((pred == 1) & (y == 1)).sum())
        fp = float(((pred == 1) & (y == 0)).sum())
        fn = float(((pred == 0) & (y == 1)).sum())
    else:
        tp = float(weight[(pred == 1) & (y == 1)].sum())
        fp = float(weight[(pred == 1) & (y == 0)].sum())
        fn = float(weight[(pred == 0) & (y == 1)].sum())
    p = tp / max(1e-9, tp + fp)
    r = tp / max(1e-9, tp + fn)
    if metric == "precision":
        return p
    if metric == "recall":
        return r
    if metric == "f2":
        # Recall-weighted F-beta with beta=2; matches the hybrid cost model
        # where missing a hot function (FN) hurts more than wrongly
        # compiling a cold one (FP).
        return 5 * p * r / max(1e-9, 4 * p + r)
    # default: F1
    return 2 * p * r / max(1e-9, p + r)


# Grid intentionally small (~40 combos) so the full sweep fits in seconds
# on the wabench/profile_log scale. Decision-tree fits are millisecond-fast
# at these depths, so K-folds * combos stays cheap.
_HW_GRID  = [1, 3, 5, 10, 22]
_CCP_GRID = [0.0, 0.001, 0.001936, 0.005]
_MSL_GRID = [2, 5]


def cv_select_hyperparams(
    X: np.ndarray,
    y: np.ndarray,
    groups: List[str],
    metric: str = "f1",
    n_splits: int = 10,
    sample_weight: Optional[np.ndarray] = None,
    score_weight: Optional[np.ndarray] = None,
) -> Tuple[Tuple[float, float, int], List[Dict[str, float]]]:
    """Pick (hot_weight, ccp_alpha, min_samples_leaf) by 5-fold CV.

    Returns ((best_hw, best_ccp, best_msl), all_results).
    Ties broken by the order grids are iterated (hw first).
    """
    print(f"\n[CV-Select] grid: {len(_HW_GRID)}×{len(_CCP_GRID)}×{len(_MSL_GRID)} "
          f"= {len(_HW_GRID)*len(_CCP_GRID)*len(_MSL_GRID)} combos, "
          f"{n_splits}-fold, metric={metric}")
    results: List[Dict[str, float]] = []
    best = None
    best_combo: Tuple[float, float, int] = (5.0, 0.001936, 2)
    for hw in _HW_GRID:
        for ccp in _CCP_GRID:
            for msl in _MSL_GRID:
                cw = {0: 1, 1: float(hw)}

                def build_fn(_cw=cw, _msl=msl, _ccp=ccp):
                    return make_model(_cw, _msl, _ccp)

                oof, _ = cross_validate_groups(
                    X, y, groups, build_fn, n_splits=n_splits, verbose=False,
                    sample_weight=sample_weight)
                valid = oof != -1
                sw = score_weight[valid] if score_weight is not None else None
                score = _score_oof(y[valid], oof[valid], metric, weight=sw) if valid.any() else 0.0
                results.append({"hw": hw, "ccp": ccp, "msl": msl,
                                "score": score})
                if best is None or score > best:
                    best = score
                    best_combo = (float(hw), float(ccp), int(msl))

    # Print top-5 for inspection.
    ranked = sorted(results, key=lambda r: -r["score"])[:5]
    print(f"[CV-Select] top {len(ranked)} by {metric}:")
    for r in ranked:
        tag = " *" if (r["hw"], r["ccp"], r["msl"]) == best_combo else ""
        print(f"    hw={r['hw']:>3} ccp={r['ccp']:.4f} msl={r['msl']:>2}  "
              f"{metric}={r['score']:.4f}{tag}")
    print(f"[CV-Select] picked  hw={best_combo[0]} ccp={best_combo[1]:.4f} "
          f"msl={best_combo[2]}  best_{metric}={best:.4f}")
    return best_combo, results


def _report_metrics(tag: str, y: np.ndarray, pred: np.ndarray) -> None:
    if len(y) == 0:
        print(f"[{tag}] empty set")
        return
    acc = float((pred == y).mean())
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    pos = int((y == 1).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    print(f"[{tag}] n={len(y)} pos={pos} acc={acc:.3f} "
          f"prec={prec:.3f} rec={rec:.3f} tp={tp} fp={fp} fn={fn} tn={tn}")


# ---------------------------------------------------------------------------
# Header emission
# ---------------------------------------------------------------------------

def _format_int_array(name: str, ctype: str, values: List[int]) -> str:
    body = ",\n    ".join(str(v) for v in values)
    return (f"constexpr {ctype} {name}[kNodeCount] = {{\n"
            f"    {body},\n}};\n")


def emit_header(model: DecisionTreeClassifier, path: Path) -> None:
    tree = model.tree_
    n = int(tree.node_count)
    feature = tree.feature.tolist()
    threshold = tree.threshold.tolist()
    left = tree.children_left.tolist()
    right = tree.children_right.tolist()
    value = tree.value.reshape(n, -1)

    feat_lines = ",\n    ".join(f'"{k}"' for k in FEATURE_NAMES)

    f_arr: List[int] = []
    t_arr: List[int] = []
    l_arr: List[int] = []
    r_arr: List[int] = []
    for i in range(n):
        if left[i] == -1:  # leaf
            cls = int(np.argmax(value[i]))
            f_arr.append(-1)
            t_arr.append(0)
            l_arr.append(cls)
            r_arr.append(cls)
        else:
            # Features are non-negative integer counts (or -1 for
            # call_graph_depth). floor(threshold) gives an equivalent integer
            # compare because sklearn picks thresholds at midpoints between
            # consecutive integers.
            f_arr.append(int(feature[i]))
            t_arr.append(int(np.floor(float(threshold[i]))))
            l_arr.append(int(left[i]))
            r_arr.append(int(right[i]))

    # int16_t threshold storage assumes |threshold| < 32768. If a future tree
    # exceeds this (call_freq_x_body is the usual culprit), widen the type in
    # both this emitter and JITPredictor.cpp rather than truncating silently.
    INT16_MAX = 32767
    INT16_MIN = -32768
    for i, t in enumerate(t_arr):
        if not (INT16_MIN <= t <= INT16_MAX):
            raise OverflowError(
                f"kNodeThreshold[{i}] = {t} exceeds int16_t range; "
                f"widen the type in emit_header() and JITPredictor.cpp"
            )

    content = f"""// Auto-generated by src/decision_model/export_model.py
// DO NOT EDIT BY HAND. Re-run the exporter to refresh.
#pragma once

#include <cstdint>

namespace Walrus {{
namespace JITPredictorModel {{

constexpr int kFeatureCount = {len(FEATURE_NAMES)};
constexpr int kNodeCount = {n};

// Feature ordering used by the trained model.
constexpr const char* kFeatureNames[kFeatureCount] = {{
    {feat_lines},
}};

// Node feature index (-1 if leaf). int8_t fits feature indices 0..127.
{_format_int_array("kNodeFeature", "int8_t", f_arr)}
// floor(threshold) — features are integer-valued, so the int compare
// `feature <= threshold` is equivalent to sklearn's float compare.
{_format_int_array("kNodeThreshold", "int16_t", t_arr)}
// At inner nodes: index of left child (X[feature] <= threshold).
// At leaf nodes: predicted class (0 or 1).
{_format_int_array("kNodeLeft", "int16_t", l_arr)}
{_format_int_array("kNodeRight", "int16_t", r_arr)}
}} // namespace JITPredictorModel
}} // namespace Walrus
"""
    path.write_text(content)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _plot_tree(model, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(16, model.tree_.node_count * 0.6), 10))
    plot_tree(
        model,
        feature_names=FEATURE_NAMES,
        class_names=["cold", "hot"],
        filled=True,
        rounded=True,
        impurity=False,
        proportion=True,
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_feature_importance(model, out_path: Path) -> None:
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    names = [FEATURE_NAMES[i] for i in order]
    values = importances[order]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(values)), values[::-1], color="steelblue")
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel("importance")
    ax.set_title("Decision tree feature importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str
) -> None:
    labels = [0, 1]
    cm = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(["cold(0)", "hot(1)"])
    ax.set_yticklabels(["cold(0)", "hot(1)"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(title)
    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def emit_visualizations(
    model,
    y_fit: np.ndarray, y_fit_pred: np.ndarray,
    y_cv: Optional[np.ndarray], y_cv_pred: Optional[np.ndarray],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tree_png = out_dir / "decision_tree.png"
    importance_png = out_dir / "feature_importance.png"
    fit_cm_png = out_dir / "confusion_matrix.png"
    tree_dot = out_dir / "decision_tree.dot"

    _plot_tree(model, tree_png)
    _plot_feature_importance(model, importance_png)
    _plot_confusion_matrix(
        y_fit, y_fit_pred, fit_cm_png, "Final-fit confusion matrix"
    )
    if y_cv is not None and y_cv_pred is not None and len(y_cv):
        _plot_confusion_matrix(
            y_cv, y_cv_pred, out_dir / "confusion_matrix_cv.png",
            "Cross-validated (OOF) confusion matrix",
        )
    export_graphviz(
        model,
        out_file=str(tree_dot),
        feature_names=FEATURE_NAMES,
        class_names=["cold", "hot"],
        filled=True,
        rounded=True,
        impurity=False,
        proportion=True,
    )

    print(f"[Viz] tree              -> {tree_png}")
    print(f"[Viz] feature importance -> {importance_png}")
    print(f"[Viz] confusion matrix   -> {fit_cm_png}")
    print(f"[Viz] graphviz dot       -> {tree_dot}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hot-weight", default="22",
                    help="hot class_weight: number, 'balanced', or 'none'")
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--min-samples-leaf", type=int, default=2)
    ap.add_argument("--ccp-alpha", type=float, default=0.001936)
    ap.add_argument("--train-dir", default=str(PROFILE_DIR),
                    help="override training profile dir")
    ap.add_argument("--test-dir", default=str(TEST_PROFILE_DIR),
                    help="override held-out test profile dir")
    ap.add_argument("--cv-select", action="store_true",
                    help="K-fold hyperparameter selection: sweep "
                         "(hot_weight, ccp_alpha, min_samples_leaf) under "
                         "5-fold CV on the train pool, pick the combo with "
                         "the best CV score, then fit one tree on the full "
                         "pool with that combo. Overrides --hot-weight, "
                         "--ccp-alpha, --min-samples-leaf for the final fit.")
    ap.add_argument("--cv-metric", default="f1",
                    choices=("f1", "f2", "precision", "recall"),
                    help="metric used by --cv-select (default: f1)")
    ap.add_argument("--time-weight", action="store_true",
                    help="option A: weight each function by floor+time_frac "
                         "in fit AND in the CV-select metric, so the tree "
                         "chases the funcs that actually own runtime.")
    ap.add_argument("--weight-floor", type=float, default=0.05,
                    help="baseline weight for every function under "
                         "--time-weight (default 0.05).")
    ap.add_argument("--cost-beta", type=float, default=1.0,
                    help="precision<->coverage knob for --time-weight: cold "
                         "FP cost = beta * (bytecode_sz / median). 0 = pure "
                         "time-weight (floods hot); larger = higher precision "
                         "(default 1.0). Use with --hot-weight 1.")
    ap.add_argument("--emit", dest="emit", action="store_true", default=True)
    ap.add_argument("--no-emit", dest="emit", action="store_false")
    args = ap.parse_args()

    # Resolve overrides into Path; train/test glob differ (wabench prefixes
    # files with NN_, cpp_bench uses bare stems), but `*.txt` is the safer
    # default and we already skip non-data files via the header parser.
    train_dir = Path(args.train_dir)
    test_dir  = Path(args.test_dir)

    # ---- Train set ----
    # `*.txt` (no NN_ prefix requirement) so legacy-relabeled mirror dirs
    # without the wabench numeric prefix also work.
    X, y, tfrac, bsize, groups, kept_benches, dropped = load_dataset(
        train_dir, glob_pat="*.txt", tag="Train")
    if len(y) == 0:
        print("error: empty train dataset", file=sys.stderr)
        sys.exit(1)

    # Option A: cost-model sample weights. `sw` is threaded into every fit
    # and the weighted CV-select metric; `tfrac` also drives the
    # time-weighted recall report.
    sw = (build_sample_weight(y, tfrac, bsize, args.weight_floor, args.cost_beta)
          if args.time_weight else None)
    if args.time_weight:
        print(f"[Policy]  time-weight ON (floor={args.weight_floor} "
              f"beta={args.cost_beta}): hot=floor+time_frac, "
              f"cold=floor+beta*size_norm; "
              f"weight range [{sw.min():.3f}, {sw.max():.3f}]")
    pos = int((y == 1).sum())
    print(f"\n[Train/Dropped] {len(dropped)} benchmarks (fast / trapped / errored)")
    for stem, wall_s, rc in dropped:
        print(f"  - {stem:<46} wall={wall_s} rc={rc}")
    print(f"[Train] benchmarks={len(kept_benches)} n={len(y)} "
          f"cold={len(y) - pos} hot={pos} hot_ratio={pos / len(y):.2%} "
          f"groups={len(set(groups))}")

    # ---- Held-out test set (CppBenchmark) ----
    if test_dir.exists():
        X_test, y_test, tfrac_test, _, _, test_benches, test_dropped = load_dataset(
            test_dir, glob_pat="*.txt", tag="Test")
        if len(y_test) > 0:
            tpos = int((y_test == 1).sum())
            print(f"\n[Test/Dropped] {len(test_dropped)} benchmarks (fast / trapped / errored)")
            for stem, wall_s, rc in test_dropped:
                print(f"  - {stem:<46} wall={wall_s} rc={rc}")
            print(f"[Test ] benchmarks={len(test_benches)} n={len(y_test)} "
                  f"cold={len(y_test) - tpos} hot={tpos} "
                  f"hot_ratio={tpos / len(y_test):.2%}")
        else:
            print(f"\n[Test ] {test_dir} held no usable rows; skipping test eval")
            X_test = y_test = tfrac_test = None
    else:
        print(f"\n[Test ] {test_dir} not found; skipping test eval")
        X_test = y_test = None
    # If --cv-select, sweep the grid first and overwrite the policy knobs
    # with whatever wins on 5-fold OOF. The final fit and emit below then
    # use the selected combo.
    if args.cv_select:
        best, _ = cv_select_hyperparams(
            X, y, groups, metric=args.cv_metric,
            sample_weight=sw, score_weight=sw)
        hw_sel, ccp_sel, msl_sel = best
        args.hot_weight = str(int(hw_sel)) if hw_sel == int(hw_sel) else str(hw_sel)
        args.ccp_alpha = ccp_sel
        args.min_samples_leaf = msl_sel

    cw = parse_weight(args.hot_weight)
    print(f"[Policy]  hot_weight={args.hot_weight} (class_weight={cw}) "
          f"min_samples_leaf={args.min_samples_leaf} "
          f"ccp_alpha={args.ccp_alpha}")

    def build_fn() -> DecisionTreeClassifier:
        return make_model(cw, args.min_samples_leaf, args.ccp_alpha,
                          max_depth=args.max_depth)

    oof, folds = cross_validate_groups(X, y, groups, build_fn, n_splits=10,
                                       sample_weight=sw)
    valid = oof != -1
    if valid.any():
        _report_metrics("CV-OOF", y[valid], oof[valid])
        print(f"[CV-Macro] acc={np.mean([s['acc'] for s in folds]):.3f} "
              f"prec={np.mean([s['prec'] for s in folds]):.3f} "
              f"rec={np.mean([s['rec'] for s in folds]):.3f}")
        # Time-weighted recall: the share of hot RUNTIME the OOF predictions
        # cover. This is the number that tracks JIT speedup, vs plain recall
        # which counts a 0.01%-time hot func the same as a 40%-time one.
        twr = time_weighted_recall(y[valid], oof[valid], tfrac[valid])
        print(f"[CV-TimeRecall] hot-runtime covered = {twr:.3f}")

    model = build_fn()
    if sw is not None:
        model.fit(X, y, sample_weight=sw)
    else:
        model.fit(X, y)
    fit_pred = model.predict(X)
    _report_metrics("Fit  ", y, fit_pred)
    print(f"[Fit-TimeRecall] hot-runtime covered = "
          f"{time_weighted_recall(y, fit_pred, tfrac):.3f}")
    print(f"[Model] nodes={model.tree_.node_count} "
          f"predicted_hot_on_pool={int(fit_pred.sum())}/{len(y)}")

    # Held-out test evaluation (CppBenchmark). Reported regardless of --emit
    # so a sweep can quickly compare policies against the real test set.
    test_pred = None
    if X_test is not None and y_test is not None and len(y_test) > 0:
        test_pred = model.predict(X_test)
        _report_metrics("Test ", y_test, test_pred)
        if tfrac_test is not None:
            print(f"[Test-TimeRecall] hot-runtime covered = "
                  f"{time_weighted_recall(y_test, test_pred, tfrac_test):.3f}")
        print(f"[Test ] predicted_hot={int(test_pred.sum())}/{len(y_test)}")
    print()

    print("[Tree]")
    print(export_text(model, feature_names=FEATURE_NAMES, show_weights=True))

    if args.emit:
        # First emit-after-overwrite saves the prior tree so the old behavior
        # can be diffed/restored if the wabench-only retrain regresses.
        backup = HEADER_PATH.with_suffix(".h.prewabench")
        if HEADER_PATH.exists() and not backup.exists():
            backup.write_text(HEADER_PATH.read_text())
            print(f"[Header] backed up original -> {backup}")
        emit_header(model, HEADER_PATH)
        print(f"[Header] wrote {HEADER_PATH}")

        # Figures are tied to the emit. Skipping them under --no-emit avoids
        # the figures/ dir falling out of sync with the active header during
        # exploratory sweeps (which is exactly why one runs --no-emit).
        emit_visualizations(
            model, y, fit_pred,
            y[valid] if valid.any() else None,
            oof[valid] if valid.any() else None,
            FIG_DIR,
        )
        if test_pred is not None:
            _plot_confusion_matrix(
                y_test, test_pred,
                FIG_DIR / "confusion_matrix_test.png",
                "Held-out test confusion matrix",
            )
            print(f"[Viz] test confusion     -> "
                  f"{FIG_DIR / 'confusion_matrix_test.png'}")
    else:
        print("[Header] --no-emit: header and figures not written "
              f"(figures/ left at last emit's state)")


if __name__ == "__main__":
    main()
