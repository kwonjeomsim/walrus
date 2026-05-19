"""Measure how well the trained predictor's selections line up with the
profile data: fraction of total cum_time covered, recall against the
heuristic labels, and number of selected indices.

Runs walrus --jit-hybrid --jit-hybrid-verbose with a 2-second cap to
capture the predict line, then cross-references against
tmp/profile_log/<name>.txt.

Note: the model was trained on these same logs (via K-fold CV — no
separate held-out set), so these numbers measure end-to-end correctness
of the export → C++ predictor pipeline, not generalization.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

WALRUS_ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = WALRUS_ROOT / "tmp/profile_log"
WALRUS_BIN = WALRUS_ROOT / "out/linux/x64/walrus"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_model import (  # noqa: E402
    parse_profile,
    read_header,
)
from jit_decision_tree import extract_features_from_wasm  # noqa: E402

PREDICT_RE = re.compile(
    r"\[jit-hybrid\] JIT-compiling (\d+) / (\d+) functions \(indices:([^)]*)\)"
)


def run_predictor(wasm: Path) -> tuple[List[int], int]:
    """Returns (predicted_indices, total_function_count)."""
    proc = subprocess.run(
        ["timeout", "2",
         str(WALRUS_BIN), "--jit-hybrid", "--jit-hybrid-verbose", str(wasm)],
        capture_output=True, text=True,
    )
    blob = proc.stdout + proc.stderr
    m = PREDICT_RE.search(blob)
    if not m:
        return [], 0
    indices = [int(x) for x in m.group(3).split()]
    return indices, int(m.group(2))


def evaluate(name: str, log_dir: Path) -> Dict[str, float]:
    txt = log_dir / f"{name}.txt"
    if not (txt.exists() and WALRUS_BIN.exists()):
        return {}
    source, _, _ = read_header(txt)
    wasm = Path(source) if source else None
    if not (wasm and wasm.exists()):
        return {}

    profile = parse_profile(txt)
    predicted, walrus_total = run_predictor(wasm)

    # walrus' total includes synthetic init/segment ModuleFunctions; the
    # predictor only emits indices in the wasm function space, so use the
    # feature extractor's wasm-aligned indices to bound the eval set.
    feats = extract_features_from_wasm(str(wasm))
    aligned_set = {f.index for f in feats}

    total_cum = sum(p.cum_time_ns for p in profile.values())
    covered_cum = sum(
        profile[i].cum_time_ns for i in predicted if i in profile
    )
    coverage = covered_cum / total_cum if total_cum else 0.0

    hot_set = {
        i for i, p in profile.items() if i in aligned_set and p.label == 1
    }
    pred_set = set(predicted)
    tp = len(hot_set & pred_set)
    fn = len(hot_set - pred_set)
    fp = len(pred_set - hot_set)
    recall = tp / len(hot_set) if hot_set else 0.0
    precision = tp / len(pred_set) if pred_set else 0.0

    # the entry function (often $_start) typically swallows >95% of cum_time
    # because it transitively includes everything; report coverage with it
    # excluded to show how much of the *non-trivial* hot work we're catching.
    if profile:
        entry_idx = max(profile, key=lambda i: profile[i].cum_time_ns)
    else:
        entry_idx = -1
    nontrivial_cum = sum(
        p.cum_time_ns for i, p in profile.items() if i != entry_idx
    )
    nontrivial_covered = sum(
        profile[i].cum_time_ns for i in predicted
        if i in profile and i != entry_idx
    )
    nontrivial_coverage = (
        nontrivial_covered / nontrivial_cum if nontrivial_cum else 0.0
    )

    return {
        "name": name,
        "selected": len(predicted),
        "total": walrus_total,
        "wasm_funcs": len(feats),
        "hot_label": len(hot_set),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "coverage": coverage,
        "coverage_excl_entry": nontrivial_coverage,
    }


def main() -> None:
    print("Labels are read directly from each profile log's 'label' column.\n")
    fmt_hdr = (
        f"{'module':<20}{'sel':>5}{'tot':>5}{'hot':>5}"
        f"{'tp':>4}{'fp':>4}{'fn':>4}"
        f"{'prec':>7}{'rec':>7}{'cov':>7}{'cov_x_e':>9}"
    )
    print(fmt_hdr)
    print("-" * len(fmt_hdr))
    sums = {"selected": 0, "tp": 0, "fp": 0, "fn": 0, "hot_label": 0}
    names = sorted(p.stem for p in PROFILE_DIR.glob("*.txt"))
    for name in names:
        r = evaluate(name, PROFILE_DIR)
        if not r:
            print(f"{name:<20} (skipped)")
            continue
        for k in sums:
            sums[k] += r[k]
        print(
            f"{r['name']:<20}{r['selected']:>5}{r['total']:>5}"
            f"{r['hot_label']:>5}{r['tp']:>4}{r['fp']:>4}{r['fn']:>4}"
            f"{r['precision']:>7.2f}{r['recall']:>7.2f}"
            f"{r['coverage']:>7.2f}{r['coverage_excl_entry']:>9.2f}"
        )

    overall_prec = sums["tp"] / max(1, sums["tp"] + sums["fp"])
    overall_rec = sums["tp"] / max(1, sums["hot_label"])
    print("-" * len(fmt_hdr))
    print(f"{'TOTAL':<20}{sums['selected']:>5}{'':>5}{sums['hot_label']:>5}"
          f"{sums['tp']:>4}{sums['fp']:>4}{sums['fn']:>4}"
          f"{overall_prec:>7.2f}{overall_rec:>7.2f}")


if __name__ == "__main__":
    main()
