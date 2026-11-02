#!/usr/bin/env python3
"""Read labeled profiles and turn them into arrays for training.

Where this file sits in the pipeline:

    profile_bench.py  ->  labeled profiles (*.txt)
    relabel.py        ->  per-theta training directories
        │
        └─ this file (load_dataset)  ->  X, y, tfrac, bsize, groups
                    │
                    └─ export_levels.py / eval_metrics.py / make_figures.py

Features are not computed here. jit_decision_tree.extract_features_from_wasm fetches the values
emitted by `walrus --dump-jit-features`, and this file only pairs them with labels into arrays.
Since the same implementation as the runtime is used, training and inference
cannot diverge.

This file was split out of export_model.py. The old export_model.py was 1002 lines that trained
a single tree and wrote the header. With the move to per-level models, header generation went
to export_levels.py and the hyperparameter search and plotting fell out of use.
Only the 137 lines below were still in use, so only those were moved. The old file is
kept as export_model.py.single-tree.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jit_decision_tree import extract_features_from_wasm  # noqa: E402

# Minimum wall time for a run to count as complete. The current label rule (cost-benefit in
# labeling.py) is valid for short runs too, so it is 0. Raise it if switching to a pure time-share label.
MIN_WALL_S = 0.0


# ---------------------------------------------------------------------------

# Each labeled row carries:
#   function module index bytecode_sz self_time_ns time_frac label
# cum_time_ns holds the per-function SELF time; time_frac = self_time / total.
# Logs written before call_count/loop_count were dropped from the profiler
# carry those two extra columns; columns are read by name so both parse.
@dataclass
class ProfileEntry:
    bytecode_sz: int
    cum_time_ns: int
    time_frac: float
    label: int




def parse_profile(path: Path) -> Dict[int, ProfileEntry]:
    """Return {walrus_func_index: ProfileEntry} from one labeled log."""
    out: Dict[int, ProfileEntry] = {}
    cols: List[str] = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "function":
                cols = parts
                continue
            name = parts[0]
            if name != "func" and not name.startswith("$"):
                continue
            if not cols or len(parts) != len(cols):
                continue
            f = dict(zip(cols, parts))
            try:
                idx = int(f["index"])
                entry = ProfileEntry(
                    bytecode_sz=int(f["bytecode_sz"]),
                    cum_time_ns=int(f["self_time_ns"]),
                    time_frac=float(f["time_frac"]),
                    label=int(f["label"]),
                )
            except (KeyError, ValueError):
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
    with_index: bool = False,
):
    """Return (X, y, tfrac, bsize, groups, kept_benches, dropped) from a dir.

    groups[i] = benchmark tag (file stem) so StratifiedGroupKFold keeps
    each wasm in a single fold. `glob_pat` defaults to wabench's numeric-
    prefixed layout; the CppBenchmark test set uses `*.txt` (no prefix).

    With `with_index=True` an extra array is appended to the tuple: the wasm
    function index of each row, which eval_metrics.py needs to look up that
    function's measured machine-code size. The default keeps the 7-tuple that
    every existing caller unpacks.

    tfrac[i] = the function's time_frac (inclusive cum_time / program total).
    bsize[i] = the function's bytecode size (compile-cost proxy). Both feed
    the cost-model sample weights and the time-weighted recall report.
    """
    X_rows: List[List[int]] = []
    y_rows: List[int] = []
    tf_rows: List[float] = []
    bs_rows: List[int] = []
    idx_rows: List[int] = []
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
            idx_rows.append(f.index)
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
    if with_index:
        findex = np.asarray(idx_rows, dtype=np.int64)
        return X, y, tfrac, bsize, groups, kept_benches, dropped, findex
    return X, y, tfrac, bsize, groups, kept_benches, dropped
