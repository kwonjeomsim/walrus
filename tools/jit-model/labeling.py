#!/usr/bin/env python3
"""Label raw Walrus per-function profiles hot/cold for JIT-hybrid training.

This is the labeling half of the profiling pipeline that feeds the decision-
tree JIT predictor:

    1. `walrus --profile-output <raw>`  writes a raw per-function dump
       (columns: function module index bytecode_sz self_time_ns) at program
       exit. See src/runtime/Profiler.cpp.
    2. THIS module reads that raw dump, assigns each function a hot/cold label,
       and rewrites it in the labeled COLS format that
       tools/jit-model/dataset.py consumes.

Used as a library (parse_dump / label / write_bench / write_aggregate / COLS)
by the profiling drivers, and runnable directly to (re)label a directory.

--------------------------------------------------------------------------
Labeling rule  (runtime share vs machine-code share)
--------------------------------------------------------------------------
A function is worth compiling when the runtime it owns pays for the machine
code it costs. Both quantities are measured, and both are turned into shares of
their module so the rule means the same thing in a 20-function benchmark and in
a 1000-function application:

    time_share(f)  = self_time(f)  / total self time of the module
    bytes_share(f) = code_bytes(f) / total machine code of the module
    hot(f)         = time_share(f) >= THETA * bytes_share(f)

THETA is the one knob. At THETA = 1 a function must merely pull its weight: own
as much of the runtime as it occupies of the code. Raising THETA demands more
runtime per byte, so fewer functions qualify and less machine code is emitted.

The rule looks at one function at a time. Whether f is hot does not depend on
what any other function scored, which is what lets the same rule serve as the
compile decision at run time -- the model predicts hot/cold and the runtime
compiles the hot ones, with no ranking, budget or tie-break in between.

THETA also defines the optimization levels. Measured on the 55-benchmark test
set, the labels themselves land at:

    THETA = 1  ->  0.980 of the runtime covered,  17.4% of all-JIT's code  (-O3)
    THETA = 2  ->  0.932                          10.1%                    (-O2)
    THETA = 4  ->  0.844                           6.6%                    (-O1)

self_time comes from `walrus --profile-output`; code_bytes from
`walrus --jit-code-dump=`. A function with no measured code size cannot be
priced and is labeled cold.

MIN_SELF_FRAC still guards near-idle runs, where total self time is small enough
that timer noise would otherwise be labeled hot.

Averaging: the drivers run each benchmark several times and average the
per-function self_time_ns before labeling, to damp timer noise.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

# Final labeled column order (must match dataset.parse_profile).
COLS = [
    "function", "module", "index", "bytecode_sz", "self_time_ns",
    "time_frac", "label",
]
# Columns dropped in the self-time-only rewrite. Logs produced before that
# still carry them; readers key off the header line, so both widths parse.
LEGACY_COLS = {"call_count", "loop_count"}

# Labeling knobs (see module docstring).
THETA = 1.0             # runtime share demanded per unit of machine-code share
MIN_SELF_FRAC = 0.002   # a hot function must own at least this share of runtime


# ---------------------------------------------------------------------------
# Raw-dump parsing
# ---------------------------------------------------------------------------

def parse_dump(path: str) -> Dict[str, object]:
    """Parse a raw `--profile-output` dump.

    Returns {"source": <wasm path or None>, "rows": [row, ...]} where each row
    is a dict with keys: function, module, index, bytecode_sz, self_time_ns.
    Header/comment lines are skipped.

    Columns are located by name from the header line rather than by position,
    so this reads raw and labeled dumps alike, and logs written before
    call_count/loop_count were dropped as well as ones written after (both
    widths would otherwise be seven columns and indistinguishable).
    """
    source = None
    cols: List[str] = []
    rows: List[dict] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("===="):
                continue
            if s.startswith("#"):
                if s.startswith("# source"):
                    source = s.split("=", 1)[1].strip()
                continue
            parts = s.split()
            if parts[0] == "function":
                cols = parts
                continue
            if parts[0] != "func" and not parts[0].startswith("$"):
                continue
            if not cols or len(parts) != len(cols):
                continue
            f = dict(zip(cols, parts))
            try:
                rows.append({
                    "function": f["function"],
                    "module": int(f["module"]),
                    "index": int(f["index"]),
                    "bytecode_sz": int(f["bytecode_sz"]),
                    "self_time_ns": int(f["self_time_ns"]),
                })
            except (KeyError, ValueError):
                continue
    return {"source": source, "rows": rows}


def average_dumps(dumps: List[Dict[str, object]]) -> Dict[str, object]:
    """Average per-function self_time_ns across repeated runs of one benchmark.

    bytecode_sz is deterministic, so the first run's value is kept;
    self_time_ns is averaged (integer) over the runs that contain each
    function index. `dumps` is a list of parse_dump() results.
    """
    if not dumps:
        return {"source": None, "rows": []}
    base = {r["index"]: dict(r) for r in dumps[0]["rows"]}
    acc: Dict[int, List[int]] = {i: [r["self_time_ns"]] for i, r in base.items()}
    for d in dumps[1:]:
        for r in d["rows"]:
            acc.setdefault(r["index"], []).append(r["self_time_ns"])
            base.setdefault(r["index"], dict(r))
    for i, times in acc.items():
        base[i]["self_time_ns"] = int(round(sum(times) / len(times)))
    rows = [base[i] for i in sorted(base)]
    return {"source": dumps[0]["source"], "rows": rows}


# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------

def label(rows: List[dict], wall_ns: int = 0,
          code_bytes: Dict[int, int] | None = None,
          theta: float = None) -> List[Tuple]:
    """Return labeled rows as 9-tuples in COLS order.

    `code_bytes` maps a function index to its measured machine code size, as
    written by `walrus --jit-code-dump=`. Without it there is no cost side and
    every function is labeled cold, since the rule cannot be evaluated.
    `theta` overrides the module-level THETA, which is how the optimization
    levels are built.
    """
    if not rows:
        return []

    th = THETA if theta is None else theta
    total_self = sum(r["self_time_ns"] for r in rows)
    denom = total_self if total_self > 0 else (wall_ns if wall_ns > 0 else 1)
    total_bytes = sum(code_bytes.get(r["index"], 0) for r in rows) if code_bytes else 0

    labeled: List[Tuple] = []
    for r in rows:
        time_share = r["self_time_ns"] / total_self if total_self > 0 else 0.0
        lab = 0
        if total_bytes > 0 and time_share >= MIN_SELF_FRAC:
            bytes_share = code_bytes.get(r["index"], 0) / total_bytes
            lab = 1 if time_share >= th * bytes_share else 0
        labeled.append((
            r["function"], r["module"], r["index"], r["bytecode_sz"],
            r["self_time_ns"], round(r["self_time_ns"] / denom, 6), lab,
        ))
    return labeled


# ---------------------------------------------------------------------------
# Labeled-file emission
# ---------------------------------------------------------------------------

def write_bench(path: str, bench: str, source: str, wall_s: float, rc: int,
                labeled: List[Tuple]) -> None:
    """Rewrite `path` in the labeled COLS format read by dataset.py."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("==== Walrus per-function profile (labeled) ====\n")
        fh.write(f"# benchmark = {bench}\n")
        fh.write(f"# source    = {source}\n")
        fh.write(f"# wall_s    = {wall_s:.2f}   rc={rc}\n")
        fh.write(
            f"{'function':<32} {'module':>6} {'index':>6} "
            f"{'bytecode_sz':>12} {'self_time_ns':>16} "
            f"{'time_frac':>11} {'label':>6}\n"
        )
        for row in labeled:
            (fn, mod, idx, bs, st, tf, lab) = row
            fh.write(
                f"{fn:<32} {mod:>6} {idx:>6} {bs:>12} "
                f"{st:>16} {tf:>11.6f} {lab:>6}\n"
            )


def write_aggregate(out_dir: str, results: List[dict]) -> None:
    """Write _summary.tsv (per-benchmark) and _labels.tsv (all rows)."""
    summary = os.path.join(out_dir, "_summary.tsv")
    labels = os.path.join(out_dir, "_labels.tsv")
    with open(summary, "w", encoding="utf-8") as fh:
        fh.write("benchmark\twall_s\trc\tn_funcs\tn_hot\tlog_size\tsource\n")
        for r in results:
            n = len(r["labeled"])
            n_hot = sum(1 for x in r["labeled"] if x[-1] == 1)
            fh.write(f"{r['benchmark']}\t{r['wall_s']:.2f}\t{r['rc']}\t{n}\t"
                     f"{n_hot}\t{r.get('log_size', 0)}\t{r['source']}\n")
    with open(labels, "w", encoding="utf-8") as fh:
        fh.write("benchmark\t" + "\t".join(COLS) + "\n")
        for r in results:
            for row in r["labeled"]:
                fh.write(r["benchmark"] + "\t" + "\t".join(str(x) for x in row) + "\n")


if __name__ == "__main__":
    print("labeling.py is a library; use profile_bench.py to drive profiling.",
          file=sys.stderr)
    sys.exit(1)
