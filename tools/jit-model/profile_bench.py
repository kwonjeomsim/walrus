#!/usr/bin/env python3
"""Profile a list of wasm benchmarks under --profile-output and write labeled
training files for the JIT-hybrid decision tree.

Each benchmark is run --repeats times and the per-function self times are
averaged before labeling, to damp timer noise. Self-time labeling only makes
sense for runs that COMPLETE -- a run cut short by the timeout leaves its hot
kernel mid-flight with no self time recorded -- so a benchmark is kept only if
every repeat exits 0 within the timeout.

The bench list is a TSV with a header line and four columns:

    tag       name for the output file; must be unique
    wall_s    interpreter wall time, carried through to the manifest
    cwd       directory to run in (benchmarks that open data files need this)
    args      everything after the walrus mode flags, e.g.
                  bench.wasm
                  --mapdirs ./data /data --args bench.wasm 100
                  --stdin input.txt bench.wasm

`--stdin <path>` belongs to this runner, not to walrus: it is stripped and the
file is fed as standard input, for benchmarks that read their workload there.

Output: <out-dir>/<tag>.txt (labeled), raw/ dumps, _manifest.tsv, _labels.tsv,
_summary.tsv.

  profile_bench.py benches.tsv --out-dir profile/profile_log
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

WALRUS_CANDIDATES = ("out/walrus", "out/release/walrus", "out/linux/x64/walrus")


def find_walrus(explicit: str | None) -> str:
    for path in (explicit, os.environ.get("WALRUS_BIN")):
        if path:
            if not os.path.exists(path):
                sys.exit(f"walrus binary not found: {path}")
            return str(Path(path).resolve())

    for rel in WALRUS_CANDIDATES:
        path = REPO_ROOT / rel
        if path.exists():
            return str(path)

    sys.exit("walrus binary not found; build it or pass --walrus / set WALRUS_BIN.\n"
             "  looked for: " + ", ".join(str(REPO_ROOT / r) for r in WALRUS_CANDIDATES))


def load_labeling() -> ModuleType:
    # Loaded by path, not `import labeling`, so this works from any cwd.
    path = SCRIPT_DIR / "labeling.py"
    if not path.exists():
        sys.exit(f"labeling.py not found next to this script: {path}")
    spec = importlib.util.spec_from_file_location("walrus_jit_labeling", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def split_args(args: str) -> tuple[list[str], str | None]:
    """Return (walrus argv tail, stdin file path or None)."""
    parts = shlex.split(args)
    stdin_path = None
    out = []
    i = 0
    while i < len(parts):
        if parts[i] == "--stdin" and i + 1 < len(parts):
            stdin_path = parts[i + 1]
            i += 2
            continue
        out.append(parts[i])
        i += 1
    return out, stdin_path


def resolve_source(source: str | None, cwd: str) -> str | None:
    """The dump records the wasm path as walrus saw it, which is relative when
    the bench list uses a relative path. Training reads it from the repo root."""
    if not source or os.path.isabs(source):
        return source
    candidate = Path(cwd) / source
    return str(candidate.resolve()) if candidate.exists() else source


def read_bench_list(path: Path) -> list[tuple[str, float, str, str]]:
    rows = []
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        if header[:4] != ["tag", "wall_s", "cwd", "args"]:
            sys.exit(f"{path}: expected a 'tag wall_s cwd args' header, got {header}")
        for n, line in enumerate(fh, 2):
            if not line.strip():
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) != 4:
                sys.exit(f"{path}:{n}: expected 4 columns, got {len(p)}")
            rows.append((p[0], float(p[1]), p[2], p[3]))
    return rows


def run_once(walrus: str, raw: Path, cwd: str, args: str,
             timeout: float) -> tuple[float, int]:
    tail, stdin_path = split_args(args)
    fin = open(stdin_path, "rb") if stdin_path else subprocess.DEVNULL
    t0 = time.time()
    try:
        proc = subprocess.run([walrus, "--profile-output", str(raw), *tail],
                              cwd=cwd, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, stdin=fin,
                              timeout=timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc = 124
    finally:
        if stdin_path:
            fin.close()
    return time.time() - t0, rc


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bench_list", help="TSV of benchmarks to profile")
    ap.add_argument("--out-dir", default=os.environ.get("WALRUS_PROFILE_OUT"),
                    help="where to write labeled files "
                         "(default: <repo>/profile/profile_log)")
    ap.add_argument("--walrus",
                    help="walrus binary (default: $WALRUS_BIN, else out/walrus)")
    ap.add_argument("--repeats", type=int, default=3,
                    help="runs per benchmark, averaged (default: 3)")
    ap.add_argument("--timeout", type=float, default=240.0,
                    help="seconds per run; profiling slows a run down several-"
                         "fold, so keep this generous (default: 240)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    walrus = find_walrus(args.walrus)
    P = load_labeling()

    # Absolute: each benchmark runs with its own cwd, so a relative
    # --profile-output would be written over there, or silently fail.
    out_dir = Path(args.out_dir).resolve() if args.out_dir \
        else REPO_ROOT / "profile/profile_log"
    raw_dir = out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows = read_bench_list(Path(args.bench_list))
    print(f"walrus  : {walrus}")
    print(f"out-dir : {out_dir}")
    print(f"{len(rows)} benchmarks x {args.repeats} runs\n")

    results, manifest = [], []
    for i, (tag, interp_s, cwd, bench_args) in enumerate(rows, 1):
        if not os.path.isdir(cwd):
            print(f"[{i:3d}/{len(rows)}] MISSING cwd  {tag}  {cwd}")
            continue

        dumps, walls, ok = [], [], True
        for rep in range(args.repeats):
            raw = raw_dir / f"{tag}.r{rep}.raw.txt"
            wall, rc = run_once(walrus, raw, cwd, bench_args, args.timeout)
            walls.append(wall)
            if rc != 0 or not raw.exists() or raw.stat().st_size == 0:
                ok = False
                print(f"[{i:3d}/{len(rows)}] DROP rep{rep} rc={rc}  {tag}")
                break
            dumps.append(P.parse_dump(str(raw)))
        if not ok:
            sys.stdout.flush()
            continue

        avg = P.average_dumps(dumps)
        if not any(r["self_time_ns"] > 0 for r in avg["rows"]):
            print(f"[{i:3d}/{len(rows)}] no-activity  {tag}")
            continue

        wall_s = sum(walls) / len(walls)
        labeled = P.label(avg["rows"], int(wall_s * 1e9))
        out_path = out_dir / f"{tag}.txt"
        P.write_bench(str(out_path), tag, resolve_source(avg["source"], cwd) or tag,
                      wall_s, 0, labeled)
        n_hot = sum(1 for x in labeled if x[-1] == 1)
        results.append({"benchmark": tag, "wall_s": wall_s, "rc": 0,
                        "source": resolve_source(avg["source"], cwd) or tag,
                        "labeled": labeled,
                        "log_size": out_path.stat().st_size})
        manifest.append((tag, cwd, bench_args, f"{interp_s:.3f}", str(n_hot),
                         str(len(labeled))))
        print(f"[{i:3d}/{len(rows)}] KEEP interp~{interp_s:6.2f}s "
              f"prof~{wall_s:6.2f}s hot={n_hot:>3}/{len(labeled):<4} {tag}")
        sys.stdout.flush()

    P.write_aggregate(str(out_dir), results)
    with open(out_dir / "_manifest.tsv", "w") as fh:
        fh.write("tag\tcwd\targs\tinterp_s\tn_hot\tn_funcs\n")
        for row in manifest:
            fh.write("\t".join(row) + "\n")
    print(f"\nprofiled {len(manifest)}/{len(rows)} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
