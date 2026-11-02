#!/usr/bin/env python3
"""Record the machine code size of every function for each benchmark.

The label rule (labeling.py) is "time share >= theta x code share", so it needs the cost side.
This script produces those numbers; relabel.py and eval_metrics.py read them.

Given a name that is not exported, `--run-export` makes walrus only compile the module
without running the workload, so each module finishes in under a second regardless of run time.

The output is one file per benchmark, one "func_index bytes" pair per line.

    dump_codesize.py benches.tsv --out-dir profile/codesize

Note: code= printed by `--jit-stats` is the **reserved** buffer size, about 30% larger than
the **actually written** bytes summed here. Do not mix the two numbers.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from profile_bench import find_walrus, read_bench_list, split_args  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bench_list", help="TSV in the same format as profile_bench.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--walrus", default=None)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--force", action="store_true", help="regenerate existing files too")
    ap.add_argument("--remap", action="append", default=[], metavar="OLD=NEW",
                    help="rewrite paths in the bench list, for running on another machine")
    args = ap.parse_args()

    walrus = find_walrus(args.walrus)
    # walrus runs in the benchmark's working directory, so a relative --jit-code-dump
    # path resolves there. The output directory does not exist there, so fopen fails
    # silently, rc stays 0 and no dump is written. Use an absolute path.
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.pop("WALRUS_PERF_DIR", None)

    rules = []
    for r in args.remap:
        old, _, new = r.partition("=")
        if not old or not new:
            raise SystemExit(f"--remap must be of the form OLD=NEW: {r}")
        rules.append((old, new))

    def entries(path: Path) -> int:
        """Count the usable lines in an existing dump."""
        if not path.exists():
            return 0
        n = 0
        for line in path.read_text(errors="replace").splitlines():
            f = line.split()
            if len(f) == 2 and f[0].isdigit() and f[1].isdigit():
                n += 1
        return n

    benches = read_bench_list(Path(args.bench_list))
    have = sum(1 for q in out_dir.glob("*.txt") if entries(q) > 0)
    print(f"walrus     : {walrus}")
    print(f"out dir    : {out_dir}  ({have} files already filled)")
    print(f"bench list : {Path(args.bench_list).resolve()}  ({len(benches)} rows)\n", flush=True)
    ok = skipped = failed = 0
    for i, (tag, _wall, cwd, argstr) in enumerate(benches, 1):
        for old, new in rules:
            cwd = cwd.replace(old, new)
            argstr = argstr.replace(old, new)
        out = out_dir / f"{tag}.txt"
        # An empty file does not count as done. Otherwise an empty file left by an
        # earlier failed run would be skipped on every rerun.
        if entries(out) > 0 and not args.force:
            skipped += 1
            continue
        if out.exists():
            out.unlink()
        wasm_args, _stdin = split_args(argstr)
        cmd = [walrus, "--jit", f"--jit-code-dump={out}",
               "--run-export", "__no_such_export__"] + wasm_args
        try:
            p = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, cwd=cwd,
                               timeout=args.timeout, env=env)
        except subprocess.TimeoutExpired:
            print(f"[{i:>4}/{len(benches)}] {tag:<46} timeout", flush=True)
            failed += 1
            continue
        n = total = 0
        if out.exists():
            for line in out.read_text(errors="replace").splitlines():
                f = line.split()
                if len(f) == 2:
                    n += 1
                    total += int(f[1])
        if n == 0:
            print(f"[{i:>4}/{len(benches)}] {tag:<46} no dump (rc={p.returncode})",
                  flush=True)
            failed += 1
            if failed == 1 and ok == 0:
                print(f"       (empty from the first bench. Check the wasm path and whether the JIT "
                      f"works on this machine:\n"
                      f"        {' '.join(cmd)})", flush=True)
            continue
        ok += 1
        print(f"[{i:>4}/{len(benches)}] {tag:<46} {n:>5} funcs  {total:>9} bytes of machine code",
              flush=True)
    print(f"\ndone {ok}, skipped {skipped}, failed {failed} -> {out_dir}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
