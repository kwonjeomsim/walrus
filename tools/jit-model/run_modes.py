#!/usr/bin/env python3
"""Run a benchmark list in several execution modes, measure time and machine code size, and write JSON.

Written to reproduce the x86 measurements exactly on another machine (e.g. a Raspberry Pi).
Only the paths need to change to reuse the same bench list.

Mode names
    interp   interpreter only (slow end of the baseline)
    alljit   compile everything, --jit (fast end of the baseline)
    tierup   tiered --jit-tierup (threshold proportional to function size)
    lazy     lazy compilation (--jit-tierup, threshold 1)
    O1/O2/O3 the three levels of our model, --jit-hybrid

Modes are **alternated** within each benchmark, so that every mode sees the same conditions
even if the machine state drifts over time. The median of the repeats of a mode is used.

    run_modes.py benches.tsv --out arm_eval.json \
        --remap /home/soonbkwon/projects/wasmDataset=/home/pi/wasmDataset \
        --modes interp,alljit,tierup,lazy,O1,O2,O3 --timeout-scale 4

Benchmarks with existing results are skipped, so an interrupted run resumes when rerun.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from profile_bench import find_walrus, read_bench_list, split_args  # noqa: E402

MODE_FLAGS = {
    "interp": ([], {}),
    "alljit": (["--jit"], {}),
    "tierup": (["--jit-tierup"], {}),
    # The variant tier-up used before cost scaling became the default: one
    # threshold for every function regardless of its size. Kept so the
    # comparison can be re-run.
    "tierup_uniform": (["--jit-tierup"], {"WALRUS_JIT_TIERUP_COST": "0"}),
    "lazy": (["--jit-tierup"], {"WALRUS_JIT_TIERUP_THRESHOLD": "1"}),
    "O1": (["-O1", "--jit-hybrid"], {}),
    "O2": (["-O2", "--jit-hybrid"], {}),
    "O3": (["-O3", "--jit-hybrid"], {}),
    # Sweeping K, the counts a function must reach per byte of its own
    # bytecode. K = 1 is the default and is what "tierup" above measures.
    "cost4": (["--jit-tierup"], {"WALRUS_JIT_TIERUP_COST": "4"}),
    "cost16": (["--jit-tierup"], {"WALRUS_JIT_TIERUP_COST": "16"}),
    "cost64": (["--jit-tierup"], {"WALRUS_JIT_TIERUP_COST": "64"}),
    # Use another binary given with --walrus-alt. Made to measure two builds that differ only
    # in the model alternately within one benchmark.

    # Oracle: compile exactly the functions the label marks hot at theta = 1,
    # read from a per-benchmark list. Not a strategy -- it bounds what the label
    # can buy. {tag} is replaced with the benchmark name.
    "oracle": (["--jit-compile-list", "{oracle_dir}/{tag}.txt"], {}),
}


def apply_remap(text: str, rules) -> str:
    for old, new in rules:
        text = text.replace(old, new)
    return text


TIME_BIN = shutil.which("time") or "/usr/bin/time"


def have_gnu_time() -> bool:
    """Check that this is GNU time. The shell builtin time does not know -v."""
    if not os.path.exists(TIME_BIN):
        return False
    try:
        r = subprocess.run([TIME_BIN, "-v", "-o", os.devnull, "true"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return r.returncode == 0
    except OSError:
        return False


def run_once(walrus, mode, wasm_args, stdin_path, cwd, timeout, base_env, rss,
             subst=None, alt_walrus=None):
    """Run once and return (elapsed time, machine code bytes, function count, max RSS).

    Max RSS is the "Maximum resident set size" printed by GNU time -v. It is written to a
    separate file with -o so it does not mix with the [jit-stats] walrus writes to stderr.
    If rss is False, the command is not wrapped.
    """
    if "@" in mode:
        base, key = mode.split("@", 1)
        flags, extra = MODE_FLAGS[base]
        walrus = alt_walrus[key]
    else:
        flags, extra = MODE_FLAGS[mode]
    if subst:
        flags = [f.format(**subst) for f in flags]
    env = dict(base_env)
    env.update(extra)
    inner = [walrus] + flags + ["--jit-stats"] + wasm_args
    tv = None
    if rss:
        fd, tv = tempfile.mkstemp(prefix="rss_", suffix=".txt")
        os.close(fd)
        cmd = [TIME_BIN, "-v", "-o", tv] + inner
    else:
        cmd = inner
    fh = open(stdin_path, "rb") if stdin_path else subprocess.DEVNULL
    t0 = time.time()
    try:
        p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                           stdin=fh, cwd=cwd, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        if tv:
            os.unlink(tv)
        return None
    finally:
        if stdin_path:
            fh.close()
    if p.returncode != 0:
        if tv:
            os.unlink(tv)
        return None
    code = nfun = 0
    for line in p.stderr.decode(errors="replace").splitlines():
        if line.startswith("[jit-stats]"):
            for tok in line.split():
                if tok.startswith("code="):
                    code = int(tok[5:])
                elif tok.startswith("functions="):
                    nfun = int(tok[10:])
    maxrss = 0
    if tv:
        for line in open(tv, errors="replace"):
            if "Maximum resident set size" in line:
                maxrss = int(line.rsplit(":", 1)[1].strip()) * 1024   # KB -> bytes
        os.unlink(tv)
    return time.time() - t0, code, nfun, maxrss


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bench_list")
    ap.add_argument("--out", required=True)
    ap.add_argument("--walrus", default=None)
    ap.add_argument("--modes", default="interp,alljit,tierup,lazy,O1,O2,O3")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--remap", action="append", default=[], metavar="OLD=NEW",
                    help="rewrite paths in the bench list. May be repeated.")
    ap.add_argument("--timeout-scale", type=float, default=1.0,
                    help="extend timeouts on slow machines (default 1.0)")
    ap.add_argument("--min-timeout", type=float, default=120.0)
    ap.add_argument("--no-rss", action="store_true",
                    help="do not measure max RSS (when GNU time is unavailable)")
    ap.add_argument("--walrus-alt", action="append", default=[], metavar="NAME=PATH",
                    help="another walrus binary, used as mode 'O3@NAME'. "
                         "May be given multiple times")
    ap.add_argument("--oracle-dir", default=None,
                    help="directory of per-benchmark compile lists read by the oracle mode")
    ap.add_argument("--baselines-out", default=None,
                    help="when both interp and alljit are measured, also write them "
                         "in profile/baselines.json format")
    args = ap.parse_args()

    alt = {}
    for spec in args.walrus_alt:
        name, _, path = spec.partition("=")
        alt[name] = path
    modes = [m for m in args.modes.split(",") if m]
    bad = [m for m in modes
           if (m.split("@", 1)[0] if "@" in m else m) not in MODE_FLAGS
           or ("@" in m and m.split("@", 1)[1] not in alt)]
    if bad:
        raise SystemExit(f"unknown modes: {bad}. available: {list(MODE_FLAGS)}")

    rules = []
    for r in args.remap:
        old, _, new = r.partition("=")
        if not old or not new:
            raise SystemExit(f"--remap must be of the form old=new: {r}")
        rules.append((old, new))

    walrus = find_walrus(args.walrus)
    base_env = dict(os.environ)
    base_env.pop("WALRUS_PERF_DIR", None)

    rss = not args.no_rss
    if rss and not have_gnu_time():
        print("GNU time not found, continuing without RSS. "
              "(Ubuntu/Debian: sudo apt install time)", flush=True)
        rss = False

    benches = read_bench_list(Path(args.bench_list))
    out_path = Path(args.out)
    res = json.loads(out_path.read_text()) if out_path.exists() else {}

    print(f"walrus: {walrus}")
    print(f"{len(benches)} benchmarks, modes {modes}, {args.repeats} repeats, "
          f"max RSS {'measured' if rss else 'not measured'}\n", flush=True)

    for i, (tag, wall, cwd, argstr) in enumerate(benches, 1):
        if tag in res:
            continue
        cwd = apply_remap(cwd, rules)
        wasm_args, stdin_path = split_args(apply_remap(argstr, rules))
        if stdin_path:
            stdin_path = apply_remap(stdin_path, rules)
        timeout = max(args.min_timeout, wall * 8 * args.timeout_scale)

        acc = {m: [] for m in modes}
        rssv = {m: [] for m in modes}
        code = {m: 0 for m in modes}
        nfun = {m: 0 for m in modes}
        failed = None
        for _ in range(args.repeats):
            for m in modes:                      # alternate the modes
                r = run_once(walrus, m, wasm_args, stdin_path, cwd, timeout,
                             base_env, rss,
                             {"tag": tag, "oracle_dir": args.oracle_dir or ""},
                             alt)
                if r is None:
                    failed = m
                    break
                t, c, n, mr = r
                acc[m].append(t)
                rssv[m].append(mr)
                code[m], nfun[m] = c, n
            if failed:
                break
        if failed:
            res[tag] = None
            print(f"[{i:>4}/{len(benches)}] {tag:<42} failed ({failed})", flush=True)
        else:
            # Keep the raw repeats. Keeping only the median hides the spread and
            # leaves no basis for judging whether there were enough repeats.
            res[tag] = {m: {"t": round(statistics.median(acc[m]), 3),
                            "reps": [round(v, 3) for v in acc[m]],
                            "code": code[m], "n": nfun[m],
                            # RSS is a maximum, so the minimum is more stable than the median.
                            # Other processes can push it up but never
                            # bring it down.
                            "rss": min(rssv[m]) if rss else 0,
                            "rss_max": max(rssv[m]) if rss else 0,
                            "rss_reps": rssv[m] if rss else []}
                       for m in modes}
            line = "  ".join(
                f"{m} {res[tag][m]['t']:>6.2f}s/{res[tag][m]['code']//1024:>4}KB"
                + (f"/{res[tag][m]['rss']//1024:>5}KB" if rss else "")
                for m in modes)
            print(f"[{i:>4}/{len(benches)}] {tag:<42} {line}", flush=True)
        out_path.write_text(json.dumps(res, indent=1, sort_keys=True))

    if args.baselines_out and "interp" in modes and "alljit" in modes:
        base = {t: {"interp_s": v["interp"]["t"], "alljit_s": v["alljit"]["t"],
                    "alljit_code": v["alljit"]["code"]}
                for t, v in res.items() if v}
        Path(args.baselines_out).write_text(json.dumps(base, indent=1, sort_keys=True))
        print(f"\nwrote {args.baselines_out} ({len(base)} benches)")

    ok = sum(1 for v in res.values() if v)
    print(f"\n{ok} done, {len(res)-ok} failed -> {out_path}")
    print("RUNMODESDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
