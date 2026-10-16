#!/usr/bin/env python3
"""Measure the speed-code curve of tier-up over a wide range of K, together with a control
that precompiles the same function set at load time.

The paper's tier-up fixes K = 1 in n_call + n_back >= K * b(f). This script collects the
measurements to answer "does any K dominate -O2?" and "how much of the speed tier-up loses
is due to compile timing (warm-up + no OSR)?"

Mode names
    interp, alljit, O1, O2, O3, lazy   same as run_modes.py
    tierup_K<k>    --jit-tierup, WALRUS_JIT_TIERUP_COST=k  (size-proportional threshold)
    uniform_T<t>   --jit-tierup, WALRUS_JIT_TIERUP_THRESHOLD=t  (same threshold for every function)
    pre_K<k>       precompile, via --jit-compile-list at load time, the functions that
                   tierup_K<k> eventually compiled. Same code, different timing.

For each benchmark, first run once per K with --jit-code-dump to collect the compile lists
(timings unused), then run all modes alternately --repeats times. The output JSON has the
same shape as run_modes.py, so bootstrap_ci.py and suite_breakdown.py can read it too.

    k_sweep.py --split profile/split_v6.json --set test --out results/ksweep_test_x86.json \\
        --walrus ~/projects/walrus-v6/out/release/x64/walrus --repeats 3 --cpu 2

Benchmarks with existing results are skipped, so an interrupted run resumes when rerun.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from profile_bench import read_bench_list, split_args  # noqa: E402
import run_modes  # noqa: E402

DEFAULT_K = [1, 4, 16, 64, 256, 1024, 4096, 16384]
DEFAULT_T = [100, 1000, 10000, 100000]
FIXED_MODES = ["interp", "alljit", "O1", "O2", "O3", "lazy"]
WASM_DIR = Path("/home/soonbkwon/projects/wasmDataset/external/wasm")   # subject to --remap


def register_modes(ks, ts, list_dir: Path):
    """Add the K, T, and precompile modes to run_modes.MODE_FLAGS."""
    for k in ks:
        run_modes.MODE_FLAGS[f"tierup_K{k}"] = (
            ["--jit-tierup"], {"WALRUS_JIT_TIERUP_COST": str(k)})
        run_modes.MODE_FLAGS[f"pre_K{k}"] = (
            ["--jit-compile-list", str(list_dir / f"{{tag}}.K{k}.txt")], {})
    for t in ts:
        run_modes.MODE_FLAGS[f"uniform_T{t}"] = (
            ["--jit-tierup"], {"WALRUS_JIT_TIERUP_THRESHOLD": str(t)})


def bench_rows(args):
    """Select the benchmarks of one split set from the bench list.

    Applications added later are not in benches.tsv. They are standalone drivers run without
    arguments, so a row is built from the wasm file alone. wall_s is only used for the timeout.
    """
    rows = {r[0]: r for r in read_bench_list(Path(args.bench_list))}
    tags = json.loads(Path(args.split).read_text())[args.set]
    hint = {}
    if args.wall_hint:
        d = json.loads(Path(args.wall_hint).read_text())
        hint = {t: v["interp"]["t"] for t, v in d.items() if v}
    rules = []
    for r in args.remap:
        old, _, new = r.partition("=")
        if not old or not new:
            raise SystemExit(f"--remap must be of the form old=new: {r}")
        rules.append((old, new))
    rm = lambda s: run_modes.apply_remap(s, rules)
    out, missing = [], []
    for tag in sorted(tags):
        if tag in rows:
            t, w, cwd, a = rows[tag]
            out.append((t, w, rm(cwd), rm(a)))
        elif Path(rm(str(WASM_DIR / f"{tag}.wasm"))).exists():
            out.append((tag, hint.get(tag, 30.0), rm(str(WASM_DIR)),
                        rm(str(WASM_DIR / f"{tag}.wasm"))))
        else:
            missing.append(tag)
    if missing:
        raise SystemExit(f"benchmarks with unknown run commands: {missing}")
    return out


def collect_list(walrus, k, tag, wasm_args, stdin_path, cwd, timeout, env, list_dir):
    """Run tierup_K<k> once and save the indices of the compiled functions to a list file."""
    dst = list_dir / f"{tag}.K{k}.txt"
    if dst.exists():
        return True
    fd, dump = tempfile.mkstemp(prefix="dump_", suffix=".txt")
    os.close(fd)
    flags, extra = run_modes.MODE_FLAGS[f"tierup_K{k}"]
    e = dict(env)
    e.update(extra)
    cmd = [walrus] + flags + [f"--jit-code-dump={dump}"] + wasm_args
    fh = open(stdin_path, "rb") if stdin_path else run_modes.subprocess.DEVNULL
    try:
        p = run_modes.subprocess.run(cmd, stdout=run_modes.subprocess.DEVNULL,
                                     stderr=run_modes.subprocess.DEVNULL, stdin=fh,
                                     cwd=cwd, timeout=timeout, env=e)
        ok = p.returncode == 0
    except run_modes.subprocess.TimeoutExpired:
        ok = False
    finally:
        if stdin_path:
            fh.close()
    if ok:
        idx = sorted({int(line.split()[0]) for line in open(dump) if line.strip()})
        dst.write_text(" ".join(map(str, idx)) + "\n")
    os.unlink(dump)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench-list", default="profile/benches.tsv")
    ap.add_argument("--split", required=True)
    ap.add_argument("--set", required=True, help="split file key (test, train)")
    ap.add_argument("--wall-hint", default=None,
                    help="results JSON used to set timeouts for benchmarks not in the list")
    ap.add_argument("--out", required=True)
    ap.add_argument("--walrus", required=True)
    ap.add_argument("--k", default=",".join(map(str, DEFAULT_K)))
    ap.add_argument("--t", default=",".join(map(str, DEFAULT_T)))
    ap.add_argument("--fixed", default=",".join(FIXED_MODES),
                    help="fixed modes to measure alongside. interp,alljit is enough for the training set")
    ap.add_argument("--no-pre", action="store_true", help="skip the precompile control")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cpu", type=int, default=None, help="pin to this single core")
    ap.add_argument("--only", default=None, help="only these comma-separated benchmarks (for testing)")
    ap.add_argument("--remap", action="append", default=[], metavar="OLD=NEW",
                    help="rewrite paths in the bench list (for running on another machine). May be repeated")
    args = ap.parse_args()

    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})   # inherited by child processes

    ks = [int(x) for x in args.k.split(",") if x]
    ts = [int(x) for x in args.t.split(",") if x]
    out_path = Path(args.out)
    list_dir = out_path.with_suffix("").parent / (out_path.stem + "_lists")
    list_dir.mkdir(parents=True, exist_ok=True)
    register_modes(ks, ts, list_dir)

    modes = [m for m in args.fixed.split(",") if m]
    modes += [f"tierup_K{k}" for k in ks]
    if not args.no_pre:
        modes += [f"pre_K{k}" for k in ks]
    modes += [f"uniform_T{t}" for t in ts]

    benches = bench_rows(args)
    if args.only:
        keep = set(args.only.split(","))
        benches = [b for b in benches if b[0] in keep]
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    env = dict(os.environ)
    env.pop("WALRUS_PERF_DIR", None)
    env.pop("WALRUS_JIT_TIERUP_COST", None)
    env.pop("WALRUS_JIT_TIERUP_THRESHOLD", None)
    rss = run_modes.have_gnu_time()

    print(f"walrus: {args.walrus}\n{len(benches)} benchmarks, {len(modes)} modes, "
          f"{args.repeats} repeats, cpu {args.cpu}, RSS {'measured' if rss else 'off'}",
          flush=True)
    t_start = time.time()
    for i, (tag, wall, cwd, argstr) in enumerate(benches, 1):
        if tag in res:
            continue
        wasm_args, stdin_path = split_args(argstr)
        timeout = max(120.0, wall * 8)
        if not args.no_pre:
            bad = [k for k in ks if not collect_list(args.walrus, k, tag, wasm_args,
                                                    stdin_path, cwd, timeout, env, list_dir)]
            if bad:
                res[tag] = None
                print(f"[{i:>3}/{len(benches)}] {tag:<40} list collection failed K={bad}", flush=True)
                out_path.write_text(json.dumps(res, indent=1, sort_keys=True))
                continue
        acc = {m: [] for m in modes}
        rssv = {m: [] for m in modes}
        code = {m: 0 for m in modes}
        nfun = {m: 0 for m in modes}
        failed = None
        for _ in range(args.repeats):
            for m in modes:
                r = run_modes.run_once(args.walrus, m, wasm_args, stdin_path, cwd,
                                       timeout, env, rss, {"tag": tag}, {})
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
            print(f"[{i:>3}/{len(benches)}] {tag:<40} failed ({failed})", flush=True)
        else:
            res[tag] = {m: {"t": round(statistics.median(acc[m]), 3),
                            "reps": [round(v, 3) for v in acc[m]],
                            "code": code[m], "n": nfun[m],
                            "rss": min(rssv[m]) if rss else 0,
                            "rss_max": max(rssv[m]) if rss else 0,
                            "rss_reps": rssv[m] if rss else []}
                        for m in modes}
            el = (time.time() - t_start) / 60
            r0 = res[tag]
            print(f"[{i:>3}/{len(benches)}] {tag:<40} interp {r0.get('interp', {}).get('t', 0):7.2f}s "
                  f"K1 {r0[f'tierup_K{ks[0]}']['t']:7.2f}s  elapsed {el:6.1f}min", flush=True)
        out_path.write_text(json.dumps(res, indent=1, sort_keys=True))

    ok = sum(1 for v in res.values() if v)
    print(f"\n{ok} done, {len(res) - ok} failed -> {out_path}\nKSWEEPDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
