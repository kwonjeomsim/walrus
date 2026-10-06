#!/usr/bin/env python3
"""Measure the cost of choosing what to compile, in instructions: tiered counters vs static prediction.

Each benchmark alternates four runs repeatedly, all with the same binary pinned to one P-core.
    A  pure interpreter, full workload
    B  --jit-tierup + threshold 4 billion: counters run but nothing is compiled
    C  load only (a name without --run-export)
    D  load only with -O0 --jit-hybrid: static prediction (feature extraction) runs, nothing is compiled
Counter cost = (B - A) / A, static prediction cost = (D - C) / A. Instructions are user-space retired instructions.
Each run also records time and peak RSS. On machines without perf, --events none measures only time and RSS.

    decision_cost.py benches.tsv --out cost.json [--repeats 10] [--cpu 2] [--events auto|none|<perf events>]
If interrupted, rerunning skips the finished benchmarks.
"""
import argparse, json, os, subprocess, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from profile_bench import read_bench_list, split_args

def pick_events(spec, cpu):
    """Use P-core events on hybrid P/E x86, common names elsewhere. None if perf does not work."""
    if spec == "none":
        return None
    if spec != "auto":
        return spec
    ev = ("cpu_core/instructions/u,cpu_core/cycles/u" if os.path.isdir("/sys/bus/event_source/devices/cpu_core")
          else "instructions:u,cycles:u")
    try:
        p = subprocess.run(["taskset", "-c", str(cpu), "perf", "stat", "-x,", "-e", ev, "--", "true"], capture_output=True, timeout=30)
        if p.returncode == 0 and "<not supported>" not in p.stderr.decode() and "<not counted>" not in p.stderr.decode():
            return ev
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None

def run(walrus, flags, wasm_args, stdin, cwd, env, cpu, timeout, events):
    rss = tempfile.NamedTemporaryFile(suffix=".rss", delete=False).name
    cmd = ["taskset", "-c", str(cpu)]
    if events:
        cmd += ["perf", "stat", "-x,", "-e", events, "--"]
    cmd += ["/usr/bin/time", "-f", "%M", "-o", rss, walrus] + flags + ["--jit-stats"] + wasm_args
    fh = open(stdin, "rb") if stdin else subprocess.DEVNULL
    t0 = time.time()
    try:
        p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, stdin=fh, cwd=cwd, env=env, timeout=timeout)
    finally:
        if stdin: fh.close()
    r = {"t": round(time.time() - t0, 4), "rc": p.returncode}
    try:
        r["rss_kb"] = int(Path(rss).read_text().split()[-1])
    except (OSError, ValueError, IndexError):
        pass
    os.unlink(rss)
    for line in p.stderr.decode(errors="replace").splitlines():
        f = line.split(",")
        if len(f) > 3 and "instructions" in f[2]: r["ins"] = int(f[0])
        elif len(f) > 3 and "cycles" in f[2]: r["cyc"] = int(f[0])
        elif line.startswith("[jit-stats]"):
            for tok in line.split():
                if "=" in tok and not tok.startswith("["):
                    k, v = tok.split("=", 1); r[k] = int(v)
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench_list"); ap.add_argument("--out", required=True)
    ap.add_argument("--walrus", default=str(Path(__file__).resolve().parents[2] / "out/walrus"))
    ap.add_argument("--repeats", type=int, default=10); ap.add_argument("--cpu", type=int, default=2)
    ap.add_argument("--events", default="auto")
    ap.add_argument("--remap", action="append", default=[], metavar="OLD=NEW")
    a = ap.parse_args()
    events = pick_events(a.events, a.cpu)
    env = dict(os.environ); env.pop("WALRUS_PERF_DIR", None)
    envT = dict(env); envT["WALRUS_JIT_TIERUP_THRESHOLD"] = "4000000000"
    NOEXP = ["--run-export", "__no_such_export__"]
    modes = {"A": ([], env, False), "B": (["--jit-tierup"], envT, False),
             "C": (NOEXP, env, True), "D": (["-O0", "--jit-hybrid"] + NOEXP, env, True)}
    out = Path(a.out); res = json.loads(out.read_text()) if out.exists() else {}
    rows = read_bench_list(Path(a.bench_list))
    print(f"walrus {a.walrus}, CPU {a.cpu}, repeats {a.repeats}, benchmarks {len(rows)}, events {events or 'none (time and RSS only)'}", flush=True)
    rules = [r.split("=", 1) for r in a.remap]
    for i, (tag, wall, cwd, args) in enumerate(rows, 1):
        if tag in res: continue
        for old, new in rules:
            cwd, args = cwd.replace(old, new), args.replace(old, new)
        wasm_args, stdin = split_args(args)
        tl = max(120.0, wall * 8)
        acc = {m: [] for m in modes}; bad = None
        for r in range(a.repeats):
            for m, (flags, e, loadonly) in modes.items():
                x = run(a.walrus, flags, wasm_args, stdin, cwd, e, a.cpu, tl, events)
                if x["rc"] != 0 or (events and "ins" not in x): bad = f"{m} rc={x['rc']}"; break
                if m in "BD" and x.get("code", 0) != 0: bad = f"{m} code={x['code']}"; break
                acc[m].append(x)
            if bad: break
        if bad:
            print(f"[{i:>3}/{len(rows)}] {tag:<44} failed {bad}", flush=True); continue
        res[tag] = acc
        med = lambda m, k: sorted(v[k] for v in acc[m])[len(acc[m]) // 2]
        msg = f"counter time {med('B','t')/med('A','t')-1:+.3%}"
        if events:
            A = med("A", "ins")
            msg += (f" instructions {med('B','ins')/A-1:+.3%} cycles {med('B','cyc')/med('A','cyc')-1:+.3%}"
                    f" | prediction instructions {(med('D','ins')-med('C','ins'))/A:+.4%}")
        msg += f" | prediction {med('D','predict_ns')/1e6:.2f}ms, load RSS +{med('D','rss_kb')-med('C','rss_kb')}KB"
        print(f"[{i:>3}/{len(rows)}] {tag:<44} {msg}", flush=True)
        out.write_text(json.dumps(res))
    print("DECISIONCOSTDONE", flush=True)

if __name__ == "__main__":
    main()
