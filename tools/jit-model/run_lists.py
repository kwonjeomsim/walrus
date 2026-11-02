#!/usr/bin/env python3
"""Runs one module under several compile lists, alternating, and records time and code.

Variants are given as JSON {"name": [function indices] | "interp" | "tierup" | "O1".."O3"}.
Standard library only, so it runs on the device as is.

    run_lists.py --walrus W --wasm app.wasm --variants v.json --out r.json [--repeats 10] [--cpu 2]
"""
import argparse
import json
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

FIXED = {"interp": [], "tierup": ["--jit-tierup"], "alljit": ["--jit"],
         "O1": ["-O1", "--jit-hybrid"], "O2": ["-O2", "--jit-hybrid"], "O3": ["-O3", "--jit-hybrid"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walrus", required=True)
    ap.add_argument("--wasm", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--cpu", type=int, default=2)
    a = ap.parse_args()
    variants = json.loads(Path(a.variants).read_text())
    lists = {}
    tmp = Path(tempfile.mkdtemp(prefix="lists_"))
    for name, v in variants.items():
        if isinstance(v, list):
            p = tmp / f"{name}.txt"
            p.write_text("\n".join(map(str, sorted(v))) + "\n")
            lists[name] = ["--jit-compile-list", str(p)]
        else:
            lists[name] = FIXED[v]
    wasm = str(Path(a.wasm).resolve())
    acc = {n: [] for n in variants}
    for r in range(a.repeats):
        for name, flags in lists.items():
            t0 = time.time()
            p = subprocess.run(["taskset", "-c", str(a.cpu), a.walrus] + flags + ["--jit-stats", wasm],
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, cwd=str(Path(wasm).parent))
            t = time.time() - t0
            code = nfun = 0
            for line in p.stderr.decode(errors="replace").splitlines():
                if line.startswith("[jit-stats]"):
                    for tok in line.split():
                        if tok.startswith("code="):
                            code = int(tok[5:])
                        elif tok.startswith("functions="):
                            nfun = int(tok[10:])
            acc[name].append({"t": round(t, 4), "rc": p.returncode, "code": code, "n": nfun})
        print(f"rep {r + 1}/{a.repeats} " + "  ".join(f"{n} {acc[n][-1]['t']:.2f}" for n in acc), flush=True)
    out = {n: {"t": statistics.median(x["t"] for x in v), "code": v[-1]["code"], "n": v[-1]["n"],
               "rc": max(x["rc"] for x in v), "reps": [x["t"] for x in v]} for n, v in acc.items()}
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
