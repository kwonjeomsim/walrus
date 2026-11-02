#!/usr/bin/env python3
"""Which profitable functions of one module the levels select, and which they miss.

    app_miss.py --labeled lab/app_xz.txt --codesize code/app_xz.txt --names map_xz.txt \
        --wasm app_xz.wasm --walrus W [--profile-label PROFILE]
"""
import argparse
import re
import subprocess
from pathlib import Path


def read_profile(path):
    lines = Path(path).read_text().splitlines()
    h = next(i for i, l in enumerate(lines) if l.startswith("function"))
    cols = lines[h].split()
    out = {}
    for l in lines[h + 1:]:
        p = l.split()
        if len(p) < len(cols):
            continue
        r = dict(zip(cols, p))
        out[int(r["index"])] = {"bc": int(r["bytecode_sz"]), "ns": int(r["self_time_ns"]),
                               "frac": float(r["time_frac"]), "label": int(r["label"])}
    return out


def selected(walrus, wasm, level):
    p = subprocess.run([walrus, f"-O{level}", "--jit-hybrid-verbose", "--run-export", "__none__", wasm],
                       capture_output=True, text=True)
    m = re.search(r"indices: ([0-9 ]*)\)", p.stdout + p.stderr)
    return set(map(int, m.group(1).split())) if m and m.group(1).strip() else set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", required=True)
    ap.add_argument("--codesize", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--wasm", required=True)
    ap.add_argument("--walrus", required=True)
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()
    prof = read_profile(a.labeled)
    code = {int(x.split()[0]): int(x.split()[1]) for x in Path(a.codesize).read_text().split("\n") if x.strip() and x.split()[0].isdigit()}
    names = dict((int(i), n) for i, n in (l.split(":", 1) for l in Path(a.names).read_text().splitlines() if ":" in l))
    sel = {L: selected(a.walrus, a.wasm, L) for L in (1, 2, 3)}
    tot_code = sum(code.values())
    hot = [i for i, r in prof.items() if r["label"]]
    print(f"functions {len(names)}, profitable at theta=1: {len(hot)}, "
          f"selected O1/O2/O3: {len(sel[1])}/{len(sel[2])}/{len(sel[3] | sel[2] | sel[1])}")
    for L in (1, 2, 3):
        s = set().union(*(sel[k] for k in range(1, L + 1)))
        tf = sum(prof[i]["frac"] for i in s if i in prof)
        hf = sum(prof[i]["frac"] for i in hot)
        print(f"  O{L}: time share of selected {tf:.3f} (profitable set {hf:.3f}), code share {sum(code.get(i, 0) for i in s) / tot_code:.3f}")
    s3 = sel[1] | sel[2] | sel[3]
    print(f"{'idx':>4} {'name':34} {'time':>7} {'code':>7} {'bc':>7}  label  O1 O2 O3")
    for i in sorted(prof, key=lambda i: -prof[i]["frac"])[:a.top]:
        r = prof[i]
        print(f"{i:4d} {names.get(i, '?')[:34]:34} {r['frac']:7.3f} {code.get(i, 0) / tot_code:7.3f} {r['bc']:7d}  "
              f"{'hot' if r['label'] else '   '}    " + "  ".join('x ' if i in set().union(*(sel[k] for k in range(1, L + 1))) else '. ' for L in (1, 2, 3)))


if __name__ == "__main__":
    main()
