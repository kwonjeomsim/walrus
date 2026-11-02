#!/usr/bin/env python3
"""Extract the indices of functions labeled hot and write them as --jit-compile-list lists.

The oracle is not a strategy. It is an upper bound measuring how much the labels can actually gain.
It compiles exactly the functions the measured profile marks hot, with no prediction.

    make_oracle_lists.py --labeled profile/test_t1.0 --out-dir profile/oracle_t1.0
"""
from __future__ import annotations

import argparse
from pathlib import Path


def hot_indices(path: Path):
    lines = path.read_text().splitlines()
    head = next((i for i, l in enumerate(lines) if l.startswith("function")), None)
    if head is None:
        return None
    cols = lines[head].split()
    try:
        ii, li = cols.index("index"), cols.index("label")
    except ValueError:
        return None
    out = []
    for l in lines[head + 1:]:
        p = l.split()
        if len(p) <= max(ii, li):
            continue
        try:
            if int(p[li]) == 1:
                out.append(int(p[ii]))
        except ValueError:
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = tot = 0
    for f in sorted(Path(args.labeled).glob("*.txt")):
        idx = hot_indices(f)
        if idx is None:
            print(f"  skipped {f.name}")
            continue
        (out / f"{f.stem}.txt").write_text("\n".join(str(i) for i in idx) + "\n")
        n += 1
        tot += len(idx)
    print(f"{n} benchmarks, {tot} hot funcs in total -> {out}")


if __name__ == "__main__":
    main()
