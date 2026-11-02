#!/usr/bin/env python3
"""Combine profiles and code size dumps and relabel them with the theta rule.

    relabel.py <profile dir> <code size dir> --out-dir <output> --theta N

Profiles come from `walrus --profile-output`, code sizes from `walrus --jit-code-dump=`.
Code sizes are attached to already collected profiles, so nothing needs to be rerun.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labeling as P


def read_meta(path: str) -> dict:
    meta = {"benchmark": os.path.basename(path)[:-4], "source": "", "wall_s": 0.0, "rc": 0}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith(("#", "=")):
                break
            if line.startswith("# benchmark"):
                meta["benchmark"] = line.split("=", 1)[1].strip()
            elif line.startswith("# source"):
                meta["source"] = line.split("=", 1)[1].strip()
            elif line.startswith("# wall_s"):
                m = re.search(r"wall_s\s*=\s*([0-9.]+).*rc=(-?\d+)", line)
                if m:
                    meta["wall_s"] = float(m.group(1))
                    meta["rc"] = int(m.group(2))
    return meta


def read_code_sizes(path: str) -> dict:
    sizes = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 2:
                try:
                    sizes[int(parts[0])] = int(parts[1])
                except ValueError:
                    pass
    return sizes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("profile_dir")
    ap.add_argument("codesize_dir")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--theta", type=float, default=P.THETA)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results, missing = [], 0
    for name in sorted(os.listdir(args.profile_dir)):
        if not name.endswith(".txt") or name.startswith("_"):
            continue
        src = os.path.join(args.profile_dir, name)
        dump = P.parse_dump(src)
        if not dump["rows"]:
            continue
        meta = read_meta(src)
        cs_path = os.path.join(args.codesize_dir, name)
        if not os.path.exists(cs_path):
            missing += 1
            continue
        sizes = read_code_sizes(cs_path)
        rows = P.label(dump["rows"], int(meta["wall_s"] * 1e9), sizes, args.theta)
        out = os.path.join(args.out_dir, name)
        P.write_bench(out, meta["benchmark"], dump["source"] or meta["source"],
                      meta["wall_s"], meta["rc"], rows)
        results.append({"benchmark": meta["benchmark"], "wall_s": meta["wall_s"],
                        "rc": meta["rc"], "source": dump["source"] or meta["source"],
                        "labeled": rows, "log_size": os.path.getsize(out)})
    P.write_aggregate(args.out_dir, results)
    hot = sum(1 for r in results for x in r["labeled"] if x[-1] == 1)
    total = sum(len(r["labeled"]) for r in results)
    print(f"theta={args.theta}: {len(results)} benches, {hot} of {total} funcs hot"
          + (f" ({missing} without code size excluded)" if missing else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
