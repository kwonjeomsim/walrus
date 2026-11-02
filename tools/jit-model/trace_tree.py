#!/usr/bin/env python3
"""Print the decision path of each level's tree for given functions.

    trace_tree.py --model JITModelData.h --features feats.tsv --funcs 37,58 [--names map.txt]
"""
import argparse
import re
from pathlib import Path


def arr(src, name, conv=int):
    m = re.search(rf"{name}\[[^\]]*\]\s*=\s*\{{(.*?)\}};", src, re.S)
    body = re.sub(r"//.*", "", m.group(1))
    return [conv(x.strip().strip('"')) for x in body.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--funcs", required=True)
    ap.add_argument("--names")
    a = ap.parse_args()
    src = Path(a.model).read_text()
    fnames = arr(src, "kFeatureNames", str)
    feat, thr = arr(src, "kNodeFeature"), arr(src, "kNodeThreshold")
    left, right = arr(src, "kNodeLeft"), arr(src, "kNodeRight")
    roots = arr(src, "kLevelRoot")
    rows = Path(a.features).read_text().splitlines()
    cols = rows[0].split("\t")
    F = {int(r.split("\t")[0]): dict(zip(cols, map(int, r.split("\t")))) for r in rows[1:] if r.strip()}
    names = {}
    if a.names:
        names = dict((int(i), n) for i, n in (l.split(":", 1) for l in Path(a.names).read_text().splitlines() if ":" in l))
    used = sorted({fnames[f] for f in feat if f >= 0})
    for i in map(int, a.funcs.split(",")):
        f = F[i]
        print(f"\n## {i} {names.get(i, '')}  " + "  ".join(f"{u}={f[u]}" for u in used))
        for L, n in enumerate(roots):
            path = []
            while feat[n] >= 0:
                v = f[fnames[feat[n]]]
                go = v <= thr[n]
                path.append(f"{fnames[feat[n]]}={v} {'<=' if go else '>'} {thr[n]}")
                n = left[n] if go else right[n]
            print(f"  O{L + 1}: {'HOT ' if left[n] == 1 else 'cold'}  " + " -> ".join(path))


if __name__ == "__main__":
    main()
