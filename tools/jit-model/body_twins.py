#!/usr/bin/env python3
"""For functions of one module, count identical bodies among the training modules
and how often the training labels mark them profitable.

    body_twins.py --wasm app_xz.wasm --funcs 58,60 --names map.txt
"""
import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def leb(b, i):
    r = s = 0
    while True:
        x = b[i]; i += 1
        r |= (x & 0x7f) << s; s += 7
        if x < 0x80:
            return r, i


def bodies(path):
    b = Path(path).read_bytes()
    i, nimp, out = 8, 0, {}
    while i < len(b):
        sid = b[i]; size, j = leb(b, i + 1); end = j + size
        if sid == 2:
            n, k = leb(b, j)
            for _ in range(n):
                l, k = leb(b, k); k += l
                l, k = leb(b, k); k += l
                kind = b[k]; k += 1
                if kind == 0:
                    _, k = leb(b, k); nimp += 1
                elif kind == 1:
                    k += 1; f, k = leb(b, k); _, k = leb(b, k) if f & 1 else (0, k); _, k = leb(b, k)
                elif kind == 2:
                    f, k = leb(b, k); _, k = leb(b, k)
                    if f & 1: _, k = leb(b, k)
                elif kind == 3:
                    k += 2
                else:
                    _, k = leb(b, k)
        if sid == 10:
            n, k = leb(b, j)
            for f in range(n):
                l, k2 = leb(b, k)
                out[nimp + f] = hashlib.sha1(b[k2:k2 + l]).hexdigest()
                k = k2 + l
        i = end
    return out


def labels(path):
    lines = Path(path).read_text().splitlines()
    h = next(i for i, l in enumerate(lines) if l.startswith("function"))
    cols = lines[h].split()
    out = {}
    for l in lines[h + 1:]:
        p = l.split()
        if len(p) >= len(cols):
            r = dict(zip(cols, p)); out[int(r["index"])] = (int(r["label"]), float(r["time_frac"]))
    return out


def wasm_of(tag, rows):
    if tag in rows:
        return next(x for x in reversed(rows[tag]) if x.endswith(".wasm"))
    p = Path.home() / "projects/wasmDataset/external/wasm" / f"{tag}.wasm"
    return str(p) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wasm", required=True)
    ap.add_argument("--funcs", required=True)
    ap.add_argument("--names")
    a = ap.parse_args()
    names = {}
    if a.names:
        names = dict((int(i), n) for i, n in (l.split(":", 1) for l in Path(a.names).read_text().splitlines() if ":" in l))
    target = bodies(a.wasm)
    want = {target[i]: i for i in map(int, a.funcs.split(","))}
    rows = {}
    for l in (REPO / "profile/benches.tsv").read_text().splitlines()[1:]:
        p = l.split("\t"); rows[p[0]] = p[3].split()
    train = json.loads((REPO / "profile/split_v6.json").read_text())["train"]
    hits = {h: [] for h in want}
    for tag in train:
        w = wasm_of(tag, rows)
        if not w:
            continue
        bb = bodies(w)
        lab = {th: labels(REPO / f"profile/train6_t{th}/{tag}.txt") for th in ("4.0", "2.0", "1.0")
               if (REPO / f"profile/train6_t{th}/{tag}.txt").exists()}
        for idx, h in bb.items():
            if h in hits and "1.0" in lab and idx in lab["1.0"]:
                hits[h].append((tag, lab["4.0"][idx][0], lab["2.0"][idx][0], lab["1.0"][idx][0], lab["1.0"][idx][1]))
    for h, i in want.items():
        v = hits[h]
        print(f"{i:4d} {names.get(i, '')[:28]:28} twins in training {len(v):3d}  "
              f"profitable at theta=4/2/1: {sum(x[1] for x in v)}/{sum(x[2] for x in v)}/{sum(x[3] for x in v)}  "
              f"median time share {sorted(x[4] for x in v)[len(v) // 2] if v else 0:.4f}")


if __name__ == "__main__":
    main()
