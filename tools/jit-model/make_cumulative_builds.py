#!/usr/bin/env python3
"""Build walrus binaries with features dropped cumulatively.

A small loss from dropping one feature does not mean several can be dropped at once.
When features share information, the rest cover for one missing feature, but nothing
is left to cover when several are gone. To measure this, build a model for each k = 1..n.

Features are dropped in order of **smallest loss at every level** in the single-drop
measurements. Do not order by the mean over levels, since a feature that hurts badly
only at -O1 is hidden by the mean and ends up first.

    make_cumulative_builds.py --order cum_order.json --bin-dir bin/
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[2]
PRED = R / "src/jit/JITPredictor.cpp"
MODEL = R / "src/jit/JITModelData.h"
TREE_PY = R / "tools/jit-model/jit_decision_tree.py"
BUILD_DIR = "out"


def feature_order():
    s = PRED.read_text()
    i = s.index("int32_t feature(int idx) const")
    j = s.index("};", i)
    return [l.strip()[len("return "):].rstrip(";")
            for l in s[i:j].splitlines()
            if l.strip().startswith("return ") and not l.strip().startswith("return 0")]


def write_variant(keep, dropped):
    s = PRED.read_text()
    sw = "    int32_t feature(int idx) const\n    {\n        switch (idx) {\n"
    for i, n in enumerate(keep):
        sw += f"        case {i}:\n            return {n};\n"
    sw += "        default:\n            return 0;\n        }\n    }"
    i = s.index("    int32_t feature(int idx) const"); j = s.index("};", i)
    s = s[:i] + sw + "\n" + s[j:]
    i = s.index("constexpr const char* kFeatureNames[] = {"); j = s.index("};", i)
    s = s[:i] + "constexpr const char* kFeatureNames[] = {\n" + \
        "".join(f'    "{n}",\n' for n in keep) + s[j:]
    PRED.write_text(s)

    s = TREE_PY.read_text()
    for n in dropped:
        s = s.replace(f'    "{n}",\n', "").replace(f"    {n}: int\n", "")
        s = s.replace(f"            self.{n},\n", "")
        for line in list(s.splitlines(True)):
            if line.strip().startswith(f"- {n}"):
                s = s.replace(line, "")
    TREE_PY.write_text(s)


def run(cmd):
    # Retraining extracts features with walrus --dump-jit-features. It must point
    # at the build just made so the feature table matches.
    env = dict(os.environ, WALRUS_BIN=str(R / BUILD_DIR / "walrus"))
    r = subprocess.run(cmd, cwd=R, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-3000:] + r.stderr[-3000:])
        raise SystemExit(f"failed: {' '.join(str(c) for c in cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", required=True, help="JSON list giving the drop order")
    ap.add_argument("--bin-dir", required=True)
    ap.add_argument("--build-dir", default="out")
    args = ap.parse_args()
    global BUILD_DIR
    BUILD_DIR = args.build_dir
    order = json.loads(Path(args.order).read_text())
    binout = Path(args.bin_dir); binout.mkdir(parents=True, exist_ok=True)
    full = feature_order()
    backup = {p: p.read_text() for p in (PRED, MODEL, TREE_PY)}
    try:
        for k in range(1, len(order) + 1):
            target = binout / f"walrus-cum{k}"
            if target.exists():
                print(f"  skip {k}"); continue
            for p, t in backup.items():
                p.write_text(t)
            dropped = order[:k]
            write_variant([n for n in full if n not in dropped], dropped)
            run(["ninja", "-C", BUILD_DIR, "walrus"])
            run([str(R / ".venv/bin/python"), "tools/jit-model/export_levels.py",
                 "--level-dir", "O1=profile/train_t4.0",
                 "--level-dir", "O2=profile/train_t2.0",
                 "--level-dir", "O3=profile/train_t1.0", "--wide-thresholds"])
            run(["ninja", "-C", BUILD_DIR, "walrus"])
            shutil.copy(R / BUILD_DIR / "walrus", target)
            print(f"  built with {k} dropped ({dropped[-1]})", flush=True)
    finally:
        for p, t in backup.items():
            p.write_text(t)
        run(["ninja", "-C", BUILD_DIR, "walrus"])
    print(f"\n{len(order)} cumulative variants in {binout}. Sources restored.")


if __name__ == "__main__":
    main()
