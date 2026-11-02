#!/usr/bin/env python3
"""Build walrus binaries that each drop one feature.

Two steps.

    generate  store pairs of a JITPredictor.cpp without the feature and a JITModelData.h
              retrained for it. Run once on the machine with the training data.
    build     apply each stored pair to the source, build, and keep the binary.
              Run on every machine to be measured.

The split exists because the model is architecture-independent. Building the model
trained on x86 as-is on ARM keeps the "transfers without retraining" claim honest.

    make_ablation_builds.py generate --out-dir build/ablations
    make_ablation_builds.py build    --in-dir  build/ablations --bin-dir bin/
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


def feature_order():
    """Read features in the order feature() returns them."""
    s = PRED.read_text()
    i = s.index("int32_t feature(int idx) const")
    j = s.index("};", i)
    out = []
    for line in s[i:j].splitlines():
        line = line.strip()
        if line.startswith("return ") and not line.startswith("return 0"):
            out.append(line[len("return "):].rstrip(";"))
    return out


def write_variant(keep):
    """Rewrite feature() and the name array with only the kept features."""
    s = PRED.read_text()
    sw = "    int32_t feature(int idx) const\n    {\n        switch (idx) {\n"
    for i, n in enumerate(keep):
        sw += f"        case {i}:\n            return {n};\n"
    sw += "        default:\n            return 0;\n        }\n    }"
    i = s.index("    int32_t feature(int idx) const")
    j = s.index("};", i)
    s = s[:i] + sw + "\n" + s[j:]
    i = s.index("constexpr const char* kFeatureNames[] = {")
    j = s.index("};", i)
    s = s[:i] + "constexpr const char* kFeatureNames[] = {\n" + \
        "".join(f'    "{n}",\n' for n in keep) + s[j:]
    PRED.write_text(s)


def write_tree_py(dropped):
    s = TREE_PY.read_text()
    s = s.replace(f'    "{dropped}",\n', "").replace(f"    {dropped}: int\n", "")
    s = s.replace(f"            self.{dropped},\n", "")
    for line in list(s.splitlines(True)):
        if line.strip().startswith(f"- {dropped}"):
            s = s.replace(line, "")
    TREE_PY.write_text(s)


BUILD_DIR = "out"


def run(cmd):
    cmd = ["ninja", "-C", BUILD_DIR, "walrus"] if cmd == ["ninja"] else cmd
    # Retraining extracts features with walrus --dump-jit-features. It must point
    # at the build just made so the feature table matches. The default (out/walrus)
    # may read another build's table and get the wrong count.
    env = dict(os.environ, WALRUS_BIN=str(R / BUILD_DIR / "walrus"))
    r = subprocess.run(cmd, cwd=R, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-3000:] + r.stderr[-3000:])
        raise SystemExit(f"failed: {' '.join(str(c) for c in cmd)}")


def cmd_generate(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    order = feature_order()
    print(f"{len(order)} features")
    backup = {p: p.read_text() for p in (PRED, MODEL, TREE_PY)}
    try:
        for name in order:
            d = out / name
            if (d / "JITModelData.h").exists():
                print(f"  skip {name}")
                continue
            for p, t in backup.items():
                p.write_text(t)
            write_variant([n for n in order if n != name])
            write_tree_py(name)
            # Retraining calls walrus --dump-jit-features, so the binary with the
            # reduced feature table must be built first.
            run(["ninja"])
            run([str(R / ".venv/bin/python"), "tools/jit-model/export_levels.py",
                 "--level-dir", f"O1={args.train_o1}",
                 "--level-dir", f"O2={args.train_o2}",
                 "--level-dir", f"O3={args.train_o3}",
                 "--wide-thresholds"])
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy(PRED, d / "JITPredictor.cpp")
            shutil.copy(MODEL, d / "JITModelData.h")
            print(f"  generated {name}", flush=True)
    finally:
        for p, t in backup.items():
            p.write_text(t)
        run(["ninja"])
    (out / "order.json").write_text(json.dumps(order, indent=1))
    print(f"\n{len(order)} in {out}. Sources restored.")


def cmd_build(args):
    src = Path(args.in_dir)
    binout = Path(args.bin_dir)
    binout.mkdir(parents=True, exist_ok=True)
    order = json.loads((src / "order.json").read_text())
    backup = {p: p.read_text() for p in (PRED, MODEL)}
    try:
        for name in order:
            target = binout / f"walrus-drop-{name}"
            if target.exists():
                print(f"  skip {name}")
                continue
            shutil.copy(src / name / "JITPredictor.cpp", PRED)
            shutil.copy(src / name / "JITModelData.h", MODEL)
            run(["ninja"])
            shutil.copy(R / BUILD_DIR / "walrus", target)
            print(f"  built {name}", flush=True)
    finally:
        for p, t in backup.items():
            p.write_text(t)
        run(["ninja"])
    shutil.copy(R / BUILD_DIR / "walrus", binout / "walrus-base")
    print(f"\n{len(order)+1} binaries in {binout} (including baseline).")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--out-dir", required=True)
    g.add_argument("--train-o1", default="profile/train_t4.0")
    g.add_argument("--train-o2", default="profile/train_t2.0")
    g.add_argument("--train-o3", default="profile/train_t1.0")
    g.set_defaults(fn=cmd_generate)
    b = sub.add_parser("build")
    b.add_argument("--in-dir", required=True)
    b.add_argument("--bin-dir", required=True)
    b.add_argument("--build-dir", default="out",
                   help="ninja build directory. Set it to use a build with a different compiler")
    b.set_defaults(fn=cmd_build)
    args = ap.parse_args()
    global BUILD_DIR
    BUILD_DIR = getattr(args, "build_dir", "out")
    args.fn(args)


if __name__ == "__main__":
    main()
