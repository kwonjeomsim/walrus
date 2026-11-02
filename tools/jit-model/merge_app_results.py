#!/usr/bin/env python3
"""Replace benchmark entries of a result JSON with those of another run.

    merge_app_results.py base.json new.json out.json
"""
import json
import sys

base, new, out = sys.argv[1:4]
b = json.load(open(base))
n = json.load(open(new))
miss = [t for t in n if t not in b]
if miss:
    sys.exit(f"not in base: {miss}")
for t, v in n.items():
    lost = set(b[t]) - set(v)
    if lost:
        print(f"{t}: modes missing from new run, dropped: {sorted(lost)}", file=sys.stderr)
    b[t] = v
json.dump(b, open(out, "w"), indent=1)
print(f"{out}: replaced {len(n)} of {len(b)}")
