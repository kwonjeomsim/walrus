#!/usr/bin/env python3
"""Numbers of the decision-cost section from decision_cost.py output.

    decision_summary.py cost.json [--interp-total S]
"""
import argparse
import json

import numpy as np


def med(runs, key):
    return float(np.median([r[key] for r in runs]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cost")
    ap.add_argument("--interp-total", type=float, help="summed interpreter time of the held-out set (s)")
    a = ap.parse_args()
    d = json.load(open(a.cost))
    tags = sorted(d)
    ct = np.array([med(d[b]["B"], "t") / med(d[b]["A"], "t") - 1 for b in tags])
    ci = np.array([med(d[b]["B"], "ins") / med(d[b]["A"], "ins") - 1 for b in tags])
    sA = sum(med(d[b]["A"], "t") for b in tags)
    sB = sum(med(d[b]["B"], "t") for b in tags)
    cA = sum(med(d[b]["A"], "cyc") for b in tags)
    cv = [np.std([r["t"] for r in d[b]["A"]]) / np.mean([r["t"] for r in d[b]["A"]]) for b in tags]
    pred_cyc = sum(med(d[b]["D"], "cyc") - med(d[b]["C"], "cyc") for b in tags)
    pn = np.array([med(d[b]["D"], "predict_ns") for b in tags]) / 1e6
    print(f"benchmarks {len(tags)}")
    print(f"counters: time median {100 * np.median(ct):.2f}%  sum {100 * (sB / sA - 1):.2f}%  "
          f"instructions median {100 * np.median(ci):.2f}%  negative {(ct < 0).sum()}  "
          f"interp CV median {100 * np.median(cv):.2f}%")
    print(f"predictor: median {np.median(pn):.3f} ms  quartiles {np.percentile(pn, 25):.2f}-{np.percentile(pn, 75):.2f} ms  "
          f"max {pn.max():.1f} ms ({tags[int(pn.argmax())]})  total {pn.sum() / 1e3:.3f} s  "
          f"added cycles {100 * pred_cyc / cA:.4f}% of run time")
    if a.interp_total:
        print(f"  total / interpreted time {100 * pn.sum() / 1e3 / a.interp_total:.4f}%")
    if all("rss_kb" in d[b]["A"][0] for b in tags):
        rc = max(abs(med(d[b]["B"], "rss_kb") / med(d[b]["A"], "rss_kb") - 1) for b in tags)
        rp = max(med(d[b]["D"], "rss_kb") - med(d[b]["C"], "rss_kb") for b in tags)
        print(f"peak RSS: counters change at most {100 * rc:.2f}%  predictor adds at most {rp / 1024:.2f} MiB at load")


if __name__ == "__main__":
    main()
