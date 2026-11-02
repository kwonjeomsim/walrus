#!/usr/bin/env python3
"""Score predictions on the same scale as the measurements.

Why this is needed
------------------
Cross-validation scores used to compute a "time share" and a "code share" per benchmark and then
**average the benchmarks with equal weight**. The values we actually report are not computed that way.

    time attainment = (Σ interpreter time − Σ measured time) / (Σ interpreter time − Σ all-JIT time)
    code ratio      = Σ generated machine code / Σ all-JIT machine code

Both are **plain sums** across benchmarks, so long-running benchmarks and benchmarks with large
machine code dominate the result. For example, if a model misses an entire 40-second program,
the measured time attainment drops sharply, but in an equal-weight average it is one of 210
benchmarks and the score moves by less than 0.5 percent. In the 2026-09-07 feature-combination
experiment, cross-validation said "reducing to five features is fine" but the measurements
disagreed, and this scale mismatch was the cause (see docs/report_list/06).

This module aligns the scale with the measurements by giving each benchmark two weights.

    gain weight gain_b = interpreter time − all-JIT time
        The maximum time compilation can save in that program. A benchmark with a small
        value barely changes in measurement whatever is compiled, so it should
        count little in the score.

    code weight code_b = machine code bytes produced by all-JIT
        This goes directly into the denominator of the code ratio.

The time side is still an approximation. How much compiling a function actually speeds things up
is only known by running it, so here we assume "the gain equals that function's self time".
Boundary cost (switching between interpreter and machine code) is not counted either. Still,
**matching the weights alone greatly reduces the mismatch with measurements.** The `check`
command below shows how well it matches the actual measurements.

Inputs
------
    profile/baselines.json   per-benchmark interpreter and all-JIT time and machine code size
    profile/codesize/<tag>.txt   "func_index bytes" list written by `walrus --jit-code-dump=`
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_BASELINES = REPO / "profile/baselines.json"
DEFAULT_CODESIZE = REPO / "profile/codesize"

# Per-level theta. Must match the label rule in labeling.py.
LEVEL_THETA = {"O1": 4.0, "O2": 2.0, "O3": 1.0}


class Weighting:
    """Holds each benchmark's weight in the measurements."""

    def __init__(self, baselines: Path = DEFAULT_BASELINES,
                 codesize_dir: Path = DEFAULT_CODESIZE):
        self.base = json.loads(Path(baselines).read_text())
        self.codesize_dir = Path(codesize_dir)
        self._codes: Dict[str, Dict[int, int]] = {}

    def codes(self, bench: str) -> Dict[int, int]:
        """Function index -> machine code bytes produced when compiling that function."""
        if bench not in self._codes:
            p = self.codesize_dir / f"{bench}.txt"
            d: Dict[int, int] = {}
            if p.exists():
                for line in p.read_text(errors="replace").splitlines():
                    f = line.split()
                    if len(f) == 2:
                        try:
                            d[int(f[0])] = int(f[1])
                        except ValueError:
                            pass
            self._codes[bench] = d
        return self._codes[bench]

    def known(self, bench: str) -> bool:
        return bench in self.base and bool(self.codes(bench))

    def gain(self, bench: str) -> float:
        b = self.base.get(bench)
        return max(0.0, b["interp_s"] - b["alljit_s"]) if b else 0.0

    def code_total(self, bench: str) -> float:
        """All-JIT machine code size. Summed from the same dump as the numerator for consistency."""
        return float(sum(self.codes(bench).values()))


class Pool:
    """Per-row information needed for scoring. Independent of the model."""

    def __init__(self, tfrac: np.ndarray, findex: np.ndarray,
                 groups: Iterable[str], weighting: Weighting):
        self.tfrac = np.asarray(tfrac, dtype=np.float64)
        self.findex = np.asarray(findex, dtype=np.int64)
        self.groups = np.asarray(list(groups))
        self.w = weighting
        self.benches = [b for b in sorted(set(self.groups)) if weighting.known(b)]
        self._mask = {b: (self.groups == b) for b in self.benches}
        # Precompute per benchmark to speed up repeated scoring.
        self._tsum = {b: float(self.tfrac[self._mask[b]].sum()) for b in self.benches}
        self._code = {b: np.array([weighting.codes(b).get(int(i), 0)
                                   for i in self.findex[self._mask[b]]], dtype=np.float64)
                      for b in self.benches}
        self._gain = {b: weighting.gain(b) for b in self.benches}
        self._ctot = {b: weighting.code_total(b) for b in self.benches}

    def subset(self, rows) -> "Pool":
        """New Pool with a subset of rows. Used to score only the validation fold in CV."""
        rows = np.asarray(rows)
        return Pool(self.tfrac[rows], self.findex[rows], self.groups[rows], self.w)

    def unknown(self) -> List[str]:
        """Benchmarks excluded from scoring due to missing baselines or code sizes."""
        return [b for b in sorted(set(self.groups)) if b not in self._mask]

    # -----------------------------------------------------------------
    def per_bench(self, pred: np.ndarray) -> Dict[str, dict]:
        out = {}
        for b in self.benches:
            m = self._mask[b]
            p = pred[m]
            ts = self._tsum[b]
            ct = self._ctot[b]
            out[b] = {
                "covered": float(self.tfrac[m][p].sum() / ts) if ts > 0 else 0.0,
                "code": float(self._code[b][p].sum()),
                "code_total": ct,
                "gain": self._gain[b],
            }
        return out

    def score(self, pred: np.ndarray, theta: Optional[float] = None) -> dict:
        """Score on the same scale as the measurements.

        time: approximate time attainment. Weighted mean of the self time fraction covered
              by predicted hot functions, weighted by the time compilation can save per benchmark.
        code: code ratio. Sum of machine code bytes of predicted hot functions divided by
              the sum of bytes produced by all-JIT.
        time_flat / code_flat: the old method (equal-weight mean over benchmarks), reported
              alongside to show the effect of the scale change.
        """
        nt = dt = 0.0
        nc = dc = 0.0
        flat_t, flat_c = [], []
        for b in self.benches:
            m = self._mask[b]
            p = pred[m]
            ts = self._tsum[b]
            cov = float(self.tfrac[m][p].sum() / ts) if ts > 0 else 0.0
            g = self._gain[b]
            nt += g * cov
            dt += g
            cb = float(self._code[b][p].sum())
            ct = self._ctot[b]
            nc += cb
            dc += ct
            flat_t.append(cov)
            flat_c.append(cb / ct if ct > 0 else 0.0)
        t = nt / dt if dt > 0 else 0.0
        c = nc / dc if dc > 0 else 0.0
        out = {"time": t, "code": c,
               "time_flat": float(np.mean(flat_t)) if flat_t else 0.0,
               "code_flat": float(np.mean(flat_c)) if flat_c else 0.0}
        if theta is not None:
            out["util"] = t - theta * c
        return out


def load_pool(profile_dir: Path, weighting: Weighting, glob_pat: str = "*.txt"):
    """Read one label directory and return (X, y, Pool)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import dataset as D

    X, y, tfrac, _bsize, groups, _kept, _dropped, findex = D.load_dataset(
        Path(profile_dir), glob_pat=glob_pat, verbose=False, with_index=True)
    return X, y, Pool(tfrac, findex, groups, weighting)
